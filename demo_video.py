# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Video demo:
1) Read frames from a video.
2) Run SAM 3D Body per frame to get 2D/3D keypoints.
3) Draw 2D keypoints onto frames and save them as an output video.
4) Aggregate 3D keypoint trajectories across frames and save a single plot.

Notes:
- We only use the first detected person per frame when building the 3D trajectory.
- Frames with no detections are skipped for the trajectory plot but still appear in the
  visualization video (without overlay).
"""

import argparse
import os
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrootutils
import torch
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from tqdm import tqdm


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)


def setup_visualizer() -> SkeletonVisualizer:
    visualizer = SkeletonVisualizer(line_width=2, radius=4)
    visualizer.set_pose_meta(mhr70_pose_info)
    return visualizer


def draw_2d_keypoints(frame_bgr: np.ndarray, outputs: List[dict], visualizer: SkeletonVisualizer):
    """Overlay 2D keypoints on a BGR frame."""
    img = frame_bgr.copy()
    for person_output in outputs:
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        img = visualizer.draw_skeleton(img, keypoints_2d)
    return img


def save_video(frames: List[np.ndarray], output_path: str, fps: float):
    if len(frames) == 0:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps if fps > 0 else 30.0, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


def plot_3d_trajectory(traj_list: List[np.ndarray], save_path: str):
    """
    traj_list: list of (J, 3) arrays (first person per frame), already root-centered.
    """
    if len(traj_list) == 0:
        return
    traj = np.stack(traj_list, axis=0)  # (T, J, 3)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot per-joint trajectory over time
    J = traj.shape[1]
    for j in range(J):
        ax.plot(traj[:, j, 0], traj[:, j, 1], traj[:, j, 2], linewidth=1)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D keypoint trajectories (first person)")
    ax.view_init(elev=20, azim=-70)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main(args):
    # Output paths
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    output_folder = args.output_folder or os.path.join("./output_video", video_name)
    os.makedirs(output_folder, exist_ok=True)

    # Optional component paths (arg > env > default)
    mhr_path = args.mhr_path or os.environ.get("SAM3D_MHR_PATH", "")
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = args.segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", "")

    # Load model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, model_cfg = load_sam_3d_body(
        args.checkpoint_path, device=device, mhr_path=mhr_path
    )

    human_detector, human_segmentor, fov_estimator = None, None, None
    if args.detector_name:
        from tools.build_detector import HumanDetector

        human_detector = HumanDetector(
            name=args.detector_name, device=device, path=detector_path
        )

    if (args.segmentor_name == "sam2" and len(segmentor_path)) or args.segmentor_name != "sam2":
        from tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(
            name=args.segmentor_name, device=device, path=segmentor_path
        )
    if args.fov_name:
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(name=args.fov_name, device=device, path=fov_path)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )
    visualizer = setup_visualizer()

    # Read video
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {args.video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)

    vis_frames = []
    traj_list = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames if total_frames > 0 else None, desc="Processing video")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference (convert BGR -> RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        outputs = estimator.process_one_image(
            rgb_frame,
            bbox_thr=args.bbox_thresh,
            use_mask=args.use_mask,
        )

        # 2D overlay
        if len(outputs) > 0:
            vis_frame = draw_2d_keypoints(frame, outputs, visualizer)
        else:
            vis_frame = frame
        vis_frames.append(vis_frame)

        # Collect 3D keypoints (first person) for trajectory
        if len(outputs) > 0:
            kp3d = outputs[0]["pred_keypoints_3d"]  # (J, 3)
            # Root-center to stabilize trajectory plot (pelvis assumed idx 0)
            kp3d_centered = kp3d - kp3d[0]
            traj_list.append(kp3d_centered)

        frame_idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()

    # Save 2D visualization video
    video_out_path = os.path.join(output_folder, f"{video_name}_2d.mp4")
    save_video(vis_frames, video_out_path, fps)

    # Save 3D trajectory plot
    traj_img_path = os.path.join(output_folder, f"{video_name}_3d_traj.png")
    plot_3d_trajectory(traj_list, traj_img_path)

    print(f"Saved 2D keypoint video to: {video_out_path}")
    print(f"Saved 3D trajectory plot to: {traj_img_path}")


# python demo_video.py \
#   --video_path /path/to/video.mp4 \
#   --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
#   --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
#   --output_folder ./output_video/demo \
#   --detector_name vitdet \
#   --segmentor_name sam2 \
#   --fov_name moge2 \
#   --use_mask       # 如需 mask-conditioned 推理

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM 3D Body Video Demo - Extract 2D/3D keypoints from video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--video_path",
        required=True,
        type=str,
        help="Path to input video",
    )
    parser.add_argument(
        "--output_folder",
        default="",
        type=str,
        help="Output folder (default: ./output_video/<video_name>)",
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        type=str,
        help="Path to SAM 3D Body model checkpoint",
    )
    parser.add_argument(
        "--detector_name",
        default="vitdet",
        type=str,
        help="Human detection model (default: vitdet)",
    )
    parser.add_argument(
        "--segmentor_name",
        default="sam2",
        type=str,
        help="Human segmentation model (default: sam2)",
    )
    parser.add_argument(
        "--fov_name",
        default="moge2",
        type=str,
        help="FOV estimator (default: moge2)",
    )
    parser.add_argument(
        "--detector_path",
        default="",
        type=str,
        help="Path to human detection model folder (or set SAM3D_DETECTOR_PATH)",
    )
    parser.add_argument(
        "--segmentor_path",
        default="",
        type=str,
        help="Path to human segmentation model folder (or set SAM3D_SEGMENTOR_PATH)",
    )
    parser.add_argument(
        "--fov_path",
        default="",
        type=str,
        help="Path to fov estimation model folder (or set SAM3D_FOV_PATH)",
    )
    parser.add_argument(
        "--mhr_path",
        default="",
        type=str,
        help="Path to MoHR/assets folder (or set SAM3D_MHR_PATH)",
    )
    parser.add_argument(
        "--bbox_thresh",
        default=0.8,
        type=float,
        help="Bounding box detection threshold",
    )
    parser.add_argument(
        "--use_mask",
        action="store_true",
        default=False,
        help="Use mask-conditioned prediction (segmentation mask is auto-generated from bbox)",
    )
    args = parser.parse_args()

    main(args)

