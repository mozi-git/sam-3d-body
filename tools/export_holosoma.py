# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Bridge script: take SAM-3D-Body demo_video outputs (*.npz) and export a (T, J, 3)
numpy array for Holosoma retargeting.

Steps:
1) Load *_results.npz from a folder (produced by demo_video.py).
2) Pick one person (default person_idx=0) per frame; skip frames with no detection.
3) Optional root-centering (subtract pelvis/root joint).
4) Optional temporal smoothing (simple moving average).
5) Save a single .npy shaped (T, J, 3).

Holosoma usage example (assuming you place the npy under /path/to/motion_dir
and name it my_motion.npy; see holosoma_retargeting README for data_format/mapping):

  python examples/robot_retarget.py \
    --data_path /path/to/motion_dir \
    --task-type robot_only \
    --task-name my_motion \
    --data_format smplh \
    --retargeter.debug --retargeter.visualize
"""

import argparse
import glob
import os
from typing import List

import numpy as np


def moving_average(traj: np.ndarray, window: int) -> np.ndarray:
    """Temporal smoothing with a simple moving average along the time axis.

    traj: (T, J, 3)
    window: odd integer >= 1
    """
    T = traj.shape[0]
    if window <= 1 or T < window:
        return traj
    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / window
    # pad on time axis
    padded = np.pad(traj, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    smoothed = np.empty_like(traj, dtype=np.float32)
    for d in range(3):
        smoothed[:, :, d] = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="valid"), 0, padded[:, :, d]
        )
    return smoothed


def main(args):
    npz_files = sorted(glob.glob(os.path.join(args.results_dir, "*_results.npz")))
    if len(npz_files) == 0:
        raise FileNotFoundError(f"No *_results.npz found under {args.results_dir}")

    traj_list: List[np.ndarray] = []
    for npz_path in npz_files:
        data = np.load(npz_path, allow_pickle=True)
        outputs = data["outputs"]
        if len(outputs) == 0:
            continue
        if args.person_idx >= len(outputs):
            # Skip if requested person not present
            continue

        person = outputs[args.person_idx].item()
        kp3d = np.array(person["pred_keypoints_3d"], dtype=np.float32)  # (J, 3)

        if args.root_center:
            kp3d = kp3d - kp3d[args.root_index : args.root_index + 1]

        traj_list.append(kp3d)

    if len(traj_list) == 0:
        raise RuntimeError("No frames with the requested person were found.")

    traj = np.stack(traj_list, axis=0)  # (T, J, 3)

    if args.smooth_window > 1:
        traj = moving_average(traj, args.smooth_window)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.save(args.output_path, traj)

    print(
        f"Saved: {args.output_path} | shape={traj.shape} | "
        f"T={traj.shape[0]}, J={traj.shape[1]}"
    )
    print(
        "Reminder: ensure your Holosoma data_format/joints_mapping matches SAM3D joint order, "
        "or add a new mapping in holosoma_retargeting/config_types/data_type.py."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export SAM-3D-Body demo_video outputs to (T, J, 3) npy for Holosoma."
    )
    parser.add_argument(
        "--results_dir",
        required=True,
        help="Folder containing *_results.npz from demo_video.py",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to save the output .npy (T, J, 3)",
    )
    parser.add_argument(
        "--person_idx",
        type=int,
        default=0,
        help="Which person index to export per frame (default: 0)",
    )
    parser.add_argument(
        "--root_center",
        action="store_true",
        help="Subtract root joint to root-center the trajectory (default: False)",
    )
    parser.add_argument(
        "--root_index",
        type=int,
        default=0,
        help="Root joint index in SAM3D keypoints (default: 0 / pelvis)",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=1,
        help="Odd window size for moving-average smoothing along time (default: 1 = off)",
    )
    args = parser.parse_args()

    main(args)

