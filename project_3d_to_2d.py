#!/usr/bin/env python3
"""
3D关键点投影到2D图片工具
功能：将NPZ文件中的3D关键点使用相机参数投影到2D平面，并生成可视化图片和视频
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
import argparse

def load_npz_data(npz_folder: str) -> List[Dict]:
    """
    从NPZ文件夹加载所有数据
    """
    npz_files = sorted([f for f in os.listdir(npz_folder) if f.endswith('.npz')])
    all_data = []
    
    for npz_file in npz_files:
        npz_path = os.path.join(npz_folder, npz_file)
        data = np.load(npz_path, allow_pickle=True)
        
        if 'outputs' in data and len(data['outputs']) > 0:
            frame_data = data['outputs'][0]  # 获取第一人的数据
            frame_data['frame_name'] = npz_file
            all_data.append(frame_data)
    
    return all_data

def create_intrinsic_matrix(focal_length: float, image_width: int = 1920, image_height: int = 1080) -> np.ndarray:
    """
    创建相机内参矩阵
    """
    # 假设主点在图像中心
    cx = image_width / 2
    cy = image_height / 2
    
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ])
    
    return K

def project_3d_to_2d_perspective(kp3d: np.ndarray, 
                                focal_length: float, 
                                cam_t: np.ndarray,
                                image_size: Tuple[int, int] = (1920, 1080)) -> np.ndarray:
    """
    使用透视投影将3D关键点投影到2D平面
    """
    width, height = image_size
    
    # 创建相机内参矩阵
    K = create_intrinsic_matrix(focal_length, width, height)
    
    # 应用相机平移（将3D点转换到相机坐标系）
    kp3d_camera = kp3d + cam_t
    
    # 透视投影
    kp2d_homogeneous = kp3d_camera @ K.T  # (J, 3)
    
    # 齐次坐标转笛卡尔坐标
    kp2d = kp2d_homogeneous[:, :2] / kp2d_homogeneous[:, 2:3]
    
    return kp2d

def project_3d_to_2d_with_bbox(kp3d: np.ndarray, 
                              focal_length: float, 
                              cam_t: np.ndarray,
                              bbox: np.ndarray,
                              image_size: Tuple[int, int] = (1920, 1080)) -> np.ndarray:
    """
    使用边界框信息进行更准确的投影
    """
    width, height = image_size
    
    # 创建相机内参矩阵，使用边界框中心作为主点
    bbox_center_x = (bbox[0] + bbox[2]) / 2
    bbox_center_y = (bbox[1] + bbox[3]) / 2
    
    K = np.array([
        [focal_length, 0, bbox_center_x],
        [0, focal_length, bbox_center_y],
        [0, 0, 1]
    ])
    
    # 应用相机平移
    kp3d_camera = kp3d + cam_t
    
    # 透视投影
    kp2d_homogeneous = kp3d_camera @ K.T  # (J, 3)
    
    # 齐次坐标转笛卡尔坐标
    kp2d = kp2d_homogeneous[:, :2] / kp2d_homogeneous[:, 2:3]
    
    return kp2d

def create_blank_canvas(width: int = 1920, height: int = 1080, bg_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    创建空白画布
    """
    canvas = np.ones((height, width, 3), dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)
    return canvas

def draw_2d_skeleton(canvas: np.ndarray, 
                    kp2d: np.ndarray, 
                    connections: List[Tuple[int, int]],
                    point_color: Tuple[int, int, int] = (0, 0, 255),
                    line_color: Tuple[int, int, int] = (0, 255, 0),
                    point_radius: int = 5,
                    line_thickness: int = 2) -> np.ndarray:
    """
    在画布上绘制2D骨架
    """
    img = canvas.copy()
    
    # 绘制关键点
    for i, point in enumerate(kp2d):
        x, y = int(point[0]), int(point[1])
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img, (x, y), point_radius, point_color, -1)
            # 添加关键点编号
            cv2.putText(img, str(i), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, point_color, 1)
    
    # 绘制骨架连接线
    for start_idx, end_idx in connections:
        if start_idx < len(kp2d) and end_idx < len(kp2d):
            start_point = (int(kp2d[start_idx][0]), int(kp2d[start_idx][1]))
            end_point = (int(kp2d[end_idx][0]), int(kp2d[end_idx][1]))
            
            if (0 <= start_point[0] < img.shape[1] and 0 <= start_point[1] < img.shape[0] and
                0 <= end_point[0] < img.shape[1] and 0 <= end_point[1] < img.shape[0]):
                cv2.line(img, start_point, end_point, line_color, line_thickness)
    
    return img

def get_skeleton_connections() -> List[Tuple[int, int]]:
    """
    定义70个关键点的骨架连接关系
    """
    connections = [
        # 腿部连接
        (13, 11), (11, 9),  # 左腿：脚踝-膝盖-髋部
        (14, 12), (12, 10),  # 右腿：脚踝-膝盖-髋部
        (9, 10),  # 髋部连接
        
        # 身体主干
        (9, 5), (10, 6),  # 髋部到肩膀
        (5, 6),  # 肩膀连接
        
        # 手臂连接
        (5, 7), (7, 62),  # 左臂：肩膀-肘部-手腕
        (6, 8), (8, 41),  # 右臂：肩膀-肘部-手腕
        
        # 头部连接
        (1, 2),  # 眼睛连接
        (0, 1), (0, 2),  # 鼻子到眼睛
        (1, 3), (2, 4),  # 眼睛到耳朵
        (3, 5), (4, 6),  # 耳朵到肩膀
        
        # 脚部连接
        (13, 15), (13, 16), (13, 17),  # 左脚：脚踝到大脚趾、小脚趾、脚跟
        (14, 18), (14, 19), (14, 20),  # 右脚：脚踝到大脚趾、小脚趾、脚跟
        
        # 左手连接
        (62, 45), (45, 44), (44, 43), (43, 42),  # 左手拇指
        (62, 49), (49, 48), (48, 47), (47, 46),  # 左手食指
        (62, 53), (53, 52), (52, 51), (51, 50),  # 左手中指
        (62, 57), (57, 56), (56, 55), (55, 54),  # 左手无名指
        (62, 61), (61, 60), (60, 59), (59, 58),  # 左手小指
        
        # 右手连接
        (41, 24), (24, 23), (23, 22), (22, 21),  # 右手拇指
        (41, 28), (28, 27), (27, 26), (26, 25),  # 右手食指
        (41, 32), (32, 31), (31, 30), (30, 29),  # 右手中指
        (41, 36), (36, 35), (35, 34), (34, 33),  # 右手无名指
        (41, 40), (40, 39), (39, 38), (38, 37),  # 右手小指
        
        # 额外身体连接
        (7, 63), (8, 64),  # 肘部
        (5, 67), (6, 68),  # 肩部
        (69, 5), (69, 6),  # 颈部到肩膀
    ]
    
    return connections

def compare_with_original_2d(canvas: np.ndarray, 
                           projected_kp2d: np.ndarray, 
                           original_kp2d: np.ndarray,
                           connections: List[Tuple[int, int]]) -> np.ndarray:
    """
    比较投影后的2D关键点与原始的2D关键点
    """
    img = canvas.copy()
    
    # 绘制原始2D关键点（红色）
    for i, point in enumerate(original_kp2d):
        x, y = int(point[0]), int(point[1])
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img, (x, y), 4, (0, 0, 255), -1)  # 红色
    
    # 绘制原始2D的骨架（橙色）
    for start_idx, end_idx in connections:
        if start_idx < len(original_kp2d) and end_idx < len(original_kp2d):
            start_point = (int(original_kp2d[start_idx][0]), int(original_kp2d[start_idx][1]))
            end_point = (int(original_kp2d[end_idx][0]), int(original_kp2d[end_idx][1]))
            
            if (0 <= start_point[0] < img.shape[1] and 0 <= start_point[1] < img.shape[0] and
                0 <= end_point[0] < img.shape[1] and 0 <= end_point[1] < img.shape[0]):
                cv2.line(img, start_point, end_point, (0, 165, 255), 2)  # 橙色

    # 绘制投影后的2D关键点（绿色）
    for i, point in enumerate(projected_kp2d):
        x, y = int(point[0]), int(point[1])
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)  # 绿色
    
    # 绘制投影后的骨架（蓝色）
    for start_idx, end_idx in connections:
        if start_idx < len(projected_kp2d) and end_idx < len(projected_kp2d):
            start_point = (int(projected_kp2d[start_idx][0]), int(projected_kp2d[start_idx][1]))
            end_point = (int(projected_kp2d[end_idx][0]), int(projected_kp2d[end_idx][1]))
            
            if (0 <= start_point[0] < img.shape[1] and 0 <= start_point[1] < img.shape[0] and
                0 <= end_point[0] < img.shape[1] and 0 <= end_point[1] < img.shape[0]):
                cv2.line(img, start_point, end_point, (255, 0, 0), 2)  # 蓝色
    
    # 添加图例
    cv2.putText(img, "Red: Original 2D Keypoints", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(img, "Green: Projected 2D Keypoints", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img, "Blue: Projected Skeleton", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    return img

def create_projection_visualization(data: Dict, 
                                  output_path: str, 
                                  image_size: Tuple[int, int] = (1920, 1080)):
    """
    创建3D到2D投影的可视化
    """
    if 'pred_keypoints_3d' not in data or 'focal_length' not in data or 'pred_cam_t' not in data:
        print(f"跳过 {data.get('frame_name', '未知帧')}: 缺少必要的3D数据")
        return
    
    kp3d = data['pred_keypoints_3d']
    focal_length = data['focal_length']
    cam_t = data['pred_cam_t']
    
    # 尝试使用边界框进行更准确的投影
    if 'bbox' in data:
        projected_kp2d = project_3d_to_2d_with_bbox(kp3d, focal_length, cam_t, data['bbox'], image_size)
        print(f"使用边界框进行投影")
    else:
        projected_kp2d = project_3d_to_2d_perspective(kp3d, focal_length, cam_t, image_size)
        print(f"使用默认投影")
    
    # 创建空白画布
    canvas = create_blank_canvas(image_size[0], image_size[1])
    
    # 获取骨架连接关系
    connections = get_skeleton_connections()
    
    # 绘制投影后的2D骨架
    result_img = draw_2d_skeleton(canvas, projected_kp2d, connections)
    
    # 如果有原始2D关键点，进行比较
    if 'pred_keypoints_2d' in data:
        original_kp2d = data['pred_keypoints_2d']
        comparison_img = compare_with_original_2d(canvas, projected_kp2d, original_kp2d, connections)
        
        # 保存比较图
        base_name, ext = os.path.splitext(output_path)
        comparison_path = f"{base_name}_comparison{ext}"
        # 统一保存到 compare文件夹
        compare_dir = os.path.join(os.path.dirname(output_path), "compare")
        os.makedirs(compare_dir, exist_ok=True)
        comparison_path = os.path.join(compare_dir, os.path.basename(comparison_path))
        cv2.imwrite(comparison_path, comparison_img)
        print(f"已保存比较图: {comparison_path}")
    
    # 保存投影图
    # 统一保存到 projection文件夹
    projection_dir = os.path.join(os.path.dirname(output_path), "projection")
    os.makedirs(projection_dir, exist_ok=True)
    output_path = os.path.join(projection_dir, os.path.basename(output_path))
    cv2.imwrite(output_path, result_img)
    print(f"已保存投影图: {output_path}")
    
    return projected_kp2d

def create_video_from_images(image_folder: str, 
                           output_video_path: str, 
                           fps: int = 30,
                           image_pattern: str = "frame_*.png"):
    """
    将图片序列合成为视频
    
    参数:
        image_folder: 包含图片的文件夹路径
        output_video_path: 输出视频文件路径
        fps: 视频帧率（默认：30）
        image_pattern: 图片文件名模式（默认：frame_*.png）
    """
    import glob
    
    # 获取所有匹配的图片文件
    image_pattern_full = os.path.join(image_folder, image_pattern)
    image_files = sorted(glob.glob(image_pattern_full))
    
    if not image_files:
        print(f"错误：在 {image_folder} 中未找到匹配 {image_pattern} 的图片")
        return False
    
    print(f"找到 {len(image_files)} 张图片，正在创建视频...")
    
    # 读取第一张图片获取尺寸
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print(f"错误：无法读取第一张图片 {image_files[0]}")
        return False
    
    height, width, _ = first_image.shape
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print(f"错误：无法创建视频文件 {output_video_path}")
        return False
    
    # 逐帧写入视频
    for i, image_file in enumerate(image_files):
        img = cv2.imread(image_file)
        if img is not None:
            video_writer.write(img)
            if i % 10 == 0:  # 每10帧打印一次进度
                print(f"已处理 {i+1}/{len(image_files)} 帧")
        else:
            print(f"警告：无法读取图片 {image_file}")
    
    # 释放视频写入器
    video_writer.release()
    
    print(f"✅ 视频创建完成: {output_video_path}")
    print(f"视频信息: {width}x{height}, {fps} fps, {len(image_files)} 帧")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='3D关键点投影到2D图片并生成视频')
    parser.add_argument('--npz_folder', type=str, required=True,
                       help='包含NPZ文件的文件夹路径')
    parser.add_argument('--output_folder', type=str, required=True,
                       help='输出图片文件夹路径')
    parser.add_argument('--image_width', type=int, default=1920,
                       help='输出图片宽度（默认：1920）')
    parser.add_argument('--image_height', type=int, default=1080,
                       help='输出图片高度（默认：1080）')
    parser.add_argument('--frame_interval', type=int, default=1,
                       help='每多少帧处理一张图片（默认：1）')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='最大处理帧数（默认：所有帧）')
    parser.add_argument('--create_video', action='store_true',
                       help='是否将图片合成为视频')
    parser.add_argument('--video_fps', type=int, default=30,
                       help='视频帧率（默认：30）')
    parser.add_argument('--video_name', type=str, default='projection_video.mp4',
                       help='输出视频文件名（默认：projection_video.mp4）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_folder, exist_ok=True)
    
    # 加载数据
    print("正在加载NPZ数据...")
    all_data = load_npz_data(args.npz_folder)
    
    if not all_data:
        print("错误：未找到有效的NPZ数据")
        return
    
    print(f"成功加载 {len(all_data)} 帧数据")
    
    # 限制帧数
    if args.max_frames:
        all_data = all_data[:args.max_frames]
    
    image_size = (args.image_width, args.image_height)
    all_projected_kp2d = []
    
    # 处理每帧数据
    for i, data in enumerate(all_data):
        if i % args.frame_interval != 0:
            continue
            
        output_path = os.path.join(args.output_folder, f"frame_{i:06d}_projection.png")
        
        try:
            projected_kp2d = create_projection_visualization(data, output_path, image_size)
            if projected_kp2d is not None:
                all_projected_kp2d.append(projected_kp2d)
        except Exception as e:
            print(f"处理帧 {i} 时出错: {e}")
    
    # 生成投影误差报告
    if all_projected_kp2d and len(all_data) > 0:
        report_path = os.path.join(args.output_folder, "projection_report.txt")
        with open(report_path, 'w') as f:
            f.write("3D到2D投影报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"处理帧数: {len(all_projected_kp2d)}\n")
            f.write(f"输出文件夹: {args.output_folder}\n")
            f.write(f"图像尺寸: {image_size}\n")
            
            # 计算投影误差（如果有原始2D关键点）
            total_error = 0
            valid_frames = 0
            
            for i, data in enumerate(all_data):
                if i % args.frame_interval != 0:
                    continue
                    
                if 'pred_keypoints_2d' in data and i < len(all_projected_kp2d):
                    original_kp2d = data['pred_keypoints_2d']
                    projected_kp2d = all_projected_kp2d[i]
                    
                    if len(original_kp2d) == len(projected_kp2d):
                        error = np.sqrt(np.sum((original_kp2d - projected_kp2d) ** 2, axis=1)).mean()
                        total_error += error
                        valid_frames += 1
                        f.write(f"帧 {i}: 平均投影误差 = {error:.4f} 像素\n")
            
            if valid_frames > 0:
                avg_error = total_error / valid_frames
                f.write(f"\n平均投影误差: {avg_error:.4f} 像素\n")
                f.write(f"有效比较帧数: {valid_frames}\n")
    
    # 创建视频（如果启用）
    if args.create_video:
        video_path = os.path.join(args.output_folder, args.video_name)
        success = create_video_from_images(os.path.join(args.output_folder, "projection"), video_path, args.video_fps)
        video_path = os.path.join(args.output_folder, "compare_video.mp4")
        success = create_video_from_images(os.path.join(args.output_folder, "compare"), video_path, args.video_fps)

        
        if success:
            print(f"✅ 视频生成完成: {video_path}")
        else:
            print("❌ 视频生成失败")
    
    print(f"\n✅ 3D到2D投影完成！")
    print(f"输出文件保存在: {args.output_folder}")

if __name__ == "__main__":
    main()

# python project_3d_to_2d.py     --npz_folder /home/zhumeng/Documents/learn/3dbody/sam-3d-body/test/output_video/demo     --output_folder /home/zhumeng/Documents/learn/3dbody/sam-3d-body/test/3d_to_2d_projection     --create_video