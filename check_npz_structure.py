import numpy as np
import os

def check_npz_structure(npz_folder):
    """检查npz文件的结构和内容"""
    
    npz_files = sorted([f for f in os.listdir(npz_folder) if f.endswith('.npz')])
    
    print(f"找到 {len(npz_files)} 个npz文件:")
    for npz_file in npz_files:
        print(f"  - {npz_file}")
    
    if not npz_files:
        return
    
    # 检查第一个文件的结构
    first_file = os.path.join(npz_folder, npz_files[0])
    data = np.load(first_file, allow_pickle=True)
    
    print(f"\n文件结构分析 ({npz_files[0]}):")
    print(f"文件中的数组: {list(data.keys())}")
    
    if 'outputs' in data:
        outputs = data['outputs']
        print(f"outputs 类型: {type(outputs)}")
        print(f"outputs 形状: {outputs.shape if hasattr(outputs, 'shape') else 'N/A'}")
        
        if len(outputs) > 0:
            first_output = outputs[0]
            print(f"第一个输出的类型: {type(first_output)}")
            
            if hasattr(first_output, '__len__') and len(first_output) > 0:
                print(f"第一个输出的键: {list(first_output.keys())}")
                
                # 检查关键点数据
                if 'pred_keypoints_3d' in first_output:
                    kp3d = first_output['pred_keypoints_3d']
                    print(f"pred_keypoints_3d 形状: {kp3d.shape}")
                    print(f"pred_keypoints_3d 数据类型: {kp3d.dtype}")
                    print(f"前5个关键点示例:")
                    for i in range(min(5, len(kp3d))):
                        print(f"  关节 {i}: {kp3d[i]}")
                
                # 检查其他可能的数据
                for key in first_output.keys():
                    if key != 'pred_keypoints_3d':
                        value = first_output[key]
                        if hasattr(value, 'shape'):
                            print(f"{key} 形状: {value.shape}")
                        else:
                            print(f"{key}: {type(value)}")
        else:
            print("outputs 为空")
    
    # 检查所有文件的一致性
    print(f"\n文件间一致性检查:")
    all_have_outputs = True
    all_have_keypoints = True
    
    for npz_file in npz_files:
        file_path = os.path.join(npz_folder, npz_file)
        data = np.load(file_path, allow_pickle=True)
        
        if 'outputs' not in data:
            all_have_outputs = False
            print(f"  {npz_file}: 缺少 outputs")
            continue
            
        outputs = data['outputs']
        if len(outputs) == 0:
            print(f"  {npz_file}: outputs 为空")
            continue
            
        if 'pred_keypoints_3d' not in outputs[0]:
            all_have_keypoints = False
            print(f"  {npz_file}: 缺少 pred_keypoints_3d")
        else:
            kp3d = outputs[0]['pred_keypoints_3d']
            print(f"  {npz_file}: 有 {len(kp3d)} 个关键点")
    
    if all_have_outputs and all_have_keypoints:
        print("✅ 所有文件结构一致")
    else:
        print("⚠️ 文件结构不一致")

def main():
    npz_folder = "/home/zhumeng/Documents/learn/3dbody/sam-3d-body/test/output_video/demo"
    check_npz_structure(npz_folder)

if __name__ == "__main__":
    main()

# 文件结构分析 (test_000000_results.npz):
# 文件中的数组: ['outputs']
# outputs 类型: <class 'numpy.ndarray'>
# outputs 形状: (1,)
# 第一个输出的类型: <class 'dict'>
# 第一个输出的键: ['bbox', 'focal_length', 'pred_keypoints_3d', 'pred_keypoints_2d', 'pred_vertices', 'pred_cam_t', 'pred_pose_raw', 'global_rot', 'body_pose_params', 'hand_pose_params', 'scale_params', 'shape_params', 'expr_params', 'mask', 'pred_joint_coords', 'pred_global_rots', 'mhr_model_params', 'lhand_bbox', 'rhand_bbox']
# pred_keypoints_3d 形状: (70, 3)
# pred_keypoints_3d 数据类型: float32
# 前5个关键点示例:
#   关节 0: [ 0.02829324 -1.4793346  -0.33799043]
#   关节 1: [ 0.06653047 -1.5264676  -0.33918506]
#   关节 2: [ 0.00213267 -1.5325384  -0.32562804]
#   关节 3: [ 0.1342294  -1.5593415  -0.25731382]
#   关节 4: [-0.01848983 -1.5725983  -0.22434863]
# bbox 形状: (4,)
# focal_length 形状: ()
# pred_keypoints_2d 形状: (70, 2)
# pred_vertices 形状: (18439, 3)
# pred_cam_t 形状: (3,)
# pred_pose_raw 形状: (266,)
# global_rot 形状: (3,)
# body_pose_params 形状: (133,)
# hand_pose_params 形状: (108,)
# scale_params 形状: (28,)
# shape_params 形状: (45,)
# expr_params 形状: (72,)
# mask: <class 'NoneType'>
# pred_joint_coords 形状: (127, 3)
# pred_global_rots 形状: (127, 3, 3)
# mhr_model_params 形状: (204,)
# lhand_bbox 形状: (4,)
# rhand_bbox 形状: (4,)