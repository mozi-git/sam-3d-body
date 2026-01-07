# SAM-3D-Body → Holosoma Retargeting 快速流程

把视频里的人体动捕结果（SAM-3D-Body 推理）转换成 Holosoma 可用的 `(T, J, 3)` 关键点轨迹，并喂给 Holosoma 的 retarget pipeline。

## 0. 前置
- 已安装本仓依赖，能够运行 `demo_video.py`。
- Holosoma 仓库已就绪（见 holosoma 根目录 README 的 setup 脚本）。

## 1. 视频推理并导出 per-frame NPZ
`demo_video.py` 已支持每帧保存 `*_results.npz`（默认开启 `--save_npz`）。

```bash
python demo_video.py \
  --video_path /path/to/video.mp4 \
  --checkpoint_path ./checkpoints/sam-3d-body-dinov3/model.ckpt \
  --mhr_path ./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt \
  --output_folder ./output_video/demo \
  --detector_name vitdet \
  --segmentor_name sam2 \
  --fov_name moge2 \
  --use_mask \
  --save_npz           # 默认 True
```

输出：
- `./output_video/demo/<video_name>_2d.mp4`：2D 关键点可视化视频
- `./output_video/demo/<video_name>_3d_traj.png`：3D 轨迹图（首人）
- `./output_video/demo/<video_name>_<frame_idx>_results.npz`：每帧 raw 结果（用于下一步）

## 2. 导出 Holosoma 需要的 `(T, J, 3)`
使用桥接脚本 `tools/export_holosoma.py`：

```bash
python tools/export_holosoma.py \
  --results_dir ./output_video/demo \
  --output_path ./output_video/demo/my_motion.npy \
  --person_idx 0 \
  --root_center \
  --smooth_window 7
```

说明：
- 自动遍历 `*_results.npz`，按文件名排序。
- `person_idx` 选择每帧第几个检测到的人（默认 0）。
- `--root_center`：用 `root_index`（默认 0/骨盆）做根平移。
- `--smooth_window`：时间滑窗均值平滑，1 表示不平滑。
- 输出 `my_motion.npy` 形状 `(T, J, 3)`。

## 3. 在 Holosoma 里做 retarget
确保关节顺序/命名和 Holosoma 的 `config_types/data_type.py` 中的 `demo_joints` / `joints_mapping` 对齐；如不一致，需在该文件新增映射。

示例（单序列）：

```bash
cd /home/zhumeng/Documents/learn/3dbody/holosoma/src/holosoma_retargeting

python examples/robot_retarget.py \
  --data_path /path/to/motion_dir \        # 包含 my_motion.npy 的目录
  --task-type robot_only \
  --task-name my_motion \                  # 文件名（去后缀）匹配 my_motion.npy
  --data_format smplh \                    # 若自定义格式，请在 data_type.py 注册
  --retargeter.debug --retargeter.visualize
```

批处理可用 `examples/parallel_robot_retarget.py`。

## 4. 后续（可选）
- 用 Holosoma 的 `data_conversion/convert_data_format_mj.py` 统一帧率/格式，喂给 RL whole-body tracking 训练。
- 如果要处理多人，可在导出脚本里为不同 `person_idx` 各导出一份轨迹，再分别 retarget。

## 快速检查
- `demo_video.py` 能产出 `_results.npz`。
- `tools/export_holosoma.py` 输出 `(T, J, 3)`，长度 T 与有效帧数一致。
- Holosoma `robot_retarget.py` 能读取你的 `data_format`；如报关节映射问题，检查/扩展 `config_types/data_type.py`。

