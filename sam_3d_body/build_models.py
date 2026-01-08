# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import torch

from .models.meta_arch import SAM3DBody
from .utils.config import get_config
from .utils.checkpoint import load_state_dict


def load_sam_3d_body(checkpoint_path: str = "", device: str = "cuda", mhr_path: str = ""):
    print("Loading SAM 3D Body model...")
    
    # Check the current directory, and if not present check the parent dir.
    model_cfg = os.path.join(os.path.dirname(checkpoint_path), "model_config.yaml")
    if not os.path.exists(model_cfg):
        # Looks at parent dir
        model_cfg = os.path.join(
            os.path.dirname(os.path.dirname(checkpoint_path)), "model_config.yaml"
        )

    model_cfg = get_config(model_cfg)

    # Disable face for inference
    model_cfg.defrost()
    model_cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = mhr_path
    model_cfg.freeze()

    # Initialze the model
    model = SAM3DBody(model_cfg)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    load_state_dict(model, state_dict, strict=False)

    model = model.to(device)
    model.eval()
    return model, model_cfg


def _hf_download(repo_id):
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(repo_id=repo_id)
    return os.path.join(local_dir, "model.ckpt"), os.path.join(local_dir, "assets", "mhr_model.pt")

import os
# 导入 ModelScope 的下载函数
from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download

def _ms_download(model_id):
    """
    替代 _hf_download 函数，从 ModelScope 下载模型。
    model_id: ModelScope 的模型ID，例如 'damo/cv_sam-3d-body-dinov3' (需要确认)
    """
    # 从 ModelScope 下载模型仓库
    local_dir = ms_snapshot_download(model_id=model_id, cache_dir=None)
    
    # 根据下载后的文件结构，构造正确的文件路径
    # 注意：以下文件路径假设与 Hugging Face 版本结构一致，可能需要调整
    ckpt_path = os.path.join(local_dir, "model.ckpt")
    mhr_path = os.path.join(local_dir, "assets", "mhr_model.pt")
    
    # 一个更安全的做法是，如果文件名不确定，可以搜索目录
    # import glob
    # ckpt_files = glob.glob(os.path.join(local_dir, "*.ckpt"))
    # if ckpt_files:
    #     ckpt_path = ckpt_files[0]
    
    return ckpt_path, mhr_path

def load_sam_3d_body_ms(model_id, **kwargs):
    """
    替代 load_sam_3d_body_hf 函数，从 ModelScope 加载模型。
    """
    ckpt_path, mhr_path = _ms_download(model_id)
    return load_sam_3d_body(checkpoint_path=ckpt_path, mhr_path=mhr_path)

def load_sam_3d_body_hf(repo_id, **kwargs):
    ckpt_path, mhr_path = _ms_download(repo_id)
    return load_sam_3d_body(checkpoint_path=ckpt_path, mhr_path=mhr_path)
