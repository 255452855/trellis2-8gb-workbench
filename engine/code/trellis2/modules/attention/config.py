# trellis2/modules/attention/config.py
from typing import *

BACKEND = None  # resolved in __from_env
DEBUG = False
# 老卡（sm < 80）上是否绕开 PyTorch 的 SDPA 后端分派，改用 bmm+softmax。
# 见 full_attn._bmm_sdpa 的注释：Pascal 上默认后端慢 14 倍。
MATH_SDPA = False

def _detect_best_backend():
    """Pick the best available attention backend based on GPU compute capability."""
    import torch
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        sm = major * 10 + minor
        if sm >= 80:
            return 'flash_attn'
        else:
            # sm < 80（Pascal/Volta/Turing）：xformers 的 fmha 要求 capability > (8,0)，
            # flash_attn 同样要 sm_80+。老卡上只有 PyTorch 原生 sdpa 能跑，
            # 而 sdpa 内部还要再绕开 mem-efficient（见 _configure_old_gpu_sdpa）。
            return 'sdpa'
    return 'flash_attn'


def _configure_old_gpu_sdpa():
    """Pre-Ampere：禁用 flash / mem-efficient 的 SDPA 后端，只留 math。

    PyTorch 的 SDPA 在 sm < 80 上会“自动”选中 mem-efficient（cutlass fmha），
    那个 kernel 是按 Tensor Core 设计的，在没有 Tensor Core 的 Pascal 上
    实测只有 0.13 TFLOP/s —— 4096x4096 的 fp16 自注意力要 810ms。
    把后端限制成 math（cuBLAS bmm + softmax）后是 1.8 TFLOP/s，快 14 倍。

    这里改的是全局开关，配合 full_attn 里的 _bmm_sdpa 双保险：
    即使某处仍然调用了 torch.nn.functional.scaled_dot_product_attention，
    也不会再掉进 mem-efficient 的坑。
    """
    import os
    import torch

    global MATH_SDPA

    if os.environ.get("TRELLIS2_MATH_SDPA", "1").lower() in ("0", "false", "no"):
        MATH_SDPA = False
        return
    if not torch.cuda.is_available():
        MATH_SDPA = False
        return
    if torch.cuda.get_device_capability()[0] >= 8:
        # Ampere 及以上：mem-efficient / flash 都很好用，保持默认。
        MATH_SDPA = False
        return

    MATH_SDPA = True
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass
    print(
        "[ATTENTION] 老架构 GPU：已禁用 flash/mem-efficient SDPA，"
        "注意力改用 bmm+softmax（快约 14 倍）"
    )


def __from_env():
    import os
    
    global BACKEND
    global DEBUG
    
    env_attn_backend = os.environ.get('ATTN_BACKEND')
    env_attn_debug = os.environ.get('ATTN_DEBUG')
    
    if env_attn_backend is not None and env_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3', 'sdpa', 'naive']:
        BACKEND = env_attn_backend
    else:
        BACKEND = _detect_best_backend()
    
    if env_attn_debug is not None:
        DEBUG = env_attn_debug == '1'

    _configure_old_gpu_sdpa()

    print(f"[ATTENTION] Using backend: {BACKEND}")
    print("Please wait...")
        

__from_env()
    

def set_backend(backend: Literal['xformers', 'flash_attn']):
    global BACKEND
    BACKEND = backend

def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug