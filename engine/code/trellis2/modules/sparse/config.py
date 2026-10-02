# trellis2/modules/sparse/config.py
from typing import *

CONV = 'flex_gemm' 
DEBUG = False
ATTN = None  # resolved in __from_env
# 老卡（sm < 80）上是否绕开 PyTorch 的 SDPA 后端分派，改用 bmm+softmax。
MATH_SDPA = False

def _detect_best_backend():
    """Pick the best available sparse attention backend based on GPU compute capability."""
    import torch
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        sm = major * 10 + minor
        if sm >= 80:
            return 'flash_attn'
        else:
            # sm < 80（Pascal/Volta/Turing 等）：
            # xformers 的 fmha 后端要求 capability > (8,0)，且 bf16 要 A100+，
            # 在老卡上会直接 NotImplementedError；改用 PyTorch 原生 sdpa。
            return 'sdpa'
    return 'flash_attn'


def _configure_old_gpu_sdpa():
    """Pre-Ampere：禁用 flash / mem-efficient 的 SDPA 后端，只留 math。

    PyTorch 的 SDPA 在 sm < 80 上会“自动”选中 mem-efficient（cutlass fmha），
    那个 kernel 是按 Tensor Core 设计的，在没有 Tensor Core 的 Pascal 上
    实测只有 0.13 TFLOP/s。限制成 math（cuBLAS bmm + softmax）后快 14 倍。
    """
    import os
    import torch

    global MATH_SDPA

    if os.environ.get("TRELLIS2_MATH_SDPA", "1").lower() in ("0", "false", "no"):
        MATH_SDPA = False
        return
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] >= 8:
        MATH_SDPA = False
        return

    MATH_SDPA = True
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass


def __from_env():
    import os
    
    global CONV
    global DEBUG
    global ATTN
    
    env_sparse_conv_backend = os.environ.get('SPARSE_CONV_BACKEND')
    env_sparse_debug = os.environ.get('SPARSE_DEBUG')
    env_sparse_attn_backend = os.environ.get('SPARSE_ATTN_BACKEND')
    if env_sparse_attn_backend is None:
        env_sparse_attn_backend = os.environ.get('ATTN_BACKEND')

    if env_sparse_conv_backend is not None and env_sparse_conv_backend in ['none', 'spconv', 'torchsparse', 'flex_gemm']:
        CONV = env_sparse_conv_backend
    if env_sparse_debug is not None:
        DEBUG = env_sparse_debug == '1'
    if env_sparse_attn_backend is not None and env_sparse_attn_backend in ['xformers', 'flash_attn', 'flash_attn_3', 'sdpa']:
        ATTN = env_sparse_attn_backend
    else:
        ATTN = _detect_best_backend()

    _configure_old_gpu_sdpa()

    print(f"[SPARSE] Conv backend: {CONV}; Attention backend: {ATTN}")
        

__from_env()
    

def set_conv_backend(backend: Literal['none', 'spconv', 'torchsparse', 'flex_gemm']):
    global CONV
    CONV = backend

def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug

def set_attn_backend(backend: Literal['xformers', 'flash_attn']):
    global ATTN
    ATTN = backend