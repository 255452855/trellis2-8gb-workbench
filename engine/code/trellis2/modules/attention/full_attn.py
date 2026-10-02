# File: trellis2/modules/attention/full_attn.py
# trellis2/modules/attention/full_attn.py
from typing import *
import torch
import math
from . import config


__all__ = [
    'scaled_dot_product_attention',
]


def _naive_sdpa(q, k, v):
    """
    Naive implementation of scaled dot product attention.
    """
    q = q.permute(0, 2, 1, 3)   # [N, H, L, C]
    k = k.permute(0, 2, 1, 3)   # [N, H, L, C]
    v = v.permute(0, 2, 1, 3)   # [N, H, L, C]
    scale_factor = 1 / math.sqrt(q.size(-1))
    attn_weight = q @ k.transpose(-2, -1) * scale_factor
    attn_weight = torch.softmax(attn_weight, dim=-1)
    out = attn_weight @ v
    out = out.permute(0, 2, 1, 3)   # [N, L, H, C]
    return out


def _bmm_sdpa(q, k, v):
    """cuBLAS batched-matmul + softmax attention (q/k/v already [N, H, L, C]).

    ⚠️ Pascal 等老卡的救命实现。
    sm < 80 没有 Tensor Core，而 PyTorch 的 SDPA 默认会自动挑
    mem-efficient（cutlass fmha）后端——那是按 Tensor Core 设计的。
    GTX 1070 实测：4096x4096 fp16 自注意力走 mem-efficient 要 810ms
    （0.13 TFLOP/s），换成这里的 bmm+softmax 只要 57ms（1.8 TFLOP/s），
    快了 14 倍。整条 ss 前向从 33s 降到 5.2s。

    用 torch.matmul 而不是 torch.bmm，是为了避免把 [N, H, L, C] 这种
    非连续张量 reshape 成 [N*H, L, C] 时产生额外拷贝。

    ⚠️ scale 要折进 q，不能写成 matmul(...) * scale：
    后者会再开一份和注意力矩阵同样大的临时张量 —— 4096 个 token、
    16 个 head、fp32 时就是 1 GB，8GB 卡上很容易把后面 NAF 上采样挤爆。
    折进 q 只多一份 q 大小的临时张量（小两个数量级）。

    ⚠️ 不要在这里做"分块/online-softmax"之类的改造来省显存：
    试过按 query 行分块（数学上等价），虽然微基准验证通过，
    但整体跑起来仍然把 WSL2 虚拟机搞崩了两次。这条路径已经稳定，
    别动它。分辨率该由 pipeline 的显存守卫去降，不是靠这里硬省。
    """
    scale = 1.0 / math.sqrt(q.size(-1))
    attn = torch.matmul(q * scale, k.transpose(-2, -1))
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


@overload
def scaled_dot_product_attention(qkv: torch.Tensor) -> torch.Tensor:
    """
    Apply scaled dot product attention.

    Args:
        qkv (torch.Tensor): A [N, L, 3, H, C] tensor containing Qs, Ks, and Vs.
    """
    ...

@overload
def scaled_dot_product_attention(q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
    """
    Apply scaled dot product attention.

    Args:
        q (torch.Tensor): A [N, L, H, C] tensor containing Qs.
        kv (torch.Tensor): A [N, L, 2, H, C] tensor containing Ks and Vs.
    """
    ...

@overload
def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Apply scaled dot product attention.

    Args:
        q (torch.Tensor): A [N, L, H, Ci] tensor containing Qs.
        k (torch.Tensor): A [N, L, H, Ci] tensor containing Ks.
        v (torch.Tensor): A [N, L, H, Co] tensor containing Vs.

    Note:
        k and v are assumed to have the same coordinate map.
    """
    ...

def scaled_dot_product_attention(*args, **kwargs):
    arg_names_dict = {
        1: ['qkv'],
        2: ['q', 'kv'],
        3: ['q', 'k', 'v']
    }
    num_all_args = len(args) + len(kwargs)
    assert num_all_args in arg_names_dict, f"Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3"
    for key in arg_names_dict[num_all_args][len(args):]:
        assert key in kwargs, f"Missing argument {key}"

    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs['qkv']
        assert len(qkv.shape) == 5 and qkv.shape[2] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, L, 3, H, C]"
        device = qkv.device

    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs['q']
        kv = args[1] if len(args) > 1 else kwargs['kv']
        assert q.shape[0] == kv.shape[0], f"Batch size mismatch, got {q.shape[0]} and {kv.shape[0]}"
        assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, C]"
        assert len(kv.shape) == 5, f"Invalid shape for kv, got {kv.shape}, expected [N, L, 2, H, C]"
        device = q.device

    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs['q']
        k = args[1] if len(args) > 1 else kwargs['k']
        v = args[2] if len(args) > 2 else kwargs['v']
        assert q.shape[0] == k.shape[0] == v.shape[0], f"Batch size mismatch, got {q.shape[0]}, {k.shape[0]}, and {v.shape[0]}"
        assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, Ci]"
        assert len(k.shape) == 4, f"Invalid shape for k, got {k.shape}, expected [N, L, H, Ci]"
        assert len(v.shape) == 4, f"Invalid shape for v, got {v.shape}, expected [N, L, H, Co]"
        device = q.device    

    if config.BACKEND == 'xformers':
        if 'xops' not in globals():
            import xformers.ops as xops
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=2)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=2)
        out = xops.memory_efficient_attention(q, k, v)
    elif config.BACKEND == 'flash_attn':
        if 'flash_attn' not in globals():
            import flash_attn
        if num_all_args == 1:
            out = flash_attn.flash_attn_qkvpacked_func(qkv)
        elif num_all_args == 2:
            out = flash_attn.flash_attn_kvpacked_func(q, kv)
        elif num_all_args == 3:
            out = flash_attn.flash_attn_func(q, k, v)
    elif config.BACKEND == 'flash_attn_3':
        if 'flash_attn_3' not in globals():
            import flash_attn_interface as flash_attn_3
            if num_all_args == 1:
                out = flash_attn_3.flash_attn_qkvpacked_func(qkv)
            elif num_all_args == 2:
                k, v = kv.unbind(dim=2)
                out = flash_attn_3.flash_attn_func(q, k, v)
            elif num_all_args == 3:
                out = flash_attn_3.flash_attn_func(q, k, v)
    elif config.BACKEND == 'sdpa':
        if 'sdpa' not in globals():
            from torch.nn.functional import scaled_dot_product_attention as sdpa
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=2)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)   # [N, H, L, C]
        k = k.permute(0, 2, 1, 3)   # [N, H, L, C]
        v = v.permute(0, 2, 1, 3)   # [N, H, L, C]
        if config.MATH_SDPA:
            # 老卡（sm < 80）上必须绕开 mem-efficient 后端，否则慢 14 倍。
            out = _bmm_sdpa(q, k, v)
        else:
            out = sdpa(q, k, v)     # [N, H, L, C]
        out = out.permute(0, 2, 1, 3)   # [N, L, H, C]
    elif config.BACKEND == 'naive':
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=2)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=2)
        out = _naive_sdpa(q, k, v)
    else:
        raise ValueError(f"Unknown attention module: {config.BACKEND}")
    
    return out
