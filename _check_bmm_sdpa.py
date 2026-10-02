"""Numerical equivalence check: _bmm_sdpa vs PyTorch math SDPA.

Run inside WSL:
    /home/ccb/trellis2-wsl-venv/bin/python -u _check_bmm_sdpa.py
"""
import math
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, "/mnt/d/IDM/TRELLIS2/engine/code")

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

from trellis2.modules.attention.full_attn import _bmm_sdpa as dense_bmm  # noqa
from trellis2.modules.sparse.attention.full_attn import _bmm_sdpa as sp_bmm  # noqa

torch.manual_seed(0)
dev = "cuda"

print("=== contiguous [B,H,L,C] ===")
for B, H, L, C in [(2, 4, 17, 32), (1, 12, 128, 64), (1, 3, 1, 8), (1, 2, 257, 128)]:
    q = torch.randn(B, H, L, C, device=dev, dtype=torch.float16)
    k = torch.randn(B, H, L, C, device=dev, dtype=torch.float16)
    v = torch.randn(B, H, L, C, device=dev, dtype=torch.float16)
    ref = F.scaled_dot_product_attention(q, k, v)
    mine = dense_bmm(q, k, v)
    d = (ref.float() - mine.float()).abs().max().item()
    rel = d / ref.float().abs().max().item()
    print(f"  {(B,H,L,C)}: max_abs_diff={d:.3e}  rel={rel:.3e}  "
          f"{'OK' if rel < 2e-3 else 'MISMATCH'}")

print("\n=== non-contiguous, [L,H,C] -> transpose(0,1)  (sparse path layout) ===")
for L, H, C in [(4096, 12, 128), (1000, 12, 128)]:
    # emulate qkv.feats [T, 3, H, C] -> slice dim1 -> [T, H, C] (non-contig)
    qkv = torch.randn(L, 3, H, C, device=dev, dtype=torch.float16)
    q3, k3, v3 = qkv.unbind(dim=1)
    print(f"  q3 contiguous? {q3.is_contiguous()}  strides={tuple(q3.stride())}")
    qi = q3.transpose(0, 1).unsqueeze(0)   # [1,H,L,C]
    ki = k3.transpose(0, 1).unsqueeze(0)
    vi = v3.transpose(0, 1).unsqueeze(0)
    ref = F.scaled_dot_product_attention(qi, ki, vi)
    mine = sp_bmm(qi, ki, vi)
    d = (ref.float() - mine.float()).abs().max().item()
    rel = d / ref.float().abs().max().item()
    print(f"  L={L}: max_abs_diff={d:.3e}  rel={rel:.3e}  "
          f"{'OK' if rel < 2e-3 else 'MISMATCH'}")

print("\n=== cross-attention (Lq != Lkv) ===")
B, H, Lq, Lkv, C = 1, 12, 4096, 1024, 128
q = torch.randn(B, H, Lq, C, device=dev, dtype=torch.float16)
k = torch.randn(B, H, Lkv, C, device=dev, dtype=torch.float16)
v = torch.randn(B, H, Lkv, C, device=dev, dtype=torch.float16)
ref = F.scaled_dot_product_attention(q, k, v)
mine = dense_bmm(q, k, v)
d = (ref.float() - mine.float()).abs().max().item()
print(f"  max_abs_diff={d:.3e}  {'OK' if d / ref.float().abs().max().item() < 2e-3 else 'MISMATCH'}")

print("\n=== 数值范围检查（fp16 是否溢出） ===")
q = torch.randn(1, 12, 4096, 128, device=dev, dtype=torch.float16) * 3
k = torch.randn(1, 12, 4096, 128, device=dev, dtype=torch.float16) * 3
v = torch.randn(1, 12, 4096, 128, device=dev, dtype=torch.float16)
mine = dense_bmm(q, k, v)
print(f"  finite={bool(torch.isfinite(mine).all())}  "
      f"max={mine.abs().max().item():.4f}")
