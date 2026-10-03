import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))

"""Verify the chunked attention path is numerically equivalent to the full one.

Chunking only changes HOW the [B,H,Lq,Lk] matrix is produced, never the math:
softmax still covers all keys. This test forces chunking on tiny inputs and
compares against the unchunked result.

Run inside WSL:
    /home/ccb/trellis2-wsl-venv/bin/python -u _check_chunked_attn.py
"""
import os
import sys

CODE = "" + _ROOT + "/engine/code"
sys.path.insert(0, CODE)

import torch  # noqa: E402

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

import trellis2.modules.attention.full_attn as dense  # noqa: E402
import trellis2.modules.sparse.attention.full_attn as sparse  # noqa: E402

NO_CHUNK = 1 << 62


def run(mod, q, k, v, chunk_bytes):
    old = mod._ATTN_CHUNK_BYTES
    mod._ATTN_CHUNK_BYTES = chunk_bytes
    try:
        return mod._bmm_sdpa(q, k, v)
    finally:
        mod._ATTN_CHUNK_BYTES = old


def cmp(tag, mod, B, H, Lq, Lkv, C, dtype):
    q = torch.randn(B, H, Lq, C, device="cuda", dtype=dtype)
    k = torch.randn(B, H, Lkv, C, device="cuda", dtype=dtype)
    v = torch.randn(B, H, Lkv, C, device="cuda", dtype=dtype)
    ref = run(mod, q, k, v, NO_CHUNK)
    # 强制分块：阈值设得极小，保证 rows=1（最碎的分块）
    got = run(mod, q, k, v, 1)
    d = (ref.float() - got.float()).abs().max().item()
    rel = d / max(1e-6, ref.float().abs().max().item())
    ok = rel < 5e-3
    print(f"  {tag:34s} {(B,H,Lq,Lkv,C)} {str(dtype).replace('torch.',''):8s} "
          f"max_abs={d:.3e} rel={rel:.3e} {'OK' if ok else 'MISMATCH'}")
    return ok


print("=== dense _bmm_sdpa: 强制分块 vs 不分块 ===")
allok = True
allok &= cmp("dense 自注意力", dense, 1, 4, 128, 128, 64, torch.float16)
allok &= cmp("dense 自注意力 fp32", dense, 1, 4, 128, 128, 64, torch.float32)
allok &= cmp("dense 交叉注意力", dense, 1, 12, 4096, 1024, 128, torch.float16)
allok &= cmp("dense 交叉注意力 fp32", dense, 1, 12, 2048, 512, 128, torch.float32)

print("\n=== sparse _bmm_sdpa: 强制分块 vs 不分块 ===")
allok &= cmp("sparse 自注意力", sparse, 1, 12, 2048, 2048, 128, torch.float16)
allok &= cmp("sparse 自注意力 fp32", sparse, 1, 12, 2048, 2048, 128, torch.float32)
allok &= cmp("sparse 交叉注意力", sparse, 1, 12, 4096, 1024, 128, torch.float16)

print("\n=== 非连续输入（真实调用路径的 [T,H,C] -> transpose(0,1) 布局） ===")
for mod, tag in ((sparse, "sparse"), (dense, "dense")):
    T, H, C = 3000, 12, 128
    qkv = torch.randn(T, 3, H, C, device="cuda", dtype=torch.float16)
    q3, k3, v3 = qkv.unbind(dim=1)
    qi = q3.transpose(0, 1).unsqueeze(0)
    ki = k3.transpose(0, 1).unsqueeze(0)
    vi = v3.transpose(0, 1).unsqueeze(0)
    ref = run(mod, qi, ki, vi, NO_CHUNK)
    got = run(mod, qi, ki, vi, 1)
    d = (ref.float() - got.float()).abs().max().item()
    rel = d / max(1e-6, ref.float().abs().max().item())
    ok = rel < 5e-3
    allok &= ok
    print(f"  {tag:8s} 非连续 T={T}      max_abs={d:.3e} rel={rel:.3e} "
          f"{'OK' if ok else 'MISMATCH'}")

print("\n=== 大 L 时是否真的触发分块 ===")
for L in (4096, 10407):
    for H in (12,):
        q = torch.randn(1, H, L, 128, device="cuda", dtype=torch.float32)
        k = torch.randn(1, H, L, 128, device="cuda", dtype=torch.float32)
        v = torch.randn(1, H, L, 128, device="cuda", dtype=torch.float32)
        full_mb = 1 * H * L * L * 4 / 1024 ** 2
        thr_mb = sparse._ATTN_CHUNK_BYTES / 1024 ** 2
        print(f"  L={L:>6} H={H}: 完整矩阵 {full_mb:8.0f} MiB, "
              f"阈值 {thr_mb:.0f} MiB → "
              f"{'分块' if full_mb > thr_mb else '不分块'}")
        del q, k, v
        torch.cuda.empty_cache()

print(f"\n结论: {'全部等价 ✓' if allok else '有不等价 ✗'}")
