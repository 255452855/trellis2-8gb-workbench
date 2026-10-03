"""Which SDPA backend is fastest on Pascal sm_61 for fp16?

The ss forward spends 835ms on a [1,12,4096,4096] self-attention that
should take ~25ms.  torch profiler says aten::_efficient_attention_forward
is being used.  Test the alternatives.

Run inside WSL:
    /home/ccb/trellis2-wsl-venv/bin/python -u _bench_attn.py
"""
import time
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

dev = "cuda"
dt = torch.float16


def bench(label, fn, n=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    el = (time.time() - t0) / n
    print(f"  {label:44s} {el*1e3:9.2f} ms")
    return el


def manual(q, k, v, scale=None):
    """Plain cuBLAS bmm + softmax."""
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    a = torch.bmm(q.reshape(-1, q.shape[-2], q.shape[-1]),
                  k.reshape(-1, k.shape[-2], k.shape[-1]).transpose(-2, -1))
    a = (a * scale)
    a = torch.softmax(a, dim=-1)
    o = torch.bmm(a, v.reshape(-1, v.shape[-2], v.shape[-1]))
    return o.reshape(q.shape[0], q.shape[1], q.shape[-2], v.shape[-1])


print("GPU:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("torch", torch.__version__)

print("\n--- SDPA backend availability on this device ---")
for name, fn in [
    ("flash_sdp", torch.backends.cuda.flash_sdp_enabled),
    ("mem_efficient_sdp", torch.backends.cuda.mem_efficient_sdp_enabled),
    ("math_sdp", torch.backends.cuda.math_sdp_enabled),
]:
    try:
        print(f"  {name}: {fn()}")
    except Exception as e:
        print(f"  {name}: {e}")

for tag, (Lq, Lkv) in [("self-attn 4096x4096", (4096, 4096)),
                       ("cross-attn 4096x1024", (4096, 1024))]:
    print(f"\n=== {tag}  (B=1, H=12, C=128, fp16) ===")
    q = torch.randn(1, 12, Lq, 128, device=dev, dtype=dt)
    k = torch.randn(1, 12, Lkv, 128, device=dev, dtype=dt)
    v = torch.randn(1, 12, Lkv, 128, device=dev, dtype=dt)

    flops = 2 * 2 * 12 * Lq * Lkv * 128
    try:
        e = bench("A. F.sdpa default (auto backend)",
                  lambda: F.scaled_dot_product_attention(q, k, v))
        print(f"     -> {flops/e/1e12:.2f} TFLOP/s")
    except Exception as ex:
        print("  A failed:", str(ex)[:120])

    try:
        e = bench("B. F.sdpa forced MATH",
                  lambda: F.scaled_dot_product_attention(q, k, v),
                  )
    except Exception as ex:
        print("  B failed:", str(ex)[:120])
    # redo B properly with context manager
    def _math():
        with sdpa_kernel([SDPBackend.MATH]):
            return F.scaled_dot_product_attention(q, k, v)
    try:
        e = bench("B2. sdpa_kernel([MATH])", _math)
        print(f"     -> {flops/e/1e12:.2f} TFLOP/s")
    except Exception as ex:
        print("  B2 failed:", str(ex)[:120])

    def _eff():
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            return F.scaled_dot_product_attention(q, k, v)
    try:
        e = bench("C. sdpa_kernel([EFFICIENT_ATTENTION])", _eff)
        print(f"     -> {flops/e/1e12:.2f} TFLOP/s")
    except Exception as ex:
        print("  C failed:", str(ex)[:120])

    try:
        e = bench("D. manual bmm + softmax (cuBLAS)", lambda: manual(q, k, v))
        print(f"     -> {flops/e/1e12:.2f} TFLOP/s")
    except Exception as ex:
        print("  D failed:", str(ex)[:120])

    # manual with fp16 softmax over fp16 (avoid upcast copies)
    def manual32(q, k, v):
        s = 1.0 / (q.shape[-1] ** 0.5)
        a = torch.bmm(q.reshape(-1, Lq, 128),
                      k.reshape(-1, Lkv, 128).transpose(-2, -1)) * s
        a = torch.softmax(a, dim=-1)
        return torch.bmm(a, v.reshape(-1, Lkv, 128)).reshape(1, 12, Lq, 128)
    try:
        e = bench("E. manual bmm, no .contiguous()", lambda: manual32(q, k, v))
        print(f"     -> {flops/e/1e12:.2f} TFLOP/s")
    except Exception as ex:
        print("  E failed:", str(ex)[:120])

print("\n--- global switches ---")
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
print("  after disabling flash+mem_efficient:")
q = torch.randn(1, 12, 4096, 128, device=dev, dtype=dt)
k = torch.randn(1, 12, 4096, 128, device=dev, dtype=dt)
v = torch.randn(1, 12, 4096, 128, device=dev, dtype=dt)
flops = 2 * 2 * 12 * 4096 * 4096 * 128
e = bench("  F.sdpa (math only, global)", lambda: F.scaled_dot_product_attention(q, k, v))
print(f"     -> {flops/e/1e12:.2f} TFLOP/s")
