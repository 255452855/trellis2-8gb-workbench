import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))

"""Why is the GPU at 100% util but only ~70W?

Runs several synthetic workloads back-to-back, printing a marker before each
so an external `nvidia-smi` sampler can attribute power readings.

Run inside WSL:
    /home/ccb/trellis2-wsl-venv/bin/python -u _bench_power.py
"""
import os
import sys
import time
import torch

CODE = "" + _ROOT + "/engine/code"
sys.path.insert(0, CODE)
os.environ.setdefault("ATTN_BACKEND", "sdpa")

dev = "cuda"


def mark(tag):
    print(f"###PHASE {tag}", flush=True)


def run(tag, fn, seconds=8.0, flops_per_call=None):
    mark(tag)
    # warmup
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    n = 0
    while time.time() - t0 < seconds:
        fn()
        n += 1
    torch.cuda.synchronize()
    el = time.time() - t0
    rate = ""
    if flops_per_call:
        tf = flops_per_call * n / el / 1e12
        rate = f"  {tf:6.2f} TFLOP/s"
    print(f"    {tag}: {el:.2f}s, {n} iters, {el/n*1e3:.2f} ms/iter{rate}",
          flush=True)
    torch.cuda.synchronize()


print("GPU:", torch.cuda.get_device_name(0),
      "| cap", torch.cuda.get_device_capability(0))
p = torch.cuda.get_device_properties(0)
print(f"SM count={p.multi_processor_count} "
      f"| 理论 fp32 峰值≈{p.multi_processor_count*128*2*1.9/1e3:.1f} TFLOP/s "
      f"(@1.9GHz)")

a_big = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
b_big = torch.randn(4096, 4096, device=dev, dtype=torch.float16)
run("A 大方阵 4096^3 fp16", lambda: a_big @ b_big,
    flops_per_call=2 * 4096 ** 3)

a_thin = torch.randn(4096, 128, device=dev, dtype=torch.float16)
b_thin = torch.randn(128, 4096, device=dev, dtype=torch.float16)
run("B 细K 4096x128x4096 fp16", lambda: a_thin @ b_thin,
    flops_per_call=2 * 4096 * 128 * 4096)

attn = torch.randn(12, 4096, 4096, device=dev, dtype=torch.float16)
run("C softmax [12,4096,4096] fp16", lambda: torch.softmax(attn, dim=-1))

q = torch.randn(1, 12, 4096, 128, device=dev, dtype=torch.float16)
k = torch.randn(1, 12, 4096, 128, device=dev, dtype=torch.float16)
v = torch.randn(1, 12, 4096, 128, device=dev, dtype=torch.float16)


def full_attn():
    s = 1.0 / (q.size(-1) ** 0.5)
    a = torch.matmul(q * s, k.transpose(-2, -1))
    a = torch.softmax(a, dim=-1)
    return torch.matmul(a, v)


run("D 完整注意力 1x12x4096 fp16", full_attn,
    flops_per_call=2 * 2 * 12 * 4096 * 4096 * 128)

# 真实模型
from safetensors.torch import load_file  # noqa: E402
import json  # noqa: E402
import trellis2.models as M  # noqa: E402

SS = os.path.join(CODE, "MODELS", "TRELLIS.2-4B", "ckpts",
                  "ss_flow_img_dit_1_3B_64_bf16")
cfg = json.load(open(SS + ".json"))
model = M.__getattr__(cfg["name"])(**cfg["args"])
model.load_state_dict(load_file(SS + ".safetensors"), strict=False)
model.eval()
if getattr(model, "dtype", None) == torch.bfloat16:
    model.dtype = torch.float16
for prm in model.parameters():
    if prm.dtype == torch.bfloat16:
        prm.data = prm.data.half()
for buf in model.buffers():
    if buf.dtype == torch.bfloat16:
        buf.data = buf.data.half()
model = model.to(dev)

res = model.resolution
x = torch.randn(1, model.in_channels, res, res, res, device=dev)
t = torch.full((1,), 500.0, device=dev)
cond = torch.randn(1, 1024, 1024, device=dev)

mark("E ss 模型单次前向")
with torch.no_grad():
    for _ in range(2):
        model(x, t, cond)
    torch.cuda.synchronize()
    t0 = time.time()
    n = 5
    for _ in range(n):
        model(x, t, cond)
    torch.cuda.synchronize()
    el = (time.time() - t0) / n
print(f"    ss forward: {el*1e3:.0f} ms  "
      f"(30 blocks → {el/30*1e3:.0f} ms/block)", flush=True)
print("###PHASE END", flush=True)
