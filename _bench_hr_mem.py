"""Measure the HR shape model's REAL peak VRAM vs token count — safely.

Only small token counts (<= 3000) are used, and the script hard-stops if free
VRAM drops below a floor, so it cannot oversubscribe and take the machine down.

Goal: get the real a + b·N + c·N² law instead of guessing, so the pipeline's
memory guard can be based on measurement rather than assumption.

Run inside WSL:
    /home/ccb/trellis2-wsl-venv/bin/python -u _bench_hr_mem.py
"""
import gc
import json
import os
import sys

CODE = "/mnt/d/IDM/TRELLIS2/engine/code"
sys.path.insert(0, CODE)
os.environ.setdefault("ATTN_BACKEND", "sdpa")

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
import trellis2.models as M  # noqa: E402
from trellis2.modules import sparse as sp  # noqa: E402
from trellis2.modules.utils import convert_module_to  # noqa: E402
import functools  # noqa: E402

CKPT = os.path.join(CODE, "MODELS", "Pixal3D-fp16", "ckpts",
                    "slat_flow_img2shape_dit_1_3B_1024_bf16")

# 安全底线：剩余显存低于这个值就立刻停，绝不冒险
FREE_FLOOR_MB = 2200
MAX_TOKENS = 3000


def mb(x):
    return x / 1024 ** 2


def free_mb():
    return torch.cuda.mem_get_info()[0] / 1024 ** 2


def build_model():
    cfg = json.load(open(CKPT + ".json"))
    model = M.__getattr__(cfg["name"])(**cfg["args"])
    model.load_state_dict(load_file(CKPT + ".safetensors"), strict=False)
    model.eval()
    if getattr(model, "dtype", None) == torch.bfloat16:
        model.dtype = torch.float16
    for prm in model.parameters():
        if prm.dtype == torch.bfloat16:
            prm.data = prm.data.half()
    for buf in model.buffers():
        if buf.dtype == torch.bfloat16:
            buf.data = buf.data.half()
    # 复现 _promote_pascal_tail：最后 6 个 block 提成 fp32
    for block in model.blocks[-6:]:
        block.apply(functools.partial(convert_module_to, dtype=torch.float32))
    return model, cfg["args"]


def make_inputs(model, n_tokens, device="cuda"):
    args = model.__dict__
    in_ch = getattr(model, "in_channels", 32)
    # 合法的唯一 3D 坐标（分辨率 64）
    grid = 64
    picks = torch.randint(0, grid ** 3, (n_tokens * 2,))
    picks = torch.unique(picks)[:n_tokens]
    while picks.numel() < n_tokens:
        extra = torch.randint(0, grid ** 3, (n_tokens,))
        picks = torch.unique(torch.cat([picks, extra]))[:n_tokens]
    z = picks % grid
    y = (picks // grid) % grid
    x = picks // (grid * grid)
    coords = torch.stack([torch.zeros_like(x), x, y, z], dim=1).int()
    coords = coords[torch.randperm(coords.shape[0])].contiguous().to(device)

    x_in = sp.SparseTensor(
        feats=torch.randn(coords.shape[0], in_ch, device=device),
        coords=coords,
    )
    # 全局条件：shape_1024 用 image_size=1024 → 64x64 = 4096 个 patch token
    global_cond = torch.randn(1, 4096, 1024, device=device)
    proj_in = getattr(model, "proj_in_channels", 2048) or 2048
    proj_cond = sp.SparseTensor(
        feats=torch.randn(coords.shape[0], proj_in, device=device),
        coords=coords,
    )
    t = torch.full((1,), 500.0, device=device)
    return x_in, t, {"global": global_cond, "proj": proj_cond}


def main():
    print("GPU:", torch.cuda.get_device_name(0),
          f"| 起始 free={free_mb():.0f}MiB")

    model, args = build_model()
    print("模型配置:", json.dumps(args))
    model = model.to("cuda")
    torch.cuda.synchronize()
    weights_mb = torch.cuda.memory_allocated() / 1024 ** 2
    print(f"权重 + 输入层占用: {weights_mb:.0f} MiB")
    print(f"（注意：checkpoint 文件 2647 MiB，fp32 tail 会再多约 550 MiB）\n")

    rows = []
    print(f"{'tokens':>8} {'峰值MiB':>10} {'净激活MiB':>11} {'剩余MiB':>9} "
          f"{'attention满矩阵MiB':>18}")
    for n in (500, 1000, 1500, 2000, 2500, 3000):
        if n > MAX_TOKENS:
            break
        if free_mb() < FREE_FLOOR_MB:
            print(f"  ⛔ 剩余显存 {free_mb():.0f}MiB < 底线 {FREE_FLOOR_MB}MiB，"
                  f"安全起见停止")
            break
        x_in, t, cond = make_inputs(model, n)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model(x_in, t, cond)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1024 ** 2
        act = peak - weights_mb
        # 完整注意力矩阵（fp32 tail 所以按 4 字节）
        full_attn = 12 * n * n * 4 / 1024 ** 2
        rows.append((n, peak, act))
        print(f"{n:>8} {peak:>10.0f} {act:>11.0f} {free_mb():>9.0f} "
              f"{full_attn:>18.0f}")
        del x_in, t, cond
        gc.collect()
        torch.cuda.empty_cache()

    if len(rows) >= 3:
        import numpy as np
        ns = np.array([r[0] for r in rows], dtype=np.float64)
        ps = np.array([r[2] for r in rows], dtype=np.float64)  # 净激活
        # 拟合 act = a*N + c*N²
        A = np.stack([ns, ns ** 2], axis=1)
        (a, c), *_ = np.linalg.lstsq(A, ps, rcond=None)
        print(f"\n拟合: 净激活_MiB ≈ {a:.4f}·N + {c:.8f}·N²")
        print(f"  二次项系数 {c:.8f} vs 理论 12·4/1024² = "
              f"{12 * 4 / 1024 ** 2:.8f}")
        for n in (5556, 7707, 10407):
            print(f"  外推 N={n}: 激活 ≈ {a*n + c*n*n:.0f} MiB "
                  f"→ 总需求 ≈ {weights_mb + a*n + c*n*n:.0f} MiB")
        print(f"\n8GB 卡可用约 8192 - 1200(上下文/其他) ≈ 7000 MiB")
        print("（外推仅供参考，不是精确值）")


if __name__ == "__main__":
    main()
