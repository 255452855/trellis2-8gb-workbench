import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))

"""Safe measurement of NAF's VRAM peak vs target size.

Only uses tiny targets (64..192) so it cannot blow up the machine, then fits
the quadratic law to extrapolate to 512 / 1024.

Run inside WSL:
    /home/ccb/trellis2-wsl-venv/bin/python -u _bench_naf.py
"""
import os
import sys
import gc
import torch

CODE = "" + _ROOT + "/engine/code"
sys.path.insert(0, CODE)
os.chdir(CODE)  # _load_naf looks for MODELS/NAF relative to cwd

NAF_DIR = os.path.join(CODE, "MODELS", "NAF")


def mb(x):
    return x / 1024 ** 2


def main():
    print("GPU:", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info()
    print(f"起始: free={mb(free):.0f}MiB / {mb(total):.0f}MiB\n")

    naf = torch.hub.load(NAF_DIR, "naf", source="local",
                         pretrained=True, device="cuda", trust_repo=True)
    naf.eval()
    naf.requires_grad_(False)
    gc.collect()
    torch.cuda.empty_cache()

    base = torch.cuda.memory_allocated()
    print(f"NAF 权重占用: {mb(base):.0f} MiB\n")

    print(f"{'target':>8} {'peak_MiB':>10} {'tokens':>9} {'MiB/token²':>12}")
    rows = []
    for t in (64, 96, 128, 160, 192):
        guide = torch.rand(1, 3, 512, 512, device="cuda")
        lr = torch.randn(1, 1024, 32, 32, device="cuda")
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            out = naf(guide, lr, (t, t))
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        rows.append((t, peak, out.shape))
        print(f"{t:>8} {mb(peak):>10.0f} {t*t:>9} "
              f"{mb(peak)/(t*t):>12.6f}")
        del out, guide, lr
        gc.collect()
        torch.cuda.empty_cache()

    # 拟合 peak = a * t^2 + b  （只取后 3 个点，避开小尺寸的固定开销）
    import numpy as np
    ts = np.array([r[0] for r in rows[-3:]], dtype=np.float64)
    ps = np.array([r[1] for r in rows[-3:]], dtype=np.float64)
    A = np.stack([ts ** 2, np.ones_like(ts)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, ps, rcond=None)
    print(f"\n拟合: peak_MiB ≈ {a:.6f} * target² + {b:.0f}")
    for t in (512, 1024):
        est = a * t * t + b
        print(f"  外推 target={t}: 约 {mb(est):.0f} MiB "
              f"({est/1024**3:.2f} GiB)")
    print("\n注意：外推值只用于定 guard 阈值，不是精确值。")


if __name__ == "__main__":
    main()
