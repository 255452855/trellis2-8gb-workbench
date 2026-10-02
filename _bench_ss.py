import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))

"""Profile one forward of the sparse-structure flow model.

Faithful to what the app actually does at load time:
  * model built from <name>.json, then load_state_dict (copy_ keeps ctor dtypes)
  * convert_to(cfg dtype) in __init__ only touches .blocks
  * pipeline_worker / _normalize_loaded_dtype then flips bf16 -> fp16
So: blocks = fp16, everything else (input_layer, out_layer, t_embedder,
adaLN_modulation, norms) = fp32.  Noise x is fp32.

Run inside WSL:
    /home/ccb/trellis2-wsl-venv/bin/python -u _bench_ss.py
"""
import os
import sys
import json
import time
import collections

CODE = "" + _ROOT + "/engine/code"
sys.path.insert(0, CODE)
os.environ.setdefault("ATTN_BACKEND", "sdpa")

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
import trellis2.models as M  # noqa: E402

MODELS = os.path.join(CODE, "MODELS")

SS_T2 = os.path.join(MODELS, "TRELLIS.2-4B", "ckpts", "ss_flow_img_dit_1_3B_64_bf16")
SS_PX = os.path.join(MODELS, "Pixal3D-fp16", "ckpts", "ss_flow_img_dit_1_3B_64_bf16")


def load_ss(path, force_fp16=True):
    cfg = json.load(open(path + ".json"))
    m = M.__getattr__(cfg["name"])(**cfg["args"])
    m.load_state_dict(load_file(path + ".safetensors"), strict=False)
    m.eval()
    # mirror _normalize_loaded_dtype() (bf16 -> fp16)
    if force_fp16 and getattr(m, "dtype", None) == torch.bfloat16:
        m.dtype = torch.float16
    for p in m.parameters():
        if p.dtype == torch.bfloat16:
            p.data = p.data.half()
    for b in m.buffers():
        if b.dtype == torch.bfloat16:
            b.data = b.data.half()
    return m, cfg["args"]


def dtype_hist(mod):
    c = collections.Counter()
    for p in mod.parameters():
        c[str(p.dtype).replace("torch.", "")] += p.numel()
    for b in mod.buffers():
        c[str(b.dtype).replace("torch.", "") + "(buf)"] += b.numel()
    return dict(c)


def run(tag, path, force_fp16=True, attn_override=None):
    print("#" * 78)
    print(f"# {tag}")
    print("#" * 78)
    m, args = load_ss(path, force_fp16=force_fp16)
    mode = args.get("image_attn_mode", "cross")
    if attn_override:
        # swap ProjectAttention back to a plain cross-attn block to isolate cost
        for blk in m.blocks:
            ca = blk.cross_attn
            if hasattr(ca, "cross_attn_block"):
                blk.cross_attn = ca.cross_attn_block
        mode = attn_override
        print(f"  [override] image_attn_mode -> {attn_override}")
    print("  blocks dtype:", dtype_hist(m).get("float16", 0), "fp16 params")
    print("  full dtype hist:", dtype_hist(m))
    b0 = m.blocks[0]
    ca = b0.cross_attn
    if hasattr(ca, "cross_attn_block"):
        print(f"  proj_linear: w={ca.proj_linear.weight.dtype} "
              f"b={ca.proj_linear.bias.dtype} "
              f"shape={tuple(ca.proj_linear.weight.shape)}")
    print(f"  input_layer: w={m.input_layer.weight.dtype} b={m.input_layer.bias.dtype}")
    print(f"  rope_phases: {m.rope_phases.dtype} {tuple(m.rope_phases.shape)} "
          f"| model.dtype={m.dtype}")

    m = m.to("cuda")
    print(f"  after .to(cuda) rope_phases.device={m.rope_phases.device}")

    res = m.resolution
    x = torch.randn(1, m.in_channels, res, res, res, device="cuda")  # fp32, like the app
    t = torch.full((1,), 500.0, device="cuda")
    Lg = 1024
    if mode == "proj":
        cond = {
            "global": torch.randn(1, Lg, 1024, device="cuda"),
            "proj": torch.randn(1, res ** 3, 1024, device="cuda"),
        }
    else:
        cond = torch.randn(1, Lg, 1024, device="cuda")

    with torch.no_grad():
        for _ in range(2):
            m(x, t, cond)
        torch.cuda.synchronize()
        t0 = time.time()
        n = 5
        for _ in range(n):
            m(x, t, cond)
        torch.cuda.synchronize()
        el = (time.time() - t0) / n
    print(f"\n>>> FORWARD TIME: {el:.3f} s   (x={tuple(x.shape)} dtype={x.dtype}, "
          f"Lg={Lg}, proj_tokens={res**3 if mode=='proj' else 0})")
    print(f"    peak VRAM: {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")

    from torch.profiler import profile, ProfilerActivity
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            m(x, t, cond)
            torch.cuda.synchronize()
    print("\n--- top CUDA ops ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    print()

    del m, x, cond, prof
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def main():
    print("GPU:", torch.cuda.get_device_name(0),
          "| cap", torch.cuda.get_device_capability(0),
          "| torch", torch.__version__, "| cuda", torch.version.cuda)

    # raw matmul throughput for the three candidate dtypes
    for dt in (torch.float16, torch.bfloat16, torch.float32):
        try:
            a = torch.randn(4096, 1536, device="cuda", dtype=dt)
            b = torch.randn(1536, 1536, device="cuda", dtype=dt)
            for _ in range(3):
                a @ b
            torch.cuda.synchronize()
            t0 = time.time()
            n = 30
            for _ in range(n):
                a @ b
            torch.cuda.synchronize()
            el = (time.time() - t0) / n
            print(f"[matmul] {str(dt):18s} 4096x1536x1536: {el*1e3:8.3f} ms = "
                  f"{2*4096*1536*1536/el/1e12:6.2f} TFLOP/s")
        except Exception as e:
            print(f"[matmul] {dt}: FAILED {type(e).__name__}: {str(e)[:80]}")
    print()

    run("TRELLIS.2  cross  (blocks fp16)", SS_T2, force_fp16=True)
    run("Pixal3D    proj   (blocks fp16)", SS_PX, force_fp16=True)
    run("Pixal3D    cross  (proj stripped)", SS_PX, force_fp16=True,
        attn_override="cross")


if __name__ == "__main__":
    main()
