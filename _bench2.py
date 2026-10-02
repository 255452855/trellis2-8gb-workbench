"""Pinpoint where the ss-forward CPU time goes.

1. Per-block CUDA-event timing (30 blocks)
2. Sub-section timing inside one block
3. profiler sorted by SELF CPU time (the earlier run showed 32.7s of CPU
   that was NOT in the top-15 by cuda_time)

Run inside WSL:
    /home/ccb/trellis2-wsl-venv/bin/python -u _bench2.py
"""
import os
import sys
import json
import time

CODE = "/mnt/d/IDM/TRELLIS2/engine/code"
sys.path.insert(0, CODE)
os.environ.setdefault("ATTN_BACKEND", "sdpa")

import torch  # noqa: E402

if os.environ.get("FORCE_MATH_SDPA", "1") == "1":
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    print("[FIX] forced MATH sdpa backend")

from safetensors.torch import load_file  # noqa: E402
import trellis2.models as M  # noqa: E402

PATH = os.path.join(CODE, "MODELS", "TRELLIS.2-4B", "ckpts",
                    "ss_flow_img_dit_1_3B_64_bf16")


def load():
    cfg = json.load(open(PATH + ".json"))
    m = M.__getattr__(cfg["name"])(**cfg["args"])
    m.load_state_dict(load_file(PATH + ".safetensors"), strict=False)
    m.eval()
    if getattr(m, "dtype", None) == torch.bfloat16:
        m.dtype = torch.float16
    for p in m.parameters():
        if p.dtype == torch.bfloat16:
            p.data = p.data.half()
    for b in m.buffers():
        if b.dtype == torch.bfloat16:
            b.data = b.data.half()
    return m, cfg["args"]


def main():
    m, args = load()
    m = m.to("cuda")
    res = m.resolution
    x = torch.randn(1, m.in_channels, res, res, res, device="cuda")
    t = torch.full((1,), 500.0, device="cuda")
    cond = torch.randn(1, 1024, 1024, device="cuda")

    # ---------- warmup ----------
    with torch.no_grad():
        for _ in range(2):
            m(x, t, cond)
    torch.cuda.synchronize()
    print("\n[1] warmup done\n")

    # ---------- per-block event timing ----------
    blocks = m.blocks
    evs = [(torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True)) for _ in blocks]
    orig = [b.forward for b in blocks]

    def make(i, fn):
        def wrapped(*a, **kw):
            evs[i][0].record()
            r = fn(*a, **kw)
            evs[i][1].record()
            return r
        return wrapped

    for i, b in enumerate(blocks):
        b.forward = make(i, orig[i])

    h = x.view(*x.shape[:2], -1).permute(0, 2, 1).contiguous()
    h = m.input_layer(h)
    t_emb = m.t_embedder(t)
    if m.share_mod:
        t_emb = m.adaLN_modulation(t_emb)
    from trellis2.modules.utils import manual_cast
    t_emb = manual_cast(t_emb, m.dtype)
    h = manual_cast(h, m.dtype)
    cond2 = manual_cast(cond, m.dtype)

    with torch.no_grad():
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for b in blocks:
            h = b(h, t_emb, cond2, m.rope_phases)
        e1.record()
        torch.cuda.synchronize()
    print(f"[2] 30 blocks total (event, GPU timeline): "
          f"{e0.elapsed_time(e1)/1000:.3f} s")
    ts = [s.elapsed_time(e) / 1000 for s, e in evs]
    print("    per-block ms:", " ".join(f"{v*1000:.1f}" for v in ts))
    print(f"    min={min(ts)*1000:.1f}ms max={max(ts)*1000:.1f}ms "
          f"sum={sum(ts):.2f}s")
    for i, b in enumerate(blocks):
        b.forward = orig[i]

    # ---------- sub-section timing on block 0 ----------
    print("\n[3] sub-section timing (block 0), wall clock with sync:")
    b0 = blocks[0]
    mod = (b0.modulation + t_emb).type(t_emb.dtype).chunk(6, dim=1)
    sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = mod
    hh = h.clone()
    with torch.no_grad():
        def timed(label, fn, n=3):
            for _ in range(n):
                r = fn()
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n):
                r = fn()
            torch.cuda.synchronize()
            el = (time.time() - t0) / n
            print(f"    {label:24s} {el*1000:9.2f} ms")
            return r

        timed("norm1", lambda: b0.norm1(hh))
        timed("self_attn", lambda: b0.self_attn(
            b0.norm1(hh) * (1 + sc_msa.unsqueeze(1)) + sh_msa.unsqueeze(1),
            phases=m.rope_phases))
        timed("norm2", lambda: b0.norm2(hh))
        timed("cross_attn", lambda: b0.cross_attn(b0.norm2(hh), cond2))
        timed("norm3", lambda: b0.norm3(hh))
        timed("mlp", lambda: b0.mlp(b0.norm3(hh)))
        timed("FULL block", lambda: b0(hh, t_emb, cond2, m.rope_phases))

    # ---------- profiler sorted by self CPU ----------
    print("\n[4] profiler sorted by self_cpu_time_total:")
    from torch.profiler import profile, ProfilerActivity
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            m(x, t, cond)
            torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=22))


if __name__ == "__main__":
    main()
