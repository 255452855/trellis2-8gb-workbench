# -*- coding: utf-8 -*-
import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))

"""一句话 → 一张图（FLUX.2-klein-4B）。

**必须用文生图自己的 venv 跑**（diffusers 0.40 需要 huggingface_hub>=1.23，
和 3D 环境的 transformers 4.57 冲突）：

    /home/ccb/flux-venv/bin/python _t2i_generate.py --prompt "..." --out x.png

显存：transformer 7.75GB + Qwen3-4B 文本编码器 8.05GB ≈ 16GB，
8GB 卡**必须**开 offload，否则一定 OOM。
"""
import argparse
import os
import time

# 只用本地权重，别让 huggingface_hub 去联网（这台机器上 HF 不通）
os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODEL_DIR = os.environ.get(
    "FLUX_MODEL_DIR",
    "" + _ROOT + "/engine/code/MODELS/FLUX.2-klein-4B",
)


def main() -> int:
    ap = argparse.ArgumentParser(description="FLUX.2-klein-4B 文生图")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prompt-file", default=None, help="从 UTF-8 文件读提示词")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=768,
                    help="正方形边长；8GB 卡建议 512~768，越大越慢越吃显存")
    ap.add_argument("--steps", type=int, default=4, help="klein 是蒸馏模型，4 步即可")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offload", choices=["model", "sequential", "none"], default="model",
                    help="model=按子模型换入换出；sequential=按层，最省显存但最慢")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    a = ap.parse_args()

    # 提示词：命令行 / 文件 / 交互式三选一。交互式是双击 .bat 的默认路径，
    # 这样提示词永远不经过 cmd→wsl→bash 的多层转义。
    prompt = a.prompt
    if not prompt and a.prompt_file:
        with open(a.prompt_file, encoding="utf-8") as fh:
            prompt = fh.read().strip()
    if not prompt:
        try:
            prompt = input("请输入提示词（英文通常效果更好）：").strip()
        except EOFError:
            prompt = ""
    if not prompt:
        print("没有提示词，退出。")
        return 1

    import torch
    from diffusers import Flux2KleinPipeline

    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float16
    print(f"[文生图] 加载 {MODEL_DIR}  dtype={a.dtype}", flush=True)
    t0 = time.time()
    pipe = Flux2KleinPipeline.from_pretrained(
        MODEL_DIR, torch_dtype=dtype, local_files_only=True
    )
    print(f"[文生图] 权重加载完成，用时 {time.time() - t0:.0f}s", flush=True)

    if a.offload == "model":
        pipe.enable_model_cpu_offload()
        print("[文生图] enable_model_cpu_offload()", flush=True)
    elif a.offload == "sequential":
        pipe.enable_sequential_cpu_offload()
        print("[文生图] enable_sequential_cpu_offload()（最省显存，最慢）", flush=True)
    else:
        pipe.to("cuda")

    # VAE 分块解码，进一步压峰值
    for fn in ("enable_slicing", "enable_tiling"):
        try:
            getattr(pipe.vae, fn)()
        except Exception:
            pass

    gen = torch.Generator("cpu").manual_seed(a.seed)
    print(f"[文生图] 采样中：{a.size}x{a.size}, {a.steps} 步 …", flush=True)
    t1 = time.time()
    image = pipe(
        prompt=prompt,
        height=a.size,
        width=a.size,
        guidance_scale=1.0,
        num_inference_steps=a.steps,
        generator=gen,
    ).images[0]

    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    image.save(out)
    print(f"[文生图] 完成，采样用时 {time.time() - t1:.0f}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
