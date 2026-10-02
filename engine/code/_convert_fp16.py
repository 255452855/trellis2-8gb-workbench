# -*- coding: utf-8 -*-
"""
把 Pixal3D 的 fp32 DiT 权重转成 fp16 safetensors，省一半空间。

转换规则抄自 raven38/pixal3d.cpp 的官方量化工具 tools/quantize_gguf.py：

    SKIP_SUBSTR = (".gamma", ".beta", "_norm.", "norm.weight", "norm.bias",
                   ".scale", ".shift", ".modulation", "_token", "pos_embed")
    可转 = (ndim == 2) and (名字不含 SKIP_SUBSTR 里的任何关键词)

为什么这些要留 F32（官方注释）：
1. norm 的 gain/bias 不经过矩阵乘，是 elementwise 相乘的，量化误差会直接变成输出误差
2. 1D/3D/4D 张量（conv / patch embed）不适合按行量化
3. 真正吃体积的是 2D 矩阵（attention / FFN 权重），那些才该转

decoder 本来就是 fp16，不动。
"""
import os
import sys
import time

import torch
from safetensors.torch import load_file, save_file

SRC = r"D:\IDM\TRELLIS2\engine\code\MODELS\Pixal3D\ckpts"
DST = r"D:\IDM\TRELLIS2\engine\code\MODELS\Pixal3D-fp16\ckpts"

SKIP_SUBSTR = (".gamma", ".beta", "_norm.", "norm.weight", "norm.bias",
               ".scale", ".shift", ".modulation", "_token", "pos_embed")


def is_weight_matrix(name: str) -> bool:
    return not any(k in name for k in SKIP_SUBSTR)


# 需要转换的 DiT（decoder 已经是 fp16，跳过）
FILES = [
    "ss_flow_img_dit_1_3B_64_bf16.safetensors",
    "slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
    "slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
    "slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
]

os.makedirs(DST, exist_ok=True)

print("=" * 60)
print("Pixal3D fp32 -> fp16 转换")
print("=" * 60)

grand_before = grand_after = 0
T0 = time.time()

for fname in FILES:
    src = os.path.join(SRC, fname)
    dst = os.path.join(DST, fname)
    if not os.path.isfile(src):
        print(f"[跳过] 不存在: {fname}")
        continue

    sz_before = os.path.getsize(src)
    print(f"\n[{fname}]  {sz_before/1073741824:.2f} GB", flush=True)

    t = time.time()
    sd = load_file(src, device="cpu")

    out = {}
    n_conv = n_keep = 0
    for k, v in sd.items():
        # 只转 2D 矩阵权重，且名字不在白名单里
        if v.dtype == torch.float32 and v.ndim == 2 and is_weight_matrix(k):
            out[k] = v.half()
            n_conv += 1
        else:
            out[k] = v          # 保持原样（norm/bias/1D/3D/4D）
            n_keep += 1

    save_file(out, dst, metadata={"format": "pt"})
    sz_after = os.path.getsize(dst)
    grand_before += sz_before
    grand_after += sz_after

    print(f"  转了 {n_conv} 个 / 保留 {n_keep} 个", flush=True)
    print(f"  {sz_before/1073741824:.2f} GB -> {sz_after/1073741824:.2f} GB"
          f"  (省 {(1-sz_after/sz_before)*100:.0f}%)  耗时 {time.time()-t:.0f}s", flush=True)

    del sd, out
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

print()
print("=" * 60)
print(f"总计: {grand_before/1073741824:.2f} GB -> {grand_after/1073741824:.2f} GB"
      f"  省 {(grand_before-grand_after)/1073741824:.2f} GB")
print(f"总耗时 {(time.time()-T0)/60:.1f} 分钟")
print("=" * 60)

# 复制不需要转换的文件（decoder + pipeline 配置）
import shutil
print("\n复制 decoder 和配置（本来就是 fp16，无需转换）...")
for name in os.listdir(SRC):
    if name.endswith(".json") or "dec" in name:
        s = os.path.join(SRC, name)
        d = os.path.join(DST, name)
        if os.path.isfile(s) and not os.path.exists(d):
            shutil.copy2(s, d)
            print(f"  {name}")

for name in ("pipeline.json", "pipeline_mv.json"):
    s = os.path.join(os.path.dirname(SRC), name)
    d = os.path.join(os.path.dirname(DST), name)
    if os.path.isfile(s):
        shutil.copy2(s, d)
        print(f"  {name}")

print("\n完成！")
