# -*- coding: utf-8 -*-
"""把 Pixal3D 的 fp32 DiT 权重转成 fp16，供 8GB 及以下的卡使用（一键包安装期用）。

转换规则与仓库里 engine/code/_convert_fp16.py 完全一致（那份是作者在**自己机器上**
跑的，路径写死成 D:\\IDM\\TRELLIS2\\...；本份只多了「路径来自命令行」这一点，
这样换一台机器、换一块盘也能用，而且不会去改 engine 里那份已经实测过的文件）：

    SKIP_SUBSTR = (".gamma", ".beta", "_norm.", "norm.weight", "norm.bias",
                   ".scale", ".shift", ".modulation", "_token", "pos_embed")
    可转 = (ndim == 2) and (名字不含 SKIP_SUBSTR 里的任何关键词)

为什么这些要留 F32（官方注释）：
1. norm 的 gain/bias 不经过矩阵乘，是 elementwise 相乘的，量化误差会直接变成输出误差
2. 1D/3D/4D 张量（conv / patch embed）不适合按行量化
3. 真正吃体积的是 2D 矩阵（attention / FFN 权重），那些才该转

decoder 本来就是 fp16，直接复制不动。

    python _convert_pixal3d_fp16.py \
        --src  engine/code/MODELS/Pixal3D-F32Flow-F16Decoder \
        --dst  engine/code/MODELS/Pixal3D-fp16
"""
import argparse
import os
import shutil
import sys
import time

import torch
from safetensors.torch import load_file, save_file

SKIP_SUBSTR = (".gamma", ".beta", "_norm.", "norm.weight", "norm.bias",
               ".scale", ".shift", ".modulation", "_token", "pos_embed")

# 需要转换的 DiT（decoder 已经是 fp16，跳过）
FILES = [
    "ss_flow_img_dit_1_3B_64_bf16.safetensors",
    "slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
    "slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
    "slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
]

CONVERTIBLE = (torch.float32, torch.bfloat16)


def is_weight_matrix(name: str) -> bool:
    return not any(k in name for k in SKIP_SUBSTR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="含 ckpts/ 的 fp32 权重目录")
    ap.add_argument("--dst", required=True, help="输出目录（如 MODELS/Pixal3D-fp16）")
    a = ap.parse_args()

    src = os.path.abspath(a.src)
    dst = os.path.abspath(a.dst)
    src_ck = os.path.join(src, "ckpts")
    dst_ck = os.path.join(dst, "ckpts")
    if not os.path.isdir(src_ck):
        print(f"[错误] 找不到 {src_ck}，--src 要指向权重根目录（里面应有 ckpts/）")
        return 1
    if os.path.abspath(src) == os.path.abspath(dst):
        print("[错误] --src 和 --dst 不能是同一个目录（会覆盖原件）")
        return 1
    os.makedirs(dst_ck, exist_ok=True)

    print("=" * 60)
    print("Pixal3D fp32 -> fp16 转换")
    print(f"  源   {src}")
    print(f"  目标 {dst}")
    print("=" * 60)

    grand_before = grand_after = 0
    t0 = time.time()
    converted = 0

    for fname in FILES:
        s = os.path.join(src_ck, fname)
        d = os.path.join(dst_ck, fname)
        if not os.path.isfile(s):
            print(f"[跳过] 不存在: {fname}")
            continue
        if os.path.isfile(d):
            print(f"[已有] {fname} —— 不重复转换")
            grand_before += os.path.getsize(s)
            grand_after += os.path.getsize(d)
            converted += 1
            continue

        sz_before = os.path.getsize(s)
        print(f"\n[{fname}]  {sz_before / 1073741824:.2f} GB", flush=True)
        t = time.time()
        sd = load_file(s, device="cpu")
        out = {}
        n_conv = n_keep = 0
        for k, v in sd.items():
            if v.dtype in CONVERTIBLE and v.ndim == 2 and is_weight_matrix(k):
                out[k] = v.half()
                n_conv += 1
            else:
                out[k] = v          # 保持原样（norm/bias/1D/3D/4D）
                n_keep += 1
        save_file(out, d, metadata={"format": "pt"})
        sz_after = os.path.getsize(d)
        grand_before += sz_before
        grand_after += sz_after
        converted += 1
        print(f"  转了 {n_conv} 个 / 保留 {n_keep} 个", flush=True)
        print(f"  {sz_before / 1073741824:.2f} GB -> {sz_after / 1073741824:.2f} GB"
              f"  (省 {(1 - sz_after / sz_before) * 100:.0f}%)  耗时 {time.time() - t:.0f}s",
              flush=True)
        del sd, out

    if converted == 0:
        print("\n[错误] 一个 DiT 文件都没找到，--src 下的 ckpts/ 内容不对")
        return 1

    # 复制不需要转换的文件（decoder + 配置）
    print("\n复制 decoder 和配置（本来就是 fp16，无需转换）...")
    for name in sorted(os.listdir(src_ck)):
        if name.endswith(".json") or "dec" in name:
            s, d = os.path.join(src_ck, name), os.path.join(dst_ck, name)
            if os.path.isfile(s) and not os.path.exists(d):
                shutil.copy2(s, d)
                print(f"  ckpts/{name}")
    for name in ("pipeline.json", "pipeline_mv.json"):
        s, d = os.path.join(src, name), os.path.join(dst, name)
        if os.path.isfile(s):
            shutil.copy2(s, d)
            print(f"  {name}")

    print("\n完成！")
    print(f"总计: {grand_before / 1073741824:.2f} GB -> {grand_after / 1073741824:.2f} GB"
          f"  省 {(grand_before - grand_after) / 1073741824:.2f} GB")
    print(f"总耗时 {(time.time() - t0) / 60:.1f} 分钟")
    if not os.path.isfile(os.path.join(dst, "pipeline.json")):
        print("[警告] 输出目录里没有 pipeline.json，加载时会找不到模型目录")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
