# -*- coding: utf-8 -*-
"""一句话 → 一个 GLB。把两个模型串起来：

    FLUX.2-klein-4B（文生图，独立 venv flux-venv）
        ↓  PNG
    TRELLIS.2 / Pixal3D（图生3D，127.0.0.1:8080 后端）
        ↓
      GLB

**本脚本跑在 3D 的 venv 里**（需要 gradio_client），文生图那步用子进程调 flux-venv：

    /home/ccb/trellis2-wsl-venv/bin/python t2i_to_3d.py --prompt "一只坐着的橘猫"

显存：两个模型**串行**跑，绝不重叠（8GB 卡同时装不下）。
"""
import argparse
import os
import subprocess
import sys
import time

# 本机配了系统代理时，autoProxy 会把 http_proxy 注入 WSL，导致 gradio_client
# 连不上 127.0.0.1:8080（报 httpx.ConnectTimeout）。先清掉。
for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_v, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

ROOT = os.path.dirname(os.path.abspath(__file__))
FLUX_PY = os.environ.get("FLUX_PY", "/home/ccb/flux-venv/bin/python")
URL = os.environ.get("TRELLIS2_API", "http://127.0.0.1:8080")
OUT_DIR = os.path.join(ROOT, "output")


def wait_ready(client, timeout=900):
    """等 3D 后端把 pipeline 加载完。端口通 != 就绪。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            html = client.predict(api_name="/_runtime_panel_html")
            html = html[0] if isinstance(html, (list, tuple)) else str(html)
        except Exception:
            time.sleep(5)
            continue
        if "就绪" in html:
            return
        if "启动失败" in html:
            raise RuntimeError("3D 后端启动失败：" + html[:300])
        time.sleep(5)
    raise TimeoutError("等 3D 后端就绪超时（15 分钟）")


def main() -> int:
    ap = argparse.ArgumentParser(description="文生3D：一句话出 GLB")
    ap.add_argument("--prompt", default=None, help="文生图提示词")
    ap.add_argument("--prompt-file", default=None,
                    help="从 UTF-8 文件读提示词（.bat 用这个，避免 cmd→wsl→bash 的引号地狱）")
    ap.add_argument("--image", default=None,
                    help="跳过文生图，直接用这张已有图片走图生3D")
    ap.add_argument("--model", default="Pixal3D", choices=["TRELLIS.2", "Pixal3D"])
    ap.add_argument("--resolution", default="512",
                    choices=["512", "1024 快速", "1024 高精", "1536"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", type=int, default=768, help="文生图边长")
    ap.add_argument("--steps", type=int, default=4, help="文生图步数")
    ap.add_argument("--offload", default="model", choices=["model", "sequential", "none"])
    ap.add_argument("--faces", type=int, default=250000,
                    help="GLB 面数上限，UI 滑块范围 10000~1000000（默认 250000，越小文件越小）")
    ap.add_argument("--texture", type=int, default=2048,
                    help="GLB 贴图尺寸，UI 滑块范围 1024~4096（默认 2048）")
    ap.add_argument("--auto-degrade", dest="auto_degrade", action="store_true", default=True,
                    help="显存不够时自动降级（默认开）")
    ap.add_argument("--no-auto-degrade", dest="auto_degrade", action="store_false",
                    help="不降级：装不下就占用系统内存，很慢")
    a = ap.parse_args()

    # 提示词：命令行直接给，或从 UTF-8 文件读。
    # .bat 走文件这条路 —— cmd → wsl.exe → bash → python 四层转义下来，
    # 带空格/引号/中文的提示词几乎必被吃掉。
    prompt = a.prompt
    if not prompt and a.prompt_file:
        with open(a.prompt_file, encoding="utf-8") as fh:
            prompt = fh.read().strip()
    if not prompt:
        # 双击 .bat 进来的正常路径：直接在窗口里问，绕开所有转义问题
        try:
            prompt = input("请输入提示词（英文通常效果更好）：").strip()
        except EOFError:
            prompt = ""
    if not prompt:
        print("没有提示词：请用 --prompt / --prompt-file 指定，或运行后直接输入。")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    # ── 第一步：文生图 ─────────────────────────────────────────
    if a.image:
        img = os.path.abspath(a.image)
        print(f"[1/2] 跳过文生图，用已有图片：{img}", flush=True)
        if not os.path.exists(img):
            print(f"图片不存在：{img}")
            return 1
    else:
        img = os.path.join(OUT_DIR, f"t2i-{stamp}.png")
        print("=" * 64)
        print(f"[1/2] 文生图  提示词：{prompt}")
        print("=" * 64, flush=True)
        if not os.path.exists(FLUX_PY):
            print(f"文生图环境不存在：{FLUX_PY}\n请先双击「一键文生3D.bat」准备环境。")
            return 1
        cmd = [FLUX_PY, os.path.join(ROOT, "_t2i_generate.py"),
               "--prompt", prompt, "--out", img,
               "--size", str(a.size), "--steps", str(a.steps),
               "--seed", str(a.seed), "--offload", a.offload]
        rc = subprocess.call(cmd)
        if rc != 0 or not os.path.exists(img):
            print(f"文生图失败（退出码 {rc}）")
            return 1

    # ── 第二步：图生3D ─────────────────────────────────────────
    print("=" * 64)
    print(f"[2/2] 图生3D  模型={a.model}  分辨率={a.resolution}")
    print("=" * 64, flush=True)
    try:
        from gradio_client import Client, handle_file
    except Exception as e:
        print(f"缺少 gradio_client（{e}），请在 3D 的 venv 里跑本脚本。")
        return 1

    client = Client(URL)
    print("等待 3D 引擎就绪 …", flush=True)
    wait_ready(client)

    t0 = time.time()
    res = client.predict(
        handle_file(img),
        [],                     # reference_images
        a.seed,
        a.model,
        a.resolution,
        7.5, 0.7, 8, 5.0,       # ss_*
        7.5, 0.5, 8, 3.0,       # shape_slat_*
        1.0, 0.0, 8, 3.0,       # tex_slat_*
        False, False, False, 80, 3, 300,   # profiler
        bool(a.auto_degrade),
        api_name="/image_to_3d",
    )
    preview, status = (res[0], res[1]) if isinstance(res, (list, tuple)) else (res, "")
    body = f"{preview}\n{status}"
    dt = time.time() - t0

    if "生成失败" in body:
        # 注意：/image_to_3d 失败时**照样返回**，必须自己检查内容
        print(f"❌ 图生3D 失败，用时 {dt:.0f}s")
        tail = body.split("生成失败", 1)[1][:400]
        print(tail)
        return 1
    print(f"✅ 图生3D 完成，用时 {dt:.0f}s", flush=True)

    # ── 导出 GLB ──────────────────────────────────────────────
    # UI 滑块的硬区间：面数 10000~1000000、贴图 1024~4096。
    # 传越界值会直接被 Gradio 拒掉（AppError: Value ... is less than minimum value），
    # 这里先夹一下，省得生成白跑 8 分钟。
    faces = max(10000, min(1000000, int(a.faces)))
    tex = max(1024, min(4096, int(a.texture)))
    if (faces, tex) != (int(a.faces), int(a.texture)):
        print(f"[导出] 参数已夹到合法区间：面数 {a.faces}->{faces}，贴图 {a.texture}->{tex}")
    try:
        exported, glb = client.predict(faces, tex, api_name="/extract_glb")
    except Exception as e:
        print(f"自动导出失败（{type(e).__name__}: {e}）")
        print("请在网页上点「导出 GLB」。")
        return 1

    # /extract_glb 返回的是 Gradio 的**临时文件**（/tmp/gradio/<hash>/sample_*.glb），
    # 随时会被清理。复制一份到 output/ 才是能长期留着的产物。
    import shutil
    final = os.path.join(OUT_DIR, f"t2i-{stamp}.glb")
    try:
        src = glb if isinstance(glb, str) else str(glb)
        if os.path.exists(src):
            shutil.copy2(src, final)
            size_mb = os.path.getsize(final) / 1024 ** 2
            print(f"[导出] 已存到 output/：{final}（{size_mb:.2f} MB）")
        else:
            print(f"[导出] 临时文件不见了，只能给原始路径：{src}")
            final = src
    except Exception as e:
        print(f"[导出] 复制到 output/ 失败（{type(e).__name__}: {e}），原始文件在 {glb}")
        final = glb

    print("=" * 64)
    print(f"✅ 完成！图片：{img}")
    print(f"✅ GLB ：{final}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
