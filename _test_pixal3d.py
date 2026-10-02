# -*- coding: utf-8 -*-
"""通过 Gradio API 触发一次 Pixal3D 生成，验证 WSL 后端能否跑通。"""
import sys
import time

from gradio_client import Client, handle_file

URL = "http://127.0.0.1:8080"
IMG = ("/mnt/d/IDM/TRELLIS2/engine/code/assets/example_image/"
       "0a34fae7ba57cb8870df5325b9c30ea474def1b0913c19c596655b85a79fdee4.webp")

print("连接 %s ..." % URL, flush=True)
c = Client(URL)
print("已连接，开始提交 Pixal3D 生成（8 步）...", flush=True)

t0 = time.time()
try:
    res = c.predict(
        handle_file(IMG),   # image
        [],                 # reference_images
        42,                 # seed
        "Pixal3D",          # model_choice
        "512",              # resolution
        7.5, 0.7, 8, 5.0,   # ss_*
        7.5, 0.5, 8, 3.0,   # shape_slat_*
        1.0, 0.0, 8, 3.0,   # tex_slat_*
        False, False, False, 80, 3, 300,   # profiler
        True,               # auto_degrade：显存不足时自动降级
        api_name="/image_to_3d",
    )
    print("✅ 完成，用时 %.1f 分钟" % ((time.time() - t0) / 60), flush=True)
    print("返回:", str(res)[:800], flush=True)
except Exception as e:
    print("❌ 失败，用时 %.1f 分钟" % ((time.time() - t0) / 60), flush=True)
    print("错误:", type(e).__name__, e, flush=True)
    sys.exit(1)
