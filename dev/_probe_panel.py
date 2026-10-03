"""Probe the LIVE backend for the runtime panel HTML (query the running process).

Run inside WSL:
    /home/ccb/trellis2-wsl-venv/bin/python -u _probe_panel.py
"""
import os
import sys

os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

from gradio_client import Client  # noqa: E402

c = Client("http://127.0.0.1:8080/")
html = c.predict(api_name="/_runtime_panel_html")
print("=== 后端返回的运行状态面板 HTML ===")
print(html)
print()
for key in ("就绪", "加载中", "启动失败", "TRELLIS.2", "Pixal3D"):
    print(f"  含「{key}」: {key in html}")
