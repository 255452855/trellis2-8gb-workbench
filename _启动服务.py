# -*- coding: utf-8 -*-
"""独立启动引擎服务（脱离当前 shell，不会被回收）。"""
import os
import subprocess
import sys
import time

CODE = r"D:\IDM\TRELLIS2\engine\code"
PY = os.path.join(CODE, "venv", "Scripts", "python.exe")
LOG = r"D:\IDM\TRELLIS2\engine\code\output\_service.log"

os.makedirs(os.path.dirname(LOG), exist_ok=True)

env = os.environ.copy()
env.update({
    "XFORMERS_IGNORE_FLASH_VERSION_CHECK": "1",
    "ATTN_BACKEND": "sdpa",
    "OPENCV_IO_ENABLE_OPENEXR": "1",
    "HF_HOME": os.path.join(CODE, "MODELS"),
    "HF_HUB_OFFLINE": "1",
    "PYTHONUNBUFFERED": "1",
    # 这个 venv 是「嵌入式 Python + virtualenv」混合布局，
    # 标准库 zip（venv/Scripts/python311.zip）排在 site-packages 之前，
    # 会让 distutils 从 zip 加载，setuptools 的 _distutils_hack 断言失败：
    #   AssertionError: ...python311.zip\distutils\core.pyc
    # 设为 stdlib 让 hack 不介入。
    "SETUPTOOLS_USE_DISTUTILS": "stdlib",
})

# ── 修复 TEMP/TMP ────────────────────────────────────────────
# 从 Git Bash / MSYS 启动时，TEMP 会是 "/tmp" 这种 Unix 风格路径，
# Windows 的 multiprocessing spawn 不认，会报
#   FileNotFoundError: [WinError 3] 系统找不到指定的路径
# 所以这里强制改回 Windows 原生路径（桌面双击 bat 时本来就是对的，
# 但通过脚本启动就会踩这个坑）。
_win_temp = os.path.join(
    os.environ.get("LOCALAPPDATA", r"C:\Users\Administrator\AppData\Local"),
    "Temp",
)
os.makedirs(_win_temp, exist_ok=True)
env["TEMP"] = _win_temp
env["TMP"] = _win_temp

# ── 修复 localhost 被代理劫持 ─────────────────────────────────
# 如果启动时 shell 里带着 HTTP_PROXY/HTTPS_PROXY（例如为了下 HF 权重设的），
# 子进程会继承，导致 Gradio 访问 127.0.0.1 也走代理，启动时报：
#   Couldn't start the app because '.../gradio_api/startup-events' failed (code 502)
# 让本地地址绕过代理即可。
#
# 注意：这里**不能**清掉 HTTP_PROXY/HTTPS_PROXY ——
# TRELLIS.2 的 pipeline.json 里 sparse_structure_decoder 指向
#   microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16
# 是个 HF 路径，加载时需要走网络（或读 HF 缓存）。清了代理就报
#   LocalEntryNotFoundError
# 所以只加 NO_PROXY，保留代理本身。
_no_proxy = "127.0.0.1,localhost,::1,0.0.0.0"
env["NO_PROXY"] = _no_proxy
env["no_proxy"] = _no_proxy
if not env.get("HTTP_PROXY") and not env.get("HTTPS_PROXY"):
    # shell 里没有代理时，给个默认值（本机常用的本地代理端口）
    env["HTTP_PROXY"] = "http://127.0.0.1:7897"
    env["HTTPS_PROXY"] = "http://127.0.0.1:7897"
    print("未检测到代理，已设默认代理 127.0.0.1:7897（HF 路径加载需要）")

print(f"TEMP 已设为: {_win_temp}")
print(f"本地地址绕过代理: {_no_proxy}")
print(f"代理: {env.get('HTTP_PROXY', '(无)')}")

# DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP —— 父进程退出后服务继续跑
DETACHED = 0x00000008 | 0x00000200

with open(LOG, "wb") as lf:
    p = subprocess.Popen(
        [PY, "app.py"],
        cwd=CODE,
        env=env,
        stdout=lf,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=DETACHED,
    )

print(f"已启动，PID={p.pid}")
print(f"日志: {LOG}")
time.sleep(3)
print(f"3 秒后进程存活: {p.poll() is None}")
