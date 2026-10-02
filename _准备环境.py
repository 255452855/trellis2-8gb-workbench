# -*- coding: utf-8 -*-
"""启动前自检：修正 venv 路径、清理引擎残留锁、检查模型权重。
一切正常时不输出任何内容；有问题才打印，由「一键启动.bat」显示。
"""
import os
import sys
import glob

ROOT = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(ROOT, "engine", "code")
PYEXE = os.path.join(ROOT, "engine", "tools", "python", "python.exe")
CFG = os.path.join(CODE, "venv", "pyvenv.cfg")
MODELS = os.path.join(CODE, "MODELS", "TRELLIS.2-4B")


def fail(msg):
    print("  [错误] " + msg)
    sys.exit(1)


# 1. 目录结构
if not os.path.exists(PYEXE):
    fail("找不到内置 Python：" + PYEXE)
if not os.path.exists(CFG):
    fail("虚拟环境不完整：" + CFG)

# 2. 修正 pyvenv.cfg 里的绝对路径（文件夹移动过就需要）
try:
    with open(CFG, "r", encoding="utf-8") as f:
        lines = f.readlines()

    old_home = None
    for ln in lines:
        if ln.lower().startswith("home"):
            old_home = ln.split("=", 1)[1].strip()
            break

    new_home = os.path.dirname(PYEXE)
    if old_home and os.path.abspath(old_home) != os.path.abspath(new_home):
        out = []
        for ln in lines:
            low = ln.lower()
            if low.startswith("home"):
                out.append("home = %s\n" % new_home)
            elif low.startswith("executable"):
                out.append("executable = %s\n" % PYEXE)
            elif low.startswith("base-executable"):
                out.append("base-executable = %s\n" % PYEXE)
            elif low.startswith("base-prefix") or low.startswith("base-exec-prefix"):
                out.append("%s = %s\n" % (low.split("=")[0].strip(), new_home))
            elif low.startswith("command"):
                out.append("command = %s -m virtualenv %s\n"
                           % (PYEXE, os.path.join(CODE, "venv")))
            else:
                out.append(ln)
        with open(CFG, "w", encoding="utf-8") as f:
            f.writelines(out)
        print("  [已修正] 虚拟环境路径 -> " + new_home)
except Exception as e:
    fail("修正虚拟环境路径失败：" + str(e))

# 3. 清理 flex_gemm 残留锁（不清会导致启动卡死）
lock = os.path.join(os.path.expanduser("~"), ".flex_gemm", "autotune_cache.json.lock")
if os.path.exists(lock):
    try:
        os.remove(lock)
        print("  [已清理] 引擎残留锁文件")
    except Exception as e:
        print("  [警告] 无法删除锁文件：" + str(e))

# 4. 模型权重
if not os.path.isdir(MODELS):
    fail("找不到模型权重目录：" + MODELS)
n = len(glob.glob(os.path.join(MODELS, "ckpts", "*.safetensors")))
if n < 9:
    print("  [警告] 权重文件只有 %d 个（应为 9 个），生成可能失败" % n)

sys.exit(0)
