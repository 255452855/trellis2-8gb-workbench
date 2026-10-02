#!/usr/bin/env bash
# 一键文生3D 的 WSL 侧入口，由 一键文生3D.bat 调用。
#
# 分两个模式，**顺序很关键**：
#   t2i    → 只跑文生图。此时 3D 后端必须**已经停掉**：
#            FLUX 开了 model_cpu_offload，16GB 权重常驻系统内存，
#            和 3D 后端同时活着会把 WSL 的 16GB 内存吃干 + 显存颠簸。
#   to3d   → 只跑图生3D，用 t2i 产出的固定文件。此时后端必须已经起来。
set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
PYFLUX="${FLUX_PY:-/home/ccb/flux-venv/bin/python}"
PY3D="${TRELLIS2_PY:-/home/ccb/trellis2-wsl-venv/bin/python}"
IMG="$ROOT/output/last-t2i.png"
SIZE="${T2I_SIZE:-512}"
STEPS="${T2I_STEPS:-4}"

MODE="${1:-t2i}"

echo "############ 文生3D  [$MODE] ############"

if [ "$MODE" = "t2i" ]; then
  if [ ! -x "$PYFLUX" ]; then
    echo "[错误] 文生图环境不存在：$PYFLUX"
    echo "       先跑一次 _t2i_setup.sh"
    exit 1
  fi
  # 顺手确认 3D 后端没在占显存
  if pgrep -f "ap[p].py" >/dev/null 2>&1; then
    echo "[警告] 检测到 3D 后端还在跑，它会和 FLUX 抢内存/显存。"
    echo "       建议先停掉后端再出图（脚本会继续，但可能很慢甚至颠簸）。"
  fi
  # 提示词来源优先级：命令行额外参数 > .prompt.txt > 交互式提问。
  # .prompt.txt 这条路是给自动化/AI 用的 —— 双击时用户直接在窗口里打字，
  # 但 `echo 提示词 | 一键文生3D.bat` 这种管道喂不进 python 的 input()。
  EXTRA=("${@:2}")
  if [ ${#EXTRA[@]} -eq 0 ] && [ -s "$ROOT/.prompt.txt" ]; then
    EXTRA=(--prompt-file "$ROOT/.prompt.txt")
  fi
  echo "[提示词来源] $([ ${#EXTRA[@]} -gt 0 ] && echo "${EXTRA[0]}" || echo 交互式输入)"
  exec "$PYFLUX" -u "$ROOT/_t2i_generate.py" \
    --out "$IMG" --size "$SIZE" --steps "$STEPS" "${EXTRA[@]}"

elif [ "$MODE" = "to3d" ]; then
  if [ ! -f "$IMG" ]; then
    echo "[错误] 没找到文生图的产物：$IMG"
    exit 1
  fi
  exec "$PY3D" -u "$ROOT/t2i_to_3d.py" \
    --prompt "(from $IMG)" --image "$IMG" "${@:2}"

else
  echo "用法: _t2i_run.sh [t2i|to3d] [额外参数...]"
  exit 2
fi
