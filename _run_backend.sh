#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# 3D 后端的**唯一**启动定义：环境变量的组装只在这里写一份。
#
# 这套变量是照「已经实测能跑」的 _启动WSL后端.ps1 一份份对抄的（cwd=engine/code、
# ATTN_BACKEND=sdpa、PYTORCH_CUDA_ALLOC_CONF 必须在启动时设好、HF_HOME=<code>/MODELS、
# 权重齐了才离线）。差别只有一处，而且是关键的：那位作者机器上的
# /home/ccb、/usr/local/cuda-12.8、TORCH_CUDA_ARCH_LIST=6.1 全是写死的，
# 换一台机器就全错；这里改成 _lib.sh 现场探测 + runtime.conf 记忆。
#
# 启动器（start.bat → _start.ps1）只负责「wsl 起进程 + 等端口 + 开浏览器」，
# 不在那里拼环境变量 —— 以前两边各写一份、改一处忘另一处，排查了半天才发现
# 是两边不一致。
#
#   bash _run_backend.sh              # 前台启动（PowerShell 用 Start-Process 持有它）
#   TRELLIS2_PORT=8090 bash _run_backend.sh
# ─────────────────────────────────────────────────────────────────────
set -u
. "$(cd "$(dirname "$0")" && pwd)/_lib.sh"
t2_init

CODE="$T2_CODE"
[ -x "$TRELLIS2_PY" ] || t2_die "3D 环境不存在：$TRELLIS2_PY
   先跑：bash _setup_3d_venv.sh   或者直接用 start.bat（一键启动）"
[ -f "$CODE/app.py" ]  || t2_die "找不到 $CODE/app.py"
cd "$CODE" || exit 1

# CUDA 编译器位置：编扩展时用过的，运行期 cublas 之类也按它找
CUDA_HOME_R="${TRELLIS2_CUDA_HOME:-$(t2_detect_cuda_home 2>/dev/null)}"
if [ -n "${CUDA_HOME_R:-}" ] && [ -d "$CUDA_HOME_R/lib64" ]; then
  export CUDA_HOME="$CUDA_HOME_R"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export TORCH_CUDA_ARCH_LIST="${TRELLIS2_ARCH:-$(t2_detect_arch 2>/dev/null || echo native)}"
export XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
export OPENCV_IO_ENABLE_OPENEXR=1
export PYTHONUNBUFFERED=1
export HF_HOME="$TRELLIS2_MODELS"

# 权重齐了才敢离线：以前无条件 HF_HUB_OFFLINE=1，缺权重的人不会看到
# "下权重"的提示，只会看到 from_pretrained 抛出来的离线错误。
if [ "${TRELLIS2_WEIGHTS_READY:-0}" = "1" ] || [ -f "$TRELLIS2_MODELS/TRELLIS.2-4B/pipeline.json" ]; then
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
else
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
  echo "  [警告] 权重看起来不全（$TRELLIS2_MODELS/TRELLIS.2-4B 不存在）。"
  echo "         本次允许联网下载；建议跑一次：bash _setup_weights.sh"
fi

# 显存碎片：expandable_segments 是 8GB 卡上不颠簸的前提。
# 注意必须**启动时**就设好 —— CUDA 分配器只读一次，之后再改 os.environ 不生效。
# app.py 里也用 setdefault 设了同一个值，但那是给「直接在 WSL 里手敲 python app.py」
# 的人兜底的；从本脚本启动时这里才是真正生效的那一次，两边值必须一致。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.65}"

echo "[run_backend] python = $TRELLIS2_PY"
echo "[run_backend] models = $HF_HOME  (offline=$HF_HUB_OFFLINE)"
echo "[run_backend] arch   = $TORCH_CUDA_ARCH_LIST  gpu=$(t2_gpu_name)"
echo "[run_backend] port   = $TRELLIS2_PORT"

# flex_gemm 的 autotune 锁文件残留会让启动直接卡死（旧版 _准备环境.py 里
# 有这条自愈逻辑，那个脚本依赖 Windows 侧的 python，这里在 WSL 里自己做）。
lock="$HOME/.flex_gemm/autotune_cache.json.lock"
if [ -f "$lock" ]; then
  # 只删没人持有的锁：有活着的进程打开它时删除反而会让状态不一致
  if ! command -v fuser >/dev/null 2>&1 || ! fuser "$lock" >/dev/null 2>&1; then
    rm -f "$lock" && echo "[run_backend] 已清理 flex_gemm 残留锁"
  else
    echo "[run_backend] flex_gemm 锁仍被占用，说明有旧后端没退干净：先跑 _stop_3d_backend.sh"
  fi
fi

exec "$TRELLIS2_PY" app.py --host 0.0.0.0 --port "$TRELLIS2_PORT" --no-browser
