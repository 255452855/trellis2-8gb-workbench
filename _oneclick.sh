#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# 一键安装的总编排（WSL 侧的大脑）
#
# 双击 start.bat（一键启动）时 PowerShell 调本脚本；本脚本按顺序补齐缺的东西，
# 每步都可重复执行（幂等），装过的自动跳过。
#
#   bash _oneclick.sh --check      # 只报告缺什么：0=就绪 1=缺环境 2=缺权重
#   bash _oneclick.sh --yes        # 非交互，全自动装
#   bash _oneclick.sh --no-flux    # 只装图生3D（省 23GB 和约 1 小时）
#   bash _oneclick.sh --only deps,venv,weights
#   bash _oneclick.sh --status     # 打印当前状态表
#
# 步骤顺序是有原因的：
#   deps 在前 —— 没有 nvcc，venv 那步的 CUDA 扩展全部编译失败；
#   venv 在前 —— 权重下载要用环境里的 huggingface_hub / modelscope；
#   nvdiffrec 在 venv 之后 —— 它要往那个 venv 里装并借 torch 编译；
#   flux 最后 —— 23GB，可选，失败了也不影响图生3D。
# ─────────────────────────────────────────────────────────────────────
set -u
. "$(cd "$(dirname "$0")" && pwd)/_lib.sh"
t2_init

ASSUME_YES=0
WITH_FLUX=1
ONLY=""
MODE=install

while [ $# -gt 0 ]; do
  case "$1" in
    --check)    MODE=check ;;
    --status)   MODE=status ;;
    --yes)      ASSUME_YES=1 ;;
    --no-flux)  WITH_FLUX=0 ;;
    --only)     shift; ONLY="${1:-}" ;;
    -h|--help)  sed -n '2,22p' "$0"; exit 0 ;;
    *)          echo "未知参数：$1" >&2; exit 2 ;;
  esac
  shift
done

# ── 状态探测：每一步一个函数，返回 0=已完成 ─────────────────────────
has_deps()   { ! t2_base_tools_missing && [ -n "$(t2_detect_cuda_home 2>/dev/null)" ]; }
has_venv()   { [ -x "$TRELLIS2_PY" ] && "$TRELLIS2_PY" -c "import torch, gradio" >/dev/null 2>&1; }
has_nvdr() {
  [ -x "$TRELLIS2_PY" ] || return 1
  # 这里要跑一次真内核（光 import 过不算：49 号遇到过 import 成功但插件建不起来）。
  # 但必须把 CUDA 的 stubs 目录一并给它 —— renderutils 是**运行期** JIT 的，
  # 缺 LIBRARY_PATH 时 -lcuda 找不到 libcuda.so（WSL 里真实驱动叫 libcuda.so.1），
  # 于是这一步判"缺"、而安装那一步判"可用"，两边自相矛盾。
  local ch="${TRELLIS2_CUDA_HOME:-$(t2_detect_cuda_home 2>/dev/null)}"
  CUDA_HOME="$ch"   LD_LIBRARY_PATH="$ch/lib64:$ch/lib64/stubs:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"   LIBRARY_PATH="$ch/lib64:$ch/lib64/stubs:/usr/lib/wsl/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"   "$TRELLIS2_PY" - >/dev/null 2>&1 <<'NVDR'
import torch
from nvdiffrec_render import renderutils
n = torch.full((1, 2, 2, 3), 0.577, device="cuda")
renderutils.prepare_shading_normal(n, n, n, n, n, n, True, False)
NVDR
}
has_weights(){ [ -f "$TRELLIS2_MODELS/TRELLIS.2-4B/pipeline.json" ] \
               && [ -d "$TRELLIS2_MODELS/dinov3" ] && [ -d "$TRELLIS2_MODELS/RMBG-2.0" ] \
               && [ -f "$TRELLIS2_MODELS/TRELLIS.2-4B/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors" ]; }
has_flux()   { [ -x "$FLUX_PY" ] && "$FLUX_PY" -c "from diffusers import Flux2KleinPipeline" >/dev/null 2>&1 \
               && [ -f "$TRELLIS2_MODELS/FLUX.2-klein-4B/model_index.json" ]; }

want_step() {
  [ -z "$ONLY" ] && return 0
  case ",$(printf '%s' "$ONLY" | tr -d ' ')," in *",$1,"*) return 0 ;; esac
  return 1
}

status_table() {
  printf '  %-10s %s\n' "deps"    "$(has_deps    && echo 已装 || echo 缺)"
  printf '  %-10s %s\n' "venv"    "$(has_venv    && echo "已建 $TRELLIS2_VENV" || echo 缺)"
  printf '  %-10s %s\n' "nvdr"    "$(has_nvdr    && echo 可用 || echo '缺（只影响 PBR 材质渲染）')"
  printf '  %-10s %s\n' "weights" "$(has_weights && echo 就绪 || echo 缺)"
  printf '  %-10s %s\n' "flux"    "$(has_flux    && echo 就绪 || echo "缺（文生3D 用，约 23GB）")"
}

if [ "$MODE" = status ]; then
  echo "== 状态  显卡=$(t2_gpu_name)  显存=$(t2_gpu_mem_mib)MiB  架构=$(t2_detect_arch 2>/dev/null || echo ?)"
  status_table
  exit 0
fi

MISSING=""
for s in deps venv nvdr weights flux; do
  want_step "$s" || continue
  ok=0
  case "$s" in
    deps)    has_deps    && ok=1 ;;
    venv)    has_venv    && ok=1 ;;
    nvdr)    has_nvdr    && ok=1 ;;
    weights) has_weights && ok=1 ;;
    flux)
      # --no-flux 时 flux 不算缺
      if [ "$WITH_FLUX" = "1" ]; then has_flux && ok=1; else ok=1; fi ;;
  esac
  [ "$ok" = "1" ] || MISSING="$MISSING $s"
done

if [ "$MODE" = check ]; then
  echo "== 缺的步骤：${MISSING:-（无，一切就绪）}"
  case "$MISSING" in
    *weights*) exit 2 ;;
    *)         [ -n "$MISSING" ] && exit 1 || exit 0 ;;
  esac
fi

echo "================================================================"
echo "  Pixal3D / TRELLIS.2 一键安装"
echo "================================================================"
echo "  显卡      : $(t2_gpu_name)  $(t2_gpu_mem_mib)MiB  sm_$(t2_detect_arch 2>/dev/null || echo '?')"
echo "  项目(WSL) : $T2_ROOT"
echo "  3D 环境   : $TRELLIS2_VENV"
echo "  文生图环境: $FLUX_VENV"
# 目录可能还没建（首次安装），t2_free_gb 拿不到路径时退回用户 home
_free_of() { printf '%s' "$(t2_free_gb "$1" 2>/dev/null || t2_free_gb "$HOME")"; }

echo "  权重目录  : $TRELLIS2_MODELS   （所在盘 $(_free_of "$TRELLIS2_MODELS") GB 空闲）"
echo "  环境盘    : $TRELLIS2_VENV 所在盘 $(_free_of "$(dirname "$TRELLIS2_VENV")") GB 空闲"
echo "  待办      : ${MISSING:-（都已就绪）}"
echo

if [ -z "$MISSING" ]; then
  t2_ok "什么都没缺，直接启动就行"
  exit 0
fi

# 首次安装是个大动作（几十 GB、一两个小时），必须让用户点头
if [ "$ASSUME_YES" != "1" ] && [ -t 0 ]; then
  printf '  现在开始安装吗？[Y/n] '
  read -r ans
  case "$ans" in
    [nN]*) echo "  已取消。想看缺什么：bash _oneclick.sh --status"; exit 0 ;;
  esac
fi

FAILED=""
run_step() {  # run_step <步骤key> <显示名> <命令...>
  local key="$1" name="$2"; shift 2
  t2_log "$name"
  if "$@"; then t2_ok "$name 完成"; return 0; fi
  echo "  [FAIL] $name 失败（见上面的输出）"
  # FAILED 只存 key（deps/venv/weights…），因为收尾会把它拼成
  #   bash _oneclick.sh --only <FAILED>
  # 这句重跑提示。存中文显示名的话提示会变成 "--only 3/5,PBR,材质渲染依赖,…"，
  # 照抄必然失败（49 号实测就是这个）。
  FAILED="$FAILED $key"
  return 1
}

run_optional() {  # run_optional <显示名> <命令...> —— 可选依赖：失败只警告，不进 FAILED
  local name="$1"; shift
  t2_log "$name"
  if "$@"; then
    t2_ok "$name 完成"
  else
    echo "  [降级] $name 没装成。**不影响图生3D**，页面照常用，只是材质(PBR)渲染阶段跳过。"
    echo "         想再试一次：bash _oneclick.sh --only nvdr   或直接 bash _setup_nvdiffrec.sh"
    echo "         （实测：安装期偶发编译失败，重跑一次即成功 —— 产物会被 torch 缓存）"
  fi
  return 0   # 可选步骤永不阻断一键流程
}

# ── 1. 系统依赖 ─────────────────────────────────────────────────────
if want_step deps && case " $MISSING " in *" deps "*) true ;; *) false ;; esac; then
  run_step deps "1/5 WSL 系统依赖（python3-venv / build-essential / CUDA 编译器）" \
    t2_ensure_system_deps
fi

# ── 2. 3D 环境 ──────────────────────────────────────────────────────
if want_step venv && case " $MISSING " in *" venv "*) true ;; *) false ;; esac; then
  run_step venv "2/5 建 3D 环境（要编译 4 个 CUDA 扩展，慢的要 30~60 分钟）" \
    bash "$T2_ROOT/_setup_3d_venv.sh" --activate
fi

# ── 3. PBR 渲染依赖（可选，缺了也能跑图生3D）───────────────────────
if want_step nvdr && case " $MISSING " in *" nvdr "*) true ;; *) false ;; esac; then
  if has_venv; then
    run_optional "3/5 PBR 材质渲染依赖 nvdiffrec_render" \
      bash "$T2_ROOT/_setup_nvdiffrec.sh"
  else
    echo; echo "  [SKIP] 3/5 nvdiffrec_render —— 3D 环境还没建好，建好后会自动补"
  fi
fi

# ── 4. 权重 ─────────────────────────────────────────────────────────
if want_step weights && case " $MISSING " in *" weights "*) true ;; *) false ;; esac; then
  if has_venv; then
    run_step weights "4/5 下载权重（约 33GB，8GB 卡还要本地转 fp16）" \
      bash "$T2_ROOT/_setup_weights.sh"
  else
    echo; echo "  [SKIP] 4/5 权重 —— 下载器在 3D 环境里，得先有它"
    FAILED="$FAILED weights"
  fi
fi

# ── 5. 文生3D（FLUX）────────────────────────────────────────────────
if [ "$WITH_FLUX" = "1" ] && want_step flux && case " $MISSING " in *" flux "*) true ;; *) false ;; esac; then
  run_optional "5/5 文生3D：FLUX.2-klein-4B 环境 + 23GB 权重" \
    bash "$T2_ROOT/_t2i_setup.sh"
fi

# ── 收尾：把这次的结果写进配置，供 PowerShell / 下次启动复用 ─────────
# 权重就绪与否用**实测**而不是"步骤没报错"：漏文件比报错更常见。
if has_weights; then t2_conf_set weights_ready 1; else t2_conf_set weights_ready 0; fi
if has_flux;   then t2_conf_set flux_ready 1;    else t2_conf_set flux_ready 0; fi
if [ -x "$TRELLIS2_PY" ]; then
  t2_conf_set venv3d "$TRELLIS2_VENV"
  t2_conf_set cuda_home "$(t2_detect_cuda_home 2>/dev/null)"
  t2_conf_set cuda_arch "$(t2_detect_arch 2>/dev/null)"
fi
t2_conf_set venvflux "$FLUX_VENV"
t2_conf_set models_dir "$TRELLIS2_MODELS"
t2_conf_set port "$TRELLIS2_PORT"
t2_conf_set wsl_root "$T2_ROOT"

echo
echo "================================================================"
if [ -z "$FAILED" ]; then
  echo "  安装完成。现在可以启动后端了。"
else
  echo "  有以下步骤失败：$FAILED"
  echo "  重跑一次即可（已完成的会自动跳过）；单独补某一步："
  echo "    bash _oneclick.sh --only $(printf '%s' "${FAILED# }" | tr ' ' ',')"
fi
echo "  当前状态："
status_table
echo "================================================================"
[ -z "$FAILED" ] && exit 0 || exit 1
