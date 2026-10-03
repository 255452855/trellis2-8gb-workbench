# shellcheck shell=bash
# ─────────────────────────────────────────────────────────────────────
# 一键包公共路径层 —— 所有 WSL 侧 .sh 都 source 这个文件
#
# 为什么要有它：venv 路径以前写死成 /home/ccb/...（作者机器的用户名），
# 换一台机器、换一个人就全线报错。现在按三级优先级解析，默认值里不含
# 任何用户名：
#
#     环境变量  >  runtime.conf（安装时自动生成，不进仓库）  >  $HOME 下默认值
#
# runtime.conf 同时被 Windows 侧的 PowerShell 读，所以键名一律小写、
# 值只有绝对路径和数字，不含 $HOME 之类的 shell 展开。
#
# 注意：文件名与内容保持 ASCII —— 中文路径经 PowerShell 传给 wsl.exe 会被吃掉。
# ─────────────────────────────────────────────────────────────────────

if [ -z "${_T2_LIB_LOADED:-}" ]; then
  _T2_LIB_LOADED=1

  # --- 项目根目录：按本文件自身位置推导，clone 到哪儿都能用 ------------
  if [ -z "${T2_ROOT:-}" ]; then
    T2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
  fi
  T2_CODE="$T2_ROOT/engine/code"
  T2_CONF="${T2_CONF:-$T2_ROOT/runtime.conf}"

  # --- 日志 -----------------------------------------------------------
  t2_log()  { printf '\n######## %s ########\n' "$*"; }
  t2_step() { printf '  --> %s\n' "$*"; }
  t2_ok()   { printf '  [OK]   %s\n' "$*"; }
  t2_warn() { printf '  [WARN] %s\n' "$*"; }
  t2_bad()  { printf '  [FAIL] %s\n' "$*"; }
  t2_die()  { printf '\n  [错误] %s\n' "$*" >&2; exit 1; }

  # --- 读 runtime.conf 里的一个键（不做 eval，避免任意命令执行）--------
  _conf_get() {
    [ -f "$T2_CONF" ] || return 0
    awk -F= -v k="$1" '
      /^[a-z0-9_]+=/{ key=$1; sub(/^[^=]*=/,""); if (key==k) { print; exit } }
    ' "$T2_CONF" 2>/dev/null
  }
fi

# ─────────────────────────────────────────────────────────────────────
# t2_init —— 解析全部路径，导出给后续脚本用
# 只调用一次；重复调用是幂等的。
# ─────────────────────────────────────────────────────────────────────
t2_init() {
  # 1) 3D 环境
  venv3d="${TRELLIS2_VENV:-$(_conf_get venv3d)}"
  [ -n "$venv3d" ] || venv3d="$HOME/trellis2-wsl-venv"
  # 2) 文生图环境
  venvflux="${FLUX_VENV:-$(_conf_get venvflux)}"
  [ -n "$venvflux" ] || venvflux="$HOME/flux-venv"
  # 3) 权重目录：留在项目里（Windows 盘上，方便手动放文件 / 换机复用）
  models_dir="${TRELLIS2_MODELS:-$(_conf_get models_dir)}"
  [ -n "$models_dir" ] || models_dir="$T2_CODE/MODELS"
  # 4) 端口 / 发行版
  port="${TRELLIS2_PORT:-$(_conf_get port)}"
  [ -n "$port" ] || port=8080
  distro="${TRELLIS2_DISTRO:-$(_conf_get distro)}"
  [ -n "$distro" ] || distro="Ubuntu-24.04"
  # 5) 就绪标记
  weights_ready="${TRELLIS2_WEIGHTS_READY:-$(_conf_get weights_ready)}"
  [ -n "$weights_ready" ] || weights_ready=0
  flux_ready="${TRELLIS2_FLUX_READY:-$(_conf_get flux_ready)}"
  [ -n "$flux_ready" ] || flux_ready=0

  export TRELLIS2_VENV="$venv3d"
  export FLUX_VENV="$venvflux"
  export TRELLIS2_PY="$venv3d/bin/python"
  export FLUX_PY="$venvflux/bin/python"
  export TRELLIS2_MODELS="$models_dir"
  export TRELLIS2_PORT="$port"
  export TRELLIS2_DISTRO="$distro"
  export TRELLIS2_WEIGHTS_READY="$weights_ready"
  export TRELLIS2_FLUX_READY="$flux_ready"
  export HF_HOME="$models_dir"
}

# ─────────────────────────────────────────────────────────────────────
# 探测：CUDA 工具链 / 显卡计算能力 / 磁盘
# 这些都必须实测而不是照抄作者机器：写死 cuda-12.8 和 6.1 意味着
# 换一张非 Pascal 卡（或换了 CUDA 小版本）就编译出跑不动的扩展。
# ─────────────────────────────────────────────────────────────────────

# 找机器上实际装了哪个 cuda-*，返回绝对路径（没有则空）
t2_detect_cuda_home() {
  local d best=""
  for d in /usr/local/cuda-* /opt/cuda-*; do
    [ -x "$d/bin/nvcc" ] || continue
    if [ -z "$best" ]; then best="$d"; else
      best="$(printf '%s\n%s\n' "$best" "$d" | sort -V | tail -1)"
    fi
  done
  if [ -n "$best" ]; then echo "$best"; return 0; fi
  d="$(command -v nvcc 2>/dev/null)" || return 1
  dirname "$(dirname "$d")"
}

# 显卡计算能力，例如 "6.1"。nvidia-smi 优先（不依赖 torch 已装好）。
t2_detect_arch() {
  local cc
  cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' \r')"
  case "$cc" in
    [0-9].[0-9]) echo "$cc"; return 0 ;;
  esac
  # 退回 torch（要求 3D 环境已经建好）
  if [ -x "${TRELLIS2_PY:-}" ]; then
    cc="$("$TRELLIS2_PY" -c 'import torch;print("%d.%d"%torch.cuda.get_device_capability(0))' 2>/dev/null)"
    case "$cc" in
      [0-9].[0-9]) echo "$cc"; return 0 ;;
    esac
  fi
  return 1
}

t2_gpu_name() {
  nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr -d '\r'
}

# 显存总量（MiB），拿不到就返回空
t2_gpu_mem_mib() {
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
    | head -1 | tr -d ' \r'
}

# 某个路径所在盘还剩多少 GB（整数）
t2_free_gb() {
  df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9'
}

# WSL 里能不能看到显卡
t2_has_gpu() {
  [ -e /dev/dxg ] || return 1
  nvidia-smi -L >/dev/null 2>&1
}

# ─────────────────────────────────────────────────────────────────────
# t2_save_conf —— 把当前解析结果写回 runtime.conf（供 PowerShell 读）
# ─────────────────────────────────────────────────────────────────────
t2_save_conf() {
  local win=""
  if command -v wslpath >/dev/null 2>&1; then
    win="$(wslpath -w "$T2_ROOT" 2>/dev/null)"
  fi
  local cuda arch wready fready
  cuda="${TRELLIS2_CUDA_HOME:-$(_conf_get cuda_home)}"
  [ -n "$cuda" ] || cuda="$(t2_detect_cuda_home 2>/dev/null)"
  arch="${TRELLIS2_ARCH:-$(_conf_get cuda_arch)}"
  [ -n "$arch" ] || arch="$(t2_detect_arch 2>/dev/null)"
  # set -u 下这两个可能从未导出过，不能直接引用
  wready="${TRELLIS2_WEIGHTS_READY:-$(_conf_get weights_ready)}"
  [ -n "$wready" ] || wready=0
  fready="${TRELLIS2_FLUX_READY:-$(_conf_get flux_ready)}"
  [ -n "$fready" ] || fready=0

  {
    echo "# 由一键安装自动生成，记录了这台机器上的实际路径。不要提交进 git。"
    echo "# 手改也行：键值对，绝对路径，别写 \$HOME。"
    echo "wsl_root=$T2_ROOT"
    [ -n "$win" ] && echo "win_root=$win"
    echo "distro=${TRELLIS2_DISTRO:-Ubuntu-24.04}"
    echo "venv3d=$TRELLIS2_VENV"
    echo "venvflux=$FLUX_VENV"
    echo "models_dir=$TRELLIS2_MODELS"
    echo "port=$TRELLIS2_PORT"
    [ -n "$cuda" ] && echo "cuda_home=$cuda"
    [ -n "$arch" ] && echo "cuda_arch=$arch"
    echo "weights_ready=$wready"
    echo "flux_ready=$fready"
  } > "$T2_CONF" || t2_die "写 $T2_CONF 失败"
  t2_ok "已写入配置：$T2_CONF"
}

# 更新 runtime.conf 里的单个键（其余行原样保留）
t2_conf_set() {
  local key="$1" val="$2" tmp
  [ -f "$T2_CONF" ] || : > "$T2_CONF"
  tmp="$T2_CONF.tmp.$$"
  awk -F= -v k="$key" -v v="$val" '
    BEGIN{done=0}
    /^[a-z0-9_]+=/ { if ($1==k) { print k"="v; done=1; next } }
    { print }
    END{ if(!done) print k"="v }
  ' "$T2_CONF" > "$tmp" && mv "$tmp" "$T2_CONF"
}

# ─────────────────────────────────────────────────────────────────────
# t2_ensure_models_link —— 把 engine/code/MODELS 指到 $TRELLIS2_MODELS
#
# 为什么需要：加载那一侧（app.py 用 __file__、各 feature extractor 用
# os.getcwd()）只认 engine/code/MODELS 这个固定位置，它**不认识**
# TRELLIS2_MODELS。想把几十 GB 权重放到别的盘，光设变量没用，
# 必须让这个路径本身可达 —— 用符号链接，不动 engine 里任何源文件。
# ─────────────────────────────────────────────────────────────────────
t2_ensure_models_link() {
  local want="${1:?用法: t2_ensure_models_link <权重目录>}"
  local def="$T2_CODE/MODELS" real
  [ "$(readlink -f "$want")" = "$(readlink -f "$def")" ] && return 0
  if [ -L "$def" ]; then
    real="$(readlink -f "$def")"
    [ "$real" = "$(readlink -f "$want")" ] && return 0
    t2_die "engine/code/MODELS 已经是指向 $real 的链接，和你要用的 $want 不一致。
     确认那个链接没用了再手工删掉：rm \"$def\"，然后重跑。"
  fi
  if [ -d "$def" ]; then
    if [ -n "$(ls -A "$def" 2>/dev/null)" ]; then
      t2_die "$def 里已经有文件，不能把它改成链接（那样权重会看起来全部丢失）。
       要么换回默认路径，要么先把里面的东西搬到 $want。"
    fi
    rmdir "$def" || t2_die "删掉空目录 $def 失败"
  fi
  mkdir -p "$(dirname "$def")" "$want"
  ln -sfn "$want" "$def" || t2_die "创建链接 $def -> $want 失败"
  t2_ok "engine/code/MODELS -> $want（加载侧找的就是这个路径）"
}

# ─────────────────────────────────────────────────────────────────────
# 装 WSL 侧的系统依赖。_setup_3d_venv.sh 编译 CUDA 扩展要用 nvcc，
# 全新 Ubuntu 里没有 —— 以前这一步缺失，导致别人 clone 后编译必失败。
# ─────────────────────────────────────────────────────────────────────
t2_base_tools_missing() {
  local p
  for p in python3 git curl unzip g++ make pkg-config; do
    command -v "$p" >/dev/null 2>&1 || return 0
  done
  python3 -c 'import ensurepip' >/dev/null 2>&1 || return 0
  return 1
}

t2_sudo_ready() {
  [ "$(id -u)" = "0" ] && return 0
  sudo -n true >/dev/null 2>&1
}

# 参数：期望的 CUDA 版本（如 12.8），可为空
t2_ensure_system_deps() {
  local want_cuda="${1:-12.8}"
  local SUDO=""
  [ "$(id -u)" = "0" ] || SUDO="sudo -n "

  # 先算清楚"到底要不要装"。以前不管三七二十一先要 sudo，结果依赖早就齐的机器
  # （或者没配免密 sudo 的机器）在第一步就被一句"需要 sudo"挡死 ——
  # 什么都不用装的时候根本不该碰 sudo。
  if ! t2_base_tools_missing && [ -n "$(t2_detect_cuda_home 2>/dev/null)" ]; then
    t2_ok "系统依赖齐全（python3 / git / g++ / nvcc），不需要 sudo"
    return 0
  fi

  # 全新 WSL 里第一次用 sudo 是要输密码的。以前只认免密 sudo，判定失败就直接退出，
  # 别人第一次跑一键包必然卡在这里 —— 而安装本来是跑在当前控制台里的
  # （_start.ps1 特意没把它藏进后台窗口），所以这里可以正当地问一次密码。
  if t2_sudo_ready; then
    :
  elif [ -t 0 ] && sudo -v; then
    SUDO="sudo "
  else
    t2_die "需要 sudo 才能装系统依赖，但没能认证成功。请先在 WSL 里执行一次：
       sudo -v
     或直接手动装：
       sudo apt update && sudo apt install -y python3-venv python3-dev git curl unzip build-essential cuda-toolkit-${want_cuda/./-}"
  fi

  if t2_base_tools_missing; then
    t2_log "安装 WSL 基础依赖（python3-venv / build-essential / git / curl）"
    $SUDO apt-get update -qq || t2_warn "apt update 失败，继续尝试用已有源"
    $SUDO apt-get install -y --no-install-recommends \
      python3-venv python3-dev python3-pip git curl ca-certificates unzip \
      build-essential pkg-config libgl1 libglib2.0-0 || t2_die "apt 安装失败，见上面的报错"
    t2_ok "基础工具已装"
  else
    t2_ok "基础工具齐全"
  fi

  # nvcc：编译 flex_gemm / cumesh / nvdiffrast / o_voxel 全都依赖它。
  # 全新 Ubuntu-24.04 里没有，所以必须在这里装 —— 以前缺这一步，
  # 别人 clone 后编译阶段必然全线失败。
  if [ -n "$(t2_detect_cuda_home 2>/dev/null)" ]; then
    t2_ok "已有 nvcc：$(t2_detect_cuda_home)"
    return 0
  fi

  local maj="${want_cuda%%.*}" min="${want_cuda##*.}"
  local dist
  dist="$( (. /etc/os-release 2>/dev/null; printf '%s' "${VERSION_ID//./}") )"
  case "$dist" in 2[0-9]*) : ;; *) dist=2404 ;; esac

  t2_step "安装 CUDA 编译器 cuda-toolkit-$maj-$min（约 2~3 GB，几分钟）"
  local deb=/tmp/nv-cuda-keyring.deb
  if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
    if $SUDO curl -fsSL -o "$deb" \
        "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${dist}/x86_64/cuda-keyring_1.1-1_all.deb"; then
      $SUDO dpkg -i "$deb" >/dev/null || t2_warn "cuda-keyring 安装报错，继续"
      $SUDO rm -f "$deb"
      $SUDO apt-get update -qq || t2_warn "加源后 apt update 失败"
    else
      t2_warn "下不到 NVIDIA 的 apt 源（网络/镜像问题），改用 Ubuntu 自带的 nvidia-cuda-toolkit"
    fi
  fi

  # 依次退让：精确小版本 -> 任意 cuda-toolkit -> Ubuntu 自带的 12.0
  if $SUDO apt-get install -y --no-install-recommends "cuda-toolkit-${maj}-${min}" 2>/dev/null \
    || $SUDO apt-get install -y --no-install-recommends "cuda-toolkit-${maj}" 2>/dev/null \
    || $SUDO apt-get install -y --no-install-recommends nvidia-cuda-toolkit 2>/dev/null; then
    t2_ok "nvcc 已装：$(t2_detect_cuda_home 2>/dev/null || echo '（仍找不到，见下方自检）')"
  else
    t2_die "装不上 CUDA 编译器。请手动执行后重跑：
       sudo apt install -y cuda-toolkit-12-8
     或从 https://developer.nvidia.com/cuda-downloads 选 'WSL-Ubuntu' 安装"
  fi
}

# ─────────────────────────────────────────────────────────────────────
# 编译并行度：写死 MAX_JOBS=4 是浪费 —— 12 核的机器要编一辈子。
# 但也不能给到核数上限：nvcc/cc1plus 单个 job 峰值 1.5~2GB，编 CUDA 扩展时
# 超内存会直接被 OOM killer 干掉，报错还长得像编译失败。所以按
# 「核数」和「(内存-2GB)/2GB」里较小的那个来，两头都不吃亏。
# 想手工指定就跑：MAX_JOBS=8 bash _setup_3d_venv.sh
# ─────────────────────────────────────────────────────────────────────
t2_build_jobs() {
  local cores mem_gb by_mem jobs
  cores="$(nproc 2>/dev/null || echo 4)"
  mem_gb="$(awk '/MemTotal/{printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null || echo 8)"
  by_mem=$(( (mem_gb - 2) / 2 ))
  [ "$by_mem" -ge 1 ] || by_mem=1
  jobs="$by_mem"
  [ "$jobs" -gt "$cores" ] && jobs="$cores"
  printf '%s' "$jobs"
}

# torch 的 CUDA 构建要和显卡对齐：
# Blackwell（sm_100/sm_120）在 cu124 wheel 里没有内核，必须用 cu128。
# 返回值给 _setup_3d_venv.sh 当 pip 的 --extra-index-url 尾巴。
t2_torch_cuda_tag() {
  local arch="${1:-$(t2_detect_arch 2>/dev/null)}"
  case "$arch" in
    1[0-9].[0-9]|12.[0-9]) echo cu128 ;;
    *) echo cu124 ;;
  esac
}

# ─────────────────────────────────────────────────────────────────────
# pip 源探测 —— 3D 环境和 FLUX 环境共用一套，别在两个脚本里各抄一份
# ─────────────────────────────────────────────────────────────────────

# 国内网络下 .wslconfig 的 autoProxy 会把 http_proxy 注入 WSL，导致连国内镜像
# 也被绕到代理上。实测阿里云走代理 1.74 MB/s、直连 12.71 MB/s（快 7 倍），
# 而 pypi.org / download.pytorch.org 又必须走代理。
# no_proxy 是"追加"不是覆盖，保留 autoProxy 原有的内网段。
t2_setup_no_proxy() {
  export no_proxy="${no_proxy:-},mirrors.aliyun.com,mirrors.cloud.tencent.com,repo.huaweicloud.com,mirror.sjtu.edu.cn,mirrors.tuna.tsinghua.edu.cn,mirrors.bfsu.edu.cn,mirrors.nju.edu.cn,mirrors.ustc.edu.cn,modelscope.cn,.modelscope.cn,hf-mirror.com,.hf-mirror.com,download.pytorch.org"
  export NO_PROXY="$no_proxy"
}

# 探一个 URL 的 HTTP 状态码（拿不到返回 000）
t2_http_code() {
  curl -s -o /dev/null -w '%{http_code}' --max-time 12 "$1" 2>/dev/null || echo 000
}

# 挑一个**当前可达且最快**的 pip 源并 echo 出来。
# 绝不写死某一个镜像：换一台机器、换一个网络，能通的镜像就不同 ——
# 写死清华源的结果是别人在代理后面卡到超时。
# 用法： IDX="$(t2_pick_pip_index "${PIP_INDEX_URL:-}")"
t2_pick_pip_index() {
  local force="${1:-}"
  if [ -n "$force" ]; then
    echo "  使用指定源：$force" >&2
    echo "$force"
    return 0
  fi
  local best="" bestt=99 u code t
  for u in \
    https://mirrors.aliyun.com/pypi/simple/ \
    https://mirrors.cloud.tencent.com/pypi/simple \
    https://repo.huaweicloud.com/repository/pypi/simple \
    https://pypi.org/simple/ \
    https://pypi.tuna.tsinghua.edu.cn/simple \
    https://mirror.sjtu.edu.cn/pypi/web/simple ; do
    code="$(t2_http_code "$u/torch/")"
    if [ "$code" = "200" ]; then
      t="$(curl -s -o /dev/null -w '%{time_total}' --max-time 12 "$u/torch/" 2>/dev/null || echo 99)"
      echo "  可用：$u  (${t}s)" >&2
      awk -v a="$t" -v b="$bestt" 'BEGIN{exit !(a<b)}' && { best="$u"; bestt="$t"; }
    else
      echo "  不可用：$u  (HTTP $code)" >&2
    fi
  done
  if [ -z "$best" ]; then
    echo "  !! 所有镜像都不可达，回退到 pypi.org" >&2
    best="https://pypi.org/simple/"
  else
    echo "  选中：$best" >&2
  fi
  echo "$best"
}
