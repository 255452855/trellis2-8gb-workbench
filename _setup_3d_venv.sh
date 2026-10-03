#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# 在一台**全新机器**上重建 3D 运行环境（TRELLIS.2 / Pixal3D）
#
# 要解决的问题：本仓库不含 venv，别人 clone 下来什么都跑不了；而以前建环境
# 是作者在自己机器上手工敲的命令，路径、CUDA 版本、显卡架构全写死。
# 结论必须回答：换一个人、换一块卡、换一张盘，这条命令还能一次跑通吗？
#
# 三条硬规矩（都是实测踩出来的）：
#   1. **绝不碰现有环境**。默认建到 ${TRELLIS2_VENV}-new，自检通过后才由
#      --activate 把 runtime.conf 指过去。全程不 mv、不 rm 老 venv ——
#      老环境是别人唯一能跑的东西，编坏一次就再也起不来了。
#   2. 路径/架构/CUDA 一律现场探测（_lib.sh 的 t2_detect_*），不写死
#      /home/ccb、cuda-12.8、sm_61。
#   3. pip 源、PyTorch 的 cuXXX 都要按网络和本卡实际选，镜像写死 = 别人超时。
#
#   bash _setup_3d_venv.sh                 # 建到 ${TRELLIS2_VENV}-new（安全）
#   bash _setup_3d_venv.sh --activate      # 自检通过后把 runtime.conf 指向它
#   bash _setup_3d_venv.sh /自定义/路径     # 建到指定路径
#
# 需要编译的扩展及其来源：
#   flex_gemm       <- github.com/JeffreyXiang/FlexGEMM
#   cumesh          <- github.com/JeffreyXiang/CuMesh
#   nvdiffrast      <- github.com/NVlabs/nvdiffrast
#   o_voxel         <- 本仓库 engine/code/o-voxel（源码自带）
#   natten          <- PyPI 只有 sdist（要本地编）；官方 release wheel 仅覆盖 torch 2.7
#   nvdiffrec_render<- 单独一步：_setup_nvdiffrec.sh（从 NVlabs 源码装）
# ─────────────────────────────────────────────────────────────────────
set -u
set -o pipefail

. "$(cd "$(dirname "$0")" && pwd)/_lib.sh"
t2_init
t2_setup_no_proxy

TARGET="${TRELLIS2_VENV_NEW:-$TRELLIS2_VENV-new}"
ACTIVATE=0
for a in "$@"; do
  case "$a" in
    --activate) ACTIVATE=1 ;;
    /*)         TARGET="$a" ;;
  esac
done
PY="$TARGET/bin/python"
LOCK="$T2_ROOT/_requirements-3d-lock.txt"
WORK="${TMPDIR:-/tmp}/trellis2-build"
FAILED=""
mkdir -p "$WORK"

# ── 0. 先看现有环境是不是已经能用了（避免白折腾 1 小时）──────────────
if [ -x "$TRELLIS2_PY" ] && "$TRELLIS2_PY" -c \
     'import torch, gradio, flex_gemm, cumesh, nvdiffrast, o_voxel' >/dev/null 2>&1 \
   && [ "$TARGET" = "$TRELLIS2_VENV" ]; then
  t2_ok "现有环境已经可用，无需重建：$TRELLIS2_VENV"
  exit 0
fi

ARCH="$(t2_detect_arch 2>/dev/null || true)"
CUDA_HOME_B="$(t2_detect_cuda_home 2>/dev/null || true)"
t2_log "建 3D 环境"
echo "  显卡      : $(t2_gpu_name)  $(t2_gpu_mem_mib)MiB  sm_${ARCH:-?}"
echo "  目标路径  : $TARGET"
echo "  现有环境  : ${TRELLIS2_VENV}（本脚本不会移动或删除它）"
echo "  nvcc      : ${CUDA_HOME_B:-尚未安装，下面会自动装}"
echo "  磁盘剩余  : $(t2_free_gb "$(dirname "$TARGET")") GB（$TARGET 所在盘，环境和编译中间产物都要落在这）"
[ -n "$ARCH" ] || t2_warn "探不到显卡计算能力 —— 编译会退回 native，换卡后需重编"

# nvcc / g++ / python3-venv：全新 Ubuntu 上这几个都没有，缺一个就全线编译失败
t2_ensure_system_deps

IDX="$(t2_pick_pip_index "${PIP_INDEX_URL:-}")"

# ── 1. venv ─────────────────────────────────────────────────────────
t2_log "1/6 建虚拟环境：$TARGET"
[ -x "$PY" ] || python3 -m venv "$TARGET" || t2_die "python3 -m venv 失败（缺 python3-venv？上面刚装过，重跑一次试试）"
"$PY" -m pip install -q --upgrade pip setuptools wheel -i "$IDX" || t2_die "pip 自升级失败：$IDX 通吗"

# ── 2. PyTorch：按本卡架构选 wheel，不能照抄 cu124 ───────────────────
# Blackwell（sm_100/sm_120）在 2.6.0 的 cu124 wheel 里**没有内核**，
# 必须用 2.7.0+cu128；Pascal/Turing 则继续用 2.6.0+cu124（本仓库实测的那版）。
t2_log "2/6 安装 PyTorch"
TAG="$(t2_torch_cuda_tag "${ARCH:-}")"
case "$TAG" in
  cu128) T_VER=2.7.0; T_VISION=0.22.0 ;;
  *)     T_VER=2.6.0; T_VISION=0.21.0 ;;
esac
PYTAG="$("$PY" -c 'import sys;print("cp%d%d" % sys.version_info[:2])' 2>/dev/null || echo cp000)"
if ! "$PY" -c 'import torch' >/dev/null 2>&1; then
  # PyPI（含国内镜像）上不带 +cuXXX 后缀的 torch **默认就是 cu124 构建**，
  # 对 Pascal 正好能用，而阿里云实测比 download.pytorch.org 快一个数量级。
  # 只有需要别的 CUDA 构建（如 Blackwell 的 cu128）才走 PyTorch 官方源。
  if [ "$TAG" = "cu124" ] && "$PY" -m pip install -i "$IDX" \
       "torch==$T_VER" "torchvision==$T_VISION" "torchaudio==$T_VER" 2>&1 | tail -4; then
    :
  else
    echo "  镜像这条路没装成（或不适用），改用 PyTorch 官方 $TAG 源"
    "$PY" -m pip install --index-url "https://download.pytorch.org/whl/$TAG" \
      --extra-index-url "$IDX" \
      "torch==$T_VER" "torchvision==$T_VISION" "torchaudio==$T_VER" 2>&1 | tail -4 \
      || FAILED="$FAILED torch"
  fi
else
  echo "  torch 已装：$("$PY" -c 'import torch;print(torch.__version__)')"
fi
"$PY" -m pip install -q -i "$IDX" ninja psutil 2>/dev/null

# ── 3. 其余依赖：拿实测锁文件，剔除必须现场编译/装的那几个 ───────────
t2_log "3/6 安装其余依赖（按 _requirements-3d-lock.txt 的实测版本）"
if [ -f "$LOCK" ]; then
  # 剔除理由：
  #   * flex_gemm/cumesh/nvdiffrast/o_voxel/natten/nvdiffrec —— 下面单独编或装
  #   * torch/torchvision/torchaudio —— 锁文件里是 `==2.6.0+cu124` 这种本地版本，
  #     PyPI 镜像上没有，直接 -r 会 "No matching distribution"
  grep -viE '^(flex[_-]gemm|natten|nvdiffrast|cumesh|nvdiffrec[_-]render|o[_-]voxel|torch|torchvision|torchaudio)([= @]|$)' \
    "$LOCK" > "$WORK/req.txt"
  echo "  $(grep -c '^[a-z]' "$WORK/req.txt") 个包"
  "$PY" -m pip install -i "$IDX" -r "$WORK/req.txt" 2>&1 | tail -6 || FAILED="$FAILED deps"
else
  echo "  [警告] 找不到 $LOCK，只装核心包"
  "$PY" -m pip install -i "$IDX" gradio==6.0.1 transformers safetensors \
    pillow opencv-python trimesh tqdm ninja psutil 2>&1 | tail -4 || FAILED="$FAILED deps"
fi

# ── 预编译 wheel：这套 ABI 对得上的人不必再等编译 ────────────────────
# 上游 FlexGEMM / CuMesh / o-voxel **都不发布 wheel**（CuMesh 仓库连 release 都没有），
# 所以只能我们自己编一次、把产物留下来复用。取用顺序：
#   本地 prebuilt-wheels/ → 同 ABI 标签的 GitHub Release 包 → 现场编译
# 现场编译走 `pip wheel` 而不是 `pip install`：同一次编译顺手把 .whl 存进
# prebuilt-wheels/，以后换 venv / 重装环境直接秒装，不必再编一遍。
# ⚠️ 标签必须整串对上（python + torch + CUDA + 显卡架构）。装错 ABI 的 wheel
#    比多等半小时危险得多，所以对不上就老老实实现编。
ABI_TAG="py${PYTAG#cp}-torch${T_VER}-cu${TAG#cu}-sm${ARCH//./}"
WHEEL_STORE="${TRELLIS2_WHEEL_DIR:-$T2_ROOT/prebuilt-wheels}"
WHEEL_URL="${TRELLIS2_WHEEL_URL:-https://github.com/255452855/trellis2-8gb-workbench/releases/download/wheels-$ABI_TAG}"
WHEEL_TARBALL="wheels-$ABI_TAG.tar"
mkdir -p "$WHEEL_STORE"

wheel_of() {  # wheel_of <包名> -> 本地已有的 wheel 路径，没有则空
  ls -1t "$WHEEL_STORE"/"$1"-*.whl 2>/dev/null | head -1
}

_fetch_wheel_pack() {  # 下载并解包一次（多个包共用同一个 tar）
  [ -n "${WHEEL_PACK_TRIED:-}" ] && return 1
  WHEEL_PACK_TRIED=1
  local url="$WHEEL_URL/$WHEEL_TARBALL" out="$WORK/$WHEEL_TARBALL" rc=0
  # 上一轮已经下完并留在这个路径里了，就别再浪费一次流量
  if tar -tf "$out" >/dev/null 2>&1; then
    tar -xf "$out" -C "$WHEEL_STORE" && { echo "  预编译包已解到 $WHEEL_STORE"; return 0; }
  fi
  # 实测（51 号）本机到 GitHub Release 只有 ~30KB/s，74MB 要 40 分钟；
  # 现编全套反而 35 分钟就完。所以这里必须能"这一轮没下完、下一轮接着下"，
  # 否则慢网络的机器永远拿不到预编译包：-C - 从本地已有字节数续传。
  echo "  试取预编译包（74MB，慢网络下可分多轮续传）：$url"
  curl -fsL --retry 3 --retry-delay 5 -C - --max-time 1800 -o "$out" "$url" 2>/dev/null || rc=$?
  if [ "$rc" -ne 0 ]; then
    # 33 = 服务器拒绝 Range，说明本地这个文件已经到大小上限却是坏的，留着也没用
    [ "$rc" = 33 ] && rm -f "$out"
    local got=""
    [ -f "$out" ] && got="，本地已存 $(du -m "$out" 2>/dev/null | cut -f1)MB"
    echo "  （预编译包没取到：curl 退出码 $rc$got；本轮走现编，下次重跑会从断点继续下载）"
    return 1
  fi
  if ! tar -xf "$out" -C "$WHEEL_STORE"; then
    echo "  预编译包解包失败，删除残缺文件（下次重新下载）"
    rm -f "$out"; return 1
  fi
  echo "  预编译包已解到 $WHEEL_STORE"
}

build_or_wheel() {  # build_or_wheel <import 名> <包名> <源码目录>
  local mod="$1" pkg="$2" dir="$3" f
  [ -z "$(wheel_of "$pkg")" ] && _fetch_wheel_pack
  f="$(wheel_of "$pkg")"
  if [ -n "$f" ] && "$PY" -m pip install --no-deps "$f" 2>&1 | tail -3 \
     && "$PY" -c "import $mod" >/dev/null 2>&1; then
    echo "  ✓ $pkg —— 用预编译 wheel，免编译（$(basename "$f")）"
    return 0
  fi
  if [ ! -d "$dir" ]; then
    echo "  ✗ $pkg：没有匹配的预编译 wheel（$ABI_TAG），也没有源码目录"
    FAILED="$FAILED $pkg"; return 1
  fi
  local blog="$WORK/$pkg.build.log"
  echo "  --- 现编 $pkg（没有可复用的预编译产物，日志 $blog）---"
  if [ -d "$dir" ] \
     && "$PY" -m pip wheel --no-build-isolation --no-deps -w "$WHEEL_STORE" "$dir" > "$blog" 2>&1 \
     && f="$(wheel_of "$pkg")" && [ -n "$f" ] \
     && "$PY" -m pip install --no-deps "$f" >> "$blog" 2>&1 \
     && "$PY" -c "import $mod" >/dev/null 2>&1; then
    echo "  ✓ $pkg 编好并留档：$(basename "$f")（下次换环境直接秒装）"
    return 0
  fi
  echo "  ✗ $pkg 失败 —— 报错尾部 40 行："
  tail -40 "$blog" 2>/dev/null | sed 's/^/      /'
  echo "      完整日志：$blog"
  FAILED="$FAILED $pkg"
  return 1
}

clone_src() {  # clone_src <git url> <目录名> -> 打印本地路径
  local url="$1" dir="$WORK/$2"
  # --recursive 是必需的：CuMesh 上游 README 就写着 clone --recursive，
  # 它的子模块里有头文件，漏了会以"编译失败"的面目出现，最难往这上面想。
  if [ -d "$dir/.git" ]; then
    # 上一次跑到这儿就失败了：目录在、子模块却没 init（submodule status 里那行
    # 前导减号就是没 init）。不补这一步，报出来的还是"编译失败"，
    # 谁也不会往"子模块空着"上想。
    git -C "$dir" submodule update --init --recursive --depth 1 2>&1 | tail -2
  else
    rm -rf "$dir"; git clone --depth 1 --recursive --shallow-submodules "$url" "$dir" 2>&1 | tail -2
  fi
  printf '%s' "$dir"
}

# 三件套一起用：预编译产物命中就完全不 clone 源码（省一次网络 + 省半小时）
build_from_git() {  # build_from_git <import 名> <包名> <git url> <目录名>
  local mod="$1" pkg="$2" url="$3" sub="$4" dir=""
  [ -n "$(wheel_of "$pkg")" ] || _fetch_wheel_pack
  [ -n "$(wheel_of "$pkg")" ] || dir="$(clone_src "$url" "$sub")"
  build_or_wheel "$mod" "$pkg" "$dir"
}

# ── 4. natten：NAF 邻域注意力的 kernel ───────────────────────────────
# 实测结论（36 号）：PyPI 上 0.21.0 **只有 sdist**，装它=本地编；
# 作者那份能跑的环境里 dist-info 写着 Generator: setuptools / cp312-linux_x86_64，
# 也正是本地编出来的。官方 release 的预编译 wheel 只覆盖 torch 2.7（cu126/cu128），
# 对这套 torch 2.6+cu124 用不上。
# 关键开关是 --no-build-isolation：natten 的 pyproject 要在构建期 import torch，
# 默认行为会另起一个隔离环境重新抓 torch，于是报
#   "Failed to build 'natten' when getting requirements to build wheel"
# 用本 venv 里刚装好的 torch 就没这个问题。
t2_log "4/6 natten（注意力 kernel）"
natten_official_wheel() {  # 只有 torch>=2.7 才有官方预编译 wheel
  case "$T_VER" in
    2.7*) printf '%s' "https://github.com/SHI-Labs/NATTEN/releases/download/v0.21.0/natten-0.21.0%2Btorch${T_VER/./}${TAG#cu}-${PYTAG}-${PYTAG}-linux_x86_64.whl" ;;
    *) return 1 ;;
  esac
}
if ! "$PY" -c 'import natten' >/dev/null 2>&1; then
  # 先查本机留档的 wheel：natten 现编实测 28 分 55 秒，是整个安装里最耗时的一步，
  # 有留档产物却不用是说不过去的。
  # ⚠️ 先试着把留档包取回来，再判 wheel_of：wheel_of/_fetch_wheel_pack 原本只在第 5 步
  #    才下载 Release 包，而 natten 在第 4 步 —— 结果新机器上明明有预编译 natten，
  #    却仍然会去现编 28 分 55 秒（本机有留档时看不出来，只有空目录才暴露）。
  [ -n "$(wheel_of natten)" ] || _fetch_wheel_pack
  NATW="$(wheel_of natten)"
  NAT="$(natten_official_wheel 2>/dev/null || true)"
  # 三条路必须是互斥的 if/elif 链：先若命中留档 wheel，就绝不能再跑一次
  # --no-build-isolation（那会打出"现编装好"这句假汇报，其实只是 already satisfied）。
  if [ -n "$NATW" ] && "$PY" -m pip install --no-deps "$NATW" > "$WORK/natten.log" 2>&1 \
     && "$PY" -c 'import natten' >/dev/null 2>&1; then
    echo "  ✓ natten 用留档的预编译 wheel：$(basename "$NATW")"
  elif [ -n "$NAT" ] && "$PY" -m pip install "$NAT" > "$WORK/natten.log" 2>&1; then
    echo "  ✓ natten 用官方预编译 wheel"
  elif "$PY" -m pip install --no-build-isolation -i "$IDX" "natten==0.21.0" > "$WORK/natten.log" 2>&1 \
       && "$PY" -c 'import natten' >/dev/null 2>&1; then
    echo "  ✓ natten 现编装好（日志 $WORK/natten.log）"
  else
    echo "  ✗ natten 失败 —— 报错尾部："
    tail -30 "$WORK/natten.log" 2>/dev/null | sed 's/^/      /'
    FAILED="$FAILED natten"
  fi
else
  echo "  natten 已装：$("$PY" -c 'import natten;print(natten.__version__)')"
fi

# ── 5. 编译 CUDA 扩展 ───────────────────────────────────────────────
# 链接期也要给 CUDA 的 stubs 目录：LD_LIBRARY_PATH 只管运行期，
# 缺 LIBRARY_PATH 时 -lcuda 找不到 libcuda.so（WSL 真实驱动叫 libcuda.so.1）。
export CUDA_HOME="${CUDA_HOME:-$CUDA_HOME_B}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:/usr/lib/wsl/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${ARCH:-native}}"
export MAX_JOBS="${MAX_JOBS:-$(t2_build_jobs)}"   # 并行度按核数和内存算，见 _lib.sh

t2_log "5/6 扩展：flex_gemm / cumesh / nvdiffrast / o_voxel（ABI 标签 $ABI_TAG）"
build_from_git flex_gemm   flex_gemm   https://github.com/JeffreyXiang/FlexGEMM.git FlexGEMM
build_from_git cumesh      cumesh      https://github.com/JeffreyXiang/CuMesh.git   CuMesh
build_from_git nvdiffrast  nvdiffrast  https://github.com/NVlabs/nvdiffrast.git     nvdiffrast
# ── o_voxel 少不得的构建依赖：third_party/eigen ──────────────────────
# ⚠️ 顺序不能乱：o_voxel 的 o_voxel/postprocess.py 里 `import cumesh`，
#    所以 cumesh 必须在 o_voxel 之前装好，否则 o_voxel 的 import 自检会
#    报"没有 cumesh"这种看着像自己坏了的假故障（41 号实测就是这个现象）。
# o-voxel/setup.py 的 include_dirs 里有 third_party/eigen，而这个目录被
# .gitignore 排掉了（不发布第三方源码是对的），结果发布物天生缺它 →
# 别人编 o_voxel 100% 失败，报的还是一句看不懂的 failed-wheel-build（36 号实测）。
# 所以编译前把它取回来：header-only、只当构建期依赖、不进仓库。
ensure_eigen() {
  local d="$T2_CODE/o-voxel/third_party/eigen"
  [ -d "$d/Eigen" ] && { echo "  eigen 已就位"; return 0; }
  mkdir -p "$T2_CODE/o-voxel/third_party"
  echo "  取 eigen 头文件到 engine/code/o-voxel/third_party/eigen（o_voxel 编译需要）"
  git clone --depth 1 --branch 3.4.0 https://gitlab.com/libeigen/eigen.git "$d" > "$WORK/eigen.log" 2>&1 \
    || git clone --depth 1 https://github.com/eigenteam/eigen-git-mirror.git "$d" >> "$WORK/eigen.log" 2>&1
  if [ -d "$d/Eigen" ]; then
    echo "  ✓ eigen 就位"
  else
    echo "  ✗ eigen 没取到（见 $WORK/eigen.log），o_voxel 这步会失败"
  fi
}
ensure_eigen
build_or_wheel o_voxel o_voxel "$T2_CODE/o-voxel"
echo "  nvdiffrec_render 不在这儿装 —— 由 _oneclick.sh 的下一步单独处理"

# ── 6. 自检 ─────────────────────────────────────────────────────────
t2_log "6/6 自检（这几个模块缺一个，页面就会在生成中途报错）"
if ! "$PY" - <<'PYEOF'
import importlib, sys
mods = ["torch", "torchvision", "gradio", "transformers", "flex_gemm",
        "natten", "nvdiffrast", "cumesh", "o_voxel"]
bad = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print("  OK   %-14s %s" % (m, getattr(mod, "__version__", "")))
    except Exception as e:
        print("  FAIL %-14s %s: %s" % (m, type(e).__name__, str(e)[:70]))
        bad.append(m)
import torch
print("  cuda :", torch.cuda.is_available(), "|", torch.version.cuda, "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
if not torch.cuda.is_available():
    bad.append("cuda")
print("  BAD  :", bad or "无")
sys.exit(1 if bad else 0)
PYEOF
then
  FAILED="$FAILED 自检"
fi

echo
if [ -n "$FAILED" ]; then
  echo "######## 有失败项：$FAILED ########"
  echo "新环境留在 $TARGET，**没有动** $TRELLIS2_VENV。"
  echo "把上面的报错整段发出来即可定位；修好重跑本脚本，已完成的步骤会自动跳过。"
  exit 1
fi

t2_ok "环境建好并自检通过：$TARGET"
if [ "$ACTIVATE" = "1" ]; then
  # 只改配置指针，不搬目录：老环境原地保留，随时能改回去
  t2_conf_set venv3d "$TARGET"
  t2_conf_set cuda_home "$CUDA_HOME"
  t2_conf_set cuda_arch "${ARCH:-}"
  t2_ok "runtime.conf 的 venv3d 已指向 $TARGET（$TRELLIS2_VENV 原样保留）"
else
  echo "下一步：确认没问题后跑 bash _setup_3d_venv.sh --activate 把它设为正式环境，"
  echo "        或者手动 TRELLIS2_VENV=$TARGET bash _oneclick.sh --status"
fi
