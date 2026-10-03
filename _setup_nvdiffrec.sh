#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# 重建 PBR 材质渲染依赖 nvdiffrec_render（NVIDIA 专有，不随本仓库分发）
#
# 为什么能在**新机器**上装：不需要任何人提供 wheel —— 纯 Python 部分就是
# NVlabs/nvdiffrec 的 render/ 目录，编译部分是它自带的 CUDA 源码，
# 由 torch 的 cpp_extension 在**安装期**编一次并缓存好。
#
# WSL 里的关键坑（这就是以前手工折腾半天的原因）：上游 ops.py 链接时要
# `-lcuda -lnvrtc`，而 libcuda.so 只在 CUDA 的 stubs 目录里，WSL 的真实驱动
# 库叫 libcuda.so.1。只设 LD_LIBRARY_PATH 不够 —— 那是运行期的；
# **链接期要 LIBRARY_PATH**。补上这一个变量，编译就过了（实测）。
#
#   bash _setup_nvdiffrec.sh              # 装进配置里的 3D 环境
#   bash _setup_nvdiffrec.sh --status     # 只检查
#   bash _setup_nvdiffrec.sh --venv <路径>
# ─────────────────────────────────────────────────────────────────────
set -u
set -o pipefail

. "$(cd "$(dirname "$0")" && pwd)/_lib.sh"
t2_init

STATUS_ONLY=0
VENV="${TRELLIS2_VENV:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --venv)   shift; VENV="${1:-}" ;;
    --status) STATUS_ONLY=1 ;;
    *)        echo "未知参数：$1" >&2; exit 2 ;;
  esac
  shift
done

PY="${VENV:+$VENV/bin/python}"
PY="${PY:-$TRELLIS2_PY}"
[ -x "$PY" ] || t2_die "3D 环境不存在：$PY（先跑 _setup_3d_venv.sh）"

SP="$("$PY" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
# DEST 下面要 rm -rf，SP 万一问出来是空就会变成 "/nvdiffrec_render" —— 先卡住
[ -n "$SP" ] && [ "$SP" != "/" ] || t2_die "问不出 site-packages 路径（python 环境坏了？）：$PY"
DEST="$SP/nvdiffrec_render"

if "$PY" -c "import nvdiffrec_render.renderutils, nvdiffrec_render.light" >/dev/null 2>&1; then
  t2_ok "nvdiffrec_render 已可用（PBR 渲染就绪）"
  exit 0
fi
if [ "$STATUS_ONLY" = "1" ]; then
  echo "  nvdiffrec_render 不可用：PBR 材质渲染阶段会跳过，其余功能正常"
  exit 1
fi

# ── 取源码：raw 逐文件（304KB，比 8.5MB 的 tarball 快得多）──────────
STAGE="${TMPDIR:-/tmp}/nvdr-stage"
BASE="https://raw.githubusercontent.com/NVlabs/nvdiffrec/main/render"
MIRROR="https://gh-proxy.com/https://raw.githubusercontent.com/NVlabs/nvdiffrec/main/render"
rm -rf "$STAGE"
mkdir -p "$STAGE/nvdiffrec_render/renderutils/c_src"
# 上游 render/ 没有 __init__.py，要自己补一个空的才能当包用
: > "$STAGE/nvdiffrec_render/__init__.py"

PYFILES="light.py material.py mesh.py mlptexture.py obj.py regularizer.py render.py texture.py util.py"
RUFILES="__init__.py bsdf.py loss.py ops.py"
CSRCFILES="bsdf.cu bsdf.h common.cpp common.h cubemap.cu cubemap.h loss.cu loss.h
           mesh.cu mesh.h normal.cu normal.h tensor.h torch_bindings.cpp vec3f.h vec4f.h"

fetch() {  # fetch <相对路径> <目标文件>
  curl -sf --max-time 60 --retry 2 -o "$2" "$BASE/$1" \
    || curl -sf --max-time 60 --retry 2 -o "$2" "$MIRROR/$1"
}

t2_log "拉取 nvdiffrec 源码"
MISS=""
for f in $PYFILES;    do fetch "$f" "$STAGE/nvdiffrec_render/$f"                    || MISS="$MISS $f"; done
for f in $RUFILES;    do fetch "renderutils/$f" "$STAGE/nvdiffrec_render/renderutils/$f" || MISS="$MISS renderutils/$f"; done
for f in $CSRCFILES;  do fetch "renderutils/c_src/$f" "$STAGE/nvdiffrec_render/renderutils/c_src/$f" || MISS="$MISS c_src/$f"; done
[ -z "$MISS" ] || t2_die "这些文件两个源都没取到：$MISS
   网络通了再重跑，或手工下 https://github.com/NVlabs/nvdiffrec"
t2_ok "源码 $(find "$STAGE" -type f | wc -l) 个文件，$(du -sh "$STAGE" | cut -f1)"

# ── 装进 site-packages ──────────────────────────────────────────────
t2_log "安装到 $DEST"
rm -rf "$DEST"
mv "$STAGE/nvdiffrec_render" "$DEST" || t2_die "复制到 site-packages 失败"

# ── 安装期就把 CUDA 扩展编好（运行期不再编，否则第一次渲染卡几分钟）──
t2_log "编译 renderutils 的 CUDA 扩展（首次约 3~8 分钟）"
CUDA_HOME_B="$(t2_detect_cuda_home 2>/dev/null || true)"
[ -n "$CUDA_HOME_B" ] || t2_die "没有 nvcc：先跑 _setup_3d_venv.sh（它会自动装 CUDA 编译器）"
export CUDA_HOME="$CUDA_HOME_B"
export PATH="$CUDA_HOME/bin:$PATH"
# 运行期 + 链接期都要给全，缺 LIBRARY_PATH 就是 08 号探针里那个 ninja 链接失败
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:/usr/lib/wsl/lib"
export LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib64/stubs:/usr/lib/wsl/lib"
export MAX_JOBS="${MAX_JOBS:-$(t2_build_jobs)}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$(t2_detect_arch 2>/dev/null || echo native)}"

if "$PY" - <<'PY'
import torch
print("  目标架构:", torch.utils.cpp_extension._get_cuda_arch_flags())
from nvdiffrec_render.renderutils import prepare_shading_normal
from nvdiffrec_render.light import EnvironmentLight
from nvdiffrec_render.mesh import Mesh
n = torch.full((1, 2, 2, 3), 0.577, device="cuda")
out = prepare_shading_normal(n, n, n, n, n, n, True, False)
print("  内核自检 OK:", tuple(out.shape), out.dtype)
PY
then
  t2_ok "nvdiffrec_render 就绪，PBR 材质渲染可用"
  t2_conf_set nvdiffrec_ready 1
  rm -rf "$STAGE"
  exit 0
else
  echo "  [FAIL] 编译或自检没通过。PBR 渲染阶段会被跳过，其余功能不受影响。"
  echo "         想手工排查：cd $DEST && python -c 'import nvdiffrec_render.renderutils'"
  echo "         报错里如果还是找不到 -lcuda / -lnvrtc，检查 LIBRARY_PATH 是否含 $CUDA_HOME/lib64/stubs"
  exit 1
fi
