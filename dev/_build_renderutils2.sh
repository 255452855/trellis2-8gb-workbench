#!/bin/bash
# 为 Linux 编译 nvdiffrec 的 renderutils._C，补上 nvdiffrec_render 缺的编译扩展
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export TORCH_CUDA_ARCH_LIST=6.1
export MAX_JOBS=4

LOG=/tmp/renderutils_compile.log
exec > "$LOG" 2>&1

RU=/tmp/nvdiffrec-main/render/renderutils
CSRC="$RU/c_src"

if [ ! -d "$CSRC" ]; then echo "!!! 源码不存在: $CSRC"; exit 1; fi

cd "$CSRC" || exit 1

cat > setup.py <<'PYEOF'
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='renderutils_C',
    version='0.0.0',
    ext_modules=[
        CUDAExtension(
            name='_C',
            sources=[
                'common.cpp',
                'torch_bindings.cpp',
                'bsdf.cu',
                'cubemap.cu',
                'loss.cu',
                'mesh.cu',
                'normal.cu',
            ],
            include_dirs=['.'],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
PYEOF

echo "=== 开始编译 renderutils._C（sm_61）==="
/home/ccb/trellis2-wsl-venv/bin/python setup.py build_ext --inplace 2>&1 | tail -45
echo "构建退出码=$?"

echo ""
echo "=== 产物 ==="
ls -la "$CSRC"/_C*.so 2>/dev/null || echo "  没有生成 .so"

echo ""
echo "=== 安装到 nvdiffrec_render/renderutils ==="
DST=/home/ccb/trellis2-wsl-venv/lib/python3.12/site-packages/nvdiffrec_render/renderutils
rm -f "$DST"/_C.cp311-win_amd64.pyd
for f in "$CSRC"/_C*.so; do
    [ -f "$f" ] && cp "$f" "$DST/" && echo "  已复制 $(basename "$f")"
done
ls -la "$DST"/ | head

echo ""
echo "=== 验证导入 ==="
/home/ccb/trellis2-wsl-venv/bin/python -c "import nvdiffrec_render.renderutils as ru; print('renderutils OK')" 2>&1 | tail -5
/home/ccb/trellis2-wsl-venv/bin/python -c "from nvdiffrec_render.light import EnvironmentLight; print('EnvironmentLight OK')" 2>&1 | tail -5
