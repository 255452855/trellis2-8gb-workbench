#!/bin/bash
# 在 WSL 内编译安装 CuMesh（cumesh）到 trellis2-wsl-venv
# 针对 GTX 1070 (sm_61)
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export TORCH_CUDA_ARCH_LIST=6.1
export MAX_JOBS=4

LOG=/tmp/cumesh_setup.log
exec > "$LOG" 2>&1

echo "=== 0. 环境 ==="
echo "CUDA_HOME=$CUDA_HOME"
echo "nvcc=$(command -v nvcc)"
echo "python=/home/ccb/trellis2-wsl-venv/bin/python"

echo ""
echo "=== 1. 克隆 CuMesh（HTTP/1.1 全量，避开 GnuTLS 中断）==="
rm -rf /tmp/CuMesh
git -c http.version=HTTP/1.1 -c http.postBuffer=524288000 clone --recursive \
    https://github.com/JeffreyXiang/CuMesh.git /tmp/CuMesh
echo "直连克隆退出码=$?"

if [ ! -d /tmp/CuMesh ]; then
    echo "直连失败，尝试镜像 gitclone.com ..."
    rm -rf /tmp/CuMesh
    git -c http.version=HTTP/1.1 clone --recursive \
        https://gitclone.com/github.com/JeffreyXiang/CuMesh.git /tmp/CuMesh
    echo "镜像克隆退出码=$?"
fi

if [ ! -d /tmp/CuMesh ]; then
    echo "!!! 克隆仍然失败，终止"
    exit 1
fi

echo ""
echo "=== 2. 仓库内容 ==="
ls -la /tmp/CuMesh

echo ""
echo "=== 3. 编译安装（sm_61）==="
/home/ccb/trellis2-wsl-venv/bin/pip install --no-build-isolation /tmp/CuMesh
echo "pip 退出码=$?"

echo ""
echo "=== 4. 验证 ==="
/home/ccb/trellis2-wsl-venv/bin/python -c "import cumesh; print('cumesh OK:', cumesh.__file__)"
echo "验证退出码=$?"
