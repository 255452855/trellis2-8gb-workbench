#!/bin/bash
# 给 WSL venv 补 nvdiffrec_render（纯 Python，直接复制）和 nvdiffrast（Linux 需编译）
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export TORCH_CUDA_ARCH_LIST=6.1
export MAX_JOBS=4

LOG=/tmp/nvdiff_setup.log
exec > "$LOG" 2>&1

WINSP=/mnt/d/IDM/TRELLIS2/engine/code/venv/Lib/site-packages
WSLSP=/home/ccb/trellis2-wsl-venv/lib/python3.12/site-packages

echo "=== 1. 复制纯 Python 包 nvdiffrec_render ==="
rm -rf "$WSLSP/nvdiffrec_render" "$WSLSP/nvdiffrec_render-0.0.0.dist-info"
cp -r "$WINSP/nvdiffrec_render" "$WSLSP/" 2>&1
cp -r "$WINSP/nvdiffrec_render-0.0.0.dist-info" "$WSLSP/" 2>&1
find "$WSLSP/nvdiffrec_render" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
echo "复制结果:"; ls "$WSLSP/nvdiffrec_render"

echo ""
echo "=== 2. 克隆 nvdiffrast（HTTP/1.1）==="
rm -rf /tmp/nvdiffrast
git -c http.version=HTTP/1.1 -c http.postBuffer=524288000 clone --recursive \
    https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast
echo "克隆退出码=$?"

if [ ! -d /tmp/nvdiffrast ]; then
    echo "直连失败，尝试镜像 gitclone.com ..."
    rm -rf /tmp/nvdiffrast
    git -c http.version=HTTP/1.1 clone --recursive \
        https://gitclone.com/github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast
    echo "镜像退出码=$?"
fi

if [ ! -d /tmp/nvdiffrast ]; then
    echo "!!! nvdiffrast 克隆失败"
else
    echo ""
    echo "=== 3. 编译安装 nvdiffrast（sm_61）==="
    /home/ccb/trellis2-wsl-venv/bin/pip install --no-build-isolation /tmp/nvdiffrast
    echo "nvdiffrast pip 退出码=$?"
fi

echo ""
echo "=== 4. 验证 ==="
/home/ccb/trellis2-wsl-venv/bin/python -c "import nvdiffrast.torch; print('nvdiffrast OK')" 2>&1 | tail -3
/home/ccb/trellis2-wsl-venv/bin/python -c "import nvdiffrec_render; print('nvdiffrec_render OK')" 2>&1 | tail -3
