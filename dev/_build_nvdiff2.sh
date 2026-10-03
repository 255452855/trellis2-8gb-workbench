#!/bin/bash
# 用 tarball 方式获取并编译 nvdiffrast（绕开 git 的 GnuTLS 卡死）
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export TORCH_CUDA_ARCH_LIST=6.1
export MAX_JOBS=4

LOG=/tmp/nvdiff2.log
exec > "$LOG" 2>&1

echo "=== 0. 杀掉卡住的 git ==="
pkill -f "nvdiffras[t].git"
sleep 2

echo ""
echo "=== 1. 下载 nvdiffrast 源码 ==="
cd /tmp
rm -rf nvdiffrast nvdiffrast.tar.gz nvdiffrast-main
OK=0
for URL in \
  "https://github.com/NVlabs/nvdiffrast/archive/refs/heads/main.tar.gz" \
  "https://ghproxy.net/https://github.com/NVlabs/nvdiffrast/archive/refs/heads/main.tar.gz" \
  "https://gh-proxy.com/https://github.com/NVlabs/nvdiffrast/archive/refs/heads/main.tar.gz" ; do
    echo "--- 尝试: $URL"
    if curl -L --connect-timeout 20 --max-time 240 -o /tmp/nvdiffrast.tar.gz "$URL" 2>&1 | tail -2; then
        if tar -tzf /tmp/nvdiffrast.tar.gz >/dev/null 2>&1; then
            echo "  下载成功: $(ls -la /tmp/nvdiffrast.tar.gz | awk '{print $5}') 字节"
            OK=1
            break
        fi
    fi
done

if [ "$OK" != "1" ]; then
    echo "!!! 所有源都下载失败"
    exit 1
fi

echo ""
echo "=== 2. 解压 ==="
tar -xzf /tmp/nvdiffrast.tar.gz
mv /tmp/nvdiffrast-main /tmp/nvdiffrast 2>/dev/null
ls /tmp/nvdiffrast | head -20

echo ""
echo "=== 3. 编译安装 nvdiffrast（sm_61）==="
/home/ccb/trellis2-wsl-venv/bin/pip install --no-build-isolation /tmp/nvdiffrast
echo "pip 退出码=$?"

echo ""
echo "=== 4. 验证 ==="
/home/ccb/trellis2-wsl-venv/bin/python -c "import nvdiffrast.torch; print('nvdiffrast OK')" 2>&1 | tail -4
/home/ccb/trellis2-wsl-venv/bin/python -c "import nvdiffrec_render; print('nvdiffrec_render OK')" 2>&1 | tail -4
