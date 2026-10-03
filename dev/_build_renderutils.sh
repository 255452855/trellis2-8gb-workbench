#!/bin/bash
# 拉取 nvdiffrec 源码，检查 renderutils 的编译方式
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export TORCH_CUDA_ARCH_LIST=6.1
export MAX_JOBS=4

LOG=/tmp/renderutils_build.log
exec > "$LOG" 2>&1

echo "=== 1. 下载 nvdiffrec 源码 ==="
cd /tmp
rm -rf nvdiffrec-main nvdiffrec.tar.gz
OK=0
for URL in \
  "https://github.com/NVlabs/nvdiffrec/archive/refs/heads/main.tar.gz" \
  "https://ghproxy.net/https://github.com/NVlabs/nvdiffrec/archive/refs/heads/main.tar.gz" ; do
    echo "--- 尝试 $URL"
    curl -L --connect-timeout 20 --max-time 300 -o nvdiffrec.tar.gz "$URL" 2>&1 | tail -1
    if tar -tzf nvdiffrec.tar.gz >/dev/null 2>&1; then echo "  下载 OK"; OK=1; break; fi
done
if [ "$OK" != "1" ]; then echo "!!! 下载失败"; exit 1; fi
tar -xzf nvdiffrec.tar.gz

echo ""
echo "=== 2. 目录结构 ==="
ls /tmp/nvdiffrec-main/ 2>/dev/null | head
echo "--- render/ ---"
ls /tmp/nvdiffrec-main/render/ 2>/dev/null | head
echo "--- render/renderutils/ ---"
ls /tmp/nvdiffrec-main/render/renderutils/ 2>/dev/null | head
echo "--- csrc/ ---"
ls /tmp/nvdiffrec-main/render/renderutils/csrc/ 2>/dev/null | head -20
echo "--- setup 文件 ---"
find /tmp/nvdiffrec-main/render -maxdepth 3 \( -name "setup.py" -o -name "*.toml" -o -name "*.cfg" \) 2>/dev/null | head
