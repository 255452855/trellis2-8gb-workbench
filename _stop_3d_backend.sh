#!/usr/bin/env bash
# 停掉 3D 后端，并**等显存真正释放**再返回。
#
# 为什么不能只 `pkill -f app.py`：
#   worker 是 multiprocessing 子进程，父进程被强杀后它会变成孤儿，
#   继续占着 2~3GB 显存。8GB 卡上这点残留足以让 FLUX（transformer 7.75GB）
#   直接 OOM —— 2026-10-02 实测就踩到了：残留 1252MiB 时文生图失败。
#   项目自带的「重启后端.bat」也是杀两次（app.py + venv python），这里照做并加等显存。
set -u

echo "[停后端] 杀 app.py"
if pkill -f 'ap[p].py' 2>/dev/null; then echo "  app.py 已停"; else echo "  app.py 本来就没跑"; fi
sleep 1

echo "[停后端] 杀孤儿 worker"
if pkill -f 'trellis2-wsl-venv/bin/python' 2>/dev/null; then echo "  worker 已停"; else echo "  无残留 worker"; fi

FLOOR=${AIR_FLOOR_MB:-1500}
echo "[停后端] 等显存降到 ${FLOOR}MiB 以下 ..."
for i in $(seq 1 20); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  if [ -z "$used" ]; then break; fi
  if [ "$used" -lt "$FLOOR" ]; then
    echo "[停后端] OK：used=${used}MiB（等了 $((i * 2)) 秒）"
    exit 0
  fi
  sleep 2
done

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
echo "[停后端] 警告：显存仍占 ${used:-?}MiB。文生图可能会 OOM，"
echo "          看一下是不是还有别的程序在用显卡（游戏/浏览器/训练）。"
exit 0
