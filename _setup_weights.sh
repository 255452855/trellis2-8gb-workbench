#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# 下载并校验「图生3D」所需的全部权重（本仓库不含权重，新机器从零装）
#
#   bash _setup_weights.sh              # 缺什么下什么，可重复跑
#   bash _setup_weights.sh --status     # 只检查，不写任何东西
#   bash _setup_weights.sh --only pixal3d,dinov3
#   bash _setup_weights.sh --source modelscope|huggingface
#   bash _setup_weights.sh --prune      # 转完 fp16 后删掉 20.5GB 的 fp32 原件
#   bash _setup_weights.sh --with-moge  # 顺带下 MoGe-2（可选功能）
#
# 为什么每一项都要单独处理（不是一个 snapshot_download 就完事）：
#   * facebook/dinov3 和 briaai/RMBG-2.0 在 Hugging Face 上是 **gated**
#     （要点同意 + 配 token 才能下）。一键包不能要求每个人都去申请，
#     所以上游项目 TRELLIS.2-stableprojectorz 打好的 release zip 优先。
#   * Pixal3D 官方只给 flow=fp32 那一版（整套 20.5GB），8GB 卡装不下
#     （实测单阶段 7619MiB/8192MiB 直接加载失败）。所以必须下完再本地转
#     成 fp16 —— 见 _convert_pixal3d_fp16.py，产物是 MODELS/Pixal3D-fp16。
#   * NAF 是 torch.hub 的 git 仓库 + release 里的 .pth，缓存落在
#     ~/.cache/torch。运行期是 HF_HUB_OFFLINE=1 的，所以必须在安装期
#     预热，否则第一次跑 Pixal3D 的 HR 阶段会去联网然后失败。
#
# 权重目录默认 $TRELLIS2_MODELS = engine/code/MODELS（见 _lib.sh 的三级解析）。
# ⚠️ 加载那一侧（app.py / pipeline_worker.py / 各 feature extractor）读的是
# **engine/code/MODELS 这个固定相对位置**（用 __file__ 和 os.getcwd() 拼的），
# 它并不认识 TRELLIS2_MODELS。所以想把权重放别的盘，只改本变量的话必须让
# engine/code/MODELS 变成一个指向它的链接 —— t2_ensure_models_link() 干这件事，
# 全程不动 engine 里的任何源文件。
# ─────────────────────────────────────────────────────────────────────
set -u
set -o pipefail

. "$(cd "$(dirname "$0")" && pwd)/_lib.sh"
t2_init

ONLY=""
PRUNE=0
WITH_MOGE=0
STATUS_ONLY=0
SOURCE="${TRELLIS2_MODEL_SOURCE:-auto}"

while [ $# -gt 0 ]; do
  case "$1" in
    --only)      shift; ONLY="${1:-}" ;;
    --source)    shift; SOURCE="${1:-auto}" ;;
    --status)    STATUS_ONLY=1 ;;
    --prune)     PRUNE=1 ;;
    --with-moge) WITH_MOGE=1 ;;
    -h|--help)   sed -n '2,28p' "$0"; exit 0 ;;
    *)           echo "未知参数：$1" >&2; exit 2 ;;
  esac
  shift
done

M="${TRELLIS2_MODELS:-$T2_CODE/MODELS}"
PY="${TRELLIS2_PY:-}"
WORK="${TMPDIR:-/tmp}/trellis2-weights"
mkdir -p "$WORK" "$M"

# ── 权重放在别处时，让 engine/code/MODELS 指向它 ─────────────────────
# 加载侧只认 engine/code/MODELS（app.py 用 __file__ 拼、feature extractor 用
# os.getcwd() 拼），所以本包唯一"碰"engine/code 的地方就是这一个符号链接，
# 不动任何源文件。链接逻辑在 _lib.sh 的 t2_ensure_models_link。

# 大陆网络下走代理访问国内 CDN 会慢 7 倍（_lib.sh 里有实测数字）；
# 同时权重下载必须允许联网 —— 运行期那个 HF_HUB_OFFLINE=1 是启动器设的。
t2_setup_no_proxy
unset HF_HUB_OFFLINE 2>/dev/null || true
export HF_HOME="$M"

http_code() { curl -s -o /dev/null -w '%{http_code}' --max-time 12 "$1" 2>/dev/null || echo 000; }
http_time() { curl -s -o /dev/null -w '%{time_total}' --max-time 15 "$1" 2>/dev/null || echo 99; }

# ── 下载源：实测挑一个，别照抄别人的网络环境 ─────────────────────────
_pick_source() {
  if [ "$SOURCE" != "auto" ]; then printf '%s' "$SOURCE"; return; fi
  local ms hf mst hft
  mst="$(http_time "https://www.modelscope.cn/api/v1/models/black-forest-labs/FLUX.2-klein-4B")"
  hft="$(http_time "https://huggingface.co/api/models/black-forest-labs/FLUX.2-klein-4B")"
  ms=0; hf=0
  [ "${mst%%.*}" -gt 0 ] 2>/dev/null && ms=1
  [ "${hft%%.*}" -gt 0 ] 2>/dev/null && hf=1
  if [ "$ms" = "1" ] && [ "$hf" = "1" ]; then
    awk -v a="$mst" -v b="$hft" 'BEGIN{exit !(a<b)}' && printf modelscope && return
    printf huggingface; return
  fi
  [ "$ms" = "1" ] && { printf modelscope; return; }
  [ "$hf" = "1" ] && { printf huggingface; return; }
  printf modelscope
}

# 需求清单：<名字>|<目标子目录>|<repo id>|<必需文件>|<约GB>|<说明>
ALL_ITEMS='trellis2|TRELLIS.2-4B|microsoft/TRELLIS.2-4B|pipeline.json;ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors;ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors|15|图生3D 主模型
image-large|TRELLIS-image-large|microsoft/TRELLIS-image-large|pipeline.json;ckpts/ss_dec_conv3d_16l8_fp16.safetensors|3.1|被 TRELLIS.2-4B/pipeline.json 用 ../ 引用
pixal3d|Pixal3D-F32Flow-F16Decoder|TencentARC/Pixal3D|pipeline.json;ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors|20.5|8GB 卡还要本地转成 fp16
dinov3|dinov3||config.json;model.safetensors|1.2|图像条件特征（gated，走 zip）
rmbg|RMBG-2.0||config.json;model.safetensors|0.9|抠图（gated + 非商用，走 zip）
naf|NAF||hubconf.py|0.1|torch.hub 仓库 + release 权重'
[ "$WITH_MOGE" = "1" ] && ALL_ITEMS="$ALL_ITEMS
moge|MoGe-2|Ruicheng/moge-2-vitl||1.3|可选：单目几何"

item_field() { printf '%s' "$1" | awk -F'|' -v n="$2" 'NF>=n{print $n}'; }

want() {
  [ -z "$ONLY" ] && return 0
  case ",$ONLY," in *",$1,"*) return 0 ;; esac
  return 1
}

missing_files() {  # missing_files <dir> <a;b;c> —— 支持 a/*.ext 这种通配
  local d="$1" spec="$2" f miss="" oldifs="$IFS"
  IFS=';'
  for f in $spec; do
    [ -n "$f" ] || continue
    if printf '%s' "$f" | grep -q '\*'; then
      ls "$d"/$f >/dev/null 2>&1 || miss="$miss ${f##*/}"
    else
      [ -e "$d/$f" ] || miss="$miss $f"
    fi
  done
  IFS="$oldifs"
  printf '%s' "$miss"
}

# ── 先报状态，再决定下什么 ──────────────────────────────────────────
t2_log "权重清单  $M"
NEED_GB=0
TODO=""
while IFS= read -r line; do
  [ -n "$line" ] || continue
  name="$(item_field "$line" 1)"
  want "$name" || continue
  dir="$M/$(item_field "$line" 2)"
  miss="$(missing_files "$dir" "$(item_field "$line" 4)")"
  if [ -z "$miss" ]; then
    printf '  %-12s %-6s 已就绪\n' "$name" "$(item_field "$line" 5)"
  else
    printf '  %-12s %-6s 缺:%s\n' "$name" "$(item_field "$line" 5)" "$miss"
    TODO="$TODO $name"
    NEED_GB="$(awk -v a="$NEED_GB" -v b="$(item_field "$line" 5)" 'BEGIN{printf "%.1f", a+b}')"
  fi
done <<ITEMS
$ALL_ITEMS
ITEMS

# 小显存卡必须有 fp16 变体才跑得动 Pixal3D
MEM_MIB="$(t2_gpu_mem_mib)"
CONVERT_NEEDED=0
if want pixal3d && [ -d "$M/Pixal3D-F32Flow-F16Decoder" ] && [ ! -d "$M/Pixal3D-fp16" ] \
   && [ -n "$MEM_MIB" ] && [ "$MEM_MIB" -lt 20000 ]; then
  CONVERT_NEEDED=1
  printf '  %-12s %-6s 需要本地转换（显存 %sMiB < 20GB）\n' "pixal3d-fp16" "13" "$MEM_MIB"
  TODO="$TODO pixal3d-fp16"
  NEED_GB="$(awk -v a="$NEED_GB" -v b="13" 'BEGIN{printf "%.1f", a+b}')"
fi

if [ "$STATUS_ONLY" = "1" ]; then
  echo
  [ -z "$TODO" ] && echo "  全部就绪。" || echo "  待处理：$TODO（约 $NEED_GB GB）"
  exit 0
fi

if [ -z "$TODO" ]; then
  t2_ok "所有权重都已就绪，无需下载"
  t2_conf_set weights_ready 1
  exit 0
fi

FREE="$(t2_free_gb "$M")"
NEED_INT="$(printf '%.0f' "$NEED_GB")"
echo
echo "  待处理：$TODO"
echo "  需要空间：约 ${NEED_GB} GB，目标盘当前可用 ${FREE:-?} GB"
if [ -n "$FREE" ] && [ "$NEED_INT" -gt "$((FREE - 20))" ]; then
  t2_die "磁盘不够：需要至少 $((NEED_INT + 20)) GB（留 20GB 给转换临时文件和缓存）。
     腾空间，或用 TRELLIS2_MODELS=<路径> 换一块盘再跑。"
fi

if [ ! -x "$PY" ]; then
  t2_die "还没建 3D 环境（$PY 不存在）。先跑：bash _setup_3d_venv.sh"
fi

SRC_CHOSEN="$(_pick_source)"
echo "  下载源：$SRC_CHOSEN（两个源都会自动兜底）"
t2_ensure_models_link "$M"

# ── 快照下载：目标目录直接写，断点续传 ──────────────────────────────
# 为什么不套临时目录再移动：/mnt/d 是 drvfs，20GB 的 mv 等于整盘复制，
# 十几分钟白费。snapshot_download 本身就是续传安全的。
_snapshot() {  # _snapshot <repo> <dest> <source> <log>
  local repo="$1" dest="$2" src="$3" log="$4"
  mkdir -p "$dest"
  if [ "$src" = "modelscope" ]; then
    SRC="$repo" DST="$dest" "$PY" - > "$log" 2>&1 <<'PY'
import os
from modelscope import snapshot_download
print(snapshot_download(os.environ["SRC"], local_dir=os.environ["DST"], max_workers=4))
PY
  else
    SRC="$repo" DST="$dest" "$PY" - > "$log" 2>&1 <<'PY'
import os
from huggingface_hub import snapshot_download
print(snapshot_download(os.environ["SRC"], local_dir=os.environ["DST"],
                        max_workers=4, resume_download=True))
PY
  fi
}

_ensure_client() {  # _ensure_client <source> —— 装缺的客户端，返回 0 表示可用
  if [ "$1" = "modelscope" ]; then
    "$PY" -c "import modelscope" 2>/dev/null && return 0
    t2_step "安装 modelscope"
    "$PY" -m pip install -q modelscope >/dev/null 2>&1 && return 0
    t2_warn "modelscope 装不上"
    return 1
  fi
  "$PY" -c "import huggingface_hub" 2>/dev/null && return 0
  t2_warn "这个环境里没有 huggingface_hub，用不了 HF 源"
  return 1
}

pull_snapshot() {  # pull_snapshot <子目录> <repo id>
  local sub="$1" repo="$2" other log
  if [ "$SRC_CHOSEN" = "modelscope" ]; then other=huggingface; else other=modelscope; fi
  for src in "$SRC_CHOSEN" "$other"; do
    _ensure_client "$src" || continue
    log="$WORK/$sub.$src.log"
    t2_step "$repo -> $M/$sub（源：$src）"
    if _snapshot "$repo" "$M/$sub" "$src" "$log"; then
      return 0
    fi
    echo "  [WARN] $src 失败，日志尾部："
    tail -8 "$log" | sed 's/^/      /'
    # HF 直连不通时试一次 hf-mirror（大陆网络常见）
    if [ "$src" = "huggingface" ] && [ -z "${HF_ENDPOINT:-}" ] \
       && [ "$(http_code https://hf-mirror.com/api)" = "200" ]; then
      export HF_ENDPOINT="https://hf-mirror.com"
      echo "  改用镜像 $HF_ENDPOINT 再试一次"
      log="$WORK/$sub.mirror.log"
      _snapshot "$repo" "$M/$sub" huggingface "$log" && return 0
      tail -8 "$log" | sed 's/^/      /'
      unset HF_ENDPOINT
    fi
  done
  return 1
}

FAILED=""

# ── 1. HF / 魔搭快照类 ─────────────────────────────────────────────
while IFS= read -r line; do
  name="$(item_field "$line" 1)"; repo="$(item_field "$line" 3)"
  [ -n "$name" ] && [ -n "$repo" ] || continue
  want "$name" || continue
  case " $TODO " in *" $name "*) : ;; *) continue ;; esac
  t2_log "下载 $name"
  if pull_snapshot "$(item_field "$line" 2)" "$repo"; then
    miss="$(missing_files "$M/$(item_field "$line" 2)" "$(item_field "$line" 4)")"
    if [ -z "$miss" ]; then t2_ok "$name 就绪"; else echo "  [FAIL] $name 下完了但仍缺：$miss"; FAILED="$FAILED $name"; fi
  else
    echo "  [FAIL] $name 两个源都没下成"
    FAILED="$FAILED $name"
  fi
done <<ITEMS
$ALL_ITEMS
ITEMS

# ── 2. gated 模型：上游 release zip（免 token）──────────────────────
zip_item() {  # zip_item <名字> <zip 文件名> <解压后要存在的文件>
  local name="$1" file="$2" check="$3"
  want "$name" || return 0
  case " $TODO " in *" $name "*) : ;; *) return 0 ;; esac
  [ -e "$M/$check" ] && return 0
  t2_log "下载 $name（上游 release zip，绕开 gated 授权）"
  local out="$WORK/$file"
  # -C - 续传 + 重试：这两个 zip 一个 1.2GB、一个 850MB
  curl -fL --retry 3 --retry-delay 5 -C - --max-time 7200 -o "$out" \
    "https://github.com/IgorAherne/TRELLIS.2-stableprojectorz/releases/download/extra-models/$file" \
    || curl -fL --retry 3 --retry-delay 5 -C - --max-time 7200 -o "$out" \
    "https://sourceforge.net/projects/trellis-2-stableprojectorz/files/extra-models/$file/download" \
    || { echo "  [FAIL] $name 两个镜像都下不动"; FAILED="$FAILED $name"; return; }
  ( cd "$M" && unzip -o -q "$out" ) || { echo "  [FAIL] $name 解压失败"; FAILED="$FAILED $name"; return; }
  if [ -e "$M/$check" ]; then t2_ok "$name 就绪"; rm -f "$out";
  else echo "  [FAIL] $name 解压后仍缺 $check"; FAILED="$FAILED $name"; fi
}
zip_item dinov3 dinov3.zip "dinov3/config.json"
zip_item rmbg  RMBG-2.0.zip "RMBG-2.0/model.safetensors"

# ── 3. NAF：git 仓库 + 预热 torch.hub 缓存 ─────────────────────────
if want naf && case " $TODO " in *" naf "*) true ;; *) false ;; esac; then
  t2_log "准备 NAF（Pixal3D HR 阶段的邻域注意力）"
  if [ ! -f "$M/NAF/hubconf.py" ]; then
    git clone --depth 1 https://github.com/valeoai/NAF.git "$M/NAF" \
      && t2_ok "NAF 源码就绪" \
      || { echo "  [FAIL] NAF clone"; FAILED="$FAILED naf"; }
  fi
  if [ -f "$M/NAF/hubconf.py" ]; then
    t2_step "预热 NAF 权重到 torch.hub 缓存（运行期离线，必须在安装期下好）"
    # 顺带验证了 natten 能真的把模型建起来
    if NAF_DIR="$M/NAF" "$PY" - <<'PY'
import os, torch
m = torch.hub.load(os.environ["NAF_DIR"], "naf", source="local",
                   pretrained=True, device="cpu", trust_repo=True)
print("  NAF OK，参数量：", sum(p.numel() for p in m.parameters()))
PY
    then t2_ok "NAF 权重已缓存"; else
      echo "  [WARN] NAF 预热失败：跑到 Pixal3D HR 阶段时才会再次尝试联网"
    fi
  fi
fi

# ── 4. Pixal3D fp16 转换（8GB 卡唯一跑得动的版本）───────────────────
if want pixal3d && [ ! -d "$M/Pixal3D-fp16" ] && [ -d "$M/Pixal3D-F32Flow-F16Decoder" ] \
   && [ -n "$MEM_MIB" ] && [ "$MEM_MIB" -lt 20000 ]; then
  t2_log "把 Pixal3D 的 DiT 转成 fp16（10~20 分钟，纯 CPU + 磁盘）"
  if "$PY" "$T2_ROOT/_convert_pixal3d_fp16.py" \
       --src "$M/Pixal3D-F32Flow-F16Decoder" --dst "$M/Pixal3D-fp16"; then
    t2_ok "Pixal3D-fp16 就绪"
    if [ "$PRUNE" = "1" ]; then
      echo "  --prune：删掉 20.5GB 的 fp32 原件"
      rm -rf "$M/Pixal3D-F32Flow-F16Decoder"
    else
      echo "  fp32 原件还在 $M/Pixal3D-F32Flow-F16Decoder（20.5GB）。"
      echo "  8GB 卡用不到它，想省空间就加 --prune 重跑一次。"
    fi
  else
    echo "  [FAIL] fp16 转换失败"; FAILED="$FAILED pixal3d-fp16"
  fi
fi

# ── 汇总 ────────────────────────────────────────────────────────────
t2_log "复核"
STILL=""
while IFS= read -r line; do
  name="$(item_field "$line" 1)"
  [ -n "$name" ] && want "$name" || continue
  miss="$(missing_files "$M/$(item_field "$line" 2)" "$(item_field "$line" 4)")"
  if [ -z "$miss" ]; then printf '  %-12s 就绪\n' "$name"
  else printf '  %-12s 仍缺:%s\n' "$name" "$miss"; STILL="$STILL $name"; fi
done <<ITEMS
$ALL_ITEMS
ITEMS
[ "$CONVERT_NEEDED" = "1" ] && [ ! -d "$M/Pixal3D-fp16" ] && STILL="$STILL pixal3d-fp16"

if [ -z "$STILL" ]; then
  echo
  echo "  ######## 权重全部就绪 ########"
  t2_conf_set weights_ready 1
  exit 0
fi
echo
echo "  还缺：$STILL"
echo "  单独补跑：bash _setup_weights.sh --only ${STILL# }"
echo "  也可以手动放：按上面的路径放进 $M，再 bash _setup_weights.sh --status 校验。"
exit 1
