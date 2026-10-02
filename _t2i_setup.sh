#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# 准备「文生图」独立环境（FLUX.2-klein-4B）
#
# 为什么单独建 venv：diffusers 0.40 要求 huggingface_hub>=1.23，而
# TRELLIS.2 的 3D 环境里 transformers 4.57 要求 huggingface_hub<1.0，
# 两者**不能共存**。所以文生图必须有自己的 venv，物理隔离。
#   3D  环境：/home/ccb/trellis2-wsl-venv   （不要动）
#   文生图：/home/ccb/flux-venv             （本脚本创建）
#
# 可重复运行：已完成的步骤自动跳过。
# 注意：文件名保持 ASCII —— 带中文的路径经 PowerShell 传给 wsl.exe 会被吃掉。
# ─────────────────────────────────────────────────────────────────────
ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="${FLUX_VENV:-/home/ccb/flux-venv}"
PY="$VENV/bin/python"
MODELS="${FLUX_MODEL_DIR:-$ROOT/engine/code/MODELS/FLUX.2-klein-4B}"
MS_MODEL=black-forest-labs/FLUX.2-klein-4B
IDX=https://pypi.tuna.tsinghua.edu.cn/simple
TORCH_PIN="torch==2.6.0 torchvision==0.21.0"

echo "############ [1/4] venv ############"
if [ ! -x "$PY" ]; then
  python3 -m venv "$VENV" || { echo "建 venv 失败"; exit 1; }
  echo "已创建 $VENV"
else
  echo "已存在 $VENV"
fi
"$PY" -m pip install -q --upgrade pip -i "$IDX" 2>&1 | tail -2

echo "############ [2/4] torch 2.6.0+cu124（Pascal 可用，与 3D 环境一致）############"
if ! "$PY" -c "import torch" 2>/dev/null; then
  "$PY" -m pip install -i "$IDX" $TORCH_PIN 2>&1 | tail -3
else
  echo "torch 已装：$("$PY" -c 'import torch;print(torch.__version__)')"
fi

echo "############ [3/4] diffusers / transformers / accelerate ############"
if ! "$PY" -c "from diffusers import Flux2KleinPipeline" 2>/dev/null; then
  echo "让 pip 自行解依赖（这个 venv 是隔离的，坏了不影响 3D 环境）"
  "$PY" -m pip install -i "$IDX" \
    diffusers transformers accelerate safetensors sentencepiece protobuf einops 2>&1 | tail -10
fi
"$PY" - <<'PYEOF'
import torch
print("torch        ", torch.__version__, "| cuda:", torch.cuda.is_available())
try:
    import diffusers
    print("diffusers    ", diffusers.__version__,
          "| Flux2KleinPipeline:", hasattr(diffusers, "Flux2KleinPipeline"))
except Exception as e:
    print("diffusers 失败:", type(e).__name__, str(e)[:200])
try:
    import transformers
    print("transformers ", transformers.__version__)
except Exception as e:
    print("transformers 失败:", type(e).__name__, str(e)[:160])
PYEOF

echo "############ [4/4] 权重 ############"
need=0
for f in transformer/diffusion_pytorch_model.safetensors \
         text_encoder/model-00001-of-00002.safetensors \
         text_encoder/model-00002-of-00002.safetensors \
         vae/diffusion_pytorch_model.safetensors \
         tokenizer/tokenizer.json model_index.json; do
  [ -f "$MODELS/$f" ] || { echo "缺少 $f"; need=1; }
done
if [ "$need" = "0" ]; then
  echo "权重已就绪，跳过下载"
else
  echo "从魔搭下载 $MS_MODEL（约 24GB，慢慢等）"
  "$PY" -m pip install -q -i "$IDX" modelscope 2>&1 | tail -2
  "$PY" - <<PYEOF
from modelscope import snapshot_download
print("下载完成:", snapshot_download("$MS_MODEL", local_dir="$MODELS"))
PYEOF
fi
echo "权重占用：$(du -sh "$MODELS" | cut -f1)"
echo "############ 准备完成 ############"
