# File: app.py
# app.py
import os
import json
os.environ.setdefault(
    'PYTORCH_CUDA_ALLOC_CONF',
    'expandable_segments:True,garbage_collection_threshold:0.65',
)
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
if os.name == 'nt':
    # Windows 嵌入式 Python：distutils 在标准库 zip 里，强制 stdlib 以免 _distutils_hack 断言失败
    os.environ.setdefault('SETUPTOOLS_USE_DISTUTILS', 'stdlib')
# 非 Windows（WSL/Linux + Python 3.12）：stdlib 已移除 distutils，
# 必须保持默认的 local，让 setuptools 用自带的 distutils，否则 import distutils 会失败

import sys
import traceback
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from datetime import datetime
import shutil
from typing import *
import numpy as np
from PIL import Image

import base64
import io
import atexit, signal
import ctypes
import threading
import subprocess
import time

from pipeline_worker import PipelineWorker
from trellis2.modules.sparse import SparseTensor
import torch

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
MODES = [
    {"name": "法线", "icon": "assets/app/normal.png", "render_key": "normal"},
    {"name": "黏土", "icon": "assets/app/clay.png", "render_key": "clay"},
    {"name": "基础色", "icon": "assets/app/basecolor.png", "render_key": "base_color"},
    {"name": "森林", "icon": "assets/app/hdri_forest.png", "render_key": "shaded_forest"},
    {"name": "日落", "icon": "assets/app/hdri_sunset.png", "render_key": "shaded_sunset"},
    {"name": "庭院", "icon": "assets/app/hdri_courtyard.png", "render_key": "shaded_courtyard"},
]
STEPS = 8
DEFAULT_MODE = 3
DEFAULT_STEP = 3

# ── 运行时状态（供浏览器界面展示）───────────────────────────
RUNTIME = {"worker": None, "ready": False, "error": None}
RUNTIME_LOCK = threading.Lock()
SERVER_PORT = 8080


css = """
/* Overwrite Gradio Default Style */
.stepper-wrapper {
    padding: 0;
}

.stepper-container {
    padding: 0;
    align-items: center;
}

.step-button {
    flex-direction: row;
}

.step-connector {
    transform: none;
}

.step-number {
    width: 16px;
    height: 16px;
}

.step-label {
    position: relative;
    bottom: 0;
}

.wrap.center.full {
    inset: 0;
    height: 100%;
}

.wrap.center.full.translucent {
    background: var(--block-background-fill);
}

.meta-text-center {
    display: block !important;
    position: absolute !important;
    top: unset !important;
    bottom: 0 !important;
    right: 0 !important;
    transform: unset !important;
}

/* Previewer */
.previewer-container {
    position: relative;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    width: 100%;
    height: clamp(420px, 64vh, 680px);
    min-height: 420px;
    margin: 0 auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

.previewer-container .tips-icon {
    position: absolute;
    right: 10px;
    top: 10px;
    z-index: 10;
    border-radius: 10px;
    color: #fff;
    background-color: var(--color-accent);
    padding: 3px 6px;
    user-select: none;
}

.previewer-container .tips-text {
    position: absolute;
    right: 10px;
    top: 50px;
    color: #fff;
    background-color: var(--color-accent);
    border-radius: 10px;
    padding: 6px;
    text-align: left;
    max-width: 300px;
    z-index: 10;
    transition: all 0.3s;
    opacity: 0%;
    user-select: none;
}

.previewer-container .tips-text p {
    font-size: 14px;
    line-height: 1.2;
}

.tips-icon:hover + .tips-text { 
    display: block;
    opacity: 100%;
}

/* Row 1: Display Modes */
.previewer-container .mode-row {
    width: 100%;
    display: flex;
    gap: 8px;
    justify-content: center;
    margin-bottom: 20px;
    flex-wrap: wrap;
}
.previewer-container .mode-btn {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    cursor: pointer;
    opacity: 0.5;
    transition: all 0.2s;
    border: 2px solid #ddd;
    object-fit: cover;
}
.previewer-container .mode-btn:hover { opacity: 0.9; transform: scale(1.1); }
.previewer-container .mode-btn.active {
    opacity: 1;
    border-color: var(--color-accent);
    transform: scale(1.1);
}

/* Row 2: Display Image */
.previewer-container .display-row {
    margin-bottom: 20px;
    min-height: 0;
    width: 100%;
    flex-grow: 1;
    display: flex;
    justify-content: center;
    align-items: center;
}
.previewer-container .previewer-main-image {
    max-width: 100%;
    max-height: 100%;
    flex-grow: 1;
    object-fit: contain;
    display: none;
}
.previewer-container .previewer-main-image.visible {
    display: block;
}

/* Row 3: Custom HTML Slider */
.previewer-container .slider-row {
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    padding: 0 10px;
}

.previewer-container input[type=range] {
    -webkit-appearance: none;
    width: 100%;
    max-width: 400px;
    background: transparent;
}
.previewer-container input[type=range]::-webkit-slider-runnable-track {
    width: 100%;
    height: 8px;
    cursor: pointer;
    background: #ddd;
    border-radius: 5px;
}
.previewer-container input[type=range]::-webkit-slider-thumb {
    height: 20px;
    width: 20px;
    border-radius: 50%;
    background: var(--color-accent);
    cursor: pointer;
    -webkit-appearance: none;
    margin-top: -6px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    transition: transform 0.1s;
}
.previewer-container input[type=range]::-webkit-slider-thumb:hover {
    transform: scale(1.2);
}

/* Overwrite Previewer Block Style */
.gradio-container .padded:has(.previewer-container) {
    padding: 0 !important;
}

.gradio-container:has(.previewer-container) [data-testid="block-label"] {
    position: absolute;
    top: 0;
    left: 0;
}

/* Compact generation workspace */
.gradio-container {
    max-width: 1480px !important;
    margin: 0 auto !important;
}
#pix-header {
    padding: 16px 4px 12px;
    margin-bottom: 12px;
    border-bottom: 1px solid var(--border-color-primary);
}
#pix-header h1 {
    margin: 0 !important;
    font-size: clamp(1.25rem, 2vw, 1.65rem) !important;
    font-weight: 650 !important;
}
#pix-header p {
    margin: 5px 0 0 !important;
    color: var(--body-text-color-subdued);
    font-size: 0.9rem;
}
#pix-left-panel, #pix-right-panel {
    padding: 10px !important;
    min-width: 0 !important;
}
#pix-left-panel {
    border-right: 1px solid var(--border-color-primary);
}
#pix-left-panel .gradio-row {
    gap: 8px !important;
}
#pix-left-panel .gradio-row > * {
    min-width: 0 !important;
}
#pix-multiview {
    overflow: hidden;
}
#pix-generate-btn {
    font-weight: 600 !important;
    min-height: 46px;
}
#pix-generate-btn:disabled {
    background: var(--button-secondary-background-fill) !important;
    color: var(--body-text-color-subdued) !important;
    cursor: not-allowed !important;
}
#pix-extract-btn {
    font-weight: 600 !important;
}
#pix-extract-btn:disabled {
    background: var(--button-secondary-background-fill) !important;
    color: var(--body-text-color-subdued) !important;
}
#pix-status {
    padding: 8px 10px;
    margin: 6px 0;
    font-size: 0.86rem;
    line-height: 1.5;
    border-left: 4px solid;
}
#pix-status.pix-idle {
    background: color-mix(in srgb, var(--color-accent) 9%, transparent);
    border-left-color: var(--color-accent);
    color: var(--body-text-color);
}
#pix-status.pix-running {
    background: color-mix(in srgb, #d99000 10%, transparent);
    border-left-color: #d99000;
    color: var(--body-text-color);
}
#pix-status.pix-done {
    background: color-mix(in srgb, #16845b 10%, transparent);
    border-left-color: #16845b;
    color: var(--body-text-color);
}
#pix-status.pix-error {
    background: color-mix(in srgb, #c43b35 10%, transparent);
    border-left-color: #c43b35;
    color: var(--body-text-color);
}
@keyframes pix-pulse {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.55; }
}
.pix-dot {
    display: inline-block;
    width: 9px; height: 9px;
    border-radius: 50%;
    margin-right: 7px;
    vertical-align: middle;
    animation: pix-pulse 1.3s ease-in-out infinite;
}
.pix-dot.idle    { background: var(--color-accent); animation: none; }
.pix-dot.running { background: #d99000; }
.pix-dot.done    { background: #16845b; animation: none; }
.pix-dot.error   { background: #c43b35; animation: none; }
#pix-runtime {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 8px;
}
#pix-runtime .pix-rt-item {
    flex: 1 1 170px;
    background: var(--block-background-fill);
    border: 1px solid var(--border-color-primary);
    border-radius: 4px;
    padding: 7px 10px;
    display: flex;
    flex-direction: column;
    gap: 3px;
}
#pix-runtime .pix-rt-label {
    font-size: 0.72rem;
    opacity: 0.6;
}
#pix-runtime .pix-rt-value {
    font-size: 0.94rem;
    font-weight: 600;
    word-break: break-all;
}
.pix-ok   { color: #10b981; }
.pix-wait { color: #f59e0b; }
.pix-err  { color: #ef4444; }
/* 引擎状态这一格要比其他格更抢眼 */
#pix-runtime .pix-rt-state {
    font-size: 1.06rem;
    letter-spacing: .02em;
}
/* 显存占用后面跟一条迷你占用条 */
#pix-runtime .pix-rt-bar {
    display: inline-block;
    width: 54px;
    height: 6px;
    margin-left: 8px;
    border-radius: 3px;
    background: var(--border-color-primary);
    overflow: hidden;
    vertical-align: middle;
}
#pix-runtime .pix-rt-bar > i {
    display: block;
    height: 100%;
    background: currentColor;
    opacity: .85;
}
/* 权重徽章。放在 .pix-rt-value.pix-* 之后，
   保证单个徽章的颜色不会被整格的红色压掉。 */
#pix-runtime .pix-ckpt {
    display: inline-block;
    margin-right: 10px;
    white-space: nowrap;
}
#pix-runtime .pix-ckpt.pix-ok  { color: #10b981; }
#pix-runtime .pix-ckpt.pix-err { color: #ef4444; }

/* --- 说明性小字（比 Gradio 默认 info 更好扫读） --- */
.pix-hint {
    font-size: 0.82rem;
    line-height: 1.75;
    color: var(--body-text-color-subdued);
    margin: -4px 0 6px;
}
.pix-hint b { color: var(--body-text-color); font-weight: 600; }
.pix-hint-warn { color: #f59e0b; font-weight: 600; }

/* --- 预览区空态 --- */
.previewer-container .pix-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    height: 100%;
    min-height: 320px;
    text-align: center;
    padding: 0 24px;
}
.previewer-container .pix-empty svg {
    width: 44px;
    height: 44px;
    opacity: .35;
}
.pix-empty-title {
    margin: 0;
    font-size: 1rem;
    font-weight: 600;
    opacity: .8;
}
.pix-empty-tip {
    margin: 0;
    max-width: 430px;
    font-size: 0.85rem;
    line-height: 1.7;
    opacity: .55;
}
#pix-runtime .pix-rt-note {
    flex-basis: 100%;
    font-size: 0.86rem;
    line-height: 1.6;
    padding: 8px 10px;
    background: color-mix(in srgb, #c43b35 10%, transparent);
    border-left: 4px solid #c43b35;
    word-break: break-all;
}

.gradio-container .label-wrap {
    font-weight: 600 !important;
    font-size: 0.92rem !important;
}

.gradio-container input[type="number"] {
    font-variant-numeric: tabular-nums;
}
#pix-progress {
    margin-top: 12px;
    padding: 10px 12px;
    border-radius: 4px;
    background: color-mix(in srgb, var(--color-accent) 7%, transparent);
    border: 1px solid color-mix(in srgb, var(--color-accent) 22%, transparent);
}
.pix-pg-head {
    display: flex;
    align-items: center;
    gap: 9px;
    margin-bottom: 10px;
}
.pix-pg-dot {
    width: 9px; height: 9px;
    border-radius: 50%;
    background: #6366f1;
    animation: pix-pulse 1.2s ease-in-out infinite;
    flex-shrink: 0;
}
.pix-pg-stage {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--body-text-color);
    flex: 1;
}
.pix-pg-pct {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--color-accent);
    font-variant-numeric: tabular-nums;
}
.pix-pg-track {
    height: 8px;
    border-radius: 5px;
    background: var(--border-color-primary);
    overflow: hidden;
}
.pix-pg-fill {
    height: 100%;
    border-radius: 5px;
    background: var(--color-accent);
    transition: width 0.5s ease;
}
.pix-pg-tip {
    margin-top: 9px;
    font-size: 0.82rem;
    color: var(--body-text-color-subdued);
}

/* --- 小屏适配 --- */
@media (max-width: 1100px) {
    #pix-left-panel { border-right: 0; border-bottom: 1px solid var(--border-color-primary); }
    .previewer-container { height: clamp(380px, 60vh, 560px); min-height: 380px; }
    #pix-left-panel .gradio-row {
        flex-wrap: wrap !important;
    }
    #pix-left-panel .gradio-row > * {
        flex: 1 1 180px !important;
    }
}
@media (max-width: 640px) {
    .previewer-container {
        height: 440px;
        min-height: 440px;
        padding: 10px;
    }
    .previewer-container .display-row {
        min-height: 250px;
    }
    .previewer-container .mode-row {
        gap: 10px;
        margin-bottom: 12px;
    }
    .previewer-container .mode-btn {
        width: 28px;
        height: 28px;
    }
    #pix-runtime .pix-rt-item {
        flex-basis: 140px;
    }
}
"""


# 预览交互脚本（js_on_load 注入，每次 HTML 渲染后执行）。
#
# 为什么不靠内联 onclick：Gradio 6 的 gr.Blocks 没有 head 参数，
# 而 innerHTML 插入的内容里内联事件处理器作用域不可靠。
# 这里直接用 addEventListener 绑定，完全不碰内联属性，最稳。
# element 是 js_on_load 自动传入的容器元素。
preview_js = """
(function () {
    var el = element;
    if (!el) return;

    function refreshView(mode, step) {
        var allImgs = el.querySelectorAll('.previewer-main-image');
        if (!allImgs.length) return;

        // 没指定就用当前显示的那张
        if (mode === -1 || step === -1) {
            for (var i = 0; i < allImgs.length; i++) {
                if (allImgs[i].classList.contains('visible')) {
                    var parts = allImgs[i].id.split('-');   // ['view','m3','s3']
                    if (mode === -1) mode = parseInt(parts[1].slice(1), 10);
                    if (step === -1) step = parseInt(parts[2].slice(1), 10);
                    break;
                }
            }
        }
        if (isNaN(mode)) mode = 0;
        if (isNaN(step)) step = 0;

        allImgs.forEach(function (img) { img.classList.remove('visible'); });

        var target = el.querySelector('#view-m' + mode + '-s' + step);
        if (target) target.classList.add('visible');

        el.querySelectorAll('.mode-btn').forEach(function (btn, idx) {
            if (idx === mode) btn.classList.add('active');
            else btn.classList.remove('active');
        });
    }

    // 绑定模式按钮
    el.querySelectorAll('.mode-btn').forEach(function (btn, idx) {
        btn.addEventListener('click', function (e) {
            e.preventDefault();
            refreshView(idx, -1);
        });
    });

    // 绑定视角滑块
    var slider = el.querySelector('#custom-slider');
    if (slider) {
        slider.addEventListener('input', function () {
            refreshView(-1, parseInt(slider.value, 10));
        });
    }

    // 兜底：也挂到 window，万一还有内联写法能用
    window.refreshView = refreshView;
    window.selectMode = function (m) { refreshView(m, -1); };
    window.onSliderChange = function (v) { refreshView(-1, parseInt(v, 10)); };
})();
"""


empty_html = """
<div class="previewer-container">
  <div class="pix-empty">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
         stroke="currentColor" stroke-width="1.4" stroke-linecap="round"
         stroke-linejoin="round" style="color: var(--body-text-color);">
      <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
      <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
      <line x1="12" y1="22.08" x2="12" y2="12"></line>
    </svg>
    <p class="pix-empty-title">生成结果会显示在这里</p>
    <p class="pix-empty-tip">
      左侧上传一张主体清晰的图片 → 点「开始生成」<br>
      完成后可在这里旋转查看，并导出带贴图的 GLB
    </p>
  </div>
</div>
"""


def image_to_base64(image):
    buffered = io.BytesIO()
    image = image.convert("RGB")
    image.save(buffered, format="jpeg", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)
    
    
def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    # ignore_errors：某些环境（含安全 shim）会提前清掉该目录，
    # 原版不带这个参数会抛 FileNotFoundError，把整个 predict 请求带崩。
    shutil.rmtree(user_dir, ignore_errors=True)


def preprocess_image(image: Image.Image) -> Image.Image:
    return _get_worker().preprocess(image)


def _save_reference_images(images: list, session_hash: str) -> list:
    """Persist only the extra views so the worker can open them one at a time."""
    if not images:
        return []
    if len(images) > 3:
        raise ValueError(
            f"Pixal3D 最多接受 3 张补充视角（加主体图共 4 张），收到 {len(images)} 张。"
            "请删除多余图片后重试，未处理任何补充图。"
        )
    user_dir = os.path.join(TMP_DIR, str(session_hash), "multiview")
    os.makedirs(user_dir, exist_ok=True)
    paths = []
    for index, item in enumerate(images, start=1):
        image = item[0] if isinstance(item, (tuple, list)) else item
        if image is None:
            raise ValueError(f"第 {index} 张补充视角为空，未处理任何补充图。")
        path = os.path.join(user_dir, f"reference_{index}.png")
        image.convert("RGB").save(path, format="PNG")
        paths.append(path)
    return paths


def pack_state(latents: Tuple[SparseTensor, SparseTensor, int]) -> dict:
    shape_slat, tex_slat, res = latents
    return {
        'shape_slat_feats': shape_slat.feats.cpu().numpy(),
        'tex_slat_feats': tex_slat.feats.cpu().numpy(),
        'coords': shape_slat.coords.cpu().numpy(),
        'res': res,
    }
    
    
def unpack_state(state: dict) -> Tuple[SparseTensor, SparseTensor, int]:
    shape_slat = SparseTensor(
        feats=torch.from_numpy(state['shape_slat_feats']).cuda(),
        coords=torch.from_numpy(state['coords']).cuda(),
    )
    tex_slat = shape_slat.replace(torch.from_numpy(state['tex_slat_feats']).cuda())
    return shape_slat, tex_slat, state['res']


def get_seed(randomize_seed: bool, seed: int) -> int:
    """
    Get the random seed.
    """
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


def image_to_3d(
    image: Image.Image,
    reference_images: list,
    seed: int,
    model_choice: str,
    resolution: str,
    ss_guidance_strength: float,
    ss_guidance_rescale: float,
    ss_sampling_steps: int,
    ss_rescale_t: float,
    shape_slat_guidance_strength: float,
    shape_slat_guidance_rescale: float,
    shape_slat_sampling_steps: int,
    shape_slat_rescale_t: float,
    tex_slat_guidance_strength: float,
    tex_slat_guidance_rescale: float,
    tex_slat_sampling_steps: int,
    tex_slat_rescale_t: float,
    enable_python_profiling: bool,
    enable_torch_profiling: bool,
    enable_sync_hunter: bool,
    profiler_delay_sec: float,
    profiler_max_duration_sec: float,
    profiler_max_events: int,
    auto_degrade: bool,
    req: gr.Request,
    progress=gr.Progress(),
) -> str:
    import time as _time
    _t0 = _time.time()
    try:
        if image is None:
            raise ValueError("请先上传一张主体参考图。")

        def keepalive():
            progress(0.5, desc="正在生成…（GPU 工作中）")

        # 模型名映射：UI 显示名 -> worker 的 key
        _model_key = {"TRELLIS.2": "trellis2", "Pixal3D": "pixal3d"}.get(
            model_choice, "trellis2"
        )

        # Pixal3D 的 proj 模式只支持 cascade（1024_cascade / 1536_cascade），
        # 传 '512' 或 '1024' 会直接 raise ValueError；而且它没有
        # tex_slat_flow_model_512。1536 虽然支持但 8GB 显存跑不动。
        # 所以选 Pixal3D 时统一固定到「1024 高精」。
        if _model_key == "pixal3d" and resolution != "1024 高精":
            print(f"[INFO] Pixal3D 只走 cascade 链路且 1536 显存不足，"
                  f"「{resolution}」→ 改用「1024 高精」", flush=True)
            resolution = "1024 高精"

        _w = _get_worker()
        # 切换模型（若与当前不同，worker 会重新加载权重）
        try:
            if getattr(_w, "_ui_model", None) != _model_key:
                _w.switch_model(_model_key)
                _w._ui_model = _model_key
        except Exception as _e:
            raise RuntimeError(f"模型切换失败，已停止本次生成: {_e}") from _e

        reference_paths = (
            _save_reference_images(reference_images or [], req.session_hash)
            if _model_key == "pixal3d"
            else []
        )
        state, images = _w.generate(
            image=image,
            reference_images=reference_paths,
            seed=seed,
            model=_model_key,
            ss_params={
                "steps": ss_sampling_steps,
                "guidance_strength": ss_guidance_strength,
                "guidance_rescale": ss_guidance_rescale,
                "rescale_t": ss_rescale_t,
            },
            shape_params={
                "steps": shape_slat_sampling_steps,
                "guidance_strength": shape_slat_guidance_strength,
                "guidance_rescale": shape_slat_guidance_rescale,
                "rescale_t": shape_slat_rescale_t,
            },
            tex_params={
                "steps": tex_slat_sampling_steps,
                "guidance_strength": tex_slat_guidance_strength,
                "guidance_rescale": tex_slat_guidance_rescale,
                "rescale_t": tex_slat_rescale_t,
            },
            pipeline_type={
                "512": "512",
                "1024 快速": "1024",
                "1024 高精": "1024_cascade",
                "1536": "1536_cascade",
            }[resolution],
            # Pixal3D 相机估计用的图像分辨率（TRELLIS.2 忽略）
            image_resolution={
                "512": 512,
                "1024 快速": 1024,
                "1024 高精": 1024,
                "1536": 1536,
            }[resolution],
            nviews=STEPS,
            profiling={
                "enable_python": enable_python_profiling,
                "enable_torch": enable_torch_profiling,
                "enable_sync_hunter": enable_sync_hunter,
                "delay_sec": profiler_delay_sec,
                "max_duration_sec": profiler_max_duration_sec,
                "max_events": profiler_max_events,
            },
            auto_degrade=auto_degrade,
            progress_callback=keepalive,
        )
    except Exception as e:
        _elapsed = _time.time() - _t0
        print(f"[ERROR] Generation failed after {_elapsed:.0f}s: {type(e).__name__}: {e}")
        traceback.print_exc()
        _error_text = str(e)
        if "内存保护拒绝" in _error_text or "显存保护拒绝" in _error_text:
            _status_text = (
                f"资源不足，已停止本次生成；当前模型已保留。"
                f"详细原因请查看右侧错误信息（耗时 {_elapsed:.0f} 秒）。"
            )
        else:
            _status_text = f"生成失败（耗时 {_elapsed:.0f} 秒）—— {type(e).__name__}"
        return (
            {},
            _error_html("生成失败", f"{type(e).__name__}: {e}"),
            _status_html("error", _status_text),
        )

    # --- HTML Construction ---
    # The Stack of 48 Images
    images_html = ""
    for m_idx, mode in enumerate(MODES):
        for s_idx in range(STEPS):
            # ID Naming Convention: view-m{mode}-s{step}
            unique_id = f"view-m{m_idx}-s{s_idx}"
            
            # Logic: Only Mode 0, Step 0 is visible initially
            is_visible = (m_idx == DEFAULT_MODE and s_idx == DEFAULT_STEP)
            vis_class = "visible" if is_visible else ""
            
            # Image Source
            img_base64 = image_to_base64(Image.fromarray(images[mode['render_key']][s_idx]))
            
            # Render the Tag
            images_html += f"""
                <img id="{unique_id}" 
                     class="previewer-main-image {vis_class}" 
                     src="{img_base64}" 
                     loading="eager">
            """
    
    # Button Row HTML
    btns_html = ""
    for idx, mode in enumerate(MODES):        
        active_class = "active" if idx == DEFAULT_MODE else ""
        # Note: onclick calls the JS function defined in Head
        btns_html += f"""
            <img src="{mode['icon_base64']}" 
                 class="mode-btn {active_class}" 
                 onclick="selectMode({idx})"
                 title="{mode['name']}">
        """
    
    # Assemble the full component
    full_html = f"""
    <div class="previewer-container">
        <div class="tips-wrapper">
            <div class="tips-icon">💡Tips</div>
            <div class="tips-text">
                <p>● <b>Render Mode</b> - Click on the circular buttons to switch between different render modes.</p>
                <p>● <b>View Angle</b> - Drag the slider to change the view angle.</p>
            </div>
        </div>
        
        <!-- Row 1: Viewport containing 48 static <img> tags -->
        <div class="display-row">
            {images_html}
        </div>
        
        <!-- Row 2 -->
        <div class="mode-row" id="btn-group">
            {btns_html}
        </div>

        <!-- Row 3: Slider -->
        <div class="slider-row">
            <input type="range" id="custom-slider" min="0" max="{STEPS - 1}" value="{DEFAULT_STEP}" step="1" oninput="onSliderChange(this.value)">
        </div>
    </div>
    """
    return state, full_html, _status_html(
        "done",
        f"生成完成！耗时 {_time.time() - _t0:.0f} 秒。"
        f"满意就点「导出 GLB」下载模型；不满意可换张图或调整随机种子重新生成。"
    )


def extract_glb(
    state: dict,
    decimation_target: int,
    texture_size: int,
    req: gr.Request,
) -> Tuple[str, str]:
    """
    Extract a GLB file from the 3D model.

    Args:
        state (dict): The state of the generated 3D model.
        decimation_target (int): The target face count for decimation.
        texture_size (int): The texture resolution.

    Returns:
        str: The path to the extracted GLB file.
    """
    if not state or 'shape_slat_feats' not in state:
        raise gr.Error(
            "还没有可导出的模型。请先点「开始生成」并等待生成完成，再点「导出 GLB」。"
        )
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%dT%H%M%S") + f".{now.microsecond // 1000:03d}"
    os.makedirs(user_dir, exist_ok=True)
    glb_path = os.path.join(user_dir, f'sample_{timestamp}.glb')
    worker = _get_worker()
    _get_worker().extract_glb(state, decimation_target, texture_size, glb_path)
    return glb_path, glb_path



# nvidia-smi 的已知绝对路径。
# 不能只靠 PATH 查找：WSL 后端启动时会把 PATH 整个覆盖掉（env PATH=...），
# 而 WSL 里的 nvidia-smi 是驱动挂载在 /usr/lib/wsl/lib 下的，不在标准 PATH 里，
# 结果顶部状态面板会显示「未检测到 NVIDIA 显卡 / 显存占用：读取失败」。
_NVIDIA_SMI_CANDIDATES = (
    "/usr/lib/wsl/lib/nvidia-smi",            # WSL2 驱动挂载点（最常见）
    "/usr/local/cuda/bin/nvidia-smi",         # CUDA toolkit
    "/usr/local/cuda-12.8/bin/nvidia-smi",
    "/usr/local/cuda-12.4/bin/nvidia-smi",
    "/usr/bin/nvidia-smi",
    "/usr/sbin/nvidia-smi",
    "/opt/nvidia/bin/nvidia-smi",
    r"C:\Windows\System32\nvidia-smi.exe",    # Windows 原生
    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
)


def _find_nvidia_smi():
    """先查 PATH，再查已知绝对路径；都没有返回 None。"""
    found = shutil.which("nvidia-smi")
    if found:
        return found
    for path in _NVIDIA_SMI_CANDIDATES:
        try:
            if os.path.isfile(path):
                return path
        except OSError:
            continue
    return None


def _gpu_snapshot_via_smi():
    """走 nvidia-smi 读显存。"""
    exe = _find_nvidia_smi()
    if not exe:
        return None
    try:
        out = subprocess.check_output(
            [exe, "--query-gpu=name,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=5, text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).strip().splitlines()[0]
        name, used, total = [x.strip() for x in out.split(",")]
        return {"name": name, "used": float(used), "total": float(total)}
    except Exception:
        return None


# ── 兜底：直接调 NVML 的 C 接口 ─────────────────────────────
# 万一 WSL 里压根没有 nvidia-smi（有的发行版确实没有，只有驱动挂载的 so），
# 就直接 ctypes 调 libnvidia-ml。NVML 只读驱动状态，**不初始化 CUDA 上下文**，
# 所以不会像 torch.cuda 那样抢占显存（8GB 卡上这点很关键）。
_NVML_LIB_NAMES = (
    "libnvidia-ml.so.1",
    "libnvidia-ml.so",
    "/usr/lib/wsl/lib/libnvidia-ml.so.1",   # WSL2 驱动挂载点
    "nvml.dll",                              # Windows
)
_NVML_LIB_CACHE = {"tried": False, "lib": None}


class _NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


def _load_nvml():
    if not _NVML_LIB_CACHE["tried"]:
        _NVML_LIB_CACHE["tried"] = True
        for name in _NVML_LIB_NAMES:
            try:
                _NVML_LIB_CACHE["lib"] = ctypes.CDLL(name)
                break
            except OSError:
                continue
    return _NVML_LIB_CACHE["lib"]


def _gpu_snapshot_via_nvml():
    lib = _load_nvml()
    if lib is None:
        return None
    try:
        init = getattr(lib, "nvmlInit_v2", None) or getattr(lib, "nvmlInit", None)
        if init is None or init() != 0:
            return None

        handle = ctypes.c_void_p()
        get_handle = (getattr(lib, "nvmlDeviceGetHandleByIndex_v2", None)
                      or getattr(lib, "nvmlDeviceGetHandleByIndex", None))
        if get_handle is None or get_handle(0, ctypes.byref(handle)) != 0:
            return None

        name = "NVIDIA GPU"
        try:
            buf = ctypes.create_string_buffer(96)
            if lib.nvmlDeviceGetName(handle, buf, 96) == 0 and buf.value:
                name = buf.value.decode("utf-8", "ignore").strip()
        except Exception:
            pass

        mem = _NvmlMemory()
        if lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem)) != 0:
            return None
        if mem.total <= 0:
            return None
        return {
            "name": name,
            "used": mem.used / (1024 * 1024),
            "total": mem.total / (1024 * 1024),
        }
    except Exception:
        return None


_GPU_SNAP_CACHE = {"at": 0.0, "value": None}


def _gpu_snapshot():
    """读显卡型号与显存占用。

    顺序：**先 NVML，后 nvidia-smi**。
    顶部「运行状态」面板由 gr.Timer(2.0) 每 2 秒刷一次，而 nvidia-smi 是
    起一个子进程去查询驱动——WSL2 上一次调用动辄几百毫秒，还会和正在跑的
    CUDA kernel 抢设备锁。生成过程中每 2 秒来一次，会明显拖慢推理。
    NVML 是同进程 ctypes 调 libnvidia-ml，几乎零成本，所以改成优先走它。

    **不要**在这里调 torch.cuda —— 那会在主进程建 CUDA 上下文白占几百 MB 显存。

    另外加了 1.5 秒的缓存：面板同时被多处调用时不会重复查询。
    """
    import time as _time

    now = _time.time()
    cached = _GPU_SNAP_CACHE["value"]
    if cached is not None and now - _GPU_SNAP_CACHE["at"] < 1.5:
        return cached
    snap = _gpu_snapshot_via_nvml() or _gpu_snapshot_via_smi()
    _GPU_SNAP_CACHE["at"] = now
    _GPU_SNAP_CACHE["value"] = snap
    return snap


def _mb(v: float) -> str:
    return f"{v / 1024:.1f} GB" if v >= 1024 else f"{v:.0f} MB"


def _read_progress():
    """读取 worker 写的生成进度。

    用 running 标志判断任务是否进行中——**不要靠时间戳新鲜度**，
    因为阶段之间（比如材质采样结束 → 渲染开始）可能有几十秒空档，
    按新鲜度判断会让进度条中途闪没。
    """
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "_progress.json")
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return None
    if not d.get("running"):
        return None
    # 兜底：worker 崩了就没人写 running=False，用较长超时兜住（10 分钟）
    if time.time() - d.get("ts", 0) > 600:
        return None
    return d


# 阶段顺序与权重（用于把"当前阶段"换算成总进度）
# 注意顺序要和真实执行顺序一致：稀疏结构 → 形状(LR) → 形状(HR) → 材质 → 解码 → 渲染
_STAGE_ORDER = ["preprocess", "sparse", "shape", "shape_hr", "texture", "decode", "render"]
_STAGE_LABEL = {
    "sparse": "稀疏结构",
    "shape": "形状",
    "shape_lr": "形状（低分辨率）",
    "shape_hr": "形状（高分辨率）",
    "texture": "材质",
    "decode": "解码网格",
    "render": "渲染预览",
    "done": "完成",
}


def _progress_html() -> str:
    """生成中的进度条。没有进行中的任务就返回空串。"""
    d = _read_progress()
    if not d:
        return ""

    key = d.get("key", "")
    label = _STAGE_LABEL.get(key, d.get("stage", "处理中"))
    step, total, pct = d.get("step", 0), d.get("total", 0), d.get("pct", -1)

    # 总进度：已完成的阶段占大头，当前阶段内部按步数细算
    try:
        idx = _STAGE_ORDER.index(key)
    except ValueError:
        idx = 0
    n = len(_STAGE_ORDER)
    if pct >= 0:
        overall = int((idx + pct / 100.0) / n * 100)
        detail = f"{label}　第 {step}/{total} 步"
    else:
        overall = int(idx / n * 100)
        detail = label

    overall = max(1, min(99, overall))

    return (
        f'<div id="pix-progress">'
        f'  <div class="pix-pg-head">'
        f'    <span class="pix-pg-dot"></span>'
        f'    <span class="pix-pg-stage">{detail}</span>'
        f'    <span class="pix-pg-pct">{overall}%</span>'
        f'  </div>'
        f'  <div class="pix-pg-track"><div class="pix-pg-fill" style="width:{overall}%"></div></div>'
        f'  <div class="pix-pg-tip">生成期间请勿关闭页面或窗口，GPU 满载、风扇变响均属正常。</div>'
        f'</div>'
    )


def _required_ckpts(model_dir):
    """按 pipeline.json 实际引用的权重，返回 (已就绪, 需要)。

    ⚠️ 不要硬编码数字。权重目录会因为换量化版本、删冗余文件而变，
    硬编码会让面板误报成「8/9」这种红字 —— 本机删掉 tex_enc 后就踩过。
    """
    import json as _json

    cfg_path = None
    for name in ("pipeline_mv.json", "pipeline.json"):
        candidate = os.path.join(model_dir, name)
        if os.path.isfile(candidate):
            cfg_path = candidate
            break
    if cfg_path is None:
        return 0, 0

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            models = _json.load(f).get("args", {}).get("models", {}) or {}
    except Exception:
        return 0, 0

    have = need = 0
    for rel in models.values():
        if not isinstance(rel, str):
            continue
        need += 1
        # 允许 "../TRELLIS-image-large/ckpts/..." 这种跨目录引用
        path = os.path.normpath(os.path.join(model_dir, rel))
        if os.path.isfile(path + ".safetensors") or os.path.isdir(path):
            have += 1
    return have, need


def _pixal3d_model_dir():
    """当前实际会用的 Pixal3D 权重目录（与 pipeline_worker 的优先级保持一致）。"""
    root = os.path.dirname(os.path.abspath(__file__))
    override = os.environ.get("PIXAL3D_MODEL_DIR")
    if override and os.path.isdir(override):
        return override
    for name in ("Pixal3D-fp16", "Pixal3D-F32Flow-F16Decoder", "Pixal3D-fp32"):
        candidate = os.path.join(root, "MODELS", name)
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(root, "MODELS", "Pixal3D-fp16")


def _runtime_panel_html() -> str:
    """界面顶部「运行状态」面板：引擎 / 显卡 / 显存 / 权重 / 地址。"""
    gpu = _gpu_snapshot()
    with RUNTIME_LOCK:
        ready, err = RUNTIME["ready"], RUNTIME["error"]

    if err:
        state_cls, state_txt = "pix-err", "启动失败"
    elif ready:
        state_cls, state_txt = "pix-ok", "就绪"
    else:
        state_cls, state_txt = "pix-wait", "加载中…"

    if gpu:
        used, total = gpu["used"], gpu["total"]
        free = total - used
        vram_cls = "pix-err" if free < 3000 else "pix-ok"
        pct = min(100.0, (used / total * 100.0) if total else 0.0)
        vram = (
            f"{_mb(used)} / {_mb(total)}"
            f'<span class="pix-rt-bar"><i style="width:{pct:.0f}%"></i></span>'
        )
        gpu_name = gpu["name"]
    else:
        vram_cls, vram, gpu_name = "pix-err", "读取失败", "未检测到 NVIDIA 显卡"

    root = os.path.dirname(os.path.abspath(__file__))
    t2_have, t2_need = _required_ckpts(os.path.join(root, "MODELS", "TRELLIS.2-4B"))
    pixal_model_dir = _pixal3d_model_dir()
    px_have, px_need = _required_ckpts(pixal_model_dir)

    def chip(label, have, need, title=""):
        ok = need > 0 and have >= need
        mark = "✓" if ok else f"{have}/{need}"
        cls = "pix-ok" if ok else "pix-err"
        attr = f' title="{title}"' if title else ""
        return f'<span class="pix-ckpt {cls}"{attr}>{label} {mark}</span>'

    weights = chip("TRELLIS.2", t2_have, t2_need) + chip(
        "Pixal3D", px_have, px_need,
        title=os.path.basename(os.path.normpath(pixal_model_dir)),
    )
    weights_cls = "pix-ok" if (
        t2_need > 0 and t2_have >= t2_need and px_need > 0 and px_have >= px_need
    ) else "pix-err"

    def item(label, value, cls=""):
        return (
            f'<div class="pix-rt-item"><span class="pix-rt-label">{label}</span>'
            f'<span class="pix-rt-value {cls}">{value}</span></div>'
        )

    html = '<div id="pix-runtime">'
    html += item("引擎状态", state_txt, f"{state_cls} pix-rt-state")
    html += item("显卡", gpu_name)
    html += item("显存占用", vram, vram_cls)
    html += item("模型权重", weights, weights_cls)
    html += item("服务地址", f"127.0.0.1:{SERVER_PORT}")
    if err:
        html += f'<div class="pix-rt-note">引擎启动失败：{err}<br>请查看启动窗口里的完整报错。</div>'
    elif not ready:
        html += (
            '<div class="pix-rt-note" style="background:rgba(245,158,11,.1);'
            'border-left-color:#f59e0b;">模型加载中（约 3~4 分钟），'
            '加载完成前无法生成，请稍候。</div>'
        )
    html += "</div>"
    html += _progress_html()
    return html


def _init_worker_async():
    """后台线程加载引擎，让网页能立刻打开并显示进度。"""
    t0 = time.time()
    try:
        w = PipelineWorker()
        with RUNTIME_LOCK:
            RUNTIME["worker"] = w
            RUNTIME["ready"] = True
        print(f"[INFO] 引擎就绪，加载用时 {time.time() - t0:.0f} 秒。", flush=True)
    except Exception as e:
        with RUNTIME_LOCK:
            RUNTIME["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()


def _get_worker() -> PipelineWorker:
    with RUNTIME_LOCK:
        w, err = RUNTIME["worker"], RUNTIME["error"]
    if w is not None:
        return w
    if err:
        raise gr.Error(f"引擎启动失败，无法生成：{err}")
    raise gr.Error("模型还在加载中（约 3~4 分钟），请稍候再试。")


def _download_models(selected_models, source_choice):
    """Download only the models explicitly selected by the user."""
    source = {
        "按系统语言选择": "auto",
        "ModelScope（中国源）": "modelscope",
        "Hugging Face": "huggingface",
    }.get(source_choice, "auto")
    model_keys = []
    for label, key in (("TRELLIS.2", "trellis2"), ("Pixal3D", "pixal3d")):
        if label in (selected_models or []):
            model_keys.append(key)
    if not model_keys:
        return '<div id="pix-status" class="pix-error"><span class="pix-dot error"></span>请至少选择一个模型。</div>'
    try:
        from install import download_selected_models
        message = download_selected_models(model_keys, source=source)
        detail = message.replace("\n", "<br>")
        return (
            '<div id="pix-status" class="pix-done">'
            '<span class="pix-dot done"></span>模型下载完成。<br>'
            f'{detail}</div>'
        )
    except Exception as exc:
        print(f"[ERROR] Model download failed: {type(exc).__name__}: {exc}", flush=True)
        return (
            '<div id="pix-status" class="pix-error">'
            '<span class="pix-dot error"></span>'
            f'模型下载失败：{type(exc).__name__}: {exc}</div>'
        )


def _status_html(kind: str, text: str) -> str:
    """构造状态提示条。kind: idle / running / done / error"""
    return (
        f'<div id="pix-status" class="pix-{kind}">'
        f'<span class="pix-dot {kind}"></span>{text}'
        f'</div>'
    )


def _error_html(title: str, detail: str = "") -> str:
    """生成失败时显示的提示（替代原来的空白占位）。"""
    d = f'<div style="margin-top:8px;font-size:0.85rem;opacity:0.8;word-break:break-all;">{detail}</div>' if detail else ''
    return f"""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                height:100%;min-height:420px;text-align:center;padding:32px;gap:6px;">
        <div style="font-size:2.6rem;line-height:1;">⚠️</div>
        <div style="font-size:1.15rem;font-weight:600;color:#ef4444;">{title}</div>
        <div style="font-size:0.9rem;opacity:0.75;max-width:560px;">
            常见原因：显存不足（请关闭游戏 / Blender / Photoshop 等）、
            分辨率选得过高、或图片主体不清晰。<br>
            可查看服务端日志获取详细错误信息。
        </div>
        {d}
    </div>
    """


def _on_generate_click(resolution, model_choice):
    """点击「开始生成」的即时反馈；引擎没就绪时不误报「生成中」。"""
    with RUNTIME_LOCK:
        ready, err = RUNTIME["ready"], RUNTIME["error"]
    if err:
        return (
            gr.update(interactive=True),
            _status_html("error", f"引擎启动失败，无法生成：{err}"),
        )
    if not ready:
        return (
            gr.update(interactive=True),
            _status_html("running", "模型仍在加载中（约 3~4 分钟），加载完成后才能生成，请稍候。"),
        )

    # 时间预期要跟着当前选择走 —— 原来一律写「约 5 分钟（512 分辨率）」，
    # 选 1024 或 Pixal3D 时完全是误导。
    if model_choice == "Pixal3D":
        eta = "约 40 分钟以上（Pixal3D 固定走 1024 级联，8GB 卡上偏慢）"
    elif resolution == "512":
        eta = "约 5 分钟"
    elif resolution == "1024 快速":
        eta = "约 15~25 分钟"
    else:
        eta = "约 1 小时以上（级联 512→1024）"

    return (
        gr.update(value="生成中… 请勿关闭页面", interactive=False),
        _status_html(
            "running",
            f"正在生成 3D 模型，预计 {eta}。"
            "GPU 满载、风扇变响都属正常，请耐心等待，不要刷新页面。",
        ),
    )


def create_input_panel():
    with gr.Column(scale=4, min_width=320, elem_id="pix-left-panel"):
        # ── 1. 上传图片 ──
        # 这是整个流程的第一步，必须放最上面。
        # 之前这里顶着一个折叠的「模型管理」（下载权重），用户第一眼找不到上传口。
        gr.Markdown("### 1. 上传图片")
        image_prompt = gr.Image(
            label="参考图片",
            format="png", image_mode="RGBA", type="pil", height=330,
            sources=["upload", "clipboard"],
        )
        reference_images = gr.Gallery(
            label="Pixal3D 多视角补充（按左、右、后顺序）",
            type="pil",
            columns=3,
            rows=1,
            height=140,
            file_types=["image"],
            allow_preview=True,
            visible=False,
            elem_id="pix-multiview",
        )

        gr.Markdown("### 2. 生成设置")
        model_choice = gr.Radio(
            ["TRELLIS.2", "Pixal3D"],
            label="生成模型",
            value="TRELLIS.2",
            info=(
                "TRELLIS.2 = 通用图生3D，细节丰富但可能脑补原图没有的东西；"
                "Pixal3D = 像素对齐，正面与输入图更忠实（SIGGRAPH 2026）。"
                "切换模型需要重新加载权重，首次约 1-2 分钟"
            ),
        )
        model_choice.change(
            lambda value: gr.update(visible=value == "Pixal3D"),
            inputs=[model_choice],
            outputs=[reference_images],
        )
        resolution = gr.Radio(
            ["512", "1024 快速", "1024 高精", "1536"],
            label="生成分辨率",
            value="512",
        )
        # 原来这一大段塞在 Gradio 的 info 里，挤成一坨没法扫读，拆成结构化小字
        gr.Markdown(
            '<div class="pix-hint">'
            '<b>512</b> ≈ 5 分钟 · 8GB 显存推荐　|　'
            '<b>1024 快速</b> 单阶段 1024，十几分钟<br>'
            '<b>1024 高精</b> / <b>1536</b> 走级联 512→1024，老显卡 2 小时以上<br>'
            '<span class="pix-hint-warn">Pixal3D 只支持级联模式</span>，'
            '选 512 / 1024 快速也会按 1024 跑，比 TRELLIS.2 慢很多。'
            '</div>'
        )
        seed = gr.Slider(0, MAX_SEED, label="随机种子", value=0, step=1)
        randomize_seed = gr.Checkbox(label="每次随机种子", value=True)
        auto_degrade = gr.Checkbox(
            label="显存不足时自动降级（推荐）",
            value=True,
            info="勾选：装不下就自动降分辨率 / NAF 尺寸，保证一定跑得完。"
                 "取消：按原设置硬跑，装不下时会占用系统内存 —— 不会失败，"
                 "但会慢很多，也可能把电脑拖卡。",
        )

        generate_btn = gr.Button("开始生成", elem_id="pix-generate-btn", variant="primary")

        status_box = gr.HTML(
            '<div id="pix-status" class="pix-idle">'
            '<span class="pix-dot idle"></span>'
            '准备就绪 —— 上传图片后点击「开始生成」'
            '</div>'
        )

        # ── 3. 导出设置 ──
        # 这两个参数只在导出 GLB 时生效，原本混在「生成设置」里容易误以为会影响生成
        with gr.Accordion(label="导出设置（生成完成后生效）", open=False):
            decimation_target = gr.Slider(
                10000, 1000000, label="模型面数上限", value=250000, step=10000,
                info="数值越小文件越小",
            )
            texture_size = gr.Slider(
                1024, 4096, label="贴图尺寸", value=2048, step=1024,
            )

        with gr.Accordion(label="高级设置：采样参数", open=False):
            gr.Markdown("**阶段 1：稀疏结构生成**")
            with gr.Row():
                ss_guidance_strength = gr.Slider(1.0, 10.0, label="引导强度", value=7.5, step=0.1)
                ss_guidance_rescale = gr.Slider(0.0, 1.0, label="引导重缩放", value=0.7, step=0.01)
                ss_sampling_steps = gr.Slider(1, 50, label="采样步数", value=14, step=1)
                ss_rescale_t = gr.Slider(1.0, 6.0, label="重缩放 T", value=5.0, step=0.1)
            gr.Markdown("**阶段 2：形状生成**")
            with gr.Row():
                shape_slat_guidance_strength = gr.Slider(1.0, 10.0, label="引导强度", value=7.5, step=0.1)
                shape_slat_guidance_rescale = gr.Slider(0.0, 1.0, label="引导重缩放", value=0.5, step=0.01)
                shape_slat_sampling_steps = gr.Slider(1, 50, label="采样步数", value=14, step=1)
                shape_slat_rescale_t = gr.Slider(1.0, 6.0, label="重缩放 T", value=3.0, step=0.1)
            gr.Markdown("**阶段 3：材质生成**")
            with gr.Row():
                tex_slat_guidance_strength = gr.Slider(1.0, 10.0, label="引导强度", value=1.0, step=0.1)
                tex_slat_guidance_rescale = gr.Slider(0.0, 1.0, label="引导重缩放", value=0.0, step=0.01)
                tex_slat_sampling_steps = gr.Slider(1, 50, label="采样步数", value=14, step=1)
                tex_slat_rescale_t = gr.Slider(1.0, 6.0, label="重缩放 T", value=3.0, step=0.1)

        # ── 模型下载 ──
        # 只在权重缺失时才需要，挪到最后，并把原来含糊的「模型管理」改清楚
        with gr.Accordion(label="模型下载（权重缺失时才需要）", open=False):
            gr.Markdown(
                '<div class="pix-hint">'
                '顶部「模型权重」变红时才需要点这里补齐。'
                '</div>'
            )
            model_download_selection = gr.CheckboxGroup(
                ["TRELLIS.2", "Pixal3D"],
                label="选择要下载的模型",
                value=[],
            )
            model_download_source = gr.Radio(
                ["按系统语言选择", "ModelScope（中国源）", "Hugging Face"],
                label="下载源",
                value="按系统语言选择",
            )
            model_download_btn = gr.Button("下载模型（检查缺失）", variant="secondary")
            model_download_status = gr.HTML(
                '<div id="pix-status" class="pix-idle">'
                '<span class="pix-dot idle"></span>未开始下载</div>'
            )

        with gr.Accordion(label="调试与性能分析", open=False):
            gr.Markdown("性能分析数据与同步日志会保存在 `./tmp/profiling`。")
            with gr.Row():
                enable_python_profiling = gr.Checkbox(label="启用 Python 性能分析", value=False)
                enable_torch_profiling = gr.Checkbox(label="启用 PyTorch 性能分析", value=False)
                enable_sync_hunter = gr.Checkbox(label="启用 CUDA 同步检测", value=False)
            with gr.Row():
                profiler_delay_sec = gr.Slider(
                    label="开始延迟（秒）",
                    minimum=0, maximum=300, value=80, step=5,
                    info="等待 X 秒后开始记录。"
                )
                profiler_max_duration_sec = gr.Slider(
                    label="记录时长（秒）",
                    minimum=0, maximum=60, value=3, step=1,
                    info="记录 X 秒后停止（0 = 不限）。"
                )
            profiler_max_events = gr.Slider(
                label="最大事件数",
                minimum=0, maximum=5000, value=300, step=50,
                info="记录 N 个操作后停止（0 = 不限）。"
            )

    return {
        "image_prompt": image_prompt,
        "reference_images": reference_images,
        "model_choice": model_choice,
        "resolution": resolution,
        "auto_degrade": auto_degrade,
        "seed": seed,
        "randomize_seed": randomize_seed,
        "decimation_target": decimation_target,
        "texture_size": texture_size,
        "generate_btn": generate_btn,
        "status_box": status_box,
        "model_download_selection": model_download_selection,
        "model_download_source": model_download_source,
        "model_download_btn": model_download_btn,
        "model_download_status": model_download_status,
        "ss_guidance_strength": ss_guidance_strength,
        "ss_guidance_rescale": ss_guidance_rescale,
        "ss_sampling_steps": ss_sampling_steps,
        "ss_rescale_t": ss_rescale_t,
        "shape_slat_guidance_strength": shape_slat_guidance_strength,
        "shape_slat_guidance_rescale": shape_slat_guidance_rescale,
        "shape_slat_sampling_steps": shape_slat_sampling_steps,
        "shape_slat_rescale_t": shape_slat_rescale_t,
        "tex_slat_guidance_strength": tex_slat_guidance_strength,
        "tex_slat_guidance_rescale": tex_slat_guidance_rescale,
        "tex_slat_sampling_steps": tex_slat_sampling_steps,
        "tex_slat_rescale_t": tex_slat_rescale_t,
        "enable_python_profiling": enable_python_profiling,
        "enable_torch_profiling": enable_torch_profiling,
        "enable_sync_hunter": enable_sync_hunter,
        "profiler_delay_sec": profiler_delay_sec,
        "profiler_max_duration_sec": profiler_max_duration_sec,
        "profiler_max_events": profiler_max_events,
    }

def create_preview_panel():
    with gr.Column(scale=7, elem_id="pix-right-panel"):
        gr.Markdown("### 3. 预览与导出")
        with gr.Walkthrough(selected=0) as walkthrough:
            with gr.Step("预览", id=0):
                preview_output = gr.HTML(empty_html, label="3D 预览", show_label=True,
                                         container=True, js_on_load=preview_js)
                extract_btn = gr.Button("导出 GLB", elem_id="pix-extract-btn", interactive=False)
                gr.Markdown(
                    '<div class="pix-hint">'
                    '生成完成后这个按钮才会亮起；'
                    '点它会渲染网格，并按左侧「导出设置」里的面数上限 / 贴图尺寸导出 GLB。'
                    '</div>'
                )
            with gr.Step("导出", id=1):
                glb_output = gr.Model3D(label="导出的模型", height=724, show_label=True, display_mode="solid", clear_color=(0.25, 0.25, 0.25, 1.0))
                download_btn = gr.DownloadButton(label="下载 GLB 文件")

    return {
        "walkthrough": walkthrough,
        "preview_output": preview_output,
        "extract_btn": extract_btn,
        "glb_output": glb_output,
        "download_btn": download_btn
    }

def create_examples_panel(image_prompt):
    with gr.Column(scale=1, min_width=172, elem_id="pix-examples"):
        example_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "example_image")
        examples = gr.Examples(
            examples=[
                os.path.join(example_dir, image)
                for image in os.listdir(example_dir)
            ],
            inputs=[image_prompt],
            fn=preprocess_image,
            outputs=[image_prompt],
            run_on_click=True,
            examples_per_page=18,
        )
    return examples


with gr.Blocks(delete_cache=(600, 600), title="Pixal3D 本地工作台") as demo:
    gr.HTML("""
    <div id="pix-header">
        <h1>Pixal3D · 本地 3D 生成工作台</h1>
        <p>上传主体图，按需补充多视角，生成后可预览并导出带纹理的 GLB。</p>
    </div>
    """)

    runtime_panel = gr.HTML(_runtime_panel_html())
    runtime_timer = gr.Timer(2.0)

    with gr.Row(equal_height=False):
        inputs = create_input_panel()
        outputs = create_preview_panel()

    with gr.Accordion("示例图片", open=False):
        create_examples_panel(inputs["image_prompt"])
                    
    output_buf = gr.State()

    # Handlers
    demo.load(start_session)
    demo.unload(end_session)

    # 顶部「运行状态」面板。
    # ⚠️ runtime_panel 的初始 HTML 是**建页面时**就烤死的（那时 worker 还在加载），
    # 所以首屏必然显示「加载中…」。这里在会话建立后立刻再刷一次，
    # 不必干等 2 秒的定时器；定时器负责后续的持续刷新。
    demo.load(fn=_runtime_panel_html, outputs=[runtime_panel])
    runtime_timer.tick(fn=_runtime_panel_html, outputs=[runtime_panel])

    inputs["model_download_btn"].click(
        _download_models,
        inputs=[inputs["model_download_selection"], inputs["model_download_source"]],
        outputs=[inputs["model_download_status"]],
    )

    inputs["image_prompt"].upload(
        preprocess_image,
        inputs=[inputs["image_prompt"]],
        outputs=[inputs["image_prompt"]],
    )

    inputs["generate_btn"].click(
        # 1. 立刻反馈：禁用按钮 + 显示"进行中"，避免用户以为没反应
        _on_generate_click,
        inputs=[inputs["resolution"], inputs["model_choice"]],
        outputs=[inputs["generate_btn"], inputs["status_box"]],
    ).then(
        get_seed,
        inputs=[inputs["randomize_seed"], inputs["seed"]],
        outputs=[inputs["seed"]],
    ).then(
        lambda: gr.Walkthrough(selected=0), outputs=outputs["walkthrough"]
    ).then(
        lambda: gr.update(interactive=False), outputs=outputs["extract_btn"],
    ).then(
        image_to_3d,
        inputs=[
            inputs["image_prompt"], inputs["reference_images"],
            inputs["seed"], inputs["model_choice"], inputs["resolution"],
            inputs["ss_guidance_strength"], inputs["ss_guidance_rescale"], inputs["ss_sampling_steps"], inputs["ss_rescale_t"],
            inputs["shape_slat_guidance_strength"], inputs["shape_slat_guidance_rescale"], inputs["shape_slat_sampling_steps"], inputs["shape_slat_rescale_t"],
            inputs["tex_slat_guidance_strength"], inputs["tex_slat_guidance_rescale"], inputs["tex_slat_sampling_steps"], inputs["tex_slat_rescale_t"],
            inputs["enable_python_profiling"], inputs["enable_torch_profiling"], inputs["enable_sync_hunter"],
            inputs["profiler_delay_sec"], inputs["profiler_max_duration_sec"], inputs["profiler_max_events"],
            inputs["auto_degrade"],
        ],
        outputs=[output_buf, outputs["preview_output"], inputs["status_box"]],
    ).then(
        # 2. 完成后恢复按钮。
        # ⚠️ 失败时 image_to_3d 返回的是空 dict {}，此时**不能**把「导出 GLB」点亮，
        #    否则按钮看着可用，点下去只会弹「还没有生成结果」，很误导。
        lambda state: (
            gr.update(value="重新生成", interactive=True),
            gr.update(interactive=bool(state)),
        ),
        inputs=[output_buf],
        outputs=[inputs["generate_btn"], outputs["extract_btn"]],
    )
    
    outputs["extract_btn"].click(
        lambda: gr.Walkthrough(selected=1), outputs=outputs["walkthrough"]
    ).then(
        extract_glb,
        inputs=[output_buf, inputs["decimation_target"], inputs["texture_size"]],
        outputs=[outputs["glb_output"], outputs["download_btn"]],
    )
        

# Launch the Gradio app
if __name__ == "__main__":
    import argparse
    import socket

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-browser", action="store_true",
                        help="启动后不自动打开浏览器")
    args, unknown = parser.parse_known_args()

    os.makedirs(TMP_DIR, exist_ok=True)

    for i in range(len(MODES)):
        MODES[i]['icon_base64'] = image_to_base64(Image.open(MODES[i]['icon']))

    # ⚠️ connect_ex 必须设超时，否则启动会「假死」约 130 秒。
    # 本机 WSL 是 networkingMode=mirrored（Windows 与 WSL 共用 loopback），
    # 连一个「没人监听」的端口不会立刻回 RST，而是变成 SYN 黑洞 ——
    # 不设超时就走 Linux 默认的 tcp_syn_retries（约 130 秒）才放弃。
    # 叠加引擎加载 21 秒，总耗时会擦到 _启动WSL后端.ps1 的 3 分钟上限。
    # 另外 0.0.0.0 不是一个「可连接」的地址，探测时统一换成 127.0.0.1。
    _probe_host = "127.0.0.1" if args.host in ("0.0.0.0", "::", "") else args.host

    def _find_free_port(start, attempts=100):
        for p in range(start, start + attempts):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                try:
                    if s.connect_ex((_probe_host, p)) == 0:
                        continue
                except OSError:
                    pass
                return p
        raise RuntimeError(f"No free port found in range {start}-{start + attempts}")

    SERVER_PORT = _find_free_port(args.port)

    # 引擎放后台线程加载，网页先打开、状态显示在页面上
    threading.Thread(target=_init_worker_async, daemon=True).start()

    def _cleanup():
        with RUNTIME_LOCK:
            w = RUNTIME["worker"]
        if w is not None:
            w.shutdown()

    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, lambda *_: (atexit._run_exitfuncs(), exit(0)))
    signal.signal(signal.SIGTERM, lambda *_: (atexit._run_exitfuncs(), exit(0)))

    demo.launch(
        css=css,
        server_name=args.host,
        server_port=SERVER_PORT,
        inbrowser=not args.no_browser,
        prevent_thread_lock=True,
    )
    print(f"[INFO] 界面已就绪： http://{args.host}:{SERVER_PORT}", flush=True)
    demo.block_thread()
