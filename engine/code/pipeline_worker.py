# pipeline_worker.py
# Runs the TRELLIS.2 pipeline in a separate process to avoid GIL contention with Gradio.
import multiprocessing as mp
import traceback
import os
import sys


# from tools.profiling_wrapper import AuraProfiler
# from tools.sync_hunter import hunt_syncs

# ── Profiler / sync-hunter stubs (not shipped) ──────────────
class AuraProfiler:
    def __init__(self, **kwargs):
        pass
    def start(self):
        pass
    def step(self):
        pass
    def stop_and_save(self, name):
        pass

from contextlib import contextmanager

@contextmanager
def hunt_syncs(**kwargs):
    yield
# ─────────────────────────────────────────────────────────────


def _apply_patches():
    """Apply Windows compatibility patches for flex_gemm."""
    import torch

    try:
        import flex_gemm.ops.spconv as spconv
        from flex_gemm.ops.spconv import Algorithm
        # Force the algorithm to EXPLICIT_GEMM
        # This bypasses the 'kernels.triton' error by using standard Torch Matrix Multiplication
        spconv.ALGORITHM = Algorithm.EXPLICIT_GEMM
        print("[WORKER] flex_gemm EXPLICIT_GEMM patch applied.")
    except ImportError:
        print("[WORKER] Could not patch flex_gemm spconv.")
    except Exception as e:
        print(f"[WORKER] flex_gemm spconv patch failed: {e}")

    try:
        import flex_gemm.kernels as _fgk
        if not hasattr(_fgk, 'triton'):
            class _TritonFallback:
                @staticmethod
                def indice_weighed_sum_fwd(feats, indices, weights):
                    N = feats.shape[0]
                    idx = indices.long().clamp(min=0, max=N - 1)  # [M, 8]
                    
                    M_shape, K = idx.shape
                    C = feats.shape[-1]
                    
                    # Accumulate sequentially to avoid a massive [M, K, C] memory spike
                    out = torch.zeros((M_shape, C), dtype=feats.dtype, device=feats.device)
                    for i in range(K):
                        out += feats[idx[:, i]] * weights[:, i].unsqueeze(-1)
                    return out

                @staticmethod
                def indice_weighed_sum_bwd_input(grad_output, indices, weights, N):
                    M, C = grad_output.shape
                    idx = indices.long().clamp(min=0, max=N - 1)
                    weighted_grad = grad_output.unsqueeze(1) * weights.unsqueeze(-1)  # [M, 8, C]
                    grad_feats = torch.zeros(N, C, device=grad_output.device, dtype=grad_output.dtype)
                    grad_feats.scatter_add_(0, idx.reshape(-1, 1).expand(-1, C), weighted_grad.reshape(-1, C))
                    return grad_feats

            _fgk.triton = _TritonFallback()
            print("[WORKER] flex_gemm Triton fallback patch applied.")
    except ImportError:
        print("[WORKER] Could not patch flex_gemm triton fallback.")
    except Exception as e:
        print(f"[WORKER] flex_gemm triton patch failed: {e}")


def _worker_main(cmd_queue, result_queue):
    """Worker process main loop. Owns the pipeline and all GPU resources."""
    # 当前加载的模型 key 和 pipeline 实例（切换模型时用）
    global _CURRENT_MODEL, _PIPELINE
    _CURRENT_MODEL = None
    _PIPELINE = None
    # Keep the autotune cache beside the app. The default user-profile path can
    # be left locked by an interrupted Windows worker, which prevents flex_gemm
    # from importing at all on the next launch.
    _code_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ.setdefault(
        "FLEX_GEMM_AUTOTUNE_CACHE_PATH",
        os.path.join(_code_dir, "output", "autotune_cache.json"),
    )
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    # expandable_segments:True —— 让分配器向驱动申请一段可扩张的虚拟地址空间，
    # 而不是每遇到一个新尺寸就向 WSL 的 dxgkrnl 要一大块物理显存。
    # 依据：2026-10-02 的 L1 端到端测试里，PyTorch 在「还有 3.53 GiB 空闲」的情况下
    # 申请 670 MiB 被拒（dmesg: dxgkio_create_allocation: Ioctl failed: -75），
    # 而宿主机 nvidia-smi 显示确实有盈余 —— 这是典型的显存碎片/分配模式问题，
    # 也正是报错信息本身推荐的设置。
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:True,garbage_collection_threshold:0.65"
    )
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
    # 生成进度写到这个文件，主进程的 UI 轮询读取
    if not os.environ.get("PIXAL3D_PROGRESS"):
        os.environ["PIXAL3D_PROGRESS"] = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "output", "_progress.json"
        )

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    try:
        _apply_patches()

        import torch
        import cv2
        import gc
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
            os.environ.setdefault("TRELLIS2_FORCE_FP16", "1")
        from trellis2.pipelines import (
            Trellis2ImageTo3DPipeline,
            Pixal3DImageTo3DPipeline,
            Pixal3DMVImageTo3DPipeline,
        )
        from trellis2.renderers import EnvMap
        from trellis2.utils import render_utils
        from trellis2.modules.sparse import SparseTensor
        import o_voxel

        def _resource_snapshot():
            import psutil

            memory = psutil.virtual_memory()
            if torch.cuda.is_available():
                free_vram, total_vram = torch.cuda.mem_get_info()
                free_vram_mb = free_vram / (1024 ** 2)
                total_vram_mb = total_vram / (1024 ** 2)
            else:
                free_vram_mb = total_vram_mb = 0.0
            return {
                "ram_available_mb": memory.available / (1024 ** 2),
                "ram_total_mb": memory.total / (1024 ** 2),
                "vram_free_mb": free_vram_mb,
                "vram_total_mb": total_vram_mb,
            }

        def _model_weight_bytes(path: str) -> int:
            total = 0
            for root, _dirs, files in os.walk(path):
                for filename in files:
                    if filename.endswith((".safetensors", ".bin", ".pt", ".pth")):
                        total += os.path.getsize(os.path.join(root, filename))
            return total

        def _largest_model_weight_bytes(path: str) -> int:
            largest = 0
            for root, _dirs, files in os.walk(path):
                for filename in files:
                    if filename.endswith((".safetensors", ".bin", ".pt", ".pth")):
                        largest = max(largest, os.path.getsize(os.path.join(root, filename)))
            return largest

        def _configured_low_vram(path: str, default: bool = True) -> bool:
            config_path = os.path.join(path, "pipeline.json")
            try:
                import json
                with open(config_path, "r", encoding="utf-8") as config_file:
                    args = json.load(config_file).get("args", {})
                return bool(args.get("low_vram", default))
            except (OSError, ValueError, TypeError):
                return default

        def _preflight_model_load(
            key: str,
            path: str,
            after_unload: bool = False,
            ram_available_override_mb: float = None,
        ):
            snapshot = _resource_snapshot()
            weight_mb = _model_weight_bytes(path) / (1024 ** 2)
            largest_weight_mb = _largest_model_weight_bytes(path) / (1024 ** 2)
            if key == "pixal3d":
                low_vram = os.environ.get("PIXAL3D_LOW_VRAM", "1") not in (
                    "0", "false", "False"
                )
            else:
                low_vram = os.environ.get(
                    "TRELLIS2_LOW_VRAM",
                    "1" if _configured_low_vram(path) else "0",
                ) not in ("0", "false", "False")
            ram_reserve_mb = max(
                512,
                int(os.environ.get("MODEL_RAM_RESERVE_MB", "512")),
            )
            ram_factor = float(os.environ.get("MODEL_RAM_FACTOR", "1.02"))
            if low_vram:
                weight_mb = largest_weight_mb
            required_ram_mb = weight_mb * ram_factor + ram_reserve_mb
            ram_available_mb = (
                snapshot["ram_available_mb"]
                if ram_available_override_mb is None
                else ram_available_override_mb
            )
            if ram_available_mb < required_ram_mb:
                raise RuntimeError(
                    f"模型加载前内存保护拒绝 {key}：当前可用系统内存 "
                    f"{ram_available_mb:.0f}/{snapshot['ram_total_mb']:.0f} MiB，"
                    f"预计至少需要 {required_ram_mb:.0f} MiB "
                    f"（权重 {weight_mb:.0f} MiB + 安全预留 {ram_reserve_mb} MiB）。"
                    "已停止加载，未丢弃输入。"
                )

            vram_reserve_mb = max(
                1024,
                int(os.environ.get("MODEL_VRAM_RESERVE_MB", "1536")),
            )
            vram_factor = float(os.environ.get("MODEL_VRAM_FACTOR", "1.15"))
            vram_base_mb = largest_weight_mb if low_vram else weight_mb
            required_vram_mb = vram_base_mb * vram_factor + vram_reserve_mb
            if snapshot["vram_free_mb"] < required_vram_mb and not (
                key == "pixal3d" and low_vram
            ):
                mode = "低显存模式" if low_vram else "标准模式"
                raise RuntimeError(
                    f"模型加载前显存保护拒绝 {key}：当前可用显存 "
                    f"{snapshot['vram_free_mb']:.0f}/{snapshot['vram_total_mb']:.0f} MiB，"
                    f"{mode}预计至少需要约 {required_vram_mb:.0f} MiB "
                    f"（最大单文件 {largest_weight_mb:.0f} MiB + 安全预留 {vram_reserve_mb} MiB）。"
                    "已停止加载，请关闭占用显存的程序后重试。"
                )
            if key == "pixal3d" and low_vram:
                print(
                    "[RESOURCE] Pixal3D 低显存模式跳过启动阶段整文件显存预检；"
                    "每个阶段读盘前按实时剩余显存检查",
                    flush=True,
                )

            print(
                f"[RESOURCE] {key} load preflight"
                f"{' after unload' if after_unload else ''}: "
                f"RAM free={snapshot['ram_available_mb']:.0f}/"
                f"{snapshot['ram_total_mb']:.0f}MiB, "
                f"VRAM free={snapshot['vram_free_mb']:.0f}/"
                f"{snapshot['vram_total_mb']:.0f}MiB, "
                f"weights={weight_mb:.0f}MiB, "
                f"largest_weight={largest_weight_mb:.0f}MiB, "
                f"VRAM required={required_vram_mb:.0f}MiB, "
                f"RAM required={required_ram_mb:.0f}MiB",
                flush=True,
            )

        # Fast-fail before spending minutes downloading a 4B model
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Check your NVIDIA drivers and PyTorch installation.\n"
                f"  torch version: {torch.__version__}\n"
                f"  Run 'nvidia-smi' in a terminal to check GPU status."
            )

        print("[WORKER] Loading pipeline, please wait...", flush=True)

        # ── 支持的模型 ─────────────────────────────────────────
        # TRELLIS.2 用 Trellis2ImageTo3DPipeline（标准 cross-attn 条件）；
        # Pixal3D 用 Pixal3DImageTo3DPipeline（pixel-aligned 投影条件），
        # 其 image_cond_model_* 需按下文的 IMAGE_COND_CONFIGS 外部构建，
        # 推理时还需 camera_params（camera_angle_x / distance / mesh_scale）。
        # Pixal3D 权重目录的选择顺序
        # ─────────────────────────────────────────────────────────────
        # 历史注释说「fp16 flow 会溢出 NaN」，那是 _promote_pascal_tail
        # 之前的事；现在最后 6 个 block 会提到 fp32，fp16 变体已经能跑通。
        #
        # 而 Pixal3D-F32Flow-F16Decoder（flow 是 fp32）在 8GB 卡上根本装不下：
        #   实测 VRAM required=7619MiB / 8192MiB → 阶段加载直接失败：
        #   "Pixal3D 内存保护：无法读盘加载 sparse flow…
        #    该阶段权重预计需要 6646 MiB"
        # fp16 变体同一阶段只要 4581MiB，加载 13s（fp32 是 21s）。
        # 所以 8GB 及以下的卡必须优先选 fp16。
        _f32_dir = os.path.join(_code_dir, "MODELS", "Pixal3D-F32Flow-F16Decoder")
        _f16_dir = os.path.join(_code_dir, "MODELS", "Pixal3D-fp16")
        # 大显存卡（≈20GB+）才用得起 fp32 flow：它整套 20.5GB，
        # 单阶段就要 7619MiB。其余一律 fp16。
        _prefer_f32 = False
        try:
            if torch.cuda.is_available():
                _total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
                _prefer_f32 = _total_mb >= 20000
        except Exception:
            _prefer_f32 = False
        _ordered = ([_f32_dir, _f16_dir] if _prefer_f32 else [_f16_dir, _f32_dir])
        _ordered.append(os.path.join(_code_dir, "MODELS", "Pixal3D-fp32"))

        _pixal_model_dir = os.environ.get("PIXAL3D_MODEL_DIR")
        if _pixal_model_dir is None or not os.path.isdir(_pixal_model_dir):
            _pixal_model_dir = next(
                (p for p in _ordered if os.path.isdir(p)), _ordered[0]
            )
        AVAILABLE_MODELS = {
            "trellis2": ("TRELLIS.2", os.path.join(_code_dir, 'MODELS', 'TRELLIS.2-4B')),
            "pixal3d":  ("Pixal3D",   _pixal_model_dir),
        }

        # Pixal3D 各阶段的 DinoV3 投影条件特征提取器配置
        # （对应 Pixal3D 官方 inference.py 的 IMAGE_COND_CONFIGS）
        PIXAL3D_DINO_NAME = "camenduru/dinov3-vitl16-pretrain-lvd1689m"
        PIXAL3D_IMAGE_COND_CONFIGS = {
            "ss": {
                "model_name": PIXAL3D_DINO_NAME,
                "image_size": 512,
                "grid_resolution": 16,
            },
            "shape_512": {
                "model_name": PIXAL3D_DINO_NAME,
                "image_size": 512,
                "grid_resolution": 32,
                "use_naf_upsample": True,
                "naf_target_size": 512,
            },
            "shape_1024": {
                "model_name": PIXAL3D_DINO_NAME,
                "image_size": 1024,
                "grid_resolution": 64,
                "use_naf_upsample": True,
                "naf_target_size": 512,
            },
            "tex_1024": {
                "model_name": PIXAL3D_DINO_NAME,
                "image_size": 1024,
                "grid_resolution": 64,
                "use_naf_upsample": True,
                "naf_target_size": 1024,
            },
        }

        class _LazyPixal3DCondModel:
            """Construct one Pixal3D image-condition model only for its stage."""

            def __init__(self, config: dict, label: str):
                object.__setattr__(self, "_config", dict(config))
                object.__setattr__(self, "_label", label)
                object.__setattr__(self, "_loaded", None)

            @property
            def loaded_model(self):
                return object.__getattribute__(self, "_loaded")

            def ensure_loaded(self):
                loaded = object.__getattribute__(self, "_loaded")
                if loaded is not None:
                    return loaded
                from trellis2.trainers.flow_matching.mixins.image_conditioned_proj import (
                    DinoV3ProjMultiViewFeatureExtractor,
                )

                loaded = DinoV3ProjMultiViewFeatureExtractor(
                    **object.__getattribute__(self, "_config")
                )
                loaded.eval()
                if getattr(loaded, "use_naf_upsample", False):
                    loaded._load_naf()
                    print(
                        f"[WORKER] {object.__getattribute__(self, '_label')}: NAF 已加载",
                        flush=True,
                    )
                object.__setattr__(self, "_loaded", loaded)
                print(
                    f"[MODEL] 按阶段从硬盘加载条件模型: "
                    f"{object.__getattribute__(self, '_label')}",
                    flush=True,
                )
                return loaded

            def unload_to_disk(self):
                import gc

                loaded = object.__getattribute__(self, "_loaded")
                if loaded is None:
                    return
                loaded.cpu()
                object.__setattr__(self, "_loaded", None)
                del loaded
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            def __call__(self, *args, **kwargs):
                return self.ensure_loaded()(*args, **kwargs)

            def __getattr__(self, name):
                if name in {"_config", "_label", "_loaded"}:
                    raise AttributeError(name)
                return getattr(self.ensure_loaded(), name)

            def __setattr__(self, name, value):
                if name in {"_config", "_label", "_loaded"} or "_loaded" not in self.__dict__:
                    object.__setattr__(self, name, value)
                    return
                loaded = object.__getattribute__(self, "_loaded")
                if loaded is None:
                    object.__setattr__(self, name, value)
                else:
                    setattr(loaded, name, value)

        def _build_pixal3d_cond_models(pipeline):
            """构建 Pixal3D 四个阶段的 proj 条件模型（与官方 inference.py 一致）。"""
            mapping = {
                "image_cond_model_ss": "ss",
                "image_cond_model_shape_512": "shape_512",
                "image_cond_model_shape_1024": "shape_1024",
                "image_cond_model_tex_1024": "tex_1024",
            }
            for attr, cfg_key in mapping.items():
                cfg = dict(PIXAL3D_IMAGE_COND_CONFIGS[cfg_key])
                setattr(
                    pipeline,
                    attr,
                    _LazyPixal3DCondModel(cfg, attr),
                )
            print("[WORKER] Pixal3D 条件模型改为按阶段懒加载，避免 DINOv3 常驻内存",
                  flush=True)
            return pipeline

        def _build_pixal3d_views(images, camera_params, image_resolution):
            """Keep resized references on CPU; the extractor transfers one view at a time."""
            import math
            import numpy as np
            from PIL import Image

            if not images:
                raise ValueError("Pixal3D 至少需要一张参考图。")
            if len(images) > 4:
                raise ValueError(
                    f"Pixal3D 多视角最多支持 4 张参考图（主体、左、右、后），收到 {len(images)} 张。"
                    "请删除多余图片后重试，未处理任何参考图。"
                )
            base = torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, -float(camera_params["distance"])],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ], dtype=torch.float32)
            poses = []
            yaw_degrees = [0.0, -90.0, 90.0, 180.0]
            view_count = len(images)
            for index in range(view_count):
                yaw = math.radians(yaw_degrees[index])
                rotation = torch.tensor([
                    [math.cos(yaw), 0.0, math.sin(yaw), 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [-math.sin(yaw), 0.0, math.cos(yaw), 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ], dtype=torch.float32)
                poses.append(rotation @ base)

            source_images = []
            for source in images:
                if isinstance(source, (str, bytes, os.PathLike)):
                    source_images.append(os.fspath(source))
                else:
                    image = source if isinstance(source, Image.Image) else Image.fromarray(np.asarray(source))
                    source_images.append(image.convert("RGB"))

            return {
                "source_images": source_images,
                "camera_angle_x": torch.full(
                    (1, view_count), float(camera_params["camera_angle_x"]), dtype=torch.float32
                ),
                "camera_distance": torch.full(
                    (1, view_count), float(camera_params["distance"]), dtype=torch.float32
                ),
                "transform_matrix": torch.stack(poses, dim=0).unsqueeze(0),
                "mesh_scale": float(camera_params["mesh_scale"]),
            }

        def _enable_pixal3d_fp32_tail(pipeline):
            """Keep the final SLat block in fp32 on pre-Ampere GPUs.

            Pixal3D's fp16 activations grow across the 30-block shape DiT and
            can hit 65504 in the tail block on Pascal. Only the tail block is
            promoted, keeping the low-VRAM path practical.
            """
            import functools
            from trellis2.modules.utils import convert_module_to

            if os.environ.get("PIXAL3D_FP32_TAIL", "1").lower() in (
                "0", "false", "no"
            ):
                return
            major, _minor = torch.cuda.get_device_capability()
            if major >= 8:
                return
            try:
                tail_blocks = max(1, int(os.environ.get("PIXAL3D_FP32_TAIL_BLOCKS", "6")))
            except ValueError:
                tail_blocks = 6
            promoted = 0
            for model in pipeline.models.values():
                if getattr(model, "loaded_model", None) is None:
                    continue
                model = model.loaded_model
                if not hasattr(model, "blocks") or not hasattr(model, "image_attn_mode"):
                    continue
                if "SLatFlowModel" not in model.__class__.__name__:
                    continue
                if model.image_attn_mode != "proj" or not model.blocks:
                    continue
                for block in model.blocks[-tail_blocks:]:
                    block.apply(
                        functools.partial(convert_module_to, dtype=torch.float32)
                    )
                promoted += 1
            if promoted:
                print(
                    f"[WORKER] Pixal3D Pascal 兼容模式：{promoted} 个 flow model "
                    f"的最后 {tail_blocks} 个 block 使用 fp32",
                    flush=True,
                )

        def _distance_from_fov(camera_angle_x, grid_point, target_point, mesh_scale, image_resolution):
            """由 FOV 反推相机距离（与 Pixal3D 官方 inference.py 一致）。"""
            rotation_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
            gp = grid_point.to(torch.float32) @ rotation_matrix.T
            gp = gp / mesh_scale / 2
            xw, yw, zw = gp[0].item(), gp[1].item(), gp[2].item()
            xt, yt = float(target_point[0].item()), float(target_point[1].item())
            focal_length = 16.0 / torch.tan(torch.tensor(camera_angle_x / 2.0))
            f_pixels = float((focal_length * image_resolution / 32.0).item())
            x_ndc = xt - image_resolution / 2.0
            y_ndc = -(yt - image_resolution / 2.0)
            distance_x = f_pixels * xw / x_ndc - yw
            return {"distance_from_x": float(distance_x), "f_pixels": f_pixels}

        def _estimate_camera_params(image, resolution=512, manual_fov=-1.0,
                                    mesh_scale=1.0, extend_pixel=0):
            """估计 Pixal3D 所需的相机参数（manual_fov>0 时跳过 MoGe）。

            image 可以是 PIL.Image 或文件路径；MoGe 在其上做单目深度/内参估计，
            与 Pixal3D 官方 inference.py 的 get_camera_params_wild_moge 等价。
            """
            import math
            camera_angle_x = None
            if manual_fov and manual_fov > 0:
                camera_angle_x = float(manual_fov)
            else:
                try:
                    import numpy as _np
                    from PIL import Image as _Image
                    from moge.model.v2 import MoGeModel
                except ImportError as _e:
                    # MoGe 不可用时不静默失败：给出明确指引，并退到保守 FOV，
                    # 让 Pixal3D 至少能出图（质量略降）。
                    print(f"[WORKER][WARN] MoGe 不可用（{_e}）。退用默认 FOV=0.2rad。"
                          f"\n  如需精确相机估计，请安装 MoGe："
                          f"\n    pip install git+https://github.com/microsoft/MoGe.git", flush=True)
                    camera_angle_x = 0.2
                else:
                    if isinstance(image, str):
                        pil = _Image.open(image).convert("RGB")
                    else:
                        pil = image.convert("RGB")
                    w, h = pil.size
                    print("[WORKER] MoGe-2 估计相机参数...", flush=True)
                    moge = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl")
                    moge = moge.to("cuda").eval()
                    arr = _np.array(pil).astype(_np.float32) / 255.0
                    ten = torch.from_numpy(arr).permute(2, 0, 1).to("cuda")
                    with torch.no_grad():
                        out = moge.infer(ten)
                    intr = out["intrinsics"].squeeze().cpu().numpy()
                    fx = float(intr[0, 0]) * w
                    camera_angle_x = 2 * math.atan(w / (2 * fx))
                    # 释放 MoGe 显存，留给主流程
                    moge.cpu()
                    del moge, ten, out
                    torch.cuda.empty_cache()
            grid_point = torch.tensor([-1.0, 0.0, 0.0])
            distance = _distance_from_fov(
                camera_angle_x, grid_point,
                torch.tensor([0 - extend_pixel, resolution - 1 + extend_pixel]),
                mesh_scale, resolution,
            )["distance_from_x"]
            print(f"[WORKER] camera_angle_x={camera_angle_x:.4f}, distance={distance:.4f}", flush=True)
            return {'camera_angle_x': camera_angle_x, 'distance': distance, 'mesh_scale': mesh_scale}

        def _load_model(key: str):
            """加载/切换模型。切换时先卸载旧的释放显存。"""
            global _CURRENT_MODEL, _PIPELINE
            if key not in AVAILABLE_MODELS:
                raise ValueError(f"未知模型: {key}（可选: {list(AVAILABLE_MODELS)}）")
            if _PIPELINE is not None and _CURRENT_MODEL == key:
                return _PIPELINE            # 已经是它了，不重复加载

            label, path = AVAILABLE_MODELS[key]
            if not os.path.isdir(path):
                raise RuntimeError(f"模型目录不存在: {path}")

            # 切换前先按旧模型可释放的权重估算目标模型的内存门槛。
            # 资源明显不足时保留当前模型，避免失败切换让工作台失去可用引擎。
            current_snapshot = _resource_snapshot()
            released_ram_mb = 0.0
            if _PIPELINE is not None and _CURRENT_MODEL in AVAILABLE_MODELS:
                _old_path = AVAILABLE_MODELS[_CURRENT_MODEL][1]
                released_ram_mb = _model_weight_bytes(_old_path) / (1024 ** 2)
            _preflight_model_load(
                key,
                path,
                after_unload=True,
                ram_available_override_mb=(
                    current_snapshot["ram_available_mb"] + released_ram_mb
                ),
            )

            # 卸载旧 pipeline
            if _PIPELINE is not None:
                print(f"[WORKER] 卸载 {_CURRENT_MODEL} ...", flush=True)
                try:
                    _PIPELINE.cpu()
                except Exception as unload_error:
                    old_model = _CURRENT_MODEL
                    _PIPELINE = None
                    _CURRENT_MODEL = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        f"卸载旧模型 {old_model} 失败，已停止切换以避免显存叠加："
                        f" {type(unload_error).__name__}: {unload_error}"
                    ) from unload_error
                del _PIPELINE
                _PIPELINE = None
                _CURRENT_MODEL = None
                gc.collect()
                torch.cuda.empty_cache()

            _preflight_model_load(key, path, after_unload=True)
            print(f"[WORKER] 加载 {label} <- {path}", flush=True)

            if key == "pixal3d":
                # Pixal3D：投影条件管线
                _pixal_config = (
                    "pipeline_mv.json"
                    if os.path.isfile(os.path.join(path, "pipeline_mv.json"))
                    else "pipeline.json"
                )
                p = Pixal3DMVImageTo3DPipeline.from_pretrained(
                    path,
                    config_file=_pixal_config,
                )
                print("[WORKER] 构建 Pixal3D 投影条件模型 (DinoV3Proj) ...", flush=True)
                _build_pixal3d_cond_models(p)

                # ── bf16 -> fp16 ────────────────────────────────
                # Pixal3D 的模型配置里 dtype=bfloat16，构造时 convert_to(bfloat16)，
                # forward 里 manual_cast 也按 bf16 算。
                # 但 Pascal（sm_61，如 GTX 1070）不支持 bf16：
                #   xformers/fa2 都会报 "bf16 is only supported on A100+ GPUs"，
                #   即使走 sdpa 也跑不了。
                # 我们的权重本来就是 fp16（2D 矩阵已转、norm/bias 留 fp32），
                # 所以这里把模型的 dtype 标记改回 float16，并同步转换参数/缓冲。
                _n_conv = 0
                for _mname, _m in p.models.items():
                    if getattr(_m, "loaded_model", None) is None:
                        continue
                    try:
                        if getattr(_m, 'dtype', None) == torch.bfloat16:
                            _m.dtype = torch.float16
                        for _prm in _m.parameters():
                            if _prm.dtype == torch.bfloat16:
                                _prm.data = _prm.data.half()
                                _n_conv += 1
                        for _buf in _m.buffers():
                            if _buf.dtype == torch.bfloat16:
                                _buf.data = _buf.data.half()
                                _n_conv += 1
                    except Exception as _e:
                        print(f"[WORKER] ⚠ {_mname} bf16→fp16 转换异常: {_e}", flush=True)
                _enable_pixal3d_fp32_tail(p)

                # ── 关于 flow 模型的 fp16 溢出问题 ────────────────
                # 实测结论：fp16 的 flow 模型在采样时会溢出产生 NaN
                # （sparse coords 正常 4716 个，但 sample_shape_slat 输出的
                #  feats 全是 NaN → upsample 时 coords 归零 → conv 报
                #  IndexError: max()）。
                # 试过把 flow 的 dtype 标记改成 float32 让 manual_cast 走 fp32，
                # 但权重本身还是 fp16，直接报
                #   RuntimeError: mat1 and mat2 must have the same dtype,
                #                 but got Float and Half
                # → 除非重下 fp32 原版权重，否则这条路走不通。
                # 官方量化工具也提过 "F32 flow, F16 decoder"，
                # 说明 flow 本来就该是 fp32。
                print("[WORKER] Pixal3D 权重布局：F32 Flow + F16 Decoder；"
                      "当前默认目录不是全模型 FP32。", flush=True)
                # 条件模型（不在 p.models 里）
                for _attr in ('image_cond_model_ss', 'image_cond_model_shape_512',
                              'image_cond_model_shape_1024', 'image_cond_model_tex_1024'):
                    _cm = getattr(p, _attr, None)
                    _loaded_cm = getattr(_cm, "loaded_model", None)
                    if _loaded_cm is None:
                        continue
                    try:
                        if getattr(_loaded_cm, 'dtype', None) == torch.bfloat16:
                            _loaded_cm.dtype = torch.float16
                        for _prm in _loaded_cm.parameters():
                            if _prm.dtype == torch.bfloat16:
                                _prm.data = _prm.data.half()
                    except Exception:
                        pass
                print(f"[WORKER] bf16→fp16 完成（{_n_conv} 个参数/缓冲）", flush=True)

                # 显存策略：Pixal3D 低显存模式按阶段从硬盘加载权重，
                # 每个阶段结束后释放 CPU/GPU 对象，避免整套权重常驻内存。
                # 默认低显存（对 8GB 卡更友好），可用 PIXAL3D_LOW_VRAM=0 强制标准模式。
                _low_vram = os.environ.get("PIXAL3D_LOW_VRAM", "1") not in ("0", "false", "False")
                p.low_vram = _low_vram
                if _low_vram:
                    p._device = torch.device("cuda")
                    print("[WORKER] Pixal3D 低显存模式已启用", flush=True)
                else:
                    p.cuda()
                    print("[WORKER] Pixal3D 标准模式（全部权重驻留 GPU）", flush=True)

                # image_cond_model_* 不在 pipeline.models 内，需单独处理；
                # 同时按需加载 NAF 上采样权重。
                if not _low_vram:
                    for attr in ('image_cond_model_ss', 'image_cond_model_shape_512',
                                 'image_cond_model_shape_1024', 'image_cond_model_tex_1024'):
                        m = getattr(p, attr, None)
                        if m is None:
                            continue
                        m.ensure_loaded()
                        if not getattr(m, 'use_naf_upsample', False):
                            continue
                        # NAF 依赖 natten（Windows 无官方 wheel，Pascal 卡基本装不上）。
                        # 这里不再阻断加载：拿不到 NAF 时，image_conditioned_proj 里
                        # 会自动回退到双线性插值上采样（维度不变，质量可能略降）。
                        try:
                            m._load_naf()
                            print(f"[WORKER] {attr}: NAF 已加载", flush=True)
                        except Exception as _naf_err:
                            raise RuntimeError(
                                f"{attr} 的 NAF 依赖不可用，已停止 Pixal3D 加载："
                                f"{type(_naf_err).__name__}: {str(_naf_err)[:160]}"
                            ) from _naf_err
            else:
                # TRELLIS.2：标准条件管线
                p = Trellis2ImageTo3DPipeline.from_pretrained(path)
                _low_vram = os.environ.get("TRELLIS2_LOW_VRAM", "1") not in (
                    "0", "false", "False"
                )
                p.low_vram = _low_vram
                if _low_vram:
                    p._device = torch.device("cuda")
                    print("[WORKER] TRELLIS.2 低显存模式已启用，主模型按阶段读盘", flush=True)
                else:
                    p.cuda()

            _PIPELINE = p
            _CURRENT_MODEL = key
            print(f"[WORKER] {label} 就绪", flush=True)
            return _PIPELINE

        # 启动时加载默认模型（可用 PIXAL3D_DEFAULT_MODEL 覆盖）
        _default = os.environ.get("PIXAL3D_DEFAULT_MODEL", "trellis2")
        if _default not in AVAILABLE_MODELS:
            _default = "trellis2"
        _load_model(_default)          # 只做预热；后续统一走 _PIPELINE，避免切换后指向旧模型
        print("[WORKER] Pipeline loaded, loading environment maps...", flush=True)


        _code_dir = os.path.dirname(os.path.abspath(__file__))
        envmap = {}
        for env_name in ('forest', 'sunset', 'courtyard'):
            exr_path = os.path.join(_code_dir, 'assets', 'hdri', f'{env_name}.exr')
            raw = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise RuntimeError(
                    f"Failed to load '{exr_path}'. "
                    f"File exists: {os.path.isfile(exr_path)}. "
                    f"OpenCV may lack OpenEXR support — try: pip install opencv-contrib-python"
                )
            envmap[env_name] = EnvMap(torch.tensor(
                cv2.cvtColor(raw, cv2.COLOR_BGR2RGB),
                dtype=torch.float32, device='cuda'
            ))

        print("[WORKER] Pipeline ready.", flush=True)
        result_queue.put({"status": "ready"})
    except Exception as e:
        traceback.print_exc()
        result_queue.put({"status": "error", "error": f"Worker init failed: {e}"})
        return

    while True:
        cmd = cmd_queue.get()
        action = cmd["action"]

        if action == "generate":
            # UI 的「显存不足时自动降级」开关。
            # 打开（默认）：装不下就自动降分辨率 / NAF 目标尺寸，保证跑得完。
            # 关掉：用满配置硬跑，装不下时由 WSL 驱动把显存页换到系统内存
            #       —— 不会失败，但会慢很多，而且可能把宿主机拖卡。
            _auto = "1" if cmd.get("auto_degrade", True) else "0"
            os.environ["PIXAL3D_AUTO_DEGRADE"] = _auto
            os.environ["PIXAL3D_NAF_ADAPTIVE"] = _auto
            print(
                f"[VRAM] 自动降级 = {'开' if _auto == '1' else '关（装不下会占用系统内存，很慢）'}",
                flush=True,
            )

        if action == "shutdown":
            print("[WORKER] Shutting down.")
            break

        elif action == "preprocess":
            try:
                image = _PIPELINE.preprocess_image(cmd["image"])
                result_queue.put({"status": "ok", "image": image})
            except Exception as e:
                traceback.print_exc()
                result_queue.put({"status": "error", "error": str(e)})

        elif action == "switch_model":
            # UI 切模型时先调这个，避免在 generate 里才发现要等加载
            try:
                key = cmd["model"]
                _load_model(key)
                result_queue.put({"status": "ok", "model": key,
                                  "label": AVAILABLE_MODELS[key][0]})
            except Exception as e:
                traceback.print_exc()
                result_queue.put({"status": "error", "error": str(e)})

        elif action == "generate":
            try:
                # 如果请求的模型和当前不同，先切换
                req_model = cmd.get("model")
                if req_model and req_model != _CURRENT_MODEL:
                    _load_model(req_model)
                prof_cfg = cmd["profiling"]
                prof_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp', 'profiling')
                master_enabled = prof_cfg["enable_python"] or prof_cfg["enable_torch"]
                profiler = AuraProfiler(
                    log_dir=prof_dir,
                    actor_name="",
                    enabled=master_enabled,
                    enable_python=prof_cfg["enable_python"],
                    enable_torch=prof_cfg["enable_torch"],
                    schedule_config={"wait": 0, "warmup": 0, "active": 10000, "repeat": 0},
                    delay_sec=prof_cfg["delay_sec"],
                    max_duration_sec=prof_cfg["max_duration_sec"],
                    max_events=prof_cfg["max_events"],
                )
                with hunt_syncs(enabled=prof_cfg["enable_sync_hunter"],
                                log_file=os.path.join(prof_dir, "sync_report.txt")):
                    profiler.start()
                    try:
                        import _progress
                        _progress.start()
                    except Exception:
                        pass
                    # Pixal3D 需要相机参数（投影条件）；TRELLIS.2 不需要。
                    _run_kwargs = dict(
                        seed=cmd["seed"],
                        preprocess_image=False,
                        sparse_structure_sampler_params=cmd["ss_params"],
                        shape_slat_sampler_params=cmd["shape_params"],
                        tex_slat_sampler_params=cmd["tex_params"],
                        pipeline_type=cmd["pipeline_type"],
                        return_latent=True,
                    )
                    if _CURRENT_MODEL == "pixal3d":
                        # Pixal3D 只支持级联类型；把 UI 的 512/1024 归一到 1024_cascade。
                        _pt = cmd["pipeline_type"]
                        if _pt not in ("1024_cascade", "1536_cascade"):
                            _pt = "1024_cascade"
                        _run_kwargs["pipeline_type"] = _pt
                        # 相机估计所用的图像分辨率：低配优先 512（GTX 1070 8GB 友好）。
                        _img_res = cmd.get("image_resolution", 512)
                        _camera_params = _estimate_camera_params(
                            cmd["image"],
                            resolution=_img_res,
                            manual_fov=cmd.get("manual_fov", -1.0),
                            mesh_scale=cmd.get("mesh_scale", 1.0),
                        )
                        _run_kwargs["max_num_tokens"] = cmd.get("max_num_tokens", 49152)
                        _run_kwargs["camera_params"] = _camera_params
                        _views = _build_pixal3d_views(
                            [cmd["image"], *(cmd.get("reference_images") or [])],
                            _camera_params,
                            _img_res,
                        )
                        outputs, latents = _PIPELINE.run_mv(_views, **_run_kwargs)
                    else:
                        outputs, latents = _PIPELINE.run(cmd["image"], **_run_kwargs)
                    profiler.step()
                profiler.stop_and_save("inference_run")
                try:
                    import _progress
                    _progress.stage("decode", "解码网格")
                except Exception:
                    pass
                def _vram_mb():
                    return torch.cuda.memory_allocated() / (1024**2)
                print(f"[VRAM] After pipeline.run: allocated={_vram_mb():.0f}MB")
                mesh = outputs[0]
                
                # Move latents to CPU BEFORE rendering to free VRAM
                shape_slat, tex_slat, res = latents
                state = {
                    'shape_slat_feats': shape_slat.feats.cpu().numpy(),
                    'tex_slat_feats': tex_slat.feats.cpu().numpy(),
                    'coords': shape_slat.coords.cpu().numpy(),
                    'res': res,
                }
                del latents, shape_slat, tex_slat, outputs
                print(f"[VRAM] After latent offload: allocated={_vram_mb():.0f}MB")
                torch.cuda.empty_cache()
                
                mesh.simplify(16777216)  # nvdiffrast limit
                print(f"[VRAM] After simplify: allocated={_vram_mb():.0f}MB")
                try:
                    import _progress
                    _progress.stage("render", "渲染预览图（6 种模式 × 8 视角）")
                except Exception:
                    pass
                
                with torch.inference_mode():
                    images = render_utils.render_snapshot(
                        mesh, 
                        resolution=512,
                        r=2, fov=36,
                        nviews=cmd["nviews"], 
                        envmap=envmap
                    )
                print(f"[VRAM] After render: allocated={_vram_mb():.0f}MB")
                torch.cuda.empty_cache()
                try:
                    import _progress
                    _progress.finish()
                except Exception:
                    pass
                result_queue.put({"status": "ok", "state": state, "images": images})
                del mesh, images, state
                torch.cuda.empty_cache()
            except Exception as e:
                traceback.print_exc()
                try:
                    import _progress
                    _progress.finish()
                except Exception:
                    pass
                result_queue.put({"status": "error", "error": str(e)})
                torch.cuda.empty_cache()

        elif action == "extract_glb":
            try:
                # Offload envmaps to CPU — not used during GLB export
                for env in envmap.values():
                    env.offload()
                torch.cuda.empty_cache()
                
                state = cmd["state"]
                shape_slat = SparseTensor(
                    feats=torch.from_numpy(state['shape_slat_feats']).cuda(),
                    coords=torch.from_numpy(state['coords']).cuda(),
                )
                tex_slat = shape_slat.replace(torch.from_numpy(state['tex_slat_feats']).cuda())
                res = state['res']
                
                mesh = _PIPELINE.decode_latent(shape_slat, tex_slat, res)[0]
                mesh.attrs = mesh.attrs.float()
                
                # Free everything possible before heavy GLB postprocessing
                del shape_slat, tex_slat, state
                for name, model in _PIPELINE.models.items():
                    model.cpu()
                torch.cuda.empty_cache()
                
                glb = o_voxel.postprocess.to_glb(
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    attr_volume=mesh.attrs,
                    coords=mesh.coords,
                    attr_layout=_PIPELINE.pbr_attr_layout,
                    grid_size=res,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    decimation_target=cmd["decimation_target"],
                    texture_size=cmd["texture_size"],
                    remesh=True,
                    remesh_band=1,
                    remesh_project=0,
                    use_tqdm=True,
                )
                glb.export(cmd["glb_path"])
                del mesh, glb
                torch.cuda.empty_cache()
                # Flush any lingering CUDA errors (e.g. from xatlas assertions)
                # so they don't poison subsequent kernel launches
                torch.cuda.synchronize()
                
                # Reload envmaps back to GPU for next render cycle
                for env in envmap.values():
                    env.reload()
                
                result_queue.put({"status": "ok", "glb_path": cmd["glb_path"]})
            except Exception as e:
                traceback.print_exc()
                # Reload envmaps even on failure, so next generate's render works
                for env in envmap.values():
                    try:
                        env.reload()
                    except Exception:
                        pass
                torch.cuda.empty_cache()
                result_queue.put({"status": "error", "error": str(e)})


class PipelineWorker:
    """Manages a subprocess that owns the GPU pipeline."""

    def __init__(self):
        ctx = mp.get_context('spawn')  # 'spawn' is required for CUDA on Windows
        self.cmd_queue = ctx.Queue()
        self.result_queue = ctx.Queue()
        self.process = ctx.Process(target=_worker_main, args=(self.cmd_queue, self.result_queue))
        self.process.daemon = True
        print("[INFO] Starting pipeline worker process...")
        self.process.start()
        # Poll for readiness, detecting early death quickly
        import queue as _queue_mod
        while True:
            if not self.process.is_alive():
                # Drain any error the worker managed to send before dying
                try:
                    msg = self.result_queue.get_nowait()
                except _queue_mod.Empty:
                    raise RuntimeError(
                        f"Worker process died unexpectedly (exit code {self.process.exitcode}). "
                        f"Check the console output above for errors."
                    )
                raise RuntimeError(f"Worker process died during init: {msg['error']}")
            try:
                msg = self.result_queue.get(timeout=None)
                break
            except _queue_mod.Empty:
                continue
        if msg["status"] == "error":
            raise RuntimeError(f"Worker init failed: {msg['error']}")
        assert msg["status"] == "ready", f"Worker failed to start: {msg}"
        print("[INFO] Pipeline worker ready.")

    def preprocess(self, image):
        self.cmd_queue.put({"action": "preprocess", "image": image})
        result = self._wait_for_result("预处理")
        if result["status"] == "error":
            raise RuntimeError(result["error"])
        return result["image"]

    def switch_model(self, model: str):
        """切换到指定模型（'trellis2' / 'pixal3d'）。会等加载完成。"""
        self.cmd_queue.put({"action": "switch_model", "model": model})
        result = self._wait_for_result("切换模型")
        if result["status"] == "error":
            raise RuntimeError(result["error"])
        return result.get("label", model)

    def generate(self, image, seed, ss_params, shape_params, tex_params, pipeline_type,
                 nviews, model=None, profiling=None, progress_callback=None,
                 image_resolution=512, manual_fov=-1.0, mesh_scale=1.0, max_num_tokens=49152,
                 reference_images=None, auto_degrade=True):
        self.cmd_queue.put({
            "action": "generate",
            "image": image,
            "reference_images": reference_images or [],
            "seed": seed,
            "ss_params": ss_params,
            "shape_params": shape_params,
            "tex_params": tex_params,
            "pipeline_type": pipeline_type,
            "nviews": nviews,
            "model": model,
            "profiling": profiling or {},
            # ── Pixal3D 相机/分辨率参数（TRELLIS.2 忽略）──
            "image_resolution": image_resolution,
            "manual_fov": manual_fov,
            "mesh_scale": mesh_scale,
            "max_num_tokens": max_num_tokens,
            # UI 的「显存不足时自动降级」开关
            "auto_degrade": bool(auto_degrade),
        })
        import queue as _queue_mod
        while True:
            try:
                result = self.result_queue.get(timeout=10)
                break
            except _queue_mod.Empty:
                if not self.process.is_alive():
                    raise RuntimeError(
                        "Worker process died during generation "
                        f"(exit code {self.process.exitcode}). "
                        "Check the worker console output for the underlying error."
                    )
                if progress_callback:
                    progress_callback()
                continue
        if result["status"] == "error":
            raise RuntimeError(result["error"])
        return result["state"], result["images"]

    def extract_glb(self, state, decimation_target, texture_size, glb_path):
        self.cmd_queue.put({
            "action": "extract_glb",
            "state": state,
            "decimation_target": decimation_target,
            "texture_size": texture_size,
            "glb_path": glb_path,
        })
        result = self._wait_for_result("导出 GLB")
        if result["status"] == "error":
            raise RuntimeError(result["error"])
        return result["glb_path"]

    def _wait_for_result(self, operation: str):
        import queue as _queue_mod
        while True:
            try:
                return self.result_queue.get(timeout=10)
            except _queue_mod.Empty:
                if not self.process.is_alive():
                    raise RuntimeError(
                        f"Worker process died during {operation} "
                        f"(exit code {self.process.exitcode}). "
                        "Check the worker console output for the underlying error."
                    )

    def shutdown(self):
        try:
            self.cmd_queue.put({"action": "shutdown"})
            self.process.join(timeout=20)
        except Exception:
            self.process.kill()
