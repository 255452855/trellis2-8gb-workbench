from typing import *
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from .base import Pipeline
from . import samplers, rembg
from ..modules.sparse import SparseTensor
from ..modules import image_feature_extractor
from ..representations import Mesh, MeshWithVoxel


class Pixal3DImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Pixal3D (proj mode) image-to-3D models.

    Based on Trellis2 pipeline, using proj mode for inference.
    Each stage (SS, Shape 512, Shape 1024, Tex 1024) has its own image_cond_model (DinoV3ProjFeatureExtractor).
    Condition building uses camera-aware projection (requires camera_angle_x, distance, mesh_scale parameters).

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        shape_slat_sampler (samplers.Sampler): The sampler for the structured latent.
        tex_slat_sampler (samplers.Sampler): The sampler for the texture latent.
        sparse_structure_sampler_params (dict): The parameters for the sparse structure sampler.
        shape_slat_sampler_params (dict): The parameters for the structured latent sampler.
        tex_slat_sampler_params (dict): The parameters for the texture latent sampler.
        shape_slat_normalization (dict): The normalization parameters for the structured latent.
        tex_slat_normalization (dict): The normalization parameters for the texture latent.
        image_cond_model_ss (nn.Module): Proj image cond model for sparse structure stage.
        image_cond_model_shape_512 (nn.Module): Proj image cond model for shape LR (512) stage.
        image_cond_model_shape_1024 (nn.Module): Proj image cond model for shape HR (1024) stage.
        image_cond_model_tex_1024 (nn.Module): Proj image cond model for texture (1024) stage.
        rembg_model (Callable): The model for removing background.
        low_vram (bool): Whether to use low-VRAM mode.
    """
    model_names_to_load = [
        'sparse_structure_flow_model',
        'sparse_structure_decoder',
        'shape_slat_flow_model_512',
        'shape_slat_flow_model_1024',
        'shape_slat_decoder',
        'tex_slat_flow_model_512',
        'tex_slat_flow_model_1024',
        'tex_slat_decoder',
    ]

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        shape_slat_sampler: samplers.Sampler = None,
        tex_slat_sampler: samplers.Sampler = None,
        sparse_structure_sampler_params: dict = None,
        shape_slat_sampler_params: dict = None,
        tex_slat_sampler_params: dict = None,
        shape_slat_normalization: dict = None,
        tex_slat_normalization: dict = None,
        image_cond_model_ss: nn.Module = None,
        image_cond_model_shape_512: nn.Module = None,
        image_cond_model_shape_1024: nn.Module = None,
        image_cond_model_tex_1024: nn.Module = None,
        rembg_model: Callable = None,
        low_vram: bool = True,
        default_pipeline_type: str = '1024_cascade',
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.shape_slat_sampler = shape_slat_sampler
        self.tex_slat_sampler = tex_slat_sampler
        self.sparse_structure_sampler_params = sparse_structure_sampler_params
        self.shape_slat_sampler_params = shape_slat_sampler_params
        self.tex_slat_sampler_params = tex_slat_sampler_params
        self.shape_slat_normalization = shape_slat_normalization
        self.tex_slat_normalization = tex_slat_normalization
        self.image_cond_model_ss = image_cond_model_ss
        self.image_cond_model_shape_512 = image_cond_model_shape_512
        self.image_cond_model_shape_1024 = image_cond_model_shape_1024
        self.image_cond_model_tex_1024 = image_cond_model_tex_1024
        self.rembg_model = rembg_model
        self.low_vram = low_vram
        self.default_pipeline_type = default_pipeline_type
        self.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        self._device = 'cpu'

    class _LazyRembg:
        def __init__(self, model_name: str):
            self.model_name = model_name
            self.model = None

        def _ensure_loaded(self):
            if self.model is None:
                self.model = rembg.BiRefNet(self.model_name)
            return self.model

        def to(self, device):
            self._ensure_loaded().to(device)

        def cuda(self):
            self._ensure_loaded().cuda()

        def cpu(self):
            if self.model is not None:
                self.model.cpu()

        def unload_to_disk(self):
            import gc

            if self.model is None:
                return
            self.model.cpu()
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        def __call__(self, image):
            return self._ensure_loaded()(image)

    @classmethod
    def from_pretrained(cls, path: str, config_file: str = "pipeline.json") -> "Pixal3DImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super().from_pretrained(path, config_file, lazy_models=True)
        args = pipeline._pretrained_args

        pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        pipeline.shape_slat_sampler = getattr(samplers, args['shape_slat_sampler']['name'])(**args['shape_slat_sampler']['args'])
        pipeline.shape_slat_sampler_params = args['shape_slat_sampler']['params']

        pipeline.tex_slat_sampler = getattr(samplers, args['tex_slat_sampler']['name'])(**args['tex_slat_sampler']['args'])
        pipeline.tex_slat_sampler_params = args['tex_slat_sampler']['params']

        pipeline.shape_slat_normalization = args['shape_slat_normalization']
        pipeline.tex_slat_normalization = args['tex_slat_normalization']

        # Proj mode: image_cond_models need to be loaded externally, set to None here
        pipeline.image_cond_model_ss = None
        pipeline.image_cond_model_shape_512 = None
        pipeline.image_cond_model_shape_1024 = None
        pipeline.image_cond_model_tex_1024 = None

        pipeline.rembg_model = pipeline._LazyRembg(
            args['rembg_model']['args'].get("model_name", "briaai/RMBG-2.0")
        )
        
        pipeline.low_vram = args.get('low_vram', True)
        pipeline.default_pipeline_type = args.get('default_pipeline_type', '1024_cascade')
        pipeline.pbr_attr_layout = {
            'base_color': slice(0, 3),
            'metallic': slice(3, 4),
            'roughness': slice(4, 5),
            'alpha': slice(5, 6),
        }
        pipeline._device = 'cpu'

        return pipeline

    def to(self, device: torch.device) -> None:
        self._device = device
        if not self.low_vram:
            super().to(device)
            if self.rembg_model is not None:
                self.rembg_model.to(device)

    @staticmethod
    def _module_storage_bytes(module: nn.Module) -> int:
        total = 0
        seen = set()
        for tensor in list(module.parameters()) + list(module.buffers()):
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            total += tensor.numel() * tensor.element_size()
        return total

    @staticmethod
    def _cuda_memory_mb() -> Tuple[float, float]:
        if not torch.cuda.is_available():
            return 0.0, 0.0
        free, total = torch.cuda.mem_get_info()
        return free / (1024 ** 2), total / (1024 ** 2)

    def _load_module_adaptive(self, module: nn.Module, label: str) -> nn.Module:
        """Move one stage module to CUDA only when the current headroom allows it."""
        ensure_loaded = getattr(module, "ensure_loaded", None)
        if ensure_loaded is not None and torch.cuda.is_available() and self.low_vram:
            checkpoint_bytes = getattr(module, "checkpoint_bytes", 0)
            if checkpoint_bytes:
                reserve_mb = max(
                    256, int(os.environ.get("PIXAL3D_GPU_RESERVE_MB", "768"))
                )
                factor = float(os.environ.get("PIXAL3D_GPU_FACTOR", "1.15"))
                required_mb = checkpoint_bytes / (1024 ** 2) * factor + reserve_mb
                free_mb, total_mb = self._cuda_memory_mb()
                if free_mb < required_mb:
                    raise RuntimeError(
                        f"Pixal3D 内存保护：无法读盘加载 {label}。"
                        f"当前 CUDA 可用 {free_mb:.0f}/{total_mb:.0f} MiB，"
                        f"该阶段权重预计需要 {required_mb:.0f} MiB，"
                        "已停止本阶段加载，未丢弃输入。"
                    )
        loaded = ensure_loaded() if ensure_loaded is not None else module
        self._normalize_loaded_dtype(loaded)
        self._promote_pascal_tail(loaded)
        if not self.low_vram:
            loaded.to(self.device)
            return loaded
        try:
            first = next(loaded.parameters())
            if first.device.type == "cuda":
                return loaded
        except StopIteration:
            pass

        reserve_mb = max(256, int(os.environ.get("PIXAL3D_GPU_RESERVE_MB", "768")))
        module_mb = self._module_storage_bytes(loaded) / (1024 ** 2)
        # CUDA module moves need temporary allocator workspace in addition to
        # the resident weights. Keep a modest multiplier without overbooking an
        # 8 GB card.
        required_mb = max(256.0, module_mb * 1.15) + reserve_mb
        free_mb, total_mb = self._cuda_memory_mb()
        torch.cuda.empty_cache()
        free_after_cache_mb, _ = self._cuda_memory_mb()
        free_mb = max(free_mb, free_after_cache_mb)
        if free_mb < required_mb:
            raise RuntimeError(
                f"Pixal3D 内存保护：无法加载 {label}。"
                f"当前 CUDA 可用 {free_mb:.0f}/{total_mb:.0f} MiB，"
                f"预计需要 {required_mb:.0f} MiB（模块约 {module_mb:.0f} MiB，"
                f"预留 {reserve_mb} MiB）。已停止继续分配，避免驱动崩溃。"
            )

        print(
            f"[VRAM] 加载 {label}: free={free_mb:.0f}MiB/"
            f"{total_mb:.0f}MiB module={module_mb:.0f}MiB "
            f"allocated={torch.cuda.memory_allocated()/(1024**2):.0f}MiB "
            f"reserved={torch.cuda.memory_reserved()/(1024**2):.0f}MiB",
            flush=True,
        )
        try:
            loaded.to(self.device)
        except RuntimeError as exc:
            loaded.cpu()
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"Pixal3D 加载 {label} 时显存不足或碎片化；"
                "已回退到 CPU 并释放缓存。"
            ) from exc
        return loaded

    @staticmethod
    def _normalize_loaded_dtype(module: nn.Module) -> None:
        if getattr(module, "dtype", None) == torch.bfloat16:
            module.dtype = torch.float16
        for parameter in module.parameters():
            if parameter.dtype == torch.bfloat16:
                parameter.data = parameter.data.half()
        for buffer in module.buffers():
            if buffer.dtype == torch.bfloat16:
                buffer.data = buffer.data.half()

    @staticmethod
    def _promote_pascal_tail(module: nn.Module) -> None:
        if os.environ.get("PIXAL3D_FP32_TAIL", "1").lower() in (
            "0", "false", "no"
        ):
            return
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] >= 8:
            return
        if "SLatFlowModel" not in module.__class__.__name__:
            return
        if getattr(module, "image_attn_mode", None) != "proj":
            return
        if getattr(module, "_pixal3d_fp32_tail_applied", False):
            return
        import functools
        from trellis2.modules.utils import convert_module_to

        try:
            tail_blocks = max(1, int(os.environ.get("PIXAL3D_FP32_TAIL_BLOCKS", "6")))
        except ValueError:
            tail_blocks = 6
        for block in module.blocks[-tail_blocks:]:
            block.apply(functools.partial(convert_module_to, dtype=torch.float32))
        module._pixal3d_fp32_tail_applied = True

    @staticmethod
    def _pascal_fp32_tail_active() -> bool:
        """Pascal 兼容模式是否会把尾部 block 提成 fp32。

        提成 fp32 后注意力矩阵也跟着从 2 字节/元素涨到 4 字节，显存需求翻倍。
        """
        if os.environ.get("PIXAL3D_FP32_TAIL", "1").lower() in ("0", "false", "no"):
            return False
        if not torch.cuda.is_available():
            return False
        return torch.cuda.get_device_capability()[0] < 8

    @staticmethod
    def _sparse_stage_bytes(
        tokens: int,
        channels: int,
        mlp_ratio: float,
        proj_in_channels: int,
        in_channels: int,
        heads: int,
        elem_bytes: int,
    ) -> float:
        """估算稀疏阶段单次前向的峰值显存（字节）。

        ⚠️⚠️ 必须把 **自注意力矩阵** 算进去。
        原实现只算了 O(tokens) 的 MLP 激活：10407 token 时给出 640 MiB 的预算，
        而注意力矩阵实际要 `12 × 10407² × 4B ≈ 5.0 GB`（fp32 tail）—— 低估 8 倍。
        低估的后果不是"慢"，是**把整台机器搞崩**：
        WSL2 的 CUDA 驱动允许显存超额订阅，装不下时它不抛 OOM，
        而是把显存页换到系统内存 → 主机内存被吃光 → 系统卡死/崩溃（实测踩过）。
        所以宁可估高，不可估低。

        注意力**分块**之后不再一次性开满 [heads, tokens, tokens]，
        峰值封顶在 2×chunk（scores + softmax 输出），所以取 min。
        """
        linear = tokens * (
            channels * mlp_ratio * 4 * 2   # fp32 tail block 的 MLP 瞬时激活
            + proj_in_channels * 2         # fp16 投影条件
            + in_channels * 4              # latent
        )
        # ⚠️ 这里**故意按未分块的完整矩阵**估，不享受 `_bmm_sdpa` 分块带来的折扣。
        # 原因：实测过按分块后的峰值（2×chunk）放行 1024 分辨率，
        # 结果仍然把 WSL2 虚拟机整个搞崩（dmesg 里能看到
        # "Init has exited. Terminating distribution" + EXT4 unmount/remount）。
        # 说明我对峰值还有没建模到的部分 —— 那就宁可估保守。
        # 分块本身仍然保留，它只会让实际峰值更低，等于额外安全余量。
        # ×2 覆盖 softmax 的临时张量。
        quad = heads * tokens * tokens * elem_bytes * 2
        return linear + quad

    def _sparse_stage_dims(self, module: nn.Module, args: dict) -> dict:
        """从 lazy 包装或已加载模块里取出估算需要的维度。"""
        loaded = getattr(module, "loaded_model", None)
        src = loaded if loaded is not None else module

        def pick(key, default):
            value = args.get(key)
            if value in (None, 0):
                value = getattr(src, key, None)
            if value in (None, 0):
                return default
            return value

        return {
            "channels": max(1, int(pick("model_channels", 1536))),
            "mlp_ratio": max(1.0, float(pick("mlp_ratio", 4.0))),
            "proj_in_channels": max(1, int(pick("proj_in_channels", 2048))),
            "in_channels": max(1, int(pick("in_channels", 32))),
            "heads": max(1, int(pick("num_heads", 12))),
        }

    def _check_sparse_stage_budget(
        self, module: nn.Module, tokens: int, label: str
    ) -> None:
        """Reject sparse stages whose estimated weights + activations exceed headroom."""
        if not self.low_vram or not torch.cuda.is_available():
            return
        ensure_loaded = getattr(module, "ensure_loaded", None)
        torch.cuda.empty_cache()
        free_mb, total_mb = self._cuda_memory_mb()
        print(
            f"[VRAM-DBG] {label}: allocated="
            f"{torch.cuda.memory_allocated()/(1024**2):.0f}MiB reserved="
            f"{torch.cuda.memory_reserved()/(1024**2):.0f}MiB free={free_mb:.0f}MiB",
            flush=True,
        )
        loaded = getattr(module, "loaded_model", None)
        if loaded is None and ensure_loaded is not None:
            args = getattr(module, "config_args", {})
            module_mb = getattr(module, "checkpoint_bytes", 0) / (1024 ** 2)
        else:
            loaded = loaded or module
            args = {}
            module_mb = self._module_storage_bytes(loaded) / (1024 ** 2)
        reserve_mb = max(256, int(os.environ.get("PIXAL3D_GPU_RESERVE_MB", "768")))
        factor = float(os.environ.get("PIXAL3D_GPU_FACTOR", "1.15"))

        dims = self._sparse_stage_dims(module, args)
        elem_bytes = 4 if self._pascal_fp32_tail_active() else 2
        activation_mb = self._sparse_stage_bytes(
            tokens,
            dims["channels"],
            dims["mlp_ratio"],
            dims["proj_in_channels"],
            dims["in_channels"],
            dims["heads"],
            elem_bytes,
        ) / (1024 ** 2)
        required_mb = module_mb * factor + activation_mb + reserve_mb
        if free_mb < required_mb:
            raise RuntimeError(
                f"Pixal3D 显存预检拒绝 {label}：tokens={tokens}，"
                f"当前可用 {free_mb:.0f}/{total_mb:.0f} MiB，"
                f"模型约 {module_mb:.0f} MiB、激活预算 {activation_mb:.0f} MiB"
                f"（含 {dims['heads']} 头 × tokens² 的注意力矩阵）、"
                f"安全预留 {reserve_mb} MiB，总需求约 {required_mb:.0f} MiB。"
                "请降低 HR 分辨率/最大 token 数，或关闭占用显存的程序。"
            )

    def _adaptive_sparse_token_limit(
        self, module: nn.Module, requested_limit: int
    ) -> int:
        """Derive a conservative token cap from live VRAM and stage dimensions."""
        if not self.low_vram or not torch.cuda.is_available():
            return max(1, int(requested_limit))
        # UI 的「显存不足时自动降级」被关掉 → 不限制 token 数，按请求的分辨率硬跑。
        # 装不下时 WSL 驱动会把显存页换到系统内存：不会失败，但很慢，且可能卡宿主机。
        if os.environ.get("PIXAL3D_AUTO_DEGRADE", "1") in ("0", "false", "False"):
            print(
                f"[VRAM] 自动降级已关闭：不限制 HR token（请求 {requested_limit}），"
                f"按原分辨率跑；装不下会占用系统内存，速度会明显变慢。",
                flush=True,
            )
            return max(1, int(requested_limit))
        ensure_loaded = getattr(module, "ensure_loaded", None)
        torch.cuda.empty_cache()
        free_mb, total_mb = self._cuda_memory_mb()
        loaded = getattr(module, "loaded_model", None)
        if loaded is None and ensure_loaded is not None:
            args = getattr(module, "config_args", {})
            module_mb = getattr(module, "checkpoint_bytes", 0) / (1024 ** 2)
        else:
            loaded = loaded or module
            args = {}
            module_mb = self._module_storage_bytes(loaded) / (1024 ** 2)

        reserve_mb = max(256, int(os.environ.get("PIXAL3D_GPU_RESERVE_MB", "768")))
        fixed_mb = module_mb * 1.15 + reserve_mb
        usable_bytes = max(0.0, free_mb - fixed_mb) * (1024 ** 2)

        dims = self._sparse_stage_dims(module, args)
        elem_bytes = 4 if self._pascal_fp32_tail_active() else 2

        # ⚠️ 原来这里是**线性**模型（bytes_per_token × tokens），但显存需求里
        # 有一项是 O(tokens²) 的自注意力矩阵。线性模型算出 live_limit=48047，
        # 真实上限只有几千 —— 于是"预检通过"然后硬塞，把主机内存吃光、系统崩。
        # 现在解 a·t + q·t² = usable。
        #
        # 但注意力**分块**之后 q·t² 会被封顶在 2×chunk，所以要先按二次方程解，
        # 解出来的 t 如果已经越过封顶点，就改用线性方程重解。
        linear_coef = (
            dims["channels"] * dims["mlp_ratio"] * 4 * 2   # MLP 瞬时激活
            + dims["proj_in_channels"] * 2                 # fp16 投影条件
            + dims["in_channels"] * 4                      # latent
        )
        quad_coef = dims["heads"] * elem_bytes * 2         # 注意力矩阵 + softmax 临时量
        # 不按分块后的封顶值打折 —— 见 `_sparse_stage_bytes` 的注释：
        # 试过放行 1024，结果把 WSL2 虚拟机搞崩了，宁可估保守。
        if quad_coef > 0:
            disc = linear_coef ** 2 + 4.0 * quad_coef * usable_bytes
            solved = (-linear_coef + disc ** 0.5) / (2.0 * quad_coef)
        else:
            solved = usable_bytes / max(1.0, linear_coef)
        live_limit = max(1, int(solved))
        token_limit = min(max(1, int(requested_limit)), live_limit)
        print(
            f"[VRAM] HR token 上限动态计算: free={free_mb:.0f}/{total_mb:.0f}MiB "
            f"weights={module_mb:.0f}MiB reserve={reserve_mb}MiB "
            f"heads={dims['heads']} elem_bytes={elem_bytes} "
            f"live_limit={live_limit} requested={requested_limit} "
            f"selected={token_limit}",
            flush=True,
        )
        return token_limit

    @staticmethod
    def _unload_module(module: nn.Module) -> None:
        unload_to_disk = getattr(module, "unload_to_disk", None)
        if unload_to_disk is not None:
            unload_to_disk()
            return
        module.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _evict_idle_modules(self, keep=(), label: str = "") -> None:
        """按需给显存：把当前阶段用不到的权重全部退回 CPU / 磁盘。

        低显存模式下每个权重都是「懒加载」包装器，自带 `unload_to_disk()`。
        这里只处理这类包装器（普通 nn.Module 会被跳过），所以不会误伤正在用的模块；
        被退回的权重下次用到时会自己从硬盘读回来。

        背景：2026-10-02 实测，材质阶段是在 8GB 卡上唯一越界的阶段 ——
        上一步的 HR 形状权重还占着显存时它就要 DINOv3+NAF 在 1024² 上做邻域注意力，
        于是撞到驱动层 `dxgkio_make_resident: Ioctl failed: -12`（ENOMEM），
        表现就是「util 100% 但功耗只有 60~70W」的颠簸，严重时把宿主机拖死。
        """
        if not self.low_vram or not torch.cuda.is_available():
            return
        keep_ids = {id(m) for m in keep if m is not None}

        named = []
        models = getattr(self, "models", None)
        if isinstance(models, dict):
            named.extend(models.items())
        for name in dir(self):
            if name.startswith("image_cond_model"):
                named.append((name, getattr(self, name, None)))
        named.append(("rembg_model", getattr(self, "rembg_model", None)))

        before_mb, _total = self._cuda_memory_mb()
        unloaded = []
        seen = set()
        for name, module in named:
            if module is None or id(module) in keep_ids or id(module) in seen:
                continue
            seen.add(id(module))
            if getattr(module, "loaded_model", None) is None:
                continue  # 本来就不在，无需处理
            unload = getattr(module, "unload_to_disk", None)
            if unload is None:
                continue
            try:
                unload()
                unloaded.append(name)
            except Exception as exc:  # noqa: BLE001 - 卸载失败不该中断生成
                print(f"[VRAM] 卸载 {name} 失败（已忽略）: {exc!r}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        after_mb, _total = self._cuda_memory_mb()
        print(
            f"[VRAM] 按需回收闲置权重{('（' + label + '）') if label else ''}: "
            f"腾出 {after_mb - before_mb:.0f}MiB，"
            f"可用 {before_mb:.0f}→{after_mb:.0f}MiB，"
            f"退回={unloaded if unloaded else '无'}",
            flush=True,
        )

    def preprocess_image(self, input: Image.Image, bg_color: tuple = (0, 0, 0)) -> Image.Image:
        """
        Preprocess the input image.

        Args:
            input: Input image (RGB or RGBA).
            bg_color: Background color (R, G, B) in 0~255. Default black (0,0,0).
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == 'RGBA':
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        if has_alpha:
            output = input
        else:
            input = input.convert('RGB')
            if self.low_vram:
                self.rembg_model.to(self.device)
            output = self.rembg_model(input)
            if self.low_vram:
                self._unload_module(self.rembg_model)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.1)
        bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
        output = output.crop(bbox)  # type: ignore
        output = np.array(output).astype(np.float32) / 255
        rgb = output[:, :, :3]
        a = output[:, :, 3:4]
        bg = np.array(bg_color, dtype=np.float32) / 255.0
        output = rgb * a + bg * (1.0 - a)
        output = Image.fromarray((np.clip(output, 0, 1) * 255).astype(np.uint8))
        return output

    # =========================================================================
    # Proj mode condition building
    # =========================================================================

    @torch.no_grad()
    def get_proj_cond_ss(
        self,
        image: list,
        camera_angle_x: float = 0.8575560450553894,
        distance: float = 2.0,
        mesh_scale: float = 1.0,
    ) -> dict:
        """
        Get proj conditioning for sparse structure stage.

        Args:
            image: List of PIL images.
            camera_angle_x: Camera horizontal FOV in radians.
            distance: Camera distance.
            mesh_scale: Mesh scale.

        Returns:
            dict with 'cond' and 'neg_cond', each containing {'global': ..., 'proj': ...}
        """
        device = self.device
        image_cond_model = self.image_cond_model_ss
        if self.low_vram:
            self._load_module_adaptive(image_cond_model, "DINO 条件 ss")
        cam_angle = torch.tensor([camera_angle_x], device=device)
        dist_tensor = torch.tensor([distance], device=device)
        scale_tensor = torch.tensor([mesh_scale], device=device)
        z_global, z_proj = image_cond_model(
            image, camera_angle_x=cam_angle, distance=dist_tensor, mesh_scale=scale_tensor,
        )
        if self.low_vram:
            self._unload_module(image_cond_model)
        return {
            'cond': {'global': z_global, 'proj': z_proj},
            'neg_cond': {'global': torch.zeros_like(z_global), 'proj': torch.zeros_like(z_proj)},
        }

    @torch.no_grad()
    def get_proj_cond_shape(
        self,
        image_cond_model: nn.Module,
        image: list,
        coords: torch.Tensor,
        camera_angle_x: float = 0.8575560450553894,
        distance: float = 2.0,
        mesh_scale: float = 1.0,
        grid_resolution_override: int = None,
    ) -> dict:
        """
        Get proj conditioning for shape/texture stages (sparse-token aligned).

        Args:
            image_cond_model: The proj image cond model for this stage.
            image: List of PIL images.
            coords: Sparse structure coordinates [N, 4] (batch_idx, x, y, z).
            camera_angle_x: Camera horizontal FOV in radians.
            distance: Camera distance.
            mesh_scale: Mesh scale.
            grid_resolution_override: Override the grid resolution if not None.

        Returns:
            dict with 'cond' and 'neg_cond', each containing {'global': ..., 'proj': SparseTensor}
        """
        device = self.device
        if self.low_vram:
            self._load_module_adaptive(image_cond_model, "DINO 条件 shape/texture")

        orig_grid_res = image_cond_model.grid_resolution
        if grid_resolution_override is not None and grid_resolution_override != orig_grid_res:
            image_cond_model.grid_resolution = grid_resolution_override
            image_cond_model.proj_grid = image_cond_model.proj_grid.__class__(
                grid_resolution=grid_resolution_override,
                image_resolution=image_cond_model.proj_grid.image_resolution,
            ).to(device)

        B = 1
        cam_angle = torch.tensor([camera_angle_x], device=device)
        dist_tensor = torch.tensor([distance], device=device)
        scale_tensor = torch.tensor([mesh_scale], device=device)
        grid_res = image_cond_model.grid_resolution
        batch_indices = coords[:, 0].long()
        x_coords = coords[:, 1].long()
        y_coords = coords[:, 2].long()
        z_coords = coords[:, 3].long()
        # The denoiser only needs projected features at sparse-token
        # coordinates. Ask the extractor for those grid points directly
        # instead of materializing the full 64^3 projection tensor.
        grid_indices = (x_coords * grid_res + y_coords) * grid_res + z_coords
        z_global, z_proj = image_cond_model(
            image,
            camera_angle_x=cam_angle,
            distance=dist_tensor,
            mesh_scale=scale_tensor,
            grid_indices=grid_indices,
        )
        # The extractor returns [B, N, C]; this pipeline builds one-object
        # sparse conditions, so remove the singleton batch dimension here.
        z_proj_sparse = z_proj[0] if z_proj.ndim == 3 and z_proj.shape[0] == B else z_proj
        z_proj_st = SparseTensor(feats=z_proj_sparse, coords=coords)

        if grid_resolution_override is not None and grid_resolution_override != orig_grid_res:
            image_cond_model.grid_resolution = orig_grid_res
            image_cond_model.proj_grid = image_cond_model.proj_grid.__class__(
                grid_resolution=orig_grid_res,
                image_resolution=image_cond_model.proj_grid.image_resolution,
            ).to(device)

        if self.low_vram:
            self._unload_module(image_cond_model)
        return {
            'cond': {'global': z_global, 'proj': z_proj_st},
            'neg_cond': {'global': torch.zeros_like(z_global), 'proj': SparseTensor(feats=torch.zeros_like(z_proj_sparse), coords=coords)},
        }

    # =========================================================================
    # Sampling methods (consistent with Trellis2)
    # =========================================================================

    @staticmethod
    def _require_finite(value, stage: str) -> None:
        """Fail early instead of turning a bad latent into an empty mesh."""
        import os

        if not os.environ.get("PIXAL3D_VALIDATE_LATENTS", "1") == "1":
            return
        feats = value.feats if hasattr(value, "feats") else value
        if not torch.is_tensor(feats):
            return
        if not torch.isfinite(feats).all():
            bad = int((~torch.isfinite(feats)).sum().item())
            raise FloatingPointError(
                f"Pixal3D {stage} produced {bad} non-finite latent values. "
                "Try the fp32 model directory, a supported GPU, or lower the "
                "sampling resolution before exporting."
            )

    def sample_sparse_structure(
        self,
        cond: dict,
        resolution: int,
        num_samples: int = 1,
        sampler_params: dict = {},
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            resolution (int): The resolution of the sparse structure.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample sparse structure latent
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        in_channels = flow_model.in_channels
        noise = torch.randn(num_samples, in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        if self.low_vram:
            self._load_module_adaptive(flow_model, "sparse flow")
        z_s = self.sparse_structure_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling sparse structure (proj)",
        ).samples
        if self.low_vram:
            self._unload_module(flow_model)
        
        # Decode sparse structure latent
        decoder = self.models['sparse_structure_decoder']
        if self.low_vram:
            self._load_module_adaptive(decoder, "sparse decoder")
        # ── 临时诊断：看 z_s 和 decoder 输出的分布 ──────────
        try:
            import os as _os
            if _os.environ.get("PIXAL3D_DIAG"):
                print(f"[DIAG] z_s: shape={tuple(z_s.shape)} dtype={z_s.dtype} "
                      f"device={z_s.device}", flush=True)
                print(f"[DIAG] z_s stats: min={z_s.min().item():.4f} "
                      f"max={z_s.max().item():.4f} mean={z_s.mean().item():.4f} "
                      f"std={z_s.std().item():.4f}", flush=True)
                print(f"[DIAG] z_s 有限值: {torch.isfinite(z_s).all().item()}", flush=True)
        except Exception as _e:
            print(f"[DIAG] z_s 统计失败: {_e}", flush=True)
        _raw = decoder(z_s)
        try:
            import os as _os
            if _os.environ.get("PIXAL3D_DIAG"):
                print(f"[DIAG] decoder(z_s): min={_raw.min().item():.4f} "
                      f"max={_raw.max().item():.4f} mean={_raw.mean().item():.4f}", flush=True)
                print(f"[DIAG] >0 的比例: {(_raw > 0).float().mean().item():.6f}", flush=True)
        except Exception as _e:
            print(f"[DIAG] decoder 统计失败: {_e}", flush=True)
        decoded = _raw > 0
        if self.low_vram:
            self._unload_module(decoder)
        if resolution != decoded.shape[2]:
            ratio = decoded.shape[2] // resolution
            decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()

        # ── 临时诊断：sparse 阶段最终给出的坐标数 ────────────
        try:
            import os as _os
            if _os.environ.get("PIXAL3D_DIAG"):
                print(f"[DIAG] sparse coords: shape={tuple(coords.shape)} "
                      f"num_voxels={coords.shape[0]}", flush=True)
                if coords.shape[0] > 0:
                    print(f"[DIAG] coords 范围: min={coords[:,1:].min().item()} "
                          f"max={coords[:,1:].max().item()}", flush=True)
        except Exception as _e:
            print(f"[DIAG] coords 统计失败: {_e}", flush=True)

        return coords

    def sample_shape_slat(
        self,
        cond: dict,
        flow_model,
        coords: torch.Tensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )

        # ── 临时诊断：看 cond（投影条件）和 noise 是否有 NaN ──
        try:
            import os as _os
            if _os.environ.get("PIXAL3D_DIAG"):
                _n = noise.feats
                print(f"[DIAG-COND] noise: nan={torch.isnan(_n).sum().item()} "
                      f"min={_n.min().item():.4f} max={_n.max().item():.4f}", flush=True)
                def _walk(_obj, _path, _depth=0):
                    if _depth > 4:
                        return
                    if torch.is_tensor(_obj):
                        print(f"[DIAG-COND] {_path}: shape={tuple(_obj.shape)} "
                              f"dtype={_obj.dtype} "
                              f"nan={torch.isnan(_obj).sum().item()} "
                              f"inf={torch.isinf(_obj).sum().item()} "
                              f"min={_obj.min().item():.4f} max={_obj.max().item():.4f}",
                              flush=True)
                    elif hasattr(_obj, 'feats') and hasattr(_obj, 'coords'):
                        _f = _obj.feats
                        print(f"[DIAG-COND] {_path} (Sparse): coords={_obj.coords.shape[0]} "
                              f"nan={torch.isnan(_f).sum().item()} "
                              f"inf={torch.isinf(_f).sum().item()} "
                              f"min={_f.min().item():.4f} max={_f.max().item():.4f}",
                              flush=True)
                    elif isinstance(_obj, dict):
                        for _k, _v in _obj.items():
                            _walk(_v, f"{_path}.{_k}", _depth + 1)
                    elif isinstance(_obj, (list, tuple)):
                        for _i, _v in enumerate(_obj):
                            _walk(_v, f"{_path}[{_i}]", _depth + 1)
                    else:
                        print(f"[DIAG-COND] {_path}: type={type(_obj).__name__}", flush=True)

                _walk(cond, "cond")
        except Exception as _e:
            print(f"[DIAG-COND] 失败: {_e}", flush=True)

        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        self._check_sparse_stage_budget(flow_model, coords.shape[0], "shape LR")
        if self.low_vram:
            self._load_module_adaptive(flow_model, "shape flow")
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling shape SLat (proj)",
        ).samples
        if self.low_vram:
            self._unload_module(flow_model)
        self._require_finite(slat, "shape SLat")

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean

        # ── 临时诊断：LR shape SLat 的输出 ───────────────────
        try:
            import os as _os
            if _os.environ.get("PIXAL3D_DIAG"):
                print(f"[DIAG] sample_shape_slat 输出: "
                      f"coords={tuple(slat.coords.shape)} "
                      f"feats={tuple(slat.feats.shape)}", flush=True)
        except Exception as _e:
            print(f"[DIAG] sample_shape_slat 统计失败: {_e}", flush=True)

        return slat
    
    def sample_shape_slat_cascade(
        self,
        lr_cond: dict,
        cond: dict,
        flow_model_lr,
        flow_model,
        lr_resolution: int,
        resolution: int,
        coords: torch.Tensor,
        sampler_params: dict = {},
        max_num_tokens: int = 49152,
    ) -> SparseTensor:
        """
        Sample structured latent with cascade (LR → HR).
        
        Args:
            lr_cond (dict): The conditioning information for LR stage.
            cond (dict): The conditioning information for HR stage.
            flow_model_lr: LR flow model.
            flow_model: HR flow model.
            lr_resolution (int): LR resolution.
            resolution (int): Target HR resolution.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
            max_num_tokens (int): Maximum number of tokens.
        """
        # LR
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model_lr.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            self._load_module_adaptive(flow_model_lr, "shape LR flow")
        slat = self.shape_slat_sampler.sample(
            flow_model_lr,
            noise,
            **lr_cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling LR shape SLat (proj, 512)",
        ).samples
        if self.low_vram:
            self._unload_module(flow_model_lr)
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        # Upsample
        if self.low_vram:
            self._load_module_adaptive(
                self.models['shape_slat_decoder'], "shape decoder"
            )
            self.models['shape_slat_decoder'].low_vram = True
        hr_coords = self.models['shape_slat_decoder'].upsample(slat, upsample_times=4)
        if self.low_vram:
            self._unload_module(self.models['shape_slat_decoder'])
            self.models['shape_slat_decoder'].low_vram = False
        hr_resolution = resolution
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens or hr_resolution == 1024:
                if hr_resolution != resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                break
            hr_resolution -= 128
        
        # Sample structured latent (HR)
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}
        if self.low_vram:
            self._load_module_adaptive(flow_model, "shape HR flow")
        slat = self.shape_slat_sampler.sample(
            flow_model,
            noise,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc=f"Sampling HR shape SLat (proj, {hr_resolution})",
        ).samples
        if self.low_vram:
            self._unload_module(flow_model)

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean

        # ── 临时诊断：看 shape SLat 的稀疏结构 ──────────────
        try:
            import os as _os
            if _os.environ.get("PIXAL3D_DIAG"):
                print(f"[DIAG] shape slat: coords={slat.coords.shape} "
                      f"feats={slat.feats.shape}", flush=True)
                print(f"[DIAG] shape slat feats: min={slat.feats.min().item():.4f} "
                      f"max={slat.feats.max().item():.4f} "
                      f"mean={slat.feats.mean().item():.4f}", flush=True)
                print(f"[DIAG] shape slat 有限值: {torch.isfinite(slat.feats).all().item()}",
                      flush=True)
        except Exception as _e:
            print(f"[DIAG] shape slat 统计失败: {_e}", flush=True)

        return slat, hr_resolution

    def decode_shape_slat(
        self,
        slat: SparseTensor,
        resolution: int,
    ) -> Tuple[List[Mesh], List[SparseTensor]]:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            List[Mesh]: The decoded meshes.
            List[SparseTensor]: The decoded substructures.
        """
        self.models['shape_slat_decoder'].set_resolution(resolution)
        if self.low_vram:
            self._load_module_adaptive(
                self.models['shape_slat_decoder'], "shape decoder"
            )
            self.models['shape_slat_decoder'].low_vram = True
        ret = self.models['shape_slat_decoder'](slat, return_subs=True)
        if self.low_vram:
            self._unload_module(self.models['shape_slat_decoder'])
            self.models['shape_slat_decoder'].low_vram = False
        return ret
    
    def sample_tex_slat(
        self,
        cond: dict,
        flow_model,
        shape_slat: SparseTensor,
        sampler_params: dict = {},
    ) -> SparseTensor:
        """
        Sample texture structured latent with the given conditioning.
        
        Args:
            cond (dict): The conditioning information.
            shape_slat (SparseTensor): The structured latent for shape.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample structured latent
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat = (shape_slat - mean) / std

        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise = shape_slat.replace(feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device))
        sampler_params = {**self.tex_slat_sampler_params, **sampler_params}
        self._check_sparse_stage_budget(
            flow_model, shape_slat.coords.shape[0], "texture"
        )
        if self.low_vram:
            self._load_module_adaptive(flow_model, "texture flow")
        slat = self.tex_slat_sampler.sample(
            flow_model,
            noise,
            concat_cond=shape_slat,
            **cond,
            **sampler_params,
            verbose=True,
            tqdm_desc="Sampling texture SLat (proj)",
        ).samples
        if self.low_vram:
            self._unload_module(flow_model)

        std = torch.tensor(self.tex_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        
        return slat

    def decode_tex_slat(
        self,
        slat: SparseTensor,
        subs: List[SparseTensor],
    ) -> SparseTensor:
        """
        Decode the structured latent.

        Args:
            slat (SparseTensor): The structured latent.

        Returns:
            SparseTensor: The decoded texture voxels
        """
        if self.low_vram:
            self._load_module_adaptive(
                self.models['tex_slat_decoder'], "texture decoder"
            )
        ret = self.models['tex_slat_decoder'](slat, guide_subs=subs) * 0.5 + 0.5
        if self.low_vram:
            self._unload_module(self.models['tex_slat_decoder'])
        return ret
    
    @torch.no_grad()
    def decode_latent(
        self,
        shape_slat: SparseTensor,
        tex_slat: SparseTensor,
        resolution: int,
    ) -> List[MeshWithVoxel]:
        """
        Decode the latent codes.

        Args:
            shape_slat (SparseTensor): The structured latent for shape.
            tex_slat (SparseTensor): The structured latent for texture.
            resolution (int): The resolution of the output.
        """
        meshes, subs = self.decode_shape_slat(shape_slat, resolution)
        tex_voxels = self.decode_tex_slat(tex_slat, subs)
        out_mesh = []
        torch.cuda.synchronize()
        for m, v in zip(meshes, tex_voxels):
            m.fill_holes()
            out_mesh.append(
                MeshWithVoxel(
                    m.vertices, m.faces,
                    origin = [-0.5, -0.5, -0.5],
                    voxel_size = 1 / resolution,
                    coords = v.coords[:, 1:],
                    attrs = v.feats,
                    voxel_shape = torch.Size([*v.shape, *v.spatial_shape]),
                    layout=self.pbr_attr_layout
                )
            )
        return out_mesh
    
    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        camera_params: dict,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        return_latent: bool = False,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
    ) -> List[MeshWithVoxel]:
        """
        Run the Pixal3D pipeline (proj mode, cascade).

        Args:
            image (Image.Image): The image prompt.
            camera_params (dict): Camera parameters with keys:
                - camera_angle_x (float): Horizontal FOV in radians.
                - distance (float): Camera distance.
                - mesh_scale (float): Mesh scale factor.
            num_samples (int): The number of samples to generate.
            seed (int): The random seed.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            shape_slat_sampler_params (dict): Additional parameters for the shape SLat sampler.
            tex_slat_sampler_params (dict): Additional parameters for the texture SLat sampler.
            preprocess_image (bool): Whether to preprocess the image.
            return_latent (bool): Whether to return the latent codes.
            pipeline_type (str): The type of the pipeline. Options: '1024_cascade', '1536_cascade'.
            max_num_tokens (int): The maximum number of tokens to use.
        """
        # Check pipeline type
        pipeline_type = pipeline_type or self.default_pipeline_type
        if pipeline_type == '1024_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
            hr_resolution = 1024
        elif pipeline_type == '1536_cascade':
            assert 'shape_slat_flow_model_512' in self.models, "No 512 resolution shape SLat flow model found."
            assert 'shape_slat_flow_model_1024' in self.models, "No 1024 resolution shape SLat flow model found."
            assert 'tex_slat_flow_model_1024' in self.models, "No 1024 resolution texture SLat flow model found."
            hr_resolution = 1536
        else:
            raise ValueError(f"Invalid pipeline type for Pixal3D proj mode: {pipeline_type}. "
                             f"Supported: '1024_cascade', '1536_cascade'.")

        # Validate image_cond_models are set
        assert self.image_cond_model_ss is not None, "image_cond_model_ss not set."
        assert self.image_cond_model_shape_512 is not None, "image_cond_model_shape_512 not set."
        assert self.image_cond_model_shape_1024 is not None, "image_cond_model_shape_1024 not set."
        assert self.image_cond_model_tex_1024 is not None, "image_cond_model_tex_1024 not set."

        # Extract camera params
        camera_angle_x = camera_params['camera_angle_x']
        distance = camera_params['distance']
        mesh_scale = camera_params.get('mesh_scale', 1.0)
        
        if preprocess_image:
            image = self.preprocess_image(image)
        torch.manual_seed(seed)

        # ---- Stage 1: Sparse Structure (proj) ----
        cond_ss = self.get_proj_cond_ss(
            [image],
            camera_angle_x=camera_angle_x,
            distance=distance,
            mesh_scale=mesh_scale,
        )
        ss_res = 32
        coords = self.sample_sparse_structure(
            cond_ss, ss_res,
            num_samples, sparse_structure_sampler_params
        )
        del cond_ss
        torch.cuda.empty_cache()

        # ---- Stage 2: Shape LR 512 (proj) ----
        cond_shape_lr = self.get_proj_cond_shape(
            self.image_cond_model_shape_512, [image], coords,
            camera_angle_x=camera_angle_x,
            distance=distance,
            mesh_scale=mesh_scale,
        )
        lr_slat = self.sample_shape_slat(
            cond_shape_lr, self.models['shape_slat_flow_model_512'],
            coords, shape_slat_sampler_params
        )
        del cond_shape_lr
        torch.cuda.empty_cache()

        # ---- Stage 3a: Upsample LR → HR ----
        if self.low_vram:
            self._load_module_adaptive(
                self.models['shape_slat_decoder'], "shape decoder"
            )
            self.models['shape_slat_decoder'].low_vram = True
        hr_coords = self.models['shape_slat_decoder'].upsample(lr_slat, upsample_times=4)
        if self.low_vram:
            self._unload_module(self.models['shape_slat_decoder'])
            self.models['shape_slat_decoder'].low_vram = False

        lr_resolution = 512
        actual_hr_resolution = hr_resolution
        min_hr_resolution = max(
            512, int(os.environ.get("PIXAL3D_MIN_HR_RESOLUTION", "512"))
        )
        hr_flow_model = self.models['shape_slat_flow_model_1024']
        token_limit = self._adaptive_sparse_token_limit(
            hr_flow_model, max_num_tokens
        )
        while True:
            grid_res = actual_hr_resolution // 16
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / lr_resolution * (grid_res - 1)).round().int(),
            ], dim=1)
            hr_coords_unique = quant_coords.unique(dim=0)
            num_tokens = hr_coords_unique.shape[0]
            if num_tokens <= token_limit or actual_hr_resolution <= min_hr_resolution:
                break
            next_resolution = max(min_hr_resolution, actual_hr_resolution - 128)
            print(
                f"[VRAM] HR tokens={num_tokens} 超过上限 {token_limit}，"
                f"将分辨率从 {actual_hr_resolution} 降到 {next_resolution}",
                flush=True,
            )
            actual_hr_resolution = next_resolution

        if num_tokens > token_limit:
            print(
                f"[VRAM][WARN] 最低 HR 分辨率 {actual_hr_resolution} 仍有 "
                f"{num_tokens} tokens（上限 {token_limit}）；将执行模型加载前的"
                "激活显存预检。",
                flush=True,
            )

        actual_grid_res = actual_hr_resolution // 16
        del lr_slat, hr_coords, quant_coords
        torch.cuda.empty_cache()

        # ---- Stage 3b: Shape HR (proj) ----
        cond_shape_hr = self.get_proj_cond_shape(
            self.image_cond_model_shape_1024, [image], hr_coords_unique,
            camera_angle_x=camera_angle_x,
            distance=distance,
            mesh_scale=mesh_scale,
            grid_resolution_override=actual_grid_res,
        )
        noise_hr = SparseTensor(
            feats=torch.randn(hr_coords_unique.shape[0], hr_flow_model.in_channels).to(self.device),
            coords=hr_coords_unique,
        )
        sampler_params_hr = {**self.shape_slat_sampler_params, **shape_slat_sampler_params}
        flow_model_hr = hr_flow_model
        self._check_sparse_stage_budget(
            flow_model_hr, hr_coords_unique.shape[0], "shape HR"
        )
        if self.low_vram:
            self._load_module_adaptive(flow_model_hr, "shape HR flow")
        hr_slat = self.shape_slat_sampler.sample(
            flow_model_hr,
            noise_hr,
            **cond_shape_hr,
            **sampler_params_hr,
            verbose=True,
            tqdm_desc=f"Sampling HR shape SLat (proj, {actual_hr_resolution})",
        ).samples
        if self.low_vram:
            self._unload_module(flow_model_hr)
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(hr_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(hr_slat.device)
        shape_slat = hr_slat * std + mean
        self._require_finite(shape_slat, "HR shape SLat")
        del cond_shape_hr, noise_hr, hr_slat, hr_coords_unique
        torch.cuda.empty_cache()

        # ---- Stage 4: Texture (proj) ----
        # 按需给显存：材质阶段要在 1024² 上跑 DINOv3+NAF，是 8GB 卡上唯一越界的
        # 阶段。先把上一步（HR 形状）遗留的权重全部退回，再让它按需读回来。
        self._evict_idle_modules(label="进入材质阶段")
        tex_grid_res = actual_hr_resolution // 16
        cond_tex = self.get_proj_cond_shape(
            self.image_cond_model_tex_1024, [image], shape_slat.coords,
            camera_angle_x=camera_angle_x,
            distance=distance,
            mesh_scale=mesh_scale,
            grid_resolution_override=tex_grid_res,
        )
        tex_slat = self.sample_tex_slat(
            cond_tex, self.models['tex_slat_flow_model_1024'],
            shape_slat, tex_slat_sampler_params
        )
        self._require_finite(tex_slat, "texture SLat")
        del cond_tex
        torch.cuda.empty_cache()

        # ---- Stage 5: Decode ----
        # 同理：解码只需要 shape/tex 两个 decoder，其余权重先退回，腾出空间。
        self._evict_idle_modules(
            keep=(self.models.get('shape_slat_decoder'),
                  self.models.get('tex_slat_decoder')),
            label="进入解码阶段",
        )
        res = actual_hr_resolution
        out_mesh = self.decode_latent(shape_slat, tex_slat, res)
        if return_latent:
            return out_mesh, (shape_slat, tex_slat, res)
        else:
            return out_mesh
