# File: trellis2/models/__init__.py
# trellis2/models/__init__.py
import importlib
import torch
import torch.nn as nn

__attributes = {
    # Sparse Structure
    'SparseStructureEncoder': 'sparse_structure_vae',
    'SparseStructureDecoder': 'sparse_structure_vae',
    'SparseStructureFlowModel': 'sparse_structure_flow',
    
    # SLat Generation
    'SLatFlowModel': 'structured_latent_flow',
    'ElasticSLatFlowModel': 'structured_latent_flow',
    
    # SC-VAEs
    'SparseUnetVaeEncoder': 'sc_vaes.sparse_unet_vae',
    'SparseUnetVaeDecoder': 'sc_vaes.sparse_unet_vae',
    'FlexiDualGridVaeEncoder': 'sc_vaes.fdg_vae',
    'FlexiDualGridVaeDecoder': 'sc_vaes.fdg_vae'
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules


class LazyPretrainedModel(nn.Module):
    """Keep a local checkpoint on disk until the model is actually used."""

    def __init__(self, path: str, kwargs: dict):
        import json
        import os

        if not (os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")):
            raise FileNotFoundError(f"本地懒加载模型文件不完整: {path}")
        with open(f"{path}.json", "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        super().__init__()
        object.__setattr__(self, "_lazy_path", path)
        object.__setattr__(self, "_lazy_config", config)
        object.__setattr__(self, "_lazy_kwargs", kwargs)
        object.__setattr__(self, "_lazy_loaded", None)

    def ensure_loaded(self):
        loaded = object.__getattribute__(self, "_lazy_loaded")
        if loaded is not None:
            return loaded
        import os
        import psutil

        checkpoint_path = f"{object.__getattribute__(self, '_lazy_path')}.safetensors"
        checkpoint_mb = os.path.getsize(checkpoint_path) / (1024 ** 2)
        reserve_mb = max(512, int(os.environ.get("MODEL_RAM_RESERVE_MB", "512")))
        factor = float(os.environ.get("MODEL_RAM_FACTOR", "1.02"))
        available_mb = psutil.virtual_memory().available / (1024 ** 2)
        required_mb = checkpoint_mb * factor + reserve_mb
        if available_mb < required_mb:
            raise RuntimeError(
                f"Pixal3D 内存保护：当前可用系统内存 {available_mb:.0f} MiB，"
                f"读盘加载 {checkpoint_mb:.0f} MiB 权重预计需要 {required_mb:.0f} MiB，"
                "已停止本阶段加载，未丢弃输入。"
            )
        config = object.__getattribute__(self, "_lazy_config")
        model_cls = __getattr__(config["name"])
        # ⚠️ 不要在 torch.device("meta") 下构建模型。
        # __init__ 里用 torch.arange / torch.tensor 计算出来的“派生属性”
        # （RoPE 的 freqs、image_feature_extractor 的 _norm_mean/_norm_std 等）
        # 在 meta 设备下会永久停在 meta；to_empty() 只处理注册的 param/buffer，
        # 搬不动这些普通属性，运行时会报：
        #   NotImplementedError: Cannot copy out of meta tensor; no data!
        # 直接正常构建（真实 CPU 张量），派生属性即被正确计算。
        model = model_cls(
            **config["args"],
            **object.__getattribute__(self, "_lazy_kwargs"),
        )

        from safetensors import safe_open

        state = model.state_dict(keep_vars=True)
        missing = set(state)
        with safe_open(
            f"{object.__getattribute__(self, '_lazy_path')}.safetensors",
            framework="pt",
            device="cpu",
        ) as checkpoint:
            with torch.no_grad():
                for key in checkpoint.keys():
                    if key not in state:
                        raise RuntimeError(
                            f"懒加载权重包含未知参数 {key}: "
                            f"{object.__getattribute__(self, '_lazy_path')}"
                        )
                    target = state[key]
                    value = checkpoint.get_tensor(key)
                    if target.shape != value.shape:
                        raise RuntimeError(
                            f"懒加载权重形状不匹配 {key}: "
                            f"模型 {tuple(target.shape)} != 文件 {tuple(value.shape)}"
                        )
                    target.copy_(value)
                    missing.discard(key)
                    del value
        missing_rope_phases = missing == {"rope_phases"}
        if missing_rope_phases:
            missing.clear()
        if missing:
            raise RuntimeError(
                f"懒加载权重缺少参数 {sorted(missing)[:8]}: "
                f"{object.__getattribute__(self, '_lazy_path')}"
            )
        if missing_rope_phases:
            from .sparse_structure_flow import SparseStructureFlowModel
            from ..modules.attention import RotaryPositionEmbedder

            if not isinstance(model, SparseStructureFlowModel):
                raise RuntimeError(
                    f"模型缺少未知派生参数 rope_phases: "
                    f"{object.__getattribute__(self, '_lazy_path')}"
                )
            coords = torch.meshgrid(
                *[
                    torch.arange(model.resolution, device="cpu")
                    for _ in range(3)
                ],
                indexing="ij",
            )
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            embedder = RotaryPositionEmbedder(
                model.model_channels // model.num_heads,
                3,
            )
            rope_phases = embedder(coords)
            model.rope_phases.copy_(rope_phases)
        model.eval()
        if (
            object.__getattribute__(self, "_lazy_config").get("args", {}).get("dtype")
            == "bfloat16"
            and os.environ.get("TRELLIS2_FORCE_FP16", "0") in ("1", "true", "True")
        ):
            for parameter in model.parameters():
                if parameter.dtype == torch.bfloat16:
                    parameter.data = parameter.data.half()
            for buffer in model.buffers():
                if buffer.dtype == torch.bfloat16:
                    buffer.data = buffer.data.half()
            for submodule in model.modules():
                for key, value in vars(submodule).items():
                    if isinstance(value, torch.Tensor) and value.dtype == torch.bfloat16:
                        setattr(submodule, key, value.half())
                    elif value is torch.bfloat16:
                        setattr(submodule, key, torch.float16)
        object.__setattr__(self, "_lazy_loaded", model)
        print(
            f"[MODEL] 按需从硬盘加载: "
            f"{object.__getattribute__(self, '_lazy_path')}",
            flush=True,
        )
        return model

    @property
    def loaded_model(self):
        return object.__getattribute__(self, "_lazy_loaded")

    @property
    def checkpoint_bytes(self):
        import os

        return os.path.getsize(
            f"{object.__getattribute__(self, '_lazy_path')}.safetensors"
        )

    @property
    def config_args(self):
        return object.__getattribute__(self, "_lazy_config")["args"]

    def unload_to_disk(self):
        import gc

        model = object.__getattribute__(self, "_lazy_loaded")
        if model is None:
            return
        model.cpu()
        object.__setattr__(self, "_lazy_loaded", None)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        loaded = object.__getattribute__(self, "_lazy_loaded")
        if loaded is not None:
            loaded.train(mode)
        return self

    def forward(self, *args, **kwargs):
        return self.ensure_loaded()(*args, **kwargs)

    def to(self, *args, **kwargs):
        device = kwargs.get("device")
        if device is None and args:
            device = args[0]
        if device is not None and torch.device(device).type == "cpu":
            loaded = object.__getattribute__(self, "_lazy_loaded")
            if loaded is None:
                return self
            self.unload_to_disk()
            return self
        return self.ensure_loaded().to(*args, **kwargs)

    def cpu(self):
        loaded = object.__getattribute__(self, "_lazy_loaded")
        if loaded is None:
            return self
        self.unload_to_disk()
        return self

    def cuda(self, device=None):
        return self.ensure_loaded().cuda(device)

    def parameters(self, recurse=True):
        loaded = object.__getattribute__(self, "_lazy_loaded")
        return loaded.parameters(recurse=recurse) if loaded is not None else iter(())

    def buffers(self, recurse=True):
        loaded = object.__getattribute__(self, "_lazy_loaded")
        return loaded.buffers(recurse=recurse) if loaded is not None else iter(())

    def __getattr__(self, name):
        if name.startswith("_lazy_"):
            raise AttributeError(name)
        loaded = object.__getattribute__(self, "_lazy_loaded")
        if loaded is None:
            config = object.__getattribute__(self, "_lazy_config")
            args = config.get("args", {})
            if name in args:
                return args[name]
            loaded = self.ensure_loaded()
        return getattr(loaded, name)


def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def from_pretrained(path: str, lazy: bool = False, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        **kwargs: Additional arguments for the model constructor.
    """
    import os
    import json
    from safetensors.torch import load_file
    is_local = os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")

    if is_local:
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors"
    else:
        from huggingface_hub import hf_hub_download
        path_parts = path.split('/')
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])
        config_file = hf_hub_download(repo_id, f"{model_name}.json")
        model_file = hf_hub_download(repo_id, f"{model_name}.safetensors")

    if lazy:
        if not is_local:
            raise RuntimeError("懒加载只支持本地 safetensors 模型。")
        return LazyPretrainedModel(path, kwargs)

    with open(config_file, 'r') as f:
        config = json.load(f)
    model = __getattr__(config['name'])(**config['args'], **kwargs)
    model.load_state_dict(load_file(model_file), strict=False)

    return model


# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import SparseStructureEncoder, SparseStructureDecoder
    from .sparse_structure_flow import SparseStructureFlowModel
    from .structured_latent_flow import SLatFlowModel, ElasticSLatFlowModel
        
    from .sc_vaes.sparse_unet_vae import SparseUnetVaeEncoder, SparseUnetVaeDecoder
    from .sc_vaes.fdg_vae import FlexiDualGridVaeEncoder, FlexiDualGridVaeDecoder
