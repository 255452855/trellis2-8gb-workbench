# File: trellis2/pipelines/base.py
# trellis2/pipelines/base.py
from typing import *
import torch
import torch.nn as nn
from .. import models


class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        config_file: str = "pipeline.json",
        lazy_models: bool = False,
    ) -> "Pipeline":
        """
        Load a pretrained model.

        Mirrors the Pixal3D codebase's base class signature so subclasses ported
        from there (Pixal3DImageTo3DPipeline, Trellis2TexturingPipeline) can call
        `super().from_pretrained(path, config_file)` and still get back an
        instance of their own class (needed for `run()` / `preprocess_image()`
        to exist on the returned object).
        """
        import os
        import json
        is_local = os.path.exists(f"{path}/{config_file}")

        if is_local:
            config_file = f"{path}/{config_file}"
        else:
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(path, config_file)

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        _models = {}
        for k, v in args['models'].items():
            # Honor model_names_to_load if the concrete pipeline declares it.
            needed = getattr(cls, 'model_names_to_load', None)
            if needed is not None and k not in needed:
                continue
            model_path = v if os.path.isabs(v) else os.path.join(path, v)
            try:
                _models[k] = models.from_pretrained(
                    model_path,
                    lazy=lazy_models,
                )
            except Exception as e:
                if lazy_models:
                    raise RuntimeError(
                        f"本地模型懒加载失败 {model_path}: {type(e).__name__}: {e}"
                    ) from e
                _models[k] = models.from_pretrained(model_path)

        new_pipeline = cls(_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    @property
    def device(self) -> torch.device:
        if hasattr(self, '_device'):
            return self._device
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
        for model in self.models.values():
            if hasattr(model, 'parameters'):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))
