from __future__ import annotations

import torch.nn as nn

try:
    import timm  # type: ignore
except Exception as exc:  # pragma: no cover
    timm = None
    _TIMM_IMPORT_ERROR = exc


class ViTProbe(nn.Module):
    def __init__(self, model_name: str, pretrained: bool, num_classes: int, drop_rate: float = 0.0, drop_path_rate: float = 0.1):
        super().__init__()
        self.model_name = model_name
        if timm is None or not hasattr(timm, "create_model"):
            raise ImportError(
                "timm is required for TGO-IV. Install timm and use the ViT-S/16 model specified in the configuration."
            ) from _TIMM_IMPORT_ERROR
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x):
        return self.model(x)

    @property
    def vit(self):
        return self.model
