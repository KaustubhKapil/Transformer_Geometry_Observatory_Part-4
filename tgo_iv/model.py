
from __future__ import annotations

import warnings

import torch.nn as nn

try:
    import timm  # type: ignore
except Exception:  # pragma: no cover
    timm = None

try:
    from torchvision.models import vit_b_16, ViT_B_16_Weights  # type: ignore
except Exception:  # pragma: no cover
    vit_b_16 = None
    ViT_B_16_Weights = None


class ViTProbe(nn.Module):
    def __init__(self, model_name: str, pretrained: bool, num_classes: int, drop_rate: float = 0.0, drop_path_rate: float = 0.1):
        super().__init__()
        self.model_name = model_name

        if timm is not None and hasattr(timm, "create_model"):
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
            )
        elif vit_b_16 is not None:
            if model_name not in {"vit_b_16", "vit_b16", "vit_base_patch16_224"}:
                warnings.warn(
                    f"timm is unavailable; falling back to torchvision vit_b_16 for '{model_name}'. "
                    "Install timm to use ViT-S/16 as specified in the paper."
                )
            weights = ViT_B_16_Weights.DEFAULT if pretrained and ViT_B_16_Weights is not None else None
            self.model = vit_b_16(weights=weights, num_classes=num_classes, dropout=drop_rate)
        else:
            raise ImportError("Neither timm nor torchvision ViT implementations are available.")

    def forward(self, x):
        return self.model(x)

    @property
    def vit(self):
        return self.model
