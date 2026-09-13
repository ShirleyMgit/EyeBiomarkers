from __future__ import annotations

import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

_RESNET_BUILDERS = {"resnet18": models.resnet18, "resnet50": models.resnet50}


def _pad_to_square(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Zero-pad a batch of images up to dim x dim, centered (no-op for sides already >= dim)."""
    height, width = x.shape[2], x.shape[3]
    pad_height = max(0, (dim - height) // 2) if height < dim else 0
    pad_width = max(0, (dim - width) // 2) if width < dim else 0
    if pad_height > 0 or pad_width > 0:
        x = F.pad(x, (pad_width, pad_width, pad_height, pad_height), mode="constant", value=0)
    return x


class ResNetRegressor(nn.Module):
    """Pupil-diameter regressor matching the authors' ResNetImageNet.

    The ResNet backbone keeps its original fc (-> 1000) and a separate regression_head
    maps those 1000 features to num_classes. The input is padded to pad_dim x pad_dim
    inside forward (the released models were trained on the raw ~16x32 crop padded to 192).
    """

    def __init__(self, base: str = "resnet18", num_classes: int = 1, pad_dim: int = 224, use_regression_head: bool = True):
        super().__init__()
        if base not in _RESNET_BUILDERS:
            raise ValueError("Only resnet18/resnet50 supported")
        self.pad_dim = pad_dim
        self.use_regression_head = use_regression_head
        self.resnet = _RESNET_BUILDERS[base](weights=None)
        if use_regression_head:
            self.regression_head = nn.Linear(1000, num_classes)
        else:
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_dim:
            x = _pad_to_square(x, self.pad_dim)
        x = self.resnet(x)
        if self.use_regression_head:
            x = self.regression_head(x)
        return x


def load_eye_model(weights_path: str | Path, base: str = "resnet18", device: str = "cpu") -> nn.Module:
    model = ResNetRegressor(base)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    if missing:
        warnings.warn(f"Model parameters not found in checkpoint (left at random init): {missing}")
    if unexpected:
        warnings.warn(f"Unexpected checkpoint keys ignored: {list(unexpected)}")
    model.to(device)
    model.eval()
    return model
