from __future__ import annotations

import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


class ResNetRegressor(nn.Module):
    """ResNet backbone with the classifier replaced by a single-output regression head."""

    def __init__(self, base: str = "resnet18"):
        super().__init__()
        if base == "resnet18":
            self.resnet = models.resnet18(weights=None)
        elif base == "resnet50":
            self.resnet = models.resnet50(weights=None)
        else:
            raise ValueError("Only resnet18/resnet50 supported")
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)


def load_eye_model(weights_path: str | Path, base: str = "resnet18", device: str = "cpu") -> nn.Module:
    model = ResNetRegressor(base)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    model_state = model.state_dict()
    compatible = {
        key: tensor
        for key, tensor in checkpoint.items()
        if key in model_state and tensor.shape == model_state[key].shape
    }
    dropped = [key for key in checkpoint if key not in compatible]
    missing, _ = model.load_state_dict(compatible, strict=False)
    if dropped:
        warnings.warn(f"Dropped {len(dropped)} checkpoint key(s) absent from or shape-mismatched vs the model: {dropped}")
    if missing:
        warnings.warn(f"{len(missing)} model parameter(s) not found in checkpoint (left at random init): {missing}")
    if any(key.startswith("resnet.fc") for key in missing):
        warnings.warn(
            "Regression head (resnet.fc) was NOT loaded from the checkpoint and remains at random "
            "init - predicted diameters will be meaningless. Verify the checkpoint's key layout."
        )
    model.to(device)
    model.eval()
    return model
