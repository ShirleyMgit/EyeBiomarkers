from __future__ import annotations

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
    state_dict = torch.load(weights_path, map_location=device)
    filtered = {k: v for k, v in state_dict.items() if not k.startswith("resnet.fc")}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if unexpected:
        print(f"Warning: unexpected keys in checkpoint: {unexpected}")
    model.to(device)
    model.eval()
    return model
