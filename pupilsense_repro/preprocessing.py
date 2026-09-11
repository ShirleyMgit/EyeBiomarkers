from __future__ import annotations

from PIL import Image
import torch
from torchvision import transforms

from .config import ReproConfig


class EyePreprocessor:
    """Turns a cropped eye image into the paper's 224x224 model input tensor."""

    def __init__(
        self,
        resize_hw: tuple[int, int] = (32, 64),
        target_size: int = 224,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        height, width = resize_hw
        pad_h = (target_size - height) // 2
        pad_w = (target_size - width) // 2
        self._transform = transforms.Compose([
            transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Pad((pad_w, pad_h, pad_w, pad_h), fill=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    @classmethod
    def from_config(cls, config: ReproConfig) -> EyePreprocessor:
        return cls(config.resize_hw, config.target_size, config.imagenet_mean, config.imagenet_std)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self._transform(img.convert("RGB"))
