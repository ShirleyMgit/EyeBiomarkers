from __future__ import annotations

from PIL import Image
import torch
from torchvision import transforms

from .config import ReproConfig


class EyePreprocessor:
    """Turns a cropped eye image into a model-input tensor.

    Matches the authors' released pipeline: convert to RGB and ToTensor (values in [0, 1]),
    with no resize and no normalization by default. The model pads the image to 192x192
    internally, so the raw ~16x32 crop is fed as-is. Resize/normalize are available for
    experiments but off by default because the deployed checkpoints were trained without them.
    """

    def __init__(
        self,
        img_size: tuple[int, int] | None = None,
        normalize: bool = False,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        img_mode: str = "RGB",
    ):
        self._img_mode = img_mode
        steps = []
        if img_size is not None:
            steps.append(
                transforms.Resize(
                    (img_size[0], img_size[-1]),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                )
            )
        steps.append(transforms.ToTensor())
        if normalize:
            steps.append(transforms.Normalize(mean=mean, std=std))
        self._transform = transforms.Compose(steps)

    @classmethod
    def from_config(cls, config: ReproConfig) -> "EyePreprocessor":
        return cls(config.img_size, config.normalize, config.imagenet_mean, config.imagenet_std, config.img_mode)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self._transform(img.convert(self._img_mode))
