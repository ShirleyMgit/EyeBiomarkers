from dataclasses import dataclass, field
from pathlib import Path

FOLD_TEST_PARTICIPANTS: dict[int, list[int]] = {
    1: [1, 4, 6, 25, 36],
    2: [2, 5, 7, 26, 37],
    3: [3, 16, 26, 38, 43],
    4: [9, 19, 29, 39, 49],
    5: [10, 14, 24, 28, 31],
}

PAPER_MAPE: dict[str, float] = {"resnet18": 3.411629, "resnet50": 3.234711}

_BASE_TO_DIR = {"resnet18": "ResNet18", "resnet50": "ResNet50"}


@dataclass(frozen=True)
class ReproConfig:
    data_root: Path
    weights_dir: Path
    device: str = "cpu"
    # Model input pipeline — matches the authors' released config: no resize, no normalization.
    # The model pads the image to 192x192 internally, so preprocessing is just ToTensor on the RGB crop.
    img_size: tuple[int, int] | None = None
    normalize: bool = False
    imagenet_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    img_mode: str = "RGB"
    batch_size: int = 128
    num_workers: int = 0
    eye: str = "left"
    results_dir: Path = field(default_factory=lambda: Path("results"))
    figures_dir: Path = field(default_factory=lambda: Path("figures"))


def weights_path_for(config: ReproConfig, base: str) -> Path:
    return config.weights_dir / _BASE_TO_DIR[base] / f"{config.eye}_eye.pt"


def target_column(config: ReproConfig) -> str:
    return f"{config.eye}_pupil"
