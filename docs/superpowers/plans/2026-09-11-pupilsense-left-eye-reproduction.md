# PupilSense Left-Eye Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable `pupilsense_repro` Python package that runs the authors' released ResNet18/ResNet50 left-eye checkpoints over the EyeDentify dataset and reproduces the paper's MAE/MAPE, including the per-participant MAPE distribution and per-fold breakdown.

**Architecture:** A small, single-responsibility package (config, preprocessing, model, dataset, metrics, evaluate, runner, plots) developed and unit-tested locally with CPU PyTorch, then driven on Colab's GPU by a thin notebook that mounts Google Drive and sets paths. Preprocessing + model + `predict_diameter` are dependency-free of the eval code so the future video pipeline and app import them directly.

**Tech Stack:** Python 3.12, PyTorch + torchvision (CPU locally, GPU on Colab), pandas, Pillow, matplotlib, tqdm, pytest; uv for local env.

## Global Constraints

- Target **Python 3.12** (matches Google Colab).
- Managed with **uv** locally: `uv add` / `uv run` / `uv sync` — never `pip install`. On Colab, torch/torchvision/pandas are preinstalled.
- **Pandas rules (user global):** never call `df.copy()`; never use `inplace=True` (reassign explicitly); never use `.apply()`, `.iterrows()`, `.itertuples()`, `.items()` — vectorize, or use list comprehensions over `zip(...)`. At package import, enable copy-on-write **only on pandas < 3.0** (`if int(pd.__version__.split(".")[0]) < 3: pd.options.mode.copy_on_write = True`) — on pandas ≥ 3.0 CoW is unconditional and setting the option emits a deprecation warning, so it must be guarded.
- **Clean Code (user global):** self-explanatory names; `@dataclass` (frozen where immutable) over boilerplate; classes over loose scripts where state groups; comments only for non-obvious "why".
- **Eye scope:** left eye only; target ground-truth column is `left_pupil`.
- **Preprocessing (paper-exact):** stored PNG is W=32×H=16 → `Resize((32,64), BICUBIC)` → zero-pad to 224×224 → `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`.
- **Weights layout:** `<weights_dir>/ResNet18/left_eye.pt` and `<weights_dir>/ResNet50/left_eye.pt`; state_dict with `resnet.`-prefixed keys, `resnet.fc.*` dropped, loaded `strict=False`.
- **MAPE definition:** `mean(|pred - true| / max(true, 1e-9)) * 100`. **MAE:** `mean(|pred - true|)` in mm.
- **5-fold test participants (paper supplement Table 3):** F1 [1,4,6,25,36]; F2 [2,5,7,26,37]; F3 [3,16,26,38,43]; F4 [9,19,29,39,49]; F5 [10,14,24,28,31].
- **Paper reference (left eye) MAPE:** ResNet18 = 3.411629%, ResNet50 = 3.234711% (LOPOCV).

---

## File Structure

```
EyeBiomarkers/
  pyproject.toml                      # uv project, deps
  pupilsense_repro/
    __init__.py                       # sets pandas CoW; exports public API
    config.py                         # ReproConfig, FOLD_TEST_PARTICIPANTS, PAPER_MAPE, weights_path_for
    preprocessing.py                  # EyePreprocessor
    model.py                          # ResNetRegressor, load_eye_model
    dataset.py                        # build_index, EyeDentifyDataset
    metrics.py                        # add_errors, overall_metrics, per_participant_mape, per_fold_metrics, summarize_distribution
    evaluate.py                       # run_inference, predict_diameter
    runner.py                         # run_reproduction
    plots.py                          # plot_* functions
  tests/
    conftest.py                       # synthetic-data fixtures
    test_config.py
    test_preprocessing.py
    test_model.py
    test_dataset.py
    test_metrics.py
    test_evaluate.py
    test_runner.py
    test_plots.py
  notebooks/
    reproduce_left_eye.ipynb          # Colab orchestration (manual run)
  README.md
```

---

### Task 1: Project scaffold, dependencies, config

**Files:**
- Create: `pyproject.toml`, `pupilsense_repro/__init__.py`, `pupilsense_repro/config.py`
- Test: `tests/test_config.py`, `tests/conftest.py`

**Interfaces:**
- Produces:
  - `ReproConfig` frozen dataclass with fields: `data_root: Path`, `weights_dir: Path`, `device: str = "cpu"`, `resize_hw: tuple[int,int] = (32,64)`, `target_size: int = 224`, `imagenet_mean: tuple[float,float,float] = (0.485,0.456,0.406)`, `imagenet_std: tuple[float,float,float] = (0.229,0.224,0.225)`, `batch_size: int = 128`, `eye: str = "left"`, `results_dir: Path = Path("results")`, `figures_dir: Path = Path("figures")`.
  - `FOLD_TEST_PARTICIPANTS: dict[int, list[int]]`
  - `PAPER_MAPE: dict[str, float]` keyed by `"resnet18"`, `"resnet50"`
  - `weights_path_for(config: ReproConfig, base: str) -> Path`
  - `target_column(config: ReproConfig) -> str` → `"left_pupil"` when `eye == "left"`, else `"right_pupil"`

- [ ] **Step 1: Initialize uv project and add dependencies**

Run:
```bash
cd "C:/Users/User/Documents/projects/EyeBiomarkers"
uv init --python 3.12 --no-workspace
uv add torch torchvision pandas pillow matplotlib tqdm
uv add --dev pytest
```
Expected: `pyproject.toml` created, `.venv` populated, `uv.lock` written. (If `uv init` created `hello.py` / `main.py`, delete it.)

- [ ] **Step 2: Write the failing test**

`tests/test_config.py`:
```python
from pathlib import Path
from pupilsense_repro.config import (
    ReproConfig, FOLD_TEST_PARTICIPANTS, PAPER_MAPE, weights_path_for, target_column,
)


def test_folds_match_paper():
    assert set(FOLD_TEST_PARTICIPANTS) == {1, 2, 3, 4, 5}
    assert FOLD_TEST_PARTICIPANTS[1] == [1, 4, 6, 25, 36]
    assert FOLD_TEST_PARTICIPANTS[5] == [10, 14, 24, 28, 31]
    assert all(len(v) == 5 for v in FOLD_TEST_PARTICIPANTS.values())


def test_weights_path_and_target_column():
    cfg = ReproConfig(data_root=Path("d"), weights_dir=Path("w"))
    assert weights_path_for(cfg, "resnet18") == Path("w") / "ResNet18" / "left_eye.pt"
    assert weights_path_for(cfg, "resnet50") == Path("w") / "ResNet50" / "left_eye.pt"
    assert target_column(cfg) == "left_pupil"
    assert PAPER_MAPE["resnet50"] == 3.234711
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: pupilsense_repro.config`.

- [ ] **Step 4: Write minimal implementation**

`pupilsense_repro/__init__.py`:
```python
import pandas as pd

pd.options.mode.copy_on_write = True
```

`pupilsense_repro/config.py`:
```python
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
    resize_hw: tuple[int, int] = (32, 64)
    target_size: int = 224
    imagenet_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    batch_size: int = 128
    eye: str = "left"
    results_dir: Path = field(default_factory=lambda: Path("results"))
    figures_dir: Path = field(default_factory=lambda: Path("figures"))


def weights_path_for(config: ReproConfig, base: str) -> Path:
    return config.weights_dir / _BASE_TO_DIR[base] / f"{config.eye}_eye.pt"


def target_column(config: ReproConfig) -> str:
    return f"{config.eye}_pupil"
```

`tests/conftest.py` (placeholder for now; fixtures added in later tasks):
```python
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock pupilsense_repro/__init__.py pupilsense_repro/config.py tests/test_config.py tests/conftest.py
git commit -m "feat(config): ReproConfig, folds, paper constants"
```

---

### Task 2: EyePreprocessor

**Files:**
- Create: `pupilsense_repro/preprocessing.py`
- Test: `tests/test_preprocessing.py`

**Interfaces:**
- Consumes: nothing (constants passed in).
- Produces: `EyePreprocessor(resize_hw=(32,64), target_size=224, mean=..., std=...)`; callable `__call__(img: PIL.Image.Image) -> torch.Tensor` returning shape `(3, 224, 224)`, dtype float32. Classmethod `from_config(config) -> EyePreprocessor`.

- [ ] **Step 1: Write the failing test**

`tests/test_preprocessing.py`:
```python
import torch
from PIL import Image
from pupilsense_repro.preprocessing import EyePreprocessor


def _sample_eye():
    return Image.new("RGB", (32, 16), color=(120, 130, 140))  # stored eye crop W=32,H=16


def test_output_shape_and_dtype():
    tensor = EyePreprocessor()(_sample_eye())
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_padding_is_zero_after_normalization():
    # Corners are pad regions; with fill=0 then ImageNet-normalize they equal -mean/std.
    tensor = EyePreprocessor()(_sample_eye())
    expected_corner = -0.485 / 0.229
    assert abs(tensor[0, 0, 0].item() - expected_corner) < 1e-4


def test_grayscale_input_is_converted():
    gray = Image.new("L", (32, 16), color=100)
    assert EyePreprocessor()(gray).shape == (3, 224, 224)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_preprocessing.py -v`
Expected: FAIL with `ModuleNotFoundError: pupilsense_repro.preprocessing`.

- [ ] **Step 3: Write minimal implementation**

`pupilsense_repro/preprocessing.py`:
```python
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
    def from_config(cls, config: ReproConfig) -> "EyePreprocessor":
        return cls(config.resize_hw, config.target_size, config.imagenet_mean, config.imagenet_std)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self._transform(img.convert("RGB"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_preprocessing.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add pupilsense_repro/preprocessing.py tests/test_preprocessing.py
git commit -m "feat(preprocessing): EyePreprocessor matching paper pipeline"
```

---

### Task 3: ResNetRegressor and load_eye_model

**Files:**
- Create: `pupilsense_repro/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `ResNetRegressor(base: str = "resnet18")` (`nn.Module`), `forward(x) -> Tensor` shape `(N, 1)`.
  - `load_eye_model(weights_path: str | Path, base: str = "resnet18", device: str = "cpu") -> nn.Module` (eval mode, on device).

- [ ] **Step 1: Write the failing test**

`tests/test_model.py`:
```python
import torch
from pupilsense_repro.model import ResNetRegressor, load_eye_model


def test_forward_output_shape():
    model = ResNetRegressor("resnet18")
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 1)


def test_invalid_base_raises():
    import pytest
    with pytest.raises(ValueError):
        ResNetRegressor("resnet101")


def test_load_eye_model_roundtrip(tmp_path):
    saved = ResNetRegressor("resnet18")
    ckpt = tmp_path / "left_eye.pt"
    torch.save(saved.state_dict(), ckpt)
    loaded = load_eye_model(ckpt, base="resnet18", device="cpu")
    assert not loaded.training  # eval mode
    with torch.no_grad():
        assert loaded(torch.randn(1, 3, 224, 224)).shape == (1, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: pupilsense_repro.model`.

- [ ] **Step 3: Write minimal implementation**

`pupilsense_repro/model.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add pupilsense_repro/model.py tests/test_model.py
git commit -m "feat(model): ResNetRegressor and load_eye_model"
```

---

### Task 4: Dataset index and EyeDentifyDataset

**Files:**
- Create: `pupilsense_repro/dataset.py`
- Modify: `tests/conftest.py` (add synthetic-data fixture)
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `EyePreprocessor` (Task 2), `target_column` (Task 1).
- Produces:
  - `build_index(data_root: Path, eye: str = "left") -> pd.DataFrame` with columns `participant_id: int`, `session_id: int`, `frame_path: str` (relative, from CSV), `image_path: str` (absolute), `true: float` (the eye's pupil value).
  - `EyeDentifyDataset(index_df: pd.DataFrame, preprocessor: EyePreprocessor)` (`torch.utils.data.Dataset`); `__len__`; `__getitem__(i) -> tuple[Tensor, float, int, int, str]` = (image_tensor, true, participant_id, session_id, frame_path).

- [ ] **Step 1: Add the synthetic-data fixture**

Append to `tests/conftest.py`:
```python
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image


@pytest.fixture
def synthetic_data_root(tmp_path) -> Path:
    """Two participants, two sessions each, two frames each, with session_data.csv."""
    root = tmp_path / "left_eyes_data"
    rows_per = []
    for pid in (1, 2):
        for sid in (1, 2):
            session_dir = root / str(pid) / str(sid)
            session_dir.mkdir(parents=True)
            csv_rows = []
            for fi in (1, 2):
                fname = f"frame_{fi:02d}.png"
                Image.new("RGB", (32, 16), color=(100 + fi, 110, 120)).save(session_dir / fname)
                csv_rows.append({
                    "session_id": sid,
                    "left_pupil": 2.0 + 0.1 * fi + 0.5 * pid,
                    "right_pupil": 2.5 + 0.1 * fi,
                    "frame_path": f"{pid}/{sid}/{fname}",
                })
            pd.DataFrame(csv_rows).to_csv(session_dir / "session_data.csv", index=False)
    return root
```

- [ ] **Step 2: Write the failing test**

`tests/test_dataset.py`:
```python
import torch
from pupilsense_repro.dataset import build_index, EyeDentifyDataset
from pupilsense_repro.preprocessing import EyePreprocessor


def test_build_index_counts_and_columns(synthetic_data_root):
    idx = build_index(synthetic_data_root, eye="left")
    assert len(idx) == 8  # 2 participants x 2 sessions x 2 frames
    assert set(["participant_id", "session_id", "frame_path", "image_path", "true"]).issubset(idx.columns)
    assert set(idx["participant_id"]) == {1, 2}


def test_index_true_matches_left_pupil(synthetic_data_root):
    idx = build_index(synthetic_data_root, eye="left")
    row = idx[(idx.participant_id == 2) & (idx.session_id == 1) & (idx.frame_path.str.endswith("frame_02.png"))].iloc[0]
    assert abs(row["true"] - (2.0 + 0.2 + 1.0)) < 1e-9


def test_dataset_getitem(synthetic_data_root):
    idx = build_index(synthetic_data_root, eye="left")
    ds = EyeDentifyDataset(idx, EyePreprocessor())
    tensor, true, pid, sid, frame_path = ds[0]
    assert tensor.shape == (3, 224, 224)
    assert isinstance(true, float) and isinstance(pid, int)
    assert len(ds) == 8
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: pupilsense_repro.dataset`.

- [ ] **Step 4: Write minimal implementation**

`pupilsense_repro/dataset.py`:
```python
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from .preprocessing import EyePreprocessor


def build_index(data_root: Path, eye: str = "left") -> pd.DataFrame:
    data_root = Path(data_root)
    pupil_col = f"{eye}_pupil"
    session_csvs = sorted(data_root.glob("*/*/session_data.csv"))
    frames = []
    for csv_path in session_csvs:
        session_dir = csv_path.parent
        participant_id = int(session_dir.parent.name)
        session_id = int(session_dir.name)
        table = pd.read_csv(csv_path)
        image_paths = [str(data_root / rel) for rel in table["frame_path"]]
        existing = [(rel, img, val)
                    for rel, img, val in zip(table["frame_path"], image_paths, table[pupil_col])
                    if Path(img).exists()]
        for rel, img, val in existing:
            frames.append({
                "participant_id": participant_id,
                "session_id": session_id,
                "frame_path": rel,
                "image_path": img,
                "true": float(val),
            })
    return pd.DataFrame(frames)


class EyeDentifyDataset(Dataset):
    def __init__(self, index_df: pd.DataFrame, preprocessor: EyePreprocessor):
        self._index = index_df.reset_index(drop=True)
        self._preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int):
        row = self._index.iloc[i]
        tensor = self._preprocessor(Image.open(row["image_path"]))
        return tensor, float(row["true"]), int(row["participant_id"]), int(row["session_id"]), row["frame_path"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add pupilsense_repro/dataset.py tests/test_dataset.py tests/conftest.py
git commit -m "feat(dataset): build_index and EyeDentifyDataset"
```

---

### Task 5: Metrics

**Files:**
- Create: `pupilsense_repro/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `FOLD_TEST_PARTICIPANTS` (Task 1).
- Produces (all operate on a results DataFrame with columns `participant_id`, `true`, `prediction`):
  - `add_errors(df) -> pd.DataFrame` adding `abs_error`, `abs_pct_error`.
  - `overall_metrics(df) -> dict` → `{"mae": float, "mape": float, "n": int}`.
  - `per_participant_mape(df) -> pd.Series` (index = participant_id, value = MAPE).
  - `per_fold_metrics(df, folds=FOLD_TEST_PARTICIPANTS) -> pd.DataFrame` columns `fold`, `mape`, `n_participants`, `n_frames`.
  - `summarize_distribution(series) -> dict` → `mean, std, median, iqr, min, max, frac_gt_5pct`.

- [ ] **Step 1: Write the failing test**

`tests/test_metrics.py`:
```python
import pandas as pd
from pupilsense_repro.metrics import (
    add_errors, overall_metrics, per_participant_mape, per_fold_metrics, summarize_distribution,
)


def _df():
    # participant 1: preds off by 0.1 on true=2.0 -> 5% ; participant 2: exact -> 0%
    return pd.DataFrame({
        "participant_id": [1, 1, 2, 2],
        "true": [2.0, 2.0, 4.0, 4.0],
        "prediction": [2.1, 1.9, 4.0, 4.0],
    })


def test_overall_metrics():
    m = overall_metrics(add_errors(_df()))
    assert abs(m["mae"] - 0.05) < 1e-9   # (0.1+0.1+0+0)/4
    assert abs(m["mape"] - 2.5) < 1e-9   # (5+5+0+0)/4
    assert m["n"] == 4


def test_per_participant_mape():
    s = per_participant_mape(add_errors(_df()))
    assert abs(s.loc[1] - 5.0) < 1e-9
    assert abs(s.loc[2] - 0.0) < 1e-9


def test_per_fold_metrics_uses_only_test_participants():
    df = add_errors(_df())
    folds = {1: [1], 2: [2]}
    result = per_fold_metrics(df, folds)
    assert set(result["fold"]) == {1, 2}
    assert abs(result.set_index("fold").loc[1, "mape"] - 5.0) < 1e-9


def test_summarize_distribution():
    s = per_participant_mape(add_errors(_df()))
    d = summarize_distribution(s)
    assert abs(d["mean"] - 2.5) < 1e-9
    assert d["max"] == 5.0
    assert 0.0 <= d["frac_gt_5pct"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: pupilsense_repro.metrics`.

- [ ] **Step 3: Write minimal implementation**

`pupilsense_repro/metrics.py`:
```python
from __future__ import annotations

import pandas as pd

from .config import FOLD_TEST_PARTICIPANTS


def add_errors(df: pd.DataFrame) -> pd.DataFrame:
    result = df.assign(abs_error=(df["prediction"] - df["true"]).abs())
    safe_true = df["true"].where(df["true"] != 0, 1e-9)
    return result.assign(abs_pct_error=(result["abs_error"] / safe_true).abs() * 100)


def overall_metrics(df: pd.DataFrame) -> dict:
    return {"mae": float(df["abs_error"].mean()), "mape": float(df["abs_pct_error"].mean()), "n": int(len(df))}


def per_participant_mape(df: pd.DataFrame) -> pd.Series:
    return df.groupby("participant_id")["abs_pct_error"].mean().rename("mape")


def per_fold_metrics(df: pd.DataFrame, folds: dict[int, list[int]] = FOLD_TEST_PARTICIPANTS) -> pd.DataFrame:
    rows = []
    for fold, participants in folds.items():
        subset = df[df["participant_id"].isin(participants)]
        if subset.empty:
            continue
        rows.append({
            "fold": fold,
            "mape": float(subset["abs_pct_error"].mean()),
            "n_participants": int(subset["participant_id"].nunique()),
            "n_frames": int(len(subset)),
        })
    return pd.DataFrame(rows)


def summarize_distribution(series: pd.Series) -> dict:
    return {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "median": float(series.median()),
        "iqr": float(series.quantile(0.75) - series.quantile(0.25)),
        "min": float(series.min()),
        "max": float(series.max()),
        "frac_gt_5pct": float((series > 5.0).mean()),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add pupilsense_repro/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): MAE/MAPE overall, per-participant, per-fold, distribution"
```

---

### Task 6: Inference (run_inference, predict_diameter)

**Files:**
- Create: `pupilsense_repro/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `EyeDentifyDataset` (Task 4), `EyePreprocessor` (Task 2).
- Produces:
  - `run_inference(model, dataset, batch_size=128, device="cpu") -> pd.DataFrame` with columns `participant_id`, `session_id`, `frame_path`, `true`, `prediction` (one row per frame, order preserved).
  - `predict_diameter(pil_image, model, preprocessor, device="cpu") -> float` (single-image inference; the app reuse hook).

- [ ] **Step 1: Write the failing test**

`tests/test_evaluate.py`:
```python
import torch
import torch.nn as nn
from PIL import Image
from pupilsense_repro.dataset import build_index, EyeDentifyDataset
from pupilsense_repro.preprocessing import EyePreprocessor
from pupilsense_repro.evaluate import run_inference, predict_diameter


class ConstModel(nn.Module):
    def __init__(self, value=2.3):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full((x.shape[0], 1), self.value)


def test_run_inference_columns_and_length(synthetic_data_root):
    idx = build_index(synthetic_data_root, eye="left")
    ds = EyeDentifyDataset(idx, EyePreprocessor())
    out = run_inference(ConstModel(2.3), ds, batch_size=4, device="cpu")
    assert len(out) == len(ds)
    assert list(out.columns) == ["participant_id", "session_id", "frame_path", "true", "prediction"]
    assert (out["prediction"].round(4) == 2.3).all()


def test_predict_diameter_single_image():
    img = Image.new("RGB", (32, 16), color=(120, 120, 120))
    value = predict_diameter(img, ConstModel(3.14), EyePreprocessor(), device="cpu")
    assert abs(value - 3.14) < 1e-4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: pupilsense_repro.evaluate`.

- [ ] **Step 3: Write minimal implementation**

`pupilsense_repro/evaluate.py`:
```python
from __future__ import annotations

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from .preprocessing import EyePreprocessor


def run_inference(model, dataset, batch_size: int = 128, device: str = "cpu") -> pd.DataFrame:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    records = []
    model.eval()
    with torch.no_grad():
        for tensors, trues, participant_ids, session_ids, frame_paths in tqdm(loader, desc="Predicting"):
            preds = model(tensors.to(device)).squeeze(1).cpu().tolist()
            batch = zip(participant_ids.tolist(), session_ids.tolist(), frame_paths,
                        trues.tolist(), preds)
            records.extend({
                "participant_id": pid, "session_id": sid, "frame_path": fp,
                "true": true, "prediction": pred,
            } for pid, sid, fp, true, pred in batch)
    return pd.DataFrame(records, columns=["participant_id", "session_id", "frame_path", "true", "prediction"])


def predict_diameter(pil_image: Image.Image, model, preprocessor: EyePreprocessor, device: str = "cpu") -> float:
    model.eval()
    with torch.no_grad():
        tensor = preprocessor(pil_image).unsqueeze(0).to(device)
        return float(model(tensor).item())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_evaluate.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add pupilsense_repro/evaluate.py tests/test_evaluate.py
git commit -m "feat(evaluate): run_inference and predict_diameter"
```

---

### Task 7: Reproduction runner

**Files:**
- Create: `pupilsense_repro/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `ReproConfig`, `weights_path_for`, `PAPER_MAPE` (Task 1); `build_index`, `EyeDentifyDataset` (Task 4); `EyePreprocessor` (Task 2); `run_inference` (Task 6); metrics (Task 5); `load_eye_model` (Task 3, injectable).
- Produces:
  - `run_reproduction(config, bases=("resnet18","resnet50"), model_loader=load_eye_model) -> pd.DataFrame` — builds the index once, evaluates each base, writes `config.results_dir/left_eye_metrics.csv` and `config.results_dir/per_participant_mape.csv`, and returns the summary DataFrame (one row per base: `base`, `overall_mae`, `overall_mape`, `per_participant_mape_mean`, `per_participant_mape_std`, `per_fold_mape_mean`, `per_fold_mape_std`, `paper_mape`). Also returns per-base results DataFrames via attribute? No — writes CSVs and returns summary; per-frame results saved to `config.results_dir/predictions_<base>.csv`.
  - `evaluate_base(config, base, index_df, preprocessor, model_loader) -> tuple[pd.DataFrame, dict, pd.Series]` = (predictions_with_errors, summary_row, per_participant_series).

- [ ] **Step 1: Write the failing test**

`tests/test_runner.py`:
```python
from pathlib import Path

import torch
import torch.nn as nn
from pupilsense_repro.config import ReproConfig
from pupilsense_repro.runner import run_reproduction


class ConstModel(nn.Module):
    def forward(self, x):
        return torch.full((x.shape[0], 1), 2.5)


def test_run_reproduction_writes_outputs(synthetic_data_root, tmp_path):
    cfg = ReproConfig(
        data_root=synthetic_data_root,
        weights_dir=tmp_path / "weights",
        results_dir=tmp_path / "results",
        batch_size=4,
    )
    summary = run_reproduction(cfg, bases=("resnet18",), model_loader=lambda *a, **k: ConstModel())
    assert list(summary["base"]) == ["resnet18"]
    assert "overall_mape" in summary.columns
    assert (tmp_path / "results" / "left_eye_metrics.csv").exists()
    assert (tmp_path / "results" / "per_participant_mape.csv").exists()
    assert (tmp_path / "results" / "predictions_resnet18.csv").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: pupilsense_repro.runner`.

- [ ] **Step 3: Write minimal implementation**

`pupilsense_repro/runner.py`:
```python
from __future__ import annotations

import pandas as pd

from .config import ReproConfig, weights_path_for, PAPER_MAPE
from .dataset import build_index, EyeDentifyDataset
from .preprocessing import EyePreprocessor
from .model import load_eye_model
from .evaluate import run_inference
from .metrics import add_errors, overall_metrics, per_participant_mape, per_fold_metrics, summarize_distribution


def evaluate_base(config, base, index_df, preprocessor, model_loader):
    model = model_loader(weights_path_for(config, base), base=base, device=config.device)
    dataset = EyeDentifyDataset(index_df, preprocessor)
    predictions = add_errors(run_inference(model, dataset, config.batch_size, config.device))
    overall = overall_metrics(predictions)
    per_participant = per_participant_mape(predictions)
    dist = summarize_distribution(per_participant)
    fold_df = per_fold_metrics(predictions)
    summary_row = {
        "base": base,
        "overall_mae": overall["mae"],
        "overall_mape": overall["mape"],
        "n_frames": overall["n"],
        "per_participant_mape_mean": dist["mean"],
        "per_participant_mape_std": dist["std"],
        "per_fold_mape_mean": float(fold_df["mape"].mean()) if not fold_df.empty else float("nan"),
        "per_fold_mape_std": float(fold_df["mape"].std()) if not fold_df.empty else float("nan"),
        "paper_mape": PAPER_MAPE.get(base, float("nan")),
    }
    return predictions, summary_row, per_participant


def run_reproduction(config: ReproConfig, bases=("resnet18", "resnet50"), model_loader=load_eye_model) -> pd.DataFrame:
    config.results_dir.mkdir(parents=True, exist_ok=True)
    preprocessor = EyePreprocessor.from_config(config)
    index_df = build_index(config.data_root, config.eye)

    summary_rows = []
    per_participant_by_base = {}
    for base in bases:
        predictions, summary_row, per_participant = evaluate_base(
            config, base, index_df, preprocessor, model_loader
        )
        predictions.to_csv(config.results_dir / f"predictions_{base}.csv", index=False)
        summary_rows.append(summary_row)
        per_participant_by_base[base] = per_participant

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(config.results_dir / "left_eye_metrics.csv", index=False)
    pd.DataFrame(per_participant_by_base).to_csv(config.results_dir / "per_participant_mape.csv")
    return summary
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_runner.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add pupilsense_repro/runner.py tests/test_runner.py
git commit -m "feat(runner): run_reproduction orchestration with CSV outputs"
```

---

### Task 8: Plots

**Files:**
- Create: `pupilsense_repro/plots.py`
- Test: `tests/test_plots.py`

**Interfaces:**
- Consumes: `FOLD_TEST_PARTICIPANTS` (Task 1); per-participant Series and predictions DataFrames.
- Produces (each saves a PNG and returns the `Path`; uses matplotlib Agg, closes the figure):
  - `plot_per_participant_mape(per_participant_by_base: dict[str, pd.Series], out_path, folds=FOLD_TEST_PARTICIPANTS) -> Path` (box + jittered strip per base, points colored by fold membership, paper mean as reference line).
  - `plot_mape_histogram(per_participant_by_base, out_path) -> Path`.
  - `plot_pred_vs_true(predictions_df, out_path) -> Path`.
  - `plot_diameter_over_frames(predictions_df, participant_id, session_id, out_path) -> Path`.

- [ ] **Step 1: Write the failing test**

`tests/test_plots.py`:
```python
import pandas as pd
from pupilsense_repro.plots import (
    plot_per_participant_mape, plot_mape_histogram, plot_pred_vs_true, plot_diameter_over_frames,
)


def _per_participant():
    return {
        "resnet18": pd.Series({1: 3.4, 4: 5.1, 10: 2.9}, name="mape"),
        "resnet50": pd.Series({1: 3.2, 4: 4.0, 10: 2.5}, name="mape"),
    }


def _predictions():
    return pd.DataFrame({
        "participant_id": [1, 1, 1],
        "session_id": [1, 1, 1],
        "frame_path": ["1/1/frame_01.png", "1/1/frame_02.png", "1/1/frame_03.png"],
        "true": [2.3, 2.35, 2.4],
        "prediction": [2.31, 2.33, 2.42],
    })


def test_plots_write_files(tmp_path):
    p1 = plot_per_participant_mape(_per_participant(), tmp_path / "dist.png")
    p2 = plot_mape_histogram(_per_participant(), tmp_path / "hist.png")
    p3 = plot_pred_vs_true(_predictions(), tmp_path / "scatter.png")
    p4 = plot_diameter_over_frames(_predictions(), 1, 1, tmp_path / "series.png")
    for p in (p1, p2, p3, p4):
        assert p.exists() and p.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_plots.py -v`
Expected: FAIL with `ModuleNotFoundError: pupilsense_repro.plots`.

- [ ] **Step 3: Write minimal implementation**

`pupilsense_repro/plots.py`:
```python
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FOLD_TEST_PARTICIPANTS, PAPER_MAPE


def _fold_of(participant_id: int, folds: dict[int, list[int]]) -> int:
    for fold, participants in folds.items():
        if participant_id in participants:
            return fold
    return 0


def plot_per_participant_mape(per_participant_by_base, out_path, folds=FOLD_TEST_PARTICIPANTS) -> Path:
    bases = list(per_participant_by_base)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([per_participant_by_base[b].values for b in bases], labels=bases, showfliers=False)
    for x, base in enumerate(bases, start=1):
        series = per_participant_by_base[base]
        colors = [_fold_of(pid, folds) for pid in series.index]
        jitter = np.random.default_rng(0).normal(0, 0.05, len(series))
        ax.scatter(np.full(len(series), x) + jitter, series.values, c=colors, cmap="tab10", s=25, alpha=0.8)
        if base in PAPER_MAPE:
            ax.hlines(PAPER_MAPE[base], x - 0.3, x + 0.3, colors="red", linestyles="--")
    ax.set_ylabel("Per-participant MAPE (%)")
    ax.set_title("Left-eye per-participant MAPE (red dash = paper mean; point color = fold)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return Path(out_path)


def plot_mape_histogram(per_participant_by_base, out_path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for base, series in per_participant_by_base.items():
        ax.hist(series.values, bins=15, alpha=0.5, label=base)
    ax.set_xlabel("Per-participant MAPE (%)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title("Distribution of per-participant MAPE")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return Path(out_path)


def plot_pred_vs_true(predictions_df, out_path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(predictions_df["true"], predictions_df["prediction"], s=8, alpha=0.3)
    lo = float(min(predictions_df["true"].min(), predictions_df["prediction"].min()))
    hi = float(max(predictions_df["true"].max(), predictions_df["prediction"].max()))
    ax.plot([lo, hi], [lo, hi], "r--")
    ax.set_xlabel("True diameter (mm)")
    ax.set_ylabel("Predicted diameter (mm)")
    ax.set_title("Predicted vs true (left eye)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return Path(out_path)


def plot_diameter_over_frames(predictions_df, participant_id, session_id, out_path) -> Path:
    subset = predictions_df[
        (predictions_df["participant_id"] == participant_id)
        & (predictions_df["session_id"] == session_id)
    ].sort_values("frame_path")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(len(subset)), subset["true"].values, label="true", marker="o", ms=3)
    ax.plot(range(len(subset)), subset["prediction"].values, label="predicted", marker="x", ms=3)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Diameter (mm)")
    ax.set_title(f"Diameter over frames — participant {participant_id}, session {session_id}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return Path(out_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_plots.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add pupilsense_repro/plots.py tests/test_plots.py
git commit -m "feat(plots): per-participant distribution, histogram, scatter, time series"
```

---

### Task 9: Colab notebook + README, full-suite check

**Files:**
- Create: `notebooks/reproduce_left_eye.ipynb`, `README.md`

**Interfaces:**
- Consumes: `ReproConfig`, `run_reproduction`, all `plots.*`.
- Produces: no code API; a runnable Colab notebook and usage docs.

- [ ] **Step 1: Run the full local test suite**

Run: `uv run pytest -v`
Expected: all tests from Tasks 1–8 PASS.

- [ ] **Step 2: Create the Colab notebook**

Create `notebooks/reproduce_left_eye.ipynb` with these cells (in order):

Cell 1 (markdown): title + one-line purpose + reminder to select a GPU runtime.

Cell 2 (code) — mount Drive:
```python
from google.colab import drive
drive.mount("/content/drive")
```

Cell 3 (code) — make the package importable and set paths **(EDIT THESE TWO PATHS to match your Drive)**:
```python
import sys
from pathlib import Path
import torch

PACKAGE_PARENT = "/content/drive/MyDrive/EyeBiomarkers"      # folder containing pupilsense_repro/
DATA_ROOT = Path("/content/drive/MyDrive/EyeBiomarkers/data/left_eyes_data/local_left_eyes_data")
WEIGHTS_DIR = Path("/content/drive/MyDrive/EyeBiomarkers/pre_trained_models")

sys.path.insert(0, PACKAGE_PARENT)
from pupilsense_repro.config import ReproConfig
from pupilsense_repro.runner import run_reproduction
from pupilsense_repro import plots

config = ReproConfig(
    data_root=DATA_ROOT,
    weights_dir=WEIGHTS_DIR,
    device="cuda" if torch.cuda.is_available() else "cpu",
    results_dir=Path("/content/results"),
    figures_dir=Path("/content/figures"),
)
print("device:", config.device)
```

Cell 4 (code) — run reproduction:
```python
summary = run_reproduction(config, bases=("resnet18", "resnet50"))
summary
```

Cell 5 (code) — plots:
```python
import pandas as pd
config.figures_dir.mkdir(parents=True, exist_ok=True)
per_part = pd.read_csv(config.results_dir / "per_participant_mape.csv", index_col=0)
per_participant_by_base = {b: per_part[b].dropna() for b in per_part.columns}

plots.plot_per_participant_mape(per_participant_by_base, config.figures_dir / "per_participant_mape.png")
plots.plot_mape_histogram(per_participant_by_base, config.figures_dir / "mape_hist.png")
for base in ("resnet18", "resnet50"):
    preds = pd.read_csv(config.results_dir / f"predictions_{base}.csv")
    plots.plot_pred_vs_true(preds, config.figures_dir / f"pred_vs_true_{base}.png")
    plots.plot_diameter_over_frames(preds, participant_id=1, session_id=1,
                                    out_path=config.figures_dir / f"series_{base}.png")

from IPython.display import Image as IPyImage, display
display(IPyImage(str(config.figures_dir / "per_participant_mape.png")))
```

Cell 6 (markdown): how to read results + the train/test-overlap caveat (single deployed checkpoint, so overall MAPE may be optimistic; compare per-fold rows to the paper's ~3.2–3.4%).

- [ ] **Step 3: Write README.md**

`README.md`:
```markdown
# EyeBiomarkers — PupilSense left-eye reproduction (Stage 1)

Reproduces the left-eye MAE/MAPE of *PupilSense* (Shah et al., ETRA '25) by running the
authors' released ResNet18/ResNet50 checkpoints over the EyeDentify dataset.

## Layout
- `pupilsense_repro/` — reusable package (config, preprocessing, model, dataset, metrics, evaluate, runner, plots)
- `notebooks/reproduce_left_eye.ipynb` — Colab runner (mounts Drive, runs everything)
- `tests/` — local pytest suite (CPU)

## Run locally (tests only)
```bash
uv sync
uv run pytest -v
```

## Run the reproduction (Colab)
1. Put this folder in Google Drive (so `pupilsense_repro/`, the dataset, and the weights are all under Drive).
2. Open `notebooks/reproduce_left_eye.ipynb` in Colab, pick a GPU runtime.
3. Edit `PACKAGE_PARENT`, `DATA_ROOT`, `WEIGHTS_DIR` in cell 3 to match your Drive.
4. Run all cells. Outputs land in `/content/results` and `/content/figures`.

## Reuse hooks (for the future app)
- `pupilsense_repro.preprocessing.EyePreprocessor`
- `pupilsense_repro.model.load_eye_model`
- `pupilsense_repro.evaluate.predict_diameter`

## Caveat
The released weights are a single deployed checkpoint per eye/architecture, not per-fold
models. Overall MAPE can be optimistic where the checkpoint trained on a participant;
compare the per-fold rows to the paper's ~3.2–3.4% (left eye).
```

- [ ] **Step 4: Commit**

```bash
git add notebooks/reproduce_left_eye.ipynb README.md
git commit -m "feat(notebook): Colab reproduction notebook and README"
```

---

## Acceptance (manual, on Colab — requires the real weights + data)

Not a local test; performed by the user after the plan is implemented:

1. Place the package + dataset + weights in Drive; open the notebook on a GPU runtime; set the three paths.
2. Run all cells; confirm `left_eye_metrics.csv` has rows for resnet18 and resnet50 with `overall_mae`, `overall_mape`, per-participant and per-fold columns populated.
3. Confirm at least one fold's `mape` lands near the paper's left-eye values (ResNet18 ≈ 3.41%, ResNet50 ≈ 3.23%); document the overall-vs-fold pattern.
4. Confirm the four figures render.

---

## Self-Review

**Spec coverage:**
- Reusable package w/ single-responsibility modules → Tasks 1–8. ✓
- Exact preprocessing (§6) → Task 2 + constants in Task 1. ✓
- Model loader (§7) → Task 3. ✓
- Dataset + `left_pupil` target (§4) → Task 4. ✓
- MAE/MAPE overall + per-participant distribution + per-fold (§8) → Tasks 5, 7. ✓
- Outputs: metrics CSV, per-participant CSV, figures (§9) → Tasks 7, 8. ✓
- Colab/Drive config, paths set once (§10) → Task 1 (config), Task 9 (notebook). ✓
- Dependencies (§11) → Task 1. ✓
- Success criteria incl. `predict_diameter` reuse hook (§12, §13) → Task 6, README Task 9. ✓
- Caveats (§14) → notebook cell 6 + README. ✓
- Right eye / retraining / video pipeline explicitly out of scope — not implemented. ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code. ✓

**Type consistency:** results-DataFrame columns (`participant_id`, `session_id`, `frame_path`, `true`, `prediction`) are consistent across `run_inference` (Task 6), `metrics` (Task 5), `runner` (Task 7), `plots` (Task 8). `weights_path_for`/`target_column`/`ReproConfig` signatures consistent across Tasks 1, 7. `model_loader` injection signature `(weights_path, base=, device=)` matches `load_eye_model` (Task 3) and the runner test stub. ✓
