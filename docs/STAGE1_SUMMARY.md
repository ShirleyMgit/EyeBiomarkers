# EyeBiomarkers — Stage 1 Summary (PupilSense reproduction)

_Last updated: 2026-09-13_

## Goal

Reproduce the pupil-diameter results of **PupilSense** (Shah, Watanabe, Moser, Dengel — ETRA '25,
arXiv:2407.11204) on the public **EyeDentify** dataset, as the validated foundation for an app that
estimates pupil diameter from a face video. Stage 1 **evaluates the authors' released checkpoints**
(no retraining).

## What we have

### Code — `pupilsense_repro/` (Python package, tested)
| Module | Responsibility |
|---|---|
| `config.py` | `ReproConfig` (frozen dataclass), 5-fold participant lists, per-eye `PAPER_MAPE`, path/target helpers |
| `preprocessing.py` | `EyePreprocessor` — image → model tensor (resize 32×64, ToTensor, **no** normalization) |
| `model.py` | `ResNetRegressor` (authors' architecture) + `load_eye_model` |
| `dataset.py` | `build_index` + `EyeDentifyDataset` (walks `<pid>/<session>/frame_XX.png` + `session_data.csv`) |
| `evaluate.py` | `run_inference` (batched, GPU) + `predict_diameter` (single-image reuse hook for the app) |
| `metrics.py` | MAE / MAPE — overall, per-participant, per-fold, distribution |
| `runner.py` | `run_reproduction` — ties it together, writes per-eye CSVs |
| `plots.py` | per-participant MAPE distribution, histogram, pred-vs-true, diameter-over-frames |

- **25 unit tests pass** (`uv run pytest`), CPU-only, using synthetic fixtures.
- **Notebook:** `notebooks/reproduce_both_eyes.ipynb` — Colab runner: mounts Drive → clones this repo →
  stages data zips to local disk → runs both eyes → plots.
- **Repo:** public at **https://github.com/ShirleyMgit/EyeBiomarkers** (branch `master`).

### Data & weights (on the user's Google Drive, not in the repo)
- Dataset: public **EyeDentify** (Kaggle `vijuls/PupilDiameterDatasets`) — 51 participants ×
  50 sessions × ~90 frames of **32×16 RGB eye crops**; `session_data.csv` per session gives
  `left_pupil` / `right_pupil` ground truth (mm, from a Tobii eye-tracker).
- Staged as `left_eyes_data.zip` and `right_eyes_data.zip` (the right zip was extracted locally from
  the full 32 GB Kaggle archive — only `right_eyes` pulled out).
- Weights: released deployed checkpoints at `pre_trained_models/{ResNet18,ResNet50}/{left,right}_eye.pt`.

## Results (ResNet50, full dataset — 212,073 frames per eye)

| Eye | Our overall MAPE | Our MAE | Per-fold MAPE (held-out) | Paper MAPE (LOPOCV) |
|---|---|---|---|---|
| Left | 5.43% | 0.132 mm | 6.25% ± 3.98 | 3.23% |
| Right | 5.12% | 0.125 mm | 5.65% ± 2.69 | 3.64% |

Progression on the left eye while debugging: **13.6% → ~5.4%** after correcting the architecture and
preprocessing.

## Key findings (paper text vs. released code)

The paper's prose and the authors' released code disagree; the **checkpoints follow the code**:

1. **Architecture:** the model keeps the ResNet's original `fc` (→1000) and adds a **separate
   `regression_head = Linear(1000, 1)`**; `forward = regression_head(resnet(pad(x)))`. Our first
   attempt used `fc = Linear(…, 1)` and no `regression_head`, so the trained head was silently dropped
   → random predictions. (Confirmed from `registrations/ResNetVariants/ImageNetBasedResNets.py` and the
   checkpoint's own keys.)
2. **Preprocessing:** upsample the 16×32 crop 2× to **32×64** (bicubic), `ToTensor`, **no ImageNet
   normalization**; the model zero-pads to 224 internally. (Confirmed from `dataset_classes.py`,
   `pt_train.yml` `normalize: false`, `predict.yml` `img_size: null`.) Resizing to 32×64 roughly
   halved MAPE; normalization hurt.

## The remaining gap (why ~5% and not ~3.2%)

- **Iris vs eye crops.** The authors trained on **iris** crops (`.../iris/...`, `.../left_iris/...`),
  but the **public dataset only contains eye crops** (verified: the Kaggle dataset has
  `left_eyes`, `right_eyes`, depth maps, and a super-res variant — **no iris folder**). The original
  full-resolution frames were withheld for privacy, so proper iris crops can't be regenerated.
- **Single deployed checkpoint** per eye/architecture vs. the paper's per-fold LOPOCV models.

Both are **data/artifact-availability limits, not bugs** — the pipeline, model, and metrics are
reproduced faithfully; the accuracy lands at ~5% because the exact training-input variant isn't public.

## How to run (Colab)

1. Ensure Drive has `.../pupilsense_data_code/data/{left,right}_eyes_data.zip` and
   `.../code/pupilsense/pre_trained_models/{ResNet18,ResNet50}/{left,right}_eye.pt`.
2. Open `notebooks/reproduce_both_eyes.ipynb` from GitHub, pick a **GPU** (L4/T4), **Run all**.
3. Output: a combined `summary` (left + right rows), per-eye CSVs and figures under `/content`.

Local tests: `uv sync && uv run pytest`.

## Reuse hooks for the app (Stage 2)
- `pupilsense_repro.preprocessing.EyePreprocessor`
- `pupilsense_repro.model.load_eye_model`
- `pupilsense_repro.evaluate.predict_diameter`

## Next — Stage 2 (the app)
Face video → MediaPipe face/eye (and **iris**) crop → blink filter (EAR) → `predict_diameter` per
frame → diameter-over-time + CSV/plot. Because we control the camera in Stage 2, we can generate
proper **iris crops** ourselves (closing the gap hit here) and optionally fine-tune on new data.
