# PupilSense Left-Eye Reproduction — Design Spec

- **Project:** EyeBiomarkers
- **Stage:** 1 of N (reproduction before app)
- **Date:** 2026-09-11
- **Paper:** Shah, Watanabe, Moser, Dengel. *PupilSense: A Novel Application for Webcam-Based Pupil Diameter Estimation*, ETRA '25 (arXiv:2407.11204v2).
- **Reference code:** authors' repo `vijulshah/eyedentify`; user's Drive copy of the paper code + trained models; user's local scripts `pupil_evaluator_pupilsense.py`, `resnet_regg_pupilsense.py`.

## 1. Goal

Reproduce the paper's **left-eye** pupil-diameter accuracy on the **EyeDentify** dataset by running the authors' **released ResNet18 and ResNet50 checkpoints** through the paper's exact preprocessing and computing **MAE + MAPE**, including the **distribution of per-participant MAPE**. Verify the numbers land in the paper's reported range (left eye: ResNet18 ≈ 3.41%, ResNet50 ≈ 3.23% MAPE).

The preprocessing, model, and a single-image `predict_diameter()` are written as reusable modules so the later stages (video → eye-crop → diameter-over-time, then the app) call the same code.

This stage **evaluates pretrained weights**; it does **not** retrain.

## 2. The paper's method (what we are reproducing)

The model maps one small cropped eye image to a single scalar (pupil diameter in mm), with **separate models per eye**. Pipeline:

1. Input eye crop: stored as **32 (W) × 16 (H)** RGB PNG.
2. Upsample 2× (bicubic) to **32 (H) × 64 (W)**.
3. Zero-pad symmetrically to **224 × 224**.
4. Normalize with ImageNet mean/std.
5. ResNet18/ResNet50 backbone (from scratch in the paper) with the final FC replaced by a **linear head → 1 output**.

Reported metrics (Table 2, leave-one-participant-out): MAE (loss) and **MAPE** = mean(|pred − true| / true) × 100.

## 3. Scope

**In scope (Stage 1):**
- Left eye only.
- Both architectures: ResNet18 and ResNet50.
- Evaluation of released checkpoints on the local/Drive EyeDentify left-eye data.
- MAE + MAPE: overall, per-participant (with full distribution), and per 5-fold test set.
- Result tables (CSV) + diagnostic plots, run from a Colab notebook.

**Out of scope (later stages), but code is shaped to grow into them:**
- Right eye.
- Retraining / cross-validation training.
- Face-video → Mediapipe face+landmarks → eye crop → EAR/ViT blink filter → diameter-over-time pipeline.
- The recording/analysis app UI.

## 4. Data

- **Dataset:** EyeDentify (public Kaggle `vijuls/PupilDiameterDatasets`), left eyes only present locally.
- **Layout:** `<data_root>/<participant_id>/<session_id>/frame_XX.png` plus one `session_data.csv` per session.
- **Counts:** 51 participants × 50 sessions × up to 90 frames; PNGs are **W=32, H=16** RGB.
- **Ground truth:** each `session_data.csv` row (90 rows/session) has columns including `left_pupil`, `right_pupil`, `pupil_diameter`, gaze, and `frame_path` (relative, e.g. `1/1/frame_01.png`). We use **`left_pupil`** as the target for the left-eye models.
- **5-fold test participants (paper supplement, Table 3):**
  - Fold 1: 1, 4, 6, 25, 36
  - Fold 2: 2, 5, 7, 26, 37
  - Fold 3: 3, 16, 26, 38, 43
  - Fold 4: 9, 19, 29, 39, 49
  - Fold 5: 10, 14, 24, 28, 31

## 5. Architecture / modules

A small package `pupilsense_repro/` (single responsibility per module):

| Module | Responsibility |
|--------|----------------|
| `config.py` | Frozen `@dataclass` `ReproConfig`: `data_root`, `weights_dir`, device, preprocessing constants (sizes, ImageNet mean/std), the 5-fold test lists, paper reference MAPE values. |
| `preprocessing.py` | `EyePreprocessor` — PIL eye image → normalized 224×224 tensor. **Reused by the app.** |
| `model.py` | `ResNetRegressor(base)` (resnet18/50, FC→1) + `load_eye_model(weights_path, base, device)` (state_dict load, drop `resnet.fc.*`, `strict=False`, `.eval()`). |
| `dataset.py` | `EyeDentifyDataset` (torch `Dataset`) yielding `(tensor, left_pupil, participant_id, session_id, frame_path)`; `build_index(data_root)` → DataFrame of every frame + GT. |
| `evaluate.py` | `run_inference(model, dataset)` → predictions DataFrame (batched, GPU). |
| `metrics.py` | `overall_metrics(df)`, `per_participant_mape(df)`, `per_fold_metrics(df, folds)`. MAE + MAPE per §8. |
| `plots.py` | Per-participant MAPE box/violin+strip (RN18 vs RN50, fold-colored), MAPE histogram, predicted-vs-true scatter, and a **diameter-over-frames** line for one session. |
| `reproduce_left_eye.ipynb` | Thin Colab notebook: mount Drive → set `ReproConfig` → build dataset → load each model → evaluate → print tables → save CSVs + plots. |

Design rule: `preprocessing.py` and `model.py` have **no dataset/eval dependencies**, so the app imports them directly.

## 6. Preprocessing spec (exact)

Matches `pupil_evaluator_pupilsense.py`:

```
Resize((32, 64), interpolation=BICUBIC)      # H=32, W=64  (2x upscale of 16x32)
Pad((pad_w, pad_h, pad_w, pad_h), fill=0)     # pad_h=(224-32)//2=96, pad_w=(224-64)//2=80 -> 224x224
ToTensor()
Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
```

## 7. Model loading spec

From `resnet_regg_pupilsense.py`: build `ResNetRegressor(base)`, `torch.load(path, map_location=device)`, drop keys starting with `resnet.fc`, `load_state_dict(filtered, strict=False)`, move to device, `.eval()`. Loader takes an explicit `weights_path` (no hard-coded relative dir) so Colab/Drive paths come from `ReproConfig`.

## 8. Evaluation & metrics spec

- **Per frame:** `abs_error = |pred − true|`; `abs_pct_error = |pred − true| / max(true, 1e-9) × 100`.
- **Overall (per model):** MAE = mean abs_error (mm); MAPE = mean abs_pct_error (%). Matches the paper's definition and the user's evaluator.
- **Per participant:** MAPE computed over all of that participant's frames → one value per participant. Reported as a **Series/table** plus summary stats (mean, std, median, IQR, min, max, count > 5%).
- **Per fold:** MAPE over each fold's test participants → 5 values; report mean ± std across folds to line up with Table 2/3, and flag which fold (if any) matches the paper's ~3.2–3.4% (the signal for which fold the checkpoint came from).
- Frames with missing image or non-finite GT are dropped and counted.

## 9. Outputs

- `results/left_eye_metrics.csv` — one row per model: overall MAE, overall MAPE, per-participant MAPE mean ± std, per-fold MAPE mean ± std.
- `results/per_participant_mape.csv` — participant × model MAPE matrix.
- `figures/` — per-participant MAPE box/violin+strip (fold-colored), MAPE histogram, predicted-vs-true scatter, and an example diameter-over-frames line.
- Printed comparison table (ours vs paper) in the notebook.

## 10. Environment & layout (Colab / Drive)

- Runs on **Google Colab** (GPU) with **Google Drive mounted**.
- Data + trained models live in the user's Drive (folders provided by the user). Exact mounted paths are set **once** at the top of the notebook via `ReproConfig` — copied from Colab's file browser — not hard-coded here.
- The `pupilsense_repro/` package is developed in this local `EyeBiomarkers` git repo, then placed in the Drive project folder (or added to `sys.path` after mount) so Colab imports it.
- Expected weight files: `<weights_dir>/ResNet18/left_eye.pt`, `<weights_dir>/ResNet50/left_eye.pt` (filenames confirmed against the user's Drive before running).

## 11. Dependencies

`torch`, `torchvision`, `pandas`, `pillow`, `tqdm`, `matplotlib` (seaborn optional for the violin/strip). Managed with `uv` locally; pre-installed on Colab.

## 12. Success criteria

- Notebook runs end-to-end in Colab on left-eye data with no manual per-file steps beyond setting the config paths.
- Produces the metrics CSVs and figures listed in §9.
- Left-eye MAPE for at least one fold's test set lands in the paper's ballpark (~3–5%); overall/per-fold pattern documented honestly (including the train/test-overlap caveat).
- `EyePreprocessor` and `load_eye_model` import cleanly and `predict_diameter(pil_image, model)` returns a mm value — the reuse hook for the app.

## 13. Reuse hooks for the app (bridge to later stages)

- `EyePreprocessor` = the single source of truth for turning an eye crop into a model input.
- `predict_diameter(pil_image, model)` = one-image inference used later per video frame.
- `ReproConfig` centralizes paths/constants so the video pipeline extends it rather than re-declaring.

## 14. Risks & caveats

- **Single deployed checkpoint, not per-fold weights:** evaluating on participants the checkpoint trained on yields optimistically low MAPE. Mitigated by per-fold reporting (§8) and explicit caveat in outputs.
- **Column/eye convention:** confirm `left_pupil` is the correct target for the left-eye crops (assumed from CSV + evaluator).
- **Checkpoint key layout:** if released keys are not `resnet.`-prefixed, the loader's filter/`strict=False` still loads the backbone; verify no unexpected missing/extra keys at load time and log them.
- **Drive path drift:** paths are user-set config, verified before the run.

## 15. Open items (confirm before/at first run)

1. Exact Drive mounted paths for `data_root` and `weights_dir`, and the weight filenames.
2. Whether the deployed checkpoint corresponds to a known fold (lets us report a strict held-out number).
