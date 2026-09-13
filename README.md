# EyeBiomarkers — PupilSense left-eye reproduction (Stage 1)

Reproduces the left-eye MAE/MAPE of *PupilSense* (Shah et al., ETRA '25) by running the
authors' released ResNet18/ResNet50 checkpoints over the EyeDentify dataset.

## Layout
- `pupilsense_repro/` — reusable package (config, preprocessing, model, dataset, metrics, evaluate, runner, plots)
- `notebooks/reproduce_both_eyes.ipynb` — Colab runner (mounts Drive, runs everything)
- `tests/` — local pytest suite (CPU)

## Run locally (tests only)
```bash
uv sync
uv run pytest -v
```

## Run the reproduction (Colab)
1. Put this folder in Google Drive (so `pupilsense_repro/`, the dataset, and the weights are all under Drive).
2. Open `notebooks/reproduce_both_eyes.ipynb` in Colab, pick a GPU runtime.
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
