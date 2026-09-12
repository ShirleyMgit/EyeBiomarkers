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
    predictions = add_errors(run_inference(model, dataset, config.batch_size, config.device, config.num_workers))
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
