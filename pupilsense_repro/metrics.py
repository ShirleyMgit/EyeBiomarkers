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
