from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import FOLD_TEST_PARTICIPANTS, paper_mape_for


def _fold_of(participant_id: int, folds: dict[int, list[int]]) -> int:
    for fold, participants in folds.items():
        if participant_id in participants:
            return fold
    return 0


def plot_per_participant_mape(per_participant_by_base, out_path, folds=FOLD_TEST_PARTICIPANTS, eye="left") -> Path:
    bases = list(per_participant_by_base)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([per_participant_by_base[b].values for b in bases], tick_labels=bases, showfliers=False)
    for x, base in enumerate(bases, start=1):
        series = per_participant_by_base[base]
        colors = [_fold_of(pid, folds) for pid in series.index]
        jitter = np.random.default_rng(0).normal(0, 0.05, len(series))
        ax.scatter(np.full(len(series), x) + jitter, series.values, c=colors, cmap="tab10", s=25, alpha=0.8)
        paper = paper_mape_for(eye, base)
        if paper == paper:  # not NaN
            ax.hlines(paper, x - 0.3, x + 0.3, colors="red", linestyles="--")
    ax.set_ylabel("Per-participant MAPE (%)")
    ax.set_title(f"{eye.capitalize()}-eye per-participant MAPE (red dash = paper mean; point color = fold)")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
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
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
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
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
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
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return Path(out_path)
