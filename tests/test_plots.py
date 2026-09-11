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
