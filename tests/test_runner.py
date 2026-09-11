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
