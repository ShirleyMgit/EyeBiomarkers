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
    assert abs(d["max"] - 5.0) < 1e-9
    assert 0.0 <= d["frac_gt_5pct"] <= 1.0
