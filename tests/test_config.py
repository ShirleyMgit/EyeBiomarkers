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
