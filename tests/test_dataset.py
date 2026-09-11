import torch
from pupilsense_repro.dataset import build_index, EyeDentifyDataset
from pupilsense_repro.preprocessing import EyePreprocessor


def test_build_index_counts_and_columns(synthetic_data_root):
    idx = build_index(synthetic_data_root, eye="left")
    assert len(idx) == 8  # 2 participants x 2 sessions x 2 frames
    assert set(["participant_id", "session_id", "frame_path", "image_path", "true"]).issubset(idx.columns)
    assert set(idx["participant_id"]) == {1, 2}


def test_index_true_matches_left_pupil(synthetic_data_root):
    idx = build_index(synthetic_data_root, eye="left")
    row = idx[(idx.participant_id == 2) & (idx.session_id == 1) & (idx.frame_path.str.endswith("frame_02.png"))].iloc[0]
    assert abs(row["true"] - (2.0 + 0.2 + 1.0)) < 1e-9


def test_dataset_getitem(synthetic_data_root):
    idx = build_index(synthetic_data_root, eye="left")
    ds = EyeDentifyDataset(idx, EyePreprocessor())
    tensor, true, pid, sid, frame_path = ds[0]
    assert tensor.shape == (3, 224, 224)
    assert isinstance(true, float) and isinstance(pid, int)
    assert len(ds) == 8
