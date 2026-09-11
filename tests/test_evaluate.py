import torch
import torch.nn as nn
from PIL import Image
from pupilsense_repro.dataset import build_index, EyeDentifyDataset
from pupilsense_repro.preprocessing import EyePreprocessor
from pupilsense_repro.evaluate import run_inference, predict_diameter


class ConstModel(nn.Module):
    def __init__(self, value=2.3):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full((x.shape[0], 1), self.value)


def test_run_inference_columns_and_length(synthetic_data_root):
    idx = build_index(synthetic_data_root, eye="left")
    ds = EyeDentifyDataset(idx, EyePreprocessor())
    out = run_inference(ConstModel(2.3), ds, batch_size=4, device="cpu")
    assert len(out) == len(ds)
    assert list(out.columns) == ["participant_id", "session_id", "frame_path", "true", "prediction"]
    assert (out["prediction"].round(4) == 2.3).all()


def test_predict_diameter_single_image():
    img = Image.new("RGB", (32, 16), color=(120, 120, 120))
    value = predict_diameter(img, ConstModel(3.14), EyePreprocessor(), device="cpu")
    assert abs(value - 3.14) < 1e-4
