import torch
from PIL import Image
from pupilsense_repro.preprocessing import EyePreprocessor


def _sample_eye():
    return Image.new("RGB", (32, 16), color=(120, 130, 140))  # stored eye crop W=32, H=16


def test_default_is_totensor_only():
    tensor = EyePreprocessor()(_sample_eye())
    assert tensor.shape == (3, 16, 32)  # native size, channels-first; model pads to 192 internally
    assert tensor.dtype == torch.float32
    # ToTensor scales to [0, 1] with no normalization applied
    assert float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0
    assert abs(tensor[0, 0, 0].item() - 120 / 255) < 1e-4


def test_grayscale_input_is_converted():
    gray = Image.new("L", (32, 16), color=100)
    assert EyePreprocessor()(gray).shape == (3, 16, 32)


def test_optional_resize():
    tensor = EyePreprocessor(img_size=(32, 64))(_sample_eye())
    assert tensor.shape == (3, 32, 64)


def test_optional_normalize_shifts_values():
    tensor = EyePreprocessor(normalize=True)(_sample_eye())
    expected = (120 / 255 - 0.485) / 0.229
    assert abs(tensor[0, 0, 0].item() - expected) < 1e-4
