import torch
from PIL import Image
from pupilsense_repro.preprocessing import EyePreprocessor


def _sample_eye():
    return Image.new("RGB", (32, 16), color=(120, 130, 140))  # stored eye crop W=32,H=16


def test_output_shape_and_dtype():
    tensor = EyePreprocessor()(_sample_eye())
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32


def test_padding_is_zero_after_normalization():
    # Corners are pad regions; with fill=0 then ImageNet-normalize they equal -mean/std.
    tensor = EyePreprocessor()(_sample_eye())
    expected_corner = -0.485 / 0.229
    assert abs(tensor[0, 0, 0].item() - expected_corner) < 1e-4


def test_grayscale_input_is_converted():
    gray = Image.new("L", (32, 16), color=100)
    assert EyePreprocessor()(gray).shape == (3, 224, 224)
