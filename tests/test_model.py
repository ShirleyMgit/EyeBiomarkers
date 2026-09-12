import warnings

import pytest
import torch
from pupilsense_repro.model import ResNetRegressor, load_eye_model


def test_forward_output_shape():
    model = ResNetRegressor("resnet18")
    out = model(torch.rand(2, 3, 16, 32))  # raw crop; padded to 192 inside forward
    assert out.shape == (2, 1)


def test_invalid_base_raises():
    with pytest.raises(ValueError):
        ResNetRegressor("resnet101")


def test_load_eye_model_roundtrip(tmp_path):
    saved = ResNetRegressor("resnet18")
    ckpt = tmp_path / "left_eye.pt"
    torch.save(saved.state_dict(), ckpt)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a matching checkpoint must load with no missing/unexpected warnings
        loaded = load_eye_model(ckpt, base="resnet18", device="cpu")
    assert not loaded.training  # eval mode
    # the trained head and backbone fc are actually carried over from the checkpoint
    assert torch.equal(loaded.regression_head.weight, saved.regression_head.weight)
    assert torch.equal(loaded.resnet.fc.weight, saved.resnet.fc.weight)
    with torch.no_grad():
        assert loaded(torch.rand(1, 3, 16, 32)).shape == (1, 1)


def test_load_warns_when_regression_head_missing(tmp_path):
    saved = ResNetRegressor("resnet18")
    state = {k: v for k, v in saved.state_dict().items() if not k.startswith("regression_head")}
    ckpt = tmp_path / "no_head.pt"
    torch.save(state, ckpt)
    with pytest.warns(UserWarning, match="not found in checkpoint"):
        load_eye_model(ckpt, base="resnet18", device="cpu")
