import warnings

import pytest
import torch
from pupilsense_repro.model import ResNetRegressor, load_eye_model


def test_forward_output_shape():
    model = ResNetRegressor("resnet18")
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 1)


def test_invalid_base_raises():
    with pytest.raises(ValueError):
        ResNetRegressor("resnet101")


def test_load_eye_model_roundtrip(tmp_path):
    saved = ResNetRegressor("resnet18")
    ckpt = tmp_path / "left_eye.pt"
    torch.save(saved.state_dict(), ckpt)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded = load_eye_model(ckpt, base="resnet18", device="cpu")
    assert not loaded.training  # eval mode
    with torch.no_grad():
        assert loaded(torch.randn(1, 3, 224, 224)).shape == (1, 1)
    assert torch.equal(loaded.resnet.fc.weight, saved.resnet.fc.weight)


def test_load_eye_model_warns_when_head_missing(tmp_path):
    saved = ResNetRegressor("resnet18")
    state_dict = {k: v for k, v in saved.state_dict().items() if not k.startswith("resnet.fc")}
    ckpt = tmp_path / "left_eye_no_head.pt"
    torch.save(state_dict, ckpt)
    with pytest.warns(UserWarning, match="Regression head"):
        load_eye_model(ckpt, base="resnet18", device="cpu")
