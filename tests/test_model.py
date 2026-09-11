import torch
from pupilsense_repro.model import ResNetRegressor, load_eye_model


def test_forward_output_shape():
    model = ResNetRegressor("resnet18")
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 1)


def test_invalid_base_raises():
    import pytest
    with pytest.raises(ValueError):
        ResNetRegressor("resnet101")


def test_load_eye_model_roundtrip(tmp_path):
    saved = ResNetRegressor("resnet18")
    ckpt = tmp_path / "left_eye.pt"
    torch.save(saved.state_dict(), ckpt)
    loaded = load_eye_model(ckpt, base="resnet18", device="cpu")
    assert not loaded.training  # eval mode
    with torch.no_grad():
        assert loaded(torch.randn(1, 3, 224, 224)).shape == (1, 1)
