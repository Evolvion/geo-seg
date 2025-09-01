import torch
from geo_seg.models.build import create_model


def test_unet_forward_shape():
    model = create_model("unet", num_classes=1)
    x = torch.randn(2, 3, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 64, 64)
