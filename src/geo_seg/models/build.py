from typing import Literal
import torch
from torch import nn


class TinyUNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)
        # Bottleneck
        self.bot = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())
        # Decoder
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Conv2d(16, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bot(p2)
        u2 = self.up2(b)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.head(d1)


class TinyDeepLab(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 1):
        super().__init__()
        # Minimal atrous-like module (not real DeepLab, just a valid placeholder)
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=4, dilation=4),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Conv2d(64, out_ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.backbone(x)
        # Keep size the same using padding above; otherwise, upsample
        if y.shape[-2:] != x.shape[-2:]:
            y = nn.functional.interpolate(
                y, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.head(y)


def create_model(name: Literal["unet", "deeplabv3"], num_classes: int) -> nn.Module:
    if name == "unet":
        return TinyUNet(in_ch=3, out_ch=num_classes)
    elif name == "deeplabv3":
        return TinyDeepLab(in_ch=3, out_ch=num_classes)
    else:
        raise ValueError(f"Unknown model name: {name}")
