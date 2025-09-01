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
        e2_c, u2_c = self._crop_to_match(e2, u2)
        d2 = self.dec2(torch.cat([u2_c, e2_c], dim=1))
        u1 = self.up1(d2)
        e1_c, u1_c = self._crop_to_match(e1, u1)
        d1 = self.dec1(torch.cat([u1_c, e1_c], dim=1))
        return self.head(d1)

    @staticmethod
    def _center_crop(t: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        _, _, h, w = t.shape
        th, tw = target_hw
        if h == th and w == tw:
            return t
        top = max((h - th) // 2, 0)
        left = max((w - tw) // 2, 0)
        bottom = top + th
        right = left + tw
        return t[:, :, top:bottom, left:right]

    def _crop_to_match(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Center-crop both tensors to the same spatial size (min of each).

        This guards against off-by-one size mismatches during upsampling/skip concat.
        """
        h = min(a.shape[-2], b.shape[-2])
        w = min(a.shape[-1], b.shape[-1])
        a2 = self._center_crop(a, (h, w))
        b2 = self._center_crop(b, (h, w))
        return a2, b2


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
