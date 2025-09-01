from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image
import torch


try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR  # Pillow >= 9
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:  # pragma: no cover
    RESAMPLE_BILINEAR = Image.BILINEAR  # type: ignore[attr-defined]
    RESAMPLE_NEAREST = Image.NEAREST  # type: ignore[attr-defined]


@dataclass
class ResizePadInfo:
    orig_size: Tuple[int, int]
    resized_size: Tuple[int, int]
    canvas_size: Tuple[int, int]
    offset_xy: Tuple[int, int]


def resize_and_pad_to_square(img: Image.Image, image_size: int) -> tuple[Image.Image, ResizePadInfo]:
    """Resize keeping aspect ratio and pad to a square canvas of size (image_size, image_size).

    Returns the padded RGB image and metadata needed to map predictions back.
    """
    src = img.convert("RGB")
    w, h = src.size
    scale = min(image_size / w, image_size / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = src.resize((new_w, new_h), RESAMPLE_BILINEAR)

    canvas = Image.new("RGB", (image_size, image_size), color=(0, 0, 0))
    off_x = (image_size - new_w) // 2
    off_y = (image_size - new_h) // 2
    canvas.paste(resized, (off_x, off_y))

    info = ResizePadInfo(
        orig_size=(w, h),
        resized_size=(new_w, new_h),
        canvas_size=(image_size, image_size),
        offset_xy=(off_x, off_y),
    )
    return canvas, info


def to_model_tensor(img_rgb: Image.Image) -> torch.Tensor:
    """Convert an RGB PIL image in [0,255] to a batched CHW torch float tensor in [0,1]."""
    arr = np.asarray(img_rgb, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return t


def project_mask_back_to_original(mask_canvas: np.ndarray, info: ResizePadInfo) -> Image.Image:
    """Map a mask from canvas space (image_size x image_size) back to the original image size.

    Expects a 2D numpy array in {0,1}. Returns a PIL L image at original size.
    """
    off_x, off_y = info.offset_xy
    new_w, new_h = info.resized_size
    crop = mask_canvas[off_y : off_y + new_h, off_x : off_x + new_w]
    # Resize cropped mask back to original size using nearest to keep it binary-ish
    mask_small = Image.fromarray((crop * 255).astype(np.uint8), mode="L")
    mask_orig = mask_small.resize(info.orig_size, RESAMPLE_NEAREST)
    return mask_orig

