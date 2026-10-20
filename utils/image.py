"""Consistent image IO using NumPy/Pillow; no per-image contrast stretching."""

from pathlib import Path

import numpy as np
from PIL import Image
import torch

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm"}


def load_luminance(path, visible=False):
    with Image.open(path) as image:
        if image.mode in ("I;16", "I;16B", "I;16L", "I"):
            values = np.array(image, dtype=np.float32) / 65535.
        elif image.mode == "F":
            values = np.array(image, dtype=np.float32)
        elif image.mode == "L":
            values = np.array(image, dtype=np.float32) / 255.
        else:
            rgb = np.array(image.convert("RGB"), dtype=np.float32) / 255.
            # Use the same intensity convention for RGB-stored IR and visible.
            values = rgb @ np.array([.299, .587, .114], dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise ValueError(f"Unsupported image intensities/shape in {path}; expected finite [0,1]")
    return torch.from_numpy(np.ascontiguousarray(values[None]))


def save_fused(tensor, path, visible_path=None, color=False):
    values = tensor.detach().float().cpu().reshape(tensor.shape[-2:]).numpy()
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected a finite single-channel fused image")
    gray = np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)
    image = Image.fromarray(gray)
    if color:
        with Image.open(visible_path) as source:
            _, cb, cr = source.convert("YCbCr").split()
        if cb.size != image.size:
            raise ValueError("Color source and fused result have different sizes")
        image = Image.merge("YCbCr", (image, cb, cr)).convert("RGB")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
