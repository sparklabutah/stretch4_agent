"""JPEG encoding and the downscaled-pixel -> full-resolution mapping used by depth queries."""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image as PILImage


@dataclass(frozen=True)
class Encoded:
    jpeg: bytes
    width: int          # width/height of the image the agent sees
    height: int
    full_width: int
    full_height: int

    @property
    def scale(self) -> float:
        return self.width / self.full_width

    def to_full(self, x: float, y: float) -> tuple[int, int]:
        """Pixel in the downscaled image -> pixel in the full-resolution image (nearest, clamped)."""
        fx = int(x * self.full_width / self.width)
        fy = int(y * self.full_height / self.height)
        return min(max(fx, 0), self.full_width - 1), min(max(fy, 0), self.full_height - 1)

    def contains(self, x: float, y: float) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height


def encode_jpeg(rgb: np.ndarray, max_dim: int = 768, quality: int = 80) -> Encoded:
    h, w = rgb.shape[:2]
    img = PILImage.fromarray(rgb)
    longest = max(h, w)
    if longest > max_dim:
        s = max_dim / longest
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return Encoded(buf.getvalue(), img.width, img.height, w, h)
