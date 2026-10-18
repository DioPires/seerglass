# SPDX-License-Identifier: Apache-2.0
"""Read an image file into the array layout the rest of the package uses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: str | Path) -> np.ndarray:
    """Read an image file as an ``(H, W, 3)`` uint8 RGB array.

    Any mode the file uses (palette, grayscale, alpha) is converted to RGB by Pillow; an alpha
    channel is dropped, not composited. The array is a writable copy.
    """
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))
