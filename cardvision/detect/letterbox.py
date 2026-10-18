# SPDX-License-Identifier: Apache-2.0
"""The detector's input contract: how an RGB frame becomes the detector's input tensor.

The frame is resized to fit a ``640 x 384`` canvas (width x height) without changing its aspect
ratio, placed in the top-left corner, and the rest of the canvas is filled with the gray value
114. Channels are reordered to blue, green, red and the values stay on the 0..255 scale: no
mean is subtracted and nothing is divided. These are the conventions of the YOLOX detector
family, which the detector was fine-tuned from, so they must be repeated exactly.

The resize is Pillow's bilinear filter. The detector was trained and measured with that filter;
another library's bilinear resize gives slightly different pixels and therefore slightly
different boxes.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

INPUT_WIDTH = 640
INPUT_HEIGHT = 384
PAD_VALUE = 114


def letterbox_ratio(frame_width: int, frame_height: int, width: int = INPUT_WIDTH, height: int = INPUT_HEIGHT) -> float:
    """The scale from frame pixels to canvas pixels: the largest that fits both axes."""
    return min(height / frame_height, width / frame_width)


def letterbox_size(frame_width: int, frame_height: int, width: int = INPUT_WIDTH, height: int = INPUT_HEIGHT) -> tuple[int, int]:
    """``(width, height)`` of the resized frame inside the canvas.

    Each side is the frame's side times ``letterbox_ratio``, rounded to the nearest integer
    (halves go to the even neighbor).
    """
    ratio = letterbox_ratio(frame_width, frame_height, width, height)
    return int(round(frame_width * ratio)), int(round(frame_height * ratio))


def letterbox_bgr(
    rgb: np.ndarray,
    width: int = INPUT_WIDTH,
    height: int = INPUT_HEIGHT,
    pad: int = PAD_VALUE,
) -> tuple[np.ndarray, float]:
    """Fit an ``(H, W, 3)`` uint8 RGB frame into a ``(height, width, 3)`` uint8 BGR canvas.

    Returns the canvas and the ratio. A coordinate on the canvas divided by the ratio is a
    coordinate in the frame; there is no offset, because the frame sits in the top-left corner.
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected an (H, W, 3) uint8 RGB array, got shape {rgb.shape} and dtype {rgb.dtype}")
    frame_height, frame_width = rgb.shape[:2]
    if frame_height == 0 or frame_width == 0:
        raise ValueError(f"cannot letterbox an empty frame (shape {rgb.shape})")
    ratio = letterbox_ratio(frame_width, frame_height, width, height)
    new_width, new_height = letterbox_size(frame_width, frame_height, width, height)
    if new_width < 1 or new_height < 1:
        raise ValueError(f"a {frame_width} x {frame_height} frame is too thin to fit a {width} x {height} canvas")
    canvas = np.full((height, width, 3), pad, np.uint8)
    resized = np.asarray(Image.fromarray(rgb).resize((new_width, new_height), Image.BILINEAR))
    canvas[:new_height, :new_width] = resized[:, :, ::-1]
    return canvas, ratio


def to_input_tensor(canvas: np.ndarray) -> np.ndarray:
    """Lay a ``(height, width, 3)`` canvas out as the ``(1, 3, height, width)`` float32 input."""
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32)
