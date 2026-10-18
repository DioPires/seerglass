# SPDX-License-Identifier: Apache-2.0
"""Cut a detected box out of a frame, the way the recognizer was trained to see it.

A box has fractional edges and a crop has whole pixels. The convention is truncation: each of
the four edges is clipped to the frame and then cut down to a whole number, so the left and top
edges move in by a fraction of a pixel and the right and bottom edges move out by less than
one. The recognizer's training crops and the embedding bank's measurements were made this way;
rounding the edges instead shifts some crops by a pixel and changes the embedding slightly.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from cardvision.core.constants import DETECTOR_MIN_CROP_PX

Box = Sequence[float]  # x, y, width, height


def crop_edges(box: Box, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    """Whole-pixel edges ``(x0, y0, x1, y1)`` of a fractional ``(x, y, width, height)`` box.

    ``x1`` and ``y1`` are exclusive. A box that lies wholly outside the frame gives an empty
    range (``x1 == x0`` or ``y1 == y0``).
    """
    x, y, width, height = box
    x0, y0 = int(max(0, x)), int(max(0, y))
    x1, y1 = int(min(frame_width, x + width)), int(min(frame_height, y + height))
    return x0, y0, max(x0, x1), max(y0, y1)


def crop_size(box: Box, frame_width: int, frame_height: int) -> tuple[int, int]:
    """``(width, height)`` in pixels of the crop a box produces."""
    x0, y0, x1, y1 = crop_edges(box, frame_width, frame_height)
    return x1 - x0, y1 - y0


def crop_truncated(frame: np.ndarray, box: Box) -> np.ndarray:
    """The crop of an ``(H, W, 3)`` frame for a fractional ``(x, y, width, height)`` box.

    Returns a contiguous copy, which is empty when the box misses the frame.
    """
    frame_height, frame_width = frame.shape[:2]
    x0, y0, x1, y1 = crop_edges(box, frame_width, frame_height)
    return frame[y0:y1, x0:x1].copy()


def large_enough(box: Box, frame_width: int, frame_height: int, min_px: int = DETECTOR_MIN_CROP_PX) -> bool:
    """Whether a box's crop is at least ``min_px`` pixels on both sides.

    Smaller crops carry too little of the card to be compared, so such boxes are dropped before
    recognition. The test is made on the crop, after clipping and truncation, in the frame the
    detector analyzed.
    """
    width, height = crop_size(box, frame_width, frame_height)
    return width >= min_px and height >= min_px
