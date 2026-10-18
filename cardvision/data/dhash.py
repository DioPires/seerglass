# SPDX-License-Identifier: Apache-2.0
"""A 64-bit fingerprint of a frame, to tell whether two images show the same frame.

The fingerprint is a *difference hash*. It is defined here completely, so that any program can
reproduce it:

1. **Grayscale.** Each pixel becomes ``round(0.299 R + 0.587 G + 0.114 B)``, with the sum taken
   in double precision and ``round`` going to the nearest whole number, halves to the even one.
2. **Nine by eight grid.** The frame is divided into 9 columns and 8 rows. The pixel at column
   ``x`` and row ``y`` of a frame ``W`` pixels wide and ``H`` high belongs to the cell
   ``(min(8, 9x // W), min(7, 8y // H))``, integer division. Each cell takes the mean gray of
   its pixels (0 for a cell that has none, which happens only in frames smaller than the grid).
3. **Bits.** For each of the 8 rows, from the top, and each of the first 8 cells of the row,
   from the left, the bit is 1 when the cell to its right is brighter than it, else 0. The 64
   bits are read in that order, the first as the most significant.

The result is written as 16 lower-case hexadecimal digits. Because the gray mean over large
cells is what it keeps, the fingerprint survives re-encoding, a change of resolution and light
compression: two decodes of one frame differ by a few bits at most, while two different frames
of a video usually differ by dozens. It holds only the coarse pattern of which parts of the
frame are brighter than their neighbors; the frame cannot be rebuilt from it.
"""

from __future__ import annotations

import re

import numpy as np

COLUMNS = 9
ROWS = 8
BITS = (COLUMNS - 1) * ROWS  # 64
HEX_DIGITS = BITS // 4  # 16
DHASH_PATTERN = r"^[0-9a-f]{16}$"

# Two fingerprints at most this many bits apart are taken to be of the same frame.
SAME_FRAME_MAX_HAMMING = 6

_FORM = re.compile(DHASH_PATTERN)


def is_dhash(text: object) -> bool:
    """Whether ``text`` is a fingerprint in the written form: exactly 16 lower-case hex digits."""
    return isinstance(text, str) and _FORM.fullmatch(text) is not None


def dhash(rgb: np.ndarray) -> str:
    """The fingerprint of a frame, given as an ``(H, W, 3)`` uint8 RGB array, as 16 lower-case
    hexadecimal digits. Raises ``ValueError`` for anything else, or an empty frame."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected an (H, W, 3) uint8 RGB array, got {rgb.dtype} with shape {rgb.shape}")
    height, width = rgb.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("the frame is empty")
    gray = np.round(0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2])
    rows = np.minimum(ROWS - 1, (np.arange(height) * ROWS) // height)
    columns = np.minimum(COLUMNS - 1, (np.arange(width) * COLUMNS) // width)
    cell = (rows[:, None] * COLUMNS + columns[None, :]).ravel()
    # Whole-number sums are exact in double precision, whatever the order of addition.
    sums = np.bincount(cell, weights=gray.ravel(), minlength=ROWS * COLUMNS)
    counts = np.bincount(cell, minlength=ROWS * COLUMNS)
    grid = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0).reshape(ROWS, COLUMNS)
    bits = (grid[:, 1:] > grid[:, :-1]).ravel()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:0{HEX_DIGITS}x}"


def hamming(a: str, b: str) -> int:
    """The number of bits in which two fingerprints differ, 0 to 64. Raises ``ValueError`` when
    either is not in the written form."""
    for name, text in (("first", a), ("second", b)):
        if not is_dhash(text):
            raise ValueError(f"the {name} fingerprint {text!r} is not 16 lower-case hexadecimal digits")
    return (int(a, 16) ^ int(b, 16)).bit_count()


__all__ = ["DHASH_PATTERN", "SAME_FRAME_MAX_HAMMING", "dhash", "hamming", "is_dhash"]
