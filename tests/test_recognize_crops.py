# SPDX-License-Identifier: Apache-2.0
"""Crop extraction: the truncation convention, especially at the edges of the frame."""

from __future__ import annotations

import numpy as np
import pytest

from cardvision.core.constants import DETECTOR_MIN_CROP_PX
from cardvision.recognize import crop_edges, crop_size, crop_truncated, large_enough
from helpers import pattern

FRAME = pattern(60, 100)   # height 60, width 100


@pytest.mark.parametrize(
    ("box", "edges"),
    [
        ((10, 5, 20, 10), (10, 5, 30, 15)),                 # whole numbers pass through
        ((10.9, 5.2, 20.3, 10.9), (10, 5, 31, 16)),         # 10.9 -> 10, 31.2 -> 31; 5.2 -> 5, 16.1 -> 16
        ((10.99, 5.99, 0.02, 0.02), (10, 5, 11, 6)),        # a sliver that crosses a pixel boundary
        ((10.2, 5.2, 0.5, 0.5), (10, 5, 10, 5)),            # a sliver inside one pixel is empty
        ((0, 0, 100, 60), (0, 0, 100, 60)),                 # the whole frame
        ((-5.5, -3.2, 20.9, 20.1), (0, 0, 15, 16)),         # over the top-left corner: clipped at 0
        ((90.6, 50.2, 30, 30), (90, 50, 100, 60)),          # over the bottom-right corner: clipped at the size
        ((99.9, 59.9, 5, 5), (99, 59, 100, 60)),            # the last pixel
        ((-0.9, -0.9, 1.8, 1.8), (0, 0, 0, 0)),             # ends before the first pixel boundary
        ((-0.5, -0.5, 1.6, 1.6), (0, 0, 1, 1)),
    ],
)
def test_edges_are_clipped_then_truncated(box, edges):
    assert crop_edges(box, 100, 60) == edges
    x0, y0, x1, y1 = edges
    crop = crop_truncated(FRAME, box)
    assert crop.shape == (y1 - y0, x1 - x0, 3)
    np.testing.assert_array_equal(crop, FRAME[y0:y1, x0:x1])
    assert crop_size(box, 100, 60) == (x1 - x0, y1 - y0)


def test_truncation_is_not_rounding():
    # Rounding would give columns 11..31; truncation gives 10..31.
    assert crop_edges((10.6, 0, 20.6, 10), 100, 60)[:3:2] == (10, 31)
    # The right edge at 30.999 stays at 30, where rounding would reach 31.
    assert crop_edges((10, 0, 20.999, 10), 100, 60)[2] == 30


@pytest.mark.parametrize(
    "box",
    [
        (150, 10, 20, 20),      # right of the frame
        (10, 80, 20, 20),       # below the frame
        (-50, 10, 20, 20),      # left of the frame
        (10, -50, 20, 20),      # above the frame
        (-50, -50, 20, 20),
        (100, 60, 10, 10),      # starts exactly at the far corner
    ],
)
def test_a_box_outside_the_frame_gives_an_empty_crop(box):
    crop = crop_truncated(FRAME, box)
    assert crop.size == 0
    assert 0 in crop_size(box, 100, 60)


def test_the_crop_is_a_contiguous_copy():
    crop = crop_truncated(FRAME, (10, 5, 20, 10))
    assert crop.flags["C_CONTIGUOUS"] and crop.flags["OWNDATA"]
    crop[:] = 0
    assert FRAME[5:15, 10:30].any()


def test_minimum_crop_size_is_24_pixels():
    assert DETECTOR_MIN_CROP_PX == 24


@pytest.mark.parametrize(
    ("box", "expected"),
    [
        ((10, 10, 24, 24), True),
        ((10, 10, 23.9, 30), False),            # 10 .. 33.9 -> 10 .. 33: 23 wide
        ((10.5, 10, 23.5, 30), True),           # 10.5 .. 34.0 -> 10 .. 34: truncation makes it 24 wide
        ((10.9, 10, 23.2, 30), True),           # 10.9 .. 34.1 -> 10 .. 34
        ((10, 10, 30, 23.999), False),
        ((80, 10, 40, 30), False),              # clipped at the right edge: 20 wide
        ((76, 10, 40, 30), True),               # clipped to exactly 24
        ((-10, 10, 34, 30), True),              # clipped at the left edge: 0 .. 24
        ((-10, 10, 33.5, 30), False),
        ((10, 40, 30, 30), False),              # clipped at the bottom: 20 high
        ((200, 10, 30, 30), False),
    ],
)
def test_large_enough_is_judged_on_the_crop(box, expected):
    assert large_enough(box, 100, 60) is expected


def test_large_enough_takes_another_minimum():
    assert large_enough((0, 0, 10, 10), 100, 60, min_px=10) is True
    assert large_enough((0, 0, 10, 10), 100, 60, min_px=11) is False
