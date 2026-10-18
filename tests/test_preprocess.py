# SPDX-License-Identifier: Apache-2.0
"""The preprocessing contracts, checked against a per-pixel reference and pinned values."""

from __future__ import annotations

import numpy as np
import pytest

from cardvision.core import preprocess as P
from helpers import digest, pattern

# Crop sizes below, at and above the 128-pixel grid, square and not.
SIZES = [(8, 8), (8, 600), (600, 8), (24, 17), (50, 37), (127, 129), (128, 128), (129, 127), (200, 143), (600, 600)]
GRIDS = [(128, 128), (96, 96), (64, 48), (9, 8)]  # (out_w, out_h)


def reference_pool(rgb: np.ndarray, out_w: int, out_h: int) -> tuple[np.ndarray, np.ndarray]:
    """The pool written as the plain per-pixel loop of its definition: (cell means, counts)."""
    h, w, _ = rgb.shape
    total = np.zeros((out_h, out_w, 3), dtype=np.float64)
    counts = np.zeros((out_h, out_w), dtype=np.int64)
    for y in range(h):
        for x in range(w):
            oy = min(out_h - 1, (y * out_h) // h)
            ox = min(out_w - 1, (x * out_w) // w)
            total[oy, ox] += rgb[y, x]
            counts[oy, ox] += 1
    return total / np.maximum(counts, 1)[:, :, None], counts


def reference_fill(pooled: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Version 2's fill, cell by cell: copy from the nearest filled row and column, ties low."""
    out_h, out_w = counts.shape
    rows = [oy for oy in range(out_h) if counts[oy].any()]
    cols = [ox for ox in range(out_w) if counts[:, ox].any()]

    def nearest(index: int, filled: list[int]) -> int:
        return min(filled, key=lambda f: (abs(f - index), f))

    out = np.empty_like(pooled)
    for oy in range(out_h):
        for ox in range(out_w):
            out[oy, ox] = pooled[nearest(oy, rows), nearest(ox, cols)]
    return out


def random_crop(seed: int, height: int, width: int) -> np.ndarray:
    return np.random.default_rng([seed, height, width]).integers(0, 256, size=(height, width, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- box_average

@pytest.mark.parametrize("size", [(8, 8), (24, 17), (50, 37), (127, 129), (129, 127), (200, 143)])
@pytest.mark.parametrize("grid", GRIDS)
def test_pool_matches_the_per_pixel_definition(size, grid):
    rgb = random_crop(1, *size)
    out_w, out_h = grid
    means, counts = reference_pool(rgb, out_w, out_h)
    v1 = P.box_average(rgb, out_w, out_h, P.POOL_V1)
    v2 = P.box_average(rgb, out_w, out_h, P.POOL_V2)
    assert v1.dtype == np.float64 and v1.shape == (out_h, out_w, 3)
    assert np.array_equal(v1, means)
    assert np.array_equal(v2, reference_fill(means, counts))


def test_pool_takes_exact_means():
    rgb = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    pooled = P.box_average(rgb, 2, 2, P.POOL_V1)
    expected = rgb.astype(np.float64).reshape(2, 2, 2, 2, 3).mean(axis=(1, 3))
    assert np.array_equal(pooled, expected)


def test_pool_cells_follow_the_floor_rule_when_the_size_does_not_divide():
    # Three columns into two cells: floor(x * 2 / 3) is 0, 0, 1.
    rgb = np.array([[[10, 10, 10], [20, 20, 20], [90, 90, 90]]], dtype=np.uint8)
    pooled = P.box_average(rgb, 2, 1, P.POOL_V1)
    assert pooled[0, :, 0].tolist() == [15.0, 90.0]


def test_version_1_leaves_unfilled_cells_at_zero_and_version_2_copies_the_nearest():
    # Two columns into eight cells: only cells 0 and 4 receive a pixel. Cell 2 is equally far
    # from both and takes the lower one.
    rgb = np.array([[[50, 60, 70], [200, 210, 220]]], dtype=np.uint8)
    v1 = P.box_average(rgb, 8, 1, P.POOL_V1)[0, :, 0]
    v2 = P.box_average(rgb, 8, 1, P.POOL_V2)[0, :, 0]
    assert v1.tolist() == [50, 0, 0, 0, 200, 0, 0, 0]
    assert v2.tolist() == [50, 50, 50, 200, 200, 200, 200, 200]


def test_version_2_fills_rows_and_columns_independently():
    rgb = np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]], dtype=np.uint8)
    v2 = P.box_average(rgb, 4, 4, P.POOL_V2)[:, :, 0]
    assert v2.tolist() == [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]


@pytest.mark.parametrize("size", [(128, 128), (129, 128), (200, 143), (600, 600), (128, 600)])
def test_versions_agree_when_the_crop_covers_the_grid(size):
    rgb = random_crop(2, *size)
    assert np.array_equal(P.box_average(rgb, 128, 128, P.POOL_V1), P.box_average(rgb, 128, 128, P.POOL_V2))


@pytest.mark.parametrize("size", [(8, 8), (8, 600), (600, 8), (50, 37), (127, 129)])
def test_version_2_leaves_no_cell_empty(size):
    rgb = np.maximum(random_crop(3, *size), 1)  # every pixel non-zero, so a zero cell is an empty one
    assert (P.box_average(rgb, 128, 128, P.POOL_V1) == 0).any()
    assert (P.box_average(rgb, 128, 128, P.POOL_V2) > 0).all()


def test_pool_stays_within_the_input_range():
    rgb = random_crop(4, 77, 311)
    for version in P.POOL_VERSIONS:
        pooled = P.box_average(rgb, 128, 128, version)
        assert pooled.min() >= 0 and pooled.max() <= 255


def test_a_constant_crop_pools_to_the_same_constant():
    rgb = np.full((33, 71, 3), 137, dtype=np.uint8)
    assert (P.box_average(rgb, 128, 128, P.POOL_V2) == 137).all()


def test_pool_rejects_bad_arguments():
    rgb = random_crop(5, 10, 10)
    with pytest.raises(ValueError, match="unknown pool version"):
        P.box_average(rgb, 8, 8, 3)
    with pytest.raises(ValueError, match="at least 1 x 1"):
        P.box_average(rgb, 0, 8, P.POOL_V1)
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        P.box_average(rgb[:, :, 0], 8, 8, P.POOL_V1)
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        P.box_average(np.zeros((10, 10, 4), np.uint8), 8, 8, P.POOL_V1)
    with pytest.raises(ValueError, match="empty crop"):
        P.box_average(np.zeros((0, 10, 3), np.uint8), 8, 8, P.POOL_V1)


# --------------------------------------------------------------------------- normalize and friends

def test_normalize_maps_the_byte_range_to_minus_one_one_channel_first():
    pooled = np.zeros((2, 3, 3), dtype=np.float64)
    pooled[:, :, 0], pooled[:, :, 1], pooled[:, :, 2] = 0.0, 127.5, 255.0
    tensor = P.normalize(pooled)
    assert tensor.shape == (3, 2, 3) and tensor.dtype == np.float32 and tensor.flags.c_contiguous
    assert (tensor[0] == -1).all() and (tensor[1] == 0).all() and (tensor[2] == 1).all()


def test_normalize_works_in_float32_steps_in_the_documented_order():
    pooled = np.random.default_rng(6).uniform(0, 255, size=(5, 7, 3))
    v = pooled.astype(np.float32)
    expected = ((v / np.float32(255.0)) - np.float32(0.5)) / np.float32(0.5)
    assert np.array_equal(P.normalize(pooled), expected.transpose(2, 0, 1))


def test_to_model_tensor_is_pool_then_normalize():
    rgb = random_crop(7, 61, 45)
    for version in P.POOL_VERSIONS:
        expected = P.normalize(P.box_average(rgb, 96, 80, version))
        tensor = P.to_model_tensor(rgb, 96, 80, version)
        assert tensor.shape == (3, 80, 96) and np.array_equal(tensor, expected)


def test_to_uint8_image_rounds_half_to_even():
    low = np.array([[[0, 0, 0], [1, 1, 1]]], dtype=np.uint8)   # mean 0.5 -> 0
    high = np.array([[[1, 1, 1], [2, 2, 2]]], dtype=np.uint8)  # mean 1.5 -> 2
    assert P.to_uint8_image(low, 1, 1, P.POOL_V1).tolist() == [[[0, 0, 0]]]
    assert P.to_uint8_image(high, 1, 1, P.POOL_V1).tolist() == [[[2, 2, 2]]]
    image = P.to_uint8_image(random_crop(8, 90, 64), 32, 48, P.POOL_V2)
    assert image.shape == (48, 32, 3) and image.dtype == np.uint8


# --------------------------------------------------------------------------- full128

@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_full128_is_deterministic_and_shape_correct(size, seed):
    rgb = random_crop(seed, *size)
    before = rgb.copy()
    first = P.full128(rgb)
    second = P.full128(rgb.copy())
    assert first.shape == (3, 128, 128) and first.dtype == np.float32 and first.flags.c_contiguous
    assert np.isfinite(first).all() and first.min() >= -1 and first.max() <= 1
    assert np.array_equal(first, second)
    assert np.array_equal(rgb, before), "the crop must not be modified"


def test_full128_is_the_version_2_pool_at_128_normalized():
    rgb = random_crop(9, 70, 50)
    assert np.array_equal(P.full128(rgb), P.to_model_tensor(rgb, 128, 128, P.POOL_V2))
    assert P.FULL128_SIZE == 128 and P.FULL128 == "full128"


def test_full128_accepts_a_view_and_matches_its_copy():
    frame = random_crop(10, 300, 400)
    view = frame[40:215:1, 399:120:-1]  # a crop cut out of a frame, not contiguous
    assert not view.flags.c_contiguous
    assert np.array_equal(P.full128(view), P.full128(np.ascontiguousarray(view)))


def test_full128_of_a_small_crop_repeats_pixels_without_blending():
    rgb = random_crop(11, 8, 8)
    tensor = P.full128(rgb)
    # 8 pixels over 128 cells: each pixel lands on every 16th cell, and the fill repeats it.
    values = np.unique(tensor[0])
    assert len(values) <= len(np.unique(rgb[:, :, 0]))


# Pinned results on an arithmetic test pattern. These are the contract: a change here means
# every embedding changes. The sums are sums of integers and each later step is a single IEEE
# operation, so the values do not depend on the platform.
PINNED_FULL128 = {
    (50, 37): "775c4b14a24cc5f210340faec965cade1a526dd8f47ea0f843ae7fe9c690e122",
    (128, 128): "1b6faae743ef3cf610e18edb7c2e2024e4940eef94c12513a9c338862c8e7b5f",
    (300, 211): "0e63a36e1ec57d3fd96b66ff61d33a7b4918acf75de14979982afdd6467f8bea",
    (9, 600): "7ee22edfe4a81ccb407a40c606f09805d550139f7ea5623c105c50d84f077560",
}


@pytest.mark.parametrize("size", sorted(PINNED_FULL128))
def test_full128_pinned_values(size):
    assert digest(P.full128(pattern(*size))) == PINNED_FULL128[size]


def test_pool_pinned_values():
    small, large = pattern(50, 37), pattern(300, 211)
    assert digest(P.box_average(small, 96, 96, 1)) == "01856a204b64248619a72903dd12dc83d32ce8d31640089d878a4e023ab4b2dd"
    assert digest(P.box_average(small, 96, 96, 2)) == "4045505976f4d066895c45110ebf2c620e0918eafca2dbe18306ecb32602ed4e"
    assert digest(P.box_average(large, 64, 48, 1)) == "7e1b1d7d7cf7c7aaf4f9d5be5ecfec358f8f0b7768c58b8d049e6602f90d2617"
    assert digest(P.to_uint8_image(large, 32, 32, 2)) == "18b08f749d5ae32fb1fddffff9f0449cd4a36a99630c77b6cc34967bed33d9c4"


# --------------------------------------------------------------------------- rot4

def test_rot4_turns_counter_clockwise_one_quarter_per_look():
    tensor = np.array([[[1, 2], [3, 4]]], dtype=np.float32)  # one channel, 2 x 2
    looks = P.rot4(tensor)
    assert looks.shape == (4, 1, 2, 2) and looks.dtype == np.float32
    assert looks[0, 0].tolist() == [[1, 2], [3, 4]]
    assert looks[1, 0].tolist() == [[2, 4], [1, 3]]
    assert looks[2, 0].tolist() == [[4, 3], [2, 1]]
    assert looks[3, 0].tolist() == [[3, 1], [4, 2]]


@pytest.mark.parametrize("size", [(8, 8), (50, 37), (300, 211)])
def test_rot4_of_a_model_tensor(size):
    tensor = P.full128(random_crop(12, *size))
    looks = P.rot4(tensor)
    assert looks.shape == (4, 3, 128, 128)
    assert np.array_equal(looks[0], tensor)
    for k in range(4):
        assert np.array_equal(looks[k], np.rot90(tensor, k, axes=(1, 2)))
        # channels are never mixed by a turn
        for channel in range(3):
            assert np.array_equal(np.sort(looks[k][channel], axis=None), np.sort(tensor[channel], axis=None))
    assert np.array_equal(np.rot90(looks[3], 1, axes=(1, 2)), tensor)


def test_rot4_rejects_tensors_that_are_not_square():
    with pytest.raises(ValueError, match="square"):
        P.rot4(np.zeros((3, 128, 96), np.float32))
    with pytest.raises(ValueError, match="square"):
        P.rot4(np.zeros((128, 128), np.float32))


# --------------------------------------------------------------------------- the registry

def test_contract_registry():
    assert sorted(P.CONTRACTS) == ["full128"]
    assert P.preprocessor("full128") is P.full128
    assert P.input_size("full128") == 128
    with pytest.raises(ValueError, match="unknown preprocessing contract"):
        P.preprocessor("nope")
    with pytest.raises(ValueError, match="unknown preprocessing contract"):
        P.input_size("nope")
