# SPDX-License-Identifier: Apache-2.0
"""Crop preprocessing contracts: how an RGB crop becomes a model input tensor.

The recognizer was trained on tensors produced by exactly these operations and the embedding
bank was built with them, so any other runtime that loads the same weights (a browser, for
example) has to repeat them step for step. Treat this module as a contract, not as an
implementation detail: any change to the arithmetic or to the order of operations changes the
embeddings and invalidates every published bank.

The pool
--------
A crop of ``H x W`` pixels is reduced to a grid of ``out_h x out_w`` cells. Source pixel
``(x, y)`` belongs to cell ``(min(out_w - 1, floor(x * out_w / W)), min(out_h - 1,
floor(y * out_h / H)))``, and a cell's value is the mean of its pixels, per channel, summed in
float64 and divided once. Library resizers are deliberately not used: their filters differ
between libraries and versions, while this rule can be written identically in any language.

Two versions exist, and they differ only when the crop is smaller than the grid along an axis,
so that some cells receive no pixel:

* version 1 leaves such cells at 0;
* version 2 copies, per axis, the nearest row (column) that did receive pixels; on an exact tie
  the lower index wins. The copy is a pure index selection, with no interpolation.

For a crop at least as large as the grid on both axes the two versions give identical results.

Contracts
---------
``full128`` is the recognizer's input contract: the whole crop pooled to 128 x 128 with
version 2, then normalized to ``[-1, 1]`` and laid out channel-first.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

POOL_V1 = 1
POOL_V2 = 2
POOL_VERSIONS = (POOL_V1, POOL_V2)

FULL128 = "full128"
FULL128_SIZE = 128


def _check_crop(rgb: np.ndarray) -> None:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB array, got shape {rgb.shape}")
    if rgb.shape[0] == 0 or rgb.shape[1] == 0:
        raise ValueError(f"cannot preprocess an empty crop (shape {rgb.shape})")


def _pool_v1(rgb: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    h, w, _ = rgb.shape
    ys = np.minimum(out_h - 1, (np.arange(h) * out_h) // h)
    xs = np.minimum(out_w - 1, (np.arange(w) * out_w) // w)
    grid = np.zeros((out_h, out_w, 3), dtype=np.float64)
    counts = np.zeros((out_h, out_w), dtype=np.int64)
    cell = ys[:, None] * out_w + xs[None, :]  # (h, w): the flat cell index of every source pixel
    np.add.at(grid.reshape(-1, 3), cell.ravel(), rgb.reshape(-1, 3).astype(np.float64))
    np.add.at(counts.reshape(-1), cell.ravel(), 1)
    return grid / np.maximum(counts, 1)[:, :, None]


def _nearest_filled_index(n_src: int, n_out: int) -> np.ndarray:
    """For each output index along one axis: itself when it receives a source index, otherwise
    the nearest output index that does (an exact tie goes to the lower index)."""
    filled = np.unique(np.minimum(n_out - 1, (np.arange(n_src) * n_out) // n_src))
    idx = np.arange(n_out)
    pos = np.searchsorted(filled, idx)
    lo = filled[np.clip(pos - 1, 0, len(filled) - 1)]
    hi = filled[np.clip(pos, 0, len(filled) - 1)]
    near = np.where((idx - lo) <= (hi - idx), lo, hi)
    return np.where(np.isin(idx, filled), idx, near)


def _pool_v2(rgb: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    pooled = _pool_v1(rgb, out_w, out_h)
    h, w = rgb.shape[:2]
    if h < out_h:
        pooled = pooled[_nearest_filled_index(h, out_h)]
    if w < out_w:
        pooled = pooled[:, _nearest_filled_index(w, out_w)]
    return pooled


def box_average(rgb: np.ndarray, out_w: int, out_h: int, version: int) -> np.ndarray:
    """Pool an ``(H, W, 3)`` crop to ``(out_h, out_w, 3)`` float64 cell means on the 0..255 scale.

    ``version`` selects the pool rule (see the module docstring) and is required: a caller must
    say which contract it implements. The result is not rounded; it feeds ``normalize`` directly.
    """
    if version not in POOL_VERSIONS:
        raise ValueError(f"unknown pool version {version!r} (known: {POOL_VERSIONS})")
    if out_w < 1 or out_h < 1:
        raise ValueError(f"the output grid must be at least 1 x 1, got {out_w} x {out_h}")
    rgb = np.asarray(rgb)
    _check_crop(rgb)
    if version == POOL_V1:
        return _pool_v1(rgb, out_w, out_h)
    return _pool_v2(rgb, out_w, out_h)


def normalize(pooled: np.ndarray) -> np.ndarray:
    """Turn pooled ``(h, w, 3)`` cell means into a ``(3, h, w)`` float32 tensor in ``[-1, 1]``.

    The value is narrowed to float32 first and every later step is a float32 operation, in this
    order: divide by 255, subtract 0.5, divide by 0.5. The order is part of the contract.
    """
    v = pooled.astype(np.float32)
    v = v / np.float32(255.0)
    v = v - np.float32(0.5)
    v = v / np.float32(0.5)
    return np.ascontiguousarray(v.transpose(2, 0, 1))


def to_model_tensor(rgb: np.ndarray, out_w: int, out_h: int, version: int) -> np.ndarray:
    """Pool a crop and normalize it: ``(H, W, 3)`` in, ``(3, out_h, out_w)`` float32 out."""
    return normalize(box_average(rgb, out_w, out_h, version))


def to_uint8_image(rgb: np.ndarray, out_w: int, out_h: int, version: int) -> np.ndarray:
    """Pool a crop and round it back to an ``(out_h, out_w, 3)`` uint8 image.

    For previews and stored thumbnails. A model input should come from ``to_model_tensor`` on the
    original crop, which keeps the fractional cell means.
    """
    return np.clip(np.round(box_average(rgb, out_w, out_h, version)), 0, 255).astype(np.uint8)


def full128(rgb: np.ndarray) -> np.ndarray:
    """The recognizer's input contract: ``(H, W, 3)`` crop in, ``(3, 128, 128)`` float32 out.

    The whole crop is pooled, without masking or padding, so the aspect ratio is not preserved.
    """
    return normalize(box_average(np.ascontiguousarray(rgb), FULL128_SIZE, FULL128_SIZE, POOL_V2))


def rot4(tensor: np.ndarray) -> np.ndarray:
    """Stack the four looks of a square ``(C, S, S)`` tensor into ``(4, C, S, S)``.

    Look ``k`` is the tensor turned ``k`` quarter turns counter-clockwise in the image plane
    (``numpy.rot90`` over the two spatial axes), so look 0 is the tensor itself. The recognizer
    embeds all four because a card may lie in any of the four orientations.
    """
    if tensor.ndim != 3 or tensor.shape[1] != tensor.shape[2]:
        raise ValueError(f"expected a square (C, S, S) tensor, got shape {tensor.shape}")
    return np.stack([
        tensor,
        np.rot90(tensor, 1, axes=(1, 2)),
        np.rot90(tensor, 2, axes=(1, 2)),
        np.rot90(tensor, 3, axes=(1, 2)),
    ])


# Contract id -> (preprocessing function, side of the square input in pixels). A release names
# its contract by this id, and the runtime that loads the release must implement the same id.
CONTRACTS: dict[str, tuple[Callable[[np.ndarray], np.ndarray], int]] = {
    FULL128: (full128, FULL128_SIZE),
}


def preprocessor(contract: str) -> Callable[[np.ndarray], np.ndarray]:
    """The preprocessing function for a contract id."""
    if contract not in CONTRACTS:
        raise ValueError(f"unknown preprocessing contract {contract!r} (known: {sorted(CONTRACTS)})")
    return CONTRACTS[contract][0]


def input_size(contract: str) -> int:
    """The side, in pixels, of the square model input a contract produces."""
    if contract not in CONTRACTS:
        raise ValueError(f"unknown preprocessing contract {contract!r} (known: {sorted(CONTRACTS)})")
    return CONTRACTS[contract][1]
