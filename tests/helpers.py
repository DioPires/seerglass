# SPDX-License-Identifier: Apache-2.0
"""Small helpers shared by the tests."""

from __future__ import annotations

import hashlib

import numpy as np

# Output width of the throwaway model built in conftest.py.
EMBEDDING_WIDTH = 128


def pattern(height: int, width: int) -> np.ndarray:
    """A deterministic ``(height, width, 3)`` uint8 image defined by arithmetic alone, so that
    values pinned against it do not depend on any random number generator."""
    y, x, c = np.meshgrid(np.arange(height), np.arange(width), np.arange(3), indexing="ij")
    return ((x * 7 + y * 13 + c * 31 + (x * y) % 11) % 256).astype(np.uint8)


def digest(array: np.ndarray) -> str:
    """SHA-256 of an array's dtype, shape and bytes."""
    h = hashlib.sha256()
    h.update(f"{array.dtype.str}{array.shape}".encode())
    h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()
