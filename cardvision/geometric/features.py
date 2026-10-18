# SPDX-License-Identifier: Apache-2.0
"""Local features: keypoints and descriptors from OpenCV, for a card image or a crop.

Every call goes through OpenCV's documented Python API (``cv2.ORB_create``,
``cv2.AKAZE_create``, ``cv2.SIFT_create``, ``detectAndCompute``). OpenCV is the one
dependency of the stage, from the ``geometric`` extra; nothing here is imported until the stage
is used.

An image is converted to grey and scaled so that its longer side is a set number of pixels, so
that a card image and a crop meet at comparable scales. The keypoints are kept in the scaled
image's pixels. When more keypoints are found than are wanted, the strongest are kept (by
OpenCV's response, ties in the order found), so the result does not depend on the detector's
own cap.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

INSTALL_HINT = 'the geometric stage needs OpenCV: python -m pip install -e ".[geometric]"'


def cv2_module() -> Any:
    """The ``cv2`` module, or an ``ImportError`` that says which extra to install."""
    try:
        import cv2
    except ImportError:
        raise ImportError(INSTALL_HINT) from None
    return cv2


@dataclass(frozen=True)
class Features:
    """Keypoints of one image: ``xy`` ``(K, 2)`` float32 pixel positions in the scaled image,
    ``descriptors`` ``(K, B)`` (uint8 for binary descriptors, float32 for SIFT), and the scaled
    image's ``size`` as ``(width, height)``."""

    xy: np.ndarray
    descriptors: np.ndarray
    size: tuple[int, int]

    def __len__(self) -> int:
        return len(self.xy)


def is_binary(kind: str) -> bool:
    """Whether a feature type's descriptors are bit strings, compared by Hamming distance."""
    return kind in ("orb", "akaze")


def descriptor_dtype(kind: str) -> np.dtype:
    return np.dtype(np.uint8) if is_binary(kind) else np.dtype("<f4")


def norm(kind: str) -> int:
    cv2 = cv2_module()
    return cv2.NORM_HAMMING if is_binary(kind) else cv2.NORM_L2


def make_detector(kind: str, max_features: int, params: Mapping[str, Any]) -> Any:
    """An OpenCV feature detector and extractor. ``params`` are the type's settings
    (``cardvision.geometric.config.FEATURE_PARAMS``)."""
    cv2 = cv2_module()
    if kind == "orb":
        return cv2.ORB_create(nfeatures=int(max_features), scaleFactor=float(params["scaleFactor"]),
                              nlevels=int(params["nLevels"]), edgeThreshold=int(params["edgeThreshold"]),
                              patchSize=int(params["patchSize"]), fastThreshold=int(params["fastThreshold"]))
    if kind == "akaze":
        create = getattr(cv2, "AKAZE_create", None)
        if create is None:
            raise ValueError(f"this OpenCV ({cv2.__version__}) has no AKAZE; OpenCV 5 moved it out of the main "
                             'package. Use "orb" or "sift", or an OpenCV 4 build')
        return create(threshold=float(params["threshold"]), nOctaves=int(params["nOctaves"]),
                      nOctaveLayers=int(params["nOctaveLayers"]))
    if kind == "sift":
        return cv2.SIFT_create(nfeatures=int(max_features), nOctaveLayers=int(params["nOctaveLayers"]),
                               contrastThreshold=float(params["contrastThreshold"]),
                               edgeThreshold=float(params["edgeThreshold"]), sigma=float(params["sigma"]))
    raise ValueError(f"unknown feature type {kind!r}")


def scaled_grey(rgb: np.ndarray, long_side: int) -> np.ndarray:
    """An ``(H, W, 3)`` uint8 RGB image in grey, scaled so its longer side is ``long_side``
    pixels (area averaging to shrink, bilinear to enlarge)."""
    cv2 = cv2_module()
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = grey.shape[:2]
    scale = long_side / max(height, width)
    if scale == 1:
        return grey
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(grey, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


def mask_image(size: tuple[int, int], rectangles: Sequence[Sequence[float]]) -> np.ndarray | None:
    """The detection mask of a ``(width, height)`` image: 255 where keypoints may be taken, 0
    inside each ``[x0, y0, x1, y1]`` fraction rectangle. ``None`` when there is no rectangle."""
    if not rectangles:
        return None
    width, height = size
    mask = np.full((height, width), 255, dtype=np.uint8)
    for x0, y0, x1, y1 in rectangles:
        mask[int(round(y0 * height)):int(round(y1 * height)), int(round(x0 * width)):int(round(x1 * width))] = 0
    return mask


def extract(rgb: np.ndarray, kind: str, long_side: int, max_features: int, params: Mapping[str, Any],
            mask: Sequence[Sequence[float]] = ()) -> Features:
    """The strongest ``max_features`` keypoints of an RGB image, scaled to ``long_side``."""
    return extract_grey(scaled_grey(rgb, long_side), kind, max_features, params, mask)


def extract_grey(grey: np.ndarray, kind: str, max_features: int, params: Mapping[str, Any],
                 mask: Sequence[Sequence[float]] = ()) -> Features:
    """``extract`` from the scaled grey image itself (``scaled_grey``'s output)."""
    size = (grey.shape[1], grey.shape[0])
    detector = make_detector(kind, max_features, params)
    keypoints, descriptors = detector.detectAndCompute(grey, mask_image(size, mask))
    dtype = descriptor_dtype(kind)
    if descriptors is None or not keypoints:
        width = detector.descriptorSize() if kind != "akaze" else 61
        return Features(np.zeros((0, 2), np.float32), np.zeros((0, width), dtype), size)
    order = sorted(range(len(keypoints)), key=lambda i: -keypoints[i].response)[:max_features]
    xy = np.array([keypoints[i].pt for i in order], dtype=np.float32).reshape(-1, 2)
    return Features(xy, np.ascontiguousarray(descriptors[order]).astype(dtype), size)
