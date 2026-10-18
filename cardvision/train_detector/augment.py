# SPDX-License-Identifier: Apache-2.0
#
# Portions adapted from YOLOX (https://github.com/Megvii-BaseDetection/YOLOX),
# Copyright (c) Megvii Inc., Apache-2.0; modified:
#
#   * fit_tile, mosaic and the mosaic branch of Tiles.__getitem__ follow
#     MosaicDetection.__getitem__ in yolox/data/datasets/mosaicdetection.py at commit
#     6ddff4824372906469a7fae2dc3206c7aa4bbaee: a random center is drawn, each tile is
#     resized to fit the canvas, the four are pasted around the center on a gray canvas twice
#     the size, the boxes are shifted and clipped with them, and the canvas goes through
#     random_affine. Changed here: the three further tiles come from named streams in stated
#     shares instead of uniformly from one dataset; boxes arrive as fractions of their frame
#     and carry a face, back or ignore code that rides through every step; frames arrive in
#     RGB; a tile may be degraded first; there is no mix-up; the steps are split into
#     functions.
#   * finish and pack keep the order of TrainTransform.__call__ in
#     yolox/data/data_augment.py at the same commit (color jitter, mirror, fit to the canvas
#     with gray padding, drop boxes with a side of 1 px or less), written out again. Changed
#     here: the jitter always runs; an image left without boxes stays without boxes; ignore
#     regions are separated from the targets and returned beside them.
#
# Both upstream files carry the notice "Copyright (c) Megvii, Inc. and its affiliates."
# The library's own augment_hsv, random_affine and get_mosaic_coordinate are called, not copied.
"""Training images for the detector: tiles drawn from streams, mosaics, and the final canvas.

One training image is made like this (``Tiles.__getitem__``):

1. A first tile is drawn from the stream that the batch position belongs to
   (``[mix] primary``), uniformly over that stream's frames.
2. With probability ``mosaic`` the image is a mosaic: three more tiles are drawn, each from a
   stream picked by the ``mosaic_extra`` shares, the four are pasted around a random center and
   the result is rotated, shifted, zoomed and sheared at random.
3. Hue, saturation and value are jittered, the image is flipped left to right half the time,
   and it is fitted to the 640 x 384 canvas, top-left, padded with gray.
4. Targets with a side of 1 px or less are dropped. Faces and backs become rows of
   ``(class, cx, cy, w, h)`` in canvas pixels; ignore regions become rows of
   ``(cx, cy, w, h)`` beside them.

During the closing phase of training (``no_augment_share``) step 2 is skipped.

The canvas is blue-green-red on the 0..255 scale: the detector's input contract
(``cardvision.detect.letterbox``).

Every random draw of an image is fixed by the run's seed and the image's position in the run,
so the images do not depend on how many loader processes prepare them. The library functions
called here read Python's and NumPy's global generators, which is why ``__getitem__`` seeds
both; a caller that relies on those generators should make its tiles in worker processes.
"""

from __future__ import annotations

import hashlib
import io
import random
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from cardvision.detect.letterbox import INPUT_HEIGHT, INPUT_WIDTH, PAD_VALUE
from cardvision.train_detector.config import Config
from cardvision.train_detector.sources import BACK, FACE, IGNORE, Stream

MIN_SIDE_PX = 1.0
MOSAIC_TILES = 4


def degrade(image: Image.Image, rng: np.random.RandomState) -> Image.Image:
    """Make a clean frame look like compressed video: half the time a Gaussian blur of radius
    0.3 to 0.9 px, then always a JPEG round trip at a quality between 35 and 75."""
    if rng.random_sample() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.3, 0.9))))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=int(rng.randint(35, 76)))
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


def fit_tile(rgb: np.ndarray, targets: Sequence[Mapping[str, Any]], width: int = INPUT_WIDTH,
             height: int = INPUT_HEIGHT) -> tuple[np.ndarray, np.ndarray]:
    """Resize a frame to fit the canvas and express its targets in the resized frame's pixels.

    Returns the frame in blue-green-red and an ``(n, 5)`` float32 array of
    ``(left, top, right, bottom, k)``. The resize is OpenCV's bilinear filter and the new size is
    truncated to whole pixels, so a frame may come out a pixel short of the canvas.
    """
    import cv2

    frame_height, frame_width = rgb.shape[:2]
    ratio = min(height / frame_height, width / frame_width)
    image = cv2.resize(rgb[:, :, ::-1], (int(frame_width * ratio), int(frame_height * ratio)),
                       interpolation=cv2.INTER_LINEAR)
    h, w = image.shape[:2]
    boxes = np.zeros((len(targets), 5), np.float32)
    for index, target in enumerate(targets):
        boxes[index] = ((target["cx"] - target["w"] / 2) * w, (target["cy"] - target["h"] / 2) * h,
                        (target["cx"] + target["w"] / 2) * w, (target["cy"] + target["h"] / 2) * h, target["k"])
    return np.ascontiguousarray(image), boxes


def mosaic(tiles: Sequence[tuple[np.ndarray, np.ndarray]], center_x: int, center_y: int, *,
           degrees: float, translate: float, scale: tuple[float, float], shear: float,
           width: int = INPUT_WIDTH, height: int = INPUT_HEIGHT) -> tuple[np.ndarray, np.ndarray]:
    """Paste four fitted tiles around a center and warp the result back to the canvas size.

    ``tiles`` are ``(image, boxes)`` pairs from ``fit_tile``, in the order top-left, top-right,
    bottom-left, bottom-right. The warp's rotation, zoom, shear and shift are drawn by the
    library's ``random_affine`` from Python's global generator.
    """
    from yolox.data.data_augment import random_affine
    from yolox.data.datasets.mosaicdetection import get_mosaic_coordinate

    canvas = np.full((height * 2, width * 2, 3), PAD_VALUE, dtype=np.uint8)
    moved = []
    for position, (image, boxes) in enumerate(tiles):
        h, w = image.shape[:2]
        (lx1, ly1, lx2, ly2), (sx1, sy1, sx2, sy2) = get_mosaic_coordinate(
            canvas, position, center_x, center_y, w, h, height, width)
        canvas[ly1:ly2, lx1:lx2] = image[sy1:sy2, sx1:sx2]
        pad_x, pad_y = lx1 - sx1, ly1 - sy1
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, [0, 2]] += pad_x
            boxes[:, [1, 3]] += pad_y
        moved.append(boxes)
    boxes = np.concatenate(moved, 0)
    np.clip(boxes[:, 0], 0, 2 * width, out=boxes[:, 0])
    np.clip(boxes[:, 1], 0, 2 * height, out=boxes[:, 1])
    np.clip(boxes[:, 2], 0, 2 * width, out=boxes[:, 2])
    np.clip(boxes[:, 3], 0, 2 * height, out=boxes[:, 3])
    image, boxes = random_affine(canvas, boxes, target_size=(width, height), degrees=degrees,
                                 translate=translate, scales=scale, shear=shear)
    return np.ascontiguousarray(image), boxes


def finish(image: np.ndarray, boxes: np.ndarray, flip: float, width: int = INPUT_WIDTH,
           height: int = INPUT_HEIGHT) -> tuple[np.ndarray, np.ndarray]:
    """Jitter the colors, flip half the time, and fit the image to the canvas.

    The jitter draws from NumPy's global generator and the flip from Python's. Returns the
    ``(height, width, 3)`` uint8 canvas and the boxes in canvas pixels.
    """
    import cv2
    from yolox.data.data_augment import augment_hsv

    augment_hsv(image)
    if random.random() < flip:
        image = np.ascontiguousarray(image[:, ::-1])
        if len(boxes):
            left = image.shape[1] - boxes[:, 2]
            right = image.shape[1] - boxes[:, 0]
            boxes[:, 0], boxes[:, 2] = left, right
    canvas = np.full((height, width, 3), PAD_VALUE, np.uint8)
    h, w = image.shape[:2]
    ratio = min(height / h, width / w)
    if ratio != 1.0:
        image = cv2.resize(image, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_LINEAR)
        boxes[:, :4] *= ratio
    canvas[:image.shape[0], :image.shape[1]] = image
    return canvas, boxes


def pack(boxes: np.ndarray, max_targets: int, max_ignore: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split boxes into the arrays the loss takes.

    ``boxes`` is ``(n, 5)``: ``(left, top, right, bottom, k)`` in canvas pixels. Boxes whose width or
    height is 1 px or less are dropped. Returns

    * ``targets``, ``(max_targets, 5)`` float32: ``(class, cx, cy, w, h)`` for each face
      (class 0) and back (class 1), in the order given, then rows of zeros;
    * ``ignore``, ``(max_ignore, 4)`` float32: ``(cx, cy, w, h)`` for each ignore region, then
      rows of zeros;
    * ``counts``, three integers: faces and backs, ignore regions, backs, each counted before
      the arrays' limits cut anything off.
    """
    if len(boxes):
        centered = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2,
                             boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], 1)
        large = np.minimum(centered[:, 2], centered[:, 3]) > MIN_SIDE_PX
        wanted = large & (boxes[:, 4] != IGNORE)
        found, classes = centered[wanted], np.where(boxes[wanted, 4] == BACK, float(BACK), float(FACE))
        regions = centered[large & (boxes[:, 4] == IGNORE)]
    else:
        found, classes, regions = np.zeros((0, 4), np.float32), np.zeros(0), np.zeros((0, 4), np.float32)
    targets = np.zeros((max_targets, 5), np.float32)
    kept = min(len(found), max_targets)
    targets[:kept, 0] = classes[:kept]
    targets[:kept, 1:] = found[:kept]
    ignore = np.zeros((max_ignore, 4), np.float32)
    kept = min(len(regions), max_ignore)
    ignore[:kept] = regions[:kept]
    counts = np.array([len(found), len(regions), int((classes == BACK).sum())], np.int64)
    return targets, ignore, counts


def tile_seed(salt: str, seed: int, index: int) -> int:
    """The 32-bit seed of training image ``index`` in a run."""
    return int.from_bytes(hashlib.sha256(f"{salt}/{seed}/{index}".encode()).digest()[:4], "little")


class Tiles:
    """The training images of one run, in order: ``steps * batch_size`` of them.

    ``tiles[j]`` is image ``j``: ``(image, targets, ignore, counts)``, where ``image`` is the
    canvas as a ``(3, height, width)`` float32 array and the rest is what ``pack`` returns.
    Image ``j`` belongs to step ``j // batch_size`` and to batch position ``j % batch_size``.
    The object can be handed to ``torch.utils.data.DataLoader`` as it is, without shuffling.
    """

    def __init__(self, streams: Mapping[str, Stream], config: Config, width: int = INPUT_WIDTH,
                 height: int = INPUT_HEIGHT) -> None:
        self.streams = dict(streams)
        self.primary = config.primary_order()
        self.extra = list(config.mosaic_extra)
        missing = sorted({*self.primary, *(name for name, _ in self.extra)} - set(self.streams))
        if missing:
            raise ValueError(f"the mix names streams that were not built: {', '.join(missing)}")
        self.steps, self.batch_size = config.steps, config.batch_size
        self.no_augment_from = config.no_augment_from
        self.seed, self.salt = config.seed, config.loader.salt
        self.augment = config.augment
        self.width, self.height = width, height

    def __len__(self) -> int:
        return self.steps * self.batch_size

    def pull(self, name: str, index: int, rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
        """Frame ``index`` of a stream as a fitted tile, degraded first if the stream asks."""
        stream = self.streams[name]
        frame = np.asarray(stream.frames[index])
        if stream.degrade > 0 and rng.random_sample() < stream.degrade:
            frame = np.asarray(degrade(Image.fromarray(frame), rng))
        return fit_tile(frame, stream.targets[index], self.width, self.height)

    def pick_extra(self, rng: np.random.RandomState) -> tuple[str, int]:
        """A stream by the ``mosaic_extra`` shares, and one of its frames."""
        draw, reached = rng.random_sample(), 0.0
        for name, share in self.extra:
            reached += share
            if draw < reached:
                break
        return name, int(rng.randint(0, len(self.streams[name])))

    def __getitem__(self, j: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not 0 <= j < len(self):
            raise IndexError(j)
        seed = tile_seed(self.salt, self.seed, j)
        random.seed(seed)
        np.random.seed(seed)
        rng = np.random.RandomState(seed ^ 0x5EED)
        name = self.primary[j % self.batch_size]
        index = int(rng.randint(0, len(self.streams[name])))
        a = self.augment
        if j // self.batch_size < self.no_augment_from and random.random() < a.mosaic:
            center_y = int(random.uniform(0.5 * self.height, 1.5 * self.height))
            center_x = int(random.uniform(0.5 * self.width, 1.5 * self.width))
            picks = [(name, index)] + [self.pick_extra(rng) for _ in range(MOSAIC_TILES - 1)]
            tiles = [self.pull(stream, frame, rng) for stream, frame in picks]
            image, boxes = mosaic(tiles, center_x, center_y, degrees=a.degrees, translate=a.translate,
                                  scale=a.scale, shear=a.shear, width=self.width, height=self.height)
        else:
            image, boxes = self.pull(name, index, rng)
        canvas, boxes = finish(image, boxes, a.flip, self.width, self.height)
        targets, ignore, counts = pack(boxes, a.max_targets, a.max_ignore)
        return np.ascontiguousarray(canvas.transpose(2, 0, 1), dtype=np.float32), targets, ignore, counts
