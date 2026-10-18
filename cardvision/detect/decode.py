# SPDX-License-Identifier: Apache-2.0
"""The detector's output contract: from the model's raw rows to a short list of boxes.

The model emits one row per anchor, already decoded to canvas pixels (the grid offsets and
strides are constants inside the graph) and with the sigmoid applied::

    cx, cy, w, h, objectness, class score 0, class score 1, ...

For the two-class card detector the class scores are ``face`` and ``back``, 5040 rows for a
640 x 384 input. Turning the rows into boxes is done here, outside the graph:

1. score of a row = objectness times its best class score; class = that best class (the first
   one on a tie);
2. keep the rows whose score is at least the threshold;
3. convert center and size to corners;
4. greedy non-maximum suppression in descending score order (equal scores keep row order),
   ignoring the class: a box is dropped when its overlap with an already kept box exceeds the
   limit;
5. keep at most the first ``max_detections`` survivors;
6. divide by the letterbox ratio to return to frame pixels.

Steps 1 to 4 are single-precision arithmetic and step 6 is double precision, written out
explicitly so that the result does not depend on the NumPy version's scalar promotion rules.
Another runtime that follows the same steps in the same precision produces the same boxes bit
for bit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cardvision.core.constants import DETECTOR_BACK_CLASS, DETECTOR_FACE_CLASS

# Operating point of the published detector.
SCORE_THRESHOLD = 0.3
NMS_IOU = 0.45
MAX_DETECTIONS = 80

CLASS_NAMES: tuple[str, ...] = ("face", "back")
assert CLASS_NAMES[DETECTOR_FACE_CLASS] == "face" and CLASS_NAMES[DETECTOR_BACK_CLASS] == "back"

_BOX_COLUMNS = 5  # cx, cy, w, h, objectness; the class scores follow
_F32 = np.float32


@dataclass(frozen=True)
class Detection:
    """One detected card: an axis-aligned box in pixels, its score and its class.

    ``x`` and ``y`` are the top-left corner. ``cls`` indexes ``CLASS_NAMES``. Coordinates are
    not clipped: a box may reach past the edge of the frame.
    """

    x: float
    y: float
    width: float
    height: float
    score: float
    cls: int

    @property
    def box(self) -> tuple[float, float, float, float]:
        """``(x, y, width, height)``."""
        return (self.x, self.y, self.width, self.height)

    @property
    def class_name(self) -> str:
        """``"face"`` or ``"back"`` (see ``class_name``)."""
        return class_name(self.cls)


def class_name(cls: int) -> str:
    """The name of a class index: ``"face"`` or ``"back"``, and the index itself as text for a
    model with further classes."""
    return CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else str(cls)


def _rows(raw: np.ndarray) -> np.ndarray:
    rows = np.asarray(raw)
    if rows.ndim == 3 and rows.shape[0] == 1:
        rows = rows[0]
    if rows.ndim != 2 or rows.shape[1] <= _BOX_COLUMNS:
        raise ValueError(
            f"expected detector rows of shape (anchors, 5 + classes) with at least one class, got {np.asarray(raw).shape}"
        )
    return rows.astype(_F32, copy=False)


def scores_and_classes(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row: the score (objectness times the best class score, float32) and the class."""
    rows = _rows(raw)
    per_class = rows[:, 4:5] * rows[:, _BOX_COLUMNS:]
    classes = np.argmax(per_class, axis=1)
    return per_class[np.arange(len(rows)), classes], classes


def best_score(raw: np.ndarray) -> float:
    """The highest score of any row, before the threshold: 0.0 when there are no rows.

    It says how close a frame came to a detection when ``decode`` returned nothing.
    """
    rows = _rows(raw)
    if not len(rows):
        return 0.0
    scores, _ = scores_and_classes(rows)
    return float(scores.max())


def corner_boxes(rows: np.ndarray) -> np.ndarray:
    """``(N, 4)`` float32 corners ``x0, y0, x1, y1`` from rows that start ``cx, cy, w, h``."""
    rows = np.asarray(rows, dtype=_F32)
    half_w = rows[:, 2] / _F32(2)
    half_h = rows[:, 3] / _F32(2)
    return np.stack([rows[:, 0] - half_w, rows[:, 1] - half_h, rows[:, 0] + half_w, rows[:, 1] + half_h], axis=1)


def iou_one_to_many(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    """Intersection over union of one corner box with each of ``others``, in float32.

    The denominator carries a small constant (1e-9), so two empty boxes give 0, not NaN.
    """
    box = np.asarray(box, dtype=_F32)
    others = np.asarray(others, dtype=_F32)
    x0 = np.maximum(box[0], others[:, 0])
    y0 = np.maximum(box[1], others[:, 1])
    x1 = np.minimum(box[2], others[:, 2])
    y1 = np.minimum(box[3], others[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return inter / (area + areas - inter + _F32(1e-9))


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float = NMS_IOU) -> np.ndarray:
    """Greedy, class-agnostic non-maximum suppression.

    ``boxes`` are ``(N, 4)`` corners and ``scores`` their ``N`` scores. Boxes are visited in
    descending score order, equal scores in their original order. The best remaining box is
    kept and every box whose overlap with it is strictly greater than ``iou`` is dropped.
    Returns the kept indices, best first.
    """
    boxes = np.asarray(boxes, dtype=_F32)
    scores = np.asarray(scores, dtype=_F32)
    limit = _F32(iou)
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while len(order):
        best = order[0]
        keep.append(int(best))
        if len(order) == 1:
            break
        overlaps = iou_one_to_many(boxes[best], boxes[order[1:]])
        order = order[1:][overlaps <= limit]
    return np.array(keep, dtype=int)


def decode(
    raw: np.ndarray,
    ratio: float,
    threshold: float = SCORE_THRESHOLD,
    nms_iou: float = NMS_IOU,
    max_detections: int = MAX_DETECTIONS,
) -> list[Detection]:
    """Turn one frame's raw detector rows into detections in frame pixels, best first.

    ``raw`` is ``(anchors, 5 + classes)`` (a leading batch axis of 1 is accepted) and ``ratio``
    the letterbox ratio the frame was scaled by. The list is the same as the list for any lower
    threshold with the weaker boxes removed, so one run at a low threshold can serve a whole
    curve. No box is filtered for size here.
    """
    rows = _rows(raw)
    scores, classes = scores_and_classes(rows)
    passing = scores >= _F32(threshold)
    rows, scores, classes = rows[passing], scores[passing], classes[passing]
    if not len(rows):
        return []
    boxes = corner_boxes(rows)
    ratio = float(ratio)
    detections = []
    for i in nms(boxes, scores, nms_iou)[:max_detections]:
        x0, y0, x1, y1 = boxes[i]
        detections.append(Detection(
            x=float(np.float64(x0) / ratio),
            y=float(np.float64(y0) / ratio),
            width=float(np.float64(x1 - x0) / ratio),
            height=float(np.float64(y1 - y0) / ratio),
            score=float(scores[i]),
            cls=int(classes[i]),
        ))
    return detections
