# SPDX-License-Identifier: Apache-2.0
"""The detector's output contract, against cases worked out by hand."""

from __future__ import annotations

import numpy as np
import pytest

from cardvision.core.constants import DETECTOR_BACK_CLASS, DETECTOR_FACE_CLASS
from cardvision.detect import (
    CLASS_NAMES,
    MAX_DETECTIONS,
    NMS_IOU,
    SCORE_THRESHOLD,
    Detection,
    best_score,
    class_name,
    corner_boxes,
    decode,
    iou_one_to_many,
    nms,
    scores_and_classes,
)

F32 = np.float32


def rows(*entries) -> np.ndarray:
    """Raw rows from ``(cx, cy, w, h, objectness, face score, back score)`` tuples."""
    return np.array(entries, dtype=np.float32).reshape(-1, 7)


def corners_to_row(x0, y0, x1, y1, objectness=1.0, face=1.0, back=0.0):
    return ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, objectness, face, back)


def test_operating_point_and_classes():
    assert (SCORE_THRESHOLD, NMS_IOU, MAX_DETECTIONS) == (0.3, 0.45, 80)
    assert CLASS_NAMES == ("face", "back")
    assert class_name(DETECTOR_FACE_CLASS) == "face" and class_name(DETECTOR_BACK_CLASS) == "back"
    assert class_name(5) == "5"


# ----------------------------------------------------------------------------- score and class


def test_score_is_objectness_times_the_best_class_score():
    scores, classes = scores_and_classes(rows(
        (0, 0, 1, 1, 0.5, 0.8, 0.2),      # face wins: 0.5 * 0.8
        (0, 0, 1, 1, 0.5, 0.25, 0.75),    # back wins: 0.5 * 0.75
        (0, 0, 1, 1, 1.0, 0.5, 0.5),      # a tie goes to the first class
    ))
    assert scores.dtype == np.float32
    assert scores.tolist() == [F32(0.5) * F32(0.8), F32(0.375), F32(0.5)]
    assert classes.tolist() == [0, 1, 0]


def test_the_product_is_taken_in_single_precision():
    objectness, face = F32(0.7), F32(0.9)
    scores, _ = scores_and_classes(rows((0, 0, 1, 1, objectness, face, 0)))
    assert scores[0] == objectness * face
    assert float(scores[0]) != float(objectness) * float(face)   # the double-precision product differs


def test_one_class_models_are_decoded_too():
    raw = np.array([[10, 10, 4, 4, 0.5, 0.8]], dtype=np.float32)
    (found,) = decode(raw, 1.0)
    assert found.cls == 0 and found.score == float(F32(0.5) * F32(0.8))


# ----------------------------------------------------------------------------- boxes


def test_corners_from_center_and_size():
    np.testing.assert_array_equal(corner_boxes(rows((100, 50, 20, 10, 1, 1, 0))[:, :4]), [[90, 45, 110, 55]])


def test_boxes_are_divided_by_the_ratio():
    (found,) = decode(rows((100, 50, 20, 10, 1.0, 0.9, 0.1)), ratio=0.5)
    assert found == Detection(x=180.0, y=90.0, width=40.0, height=20.0, score=float(F32(0.9)), cls=0)
    assert found.box == (180.0, 90.0, 40.0, 20.0) and found.class_name == "face"


def test_corners_are_single_precision_and_the_division_is_double():
    cx, cy, w, h = F32(100.3), F32(50.7), F32(20.1), F32(9.9)
    ratio = 1 / 3
    (found,) = decode(rows((cx, cy, w, h, 1.0, 1.0, 0.0)), ratio)
    x0, x1 = cx - w / F32(2), cx + w / F32(2)
    y0, y1 = cy - h / F32(2), cy + h / F32(2)
    assert isinstance(x0, np.float32)
    assert found.x == float(np.float64(x0) / ratio)
    assert found.y == float(np.float64(y0) / ratio)
    assert found.width == float(np.float64(x1 - x0) / ratio)
    assert found.height == float(np.float64(y1 - y0) / ratio)
    # The width is the difference of the rounded corners, not the model's w.
    assert found.width != float(np.float64(w) / ratio)


def test_boxes_are_not_clipped():
    (found,) = decode(rows((5, 5, 30, 30, 1.0, 1.0, 0.0)), 1.0)
    assert (found.x, found.y, found.width, found.height) == (-10.0, -10.0, 30.0, 30.0)


# ----------------------------------------------------------------------------- threshold


def test_the_threshold_is_inclusive_and_compared_in_single_precision():
    at = F32(0.3)                         # 0.300000011920..., the nearest single to 0.3
    below = np.nextafter(at, F32(0))      # 0.299999982118...
    raw = rows((10, 10, 4, 4, 1.0, at, 0), (50, 50, 4, 4, 1.0, below, 0))
    assert [d.score for d in decode(raw, 1.0, threshold=0.3)] == [float(at)]
    # 0.300000012 is above float32(0.3) as a double but narrows to it, so the row is still kept:
    # the threshold is narrowed before the comparison, not the score widened.
    assert float(at) < 0.300000012
    assert [d.score for d in decode(raw, 1.0, threshold=0.300000012)] == [float(at)]


def test_no_row_above_the_threshold_gives_an_empty_list():
    assert decode(rows((10, 10, 4, 4, 0.1, 0.9, 0.9)), 1.0) == []
    assert decode(np.zeros((0, 7), dtype=np.float32), 1.0) == []


def test_the_best_score_is_the_highest_of_any_row_before_the_threshold():
    raw = rows((10, 10, 4, 4, 0.5, 0.4, 0), (50, 50, 4, 4, 1.0, 0, 0.21), (90, 90, 4, 4, 0.1, 0.9, 0.9))
    assert best_score(raw) == float(F32(0.21))
    assert decode(raw, 1.0) == []                  # all under 0.3: decode returns nothing, best_score says how near
    assert best_score(raw[None]) == best_score(raw)
    assert best_score(np.zeros((0, 7), dtype=np.float32)) == 0.0


def test_the_best_score_does_not_change_what_decode_returns():
    raw = rows(corners_to_row(0, 0, 10, 10, face=0.9), corners_to_row(50, 50, 60, 60, face=0.2))
    before = decode(raw, 0.5)
    best_score(raw)
    assert decode(raw, 0.5) == before and len(before) == 1


# ----------------------------------------------------------------------------- overlap and NMS


def test_iou_by_hand():
    a = np.array([0, 0, 10, 10], dtype=np.float32)
    others = np.array([
        [5, 0, 15, 10],     # half overlap: 50 / 150
        [1, 0, 11, 10],     # 90 / 110
        [0, 0, 10, 10],     # itself
        [20, 20, 30, 30],   # apart
        [10, 0, 20, 10],    # touching edges share no area
        [0, 0, 10, 5],      # contained half: 50 / 100
    ], dtype=np.float32)
    overlaps = iou_one_to_many(a, others)
    assert overlaps.dtype == np.float32
    np.testing.assert_allclose(overlaps, [1 / 3, 9 / 11, 1.0, 0.0, 0.0, 0.5], rtol=1e-6)
    assert overlaps[5] == F32(0.5)


def test_two_empty_boxes_overlap_by_zero_not_nan():
    empty = np.array([3, 3, 3, 3], dtype=np.float32)
    assert iou_one_to_many(empty, empty[None]).tolist() == [0.0]


def test_nms_keeps_the_best_and_drops_what_overlaps_it_too_much():
    boxes = np.array([
        [0, 0, 10, 10],      # 0: best
        [1, 0, 11, 10],      # 1: 0.818 with box 0 -> dropped
        [5, 0, 15, 10],      # 2: 0.333 with box 0 -> kept
        [100, 100, 110, 110],  # 3: apart -> kept
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    assert nms(boxes, scores, 0.45).tolist() == [0, 2, 3]


def test_nms_visits_boxes_in_descending_score_order():
    boxes = np.array([[1, 0, 11, 10], [0, 0, 10, 10], [100, 100, 110, 110]], dtype=np.float32)
    scores = np.array([0.5, 0.9, 0.7], dtype=np.float32)
    assert nms(boxes, scores, 0.45).tolist() == [1, 2]


def test_a_dropped_box_does_not_suppress_others():
    # 1 overlaps 0 and is dropped; 2 overlaps 1 heavily but 0 only a little, so it survives.
    boxes = np.array([[0, 0, 10, 10], [4, 0, 14, 10], [8, 0, 18, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    # IoU(0, 1) = 60 / 140 = 0.43; with a limit of 0.4 box 1 is dropped. IoU(0, 2) = 20 / 180.
    assert nms(boxes, scores, 0.4).tolist() == [0, 2]


def test_overlap_equal_to_the_limit_is_kept():
    boxes = np.array([[0, 0, 2, 2], [0, 0, 2, 1]], dtype=np.float32)   # IoU exactly 0.5
    scores = np.array([0.9, 0.8], dtype=np.float32)
    assert nms(boxes, scores, 0.5).tolist() == [0, 1]
    assert nms(boxes, scores, 0.49).tolist() == [0]


def test_equal_scores_keep_row_order():
    boxes = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [50, 50, 60, 60], [51, 50, 61, 60]], dtype=np.float32)
    scores = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    assert nms(boxes, scores, 0.45).tolist() == [0, 2]


def test_nms_of_one_box_and_of_none():
    assert nms(np.array([[0, 0, 1, 1]], dtype=np.float32), np.array([0.5], dtype=np.float32)).tolist() == [0]
    assert nms(np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32)).tolist() == []


# ----------------------------------------------------------------------------- the whole decode


def test_decode_end_to_end_by_hand():
    raw = rows(
        corners_to_row(0, 0, 10, 10, objectness=0.9, face=1.0),             # kept, face 0.9
        corners_to_row(1, 0, 11, 10, objectness=0.8, face=1.0),             # overlaps the first: dropped
        corners_to_row(5, 0, 15, 10, objectness=0.7, face=0.0, back=1.0),   # 0.33 overlap: kept, back 0.7
        corners_to_row(100, 100, 110, 110, objectness=0.2, face=1.0),       # under the threshold
        corners_to_row(200, 100, 240, 160, objectness=1.0, face=0.95, back=0.9),  # kept first, face 0.95
    )
    found = decode(raw, ratio=0.5)
    assert [(d.x, d.y, d.width, d.height, d.cls) for d in found] == [
        (400.0, 200.0, 80.0, 120.0, 0),
        (0.0, 0.0, 20.0, 20.0, 0),
        (10.0, 0.0, 20.0, 20.0, 1),
    ]
    assert [d.score for d in found] == [float(F32(0.95)), float(F32(0.9)), float(F32(0.7))]


def test_suppression_ignores_the_class():
    raw = rows(
        corners_to_row(0, 0, 10, 10, objectness=0.9, face=1.0, back=0.0),
        corners_to_row(0, 0, 10, 10, objectness=0.8, face=0.0, back=1.0),
    )
    found = decode(raw, 1.0)
    assert len(found) == 1 and found[0].cls == 0

    swapped = decode(raw[::-1].copy(), 1.0)
    assert len(swapped) == 1 and swapped[0].cls == 0     # the higher score wins, whatever the row order


def test_at_most_the_best_80_are_returned():
    entries = [corners_to_row(20 * i, 0, 20 * i + 10, 10, objectness=0.31 + 0.005 * i) for i in range(100)]
    found = decode(rows(*entries), 1.0)
    assert len(found) == 80
    assert found[0].x == 20.0 * 99 and found[-1].x == 20.0 * 20
    assert [d.score for d in found] == sorted((d.score for d in found), reverse=True)
    assert len(decode(rows(*entries), 1.0, max_detections=5)) == 5


def test_a_lower_threshold_only_adds_weaker_boxes():
    rng = np.random.default_rng(3)
    n = 300
    raw = np.concatenate([
        rng.uniform(0, 600, (n, 2)), rng.uniform(10, 120, (n, 2)), rng.uniform(0, 1, (n, 3)),
    ], axis=1).astype(np.float32)
    low = decode(raw, 0.5, threshold=0.05, max_detections=10_000)
    for threshold in (0.1, 0.3, 0.6):
        high = decode(raw, 0.5, threshold=threshold, max_detections=10_000)
        assert high == [d for d in low if d.score >= float(F32(threshold))]
        assert high


def test_a_leading_batch_axis_of_one_is_accepted():
    raw = rows(corners_to_row(0, 0, 10, 10))
    assert decode(raw[None], 1.0) == decode(raw, 1.0)


@pytest.mark.parametrize("shape", [(5040,), (2, 5040, 7), (5040, 5), (3, 4)])
def test_rows_of_the_wrong_shape_are_refused(shape):
    with pytest.raises(ValueError, match="detector rows"):
        decode(np.zeros(shape, dtype=np.float32), 1.0)


def test_double_precision_rows_are_narrowed_first():
    raw64 = np.array([corners_to_row(0.1, 0.2, 10.3, 10.4, objectness=0.7, face=0.9)], dtype=np.float64)
    assert decode(raw64, 0.37) == decode(raw64.astype(np.float32), 0.37)
