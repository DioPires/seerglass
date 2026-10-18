# SPDX-License-Identifier: Apache-2.0
"""The detector measures, on hand-built frames: the two overlaps, matching, recall at caps and AP@0.5."""

from __future__ import annotations

import numpy as np
import pytest

from cardvision.data.schema import Box, Row
from cardvision.data.targets import TargetRules
from cardvision.evaluate.detector import (
    FrameRecord,
    average_precision,
    bounds,
    cap_thresholds,
    clip_to_rect,
    corners_iou,
    detector_results,
    frame_truth,
    greedy_match,
    polygon_area,
    polygon_iou,
    polygon_mask,
    score_frame,
    share_inside,
    walk_down,
)

FACE, BACK = 0, 1


def square(x0, y0, x1, y1):
    return [(float(x0), float(y0)), (float(x1), float(y0)), (float(x1), float(y1)), (float(x0), float(y1))]


def record(faces=(), backs=(), ignore=(), not_card=(), detections=(), group="g-a", source_type="digital", size=100):
    return FrameRecord(group=group, source_type=source_type, width=size, height=size, faces=list(faces), backs=list(backs),
                       ignore=list(ignore), not_card=list(not_card),
                       detections=[(tuple(float(v) for v in box), score, cls) for box, score, cls in detections])


# ---------------------------------------------------------------------------------- geometry
def test_polygon_area_and_clipping_by_hand():
    assert polygon_area(square(0, 0, 10, 20)) == 200.0
    assert polygon_area([(0, 0), (10, 0)]) == 0.0
    diamond = [(5, 0), (10, 5), (5, 10), (0, 5)]
    assert polygon_area(diamond) == 50.0
    # The right half of the diamond is a triangle of area 25.
    assert polygon_area(clip_to_rect(diamond, 5, 0, 10, 10)) == pytest.approx(25.0)
    assert clip_to_rect(square(0, 0, 10, 10), 20, 20, 30, 30) == []
    assert bounds(diamond) == (0.0, 0.0, 10.0, 10.0)


def test_the_two_overlap_measures_agree_on_an_upright_card_and_differ_on_one_at_an_angle():
    # Recall at caps uses corners_iou on the upright box around the outline; AP uses polygon_iou.
    assert polygon_iou(square(0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert corners_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    # A card turned 45 degrees fills half of the upright box around it: a detection of that box
    # overlaps the card's outline by 0.5, though it matches the card's bounding box exactly.
    diamond = [(5, 0), (10, 5), (5, 10), (0, 5)]
    assert polygon_iou(diamond, (0, 0, 10, 10)) == pytest.approx(0.5)
    assert corners_iou(bounds(diamond), (0, 0, 10, 10)) == 1.0
    assert polygon_iou(square(0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_mask_and_area_share():
    mask = polygon_mask(100, 100, [square(60, 60, 100, 100)])
    assert mask.shape == (100, 100) and mask[80, 80] and not mask[10, 10]
    assert share_inside(mask, (70, 70, 90, 90)) == 1.0
    assert share_inside(mask, (0, 0, 20, 20)) == 0.0
    # Half of the box x 50..70 lies right of x = 60 (the mask's edge pixel column included).
    assert share_inside(mask, (50, 70, 70, 90)) == pytest.approx(0.5, abs=0.06)


def test_greedy_matching_by_hand():
    boxes = [(0, 0, 10, 10), (2, 0, 12, 10)]
    targets = [(0, 0, 10, 10), (4, 0, 14, 10)]
    # Overlaps: d0-t0 1.0; d1-t0 80/120; d1-t1 80/120; d0-t1 60/140. The best pair goes first;
    # d1 then finds t0 taken and takes t1. Each target is returned with its pair's overlap.
    assert greedy_match(boxes, targets, 0.5) == ({0: 0, 1: 1}, {0: 1.0, 1: pytest.approx(2 / 3)})
    # At a limit above 2/3 only the exact pair matches.
    assert greedy_match(boxes, targets, 0.7) == ({0: 0}, {0: 1.0})
    # Two detections on one target: the better one has it, the other is unmatched.
    assert greedy_match([(0, 0, 10, 8), (0, 0, 10, 10)], [(0, 0, 10, 10)], 0.5) == ({1: 0}, {0: 1.0})
    assert greedy_match([], targets, 0.5) == ({}, {})
    assert greedy_match(boxes, [], 0.5) == ({}, {})


def test_greedy_matching_uses_upright_boxes_and_matches_from_an_overlap_of_0_3():
    # 40 % of the target: matched at the default limit, and returned with its overlap so that
    # the caller can tell a target found at 0.5 from one found at 0.3 only.
    assert greedy_match([(0, 0, 10, 4)], [(0, 0, 10, 10)]) == ({0: 0}, {0: pytest.approx(0.4)})
    assert greedy_match([(0, 0, 10, 2.9)], [(0, 0, 10, 10)]) == ({}, {})
    # A pair is never held back for a better total. Overlaps: d0-t0 90/110, d1-t0 0.6, d0-t1 1/3,
    # d1-t1 none. d0 takes t0; d1, which only overlaps t0, stays unmatched, and so does t1,
    # though d1 on t0 and d0 on t1 would have matched everything.
    boxes = [(0, 1, 10, 11), (0, 0, 10, 6)]
    targets = [(0, 0, 10, 10), (0, 6, 10, 16)]
    assert greedy_match(boxes, targets) == ({0: 0}, {0: pytest.approx(9 / 11)})


def test_equal_overlaps_go_to_the_later_detection_then_the_later_target():
    # Two detections of the same box on one target: the later one has it.
    assert greedy_match([(0, 0, 10, 10), (0, 0, 10, 10)], [(0, 0, 10, 10)]) == ({1: 0}, {0: 1.0})
    # One detection lying the same way across two targets: it takes the later one.
    assert greedy_match([(0, 0, 10, 10)], [(0, 0, 10, 5), (0, 5, 10, 10)]) == ({0: 1}, {1: 0.5})
    # Both at once: the later detection takes the later target, the earlier one the earlier.
    assert greedy_match([(0, 0, 10, 10), (0, 0, 10, 10)], [(0, 0, 10, 5), (0, 5, 10, 10)]) == ({1: 1, 0: 0}, {1: 0.5, 0: 0.5})


def test_greedy_matching_equals_taking_the_best_remaining_pair_again_and_again():
    """The rule restated the slow way: among the pairs still free on both sides, take the one
    with the highest overlap (ties: the later detection, then the later target) until none
    reaches the limit. Boxes on a coarse grid, so that equal overlaps are common."""
    rng = np.random.default_rng(31)

    def box():
        x, y = (int(v) * 10 for v in rng.integers(0, 8, size=2))
        width, height = (int(v) * 10 for v in rng.integers(1, 4, size=2))
        return (x, y, x + width, y + height)

    ties = 0
    for _ in range(300):
        boxes = [box() for _ in range(int(rng.integers(0, 8)))]
        targets = [box() for _ in range(int(rng.integers(0, 8)))]
        overlaps = {(d, t): corners_iou(a, b) for d, a in enumerate(boxes) for t, b in enumerate(targets)}
        ties += len(set(overlaps.values())) < len(overlaps)
        expected_pairs, expected_overlaps = {}, {}
        while True:
            free = [(overlap, d, t) for (d, t), overlap in overlaps.items()
                    if overlap >= 0.3 and d not in expected_pairs and t not in expected_overlaps]
            if not free:
                break
            overlap, d, t = max(free)
            expected_pairs[d], expected_overlaps[t] = t, overlap
        assert greedy_match(boxes, targets) == (expected_pairs, expected_overlaps)
        # One to one, and never under the limit.
        assert len(set(expected_pairs.values())) == len(expected_pairs) == len(expected_overlaps)
        assert all(overlap >= 0.3 for overlap in expected_overlaps.values())
    assert ties > 100


# -------------------------------------------------------------------------- recall at caps
def busy_frame():
    """One 100 x 100 frame: two faces, one back, one ignore region, and seven detections."""
    return record(
        faces=[square(10, 10, 30, 40), square(50, 10, 70, 40)],
        backs=[square(10, 60, 30, 90)],
        ignore=[square(60, 60, 100, 100)],
        detections=[
            ((10, 10, 30, 40), 0.90, FACE),   # the first face exactly
            ((50, 10, 70, 22), 0.80, FACE),   # 40 % of the second face: matched, found at 0.3 and not at 0.5
            ((10, 60, 30, 90), 0.70, FACE),   # a face detection on the back
            ((10, 60, 30, 90), 0.65, BACK),   # the back, found as a back
            ((70, 70, 90, 90), 0.60, FACE),   # inside the ignore region
            ((40, 60, 55, 80), 0.50, FACE),   # on the table
            ((80, 5, 95, 20), 0.20, FACE),    # on the table, weak
        ],
    )


def test_one_frame_scored_by_hand():
    frame = busy_frame()
    # At 0.3: the exact box and the part-box are matched; the box on the back and the box on the
    # table are false positives; the box in the ignore region is neutral.
    assert score_frame(frame, 0.3) == {
        "frames": 1, "faces": 2, "hits": 1, "hitsLoose": 2,
        "falsePositives": 2, "falsePositivesOnBacks": 1, "neutral": 1, "backs": 1, "backHits": 1,
    }
    # Above the back detection's score: no back found. The part-box is matched at 0.3, so it is
    # not a false positive, and its face is found at 0.3 only.
    assert score_frame(frame, 0.75) == {
        "frames": 1, "faces": 2, "hits": 1, "hitsLoose": 2,
        "falsePositives": 0, "falsePositivesOnBacks": 0, "neutral": 0, "backs": 1, "backHits": 0,
    }
    assert score_frame(frame, 0.95)["falsePositives"] == 0 and score_frame(frame, 0.95)["hits"] == 0
    assert score_frame(frame, 0.05)["falsePositives"] == 3


def test_recall_is_measured_on_the_upright_box_around_a_card_at_an_angle():
    # A card twice as long as wide, lying at 45 degrees, fills four ninths of the upright box
    # around it. A detection of that upright box is what the detector is trained to give: the
    # card is found. AP measures the same detection against the outline, and it falls short.
    card = [(0, 10), (10, 0), (30, 20), (20, 30)]
    assert bounds(card) == (0.0, 0.0, 30.0, 30.0) and polygon_iou(card, (0, 0, 30, 30)) == pytest.approx(4 / 9)
    frame = record(faces=[card], detections=[((0, 0, 30, 30), 0.9, FACE)])
    found = score_frame(frame, 0.3)
    assert (found["hits"], found["hitsLoose"], found["falsePositives"]) == (1, 1, 0)
    assert average_precision([frame], FACE)["truePositives"] == 0


def test_faces_and_backs_are_matched_apart():
    # A back detection exactly on a face target, and a face detection exactly on a back target:
    # neither target is found, and the face detection is a false positive on a back.
    frame = record(faces=[square(0, 0, 20, 30)], backs=[square(50, 0, 70, 30)],
                   detections=[((0, 0, 20, 30), 0.9, BACK), ((50, 0, 70, 30), 0.8, FACE)])
    assert score_frame(frame, 0.3) == {
        "frames": 1, "faces": 1, "hits": 0, "hitsLoose": 0,
        "falsePositives": 1, "falsePositivesOnBacks": 1, "neutral": 0, "backs": 1, "backHits": 0,
    }


def test_a_face_detection_is_on_a_back_only_when_it_overlaps_a_back_s_box_by_0_3():
    # A spread pile of backs labeled as one wide box. A detection covering a small part of it
    # overlaps the pile's box by less than 0.3: a plain false positive. One covering most of the
    # pile is a false positive on a back.
    frame = record(backs=[square(0, 0, 100, 40)], detections=[((10, 5, 30, 35), 0.9, FACE)])
    assert corners_iou((10, 5, 30, 35), (0, 0, 100, 40)) < 0.3
    found = score_frame(frame, 0.3)
    assert (found["falsePositives"], found["falsePositivesOnBacks"], found["neutral"]) == (1, 0, 0)
    frame = record(backs=[square(0, 0, 100, 40)], detections=[((10, 0, 90, 40), 0.9, FACE)])
    found = score_frame(frame, 0.3)
    assert (found["falsePositives"], found["falsePositivesOnBacks"], found["neutral"]) == (1, 1, 0)


def test_walk_down_keeps_the_lowest_threshold_whose_whole_upper_range_meets_the_limit():
    counts = {0.05: 4, 0.2: 4, 0.5: 3, 0.6: 2, 0.7: 2, 0.8: 1, 0.9: 0}
    candidates = sorted(counts)
    assert walk_down(candidates, counts.get, 0) == 0.9
    assert walk_down(candidates, counts.get, 1) == 0.8
    assert walk_down(candidates, counts.get, 2) == 0.6
    assert walk_down(candidates, counts.get, 3.5) == 0.5
    assert walk_down(candidates, counts.get, 4) == 0.05
    assert walk_down(candidates, lambda t: 9, 1) == float("inf")
    # A lower threshold that happens to meet the limit again does not count once one above it failed.
    dipping = {0.1: 0, 0.5: 2, 0.9: 1}
    assert walk_down(sorted(dipping), dipping.get, 1) == 0.9


def test_cap_thresholds_on_the_hand_frame():
    frame = busy_frame()
    # False positives at each face detection's score, going down: 0.9 none, 0.8 none (the
    # part-box is matched), 0.7 one (on the back), 0.6 one (the next is neutral), 0.5 two, 0.2 three.
    found = cap_thresholds([frame], [0, 1, 2, 3.5, 4])
    assert found == {0: 0.8, 1: 0.6, 2: 0.5, 3.5: 0.2, 4: 0.2}
    # Two frames, the second empty and quiet: the same counts over twice the frames.
    assert cap_thresholds([frame, record()], [0.5, 1.0]) == {0.5: 0.6, 1.0: 0.5}
    # A frame whose best detection is already a false positive: no threshold meets a cap of 0.
    assert cap_thresholds([record(detections=[((0, 0, 10, 10), 0.9, FACE)])], [0.0]) == {0.0: None}


def test_there_is_no_minimum_score_and_no_threshold_without_a_face_detection():
    # The lowest threshold a cap can give is the lowest score of a face detection, however low.
    frame = record(faces=[square(0, 0, 20, 30)], detections=[((0, 0, 20, 30), 0.9, FACE), ((40, 40, 60, 70), 0.0007, FACE)])
    assert cap_thresholds([frame], [0.5, 1.0]) == {0.5: 0.9, 1.0: 0.0007}
    # Without any face detection there is no candidate threshold at all.
    assert cap_thresholds([], [1.0]) == {1.0: None}
    assert cap_thresholds([record(faces=[square(0, 0, 20, 30)], detections=[((0, 0, 20, 30), 0.9, BACK)])], [1.0]) == {1.0: None}


def test_cap_thresholds_equal_the_plain_walk_over_every_candidate():
    rng = np.random.default_rng(5)
    frames = []
    for _ in range(12):
        faces = [square(x, y, x + 20, y + 30) for x, y in rng.integers(0, 70, size=(3, 2))]
        detections = []
        for _ in range(int(rng.integers(0, 7))):
            x, y = rng.integers(0, 70, size=2)
            if rng.random() < 0.5:
                x, y = (int(v) for v in faces[int(rng.integers(0, 3))][0])
                x, y = x + int(rng.integers(-4, 5)), y + int(rng.integers(-4, 5))
            detections.append(((x, y, x + 20, y + 30), round(float(rng.uniform(0.001, 1.0)), 4), int(rng.random() < 0.2)))
        detections.sort(key=lambda item: -item[1])
        frames.append(record(faces=faces, ignore=[square(0, 80, 40, 100)], detections=detections))
    candidates = sorted({score for frame in frames for _, score, cls in frame.detections if cls == FACE})

    def per_frame(threshold):
        return sum(score_frame(frame, threshold)["falsePositives"] for frame in frames) / len(frames)

    caps = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 2.2]
    found = cap_thresholds(frames, caps)
    for cap in caps:
        expected = walk_down(candidates, per_frame, cap)
        assert found[cap] == (None if expected == float("inf") else expected), cap
    assert len({value for value in found.values()}) > 2  # the caps really pick different thresholds


# --------------------------------------------------------------------------------------- AP
def ap_frames(ignore=True, not_card=False):
    first = record(
        faces=[square(0, 0, 20, 20), square(40, 0, 60, 20)],
        ignore=[square(0, 50, 40, 90)] if ignore else [],
        not_card=[square(70, 70, 90, 90)] if not_card else [],
        detections=[
            ((0, 0, 20, 20), 0.9, FACE),      # the first target exactly: true positive
            ((5, 55, 25, 75), 0.8, FACE),     # inside the ignore region: left out
            ((70, 70, 90, 90), 0.7, FACE),    # on the table: false positive
            ((40, 0, 60, 18), 0.6, FACE),     # 90 % of the second target: true positive
        ],
    )
    second = record(
        faces=[square(10, 10, 30, 30), square(60, 60, 80, 80)],   # the second is never detected
        backs=[square(40, 40, 55, 60)],
        detections=[
            ((10, 10, 30, 26), 0.5, FACE),    # 80 % of the first target: true positive
            ((40, 40, 55, 60), 0.45, BACK),
            ((10, 10, 30, 30), 0.4, FACE),    # the same target again: it is taken, so a false positive
        ],
        group="g-b",
    )
    return [first, second]


def test_average_precision_by_hand_with_an_ignore_region():
    # Ranked face detections: TP, (left out), FP, TP, TP, FP over 4 targets.
    # Precision after each counted one: 1, 1/2, 2/3, 3/4, 3/5; recall: 1/4, 1/4, 2/4, 3/4, 3/4.
    # Envelope at the three recall steps: 1, 3/4, 3/4. AP = 1/4 * (1 + 3/4 + 3/4) = 0.625.
    found = average_precision(ap_frames(), FACE)
    assert found == {"ap": pytest.approx(0.625), "targets": 4, "detections": 6, "truePositives": 3,
                     "falsePositives": 2, "leftOut": 1}


def test_without_the_ignore_region_the_same_detection_is_a_false_positive():
    # TP, FP, FP, TP, TP, FP: precision 1, 1/2, 1/3, 2/4, 3/5, 3/6; envelope at the recall steps
    # 1, 3/5, 3/5. AP = 1/4 * (1 + 0.6 + 0.6) = 0.55.
    found = average_precision(ap_frames(ignore=False), FACE)
    assert found["ap"] == pytest.approx(0.55) and found["leftOut"] == 0 and found["falsePositives"] == 3


def test_a_not_card_box_is_neither_a_target_nor_a_false_positive():
    # The detection on the table now sits on a box a person marked as not a card: left out too.
    # TP, TP, TP, FP: AP = 3/4.
    found = average_precision(ap_frames(not_card=True), FACE)
    assert found["ap"] == pytest.approx(0.75) and found["leftOut"] == 2 and found["targets"] == 4


def test_average_precision_edges():
    frames = ap_frames()
    assert average_precision(frames, BACK) == {"ap": pytest.approx(1.0), "targets": 1, "detections": 1,
                                                "truePositives": 1, "falsePositives": 0, "leftOut": 0}
    assert average_precision([record(detections=[((0, 0, 10, 10), 0.9, FACE)])], FACE)["ap"] is None
    assert average_precision([record(faces=[square(0, 0, 10, 10)])], FACE)["ap"] == 0.0
    # A stricter overlap turns the 80 % and 90 % detections into false positives, and the last
    # detection now finds its target free: TP, FP, FP, FP, TP. Precision at the second recall
    # step is 2/5, so AP = 1/4 * (1 + 2/5) = 0.35.
    strict = average_precision(frames, FACE, iou=0.95)
    assert strict["truePositives"] == 2 and strict["ap"] == pytest.approx(0.35)
    # Ties in score keep frame order: the result does not depend on anything unstated.
    assert average_precision(frames, FACE) == average_precision(ap_frames(), FACE)


# ----------------------------------------------------------------------------- rows to targets
def manifest_row(boxes, source_type="digital"):
    return Row(platform="youtube", video_id="AAAAAAAAAAA", t=2.0, frame_width=1000, frame_height=1000,
               source_type=source_type, group="g-a", role="evaluation", trained_on=False,
               completeness="exhaustive-boxes", credit="tests", boxes=tuple(boxes))


def fraction_quad(x0, y0, x1, y1):
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def test_boxes_become_targets_ignore_regions_and_not_card_boxes():
    def face(quad, **extra):
        return Box(kind="face", quad=quad, card_id="T-1", identity="named", readable="yes", **extra)

    row = manifest_row([
        face(fraction_quad(0.10, 0.10, 0.31, 0.40)),                # a whole card (21 : 30)
        face(fraction_quad(0.50, 0.10, 0.80, 0.20)),                # far too wide: an ignore region
        face(fraction_quad(0.90, 0.10, 1.11, 0.40)),                # runs past the right edge: clipped
        face(fraction_quad(0.10, 0.50, 0.31, 0.80), ignore=True),
        Box(kind="back", quad=fraction_quad(0.50, 0.50, 0.71, 0.80)),
        Box(kind="unreadable", quad=fraction_quad(0.10, 0.85, 0.20, 0.95)),
        Box(kind="not-card", quad=fraction_quad(0.80, 0.85, 0.90, 0.95)),
    ])
    found = frame_truth(row, 500, 500)
    assert (len(found.faces), len(found.backs), len(found.ignore), len(found.not_card)) == (2, 1, 3, 1)
    assert (found.width, found.height, found.group, found.source_type) == (500, 500, "g-a", "digital")
    assert bounds(found.faces[0]) == pytest.approx((50, 50, 155, 200))
    assert bounds(found.faces[1]) == pytest.approx((450, 50, 500, 200))     # clipped to the frame
    assert found.key == "youtube-AAAAAAAAAAA-t0000002000"
    # With the aspect filter off, the wide face is a target.
    assert len(frame_truth(row, 500, 500, TargetRules(aspect_range=None)).faces) == 3


def test_results_report_each_point_and_both_classes():
    frames = [busy_frame(), record(group="g-b", source_type="physical/webcam",
                                   faces=[square(0, 0, 20, 30)], detections=[((0, 0, 20, 30), 0.9, FACE)])]
    found = detector_results(frames, 0.3, caps=(0.5, 1.0), seed=1, replicates=100)
    assert "minScore" not in found
    assert found["overlap"] == {"recall": "upright boxes", "ap50": "outline polygon"}
    assert found["thresholds"] == {"operatingPoint": 0.3, "falsePositivesPerFrame<=0.5": 0.6,
                                   "falsePositivesPerFrame<=1": 0.5}
    overall = found["overall"]
    assert (overall["frames"], overall["faceTargets"], overall["backTargets"], overall["ignoreRegions"]) == (2, 3, 1, 1)
    at_op = overall["points"]["operatingPoint"]
    assert (at_op["recall"]["numerator"], at_op["recall"]["denominator"]) == (2, 3)
    assert (at_op["recallLoose"]["numerator"], at_op["falsePositivesPerFrame"]["numerator"]) == (3, 2)
    assert at_op["falsePositivesPerFrame"]["value"] == 1.0 and "exact95" not in at_op["falsePositivesPerFrame"]
    assert at_op["recall"]["exact95"]["side"] == "lower" and at_op["recall"]["groupBootstrap95"]["groups"] == 2
    assert (at_op["falsePositivesOnBacks"], at_op["neutralDetections"]) == (1, 1)
    assert at_op["backRecall"]["numerator"] == 1
    capped = overall["points"]["falsePositivesPerFrame<=0.5"]
    assert capped["threshold"] == 0.6 and capped["falsePositivesPerFrame"]["value"] == 0.5
    # Each slice is read at the thresholds chosen on all frames.
    webcam = found["bySourceType"]["physical/webcam"]["points"]["falsePositivesPerFrame<=0.5"]
    assert webcam["threshold"] == 0.6 and webcam["recall"]["value"] == 1.0
    assert set(found["bySourceFamily"]) == {"digital", "physical"}
    assert overall["ap50"]["back"]["ap"] == pytest.approx(1.0)
    # AP is a measure of its own. Faces ranked: TP 0.9, TP 0.9, FP 0.8 (the part-box overlaps its
    # face by 0.4, under AP's 0.5), FP 0.7, (0.6 left out), FP 0.5, FP 0.2; two of three targets
    # found at full precision: AP = 2/3.
    assert overall["ap50"]["face"]["ap"] == pytest.approx(2 / 3)
    assert detector_results(frames, 0.3, caps=(0.0,), seed=1, replicates=10)["overall"]["points"][
        "falsePositivesPerFrame<=0"]["threshold"] == 0.8


def test_the_default_caps():
    from cardvision.evaluate.detector import DEFAULT_CAPS

    assert DEFAULT_CAPS == (0.5, 2.0, 2.2)
    found = detector_results([busy_frame()], 0.3, seed=1, replicates=10)
    assert list(found["thresholds"]) == ["operatingPoint", "falsePositivesPerFrame<=0.5", "falsePositivesPerFrame<=2",
                                         "falsePositivesPerFrame<=2.2"]
    # One frame: at most 0.5 false positives means none (0.8); at most 2 or 2.2 means two (0.5).
    assert list(found["thresholds"].values()) == [0.3, 0.8, 0.5, 0.5]
