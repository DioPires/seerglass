# SPDX-License-Identifier: Apache-2.0
"""The hover rule: stacking, the hit test and what is shown, against cases worked by hand."""

from __future__ import annotations

import pytest

from cardvision.evaluate.hover import (
    ACCEPTED_NMS,
    BEST_SCORE,
    DETECTOR_SCORE,
    HoverLayer,
    Proposal,
    box_iou,
    hit,
    hitbox_quad,
    hover_order,
    kept_after_nms,
    point_in_convex_quad,
    resolve,
)

FRAME = (1000, 500)  # width, height


def flat(quad):
    return [value for point in quad for value in point]


def proposal(box, similarity=0.9, *, detector_score=0.8, look=0, tau_passed=True, accepted=True, name="A"):
    return Proposal(box=tuple(float(v) for v in box), detector_score=detector_score, similarity=similarity, look=look,
                    tau_passed=tau_passed, accepted=accepted, name=name if accepted else None,
                    card_id=f"id-{name}" if accepted else None, top_name=name)


# ---------------------------------------------------------------------------------- the hit test
def test_hitbox_corners_by_hand():
    # x 100..300 and y 50..150 in a 1000 x 500 frame: fractions 0.1, 0.1, 0.2, 0.2; center
    # (0.2, 0.2) -> pixels (200, 100); half sizes 100 and 50.
    quad = hitbox_quad((100, 50, 200, 100), *FRAME)
    assert flat(quad) == pytest.approx([100, 50, 300, 50, 300, 150, 100, 150])


def test_a_point_on_an_edge_or_a_corner_is_inside():
    box = (100, 50, 200, 100)
    for point in [(200, 100), (100, 100), (300, 100), (200, 50), (200, 150), (100, 50), (300, 150)]:
        assert hit(box, *point, *FRAME), point
    for point in [(99.999, 100), (300.001, 100), (200, 49.999), (200, 150.001), (0, 0)]:
        assert not hit(box, *point, *FRAME), point


def test_the_hitbox_is_clamped_to_the_frame_as_the_runtime_clamps_it():
    # A box that starts left of the frame: its left edge clamps to 0 but its width is kept, so
    # the outline is x 0..200, not -50..150. The runtime clamps the four numbers one by one.
    quad = hitbox_quad((-50, 100, 200, 100), *FRAME)
    assert flat(quad) == pytest.approx([0, 100, 200, 100, 200, 200, 0, 200])
    assert hit((-50, 100, 200, 100), 180, 150, *FRAME)
    # A box wider than the frame is as wide as the frame.
    assert flat(hitbox_quad((0, 0, 1500, 500), *FRAME)) == pytest.approx([0, 0, 1000, 0, 1000, 500, 0, 500])


def test_point_in_convex_quad_works_for_either_winding_and_a_turned_outline():
    diamond = [(0, -10), (10, 0), (0, 10), (-10, 0)]
    assert point_in_convex_quad(0, 0, diamond) and point_in_convex_quad(0, 0, diamond[::-1])
    assert point_in_convex_quad(5, 5, diamond)  # on the edge x + y = 10
    assert not point_in_convex_quad(6, 5, diamond)
    assert not point_in_convex_quad(9, 9, diamond)


# -------------------------------------------------------------------------------------- stacking
def test_box_iou_by_hand():
    assert box_iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)
    assert box_iou((0, 0, 10, 10), (10, 0, 10, 10)) == 0.0
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_accepted_nms_keeps_tau_passed_boxes_and_drops_overlaps_by_similarity():
    proposals = [
        proposal((0, 0, 100, 100), 0.80),                      # 0: overlaps 1 by a third; lower similarity
        proposal((50, 0, 100, 100), 0.95),                     # 1: kept first
        proposal((400, 0, 100, 100), 0.99, tau_passed=False, accepted=False),  # 2: under tau, never a member
        proposal((60, 0, 100, 100), 0.90),                     # 3: overlaps 1 heavily, dropped
        proposal((600, 0, 100, 100), 0.70, accepted=False),    # 4: tau passed, margin failed: still a member
    ]
    # 0 against 1: intersection 50 x 100 over 15000 = 1/3 >= 0.3, dropped.
    assert kept_after_nms(proposals) == [1, 4]
    # At a limit just above a third, box 0 survives.
    assert kept_after_nms(proposals, iou=0.34) == [1, 0, 4]


def test_on_equal_similarity_an_upright_look_is_visited_before_a_turned_one():
    turned = proposal((0, 0, 100, 100), 0.9, look=1, name="turned")
    upright = proposal((10, 0, 100, 100), 0.9, look=0, name="upright")
    assert kept_after_nms([turned, upright]) == [1]
    assert kept_after_nms([upright, turned]) == [0]
    both_upright = [proposal((0, 0, 100, 100), 0.9), proposal((10, 0, 100, 100), 0.9)]
    assert kept_after_nms(both_upright) == [0]


def test_the_smallest_accepted_box_under_the_point_wins():
    big = proposal((0, 0, 400, 400), 0.99, name="big")
    small = proposal((100, 100, 50, 50), 0.70, name="small")   # inside big; overlap 2500/160000, both kept
    layer = HoverLayer([big, small], *FRAME)
    assert layer.order == [0, 1]                               # bottom first: the larger box is below
    assert layer.at(120, 120).name == "small"
    assert layer.at(300, 300).name == "big"
    assert layer.at(450, 450).index is None and layer.at(450, 450).name is None
    # The order of the proposals does not matter.
    assert HoverLayer([small, big], *FRAME).at(120, 120).name == "small"


def test_ties_on_area_go_to_the_higher_similarity_then_the_earlier_box():
    # Two boxes of equal area that overlap by 1/7 (kept both), then two identical in every key.
    a = proposal((0, 0, 100, 100), 0.80, name="a")
    b = proposal((75, 0, 100, 100), 0.90, name="b")
    assert box_iou(a.box, b.box) == pytest.approx(2500 / 17500)
    assert hover_order([a, b]) == [0, 1] and resolve([a, b], 90, 50, *FRAME).name == "b"
    assert hover_order([b, a]) == [1, 0] and resolve([b, a], 90, 50, *FRAME).name == "b"
    c = proposal((75, 0, 100, 100), 0.80, name="c")
    assert hover_order([a, c]) == [1, 0] and resolve([a, c], 90, 50, *FRAME).name == "a"


def test_a_kept_box_that_fails_the_margin_wins_the_pointer_and_shows_nothing():
    under = proposal((0, 0, 400, 400), 0.99, name="under")
    doubtful = proposal((100, 100, 50, 50), 0.70, accepted=False, name="doubtful")
    shown = resolve([under, doubtful], 120, 120, *FRAME)
    assert shown.index == 1 and shown.name is None and shown.card_id is None and shown.box == doubtful.box
    # A box under tau is not a member at all, so the card beneath shows through.
    rejected = proposal((100, 100, 50, 50), 0.40, tau_passed=False, accepted=False)
    assert resolve([under, rejected], 120, 120, *FRAME).name == "under"


def test_the_other_two_selections():
    big = proposal((0, 0, 400, 400), 0.99, detector_score=0.5, name="big")
    small = proposal((100, 100, 50, 50), 0.70, detector_score=0.9, name="small")
    weak = proposal((100, 100, 60, 60), 0.30, detector_score=0.95, tau_passed=False, accepted=False, name="weak")
    proposals = [big, small, weak]
    # accepted-nms: weak is not a member; the smaller of big and small is on top.
    assert resolve(proposals, 120, 120, *FRAME, ACCEPTED_NMS).name == "small"
    # best-score: every box takes part and the highest similarity is on top.
    assert hover_order(proposals, BEST_SCORE) == [2, 1, 0]
    assert resolve(proposals, 120, 120, *FRAME, BEST_SCORE).name == "big"
    # detector-score: the highest detector score is on top, and weak is not accepted: nothing.
    assert hover_order(proposals, DETECTOR_SCORE) == [0, 1, 2]
    shown = resolve(proposals, 120, 120, *FRAME, DETECTOR_SCORE)
    assert shown.index == 2 and shown.name is None
    # On equal detector scores the earlier box is on top.
    twins = [proposal((0, 0, 100, 100), detector_score=0.7, name="first"),
             proposal((0, 0, 100, 100), detector_score=0.7, name="second")]
    assert resolve(twins, 50, 50, *FRAME, DETECTOR_SCORE).name == "first"
    with pytest.raises(ValueError):
        hover_order(proposals, "no-such-selection")
