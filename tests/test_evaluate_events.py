# SPDX-License-Identifier: Apache-2.0
"""Hover events and their outcomes, on hand-built rows and hand-built detections."""

from __future__ import annotations

from collections import Counter

import pytest

from cardvision.data.schema import Box, Point, Row
from cardvision.evaluate.events import (
    ADJACENT,
    BACK,
    CENTER_INSIDE_ANOTHER_BOX,
    CENTER_INSIDE_ANOTHER_BOX_DEFINITION,
    FACE,
    FACE_UNNAMED,
    FACE_UNREADABLE,
    NOTHING,
    OFF_CARD,
    RIGHT,
    SHOWN,
    UNREADABLE,
    WRONG,
    box_contains,
    classify,
    inside_another_box,
    is_adjacent,
    named_face_centers,
    rates,
    row_events,
    score_row,
    tally,
)
from cardvision.evaluate.hover import HoverLayer, Proposal
from cardvision.evaluate.stats import exact_lower, exact_upper

# Two printings of one name, and two other cards.
NAMES = {"T-1": "Alpha", "T-1A": "Alpha", "T-2": "Beta", "T-3": "Gamma"}
W, H = 1000, 500


def quad(x, y, w, h):
    """An upright box in pixels as fractions of the 1000 x 500 frame."""
    return ((x / W, y / H), ((x + w) / W, y / H), ((x + w) / W, (y + h) / H), (x / W, (y + h) / H))


def face(x, y, w, h, card_id, *, readable="yes", identity="named", ignore=False, box_id=None):
    return Box(kind="face", quad=quad(x, y, w, h), ignore=ignore, card_id=card_id, identity=identity, readable=readable,
               id=box_id)


def row(boxes=None, points=None, *, group="g-one", source_type="digital", video="AAAAAAAAAAA", t=1.5,
        completeness=None):
    completeness = completeness or ("exhaustive-boxes" if boxes is not None else "points")
    return Row(platform="youtube", video_id=video, t=t, frame_width=W, frame_height=H, source_type=source_type,
               group=group, role="evaluation", trained_on=False, completeness=completeness, credit="tests",
               boxes=None if boxes is None else tuple(boxes), points=None if points is None else tuple(points))


def proposal(x, y, w, h, name, *, similarity=0.9, accepted=True, tau_passed=True):
    return Proposal(box=(float(x), float(y), float(w), float(h)), detector_score=0.9, similarity=similarity, look=0,
                    tau_passed=tau_passed, accepted=accepted, name=name if accepted else None,
                    card_id=None, top_name=name)


# ------------------------------------------------------------------------------------- events
def test_points_and_box_centers_become_events_with_roles():
    labeled = row(
        boxes=[
            face(100, 100, 100, 200, "T-1", box_id="alpha"),                 # readable face
            face(300, 100, 100, 200, "T-2", readable="no"),                  # named, not readable from the frame
            face(500, 100, 100, 200, None, identity="unknown", readable=None),  # nobody could name it
            face(700, 100, 100, 200, "T-3", readable=None),                  # readability never recorded: readable
            Box(kind="back", quad=quad(100, 350, 100, 100)),
            Box(kind="not-card", quad=quad(300, 350, 100, 100)),
            Box(kind="unreadable", quad=quad(500, 350, 100, 100)),           # no event
            face(700, 350, 100, 100, "T-1", ignore=True),                    # an ignore region: no event
            face(950, 100, 200, 200, "T-2"),                                 # center at x = 1050: outside the frame
            face(0, 0, 50, 50, "T-9"),                                       # an id the catalog lacks
        ],
        points=[Point(x=0.9, y=0.9, kind="not-card", id="far"), Point(x=0.15, y=0.4, kind="face", card_id="T-1A",
                                                                     identity="named", readable="maybe"),
                Point(x=0.5, y=0.5, kind="unreadable")],
    )
    skipped = Counter()
    events = row_events(labeled, NAMES, skipped)
    assert [(e.kind, e.label, e.role, e.truth) for e in events] == [
        ("point", "far", OFF_CARD, None),
        ("point", "p1", FACE, "Alpha"),
        ("point", "p2", UNREADABLE, None),
        ("center", "alpha", FACE, "Alpha"),
        ("center", "b1", FACE_UNREADABLE, "Beta"),
        ("center", "b2", FACE_UNNAMED, None),
        ("center", "b3", FACE, "Gamma"),
        ("center", "b4", BACK, None),
        ("center", "b5", OFF_CARD, None),
    ]
    assert skipped == {"unreadable-box": 1, "ignore-region": 1, "center-outside-frame": 1, "card-not-in-catalog": 1}
    center = events[3]
    assert (center.x, center.y) == pytest.approx((0.15, 0.4))
    assert center.key == "youtube-AAAAAAAAAAA-t0000001500/center/alpha"
    assert (center.group, center.source_type, center.truth_id) == ("g-one", "digital", "T-1")


def test_a_points_row_gives_no_center_events_and_a_boxes_row_keeps_its_points():
    only_points = row(points=[Point(x=0.5, y=0.5, kind="back")])
    assert [(e.kind, e.role) for e in row_events(only_points, NAMES)] == [("point", BACK)]
    both = row(boxes=[Box(kind="back", quad=quad(0, 0, 100, 100))], points=[Point(x=0.5, y=0.5, kind="not-card")])
    assert [(e.kind, e.role) for e in row_events(both, NAMES)] == [("point", OFF_CARD), ("center", BACK)]


def test_the_center_of_a_turned_box_is_the_mean_of_its_corners():
    turned = Box(kind="back", quad=((0.5, 0.2), (0.7, 0.5), (0.5, 0.8), (0.3, 0.5)))
    (event,) = row_events(row(boxes=[turned]), NAMES)
    assert (event.x, event.y) == pytest.approx((0.5, 0.5))


def test_a_card_without_art_is_scored_as_a_card_the_recognizer_cannot_know():
    # "art": null: the card is in the catalog, so a label naming it is an ordinary face event,
    # and since it has no bank row the recognizer shows nothing there or another card.
    from cardvision.core.catalog import parse_catalog

    catalog = parse_catalog({"game": "toy", "locale": "en", "cards": [
        {"id": "T-1", "name": "Alpha", "art": "T-1.png"}, {"id": "T-2", "name": "Beta", "art": "T-2.png"},
        {"id": "T-NEW", "name": "Lantern Moth", "art": None}]})
    names, identities = catalog.names(), catalog.identities()
    labeled = row(boxes=[face(100, 100, 100, 200, "T-NEW", box_id="moth"), face(300, 100, 100, 200, "T-NEW", box_id="moth2"),
                         face(500, 100, 100, 200, "T-2", box_id="beta")])
    skipped = Counter()
    events = row_events(labeled, names, skipped)
    assert [(e.label, e.role, e.truth, e.truth_id) for e in events] == [
        ("moth", FACE, "Lantern Moth", "T-NEW"), ("moth2", FACE, "Lantern Moth", "T-NEW"), ("beta", FACE, "Beta", "T-2")]
    assert skipped == {}
    layer = HoverLayer([proposal(100, 100, 100, 200, "Alpha"), proposal(500, 100, 100, 200, "Beta")], W, H)
    scored = score_row(labeled, events, layer, names, identities)
    assert [(s.event.label, s.outcome) for s in scored] == [("moth", WRONG), ("moth2", NOTHING), ("beta", RIGHT)]
    counts = tally(scored)
    assert (counts["faceRight"], counts["faceWrong"], counts["faceNothing"]) == (1, 1, 1)


@pytest.mark.parametrize("rule", ["name", "printing"])
def test_a_card_without_art_is_right_when_a_printing_of_it_with_art_is_shown(rule):
    # The card without art is never named by its own row; its printing with art names its identity.
    from cardvision.core.catalog import parse_catalog

    catalog = parse_catalog({"game": "toy", "locale": "en", "identity": rule, "cards": [
        {"id": "T-NEW", "name": "Lantern Moth", "art": None},
        {"id": "T-NEWA", "name": "Lantern Moth", "printingOf": "T-NEW", "art": "T-NEWA.png"},
        {"id": "T-2", "name": "Beta", "art": "T-2.png"}]})
    names, identities = catalog.names(), catalog.identities()
    labeled = row(boxes=[face(100, 100, 100, 200, "T-NEW", box_id="moth")])
    sibling = Proposal(box=(100.0, 100.0, 100.0, 200.0), detector_score=0.9, similarity=0.9, look=0, tau_passed=True,
                       accepted=True, name="Lantern Moth", card_id="T-NEWA", top_name="Lantern Moth")
    scored = score_row(labeled, row_events(labeled, names), HoverLayer([sibling], W, H), names, identities)
    assert [(s.event.label, s.outcome) for s in scored] == [("moth", RIGHT)]


# ----------------------------------------------------------- a center inside another box
COVERED = CENTER_INSIDE_ANOTHER_BOX


def centers(labeled, skipped=None):
    return [event.label for event in row_events(labeled, NAMES, skipped) if event.kind == "center"]


def test_two_overlapping_boxes_whose_centers_lie_in_each_other_give_no_event():
    # Two cards stacked with a small offset: (150, 200) is inside the second box and
    # (170, 220) inside the first. Neither label says which card is on top.
    stacked = row(boxes=[face(100, 100, 100, 200, "T-1", box_id="under"), face(120, 120, 100, 200, "T-2", box_id="over")])
    skipped = Counter()
    assert centers(stacked, skipped) == [] and skipped == {COVERED: 2}


def test_of_two_overlapping_boxes_only_the_covered_center_is_left_out():
    # A small box inside a large one, away from the large one's center (200, 250).
    nested = row(boxes=[face(100, 100, 200, 300, "T-1", box_id="large"), face(110, 110, 60, 80, "T-2", box_id="small")])
    skipped = Counter()
    assert centers(nested, skipped) == ["large"] and skipped == {COVERED: 1}
    # The order of the labels does not matter.
    skipped = Counter()
    assert centers(row(boxes=nested.boxes[::-1]), skipped) == ["large"] and skipped == {COVERED: 1}


def test_overlapping_boxes_whose_centers_are_clear_both_give_events():
    # The boxes share the corner region x 180..200, y 250..300; the centers (150, 200) and
    # (230, 350) are outside it.
    corner = row(boxes=[face(100, 100, 100, 200, "T-1", box_id="one"), face(180, 250, 100, 200, "T-2", box_id="two")])
    skipped = Counter()
    assert centers(corner, skipped) == ["one", "two"] and skipped == {}
    apart = row(boxes=[face(100, 100, 100, 200, "T-1", box_id="one"), face(400, 100, 100, 200, "T-2", box_id="two")])
    assert centers(apart, skipped) == ["one", "two"] and skipped == {}


@pytest.mark.parametrize("cover", [
    Box(kind="face", quad=quad(100, 100, 300, 300), ignore=True),                      # an ignore region
    Box(kind="back", quad=quad(100, 100, 300, 300), ignore=True),
    Box(kind="not-card", quad=quad(100, 100, 300, 300), id="cover"),
    Box(kind="unreadable", quad=quad(100, 100, 300, 300)),
    Box(kind="back", quad=quad(100, 100, 300, 300), id="cover"),
    Box(kind="face", quad=quad(100, 100, 300, 300), card_id=None, identity="unknown", id="cover"),
    Box(kind="face", quad=quad(100, 100, 300, 300), card_id="T-9", identity="named", readable="yes", id="cover"),
])
def test_a_box_of_any_kind_covers_a_center(cover):
    # The covering box spans 100..400 both ways, center (250, 250); each covered box's center
    # is inside it and its own outline does not reach (250, 250).
    covered = [face(120, 120, 60, 80, "T-1", box_id="a-face"), Box(kind="back", quad=quad(300, 120, 60, 80), id="a-back"),
               Box(kind="not-card", quad=quad(120, 300, 60, 80), id="a-table")]
    skipped = Counter()
    found = centers(row(boxes=[cover, *covered]), skipped)
    assert not {"a-face", "a-back", "a-table"} & set(found) and skipped[COVERED] == 3
    # The covering box itself is counted once, by its own kind: an event, or the reason it gives none.
    own = {"ignore-region": 1} if cover.ignore else {"unreadable-box": 1} if cover.kind == "unreadable" \
        else {"card-not-in-catalog": 1} if cover.card_id == "T-9" else {}
    assert found == ([] if own else ["cover"]) and skipped == {COVERED: 3, **own}


def test_containment_is_the_hover_rule_s_test_of_a_point_against_an_outline():
    # Fractions that binary floating point holds exactly: the first box's center (0.25, 0.5)
    # lies on the left edge of the second, and a point on an edge is inside.
    first = Box(kind="back", quad=((0.125, 0.25), (0.375, 0.25), (0.375, 0.75), (0.125, 0.75)), id="first")
    on_edge = Box(kind="back", quad=((0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)), id="second")
    past_edge = Box(kind="back", quad=((0.2501, 0.25), (0.75, 0.25), (0.75, 0.75), (0.2501, 0.75)), id="second")
    assert centers(row(boxes=[first, on_edge])) == ["second"]
    assert centers(row(boxes=[first, past_edge])) == ["first", "second"]
    # A turned outline is tested as drawn, not by the upright box around it: (0.35, 0.25) is
    # inside the upright box around the diamond and outside the diamond.
    diamond = Box(kind="not-card", quad=((0.5, 0.2), (0.7, 0.5), (0.5, 0.8), (0.3, 0.5)), id="diamond")
    inside = Box(kind="back", quad=((0.52, 0.45), (0.58, 0.45), (0.58, 0.55), (0.52, 0.55)), id="inside")
    in_the_corner = Box(kind="back", quad=((0.33, 0.2), (0.37, 0.2), (0.37, 0.3), (0.33, 0.3)), id="corner")
    labeled = row(boxes=[diamond, inside, in_the_corner])
    skipped = Counter()
    assert centers(labeled, skipped) == ["diamond", "corner"] and skipped == {COVERED: 1}
    assert inside_another_box(labeled, 1, 0.55, 0.5) and not inside_another_box(labeled, 2, 0.35, 0.25)
    # A box never covers its own center, and a row without boxes covers nothing.
    assert not inside_another_box(labeled, 0, 0.5, 0.5) and inside_another_box(labeled, 1, 0.5, 0.5)
    assert not inside_another_box(row(points=[Point(x=0.5, y=0.5, kind="back")]), 0, 0.5, 0.5)


def test_labeled_points_are_events_wherever_they_lie():
    # A point on a stack of two boxes, and a points row: the rule is about box centers only.
    stacked = [face(100, 100, 100, 200, "T-1", box_id="under"), face(120, 120, 100, 200, "T-2", box_id="over")]
    on_the_stack = Point(x=0.15, y=0.4, kind="face", card_id="T-2", identity="named", readable="yes", id="chosen")
    skipped = Counter()
    events = row_events(row(boxes=stacked, points=[on_the_stack, Point(x=0.9, y=0.9, kind="not-card", id="table")]),
                        NAMES, skipped)
    assert [(e.kind, e.label, e.role) for e in events] == [("point", "chosen", FACE), ("point", "table", OFF_CARD)]
    assert skipped == {COVERED: 2}
    # A row labeled with points only, whose boxes (if it lists any) are not exhaustive: no box is read.
    points_row = row(boxes=stacked, points=[on_the_stack], completeness="points")
    skipped = Counter()
    assert [e.label for e in row_events(points_row, NAMES, skipped)] == ["chosen"] and skipped == {}


def test_every_box_is_an_event_or_is_counted_once():
    labeled = row(boxes=[
        face(100, 100, 100, 200, "T-1", box_id="under"), face(120, 120, 100, 200, "T-2", box_id="over"),   # each covers the other
        face(500, 100, 100, 200, "T-3", box_id="clear"),                                                    # an event
        face(950, 100, 200, 200, "T-2"),                               # center (1050, 200): outside the frame
        Box(kind="face", quad=quad(900, 150, 200, 100), ignore=True),  # an ignore region, around that center
        face(510, 110, 30, 40, "T-9"),                                 # not in the catalog, and inside "clear"
        Box(kind="unreadable", quad=quad(700, 350, 100, 100)),
        Box(kind="back", quad=quad(710, 360, 20, 20), id="in-the-blur"),   # inside the unreadable box
    ])
    skipped = Counter()
    events = row_events(labeled, NAMES, skipped)
    assert [event.label for event in events] == ["clear"]
    # Each box is counted under the first reason that applies: the frame's edge before the
    # covering box, the covering box before the catalog.
    assert skipped == {COVERED: 4, "center-outside-frame": 1, "ignore-region": 1, "unreadable-box": 1}
    assert len(events) + sum(skipped.values()) == len(labeled.boxes)


def test_the_rule_is_stated_in_one_sentence():
    text = CENTER_INSIDE_ANOTHER_BOX_DEFINITION
    assert text.startswith("center inside another box = ") and COVERED == "center-inside-another-box" and COVERED in text
    assert "ignore regions and not-card boxes included" in text and ". " not in text


def test_a_covered_center_changes_no_other_event_s_outcome():
    labeled, layer = scene()
    # A second label on Alpha's card, offset a little: both centers are now covered. The name
    # shown beside the card is still adjacent, since the labeled faces still supply their centers.
    more = row(boxes=[*labeled.boxes, face(110, 110, 100, 200, "T-1", box_id="again")], points=labeled.points)
    skipped = Counter()
    before = {s.event.label: s.outcome for s in score_row(labeled, row_events(labeled, NAMES), layer, NAMES)}
    after = {s.event.label: s.outcome for s in score_row(more, row_events(more, NAMES, skipped), layer, NAMES)}
    assert skipped == {COVERED: 2} and "b0" in before and "b0" not in after and "again" not in after
    assert after == {label: outcome for label, outcome in before.items() if label != "b0"}
    assert after["near-alpha"] == ADJACENT


def test_named_faces_that_can_make_a_tooltip_adjacent():
    labeled = row(
        boxes=[face(100, 100, 100, 200, "T-1"), face(300, 100, 100, 200, "T-2", readable="no"),
               face(500, 100, 100, 200, None, identity="unlabeled", readable=None),
               face(700, 100, 100, 200, "T-3", ignore=True), Box(kind="back", quad=quad(0, 0, 50, 50))],
        points=[Point(x=0.9, y=0.9, kind="face", card_id="T-1A", identity="named", readable="yes"),
                Point(x=0.1, y=0.9, kind="not-card")],
    )
    # A face labeled unreadable still supplies its center; an ignore region and an unnamed face do not.
    found = named_face_centers(labeled, NAMES)
    assert [name for _, _, name in found] == ["Alpha", "Alpha", "Beta"]
    assert [value for x, y, _ in found for value in (x, y)] == pytest.approx([0.9, 0.9, 0.15, 0.4, 0.35, 0.4])


# ----------------------------------------------------------------------------------- outcomes
@pytest.mark.parametrize(("role", "shown", "truth", "adjacent", "expected"), [
    (FACE, "Alpha", "Alpha", False, RIGHT),
    (FACE, "Beta", "Alpha", False, WRONG),
    (FACE, None, "Alpha", False, NOTHING),
    (BACK, "Alpha", None, False, WRONG),
    (BACK, "Alpha", None, True, WRONG),          # adjacency never excuses a name on a back
    (BACK, None, None, False, NOTHING),
    (OFF_CARD, "Alpha", None, False, WRONG),
    (OFF_CARD, "Alpha", None, True, ADJACENT),
    (OFF_CARD, None, None, True, NOTHING),
    (FACE_UNREADABLE, "Alpha", "Alpha", False, SHOWN),
    (FACE_UNREADABLE, None, "Alpha", False, NOTHING),
    (FACE_UNNAMED, "Alpha", None, False, SHOWN),
    (UNREADABLE, "Alpha", None, False, SHOWN),
])
def test_classification(role, shown, truth, adjacent, expected):
    assert classify(role, shown, truth, adjacent) == expected


def test_adjacency_by_hand():
    faces = [(150.0, 200.0, "Alpha"), (800.0, 200.0, "Beta")]
    box = (100.0, 100.0, 100.0, 200.0)                       # x 100..200, y 100..300
    assert is_adjacent("Alpha", box, faces)                  # Alpha's center (150, 200) is inside
    assert not is_adjacent("Beta", box, faces)               # Beta's center is elsewhere
    assert not is_adjacent("Gamma", box, faces)              # no face of that name is labeled
    assert is_adjacent("Alpha", (150.0, 200.0, 10.0, 10.0), faces)   # a center on the box's corner counts
    assert not is_adjacent("Alpha", (150.001, 200.0, 10.0, 10.0), faces)
    assert not is_adjacent(None, box, faces) and not is_adjacent("Alpha", None, faces)
    assert box_contains(box, 200.0, 300.0) and not box_contains(box, 200.1, 300.0)


def scene():
    """One frame with every case. Detections: Alpha's card (named Alpha), Beta's card (named
    Gamma, a wrong name), Gamma's card (found, margin failed: shows nothing)."""
    labeled = row(
        boxes=[
            face(100, 100, 100, 200, "T-1A"),                         # shown Alpha: right (another printing)
            face(300, 100, 100, 200, "T-2"),                          # shown Gamma: wrong
            face(500, 100, 100, 200, "T-3"),                          # box found, nothing shown
            face(700, 100, 100, 200, "T-1"),                          # no box at all: nothing
            face(850, 100, 100, 200, "T-2", readable="no"),           # outside every rate
            Box(kind="back", quad=quad(0, 350, 80, 100)),             # no box: nothing
        ],
        points=[
            Point(x=0.11, y=0.21, kind="not-card", id="near-alpha"),  # on Alpha's box; Alpha's center inside: adjacent
            Point(x=0.31, y=0.21, kind="not-card", id="near-beta"),   # on the box showing Gamma; no Gamma center in it: wrong
            Point(x=0.51, y=0.21, kind="not-card", id="near-gamma"),  # box shows nothing
            Point(x=0.95, y=0.95, kind="not-card", id="table"),       # no box
            Point(x=0.19, y=0.59, kind="back", id="sleeve"),          # on Alpha's box: a name on a back
            Point(x=0.5, y=0.9, kind="unreadable", id="blur"),
        ],
    )
    detections = [proposal(100, 100, 100, 200, "Alpha"), proposal(300, 100, 100, 200, "Gamma"),
                  proposal(500, 100, 100, 200, "Gamma", accepted=False)]
    return labeled, HoverLayer(detections, W, H)


def test_a_frame_scored_by_hand():
    labeled, layer = scene()
    scored = score_row(labeled, row_events(labeled, NAMES), layer, NAMES)
    assert {s.event.label: s.outcome for s in scored} == {
        "near-alpha": ADJACENT, "near-beta": WRONG, "near-gamma": NOTHING, "table": NOTHING, "sleeve": WRONG,
        "blur": NOTHING, "b0": RIGHT, "b1": WRONG, "b2": NOTHING, "b3": NOTHING, "b4": NOTHING, "b5": NOTHING,
    }
    by_label = {s.event.label: s for s in scored}
    assert by_label["b1"].shown == "Gamma" and by_label["b1"].box == (300.0, 100.0, 100.0, 200.0)
    assert by_label["b2"].shown is None and by_label["b2"].box == (500.0, 100.0, 100.0, 200.0)
    assert by_label["b3"].box is None
    assert by_label["b0"].to_json()["truth"] == "Alpha" and by_label["b0"].to_json()["outcome"] == "right"
    assert tally(scored) == {
        "faceRight": 1, "faceWrong": 1, "faceNothing": 2,
        "backShown": 1, "backNothing": 1,
        "offCardShownNonAdjacent": 1, "offCardShownAdjacent": 1, "offCardNothing": 2,
        "faceUnreadableShown": 0, "faceUnreadableNothing": 1, "faceUnnamedShown": 0, "faceUnnamedNothing": 0,
        "unreadableShown": 0, "unreadableNothing": 1,
    }


def test_the_two_rates_by_hand():
    labeled, layer = scene()
    scored = score_row(labeled, row_events(labeled, NAMES), layer, NAMES)
    found = rates(scored, seed=0, replicates=100)
    # named right = right / (right + wrong + nothing) over readable faces = 1 / 4.
    assert (found["namedRight"]["numerator"], found["namedRight"]["denominator"]) == (1, 4)
    assert found["namedRight"]["exact95"] == {"side": "lower", "value": exact_lower(1, 4), "note": "assumes independent events"}
    # wrong of shown = (1 face + 1 back + 1 non-adjacent off-card) / (1 right + 3) = 3 / 4; the
    # adjacent tooltip is in neither the numerator nor the denominator.
    assert (found["wrongOfShown"]["numerator"], found["wrongOfShown"]["denominator"]) == (3, 4)
    assert found["wrongOfShown"]["exact95"]["value"] == exact_upper(3, 4)
    assert found["events"] == 12 and found["groups"] == 1
    assert found["wrongOfShown"]["groupBootstrap95"]["undefinedReason"] == "fewer than two recording groups"


def test_a_face_labeled_unreadable_never_enters_a_rate_even_when_named_wrongly():
    labeled = row(boxes=[face(100, 100, 100, 200, "T-1", readable="no"), face(300, 100, 100, 200, "T-2")])
    layer = HoverLayer([proposal(100, 100, 100, 200, "Gamma"), proposal(300, 100, 100, 200, "Beta")], W, H)
    scored = score_row(labeled, row_events(labeled, NAMES), layer, NAMES)
    assert [s.outcome for s in scored] == [SHOWN, RIGHT]
    found = rates(scored, 0, 10)
    assert (found["namedRight"]["numerator"], found["namedRight"]["denominator"]) == (1, 1)
    assert (found["wrongOfShown"]["numerator"], found["wrongOfShown"]["denominator"]) == (0, 1)
    assert found["counts"]["faceUnreadableShown"] == 1


def test_rates_over_groups_resample_the_groups_that_have_events():
    scored = []
    for index, group in enumerate(["g-a", "g-a", "g-b", "g-c"]):
        labeled = row(boxes=[face(100, 100, 100, 200, "T-1")], group=group, video=f"VIDEO00000{index}")
        name = "Alpha" if index != 2 else "Beta"
        scored += score_row(labeled, row_events(labeled, NAMES), HoverLayer([proposal(100, 100, 100, 200, name)], W, H), NAMES)
    found = rates(scored, seed=1, replicates=300)
    assert found["groups"] == 3
    assert (found["namedRight"]["numerator"], found["namedRight"]["denominator"]) == (3, 4)
    interval = found["namedRight"]["groupBootstrap95"]
    assert interval["groups"] == 3 and 0.0 <= interval["low"] <= 0.75 <= interval["high"] <= 1.0
    assert rates(scored, seed=1, replicates=300) == found
    assert rates([], 0, 10)["namedRight"]["value"] is None
