# SPDX-License-Identifier: Apache-2.0
"""Fractions and pixels, and the conversions from rows to detector targets and recognizer
crops, against cases worked out by hand."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest
from PIL import Image

from cardvision.data import geometry, targets
from cardvision.data.extract import FrameStore, frame_file_name
from cardvision.data.schema import Box, Point, Row
from cardvision.data.sources import SourceConfig, SourceConfigError, parse_source_config
from cardvision.data.targets import TargetRules
from support_data import VIDEO_ID

WIDTH, HEIGHT = 1000, 500
HAND_SOURCE = "digital/client-b"


def fractions(points) -> tuple:
    return tuple((x / WIDTH, y / HEIGHT) for x, y in points)


def upright(x, y, width, height) -> tuple:
    return fractions(geometry.quad_from_box(x, y, width, height))


def face(quad, card_id="SET-001", **fields) -> Box:
    values = {"oriented": True, "shape": "rotated-box", "card_id": card_id,
              "identity": "named" if card_id else "unknown", "readable": "yes"}
    values.update(fields)
    return Box(kind="face", quad=quad, **values)


def row(*boxes: Box, **fields) -> Row:
    values = {"platform": "youtube", "video_id": VIDEO_ID, "t": 12.517, "frame_width": WIDTH, "frame_height": HEIGHT,
              "source_type": "physical/tournament", "group": "g-test-01", "role": "train", "trained_on": False,
              "completeness": "exhaustive-boxes", "credit": "Example Labeler", "boxes": tuple(boxes)}
    values.update(fields)
    return Row(**values)


# ------------------------------------------------------------------------------- geometry
def test_fractions_become_pixels_and_back():
    quad = [[0.1, 0.2], [0.3, 0.2], [0.3, 0.6], [0.1, 0.6]]
    pixels = geometry.to_pixels(quad, 1920, 1080)
    assert pixels.tolist() == [[192.0, 216.0], [576.0, 216.0], [576.0, 648.0], [192.0, 648.0]]
    assert geometry.to_fractions(pixels, 1920, 1080) == quad
    assert geometry.to_pixels(quad, 960, 540).tolist() == (pixels / 2).tolist()
    assert geometry.to_fractions([[1, 1]], 3, 7, digits=None) == [[1 / 3, 1 / 7]]
    assert geometry.to_fractions([[1, 1]], 3, 7) == [[0.333333, 0.142857]]


def test_a_box_gives_its_pixel_box_for_any_frame_size_clipped_to_the_frame():
    box = Box(kind="back", quad=((0.95, 0.2), (1.05, 0.2), (1.05, 0.6), (0.95, 0.6)))
    assert box.pixel_box(1000, 500) == pytest.approx((950, 100, 50, 200))
    assert box.pixel_box(1000, 500, clip=False) == pytest.approx((950, 100, 100, 200))
    assert box.pixel_box(2000, 1000) == pytest.approx((1900, 200, 100, 400))
    assert np.allclose(box.pixel_quad(1000, 500), [[950, 100], [1050, 100], [1050, 300], [950, 300]])


def test_a_row_gives_its_boxes_and_points_in_pixels():
    labeled = row(face(upright(100, 100, 144, 200)), Box(kind="back", quad=upright(950, 300, 100, 140)),
                  points=(Point(x=0.25, y=0.5, kind="not-card"),))
    assert labeled.pixel_boxes(WIDTH, HEIGHT) == [pytest.approx((100, 100, 144, 200)), pytest.approx((950, 300, 50, 140))]
    assert labeled.pixel_boxes(WIDTH, HEIGHT, clip=False)[1] == pytest.approx((950, 300, 100, 140))
    assert labeled.pixel_boxes(500, 250)[0] == pytest.approx((50, 50, 72, 100))
    assert labeled.pixel_points(1920, 1080) == [(480.0, 540.0)]
    assert row(completeness="points", boxes=None).pixel_boxes(WIDTH, HEIGHT) == []


def test_a_rectangle_turned_a_quarter_clockwise():
    # A card 40 wide and 60 tall, turned so that its top edge points down the screen: its own
    # top-left corner is then at the top-right of the space it takes up on screen.
    quad = geometry.quad_from_rotated_box(100, 100, 40, 60, 90)
    assert np.allclose(quad, [[130, 80], [130, 120], [70, 120], [70, 80]])
    assert geometry.signed_area(quad) == pytest.approx(40 * 60)
    assert geometry.side_lengths(quad) == pytest.approx((40, 60))
    assert geometry.side_aspect(quad) == pytest.approx(40 / 60)
    assert geometry.short_side(quad) == pytest.approx(40)
    assert geometry.top_edge_angle(quad) == pytest.approx(90)
    assert geometry.bounding_box(quad) == pytest.approx((70, 80, 60, 40))


@pytest.mark.parametrize(("angle", "turns"), [(0, 0), (90, 1), (180, 2), (-90, 3), (270, 3), (44, 0), (46, 1), (-44, 0), (-46, 3), (170, 2)])
def test_quarter_turns_follow_the_top_edge(angle, turns):
    quad = geometry.quad_from_rotated_box(200, 200, 72, 100, angle)
    assert geometry.quarter_turns(quad) == turns
    # For a rectangle the up direction and the top edge turn together.
    assert (geometry.up_angle(quad) - geometry.top_edge_angle(quad) + 180) % 360 - 180 == pytest.approx(0, abs=1e-9)


def test_quarter_turns_go_by_the_way_from_the_center_to_the_top_edge():
    """A quad that follows perspective: its top edge slopes 50 degrees, which alone would say
    "a quarter turn", but the middle of that edge lies almost straight above the quad's center,
    so the card is upright."""
    quad = [[100, 100], [160, 171.5], [150, 300], [90, 240]]
    assert geometry.is_convex(quad) and geometry.signed_area(quad) > 0
    assert geometry.top_edge_angle(quad) == pytest.approx(50, abs=0.1)
    assert abs(geometry.up_angle(quad)) < 10
    assert geometry.quarter_turns(quad) == 0
    assert geometry.up_angle([[0, 0], [10, 0], [10, 20], [0, 20]]) == 0
    assert geometry.up_angle([[10, 0], [10, 10], [0, 10], [0, 0]]) == pytest.approx(90)
    assert geometry.up_angle([[10, 10], [0, 10], [0, 0], [10, 0]]) == pytest.approx(180)
    assert geometry.up_angle([[0, 10], [0, 0], [10, 0], [10, 10]]) == pytest.approx(-90)


def test_corners_are_put_clockwise_without_moving_the_first():
    counter = [[0, 0], [0, 10], [5, 10], [5, 0]]  # top-left, bottom-left, bottom-right, top-right
    assert geometry.clockwise(counter) == [[0, 0], [5, 0], [5, 10], [0, 10]]
    assert geometry.clockwise(geometry.clockwise(counter)) == geometry.clockwise(counter)
    assert geometry.signed_area(counter) == -50 and geometry.signed_area(geometry.clockwise(counter)) == 50


def test_convexity():
    assert geometry.is_convex([[0, 0], [5, 0], [5, 10], [0, 10]])
    assert geometry.is_convex([[0, 0], [5, 1], [6, 10], [-1, 8]])
    assert not geometry.is_convex([[0, 0], [5, 10], [5, 0], [0, 10]])  # crossed
    assert not geometry.is_convex([[0, 0], [10, 0], [2, 2], [0, 10]])  # dented
    assert not geometry.is_convex([[0, 0], [5, 0], [10, 0], [0, 10]])  # three in a line


@pytest.mark.parametrize("turns", [0, 1, 2, 3, 5])
def test_a_box_is_followed_through_a_turn_of_its_crop(turns):
    crop = np.zeros((6, 10), np.uint8)  # 10 wide, 6 tall
    crop[2:3, 1:4] = 1  # the box x=1, y=2, 3 wide, 1 tall
    turned = np.rot90(crop, turns)
    ys, xs = np.nonzero(turned)
    expected = (xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
    assert geometry.rotate_box_with_crop((1, 2, 3, 1), 10, 6, turns) == expected


# ------------------------------------------------------------------------ detector targets
def test_each_kind_of_box_becomes_the_right_detector_target():
    labeled = row(
        face(upright(100, 100, 144, 200)),                                              # a whole card, 0.72
        face(fractions(geometry.quad_from_rotated_box(500, 250, 144, 200, 90))),        # the same, on its side
        Box(kind="back", quad=upright(800, 300, 100, 140)),
        face(upright(100, 350, 144, 200), ignore=True),                                 # marked ignore
        Box(kind="unreadable", quad=upright(300, 20, 60, 60)),
        Box(kind="not-card", quad=upright(600, 20, 100, 140)),                          # background
        Box(kind="back", quad=upright(700, 20, 100, 140), ignore=True),
        Box(kind="not-card", quad=upright(850, 20, 100, 140), ignore=True),             # ignore wins
    )
    found = targets.detector_targets(labeled)
    assert [(target.kind, target.reason, target.box_index) for target in found] == [
        ("face", "face", 0), ("face", "face", 1), ("back", "back", 2), ("ignore", "ignore", 3),
        ("ignore", "unreadable", 4), ("ignore", "ignore", 6), ("ignore", "ignore", 7),
    ]
    first, second, back = found[0], found[1], found[2]
    assert (first.cx, first.cy, first.w, first.h) == pytest.approx((0.172, 0.4, 0.144, 0.4))
    # On its side the card takes up 200 x 144 pixels around (500, 250).
    assert (second.cx, second.cy, second.w, second.h) == pytest.approx((0.5, 0.5, 0.2, 0.288))
    assert (back.cx, back.cy, back.w, back.h) == pytest.approx((0.85, 0.74, 0.1, 0.28))
    assert first.as_dict() == pytest.approx({"cx": 0.172, "cy": 0.4, "w": 0.144, "h": 0.4, "k": 0})
    assert back.as_dict()["k"] == 1 and found[3].as_dict()["k"] == targets.DETECTOR_IGNORE_CODE
    assert back.as_dict({"face": 5, "back": 6, "ignore": 7})["k"] == 6


def test_the_detector_codes_are_the_trainer_s():
    """Faces and backs are the detector's two classes, 0 and 1; an ignore region is 2. The
    detector trainer reads ``k`` with the same three meanings."""
    import inspect

    from cardvision.core.constants import DETECTOR_BACK_CLASS, DETECTOR_FACE_CLASS, DETECTOR_MIN_CROP_PX

    assert targets.DETECTOR_CODES == {"face": 0, "back": 1, "ignore": 2}
    assert (DETECTOR_FACE_CLASS, DETECTOR_BACK_CLASS, targets.DETECTOR_IGNORE_CODE) == (0, 1, 2)
    assert inspect.signature(targets.detector_labels).parameters["codes"].default is targets.DETECTOR_CODES
    labeled = row(face(UPRIGHT), Box(kind="back", quad=upright(600, 100, 144, 200)), face(upright(300, 300, 144, 200), ignore=True))
    [record] = targets.detector_labels([labeled])
    assert [card["k"] for card in record["cards"]] == [0, 1, 2]
    # A negative crop must be at least the detector's smallest valid box on both sides.
    assert inspect.signature(targets.negative_crops).parameters["min_side"].default == DETECTOR_MIN_CROP_PX == 24


def test_the_aspect_filter_and_its_two_exemptions():
    square = upright(300, 100, 190, 200)        # 0.95: too square for a whole card
    strip = upright(300, 100, 200, 60)          # 0.30: a strip
    narrow_ok = upright(300, 100, 112, 200)     # 0.56
    narrow_out = upright(300, 100, 108, 200)    # 0.54
    wide_ok = upright(300, 100, 178, 200)       # 0.89
    hand = upright(300, 360, 150, 140)          # 0.933, center 430 / 500 = 0.86 of the way down
    hand_high = upright(300, 300, 150, 140)     # the same shape, center at 0.74
    hand_strip = upright(300, 440, 200, 60)     # in the hand row, but a strip

    def decide(quad, source_type="physical/tournament", rules=TargetRules(), **fields):
        return targets.classify_box(row(source_type=source_type), face(quad, **fields), rules)

    assert decide(square) == ("ignore", "aspect")
    assert decide(strip) == ("ignore", "aspect")
    assert decide(narrow_ok) == ("face", "face") and decide(wide_ok) == ("face", "face")
    assert decide(narrow_out) == ("ignore", "aspect")
    # Unnamed faces are filtered like named ones.
    assert decide(square, card_id=None) == ("ignore", "aspect")

    assert decide(square, attributes={"visible_share": "half"}) == ("face", "face-half")
    assert decide(strip, attributes={"visible_share": "half"}) == ("face", "face-half")
    assert decide(square, attributes={"visible_share": "full"}) == ("ignore", "aspect")
    assert decide(square, rules=TargetRules(keep_half_cards=False), attributes={"visible_share": "half"}) == ("ignore", "aspect")

    hand_rules = TargetRules(hand_row_source_types=(HAND_SOURCE,))
    assert decide(hand, HAND_SOURCE, hand_rules) == ("face", "face-hand-row")
    assert decide(hand, HAND_SOURCE) == ("ignore", "aspect")                      # the rule is off by default
    assert decide(hand, "digital/client-a", hand_rules) == ("ignore", "aspect")   # another client
    assert decide(hand_high, HAND_SOURCE, hand_rules) == ("ignore", "aspect")     # not in the hand row
    assert decide(hand_strip, HAND_SOURCE, hand_rules) == ("ignore", "aspect")    # only squarer cards are exempt
    assert decide(hand_high, HAND_SOURCE, TargetRules(hand_row_source_types=(HAND_SOURCE,), hand_row_min_center_y=0.7)) \
        == ("face", "face-hand-row")

    assert decide(square, rules=TargetRules(aspect_range=None)) == ("face", "face")
    assert decide(narrow_out, rules=TargetRules(aspect_range=(0.5, 0.9))) == ("face", "face")
    # The filter measures real pixels: the same fractions on a frame of another shape differ.
    tall_frame = row(frame_width=500, frame_height=1000)
    assert targets.classify_box(tall_frame, face(upright(100, 100, 144, 200))) == ("ignore", "aspect")


def test_the_rules_take_the_hand_row_from_a_sources_file():
    config = parse_source_config({"sourceTypes": ["digital/client-a", HAND_SOURCE],
                                  "handRow": {"sourceTypes": [HAND_SOURCE], "minCenterY": 0.8}})
    assert config.source_types[-2:] == ("digital/client-a", HAND_SOURCE) and "physical/webcam" in config.source_types
    rules = TargetRules.for_sources(config, keep_half_cards=False)
    assert rules.hand_row_source_types == (HAND_SOURCE,) and rules.hand_row_min_center_y == 0.8
    assert rules.keep_half_cards is False and rules.aspect_range == (0.55, 0.90)
    assert parse_source_config({}) == SourceConfig()
    for broken in ([], {"sourceTypes": "digital/x"}, {"sourceTypes": ["studio"]}, {"other": 1},
                   {"handRow": {"sourceTypes": ["digital/unlisted"]}}, {"handRow": {"minCenterY": 2}},
                   {"handRow": {"rows": 1}}):
        with pytest.raises(SourceConfigError):
            parse_source_config(broken)


def test_a_target_box_is_clipped_to_the_frame_and_dropped_when_nothing_is_left():
    labeled = row(
        face(upright(950, 100, 144, 200)),          # runs past the right edge
        face(upright(-44, -100, 144, 200)),         # runs past the top-left corner
        Box(kind="back", quad=upright(1100, 100, 100, 140)),  # wholly outside
    )
    found = targets.detector_targets(labeled)
    assert [target.box_index for target in found] == [0, 1]
    assert (found[0].cx, found[0].cy, found[0].w, found[0].h) == pytest.approx((0.975, 0.4, 0.05, 0.4))
    assert (found[1].cx, found[1].cy, found[1].w, found[1].h) == pytest.approx((0.05, 0.1, 0.1, 0.2))


def test_detector_labels_and_counts_over_rows():
    first = row(face(upright(100, 100, 144, 200)), Box(kind="back", quad=upright(800, 300, 100, 140)),
                face(upright(300, 100, 190, 200)), Box(kind="not-card", quad=upright(600, 20, 100, 140)))
    empty = row(t=20.0)
    records = targets.detector_labels([first, empty])
    assert [record["t"] for record in records] == [12.517, 20.0]
    assert records[0]["W"] == WIDTH and records[0]["H"] == HEIGHT and records[0]["group"] == "g-test-01"
    assert [card["k"] for card in records[0]["cards"]] == [0, 1, 2] and records[1]["cards"] == []
    assert set(records[0]["cards"][0]) == {"cx", "cy", "w", "h", "k"}
    assert targets.count_targets([first, empty]) == {
        "frames": 2, "face": 1, "back": 1, "ignore": 1,
        "reasons": {"face": 1, "back": 1, "aspect": 1, "not-card": 1},
    }


def test_detector_targets_refuse_a_row_that_is_not_boxed_exhaustively():
    points_only = row(completeness="points", boxes=None)
    with pytest.raises(ValueError, match="not boxed exhaustively"):
        targets.detector_targets(points_only)


# ------------------------------------------------------------------------- recognizer crops
def card_picture(width=144, height=200) -> np.ndarray:
    """An upright card: a gradient with a white patch in its own top-left corner."""
    y, x = np.mgrid[0:height, 0:width]
    card = np.stack([x % 256, y % 256, (x + y) % 256], axis=-1).astype(np.uint8)
    card[:20, :20] = 255
    return card


def scene() -> np.ndarray:
    """A 1000 x 500 frame with one upright card at (100, 100) and the same card turned a
    quarter clockwise in the 200 x 144 space at (400, 178)."""
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    frame = np.stack([(x * 3) % 200, (y * 5) % 200, (x + 2 * y) % 200], axis=-1).astype(np.uint8)
    frame[100:300, 100:244] = card_picture()
    frame[178:322, 400:600] = np.rot90(card_picture(), -1)
    return frame


UPRIGHT = upright(100, 100, 144, 200)
ON_ITS_SIDE = fractions(geometry.quad_from_rotated_box(500, 250, 144, 200, 90))


def store_frame(folder, labeled: Row, frame: np.ndarray) -> FrameStore:
    Image.fromarray(frame).save(folder / frame_file_name(labeled, ".png"))
    return FrameStore(folder)


def same(frame):
    return frame


def test_a_crop_is_the_box_with_its_margin(tmp_path):
    labeled = row(face(UPRIGHT, id="b1"))
    store = store_frame(tmp_path, labeled, scene())
    [crop] = targets.recognizer_crops([labeled], store, margin=0.25)
    # 144 x 200 at (100, 100), widened by 36 left and right and 50 above and below.
    assert crop.image.shape == (300, 216, 3)
    assert np.array_equal(crop.image, scene()[50:350, 64:280])
    assert crop.inner == pytest.approx((36, 50, 144, 200))
    assert np.array_equal(crop.image[50:250, 36:180], card_picture())
    assert (crop.card_id, crop.label, crop.turns, crop.box_index, crop.half) == ("SET-001", "face", 0, 0, None)
    assert crop.angle == pytest.approx(0) and crop.aspect == pytest.approx(0.72) and crop.short_side == pytest.approx(144)
    assert crop.name == "youtube-EXAMPLE0001-t0000012517-b1.png"

    [tight] = targets.recognizer_crops([labeled], tmp_path, margin=0)
    assert np.array_equal(tight.image, card_picture()) and tight.inner == pytest.approx((0, 0, 144, 200))


def test_a_turned_card_comes_out_upright(tmp_path):
    labeled = row(face(ON_ITS_SIDE))
    store = store_frame(tmp_path, labeled, scene())
    [tight] = targets.recognizer_crops([labeled], store, margin=0)
    assert tight.turns == 1 and tight.angle == pytest.approx(90)
    assert np.array_equal(tight.image, card_picture())
    assert tight.inner == pytest.approx((0, 0, 144, 200))
    assert tight.name == "youtube-EXAMPLE0001-t0000012517-box000.png"

    # With a margin of a tenth: the space on screen is (400, 178, 200, 144); widened it is
    # (380, 163.6, 240, 172.8), which truncates to columns 380..620 and rows 163..336.
    [crop] = targets.recognizer_crops([labeled], store, margin=0.1)
    assert crop.image.shape == (240, 173, 3)
    assert np.array_equal(crop.image, np.rot90(scene()[163:336, 380:620], 1))
    assert crop.inner == pytest.approx((15, 20, 144, 200))
    assert np.array_equal(crop.image[20:220, 15:159], card_picture())

    # Without a known orientation the crop is left as it lies.
    unoriented = row(face(ON_ITS_SIDE, oriented=False))
    [flat] = targets.recognizer_crops([unoriented], store, margin=0)
    assert flat.turns == 0 and np.array_equal(flat.image, scene()[178:322, 400:600])


def test_a_margin_is_cut_short_by_the_edge_of_the_frame(tmp_path):
    labeled = row(face(upright(10, 20, 144, 200)), face(upright(900, 350, 144, 200)))
    store = store_frame(tmp_path, labeled, scene())
    near, far = targets.recognizer_crops([labeled], store, margin=0.25)
    assert near.image.shape == (270, 190, 3) and near.inner == pytest.approx((10, 20, 144, 200))
    assert np.array_equal(near.image, scene()[0:270, 0:190])
    # The second card runs past the right and bottom edges; its box is clipped with the crop.
    assert far.image.shape == (200, 136, 3) and far.inner == (36, 50, 100, 150)
    assert all(isinstance(value, int) for value in (*near.inner, *far.inner))
    with pytest.raises(ValueError, match="margin"):
        targets.margin_crop(scene(), (10, 20, 144, 200), -0.1)


def marked_pixels(frame_shape, box, margin, turns):
    """The definition ``margin_crop`` is held to, written out with a mask: cut the widened box
    to whole pixels, mark in that cut the pixels of the box's own whole-pixel crop, turn both,
    and read off where the marks lie. Returns the cut's edges and the marks' box."""
    frame_height, frame_width = frame_shape
    x, y, w, h = box
    bx0, by0 = int(max(0, x)), int(max(0, y))
    bx1, by1 = int(min(frame_width, x + w)), int(min(frame_height, y + h))
    x0, y0 = int(max(0, x - margin * w)), int(max(0, y - margin * h))
    x1, y1 = int(min(frame_width, x + w + margin * w)), int(min(frame_height, y + h + margin * h))
    mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    mask[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0] = True
    rows, columns = np.where(np.rot90(mask, turns))
    marks = (int(columns.min()), int(rows.min()), int(columns.max() - columns.min() + 1), int(rows.max() - rows.min() + 1))
    return (x0, y0, x1, y1), marks


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
@pytest.mark.parametrize("grid", [None, 1, 0])
def test_inner_is_the_box_s_own_pixels_turned_with_the_crop(turns, grid):
    """Random boxes, with edges anywhere, on tenths of a pixel and on whole pixels: the cut is
    the widened box truncated edge by edge, and ``inner`` is the box's own truncated crop."""
    rng = np.random.default_rng(7 * turns + (0 if grid is None else grid + 1))
    frame = rng.integers(0, 256, size=(360, 640, 3), dtype=np.uint8)
    checked = 0
    for _ in range(1500):
        box = (rng.uniform(-40, 660), rng.uniform(-40, 380), rng.uniform(3, 300), rng.uniform(3, 300))
        if grid is not None:
            box = tuple(round(float(value), grid) for value in box)
        x, y, w, h = box
        if int(min(640, x + w)) <= int(max(0, x)) or int(min(360, y + h)) <= int(max(0, y)):
            continue                                    # the box has no pixel in the frame
        margin = float(rng.choice([0.0, 0.1, 0.25, 0.4]))
        crop, inner = targets.margin_crop(frame, box, margin, turns)
        (x0, y0, x1, y1), marks = marked_pixels(frame.shape[:2], box, margin, turns)
        assert np.array_equal(crop, np.rot90(frame[y0:y1, x0:x1], turns)), (box, margin)
        assert inner == marks, (box, margin)
        assert 0 <= inner[0] and 0 <= inner[1] and inner[0] + inner[2] <= crop.shape[1] and inner[1] + inner[3] <= crop.shape[0]
        checked += 1
    assert checked > 1200


def test_which_faces_become_crops(tmp_path):
    hand = upright(300, 360, 150, 140)
    labeled = row(
        face(UPRIGHT),                                                                 # 0 kept
        face(upright(300, 20, 72, 100), card_id=None, identity="unknown"),             # 1
        face(upright(400, 20, 72, 100), card_id=None, identity="unlabeled"),          # 2
        face(upright(500, 20, 72, 100), readable="no"),                                # 3
        face(upright(600, 20, 72, 100), readable="maybe"),                             # 4 kept
        face(upright(700, 20, 72, 100), readable=None),                                # 5 kept
        face(upright(800, 20, 72, 100), ignore=True),                                  # 6
        face(upright(900, 20, 72, 100), card_id="SET-099"),                            # 7
        face(upright(700, 200, 95, 100)),                                              # 8 too square
        face(upright(800, 200, 95, 100), attributes={"visible_share": "half"}),        # 9 kept
        face(hand),                                                                    # 10 kept by the hand rule
        Box(kind="back", quad=upright(600, 350, 72, 100)),                             # 11
        Box(kind="not-card", quad=upright(700, 350, 72, 100)),                         # 12
        source_type=HAND_SOURCE,
    )
    rules = TargetRules(hand_row_source_types=(HAND_SOURCE,))
    reasons = [targets.crop_exclusion(labeled, box, rules, exclude_ids={"SET-099"}) for box in labeled.boxes]
    assert reasons == [None, "unknown", "unlabeled", "unreadable", None, None, "ignore", "excluded-id", "aspect",
                       None, None, "not-face", "not-face"]

    store = store_frame(tmp_path, labeled, scene())
    skipped: Counter = Counter()
    crops = list(targets.recognizer_crops([labeled], store, margin=0, rules=rules, exclude_ids={"SET-099"},
                                          skipped=skipped, prepare=same))
    assert [crop.box_index for crop in crops] == [0, 4, 5, 9, 10]
    assert [crop.half for crop in crops] == [None, None, None, "half", "hand-row"]
    assert skipped == {"unknown": 1, "unlabeled": 1, "unreadable": 1, "ignore": 1, "excluded-id": 1, "aspect": 1}
    # Without the hand rule the hand card is filtered out too.
    assert [crop.box_index for crop in targets.recognizer_crops([labeled], store, margin=0, exclude_ids={"SET-099"})] \
        == [0, 4, 5, 9]
    with pytest.raises(ValueError, match='"SET-099" is not a known id'):
        list(targets.recognizer_crops([labeled], store, margin=0, known_ids={"SET-001"}))
    assert len(list(targets.recognizer_crops([labeled], store, margin=0, known_ids={"SET-001", "SET-099"}))) == 5


def test_crops_are_cut_from_the_frame_the_detector_would_analyze(tmp_path):
    # A frame 500 pixels wide is analyzed at 720: a box a fifth of the frame wide is 144 there.
    small = row(face(((0.2, 0.2), (0.4, 0.2), (0.4, 0.6), (0.2, 0.6))), frame_width=500, frame_height=250)
    y, x = np.mgrid[0:250, 0:500]
    store = store_frame(tmp_path, small, np.stack([x % 256, y % 256, x % 256], axis=-1).astype(np.uint8))
    rules = TargetRules(aspect_range=None)
    [analyzed] = targets.recognizer_crops([small], store, margin=0, rules=rules)
    assert analyzed.image.shape == (144, 144, 3) and analyzed.short_side == pytest.approx(144)
    [native] = targets.recognizer_crops([small], store, margin=0, rules=rules, prepare=same)
    assert native.image.shape == (100, 100, 3)


def test_a_frame_of_another_size_serves_and_a_frame_of_another_shape_does_not(tmp_path):
    labeled = row(face(UPRIGHT))
    half_size = np.asarray(Image.fromarray(scene()).resize((800, 400), Image.NEAREST))
    store = store_frame(tmp_path, labeled, half_size)
    [crop] = targets.recognizer_crops([labeled], store, margin=0, prepare=same)
    assert crop.image.shape == (160, 115, 3)  # 144 x 200 at four fifths, edges truncated

    other = row(face(UPRIGHT), t=99.0)
    store = store_frame(tmp_path, other, np.zeros((600, 800, 3), np.uint8))
    skipped: Counter = Counter()
    assert list(targets.recognizer_crops([other, row(face(UPRIGHT), t=100.0)], store, margin=0, skipped=skipped)) == []
    assert skipped == {"no-frame": 2}


def test_backs_and_boxes_that_are_not_cards_give_negative_crops(tmp_path):
    labeled = row(
        Box(kind="back", quad=upright(100, 100, 144, 200)),
        Box(kind="not-card", quad=ON_ITS_SIDE, oriented=True),
        Box(kind="back", quad=upright(700, 20, 20, 28)),                  # under 24 pixels
        Box(kind="back", quad=upright(800, 20, 72, 100), ignore=True),
        Box(kind="unreadable", quad=upright(900, 20, 72, 100)),
        Box(kind="back", quad=upright(980, 100, 144, 200)),               # only 20 pixels are in frame
        face(upright(300, 20, 72, 100)),
    )
    store = store_frame(tmp_path, labeled, scene())
    skipped: Counter = Counter()
    crops = list(targets.negative_crops([labeled], store, margin=0.25, skipped=skipped))
    assert [(crop.label, crop.box_index, crop.card_id, crop.turns) for crop in crops] == [("back", 0, None, 0), ("not-card", 1, None, 0)]
    assert np.array_equal(crops[0].image, scene()[50:350, 64:280]) and crops[0].inner == pytest.approx((36, 50, 144, 200))
    # A negative is never turned: (400, 178, 200, 144) with a quarter margin.
    assert np.array_equal(crops[1].image, scene()[142:358, 350:650]) and crops[1].inner == pytest.approx((50, 36, 200, 144))
    assert skipped == {"small": 2}
    assert [crop.box_index for crop in targets.negative_crops([labeled], store, margin=0, kinds=("back",), min_side=10)] == [0, 2, 5]


def test_crops_are_written_with_an_index(tmp_path):
    labeled = row(face(UPRIGHT, id="b1", origin="prefill"), face(ON_ITS_SIDE, card_id="SET-002", readable="maybe"))
    frames = tmp_path / "frames"
    frames.mkdir()
    store = store_frame(frames, labeled, scene())
    out = tmp_path / "crops"
    assert targets.write_crops(targets.recognizer_crops([labeled], store, margin=0.25), out) == 2
    lines = [json.loads(line) for line in (out / "index.jsonl").read_text().splitlines()]
    assert [line["file"] for line in lines] == ["youtube-EXAMPLE0001-t0000012517-b1.png",
                                                "youtube-EXAMPLE0001-t0000012517-box001.png"]
    assert {"file", "inner", "cardId", "group"} <= set(lines[0])
    assert lines[0]["inner"] == [36, 50, 144, 200] and lines[0]["cardId"] == "SET-001" and lines[0]["group"] == "g-test-01"
    assert (lines[0]["origin"], lines[0]["readable"], lines[0]["turns"], lines[0]["hw"]) == ("prefill", "yes", 0, [300, 216])
    assert (lines[1]["cardId"], lines[1]["readable"], lines[1]["turns"], lines[1]["angle"]) == ("SET-002", "maybe", 1, 90.0)
    with Image.open(out / lines[0]["file"]) as saved:
        assert np.array_equal(np.asarray(saved), scene()[50:350, 64:280])
