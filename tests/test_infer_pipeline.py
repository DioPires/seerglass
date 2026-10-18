# SPDX-License-Identifier: Apache-2.0
"""The pipeline from a frame to boxes and names, on a generated table of toy cards."""

from __future__ import annotations

import json

import numpy as np
import pytest

from cardvision.core.catalog import load_catalog
from cardvision.core.images import load_rgb
from cardvision.detect import Detection
from cardvision.evaluate.hover import HoverLayer
from cardvision.infer import Pipeline, analysis_frame, analysis_size, card_json, frame_box, prefill_box, reported_box
from cardvision.infer.frames import MAX_ANALYSIS_WIDTH, MIN_ANALYSIS_WIDTH, round_half_up
from cardvision.recognize import load_accept
from helpers import pattern
from test_infer_toy_models import (
    EXPECTED_BOXES,
    EXPECTED_CLASSES,
    EXPECTED_NAMES,
    EXPECTED_SCORES,
    SCENE_CARDS,
    SCENE_SIZE,
    ToySetup,
    build_detector,
    detector_rows,
)


@pytest.fixture(scope="module")
def setup(tmp_path_factory, toy) -> ToySetup:
    return ToySetup(tmp_path_factory.mktemp("pipeline"), *toy)


@pytest.fixture(scope="module")
def pipeline(setup) -> Pipeline:
    return Pipeline.open(setup.detector, setup.recognizer, setup.bank, load_catalog(setup.catalog), load_accept(setup.accept))


@pytest.fixture(scope="module")
def result(setup, pipeline):
    return pipeline.analyze(setup.frame, name="table.png", source="memory", t=1.5)


# ----------------------------------------------------------------------------- the analysis frame


def test_analysis_width_range():
    assert (MIN_ANALYSIS_WIDTH, MAX_ANALYSIS_WIDTH) == (720, 1920)


def test_a_frame_within_range_is_analyzed_as_it_is():
    frame = pattern(405, 720)
    assert analysis_frame(frame) is frame
    wide = pattern(1080, 1920)
    assert analysis_frame(wide) is wide


def test_a_narrow_frame_is_enlarged_and_a_wide_one_reduced():
    assert analysis_frame(pattern(360, 640)).shape == (405, 720, 3)
    assert analysis_frame(pattern(1440, 2560)).shape == (1080, 1920, 3)
    assert analysis_frame(pattern(900, 500)).shape == (1296, 720, 3)       # portrait: the width still decides
    assert analysis_frame(pattern(407, 719)).shape == (408, 720, 3)        # 407 * 720 / 719 = 407.57 -> 408


def test_a_height_exactly_between_two_whole_numbers_is_rounded_up():
    """The browser runtime's rule (``Math.round``). Python's ``round`` would send 406.5 to 406."""
    assert [round_half_up(value) for value in (0.5, 1.5, 2.5, 406.5, 407.5, 2.4, 2.6, 3.0)] == [1, 2, 3, 407, 408, 2, 3, 3]
    # 480 x 271 enlarged to 720 wide is 406.5 high.
    assert analysis_size(480, 271) == (720, 407)
    assert analysis_frame(pattern(271, 480)).shape == (407, 720, 3)
    # The halves on either side of it: 403.5, 409.5 and 412.5.
    assert [analysis_size(480, height) for height in (269, 273, 275)] == [(720, 404), (720, 410), (720, 413)]
    # A wide frame reduced: 3840 x 2161 to 1920 wide is 1080.5 high, and 3840 x 2159 is 1079.5.
    assert analysis_size(3840, 2161) == (1920, 1081) and analysis_size(3840, 2159) == (1920, 1080)
    assert analysis_frame(pattern(2161, 3840)).shape == (1081, 1920, 3)
    # Away from a half, and for a frame within the range, nothing changes.
    assert analysis_size(719, 407) == (720, 408) and analysis_size(854, 480) == (854, 480)
    assert analysis_size(1280, 720, 320, 640) == (640, 360)


def test_the_resize_is_pillows_bilinear_filter():
    from PIL import Image

    frame = pattern(360, 640)
    expected = np.asarray(Image.fromarray(frame).resize((720, 405), Image.BILINEAR))
    np.testing.assert_array_equal(analysis_frame(frame), expected)


# ----------------------------------------------------------------------------- boxes in the frame


def test_frame_box_scales_and_clips_and_does_not_round():
    # Scaled by 2 into an 80 x 600 frame: x from -10 to 90 is clipped to 0..80, y from 20 to 2020 to 20..600.
    assert frame_box(Detection(-5, 10, 50, 1000, 0.9, 0), 2.0, 2.0, 80, 600) == (0.0, 20.0, 80.0, 580.0)
    # Not rounded: 1/3 * 4/3 = 0.4444..., (1/3 + 10) * 4/3 = 13.7777...
    scaled = frame_box(Detection(1 / 3, 1 / 3, 10, 20, 0.9, 0), 4 / 3, 4 / 3, 2560, 1440)
    assert scaled == pytest.approx((4 / 9, 4 / 9, 40 / 3, 80 / 3), abs=1e-12) and scaled[0] != 0.444
    # In the analyzed frame itself (both scales 1) a box that lies inside is returned as it is.
    inside = Detection(299.99998, 100.2, 336.1, 240.3, 0.9, 0)
    assert frame_box(inside, 1.0, 1.0, 854, 480) == (299.99998, 100.2, 336.1, 240.3)
    # One that hangs over is clipped, and one wholly outside is left with no width.
    assert frame_box(Detection(-5.5, 10.25, 50, 1000, 0.9, 0), 1.0, 1.0, 80, 600) == (0.0, 10.25, 44.5, 589.75)
    assert frame_box(Detection(90, 10, 50, 100, 0.9, 0), 1.0, 1.0, 80, 600) == (90.0, 10.0, 0.0, 100.0)


def test_a_box_is_rounded_to_three_decimals_where_it_is_reported():
    assert reported_box((4 / 9, 4 / 9, 40 / 3, 80 / 3)) == [0.444, 0.444, 13.333, 26.667]
    assert reported_box((299.9997, 100, 336.0004, 239.9996)) == [300.0, 100.0, 336.0, 240.0]


class FixedDetector:
    """Stands in for a detector model: it returns the detections it was given, whatever the frame."""

    def __init__(self, *detections: Detection) -> None:
        self.detections = list(detections)

    def detect(self, rgb: np.ndarray, threshold: float | None = None) -> list[Detection]:
        return list(self.detections)


def test_the_crop_is_cut_from_the_box_before_rounding(pipeline):
    """Edges within 0.0005 px under a whole number. Rounded to three decimals the box says 300
    and 240; the crop is cut from the box as decoded, so it starts at column 299 and is 239 rows
    high. The browser runtime cuts the same pixels."""
    frame = pattern(480, 854)
    under = Detection(299.9997, 100.0, 336.0, 239.9997, 0.9, 0)   # x from 299.9997 to 635.9997, y from 100 to 339.9997
    (card,) = Pipeline(FixedDetector(under), pipeline.recognizer).analyze(frame).cards
    assert card.box == (299.9997, 100.0, 336.0, 239.9997)
    assert card_json(card)["box"] == prefill_box(card)["box"] == [300.0, 100.0, 336.0, 240.0]
    np.testing.assert_array_equal(card.crop, frame[100:339, 299:635])
    assert not np.array_equal(card.crop[:, :-1], frame[100:339, 300:635])   # the patterned frame tells a column's shift

    over = Detection(300.0003, 100.0, 336.0, 240.0003, 0.9, 0)    # just over: x from 300.0003 to 636.0003
    (card,) = Pipeline(FixedDetector(over), pipeline.recognizer).analyze(frame).cards
    assert card_json(card)["box"] == [300.0, 100.0, 336.0, 240.0]
    np.testing.assert_array_equal(card.crop, frame[100:340, 300:636])


def test_a_far_edge_is_not_the_rounded_top_plus_the_rounded_height(pipeline):
    """The bottom edge is the decoded top plus the decoded height, 163.99988, so the crop ends
    with row 162. Printed, the top is 60.4 and the height 103.6, which add up to 164; a crop cut
    from those would hold one row more. The right edge, 269.99988, goes the same way."""
    frame = pattern(480, 854)
    detection = Detection(30.4, 60.4, 239.59988, 103.59988, 0.9, 0)
    (card,) = Pipeline(FixedDetector(detection), pipeline.recognizer).analyze(frame).cards
    assert card_json(card)["box"] == [30.4, 60.4, 239.6, 103.6]
    assert 163.9998 < card.box[1] + card.box[3] < 163.9999
    np.testing.assert_array_equal(card.crop, frame[60:163, 30:269])
    assert card.crop.shape[:2] == (103, 239)


def test_a_resized_frame_is_cropped_from_the_box_in_the_analysis_frame(pipeline):
    """A 2560 px frame is analyzed at 1920 px. The box is reported in the frame's own pixels,
    and the crop is cut from the detector's box in the analysis frame, not from the reported box
    scaled back: 299.9998 * 4 / 3 is reported as 400, and 400 * 3 / 4 would start the crop at 300."""
    frame = pattern(1440, 2560)
    analyzed = analysis_frame(frame)
    detection = Detection(299.9998, 150.0, 240.0, 336.0, 0.9, 0)
    result = Pipeline(FixedDetector(detection), pipeline.recognizer).analyze(frame)
    (card,) = result.cards
    assert (result.analysis_width, result.analysis_height) == (1920, 1080)
    assert card_json(card)["box"] == [400.0, 200.0, 320.0, 448.0]
    assert card.box[0] == pytest.approx(399.99973333, abs=1e-6)
    np.testing.assert_array_equal(card.crop, analyzed[150:486, 299:539])


def test_the_minimum_crop_size_is_judged_before_rounding(pipeline):
    frame = pattern(480, 854)
    one_short = Detection(10.0, 10.0, 23.9997, 100.0, 0.9, 0)      # columns 10 to 32: 23, though the rounded box is 24 wide
    just_enough = Detection(10.9997, 200.0, 23.0006, 100.0, 0.8, 0)   # columns 10 to 33: 24, though the rounded box is 23.001 wide
    cards = Pipeline(FixedDetector(one_short, just_enough), pipeline.recognizer).analyze(frame).cards
    assert [card_json(card)["box"] for card in cards] == [[11.0, 200.0, 23.001, 100.0]]
    assert cards[0].crop.shape[:2] == (100, 24)


def test_the_hover_rule_is_asked_about_the_box_before_rounding(pipeline):
    """A pointer between the decoded edge and the rounded one: inside when the edge lies just
    under a whole number, outside when it lies just over."""
    frame = pattern(480, 854)
    # The patterned frame resembles no card, so the selection that takes every box is asked.
    under = Detection(299.9997, 100.0, 336.0, 240.0, 0.9, 0)
    layer = HoverLayer.of(Pipeline(FixedDetector(under), pipeline.recognizer).analyze(frame), "detector-score")
    assert layer.at(299.99985, 200.0).index == 0 and layer.at(299.9995, 200.0).index is None
    over = Detection(300.0003, 100.0, 336.0, 240.0, 0.9, 0)
    layer = HoverLayer.of(Pipeline(FixedDetector(over), pipeline.recognizer).analyze(frame), "detector-score")
    assert layer.at(300.00015, 200.0).index is None and layer.at(300.0005, 200.0).index == 0


def test_boxes_classes_and_scores(result):
    assert [list(card.box) for card in result.cards] == EXPECTED_BOXES
    assert [card.class_name for card in result.cards] == EXPECTED_CLASSES
    assert [round(card.score, 6) for card in result.cards] == EXPECTED_SCORES
    assert [card.index for card in result.cards] == list(range(7))
    assert (result.frame_width, result.frame_height) == SCENE_SIZE
    assert (result.analysis_width, result.analysis_height) == SCENE_SIZE
    assert (result.name, result.source, result.t) == ("table.png", "memory", 1.5)
    assert result.ms >= 0


def test_names(result):
    faces = result.faces
    assert len(faces) == 6 and all(card.is_face for card in faces)
    assert [card.recognition.name for card in faces] == EXPECTED_NAMES
    assert [card.recognition.outcome for card in faces] == ["named", "named", "nothing", "nothing", "named", "nothing"]

    upright, turned, unknown, doubtful, landscape, clipped = (card.recognition for card in faces)
    assert (upright.card_id, upright.look, upright.retried) == ("TOY-001", 0, False)
    assert (turned.card_id, turned.look, turned.retried) == ("TOY-003", 3, True)
    assert (unknown.tau_passed, unknown.accepted, unknown.retried) == (False, False, True)
    assert (doubtful.tau_passed, doubtful.accepted) == (True, False) and doubtful.top_id in ("TOY-015", "TOY-016")
    assert (landscape.card_id, landscape.look, landscape.retried) == ("TOY-017", 0, False)
    assert clipped.accepted is False


def test_backs_are_reported_but_not_cropped_or_named(result):
    (back,) = [card for card in result.cards if not card.is_face]
    assert back.class_name == "back" and back.recognition is None and back.crop is None
    assert set(card_json(back)) == {"i", "box", "score", "class"}


def test_the_crop_is_exactly_the_pixels_under_the_box(setup, result):
    for card, (card_id, _x, _y, turns) in zip(result.faces[:5], SCENE_CARDS[:5]):
        expected = np.rot90(load_rgb(setup.art_dir / f"{card_id}.png"), turns)
        np.testing.assert_array_equal(card.crop, expected)
    clipped = result.faces[5]
    np.testing.assert_array_equal(clipped.crop, load_rgb(setup.art_dir / "TOY-005.png")[:, :120])


def test_json_shape(result):
    doc = json.loads(json.dumps(result.to_json()))
    assert set(doc) == {"source", "frame", "t", "frameW", "frameH", "analysisW", "analysisH", "boxes"}
    assert (doc["frame"], doc["t"], doc["frameW"], doc["frameH"]) == ("table.png", 1.5, 1280, 768)
    assert [box["box"] for box in doc["boxes"]] == EXPECTED_BOXES
    first = doc["boxes"][0]
    assert set(first) == {"i", "box", "score", "class", "outcome", "name", "cardId", "topId", "topName",
                          "similarity", "margin", "look", "retried", "tauPassed", "accepted"}
    assert (first["outcome"], first["name"], first["cardId"], first["look"], first["retried"]) == \
        ("named", "Amber Circle", "TOY-001", 0, False)
    assert first["tauPassed"] is True and first["accepted"] is True
    assert first["similarity"] == round(first["similarity"], 6) and first["similarity"] > 0.9999
    unknown = doc["boxes"][2]
    assert (unknown["outcome"], unknown["name"], unknown["cardId"]) == ("nothing", None, None)
    assert unknown["topId"] and unknown["topName"]


def test_the_models_are_run_on_the_analysis_frame_and_boxes_come_back_in_frame_pixels(setup, tmp_path):
    """The same table at twice the size: analyzed at 1920 px wide, reported at 2560."""
    big = np.repeat(np.repeat(setup.frame, 2, axis=0), 2, axis=1)
    boxes = [(80, 80, 480, 672), (640, 80, 672, 480)]
    rows = detector_rows(boxes, [0.9, 0.8], [0, 0], frame_size=(2560, 1536))
    detector = build_detector(tmp_path / "detector.onnx", rows)
    pipeline = Pipeline.open(detector, setup.recognizer, setup.bank, load_catalog(setup.catalog), load_accept(setup.accept))
    result = pipeline.analyze(big)
    assert (result.frame_width, result.frame_height) == (2560, 1536)
    assert (result.analysis_width, result.analysis_height) == (1920, 1152)
    assert [list(card.box) for card in result.cards] == [[80.0, 80.0, 480.0, 672.0], [640.0, 80.0, 672.0, 480.0]]
    # The crops are cut from the analysis frame: three quarters of the reported size.
    assert [card.crop.shape[:2] for card in result.cards] == [(504, 360), (360, 504)]
    # Enlarged and reduced again, the cards are no longer pixel-exact, yet they are still recognizable.
    assert [card.recognition.top_id for card in result.cards] == ["TOY-001", "TOY-003"]


def test_a_frame_with_no_cards(setup, tmp_path):
    detector = build_detector(tmp_path / "empty.onnx", detector_rows([(10, 10, 100, 100)], [0.1], [0]))
    pipeline = Pipeline.open(detector, setup.recognizer, setup.bank, load_catalog(setup.catalog))
    result = pipeline.analyze(setup.frame, name="empty.png")
    assert result.cards == () and result.faces == []
    assert result.to_json()["boxes"] == []


def test_a_frame_with_no_cards_keeps_the_best_score_the_detector_gave(setup, tmp_path):
    detector = build_detector(tmp_path / "near.onnx", detector_rows([(10, 10, 100, 100), (400, 10, 100, 100)], [0.21, 0.1], [0, 1]))
    pipeline = Pipeline.open(detector, setup.recognizer, setup.bank, load_catalog(setup.catalog))
    result = pipeline.analyze(setup.frame)
    assert result.cards == () and result.too_small == 0
    assert result.best_score == pytest.approx(0.21)
    assert result.to_json()["boxes"] == []


def test_boxes_dropped_for_size_are_counted_apart_from_the_best_score(setup, tmp_path):
    detector = build_detector(tmp_path / "tiny.onnx", detector_rows([(10, 10, 20, 20), (400, 10, 20, 20)], [0.9, 0.8], [0, 0]))
    pipeline = Pipeline.open(detector, setup.recognizer, setup.bank, load_catalog(setup.catalog))
    result = pipeline.analyze(setup.frame)
    assert result.cards == () and result.too_small == 2 and result.best_score == pytest.approx(0.9)


def test_the_best_score_changes_nothing_that_was_reported_before(setup, pipeline):
    """The same detector shown to the pipeline through ``detect`` alone, as a stand-in has it, gives
    the same cards and the same JSON; it only lacks the best score."""

    class DetectOnly:
        def __init__(self, detector) -> None:
            self.detect = detector.detect

    plain = Pipeline(DetectOnly(pipeline.detector), pipeline.recognizer).analyze(setup.frame, name="table.png", source="memory", t=1.5)
    assert plain.best_score is None
    assert plain.to_json() == result_json(pipeline, setup)
    assert plain.cards == pipeline.analyze(setup.frame, name="table.png", source="memory", t=1.5).cards
    found, best = pipeline.detector.look(setup.frame)
    assert found == pipeline.detector.detect(setup.frame) and best == pytest.approx(0.95)


def result_json(pipeline, setup) -> dict:
    return pipeline.analyze(setup.frame, name="table.png", source="memory", t=1.5).to_json()


def test_the_minimum_crop_size_is_a_setting(setup, pipeline):
    lenient = Pipeline(pipeline.detector, pipeline.recognizer, min_crop_px=16)
    assert len(lenient.analyze(setup.frame).cards) == 8        # the 20 x 20 box is now kept
    strict = Pipeline(pipeline.detector, pipeline.recognizer, min_crop_px=200)
    assert [list(card.box) for card in strict.analyze(setup.frame).cards] == EXPECTED_BOXES[:6]   # the clipped card is 120 wide
