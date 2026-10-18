# SPDX-License-Identifier: Apache-2.0
"""The detector's input contract: letterbox geometry, padding, channel order, value scale."""

from __future__ import annotations

import numpy as np
import pytest

from cardvision.core.onnx import create_session
from cardvision.detect import (
    INPUT_HEIGHT,
    INPUT_WIDTH,
    PAD_VALUE,
    Detector,
    letterbox_bgr,
    letterbox_ratio,
    letterbox_size,
    to_input_tensor,
)
from helpers import pattern
from test_infer_toy_models import build_probe_detector


def flat(width: int, height: int, color: tuple[int, int, int]) -> np.ndarray:
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:] = color
    return frame


def test_the_canvas_is_640_by_384_padded_with_114():
    assert (INPUT_WIDTH, INPUT_HEIGHT, PAD_VALUE) == (640, 384, 114)


@pytest.mark.parametrize(
    ("frame", "ratio", "size"),
    [
        ((1920, 1080), 1 / 3, (640, 360)),       # 16:9 is limited by the width
        ((1280, 720), 0.5, (640, 360)),
        ((1280, 768), 0.5, (640, 384)),          # the canvas's own shape fills it
        ((640, 384), 1.0, (640, 384)),
        ((300, 900), 384 / 900, (128, 384)),     # a tall frame is limited by the height
        ((1000, 1000), 0.384, (384, 384)),
        ((320, 192), 2.0, (640, 384)),           # a small frame is enlarged
    ],
)
def test_ratio_and_resized_size(frame, ratio, size):
    assert letterbox_ratio(*frame) == pytest.approx(ratio, rel=0, abs=1e-15)
    assert letterbox_size(*frame) == size


def test_resized_side_rounds_half_to_even():
    # 100 x 25: ratio 6.4, height 160.
    assert letterbox_size(100, 25) == (640, 160)
    # 256 x 1: ratio 2.5, width 640, height 2.5 -> 2 (half to even), not 3.
    assert letterbox_ratio(256, 1) == 2.5
    assert letterbox_size(256, 1) == (640, 2)
    # 256 x 3: height 7.5 -> 8.
    assert letterbox_size(256, 3) == (640, 8)


def test_wide_frame_sits_top_left_with_padding_below():
    canvas, ratio = letterbox_bgr(flat(1280, 720, (200, 100, 50)))
    assert canvas.shape == (384, 640, 3) and canvas.dtype == np.uint8
    assert ratio == 0.5
    # Red, green, blue 200, 100, 50 becomes blue, green, red 50, 100, 200.
    assert (canvas[:360] == (50, 100, 200)).all()
    assert (canvas[360:] == 114).all()


def test_tall_frame_sits_top_left_with_padding_to_the_right():
    canvas, ratio = letterbox_bgr(flat(300, 900, (10, 20, 30)))
    assert ratio == 384 / 900
    assert (canvas[:, :128] == (30, 20, 10)).all()
    assert (canvas[:, 128:] == 114).all()


def test_a_frame_of_the_canvas_size_is_only_channel_swapped():
    frame = pattern(384, 640)
    canvas, ratio = letterbox_bgr(frame)
    assert ratio == 1.0
    np.testing.assert_array_equal(canvas, frame[:, :, ::-1])


def test_the_resize_is_pillows_bilinear_filter():
    from PIL import Image

    frame = pattern(720, 1280)
    canvas, _ = letterbox_bgr(frame)
    expected = np.asarray(Image.fromarray(frame).resize((640, 360), Image.BILINEAR))[:, :, ::-1]
    np.testing.assert_array_equal(canvas[:360], expected)


def test_the_input_tensor_is_channel_first_float32_on_the_0_to_255_scale():
    canvas, _ = letterbox_bgr(flat(1280, 720, (200, 100, 50)))
    tensor = to_input_tensor(canvas)
    assert tensor.shape == (1, 3, 384, 640) and tensor.dtype == np.float32
    assert tensor.flags["C_CONTIGUOUS"]
    assert tensor[0, 0, 0, 0] == 50.0 and tensor[0, 1, 0, 0] == 100.0 and tensor[0, 2, 0, 0] == 200.0
    assert tensor[0, :, 383, 0].tolist() == [114.0, 114.0, 114.0]


def test_other_canvas_sizes_and_pad_values_can_be_asked_for():
    canvas, ratio = letterbox_bgr(flat(100, 100, (1, 2, 3)), width=64, height=32, pad=0)
    assert canvas.shape == (32, 64, 3) and ratio == 0.32
    assert (canvas[:, :32] == (3, 2, 1)).all() and (canvas[:, 32:] == 0).all()


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10, 4), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.float32),
        np.zeros((0, 10, 3), dtype=np.uint8),
    ],
)
def test_frames_of_the_wrong_kind_are_refused(frame):
    with pytest.raises(ValueError):
        letterbox_bgr(frame)


def test_a_frame_too_thin_to_occupy_a_pixel_is_refused():
    with pytest.raises(ValueError, match="too thin"):
        letterbox_bgr(np.zeros((1, 2000, 3), dtype=np.uint8))


def test_what_the_model_receives(tmp_path):
    """Observed through a model that reports the mean of each input channel."""
    detector = Detector(create_session(build_probe_detector(tmp_path / "probe.onnx")))
    assert (detector.input_width, detector.input_height, detector.input_name) == (640, 384, "images")

    rows, ratio = detector.raw(flat(1280, 768, (255, 0, 0)))          # pure red, fills the canvas
    assert ratio == 0.5 and rows.shape == (1, 7)
    assert rows[0, :3].tolist() == [0.0, 0.0, 255.0]                  # channel 0 is blue, channel 2 is red

    rows, _ = detector.raw(flat(1280, 384, (0, 0, 0)))                # black, covers the top half only
    np.testing.assert_allclose(rows[0, :3], [57.0, 57.0, 57.0])       # the other half is the 114 padding
