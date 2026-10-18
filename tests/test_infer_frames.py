# SPDX-License-Identifier: Apache-2.0
"""Reading frames from image files, folders and (with the optional dependency) video."""

from __future__ import annotations

import sys

import numpy as np
import pytest
from PIL import Image

from cardvision.infer import Frame, VideoSupportMissing, expand_inputs, is_video, read_frames, read_video
from helpers import pattern


def save(path, rgb):
    Image.fromarray(rgb).save(path)
    return path


def test_video_is_told_from_images_by_suffix():
    assert is_video("clip.mp4") and is_video("CLIP.MKV") and is_video("a/b/c.webm")
    assert not is_video("frame.png") and not is_video("frame.JPG") and not is_video("frame")


def test_an_image_is_one_frame(tmp_path):
    path = save(tmp_path / "frame.png", pattern(40, 60))
    (frame,) = read_frames([path])
    assert isinstance(frame, Frame)
    assert (frame.name, frame.source, frame.t) == ("frame.png", str(path), 0.0)
    np.testing.assert_array_equal(frame.rgb, pattern(40, 60))


def test_a_folder_is_its_images_in_name_order(tmp_path):
    for name in ("b.png", "a.jpg", "c.PNG"):
        save(tmp_path / name, pattern(8, 8))
    (tmp_path / "notes.txt").write_text("not an image")
    (tmp_path / "sub").mkdir()
    assert [path.name for path in expand_inputs([tmp_path])] == ["a.jpg", "b.png", "c.PNG"]
    assert [frame.name for frame in read_frames([tmp_path])] == ["a.jpg", "b.png", "c.PNG"]


def test_inputs_keep_the_order_given(tmp_path):
    first, second = save(tmp_path / "z.png", pattern(8, 8)), save(tmp_path / "a.png", pattern(8, 8))
    assert [frame.name for frame in read_frames([first, second])] == ["z.png", "a.png"]


def test_missing_and_empty_inputs_are_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="no such file or folder"):
        expand_inputs([tmp_path / "missing.png"])
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="holds no image files"):
        expand_inputs([tmp_path / "empty"])


def test_two_frames_may_not_share_a_name(tmp_path):
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    a, b = save(tmp_path / "one" / "frame.png", pattern(8, 8)), save(tmp_path / "two" / "frame.png", pattern(8, 8))
    with pytest.raises(ValueError, match="would both produce a frame named frame.png"):
        list(read_frames([a, b]))


def test_video_without_opencv_says_how_to_get_it(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)       # makes "import cv2" fail
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"")
    with pytest.raises(VideoSupportMissing, match=r'python -m pip install -e "\.\[video\]" in the repository folder'):
        list(read_frames([clip]))


def test_the_sampling_interval_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="must be positive"):
        list(read_video(tmp_path / "clip.mp4", every=0))


# ----------------------------------------------------------------------------- with OpenCV


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """Three seconds of generated video at 10 frames a second; each frame is one flat gray
    level, 8 times its index, so that a decoded frame tells which one it is."""
    cv2 = pytest.importorskip("cv2")
    path = tmp_path_factory.mktemp("video") / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 64))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write the test video")
    for index in range(30):
        writer.write(np.full((64, 96, 3), index * 8, dtype=np.uint8))
    writer.release()
    return path


def frame_index(frame: Frame) -> int:
    return round(float(frame.rgb.mean()) / 8)


def test_video_is_sampled_once_a_second_by_default(clip):
    frames = list(read_video(clip))
    assert [frame.name for frame in frames] == ["clip-f00001.jpg", "clip-f00002.jpg", "clip-f00003.jpg"]
    assert [frame.t for frame in frames] == [0.05, 1.05, 2.05]          # the middle of each frame's time on screen
    assert [frame_index(frame) for frame in frames] == [0, 10, 20]
    assert all(frame.rgb.shape == (64, 96, 3) and frame.rgb.dtype == np.uint8 for frame in frames)
    assert all(frame.source == str(clip) for frame in frames)


def test_video_sampling_interval(clip):
    frames = list(read_video(clip, every=0.5))
    assert [frame.t for frame in frames] == [0.05, 0.55, 1.05, 1.55, 2.05, 2.55]
    assert [frame_index(frame) for frame in frames] == [0, 5, 10, 15, 20, 25]
    assert len(list(read_video(clip, every=100))) == 1                 # the first frame is always taken
    assert len(list(read_video(clip, every=0.01))) == 30               # never more than the video has


# The frame rates that recordings and broadcasts commonly have, by name and as the exact rate.
FRAME_RATES = {"23.976": 24000 / 1001, "24": 24.0, "25": 25.0, "29.97": 30000 / 1001, "30": 30.0, "50": 50.0,
               "59.94": 60000 / 1001, "60": 60.0}


@pytest.mark.parametrize("rate", sorted(FRAME_RATES, key=float))
def test_a_frame_s_time_finds_the_same_frame_again(rate, tmp_path):
    """A row made from a frame of ``read_video`` carries the frame's ``t``. Read back by the
    manifest's rule (``VideoReader.frame_at``), that time gives the very frame it came from:
    every one of the first 300 frames, at each common frame rate."""
    pytest.importorskip("cv2")
    from cardvision.data.extract import VideoReader
    from support_data import read_counter, write_counter_video

    path = write_counter_video(tmp_path / "counter.mp4", rate=FRAME_RATES[rate], frames=300)
    frames = list(read_video(path, every=0.001))                         # every frame of the file
    assert [read_counter(frame.rgb) for frame in frames] == list(range(300))
    assert all(round(frame.t, 3) == frame.t for frame in frames) and [frame.t for frame in frames] == sorted({frame.t for frame in frames})
    with VideoReader(path) as reader:
        found = [reader.frame_at(frame.t) for frame in frames]
        assert [frame.index for frame in found] == list(range(300))
        assert [read_counter(frame.rgb) for frame in found] == list(range(300))
        # The time lies inside the frame's own time on screen: after its timestamp, before the next one's.
        assert all(before.time < frame.t < after.time for before, frame, after in zip(found, frames, found[1:]))
        # One frame a second, as the command samples by default, and counting from the start of the file.
        sampled = list(read_video(path))
        assert [read_counter(frame.rgb) for frame in sampled] == [reader.frame_at(frame.t).index for frame in sampled]
    with VideoReader(path, exact=True) as counting:
        assert [counting.frame_at(frame.t).index for frame in frames[::7]] == list(range(0, 300, 7))


def test_a_frame_s_own_timestamp_rounded_is_not_enough(tmp_path):
    """Why ``t`` is not the frame's start: at 29.97 frames a second the frames taken once a
    second are numbers 30, 60 and 90, which start at 1.001001, 2.002002 and 3.003003 seconds.
    Rounded to the millisecond those times fall just before the frames, on numbers 29, 59 and 89."""
    pytest.importorskip("cv2")
    from cardvision.data.extract import VideoReader, mid_frame_time
    from support_data import read_counter, write_counter_video

    path = write_counter_video(tmp_path / "counter.mp4", rate=30000 / 1001, frames=100)
    sampled = list(read_video(path))
    assert [read_counter(frame.rgb) for frame in sampled] == [0, 30, 60, 90]
    with VideoReader(path) as reader:
        rate = reader.rate
        assert [reader.frame_at(round(number / rate, 3)).index for number in (30, 60, 90)] == [29, 59, 89]
        assert [reader.frame_at(frame.t).index for frame in sampled] == [0, 30, 60, 90]
    # On a file of constant frame rate the time is the one published rows carry.
    assert [frame.t for frame in sampled] == [mid_frame_time(number, rate) for number in (0, 30, 60, 90)]


def test_the_time_on_screen_of_a_frame():
    from cardvision.data.extract import frame_stamp, mid_frame_time, on_screen_time

    assert on_screen_time(1.0, 1.04, 25.0) == 1.02 and on_screen_time(0.0, 0.2, 5.0) == 0.1
    assert on_screen_time(1.0, None, 25.0) == 1.02                      # the last frame: half a frame's time on
    assert on_screen_time(1.0, 1.0, 25.0) == 1.02                       # a next timestamp that is no later tells nothing
    # Frames that are not evenly spaced: the time stays between this frame and the next.
    assert on_screen_time(2.0, 2.5, 30.0) == 2.25 and on_screen_time(2.5, 2.51, 30.0) == 2.505
    rate = 30000 / 1001
    assert all(on_screen_time(n / rate, (n + 1) / rate, rate) == mid_frame_time(n, rate) for n in range(2000))
    # A backend that gives no timestamps reports 0 for every frame: the count stands in.
    assert frame_stamp(0.0, 0, 25.0) == 0.0 and frame_stamp(0.0, 50, 25.0) == 2.0 and frame_stamp(2.002, 50, 25.0) == 2.002
    assert frame_stamp(float("nan"), 50, 25.0) == 2.0


def test_video_frames_are_rgb(tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "red.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (64, 64))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write the test video")
    blue_green_red = np.zeros((64, 64, 3), dtype=np.uint8)
    blue_green_red[:, :, 2] = 250                                       # OpenCV's order: red is last
    for _ in range(3):
        writer.write(blue_green_red)
    writer.release()
    (frame,) = read_video(path, every=10)
    red, green, blue = (float(frame.rgb[:, :, channel].mean()) for channel in range(3))
    assert red > 200 and green < 40 and blue < 40


def test_a_file_that_is_not_a_video(tmp_path):
    pytest.importorskip("cv2")
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"this is not a video")
    with pytest.raises(ValueError, match="cannot be opened as a video|no frame could be read|frame rate"):
        list(read_video(path))


def test_images_and_video_together(clip, tmp_path):
    image = save(tmp_path / "still.png", pattern(8, 8))
    names = [frame.name for frame in read_frames([image, clip], every=2.0)]
    assert names == ["still.png", "clip-f00001.jpg", "clip-f00002.jpg"]
