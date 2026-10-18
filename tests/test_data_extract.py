# SPDX-License-Identifier: Apache-2.0
"""The extraction contract: frame times, the shape check, reading a frame from a video file
and the frame folder. The video is generated at run time; tests that read it are skipped when
OpenCV is not installed."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from cardvision.data import extract
from cardvision.data.extract import FrameMissing, FrameNotFound, FrameStore, IndexEntry
from cardvision.data.schema import Row
from support_data import VIDEO_ID, counter_frame, read_counter, write_counter_video


def make_row(t: float = 2.517, width: int = 1920, height: int = 1080, video_id: str = VIDEO_ID) -> Row:
    return Row(platform="youtube", video_id=video_id, t=t, frame_width=width, frame_height=height,
               source_type="physical", group="g-test-01", role="extra", trained_on=False,
               completeness="exhaustive-boxes", credit="Example Labeler", boxes=())


# ------------------------------------------------------------------------------ frame times
def test_the_frame_for_a_time_is_the_last_frame_at_or_before_it():
    assert extract.frame_index(0.0, 30) == 0
    assert extract.frame_index(2.0, 30) == 60          # exactly on a frame
    assert extract.frame_index(2.03, 30) == 60         # still on screen
    assert extract.frame_index(2.034, 30) == 61
    assert extract.frame_index(7.007, 30000 / 1001) == 210    # 210 * 1001 / 30000 s
    assert extract.frame_index(7.006, 30000 / 1001) == 209
    # A rounded rate misplaces a time that sits exactly on a frame, which is why published
    # times sit in the middle of one.
    assert extract.frame_index(7.007, 29.97) == 209
    assert extract.frame_index(extract.mid_frame_time(210, 30000 / 1001), 29.97) == 210
    with pytest.raises(ValueError):
        extract.frame_index(-1, 30)


@pytest.mark.parametrize("rate", [24000 / 1001, 24, 25, 30000 / 1001, 30, 50, 60000 / 1001, 60])
def test_a_time_in_the_middle_of_a_frame_names_that_frame_after_rounding(rate):
    for index in (0, 1, 2, 59, 60, 61, 1234, 100_000, 647_999):
        t = extract.mid_frame_time(index, rate)
        assert round(t, 3) == t
        assert extract.frame_index(t, rate) == index


def test_the_time_of_a_frame_taken_by_the_fixed_rate_sampler():
    # At 30 frames a second the sampler's first frame is number 14, the last before 0.5 s.
    assert extract.sampled_frame_time(1, 30) == pytest.approx(14.5 / 30, abs=1e-3)
    assert extract.frame_index(extract.sampled_frame_time(1, 30), 30) == 14
    assert [extract.frame_index(extract.sampled_frame_time(n, 30), 30) for n in (2, 3, 12)] == [44, 74, 344]
    assert [extract.frame_index(extract.sampled_frame_time(n, 25), 25) for n in (1, 2, 12)] == [12, 37, 287]
    # At 60 the frame at exactly 0.5 s belongs to the next sample.
    assert [extract.frame_index(extract.sampled_frame_time(n, 60), 60) for n in (1, 2)] == [29, 89]
    assert [extract.frame_index(extract.sampled_frame_time(n, 29.97), 29.97) for n in (1, 2, 12)] == [14, 44, 344]
    # A section that began 1800 s into the video.
    assert extract.sampled_frame_time(143, 30, start=1800.0) == pytest.approx(1800 + 142.5 - 0.5 / 30, abs=1e-3)
    with pytest.raises(ValueError):
        extract.sampled_frame_time(0, 30)


def test_the_time_of_a_single_frame_taken_at_a_chosen_second():
    assert extract.frame_index(extract.frame_time_at_or_after(2, 30), 30) == 60
    assert extract.frame_index(extract.frame_time_at_or_after(2, 29.97), 29.97) == 60
    assert extract.frame_index(extract.frame_time_at_or_after(2.5, 25), 25) == 63
    assert extract.frame_index(extract.frame_time_at_or_after(7, 60), 60) == 420


def test_where_the_sampled_stretches_of_a_video_began():
    assert extract.section_starts(45 * 60) == [0.0]
    assert extract.section_starts(120.5) == [0.0]
    assert extract.section_starts(2 * 3600) == [1800.0, 3600.0, 5400.0]
    assert extract.section_starts(45 * 60 + 1) == [675.0, 1350.0, 2025.0]
    assert extract.FORMAT_SELECTOR.startswith("bv[height<=1080][ext=mp4]/bv[height<=1080]/")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs the ffmpeg program")
@pytest.mark.parametrize("rate", [30.0, 29.97, 25.0, 60.0])
def test_the_sampler_and_the_single_grab_behave_as_the_contract_says(tmp_path, rate):
    cv2 = pytest.importorskip("cv2")
    video = write_counter_video(tmp_path / "v.mp4", rate=rate, frames=int(rate * 6))
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(video), "-vf", "fps=1", "-q:v", "3",
                    str(tmp_path / "f%05d.jpg")], check=True)
    sampled = sorted(tmp_path.glob("f*.jpg"))
    assert len(sampled) >= 5
    for number, path in enumerate(sampled[:5], start=1):
        shown = read_counter(cv2.imread(str(path))[:, :, ::-1])
        assert shown == extract.frame_index(extract.sampled_frame_time(number, rate), rate)
    for second in (2, 2.5):
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", str(second), "-i", str(video),
                        "-frames:v", "1", "-q:v", "3", str(tmp_path / "one.jpg")], check=True)
        shown = read_counter(cv2.imread(str(tmp_path / "one.jpg"))[:, :, ::-1])
        assert shown == extract.frame_index(extract.frame_time_at_or_after(second, rate), rate)


# ----------------------------------------------------------------------------- shape check
def test_a_frame_must_have_the_shape_the_labels_were_made_on():
    row = make_row(width=1920, height=1080)
    assert extract.frame_check(row, 1920, 1080) is None
    assert extract.frame_check(row, 1280, 720) is None        # another size, the same shape
    assert extract.frame_check(row, 854, 480) is None         # 0.08 % off
    assert extract.frame_check(row, 1920, 1088) is None       # 0.7 % off
    assert "shapes differ" in extract.frame_check(row, 1920, 1104)   # 2.2 % off
    assert "shapes differ" in extract.frame_check(row, 1440, 1080)   # 4 : 3
    assert "shapes differ" in extract.frame_check(row, 1080, 1920)   # upright
    assert extract.frame_check(row, 1440, 1080, tolerance=0.5) is None
    assert extract.frame_check(row, 0, 0) == "the frame is empty"
    assert extract.RATIO_TOLERANCE == 0.01


def test_frames_are_named_after_their_row():
    assert extract.frame_file_name(make_row(754.517)) == "youtube-EXAMPLE0001-t0000754517.jpg"
    assert extract.frame_file_name(make_row(0.0), ".png") == "youtube-EXAMPLE0001-t0000000000.png"
    assert extract.frame_file_name(make_row(86400.25)) == "youtube-EXAMPLE0001-t0086400250.jpg"


# ------------------------------------------------------------------------- reading a video
@pytest.fixture(scope="module", params=[30.0, 29.97, 25.0])
def counter_video(request, tmp_path_factory):
    pytest.importorskip("cv2")
    rate = request.param
    return write_counter_video(tmp_path_factory.mktemp("video") / "counter.mp4", rate=rate, frames=150), rate


@pytest.mark.parametrize("exact", [False, True])
def test_the_frame_read_is_the_frame_the_contract_names(counter_video, exact):
    path, rate = counter_video
    with extract.VideoReader(path, exact=exact) as reader:
        assert reader.rate == pytest.approx(rate, abs=0.01) and (reader.width, reader.height) == (320, 180)
        for index in (0, 1, 29, 30, 31, 75, 149, 100, 3, 148, 0):   # forwards and backwards
            frame = reader.frame_at(extract.mid_frame_time(index, rate))
            assert read_counter(frame.rgb) == index and frame.index == index
            assert frame.time == pytest.approx(index / rate, abs=1e-3)
            assert frame.rgb.shape == (180, 320, 3) and frame.rgb.dtype == np.uint8


def test_a_time_exactly_on_a_frame_names_that_frame(counter_video):
    path, rate = counter_video
    for index in (0, 30, 60, 90):
        frame = extract.read_frame_at(path, index / rate + 1e-4)
        assert read_counter(frame.rgb) == index
    # Just before a frame's time, the frame before it is still on screen.
    assert read_counter(extract.read_frame_at(path, 60 / rate - 0.002).rgb) == 59


def test_a_time_outside_the_video_is_an_error(counter_video):
    path, rate = counter_video
    last = extract.read_frame_at(path, extract.mid_frame_time(149, rate))
    assert read_counter(last.rgb) == 149
    for exact in (False, True):
        with pytest.raises(FrameNotFound, match="ends before"):
            extract.read_frame_at(path, 150 / rate + 1.0, exact=exact)
        with pytest.raises(FrameNotFound, match="ends before"):
            extract.read_frame_at(path, 3600.0, exact=exact)


def test_a_file_that_holds_part_of_a_video_is_read_with_its_offset(counter_video):
    path, rate = counter_video
    with extract.VideoReader(path, offset=1800.0) as reader:
        frame = reader.frame_at(1800.0 + extract.mid_frame_time(42, rate))
        assert read_counter(frame.rgb) == 42 and frame.time == pytest.approx(1800 + 42 / rate, abs=1e-3)
        with pytest.raises(FrameNotFound, match="before the start"):
            reader.frame_at(1799.0)


def test_a_file_that_is_not_a_video_is_an_error(tmp_path):
    pytest.importorskip("cv2")
    with pytest.raises(FrameNotFound, match="no such video file"):
        extract.read_frame_at(tmp_path / "absent.mp4", 1.0)
    text = tmp_path / "notes.mp4"
    text.write_text("not a video")
    with pytest.raises(FrameNotFound):
        extract.read_frame_at(text, 1.0)


def test_reading_video_without_opencv_says_what_to_install(monkeypatch, tmp_path):
    monkeypatch.setitem(__import__("sys").modules, "cv2", None)
    with pytest.raises(extract.VideoSupportMissing, match=r'pip install -e "\.\[video\]" in the repository folder'):
        extract.VideoReader(tmp_path / "v.mp4")


# ---------------------------------------------------------------------------- frame folder
def test_the_index_keeps_the_last_entry_of_a_row(tmp_path):
    row = make_row()
    assert extract.read_index(tmp_path) == {}
    extract.append_index(tmp_path, IndexEntry("youtube", VIDEO_ID, row.t, "failed", reason="no frame"))
    extract.append_index(tmp_path, IndexEntry("youtube", VIDEO_ID, 9.0, "no-video", reason="none given"))
    done = IndexEntry("youtube", VIDEO_ID, row.t, "ok", file="frames/a.jpg", width=1280, height=720,
                      frame_time=2.5, scale=0.6667, source="local")
    extract.append_index(tmp_path, done)
    index = extract.read_index(tmp_path)
    assert index[row.key] == done and index[("youtube", VIDEO_ID, 9.0)].status == "no-video"
    assert IndexEntry.from_json(done.to_json()) == done and "reason" not in done.to_json()


def test_a_frame_folder_is_read_by_row(tmp_path):
    row, other, absent, unusable = make_row(1.0), make_row(2.0), make_row(3.0), make_row(4.0)
    picture = counter_frame(5, 1280, 720)
    (tmp_path / "frames").mkdir()
    Image.fromarray(picture).save(tmp_path / "frames" / "any-name.png")
    extract.append_index(tmp_path, IndexEntry("youtube", VIDEO_ID, 1.0, "ok", file="frames/any-name.png"))
    extract.append_index(tmp_path, IndexEntry("youtube", VIDEO_ID, 4.0, "unusable", reason="the shapes differ"))
    # A frame without an index entry is found by its name.
    Image.fromarray(picture).save(tmp_path / "frames" / extract.frame_file_name(other, ".png"))

    store = FrameStore(tmp_path)
    assert store.path(row).name == "any-name.png" and np.array_equal(store.load(row), picture)
    assert np.array_equal(store.load(other), picture)
    assert store.has(row) and store.has(other) and not store.has(absent) and not store.has(unusable)
    with pytest.raises(FrameMissing, match="no frame in"):
        store.load(absent)
    with pytest.raises(FrameMissing, match="unusable"):
        store.path(unusable)
    assert [found.t for found, _ in store.frames([row, absent, other, unusable])] == [1.0, 2.0]


def test_a_stored_frame_of_the_wrong_shape_is_not_served(tmp_path):
    row = make_row(1.0)
    Image.fromarray(np.zeros((600, 800, 3), np.uint8)).save(tmp_path / extract.frame_file_name(row, ".png"))
    store = FrameStore(tmp_path)
    assert store.has(row)
    with pytest.raises(FrameMissing, match="shapes differ"):
        store.load(row)
    assert np.array_equal(FrameStore(tmp_path, tolerance=0.5).load(row), np.zeros((600, 800, 3), np.uint8))


def test_an_index_entry_whose_file_is_gone_is_missing(tmp_path):
    row = make_row(1.0)
    extract.append_index(tmp_path, IndexEntry("youtube", VIDEO_ID, 1.0, "ok", file="frames/gone.jpg"))
    with pytest.raises(FrameMissing, match="absent"):
        FrameStore(tmp_path).path(row)
