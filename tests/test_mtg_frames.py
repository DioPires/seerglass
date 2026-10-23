# SPDX-License-Identifier: Apache-2.0
"""cut-frames: the frame to label is the frame cardvision fetch takes again. Needs OpenCV."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image
from support_data import TWITCH_ID, counter_frame, read_counter

from cardvision.data import fetch
from cardvision.mtg import frames
from cardvision.mtg.__main__ import main

RATE = 30.0


class StretchDownloader:
    """Writes the asked stretch of a counter video: the frame shown n frame times into the
    whole video shows the number n, and the file starts at ``start``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float, float]] = []

    def __call__(self, platform, video_id, start, end, folder):
        import cv2

        self.calls.append((platform, video_id, start, end))
        first = round(start * RATE)
        writer = cv2.VideoWriter(str(folder / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), RATE, (320, 180))
        for index in range(first, first + round((end - start) * RATE)):
            writer.write(np.ascontiguousarray(counter_frame(index)[:, :, ::-1]))
        writer.release()
        return fetch.Download(path=folder / "video.mp4", offset=start)


@pytest.fixture(autouse=True)
def _needs_video():
    pytest.importorskip("cv2")


def test_a_frame_is_named_by_its_time_to_the_millisecond():
    assert frames.frame_name(33588.261) == "t33588_261.png"
    assert frames.frame_name(158.6375) == "t158_638.png"
    assert frames.frame_name(7.0) == "t7_000.png"


def test_each_frame_is_the_one_on_screen_at_its_time_from_the_stretch_fetch_would_download(tmp_path):
    downloader = StretchDownloader()
    times = [12.51, 13.21, 80.05]
    entries = frames.cut_frames("twitch", TWITCH_ID, times, tmp_path / "out", downloader=downloader, say=lambda _: None)
    # The stretches are fetch's own: 12.51 s and 13.21 s share one, 80.05 s has its own.
    assert [call[2:] for call in downloader.calls] == [(s, e) for s, e, _ in fetch.plan_sections(times)]
    assert len(downloader.calls) == 2
    for entry, t in zip(entries, times):
        shown = read_counter(np.asarray(Image.open(tmp_path / "out" / entry["name"])))
        assert shown == int(t * RATE)  # the last frame that came on screen at or before t
        assert entry["t"] == t and entry["frameTime"] == pytest.approx(shown / RATE, abs=1e-3)


def test_the_times_file_lists_every_frame_cut_into_the_folder_by_time(tmp_path):
    out = tmp_path / "out"
    frames.cut_frames("twitch", TWITCH_ID, [50.0], out, downloader=StretchDownloader(), say=lambda _: None)
    frames.cut_frames("twitch", TWITCH_ID, [20.0, 50.0], out, downloader=StretchDownloader(), say=lambda _: None)
    rows = [json.loads(line) for line in (out / frames.TIMES_NAME).read_text().splitlines()]
    assert [(row["name"], row["t"]) for row in rows] == [("t20_000.png", 20.0), ("t50_000.png", 50.0)]
    assert sorted(path.name for path in out.iterdir()) == ["t20_000.png", "t50_000.png", frames.TIMES_NAME]


def test_an_unknown_platform_is_refused_before_anything_is_downloaded(tmp_path, capsys):
    downloader = StretchDownloader()
    with pytest.raises(ValueError, match="twitch, youtube"):
        frames.cut_frames("vimeo", "1", [1.0], tmp_path, downloader=downloader)
    assert downloader.calls == []
    assert main(["cut-frames", "--platform", "vimeo", "--video", "1", "--t", "1", "--out", str(tmp_path)]) == 2
    assert "unknown platform" in capsys.readouterr().err


def test_without_ffmpeg_nothing_is_downloaded(tmp_path, monkeypatch):
    monkeypatch.setattr(frames, "sections_supported", lambda: False)
    with pytest.raises(RuntimeError, match="needs ffmpeg"):
        frames.cut_frames("youtube", "abcdefghijk", [1.0], tmp_path)
