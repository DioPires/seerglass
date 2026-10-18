# SPDX-License-Identifier: Apache-2.0
"""A recording of your own, from the video file to a score: the route the guides give for a
new game, run on the toy game.

``cardvision infer`` writes frames of the recording as PNG files, the labeling tool's
``make_sample.py --local`` lists them as a recording that is on no platform, a stand-in for the
person labels them through the tool's own save path, ``export.py manifest`` writes rows with
the platform ``local`` and each frame's fingerprint, and ``cardvision validate-manifest``,
``fetch --video-file`` and ``evaluate`` take those rows like any other. Nothing touches the
network.

The route is run twice: on a recording of 5 frames a second, and on one of 29.97, the rate at
which a frame's start time, rounded to the millisecond, falls just before the frame.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from cardvision import cli
from cardvision.data import read_manifest
from cardvision.data.dhash import is_dhash
from cardvision.data.extract import read_index
from support_labeling import LABELER, label_toy_scene
from test_infer_toy_models import ToySetup

RECORDING = "kitchen-table"
# The recording's frame rate, and the time of each frame the route takes from it (one every
# five seconds): the middle of the frame's time on screen, to the millisecond.
TIMES = {5.0: (0.1, 5.1, 10.1), 30000 / 1001: (0.017, 5.022, 10.027)}


@pytest.fixture(scope="module")
def setup(tmp_path_factory, toy) -> ToySetup:
    return ToySetup(tmp_path_factory.mktemp("own-models"), *toy)


@pytest.fixture(scope="module", params=sorted(TIMES), ids=["5-frames-a-second", "29.97-frames-a-second"])
def rate(request) -> float:
    return request.param


@pytest.fixture(scope="module")
def times(rate) -> tuple[float, ...]:
    return TIMES[rate]


@pytest.fixture(scope="module")
def recording(setup, rate, tmp_path_factory) -> Path:
    """Fifteen seconds of the toy scene as a video file: the recording of one's own."""
    cv2 = pytest.importorskip("cv2")
    path = tmp_path_factory.mktemp("recording") / "phone-clip.mp4"
    height, width = setup.frame.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), rate, (width, height))
    for _ in range(round(15 * rate)):
        writer.write(np.ascontiguousarray(setup.frame[:, :, ::-1]))
    writer.release()
    return path


def labeler_command(name: str, arguments: list[str]) -> int:
    """Run one of the labeling tool's scripts in this process, imported as the tool imports it."""
    sys.path.insert(0, str(LABELER))
    try:
        return __import__(name).main(arguments)
    finally:
        sys.path.remove(str(LABELER))


@pytest.fixture(scope="module")
def manifest(setup, recording, tmp_path_factory) -> Path:
    """The manifest the route writes for the recording, in a folder of its own."""
    work = tmp_path_factory.mktemp("own")
    assert cli.main(["infer", str(recording), "--every", "5", "--frame-format", "png", *setup.arguments(),
                     "--frames", str(work / "frames" / RECORDING), "--prefill-out", str(work / "times.jsonl")]) == 0
    assert labeler_command("make_sample", ["--out", str(work / "sample.json"), "--asset", f"{RECORDING}={work / 'frames' / RECORDING}",
                                           "--all", "--times", str(work / "times.jsonl"), "--local"]) == 0
    assert label_toy_scene(work / "sample.json", setup.catalog) == 3
    out = work / "manifest" / "own.manifest.jsonl"
    assert labeler_command("export", ["manifest", "--labels-dir", str(work / "labels"), "--sample", str(work / "sample.json"),
                                      "--credit", "Toy Labeler", "--out", str(out)]) == 0
    return out


def test_the_labeling_tool_writes_rows_of_the_platform_local(manifest, times):
    rows = read_manifest(manifest)
    assert [(row.platform, row.video_id, row.t) for row in rows] == [("local", RECORDING, t) for t in times]
    assert all((row.role, row.reachable, row.reachable_checked, row.source_type) == ("extra", None, None, "physical/phone")
               for row in rows)
    assert all(len(row.boxes) == 7 and sum(box.ignore for box in row.boxes) == 1 for row in rows)
    # The frames were stored as PNG, so each row carries its frame's fingerprint.
    assert all(is_dhash(row.frame_dhash) for row in rows)


def test_the_rows_are_valid_and_cannot_be_offered(manifest, setup, capsys):
    arguments = ["validate-manifest", str(manifest), "--catalog", str(setup.catalog)]
    assert cli.main(arguments) == 0
    assert capsys.readouterr().out == "valid: 3 row(s) in 1 file(s); card ids checked against the catalog\n"
    assert cli.main([*arguments, "--contribution"]) == 1
    assert capsys.readouterr().err.count('[contribution] platform is "local"') == 3


def test_each_row_s_time_is_on_its_own_frame(manifest, recording, rate, times):
    """The frames taken are the first, and the first at or after 5 and 10 seconds. Each row's
    time gives that frame back by the manifest's rule, at either frame rate."""
    from cardvision.data.extract import VideoReader

    with VideoReader(recording) as reader:
        taken = [0, round(5 * reader.rate), round(10 * reader.rate)]
        assert [reader.frame_at(row.t).index for row in read_manifest(manifest)] == taken
    assert taken == [0, round(5 * rate), round(10 * rate)] and list(times) == [round((index + 0.5) / rate, 3) for index in taken]


def test_the_frames_come_from_the_file_given_and_the_rows_are_scored(manifest, setup, recording, times, tmp_path, capsys):
    frames = tmp_path / "frames"
    fetch = ["fetch", str(manifest), "--out", str(frames)]
    assert cli.main(fetch) == 1
    captured = capsys.readouterr()
    assert "0 frame(s) written, 0 already present, 0 unusable, 0 failed, 3 without a video" in captured.out
    assert f"local {RECORDING}: no video file given; pass --video-file {RECORDING}=PATH; 3 row(s) left without a frame" in captured.err
    assert cli.main([*fetch, "--video-file", f"{RECORDING}={recording}"]) == 0
    assert "3 frame(s) written, 0 already present, 0 unusable, 0 failed, 0 without a video" in capsys.readouterr().out
    index = read_index(frames)
    assert [(entry.status, entry.source, entry.file) for entry in index.values()] == [
        ("ok", "local", f"frames/local-{RECORDING}-t{round(t * 1000):010d}.jpg") for t in times]
    assert all(entry.distance == 0 for entry in index.values())      # each frame was found by its content

    out = tmp_path / "results"
    assert cli.main(["evaluate", "hover", "--role", "extra", "--manifest", str(manifest), "--frames", str(frames),
                     *setup.arguments(), "--out", str(out)]) == 0
    report = capsys.readouterr().out
    assert "- role evaluated: `extra`" in report and "- rows used: 3" in report and "- rows unusable (not scored): none" in report
    # The staged toy bank names three of the five whole faces on each of the three frames.
    assert "- readable faces: right 9, wrong 0, nothing 6" in report and "- backs: name shown 0, nothing 3" in report
    assert "### Source type `physical/phone`" in report
    assert "- labels left out of used rows: ignore-region 3" in report
    assert json.loads((out / "evaluate-hover.json").read_text(encoding="utf-8"))["measure"] == "hover"
    assert cli.main(["evaluate", "detector", "--role", "extra", "--manifest", str(manifest), "--frames", str(frames),
                     *setup.arguments()]) == 0
    assert "- frames: 3; face targets 15, back targets 3, ignore regions 3, not-card boxes 0" in capsys.readouterr().out
