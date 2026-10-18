# SPDX-License-Identifier: Apache-2.0
"""Builders shared by the label-manifest tests: made-up rows, manifests and a generated video.

Nothing here is real: the video ids are placeholders, the card ids belong to no game, and the
video is drawn frame by frame at run time.
"""

from __future__ import annotations

import copy
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

VIDEO_ID = "EXAMPLE0001"
OTHER_VIDEO_ID = "EXAMPLE0002"
TWITCH_ID = "0000000001"


def box_doc(**fields: Any) -> dict[str, Any]:
    """A valid face box as JSON, with any field replaced. ``None`` removes a field."""
    doc: dict[str, Any] = {
        "kind": "face",
        "ignore": False,
        "quad": [[0.25, 0.25], [0.35, 0.25], [0.35, 0.5], [0.25, 0.5]],
        "oriented": True,
        "shape": "rotated-box",
        "card_id": "SET-001",
        "identity": "named",
        "readable": "yes",
    }
    return _with(doc, fields)


def plain_box_doc(kind: str = "back", **fields: Any) -> dict[str, Any]:
    """A valid box of a kind that carries no identity fields."""
    doc: dict[str, Any] = {
        "kind": kind,
        "ignore": False,
        "quad": [[0.6, 0.6], [0.7, 0.6], [0.7, 0.85], [0.6, 0.85]],
        "oriented": False,
        "shape": "rotated-box",
    }
    return _with(doc, fields)


def point_doc(**fields: Any) -> dict[str, Any]:
    """A valid face point as JSON."""
    doc: dict[str, Any] = {"x": 0.5, "y": 0.5, "kind": "face", "card_id": "SET-001", "identity": "named", "readable": "yes"}
    return _with(doc, fields)


def row_doc(**fields: Any) -> dict[str, Any]:
    """A valid exhaustive-boxes row as JSON, with any field replaced. ``None`` values are
    kept (several fields are nullable); use ``drop`` to remove fields."""
    doc: dict[str, Any] = {
        "manifest_version": 1,
        "platform": "youtube",
        "video_id": VIDEO_ID,
        "t": 12.517,
        "frame_width": 1920,
        "frame_height": 1080,
        "source_type": "physical/tournament",
        "group": "g-test-01",
        "role": "train",
        "trained_on": False,
        "completeness": "exhaustive-boxes",
        "credit": "Example Labeler",
        "reachable": None,
        "reachable_checked": None,
        "boxes": [box_doc()],
    }
    drop = fields.pop("drop", ())
    doc.update(copy.deepcopy(fields))
    for name in drop:
        doc.pop(name, None)
    return doc


def points_row_doc(**fields: Any) -> dict[str, Any]:
    """A valid points row as JSON."""
    base = {"completeness": "points", "points": [point_doc()], "drop": ("boxes",)}
    base.update(fields)
    return row_doc(**base)


def _with(doc: dict[str, Any], fields: dict[str, Any]) -> dict[str, Any]:
    for name, value in fields.items():
        if value is None and name not in ("card_id", "readable"):
            doc.pop(name, None)
        else:
            doc[name] = copy.deepcopy(value)
    return doc


def write_lines(path: Path, docs: list[Any]) -> Path:
    """Write JSON values as a manifest file; a string is written as it is (for broken lines)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join((doc if isinstance(doc, str) else json.dumps(doc)) + "\n" for doc in docs), encoding="utf-8")
    return path


# ------------------------------------------------------------------------- a generated video
COUNTER_BITS = 12


def counter_frame(index: int, width: int = 320, height: int = 180) -> np.ndarray:
    """A frame that shows its own number as a row of black or white blocks (RGB)."""
    frame = np.zeros((height, width, 3), np.uint8)
    block = width // (COUNTER_BITS + 1)
    for bit in range(COUNTER_BITS):
        if (index >> bit) & 1:
            frame[height // 4: 3 * height // 4, bit * block + block // 4: bit * block + 3 * block // 4] = 255
    return frame


def read_counter(rgb: np.ndarray) -> int:
    """The number a frame made by ``counter_frame`` shows, after any compression."""
    height, width = rgb.shape[:2]
    block = width // (COUNTER_BITS + 1)
    value = 0
    for bit in range(COUNTER_BITS):
        center = rgb[3 * height // 8: 5 * height // 8, bit * block + 3 * block // 8: bit * block + 5 * block // 8]
        if center.mean() > 127:
            value |= 1 << bit
    return value


def write_counter_video(path: Path, rate: float = 30.0, frames: int = 150, width: int = 320, height: int = 180) -> Path:
    """Write a short video whose n-th frame (from 0) shows the number n. Needs OpenCV."""
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), rate, (width, height))
    assert writer.isOpened()
    for index in range(frames):
        writer.write(np.ascontiguousarray(counter_frame(index, width, height)[:, :, ::-1]))
    writer.release()
    return path


# --------------------------------------------------------- a video of variable frame rate
def write_timed_video(path: Path, frames: Sequence[np.ndarray], ticks: Sequence[int], tick_rate: int = 60) -> bool:
    """Write a video whose frames do not last equally long, with the ``ffmpeg`` program.

    Frame ``k`` comes on screen ``sum(ticks[:k]) / tick_rate`` seconds in, so it lasts
    ``ticks[k] / tick_rate`` seconds; how long the last frame lasts is not kept. The codec
    follows the suffix: H.264 for ``.mp4``, VP9 for ``.webm``. Two frames that follow each
    other must be different pictures.

    Each frame is handed to ffmpeg once for every tick it lasts, at ``tick_rate`` frames a
    second. ffmpeg drops the repeats and keeps the times of the frames that stay. Returns
    False when this ffmpeg cannot write the file: an encoder it lacks, for instance.
    """
    height, width = frames[0].shape[:2]
    codec = {".mp4": ["-c:v", "libx264"], ".webm": ["-c:v", "libvpx-vp9"]}[path.suffix]
    pictures = b"".join(frame.tobytes() * count for frame, count in zip(frames, ticks))
    for keep_times in (["-fps_mode", "vfr"], ["-vsync", "vfr"]):      # the newer and the older name of one option
        done = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                               "-s", f"{width}x{height}", "-r", str(tick_rate), "-i", "-", "-vf", "mpdecimate", *keep_times,
                               "-pix_fmt", "yuv420p", *codec, str(path)], input=pictures, capture_output=True)
        if done.returncode == 0 and path.is_file():
            return True
    return False
