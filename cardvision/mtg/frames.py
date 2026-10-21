# SPDX-License-Identifier: Apache-2.0
"""Frames to label, cut from a video on its platform the way ``cardvision fetch`` cuts them again.

A row the labelling tool exports names its frame by the video and the time ``t`` (and, for a
frame saved as PNG, by its fingerprint ``frame_dhash``). Whoever fetches the row again downloads a
stretch of the video around ``t`` and takes the frame on screen at ``t``
(``cardvision.data.fetch``, ``cardvision.data.extract``). The frame that is labelled must be that
same frame, so it is cut here with the same pieces: ``plan_sections`` and the downloader for the
stretch, ``VideoReader.frame_at`` for the frame. It is saved as PNG, which the export fingerprints.

Seeking in the platform's stream with ffmpeg alone is not the same thing. On a Twitch broadcast
(an HLS playlist) ``ffmpeg -ss T -i PLAYLIST`` gave a frame 6 s away from the one at ``T`` on
2026-10-21, while the stretch download gave the frame at ``T`` (checked against the playlist's
segment times).

Each frame is ``t<seconds>_<milliseconds>.png`` in the output folder, and ``times.jsonl`` there
lists every frame cut into it (``name``, ``t``, ``frameTime``: the time of the frame shown),
sorted by ``t``, for ``labeler/make_sample.py --times``. No video file is kept.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path

from PIL import Image

from cardvision.data.extract import VideoReader
from cardvision.data.fetch import (
    VIDEO_URLS,
    Downloader,
    download_video,
    plan_sections,
    sections_supported,
)

TIMES_NAME = "times.jsonl"
PLATFORMS = tuple(sorted(VIDEO_URLS))


def frame_name(t: float) -> str:
    """The file name of the frame at ``t`` seconds: ``t33588_261.png`` for 33588.261."""
    millis = round(t * 1000)
    return f"t{millis // 1000}_{millis % 1000:03d}.png"


def _save_png(rgb, path: Path) -> None:
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".partial-", suffix=".png")
    os.close(handle)
    try:
        Image.fromarray(rgb).save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _write_times(out: Path, entries: Iterable[dict]) -> Path:
    path = out / TIMES_NAME
    known: dict[str, dict] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                known[row["name"]] = row
    for entry in entries:
        known[entry["name"]] = entry
    text = "".join(json.dumps(row) + "\n" for row in sorted(known.values(), key=lambda row: row["t"]))
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text, encoding="utf-8")
    partial.replace(path)
    return path


def cut_frames(platform: str, video_id: str, times: Iterable[float], out: str | Path,
               downloader: Downloader | None = None, say: Callable[[str], None] = print) -> list[dict]:
    """Cut the frame on screen at each of ``times`` (seconds) of a video into ``out``.

    Returns one ``{"name", "t", "frameTime"}`` per time, in order of time, as ``times.jsonl``
    records them. Raises ``ValueError`` for a platform the fetcher does not know, and
    ``RuntimeError`` when ffmpeg is missing: without it only the whole video can be downloaded.
    """
    if platform not in VIDEO_URLS:
        raise ValueError(f"unknown platform {platform!r}: one of {', '.join(PLATFORMS)}")
    if downloader is None:
        if not sections_supported():
            raise RuntimeError("cutting frames downloads a stretch of the video around each time, which needs ffmpeg")
        downloader = download_video
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    for start, end, served in plan_sections(times):
        say(f"{platform} {video_id}: downloading {start:g} s to {end:g} s for {len(served)} frame(s)")
        with tempfile.TemporaryDirectory(prefix="cardvision-cut-") as scratch:
            result = downloader(platform, video_id, start, end, Path(scratch))
            with VideoReader(result.path, offset=result.offset) as reader:
                for t in served:
                    frame = reader.frame_at(t)
                    name = frame_name(t)
                    _save_png(frame.rgb, out / name)
                    entries.append({"name": name, "t": t, "frameTime": round(frame.time, 6)})
    path = _write_times(out, entries)
    say(f"{len(entries)} frame(s) in {out}; times in {path.name}")
    return entries
