# SPDX-License-Identifier: Apache-2.0
"""Builders shared by the detector-trainer tests: toy training data, made at run time.

Everything is generated from the toy game: scenes from ``cardvision.composites`` serve as the
composites stream, more scenes stand in for frames of a labeled video (their labels are
rewritten as a label manifest, with made-up video ids), and card-free scenes are the negatives.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from cardvision.composites import generate, read_labels
from cardvision.core.catalog import load_catalog
from cardvision.data.extract import frame_file_name
from cardvision.data.manifest import write_manifest
from cardvision.data.schema import Box, Row
from cardvision.train_detector.sources import Stream

VIDEO_IDS = ("EXAMPLE0001", "EXAMPLE0002")
EXTRA_FRAMES = 3


def make_composites(toy: tuple[Path, Path], out: Path, n: int, seed: int, split: str = "train") -> Path:
    generate(load_catalog(toy[0]), toy[1], out, n=n, seed=seed, split=split)
    return out


def scene_row(scene: dict[str, Any], video_id: str, t: float, **fields: Any) -> Row:
    """A generated scene's cards as an exhaustively boxed manifest row."""
    width, height = scene["width"], scene["height"]
    boxes = []
    for item in scene["objects"]:
        if item["kind"] not in ("face", "back"):
            continue
        quad = tuple((float(x) / width, float(y) / height) for x, y in item["corners"])
        if item["kind"] == "face":
            boxes.append(Box(kind="face", quad=quad, ignore=bool(item["ignore"]), oriented=True, shape="quad",
                             card_id=item["cardId"], identity="named", readable="yes"))
        else:
            boxes.append(Box(kind="back", quad=quad, ignore=(item.get("visible") or 0) < 0.5))
    values = {"platform": "youtube", "video_id": video_id, "t": t, "frame_width": width, "frame_height": height,
              "source_type": "physical/tournament", "group": "g-toy-01", "role": "train", "trained_on": False,
              "completeness": "exhaustive-boxes", "credit": "Toy Labeler", "boxes": tuple(boxes)}
    values.update(fields)
    return Row(**values)


def make_real(toy: tuple[Path, Path], folder: Path, n: int, seed: int) -> tuple[Path, Path, list[Row]]:
    """``(manifest, frames folder, rows)``: ``n`` generated scenes posing as labeled video frames."""
    scenes_dir = make_composites(toy, folder / "scenes", n, seed, "val")
    frames_dir = folder / "frames"
    frames_dir.mkdir(parents=True)
    rows = []
    for index, scene in enumerate(read_labels(scenes_dir)):
        row = scene_row(scene, VIDEO_IDS[index % len(VIDEO_IDS)], 10.0 + index)
        shutil.copy(scenes_dir / scene["file"], frames_dir / frame_file_name(row))
        rows.append(row)
    manifest = folder / "toy.manifest.jsonl"
    write_manifest(manifest, rows)
    return manifest, frames_dir, rows


def make_negatives(toy: tuple[Path, Path], folder: Path, n: int, seed: int) -> Path:
    """A folder of card-free generated scenes."""
    scenes = make_composites(toy, folder / "probe", n, seed, "probe")
    assert all(not scene["objects"] or all(o["kind"] not in ("face", "back") for o in scene["objects"])
               for scene in read_labels(scenes))
    return scenes / "images"


def make_data(toy: tuple[Path, Path], folder: Path) -> dict[str, Any]:
    """Toy inputs for every built-in stream, and the ``[streams]`` tables that point at them."""
    manifest, frames, rows = make_real(toy, folder / "real", 6, seed=21)
    composites = make_composites(toy, folder / "composites", 8, seed=22)
    negatives = make_negatives(toy, folder / "negatives", 4, seed=23)
    return {
        "rows": rows,
        "streams": {
            "real": {"manifest": str(manifest), "frames": str(frames)},
            "composites": {"folder": str(composites)},
            "negatives": {"folder": str(negatives)},
        },
    }


def extra_stream(config: dict[str, Any]) -> Stream:
    """A plug-in stream, as a config's ``extra_sources`` would name it: a few flat frames with
    one face and one ignore region each. Its settings come from a table of its own."""
    settings = config.get("streams", {}).get("extra", {})
    count = settings.get("frames", EXTRA_FRAMES) if isinstance(settings, dict) else EXTRA_FRAMES
    rng = np.random.default_rng(5)
    frames, targets = [], []
    for index in range(count):
        frame = np.full((180, 320, 3), 40 + 20 * index, np.uint8)
        frame[40:140, 100:172] = rng.integers(0, 256, (100, 72, 3), dtype=np.uint8)
        frames.append(frame)
        targets.append([{"cx": 136 / 320, "cy": 90 / 180, "w": 72 / 320, "h": 100 / 180, "k": 0},
                        {"cx": 0.85, "cy": 0.5, "w": 0.2, "h": 0.6, "k": 2}])
    return Stream("extra", frames, targets, record={"origin": "made in the test"})
