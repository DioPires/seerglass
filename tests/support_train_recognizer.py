# SPDX-License-Identifier: Apache-2.0
"""Toy training data for the recognizer trainer's tests: the toy game's card images, generated
composite scenes, and a made-up label manifest with the frames it describes.

Nothing here needs PyTorch. ``build_toy_data`` writes everything into one folder; ``toy_config``
is a small training config over it.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import toy_game
from cardvision.data import Box, Row, frame_file_name, write_manifest

FRAME_SIZE = (1280, 720)  # width, height
CARD_WIDTH = 110
GROUPS = ("g-toy-01", "g-toy-02", "g-toy-03", "g-toy-04")
HELD_OUT_GROUP = "g-toy-04"
FRAMES_PER_GROUP = 3
# Cards kept out of training in the toy runs.
UNSEEN = ("TOY-005", "TOY-012", "TOY-015")
MARGIN = 0.4


@dataclass(frozen=True)
class ToyData:
    """Where the toy data lies."""

    root: Path
    catalog: Path
    art: Path
    composites: Path
    manifest: Path
    frames: Path


def _background(rng: np.random.Generator) -> np.ndarray:
    width, height = FRAME_SIZE
    base = rng.integers(40, 120, 3)
    noise = rng.integers(-12, 13, (height // 8, width // 8, 3))
    small = np.clip(base + noise, 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(small).resize((width, height), Image.BILINEAR)).copy()


def _fractions(points) -> tuple:
    width, height = FRAME_SIZE
    return tuple((x / width, y / height) for x, y in points)


def _place(frame: np.ndarray, picture: Image.Image, x: int, y: int, on_its_side: bool):
    """Paste a card picture with its top-left at ``(x, y)``; turned a quarter clockwise when
    ``on_its_side``. Returns the quad, clockwise from the card's own top-left."""
    scale = CARD_WIDTH / min(picture.size)
    picture = picture.resize((round(picture.width * scale), round(picture.height * scale)), Image.BILINEAR)
    pixels = np.asarray(picture)
    if on_its_side:
        pixels = np.rot90(pixels, -1)
    h, w = pixels.shape[:2]
    frame[y:y + h, x:x + w] = pixels
    if on_its_side:
        return _fractions([(x + w, y), (x + w, y + h), (x, y + h), (x, y)])
    return _fractions([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])


def build_toy_frames(root: Path, art_dir: Path) -> tuple[Path, Path]:
    """Write a manifest and its frames: four recording groups of three frames, each frame with
    six named faces (every third one lying on its side), one card back and one box that a
    labeler judged not to be a card."""
    frames_dir = root / "frames"
    frames_dir.mkdir(parents=True)
    ids = [card_id for card_id, _, _, _ in toy_game.CARDS]
    pictures = {card_id: Image.open(art_dir / f"{card_id}.png").convert("RGB") for card_id in ids}
    rng = np.random.default_rng(20)
    rows = []
    number = 0
    for group_index, group in enumerate(GROUPS):
        for frame_index in range(FRAMES_PER_GROUP):
            frame = _background(rng)
            boxes = []
            for slot in range(6):
                card_id = ids[(number * 6 + slot * 7 + frame_index) % len(ids)]
                x, y = 60 + slot * 200 + int(rng.integers(0, 20)), 60 + int(rng.integers(0, 30))
                quad = _place(frame, pictures[card_id], x, y, on_its_side=slot % 3 == 2)
                boxes.append(Box(kind="face", quad=quad, oriented=True, shape="rotated-box", card_id=card_id,
                                 identity="named", readable="yes", id=f"b{slot + 1}"))
            back = Image.fromarray(np.full((154, 110, 3), (30, 40, 90), np.uint8))
            boxes.append(Box(kind="back", quad=_place(frame, back, 80, 420, False), id="b7"))
            frame[440:560, 400:500] = (200, 180, 60)
            boxes.append(Box(kind="not-card", quad=_fractions([(400, 440), (500, 440), (500, 560), (400, 560)]), id="b8"))
            row = Row(platform="youtube", video_id=f"TOYVIDEO{group_index:03d}", t=float(10 + frame_index),
                      frame_width=FRAME_SIZE[0], frame_height=FRAME_SIZE[1], source_type="physical", group=group,
                      role="train", trained_on=False, completeness="exhaustive-boxes", credit="Toy Labeler",
                      boxes=tuple(boxes))
            Image.fromarray(frame).save(frames_dir / frame_file_name(row, ".png"))
            rows.append(row)
            number += 1
    manifest = root / "toy.manifest.jsonl"
    write_manifest(manifest, rows)
    return manifest, frames_dir


def build_toy_data(root: Path, scenes: int = 36) -> ToyData:
    """The toy game, ``scenes`` composite scenes of it, and the manifest with its frames.

    About three positive faces per scene pass the trainer's limits on composite crops (nearly
    whole, 48 px or more, near a quarter turn); 36 scenes give the toy runs enough of them,
    of the cards trained on and of the cards kept out."""
    from cardvision.composites import generate
    from cardvision.core.catalog import load_catalog

    root = Path(root)
    catalog_path, art_dir = toy_game.write_toy_game(root / "game")
    composites = root / "composites"
    generate(load_catalog(catalog_path), art_dir, composites, n=scenes, seed=5)
    manifest, frames = build_toy_frames(root, art_dir)
    return ToyData(root=root, catalog=catalog_path, art=art_dir, composites=composites, manifest=manifest,
                   frames=frames)


_SHARED: dict[str, ToyData] = {}


def shared_toy_data(tmp_path_factory) -> ToyData:
    """The toy data, built once for the whole test run."""
    if "data" not in _SHARED:
        _SHARED["data"] = build_toy_data(tmp_path_factory.mktemp("toy-training"))
    return _SHARED["data"]


def toy_document(toy: ToyData, **over) -> dict:
    """A config document (what a TOML file would hold) for a short run on the toy data. A
    keyword replaces a top-level key; a table given as a keyword is merged into the table."""
    doc = {
        "seed": 3,
        "epochs": 2,
        "batch": 16,
        "workers": 0,
        "model": {"init": "random"},
        "optimizer": {"lr": 2e-3, "warmup_epochs": 0},
        "data": {"catalog": str(toy.catalog), "art": str(toy.art), "manifests": [str(toy.manifest)],
                 "frames": str(toy.frames), "composites": str(toy.composites), "margin": MARGIN},
        "background_patches": {"count": 8, "size": 96},
        "validation": {"groups": [HELD_OUT_GROUP], "unseen_cards": list(UNSEEN), "unseen_count": 40, "every": 1},
        "epoch": {"base": 96},
    }
    for key, value in over.items():
        doc[key] = {**doc[key], **value} if isinstance(doc.get(key), dict) and isinstance(value, dict) else copy.deepcopy(value)
    return doc


def toml_text(doc: dict) -> str:
    """A config document as TOML text (tables, lists of tables, and plain values)."""
    def value(item) -> str:
        if isinstance(item, bool):
            return "true" if item else "false"
        if isinstance(item, str):
            return '"' + item.replace("\\", "\\\\").replace('"', '\\"') + '"'
        if isinstance(item, (list, tuple)):
            return "[" + ", ".join(value(part) for part in item) + "]"
        return repr(item)

    lines: list[str] = []

    def table(name: str, content: dict) -> None:
        plain = {k: v for k, v in content.items() if not isinstance(v, dict) and not _is_tables(v)}
        if name and (plain or not content):
            lines.append(f"[{name}]")
        lines.extend(f"{k} = {value(v)}" for k, v in plain.items())
        for k, v in content.items():
            if isinstance(v, dict):
                table(f"{name}.{k}" if name else k, v)
            elif _is_tables(v):
                for item in v:
                    lines.append(f"[[{name}.{k}]]" if name else f"[[{k}]]")
                    lines.extend(f"{a} = {value(b)}" for a, b in item.items())

    table("", doc)
    return "\n".join(lines) + "\n"


def _is_tables(item) -> bool:
    return isinstance(item, list) and bool(item) and all(isinstance(part, dict) for part in item)
