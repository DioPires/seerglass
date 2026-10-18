# SPDX-License-Identifier: Apache-2.0
"""A toy calibration set: labeled frames on which the accept rule's two thresholds matter.

``cardvision calibrate`` needs rows whose outcome changes with ``tau`` and ``delta``. The toy
scene of the other tests does not have them, so this module stages some. Every frame is a
table with eight places, and the toy detector reports the same eight face boxes for each
(``test_infer_toy_models.build_detector``). What lies in a place decides what the toy
recognizer makes of it:

==========  =====================================================================================
``card``    a card, upright: its own bank row scores 1, by a margin that differs from card to card
``turned``  a card lying sideways or upside down. Its upright look scores well against *another*
            card, so whether the turned looks are tried, and so which name is shown, depends on
            the thresholds
``faded``   a card washed out towards the table's color: its own row scores less
``new``     a card the bank has no row for: the best row is another card's
``back``    a card back that the detector took for a face
``token``   a plain gray token that the detector took for a face
``mat``     a patch of a red playmat that the detector took for a face
==========  =====================================================================================

The labels say what is there: a named face, a card back or no card. Six frames, in three
recording groups, have the role ``train``; two more, in two other groups, have the role
``evaluation``. Nothing here has any skill, and the figures mean nothing; they only move when
the thresholds do.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from cardvision.core.images import load_rgb
from cardvision.data.extract import frame_file_name
from cardvision.data.manifest import write_manifest
from cardvision.data.schema import Box, Row
from test_infer_toy_models import SCENE_SIZE, TABLE_COLOR, build_bank, build_detector, build_recognizer, detector_rows, save_image

W, H = SCENE_SIZE
PLACES = [(x, y, 240, 336) for y in (40, 420) for x in (40, 320, 600, 880)]
# Flat patches that are not card faces, and that the toy recognizer takes for a card all the same.
PATCHES = {"back": (35, 45, 75), "token": (70, 70, 70), "mat": (150, 40, 40)}
# How far a faded card is washed out towards the table's color.
FADE = 0.3
# The card the bank has no row for.
NEW_CARD = "TOY-012"
# The rule the toy set starts from: the values of the published rule.
START = {"tau": 0.62, "delta": 0.05, "rotationRetry": True}

# One list per frame: what lies in each of the eight places.
FRAMES = [
    [("card", "TOY-001"), ("card", "TOY-009"), ("turned", "TOY-017", 1), ("faded", "TOY-003A"),
     ("new",), ("back",), ("token",), ("card", "TOY-003")],
    [("card", "TOY-003A"), ("turned", "TOY-003", 2), ("card", "TOY-004"), ("faded", "TOY-003A"),
     ("mat",), ("card", "TOY-016"), ("back",), ("card", "TOY-005")],
    [("turned", "TOY-017", 3), ("card", "TOY-011"), ("faded", "TOY-003A"), ("card", "TOY-014"),
     ("new",), ("token",), ("card", "TOY-P01"), ("back",)],
    [("card", "TOY-008"), ("faded", "TOY-003A"), ("card", "TOY-009"), ("turned", "TOY-003", 2),
     ("back",), ("card", "TOY-013"), ("mat",), ("card", "TOY-016")],
    [("card", "TOY-014"), ("card", "TOY-001"), ("turned", "TOY-017", 1), ("new",),
     ("token",), ("card", "TOY-005"), ("faded", "TOY-003A"), ("card", "TOY-004")],
    [("card", "TOY-011"), ("card", "TOY-P01"), ("back",), ("card", "TOY-003A"),
     ("faded", "TOY-003A"), ("mat",), ("token",), ("card", "TOY-003")],
]
# The recording group of each frame, and the kind of footage it stands for.
GROUPS = ["g-alpha", "g-alpha", "g-beta", "g-beta", "g-gamma", "g-gamma"]
SOURCE_TYPES = ["digital/toy-client", "digital/toy-client", "physical/webcam", "physical/webcam",
                "physical/webcam", "physical/webcam"]

# Two more frames, of two other recording groups, with the role ``evaluation``: the rows a
# figure is reported on, which a calibration must leave alone.
HELD_OUT_FRAMES = [
    [("card", "TOY-005"), ("turned", "TOY-017", 1), ("faded", "TOY-003A"), ("card", "TOY-016"),
     ("new",), ("back",), ("token",), ("card", "TOY-013")],
    [("card", "TOY-003"), ("card", "TOY-009"), ("turned", "TOY-003", 2), ("mat",),
     ("card", "TOY-001"), ("faded", "TOY-003A"), ("back",), ("card", "TOY-014")],
]
HELD_OUT_GROUPS = ["g-delta", "g-epsilon"]
HELD_OUT_SOURCE_TYPES = ["digital/toy-client", "physical/webcam"]


def _pixels(art_dir: Path, what: tuple) -> np.ndarray | None:
    """The 240 x 336 picture that lies in a place."""
    kind = what[0]
    if kind in PATCHES:
        return np.full((336, 240, 3), PATCHES[kind], dtype=np.uint8)
    card = load_rgb(Path(art_dir) / f"{NEW_CARD if kind == 'new' else what[1]}.png")
    if kind == "turned":
        card = np.rot90(card, what[2])
    if kind == "faded":
        card = (card * (1 - FADE) + np.array(TABLE_COLOR) * FADE).astype(np.uint8)
    assert card.shape[:2] == (336, 240), f"{what} does not fill a place"
    return card


def draw_frame(art_dir: Path, places: list[tuple]) -> np.ndarray:
    """One frame of the set: the table with a picture in each place."""
    frame = np.empty((H, W, 3), dtype=np.uint8)
    frame[:] = TABLE_COLOR
    for (x, y, width, height), what in zip(PLACES, places):
        frame[y:y + height, x:x + width] = _pixels(art_dir, what)
    return frame


def _label(place: tuple[int, int, int, int], what: tuple, number: int) -> Box:
    x, y, width, height = place
    quad = ((x / W, y / H), ((x + width) / W, y / H), ((x + width) / W, (y + height) / H), (x / W, (y + height) / H))
    kind = what[0]
    if kind in PATCHES:
        return Box(kind="back" if kind == "back" else "not-card", quad=quad, id=f"p{number}")
    return Box(kind="face", quad=quad, card_id=NEW_CARD if kind == "new" else what[1], identity="named", readable="yes",
               id=f"p{number}")


def _row(video: str, t: float, places: list[tuple], group: str, source_type: str, role: str) -> Row:
    return Row(platform="youtube", video_id=video, t=t, frame_width=W, frame_height=H, source_type=source_type,
               group=group, role=role, trained_on=False, credit="tests", completeness="exhaustive-boxes",
               boxes=tuple(_label(place, what, number) for number, (place, what) in enumerate(zip(PLACES, places), start=1)))


def rows() -> list[tuple[Row, list[tuple]]]:
    """The set's manifest rows, each with what its frame shows: six rows with the role
    ``train``, one exhaustively boxed row per frame, and then the two held-out rows."""
    train = [(_row(f"CALIBRATE{index:02d}", 10.0 + index, places, GROUPS[index], SOURCE_TYPES[index], "train"), places)
             for index, places in enumerate(FRAMES)]
    held_out = [(_row(f"HELDOUT{index:04d}", 30.0 + index, places, HELD_OUT_GROUPS[index], HELD_OUT_SOURCE_TYPES[index],
                      "evaluation"), places)
                for index, places in enumerate(HELD_OUT_FRAMES)]
    return [*train, *held_out]


class Calibration:
    """The toy calibration set in a folder: stand-in models, a manifest and its frames.

    ``folder`` gets ``models/`` (detector, recognizer, a bank without ``NEW_CARD``, and the
    accept rule to start from), ``labels/toy-calibration.manifest.jsonl`` and ``frames/``.
    ``rows`` are the manifest's rows: the first six have the role ``train``, the last two
    ``evaluation``. ``extra_boxes`` are further face boxes for the detector to report in every
    frame, on top of the eight places: boxes that overlap a place, to put the hover rule's
    stacking to work.
    """

    def __init__(self, folder: Path, catalog_path: Path, art_dir: Path,
                 extra_boxes: Sequence[tuple[int, int, int, int]] = ()) -> None:
        folder = Path(folder)
        models = folder / "models"
        models.mkdir(parents=True)
        self.catalog = Path(catalog_path)
        self.recognizer = build_recognizer(models / "recognizer.onnx")
        self.bank = build_bank(models / "bank.json", self.recognizer, self.catalog, art_dir, leave_out=[NEW_CARD])
        boxes = [*PLACES, *extra_boxes]
        self.detector = build_detector(models / "detector.onnx",
                                       detector_rows(boxes, [0.95 - 0.05 * index for index in range(len(boxes))],
                                                     [0] * len(boxes)))
        self.accept = models / "accept.json"
        self.accept.write_text(json.dumps(START), encoding="utf-8")
        drawn = rows()
        self.rows = [row for row, _ in drawn]
        self.manifest = folder / "labels" / "toy-calibration.manifest.jsonl"
        self.manifest.parent.mkdir()
        write_manifest(self.manifest, self.rows)
        self.frames = folder / "frames"
        self.frames.mkdir()
        for row, places in drawn:
            save_image(self.frames / frame_file_name(row, ".png"), draw_frame(art_dir, places))

    def models(self, accept: Path | None = None) -> list[str]:
        """The model arguments of ``cardvision calibrate`` and ``cardvision evaluate``."""
        return ["--detector", str(self.detector), "--recognizer", str(self.recognizer), "--bank", str(self.bank),
                "--catalog", str(self.catalog), "--accept", str(accept or self.accept)]

    def arguments(self, *extra: str, accept: Path | None = None) -> list[str]:
        return ["--manifest", str(self.manifest), "--frames", str(self.frames), *self.models(accept), *extra]
