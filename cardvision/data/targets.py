# SPDX-License-Identifier: Apache-2.0
"""From manifest rows to what the trainers consume: detector targets and recognizer crops.

**Detector targets.** Every box of an exhaustively boxed frame becomes a face, a back, an
ignore region or nothing (``classify_box``):

* a box marked ``ignore`` is an ignore region, whatever its kind;
* a back is a back;
* an ``unreadable`` box is an ignore region;
* a ``not-card`` box is nothing: a person judged it not to be a card, so it is background;
* a face is a face, unless its outline has the wrong proportions for a whole card (the aspect
  filter), in which case it is an ignore region. Two cases are exempt and stay faces: a card
  labeled as showing only half (``attributes.visible_share == "half"``), and a card in the
  hand row of a digital client that draws its hand cut off (the hand rule).

The target box is the upright box around the four corners, clipped to the frame; a box left
with no area is dropped.

**Recognizer crops.** A face becomes a training crop when it is named, not marked ignore, not
labeled unreadable from the frame alone, and passes the same aspect filter with the same two
exemptions (``crop_exclusion``). The crop is the upright box around the corners, widened by a
margin, cut from the frame the detector would analyze, and turned so that the card's top is
up to the nearest quarter turn; ``inner`` says where the card's box lies inside it, in whole
pixels.

Backs and ``not-card`` boxes give negative crops: pictures the recognizer should name nothing
for (``negative_crops``).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cardvision.core.constants import DETECTOR_BACK_CLASS, DETECTOR_FACE_CLASS, DETECTOR_MIN_CROP_PX
from cardvision.data import geometry
from cardvision.data.extract import FrameMissing, FrameStore, frame_file_name
from cardvision.data.schema import Box, Row
from cardvision.data.sources import DEFAULT_HAND_ROW_MIN_CENTER_Y, SourceConfig

FACE = "face"
BACK = "back"
IGNORE = "ignore"

# The number the detector's training code uses for each target. Faces and backs are the
# detector's two classes; an ignore region is not a class, only a marker for the loss.
DETECTOR_IGNORE_CODE = 2
DETECTOR_CODES = {FACE: DETECTOR_FACE_CLASS, BACK: DETECTOR_BACK_CLASS, IGNORE: DETECTOR_IGNORE_CODE}

DEFAULT_ASPECT_RANGE = (0.55, 0.90)


@dataclass(frozen=True)
class TargetRules:
    """The settings of the label conversions.

    Attributes:
        aspect_range: a face whose outline, with the turn taken out, has a shorter side over
            longer side outside this range is not treated as a whole card. ``None`` turns the
            filter off.
        keep_half_cards: a face labeled ``visible_share: half`` is exempt from the filter.
        hand_row_source_types: source types whose client draws the hand cut off at the bottom
            of the screen. A face from one of them that is squarer than the range allows and
            sits in that row is exempt from the filter.
        hand_row_min_center_y: the hand row starts this fraction of the way down the frame,
            measured at the center of the box.
    """

    aspect_range: tuple[float, float] | None = DEFAULT_ASPECT_RANGE
    keep_half_cards: bool = True
    hand_row_source_types: Collection[str] = ()
    hand_row_min_center_y: float = DEFAULT_HAND_ROW_MIN_CENTER_Y

    @classmethod
    def for_sources(cls, config: SourceConfig, **settings) -> TargetRules:
        """The rules with the hand-row setting of a game's sources file."""
        return cls(hand_row_source_types=tuple(config.hand_row_source_types),
                   hand_row_min_center_y=config.hand_row_min_center_y, **settings)


def face_exemption(row: Row, box: Box, rules: TargetRules = TargetRules()) -> tuple[bool, str | None, float]:
    """Whether a face passes the aspect filter, and how.

    Returns ``(passes, how, aspect)``. ``how`` is ``None`` for a face inside the range,
    ``"half"`` for one kept by its ``visible_share`` label, ``"hand-row"`` for one kept by the
    hand rule, and ``"aspect"`` for one the filter rejects. The aspect is measured in pixels
    of the frame the labels were made on.
    """
    corners = box.pixel_quad(row.frame_width, row.frame_height)
    aspect = geometry.side_aspect(corners)
    if rules.aspect_range is None:
        return True, None, aspect
    low, high = rules.aspect_range
    if low <= aspect <= high:
        return True, None, aspect
    if rules.keep_half_cards and box.attributes.get("visible_share") == "half":
        return True, "half", aspect
    center_y = float(np.mean([y for _, y in box.quad]))
    if row.source_type in rules.hand_row_source_types and aspect > high and center_y >= rules.hand_row_min_center_y:
        return True, "hand-row", aspect
    return False, "aspect", aspect


def classify_box(row: Row, box: Box, rules: TargetRules = TargetRules()) -> tuple[str | None, str]:
    """What a box is to the detector: ``(target, reason)``.

    ``target`` is ``"face"``, ``"back"``, ``"ignore"`` or ``None`` (no target: background).
    ``reason`` says which rule decided: ``ignore``, ``back``, ``unreadable``, ``not-card``,
    ``face``, ``face-half``, ``face-hand-row`` or ``aspect``.
    """
    if box.ignore:
        return IGNORE, "ignore"
    if box.kind == "back":
        return BACK, "back"
    if box.kind == "unreadable":
        return IGNORE, "unreadable"
    if box.kind == "not-card":
        return None, "not-card"
    passes, how, _ = face_exemption(row, box, rules)
    if not passes:
        return IGNORE, "aspect"
    return FACE, "face" if how is None else f"face-{how}"


@dataclass(frozen=True)
class DetectorTarget:
    """One target of a frame, as fractions of the frame.

    Attributes:
        cx, cy: the center of the box.
        w, h: its width and height.
        kind: ``face``, ``back`` or ``ignore``.
        reason: the rule that decided the kind (see ``classify_box``).
        box_index: the position of the source box in the row's ``boxes``.
        origin: the source box's ``origin``, if it has one.
    """

    cx: float
    cy: float
    w: float
    h: float
    kind: str
    reason: str
    box_index: int
    origin: str | None = None

    def as_dict(self, codes: dict[str, int] = DETECTOR_CODES) -> dict[str, float | int]:
        """``{"cx", "cy", "w", "h", "k"}``, with the kind as the trainer's number."""
        return {"cx": self.cx, "cy": self.cy, "w": self.w, "h": self.h, "k": codes[self.kind]}


def _require_boxes(row: Row) -> tuple[Box, ...]:
    if row.completeness != "exhaustive-boxes" or row.boxes is None:
        raise ValueError(
            f"{row.platform} {row.video_id} t={row.t}: the row is not boxed exhaustively "
            f"(completeness {row.completeness}); an unboxed card would be learned as background"
        )
    return row.boxes


def detector_targets(row: Row, rules: TargetRules = TargetRules()) -> list[DetectorTarget]:
    """The detector's targets for one exhaustively boxed row, in the order of its boxes.

    Raises ``ValueError`` for a row that is not ``exhaustive-boxes``.
    """
    targets = []
    for index, box in enumerate(_require_boxes(row)):
        kind, reason = classify_box(row, box, rules)
        if kind is None:
            continue
        # Fractions of the frame: the clip is to the unit square.
        x, y, width, height = geometry.bounding_box(box.quad, 1.0, 1.0)
        if width <= 0 or height <= 0:
            continue
        targets.append(DetectorTarget(cx=x + width / 2, cy=y + height / 2, w=width, h=height, kind=kind,
                                      reason=reason, box_index=index, origin=box.origin))
    return targets


def detector_labels(
    rows: Iterable[Row], rules: TargetRules = TargetRules(), codes: dict[str, int] = DETECTOR_CODES
) -> list[dict]:
    """One record per row for the detector's trainer:
    ``{"platform", "video_id", "t", "group", "W", "H", "cards": [{"cx", "cy", "w", "h", "k"}]}``.

    ``W`` and ``H`` are the size of the frame the labels were made on; the boxes are fractions.
    """
    return [
        {"platform": row.platform, "video_id": row.video_id, "t": row.t, "group": row.group,
         "W": row.frame_width, "H": row.frame_height,
         "cards": [target.as_dict(codes) for target in detector_targets(row, rules)]}
        for row in rows
    ]


def count_targets(rows: Iterable[Row], rules: TargetRules = TargetRules()) -> dict:
    """Totals over rows: frames, targets of each kind, and how many boxes each rule decided."""
    kinds: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    frames = 0
    for row in rows:
        frames += 1
        for box in _require_boxes(row):
            kind, reason = classify_box(row, box, rules)
            reasons[reason] += 1
            if kind is not None:
                kinds[kind] += 1
    return {"frames": frames, FACE: kinds[FACE], BACK: kinds[BACK], IGNORE: kinds[IGNORE], "reasons": dict(reasons)}


# -------------------------------------------------------------------------- recognizer crops
def crop_exclusion(
    row: Row,
    box: Box,
    rules: TargetRules = TargetRules(),
    exclude_ids: Collection[str] = (),
) -> str | None:
    """Why a box is not a recognizer training crop, or ``None`` when it is one.

    The reasons, in the order they are tested: ``not-face``, ``ignore``, ``unlabeled`` (nobody
    was asked to name it), ``unknown`` (it could not be named), ``unreadable`` (named, but not
    from this frame alone), ``excluded-id`` (its card is in ``exclude_ids``), ``aspect``.
    """
    if box.kind != "face":
        return "not-face"
    if box.ignore:
        return "ignore"
    if box.identity != "named" or box.card_id is None:
        return "unknown" if box.identity == "unknown" else "unlabeled"
    if box.readable == "no":
        return "unreadable"
    if box.card_id in exclude_ids:
        return "excluded-id"
    passes, _, _ = face_exemption(row, box, rules)
    return None if passes else "aspect"


def margin_crop(
    frame: np.ndarray, box: geometry.PixelBox, margin: float, turns: int = 0
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Cut a box out of a frame with room around it, then turn the cut.

    The box ``(x, y, width, height)``, in pixels of ``frame``, is widened by ``margin`` times
    its width on the left and right and ``margin`` times its height above and below, clipped
    to the frame and cut to whole pixels (edges truncated, as every crop in this package is).
    The cut is then turned by ``numpy.rot90(cut, turns)``.

    Returns ``(crop, inner)``: ``inner`` is the original box as ``(x, y, width, height)`` in
    whole pixels of the returned crop. The box is clipped to the frame and cut to whole pixels
    the same way as the crop, before the turn, and then turned with it; so ``inner`` is exactly
    the pixels of the box's own crop, and it never runs past the crop. Where the frame's edge
    cut the margin short, ``inner`` sits off-center; where it cut the box itself, ``inner`` is
    the part of the box that is in the frame, which is nothing (no width, or no height) for a
    box that lies wholly off the frame.
    """
    from cardvision.recognize.crops import crop_edges

    if margin < 0:
        raise ValueError(f"the margin cannot be negative, got {margin}")
    x, y, width, height = box
    frame_height, frame_width = frame.shape[:2]
    # Each widened edge is computed from the box's own edge (x + width + margin * width, not
    # x - margin * width + width * (1 + 2 * margin)): the two agree on paper and can fall on
    # either side of a whole number in floating point.
    x0, y0 = int(max(0, x - margin * width)), int(max(0, y - margin * height))
    x1 = max(x0, int(min(frame_width, x + width + margin * width)))
    y1 = max(y0, int(min(frame_height, y + height + margin * height)))
    cut = frame[y0:y1, x0:x1]
    bx0, by0, bx1, by1 = crop_edges(box, frame_width, frame_height)
    # A box with a pixel in the frame lies inside the cut already. One that lies wholly off the
    # frame has no pixel of its own; it is held to the cut's edge, with no width or no height.
    bx0, by0 = min(max(bx0, x0), x1), min(max(by0, y0), y1)
    bx1, by1 = min(max(bx1, bx0), x1), min(max(by1, by0), y1)
    inner = geometry.rotate_box_with_crop((bx0 - x0, by0 - y0, bx1 - bx0, by1 - by0), x1 - x0, y1 - y0, turns)
    return np.ascontiguousarray(np.rot90(cut, turns % 4)), tuple(int(value) for value in inner)


@dataclass(frozen=True)
class Crop:
    """A crop cut from a labeled frame.

    Attributes:
        image: the pixels, ``(h, w, 3)`` uint8, turned upright for a positive.
        inner: the labeled box inside the crop, ``(x, y, width, height)`` in whole crop pixels
            (see ``margin_crop``).
        card_id: the card, or ``None`` for a negative.
        label: ``face`` for a positive; ``back`` or ``not-card`` for a negative.
        row: the row the crop comes from.
        box_index: the position of the box in the row's ``boxes``.
        turns: the quarter turns applied (``numpy.rot90``'s ``k``).
        angle: the direction of the card's top edge in the frame, in degrees clockwise.
        aspect: shorter side over longer side of the outline.
        short_side: the outline's shorter side in pixels of the frame the crop was cut from.
        half: ``"half"`` or ``"hand-row"`` when the face passed the aspect filter by an
            exemption, else ``None``.
    """

    image: np.ndarray
    inner: tuple[int, int, int, int]
    card_id: str | None
    label: str
    row: Row
    box_index: int
    turns: int = 0
    angle: float = 0.0
    aspect: float = 0.0
    short_side: float = 0.0
    half: str | None = None

    @property
    def name(self) -> str:
        """A file name for the crop, unique across a manifest."""
        box = self.row.boxes[self.box_index] if self.row.boxes else None
        tag = box.id if box is not None and box.id else f"box{self.box_index:03d}"
        return f"{frame_file_name(self.row, '')}-{tag}.png"

    def index_row(self) -> dict:
        """The crop's line of a crop folder's index."""
        box = self.row.boxes[self.box_index] if self.row.boxes else None
        return {
            "file": self.name,
            "inner": list(self.inner),
            "cardId": self.card_id,
            "label": self.label,
            "group": self.row.group,
            "platform": self.row.platform,
            "video_id": self.row.video_id,
            "t": self.row.t,
            "box": self.box_index,
            "source_type": self.row.source_type,
            "turns": self.turns,
            "angle": round(self.angle, 2),
            "aspect": round(self.aspect, 3),
            "shortSidePx": round(self.short_side, 1),
            "half": self.half,
            "readable": None if box is None else box.readable,
            "origin": None if box is None else box.origin,
            "hw": list(self.image.shape[:2]),
        }


Prepare = Callable[[np.ndarray], np.ndarray]


def _analysis_frame(rgb: np.ndarray) -> np.ndarray:
    from cardvision.infer.frames import analysis_frame

    return analysis_frame(rgb)


def _boxed_rows(rows: Iterable[Row]) -> Iterator[Row]:
    for row in rows:
        if row.boxes:
            yield row


def recognizer_crops(
    rows: Iterable[Row],
    frames: FrameStore | str | Path,
    *,
    margin: float,
    rules: TargetRules = TargetRules(),
    exclude_ids: Collection[str] = (),
    known_ids: Collection[str] | None = None,
    prepare: Prepare = _analysis_frame,
    skipped: Counter | None = None,
) -> Iterator[Crop]:
    """The recognizer's positive crops from labeled rows and their frames.

    Args:
        rows: manifest rows; rows without boxes give nothing.
        frames: the folder ``cardvision fetch`` wrote, or a ``FrameStore`` over it.
        margin: the room left around each box, as a fraction of its width and height on each
            side. It has no default: it shapes the training data, so the caller states it.
        rules: the aspect filter and its exemptions.
        exclude_ids: cards to leave out (cards held back from training, for instance).
        known_ids: when given, a named card outside this set is an error, not a crop.
        prepare: applied to each frame before cutting. The default resizes the frame to the
            width the detector analyzes, as inference does, so crops have the size they have
            in use.
        skipped: when given, counts what was left out by reason (those of ``crop_exclusion``,
            ``no-frame`` for a row without a usable frame, ``empty`` for a box off the frame).
    """
    store = frames if isinstance(frames, FrameStore) else FrameStore(frames)
    tally = skipped if skipped is not None else Counter()
    for row in _boxed_rows(rows):
        wanted = []
        for index, box in enumerate(row.boxes or ()):
            reason = crop_exclusion(row, box, rules, exclude_ids)
            if reason is None:
                if known_ids is not None and box.card_id not in known_ids:
                    raise ValueError(f'{row.platform} {row.video_id} t={row.t}: card "{box.card_id}" is not a known id')
                wanted.append((index, box))
            elif reason != "not-face":
                tally[reason] += 1
        if not wanted:
            continue
        try:
            frame = prepare(store.load(row))
        except FrameMissing:
            tally["no-frame"] += len(wanted)
            continue
        height, width = frame.shape[:2]
        for index, box in wanted:
            corners = box.pixel_quad(width, height)
            turns = geometry.quarter_turns(corners) if box.oriented else 0
            image, inner = margin_crop(frame, geometry.bounding_box(corners), margin, turns)
            if image.size == 0:
                tally["empty"] += 1
                continue
            _, how, aspect = face_exemption(row, box, rules)
            yield Crop(image=image, inner=inner, card_id=box.card_id, label=FACE, row=row, box_index=index,
                       turns=turns, angle=geometry.top_edge_angle(corners), aspect=aspect,
                       short_side=geometry.short_side(corners), half=how)


def negative_crops(
    rows: Iterable[Row],
    frames: FrameStore | str | Path,
    *,
    margin: float,
    kinds: Collection[str] = ("back", "not-card"),
    min_side: int = DETECTOR_MIN_CROP_PX,
    prepare: Prepare = _analysis_frame,
    skipped: Counter | None = None,
) -> Iterator[Crop]:
    """Crops the recognizer should name nothing for: labeled backs and ``not-card`` boxes.

    A box is used when its kind is one of ``kinds``, it is not marked ignore, and its upright
    box, clipped to the frame, is at least ``min_side`` pixels on both sides in the prepared
    frame. The crop is not turned.
    """
    store = frames if isinstance(frames, FrameStore) else FrameStore(frames)
    tally = skipped if skipped is not None else Counter()
    for row in _boxed_rows(rows):
        wanted = [(index, box) for index, box in enumerate(row.boxes or ()) if box.kind in kinds and not box.ignore]
        if not wanted:
            continue
        try:
            frame = prepare(store.load(row))
        except FrameMissing:
            tally["no-frame"] += len(wanted)
            continue
        height, width = frame.shape[:2]
        for index, box in wanted:
            corners = box.pixel_quad(width, height)
            clipped = geometry.bounding_box(corners, width, height)
            if min(clipped[2], clipped[3]) < min_side:
                tally["small"] += 1
                continue
            image, inner = margin_crop(frame, clipped, margin, 0)
            yield Crop(image=image, inner=inner, card_id=None, label=box.kind, row=row, box_index=index,
                       aspect=geometry.side_aspect(corners), short_side=geometry.short_side(corners))


def write_crops(crops: Iterable[Crop], out: str | Path) -> int:
    """Save crops as PNG files in ``out`` with an ``index.jsonl`` beside them; returns the
    count. Each index line holds at least ``file``, ``inner``, ``cardId`` and ``group``."""
    from PIL import Image

    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    lines = []
    for crop in crops:
        Image.fromarray(crop.image).save(folder / crop.name, compress_level=1)
        lines.append(json.dumps(crop.index_row(), ensure_ascii=False))
    (folder / "index.jsonl").write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return len(lines)
