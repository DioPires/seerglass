# SPDX-License-Identifier: Apache-2.0
"""The detector alone, on exhaustively boxed frames: recall at false-positive caps, and AP@0.5.

Only rows whose ``completeness`` is ``exhaustive-boxes`` can be used: a detection away from
every label is a false positive only when every card in the frame is known to have a box.

**Targets.** Each box of a row becomes a face, a back, an ignore region or nothing, by the same
rule the detector's trainer uses (``cardvision.data.targets.classify_box``): a box marked
``ignore`` and an ``unreadable`` box are ignore regions, a ``not-card`` box is kept apart, and
a face whose outline does not have a whole card's proportions is an ignore region unless one of
the two exemptions applies. A target's outline is its four labeled corners clipped to the
frame.

So *face targets* is not the number of labeled faces, and it is not the number of crops that
``cardvision evaluate crops`` cuts from the same rows: a face set aside for its proportions
(a card cut off by the frame, or foreshortened by a steep camera) is an ignore region here and
is still a crop there, and a hover event. The result counts those faces
(``facesIgnoredForProportions``) and the report states the number beside the targets, so that
the two reports can be set against each other.

**Overlap: two measures, one per result.** A detection is an upright box and a label is a
four-cornered outline.

* *Recall at false-positive caps* compares **upright boxes**: the detection's box with the
  upright box around the target's outline (intersection over union). The detector is trained
  to predict these upright boxes (``cardvision.data.targets``).
* *AP@0.5* compares the detection's box with the **outline's polygon**, not with the upright
  box around it: a card lying at an angle is not credited to a detection that covers mostly
  table.

The two agree for an upright card and differ for a card at an angle, where the polygon measure
is the stricter one.

**Detections.** The detector runs once per frame with no score threshold, under the pipeline's
analysis width and smallest-crop rule: every box its decode keeps after non-maximum
suppression, up to the decode's limit on boxes per frame. There is no fixed minimum score;
every threshold is read from that one list.

Recall at false-positive caps
-----------------------------
At one score threshold, per frame, among the detections at or above it:

* face detections and face targets are matched one to one, greedily, on the overlap of
  upright boxes: every pair with an overlap of at least 0.3 is ranked, highest overlap first,
  and a pair is taken when neither side is taken yet. Equal overlaps go to the later detection,
  then the later target; the order is fixed so that a result can be reproduced. Back detections
  and back targets are matched the same way, apart from the faces: a detection only ever
  matches a target of its own class;
* a face target is **found** when the pair it was matched in overlaps by at least 0.5, and
  **found at 0.3** when it was matched at all. The same holds for a back target;
* a matched detection is never a false positive, whatever its overlap. An unmatched face
  detection is a **false positive on a back** when it overlaps a back's upright box by 0.3 or
  more; otherwise it is **neutral** (not counted) when half its area or more lies in ignore
  regions; otherwise it is a **false positive**. The area share is measured on a pixel mask of
  the frame. A detection on a ``not-card`` box is a false positive here, like one on bare
  table: a person judged that no card is there.

*Recall* is face targets found / face targets, and *recall at 0.3* is face targets found at
0.3 / face targets. *False positives per frame* is false positives (those on backs included)
/ frames. For a cap ``c``, the threshold reported is the lowest score threshold such that it
and every higher candidate threshold keep false positives per frame at or below ``c``; the
candidates are the scores of the face detections themselves, on all frames pooled. Recall is
then read at that threshold. The thresholds are chosen on all frames together and each slice
is reported at them. The detector's own operating threshold is reported the same way. The
default caps are 0.5, 2.0 and 2.2 false positives per frame.

AP@0.5
------
For one class (face or back), over all frames together, with the overlap measured on the
target's **polygon**:

1. the class's detections are ranked by score, highest first (ties keep frame order, then the
   detector's order);
2. going down the ranking, a detection is a **true positive** when its best overlap with a
   target of its class in its frame that no higher-ranked detection has taken is at least 0.5;
   it takes that target. A second detection of an already taken target is therefore not a
   true positive;
3. a detection that is not a true positive and has half its area or more inside ignore regions
   and ``not-card`` boxes is **left out**: it is neither a true nor a false positive. Ignore
   regions and ``not-card`` boxes are never targets;
4. any other detection is a **false positive**;
5. precision and recall are computed after each ranked detection; precision at a recall level
   is replaced by the highest precision at that recall or any higher one (the monotone
   envelope), and AP is the area under that envelope over recall from 0 to 1 (all-point
   interpolation, no fixed recall grid). A recall never reached contributes nothing.

The ranking holds the boxes the decode keeps, so a target the detector only finds beyond the
decode's limit on boxes per frame earns no credit.

The two results differ on purpose in three places. Recall at caps measures overlap on upright
boxes, matches all pairs by overlap, and counts a detection on a ``not-card`` box as a false
positive; AP measures overlap on the polygon, matches in score order, as AP is usually
defined, and leaves such a detection out. Neither is computed from the other.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from cardvision.core.constants import DETECTOR_BACK_CLASS, DETECTOR_FACE_CLASS
from cardvision.data.extract import FrameStore
from cardvision.data.schema import Row
from cardvision.data.targets import BACK, FACE, IGNORE, TargetRules, classify_box
from cardvision.evaluate.config import Configuration
from cardvision.evaluate.rows import Account, scorable_rows, slices
from cardvision.evaluate.stats import GroupBootstrap, rate
from cardvision.infer.frames import analysis_frame
from cardvision.infer.pipeline import Pipeline, frame_box
from cardvision.recognize import large_enough

Corners = tuple[float, float, float, float]  # x0, y0, x1, y1
Polygon = list[tuple[float, float]]
Detection = tuple[Corners, float, int]  # box, score, class

DEFAULT_CAPS = (0.5, 2.0, 2.2)
MATCH_IOU = 0.5   # a target is found when its matched pair overlaps by this much; also AP's limit
LOOSE_IOU = 0.3   # the limit a pair needs to be matched at all
BACK_TOUCH_IOU = 0.3
AREA_SHARE = 0.5
OPERATING_POINT = "operatingPoint"

DEFINITIONS = {
    "targets": "targets = the labeled boxes of the used frames, read as the detector's trainer reads them: a face "
               "is a face target unless it is an ignore region, and the ignore regions are the boxes marked ignore, "
               "the unreadable boxes, and the faces whose outline does not have a whole card's proportions (cut off, "
               "or seen at a steep angle). Those last faces are counted apart. cardvision evaluate crops and "
               "evaluate hover still count them as faces, so their totals can be larger than face targets",
    "matching": "matching, for recall and false positives: detections and targets of one class are matched one to "
                "one, pairs taken highest overlap first, at an overlap of at least 0.3; the overlap is the "
                "intersection over union of the detection's box with the upright box around the target's outline",
    "recall": "recall = face targets whose matched pair overlaps by at least 0.5 / face targets",
    "recallLoose": "recall at 0.3 = face targets matched (an overlap of at least 0.3) / face targets",
    "falsePositivesPerFrame": "false positives per frame = unmatched face detections that are not neutral / frames",
    "cap": "at a cap c: the lowest score threshold such that it and every higher one keep false positives per "
           "frame at or below c, on all frames pooled; no fixed minimum score",
    "ap": "AP@0.5 = area under the precision-recall curve's monotone envelope (all-point interpolation), "
          "detections ranked by score, each target matched once at an overlap of at least 0.5; here the overlap "
          "is the intersection over union of the detection's box with the polygon of the target's outline, not "
          "with the upright box around it; detections mostly inside ignore regions or not-card boxes are left out",
}


# ---------------------------------------------------------------------------------- geometry
def polygon_area(polygon: Sequence[Sequence[float]]) -> float:
    """Area of a polygon by the shoelace rule; 0 for fewer than three corners."""
    if len(polygon) < 3:
        return 0.0
    points = np.asarray(polygon, dtype=np.float64)
    return 0.5 * abs(float(np.dot(points[:, 0], np.roll(points[:, 1], -1)) - np.dot(points[:, 1], np.roll(points[:, 0], -1))))


def clip_to_rect(polygon: Sequence[Sequence[float]], x0: float, y0: float, x1: float, y1: float) -> Polygon:
    """A convex polygon clipped to an upright rectangle (Sutherland-Hodgman)."""
    def clip(points, inside, cross):
        out = []
        for i in range(len(points)):
            a, b = points[i - 1], points[i]
            a_in, b_in = inside(a), inside(b)
            if b_in:
                if not a_in:
                    out.append(cross(a, b))
                out.append(b)
            elif a_in:
                out.append(cross(a, b))
        return out

    def at_x(x):
        return lambda a, b: (x, a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0]))

    def at_y(y):
        return lambda a, b: (a[0] + (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]), y)

    points = [(float(x), float(y)) for x, y in polygon]
    for inside, cross in ((lambda p: p[0] >= x0, at_x(x0)), (lambda p: p[0] <= x1, at_x(x1)),
                          (lambda p: p[1] >= y0, at_y(y0)), (lambda p: p[1] <= y1, at_y(y1))):
        if not points:
            return []
        points = clip(points, inside, cross)
    return points


def bounds(polygon: Sequence[Sequence[float]]) -> Corners:
    """The upright box ``(x0, y0, x1, y1)`` around a polygon."""
    points = np.asarray(polygon, dtype=np.float64)
    return (float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max()))


def corners_iou(a: Corners, b: Corners) -> float:
    """Intersection over union of two ``(x0, y0, x1, y1)`` boxes."""
    wide = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    tall = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = wide * tall
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def polygon_iou(polygon: Sequence[Sequence[float]], box: Corners) -> float:
    """Intersection over union of a label's polygon with an upright detection box."""
    inter = polygon_area(clip_to_rect(polygon, *box))
    union = polygon_area(polygon) + (box[2] - box[0]) * (box[3] - box[1]) - inter
    return inter / union if union > 0 else 0.0


def polygon_mask(width: int, height: int, polygons: Sequence[Sequence[Sequence[float]]]) -> np.ndarray:
    """A ``(height, width)`` boolean mask that is true inside any of the polygons."""
    image = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        if len(polygon) >= 3:
            draw.polygon([tuple(point) for point in polygon], fill=1)
    return np.asarray(image, dtype=bool)


def share_inside(mask: np.ndarray, box: Corners) -> float:
    """The share of a box's area that lies on the mask, with the box rounded to whole pixels."""
    x0, y0, x1, y1 = (int(round(value)) for value in box)
    area = max(1, (x1 - x0) * (y1 - y0))
    return float(mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)].sum()) / area


def greedy_match(
    boxes: Sequence[Corners], targets: Sequence[Corners], limit: float = LOOSE_IOU
) -> tuple[dict[int, int], dict[int, float]]:
    """Match detections to targets one to one by the overlap of upright boxes, best pairs first.

    ``boxes`` and ``targets`` are ``(x0, y0, x1, y1)``; a target is the upright box around its
    outline (``bounds``). Returns ``(detection -> target, target -> overlap of its pair)``.
    Every pair whose overlap is at least ``limit`` is ranked by overlap, highest first; equal
    overlaps go to the later detection, then the later target. A pair is taken when neither of
    its sides is taken yet.
    """
    pairs = sorted(
        ((overlap, d, t) for d, box in enumerate(boxes) for t, target in enumerate(targets)
         if (overlap := corners_iou(box, target)) >= limit),
        reverse=True,
    )
    by_detection: dict[int, int] = {}
    overlap_of: dict[int, float] = {}
    for overlap, d, t in pairs:
        if d in by_detection or t in overlap_of:
            continue
        by_detection[d] = t
        overlap_of[t] = overlap
    return by_detection, overlap_of


# ------------------------------------------------------------------------------------ frames
@dataclass
class FrameRecord:
    """One frame's targets and detections, in the frame's own pixels.

    ``faces`` and ``backs`` are target polygons; ``ignore`` and ``not_card`` are the polygons
    of ignore regions and ``not-card`` boxes. ``detections`` are ``(box, score, class)``, best
    first: every box the decode kept, with no score threshold. ``set_aside`` counts the faces
    among the ignore regions that were not marked ignore: the ones whose outline does not have
    a whole card's proportions.
    """

    group: str
    source_type: str
    width: int
    height: int
    faces: list[Polygon] = field(default_factory=list)
    backs: list[Polygon] = field(default_factory=list)
    ignore: list[Polygon] = field(default_factory=list)
    not_card: list[Polygon] = field(default_factory=list)
    detections: list[Detection] = field(default_factory=list)
    key: str = ""
    set_aside: int = 0
    _masks: dict[str, np.ndarray] = field(default_factory=dict, repr=False, compare=False)

    def mask(self, name: str) -> np.ndarray:
        """A cached pixel mask: ``ignore``, or ``neutral`` (ignore regions and ``not-card``
        boxes together)."""
        if name not in self._masks:
            polygons = {"ignore": self.ignore, "neutral": [*self.ignore, *self.not_card]}[name]
            self._masks[name] = polygon_mask(self.width, self.height, polygons)
        return self._masks[name]


def frame_truth(row: Row, width: int, height: int, rules: TargetRules = TargetRules()) -> FrameRecord:
    """The targets of an exhaustively boxed row, in pixels of a frame of the given size."""
    record = FrameRecord(group=row.group, source_type=row.source_type, width=width, height=height,
                         key=f"{row.platform}-{row.video_id}-t{round(row.t * 1000):010d}")
    for box in row.boxes or ():
        kind, reason = classify_box(row, box, rules)
        polygon = clip_to_rect(box.pixel_quad(width, height).tolist(), 0.0, 0.0, float(width), float(height))
        if polygon_area(polygon) <= 0:
            continue
        {FACE: record.faces, BACK: record.backs, IGNORE: record.ignore, None: record.not_card}[kind].append(polygon)
        record.set_aside += reason == "aspect"
    return record


def detect_frame(pipeline: Pipeline, rgb: np.ndarray) -> list[Detection]:
    """The pipeline's detections for a frame with no score threshold: ``(box, score, class)``
    in frame pixels, best first, after the pipeline's analysis resize and smallest-crop rule.
    The decode still applies its non-maximum suppression and its limit on boxes per frame."""
    frame_height, frame_width = rgb.shape[:2]
    analyzed = analysis_frame(rgb, pipeline.min_width, pipeline.max_width)
    analysis_height, analysis_width = analyzed.shape[:2]
    scale_x, scale_y = frame_width / analysis_width, frame_height / analysis_height
    out = []
    for detection in pipeline.detector.detect(analyzed, threshold=0.0):
        if not large_enough(detection.box, analysis_width, analysis_height, pipeline.min_crop_px):
            continue
        x, y, width, height = frame_box(detection, scale_x, scale_y, frame_width, frame_height)
        out.append(((x, y, x + width, y + height), detection.score, detection.cls))
    return out


def _labels(row: Row) -> Counter:
    if row.completeness != "exhaustive-boxes":
        return Counter()
    return Counter({"frame": 1})


def run_detector(
    rows: Sequence[Row],
    frames: FrameStore | str | Path,
    config: Configuration,
    account: Account,
    *,
    rules: TargetRules = TargetRules(),
) -> list[FrameRecord]:
    """Detect on every exhaustively boxed row's frame and pair the detections with the targets."""
    records = []
    for row, rgb in scorable_rows(rows, frames, account, _labels):
        record = frame_truth(row, rgb.shape[1], rgb.shape[0], rules)
        record.detections = detect_frame(config.pipeline, rgb)
        records.append(record)
    return records


# ------------------------------------------------------------------- recall at false-positive caps
def score_frame(record: FrameRecord, threshold: float) -> dict[str, int]:
    """One frame's counts at a score threshold (see the module's rules)."""
    faces = [box for box, score, cls in record.detections if cls == DETECTOR_FACE_CLASS and score >= threshold]
    backs = [box for box, score, cls in record.detections if cls == DETECTOR_BACK_CLASS and score >= threshold]
    face_bounds = [bounds(polygon) for polygon in record.faces]
    back_bounds = [bounds(polygon) for polygon in record.backs]
    # One matching per class, at the looser limit: a target found at 0.5 is one whose pair
    # overlaps that much, and a detection matched at 0.3 is not a false positive.
    matched, face_overlap = greedy_match(faces, face_bounds, LOOSE_IOU)
    _, back_overlap = greedy_match(backs, back_bounds, LOOSE_IOU)
    false_positives = on_back = neutral = 0
    for index, box in enumerate(faces):
        if index in matched:
            continue
        if any(corners_iou(box, other) >= BACK_TOUCH_IOU for other in back_bounds):
            false_positives += 1
            on_back += 1
        elif record.ignore and share_inside(record.mask("ignore"), box) >= AREA_SHARE:
            neutral += 1
        else:
            false_positives += 1
    return {"frames": 1, "faces": len(record.faces),
            "hits": sum(1 for overlap in face_overlap.values() if overlap >= MATCH_IOU), "hitsLoose": len(face_overlap),
            "falsePositives": false_positives, "falsePositivesOnBacks": on_back, "neutral": neutral,
            "backs": len(record.backs),
            "backHits": sum(1 for overlap in back_overlap.values() if overlap >= MATCH_IOU)}


def walk_down(candidates: Sequence[float], false_positives_at, limit: float) -> float:
    """The lowest of ``candidates`` (ascending) such that it and every higher one keep
    ``false_positives_at(threshold)`` at or below ``limit``; infinity when even the highest
    does not."""
    fixed = None
    for threshold in reversed(candidates):
        if false_positives_at(threshold) <= limit + 1e-9:
            fixed = threshold
        else:
            break
    return fixed if fixed is not None else float("inf")


def cap_thresholds(records: Sequence[FrameRecord], caps: Sequence[float]) -> dict[float, float | None]:
    """The score threshold for each false-positive cap (``None`` when no threshold meets it, and
    when no frame has a face detection: there is then no candidate threshold).

    The candidates are the scores of the face detections of all the frames, pooled; there is no
    fixed minimum score. The same result as ``walk_down`` over every candidate, computed by
    rescoring only the frames whose own detections enter at each step.
    """
    owners: dict[float, set[int]] = {}
    for index, record in enumerate(records):
        for _, score, cls in record.detections:
            if cls == DETECTOR_FACE_CLASS:
                owners.setdefault(score, set()).add(index)
    candidates = sorted(owners, reverse=True)
    per_frame = [0] * len(records)
    total = 0
    fixed: dict[float, float | None] = {cap: None for cap in caps}
    alive = set(caps)
    for threshold in candidates:
        for index in owners.get(threshold, ()):
            count = score_frame(records[index], threshold)["falsePositives"]
            total += count - per_frame[index]
            per_frame[index] = count
        for cap in sorted(alive):
            if records and total / len(records) <= cap + 1e-9:
                fixed[cap] = threshold
            else:
                alive.discard(cap)
        if not alive:
            break
    return fixed


def _point(records: Sequence[FrameRecord], threshold: float | None, seed: int, replicates: int) -> dict[str, Any]:
    if threshold is None:
        return {"threshold": None, "note": "no threshold: there is no face detection, or no face detection's score "
                                           "keeps false positives per frame within the cap"}
    scores = [score_frame(record, threshold) for record in records]
    groups = [record.group for record in records]
    boot = GroupBootstrap(set(groups), seed, replicates)
    total = Counter()
    for score in scores:
        total.update(score)
    column = {name: boot.per_group(groups, [float(score[name]) for score in scores]) for name in
              ("frames", "faces", "hits", "hitsLoose", "falsePositives", "backs", "backHits")}
    return {
        "threshold": threshold,
        "frames": total["frames"],
        "groups": len(boot.groups),
        "recall": rate(total["hits"], total["faces"], bound="lower", bootstrap=boot,
                       per_group=(column["hits"], column["faces"])),
        "recallLoose": rate(total["hitsLoose"], total["faces"], bound="lower", bootstrap=boot,
                            per_group=(column["hitsLoose"], column["faces"])),
        "falsePositivesPerFrame": rate(total["falsePositives"], total["frames"], bootstrap=boot,
                                       per_group=(column["falsePositives"], column["frames"])),
        "falsePositivesOnBacks": total["falsePositivesOnBacks"],
        "neutralDetections": total["neutral"],
        "backRecall": rate(total["backHits"], total["backs"], bound="lower", bootstrap=boot,
                           per_group=(column["backHits"], column["backs"])),
    }


# --------------------------------------------------------------------------------------- AP
def average_precision(records: Sequence[FrameRecord], cls: int, iou: float = MATCH_IOU) -> dict[str, Any]:
    """AP at one overlap limit for one class over a set of frames (see the module's rules).

    The overlap is ``polygon_iou``: the intersection over union of the detection's upright box
    with the polygon of the target's labeled outline. That is not the measure recall at
    false-positive caps uses, which compares the detection's box with the upright box around
    the outline (``greedy_match``); for a card at an angle the polygon measure is the stricter.

    Returns the AP (``None`` when there is no target) and the counts behind it.
    """
    ranked = sorted(
        ((score, f, d) for f, record in enumerate(records)
         for d, (_, score, det_cls) in enumerate(record.detections) if det_cls == cls),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    targets = [record.faces if cls == DETECTOR_FACE_CLASS else record.backs for record in records]
    taken: list[set[int]] = [set() for _ in records]
    total = sum(len(polygons) for polygons in targets)
    outcomes: list[bool] = []
    left_out = 0
    for _, f, d in ranked:
        box = records[f].detections[d][0]
        best, best_overlap = -1, iou
        for t, polygon in enumerate(targets[f]):
            if t in taken[f]:
                continue
            overlap = polygon_iou(polygon, box)
            if overlap >= best_overlap and (best < 0 or overlap > best_overlap):
                best, best_overlap = t, overlap
        if best >= 0:
            taken[f].add(best)
            outcomes.append(True)
        elif (records[f].ignore or records[f].not_card) and share_inside(records[f].mask("neutral"), box) >= AREA_SHARE:
            left_out += 1
        else:
            outcomes.append(False)
    true_positives = int(sum(outcomes))
    out = {"ap": None, "targets": total, "detections": len(ranked), "truePositives": true_positives,
           "falsePositives": len(outcomes) - true_positives, "leftOut": left_out}
    if total == 0:
        return out
    hits = np.cumsum(outcomes, dtype=np.float64) if outcomes else np.zeros(0)
    ranks = np.arange(1, len(outcomes) + 1, dtype=np.float64)
    precision = hits / ranks if len(outcomes) else np.zeros(0)
    recall = hits / total if len(outcomes) else np.zeros(0)
    envelope = np.maximum.accumulate(precision[::-1])[::-1] if len(outcomes) else precision
    previous = np.concatenate([[0.0], recall[:-1]]) if len(outcomes) else recall
    out["ap"] = float(np.sum((recall - previous) * envelope))
    return out


# ---------------------------------------------------------------------------------- results
def detector_results(
    records: Sequence[FrameRecord],
    operating_threshold: float,
    *,
    caps: Sequence[float] = DEFAULT_CAPS,
    seed: int = 0,
    replicates: int = 10000,
) -> dict[str, Any]:
    """Recall at the operating point and at each cap, and AP@0.5 for faces and backs: overall,
    by source type and by source family. The cap thresholds are chosen on all frames."""
    thresholds: dict[str, float | None] = {OPERATING_POINT: operating_threshold}
    for cap, threshold in cap_thresholds(records, caps).items():
        thresholds[f"falsePositivesPerFrame<={cap:g}"] = threshold

    def measure(part: list[FrameRecord]) -> dict[str, Any]:
        return {
            "frames": len(part),
            "faceTargets": sum(len(record.faces) for record in part),
            "backTargets": sum(len(record.backs) for record in part),
            "ignoreRegions": sum(len(record.ignore) for record in part),
            "facesIgnoredForProportions": sum(record.set_aside for record in part),
            "notCardBoxes": sum(len(record.not_card) for record in part),
            "points": {name: _point(part, threshold, seed, replicates) for name, threshold in thresholds.items()},
            "ap50": {"face": average_precision(part, DETECTOR_FACE_CLASS), "back": average_precision(part, DETECTOR_BACK_CLASS)},
        }

    parts = slices(records, lambda record: record.source_type)
    return {
        "overlap": {"recall": "upright boxes", "ap50": "outline polygon"},
        "thresholds": thresholds,
        "overall": measure(parts["overall"]["all"]),
        "bySourceType": {name: measure(part) for name, part in parts["bySourceType"].items()},
        "bySourceFamily": {name: measure(part) for name, part in parts["bySourceFamily"].items()},
    }
