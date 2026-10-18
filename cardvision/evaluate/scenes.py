# SPDX-License-Identifier: Apache-2.0
"""Generated scenes as rows and frames that an evaluation scores.

``cardvision make-composites`` writes scenes with exact labels: it knows where every card of a
scene is and which card it is. ``read_scenes`` reads such a folder as the rows and the frames
the measures of this package take, so that a scene set is scored by the code that scores a
label manifest. Nothing is matched, counted or reported here.

**Generated scenes are not footage.** They are a harder and a different test from a real
table, and a figure on them says whether the tools run and roughly where the models stand. It
does not say how the models do on real video. The rows are marked so that the report says so
in its first lines (``Account.scenes``).

**Which folders are read.** A folder is refused unless it is one the generator wrote and still
holds what it wrote: ``manifest.json`` names the generator's format, ``labels.jsonl`` has the
SHA-256 that ``manifest.json`` records, and the labels have none of the problems that
``make-composites --check`` counts (``cardvision.composites.validate.label_problems``): every
box lies in its frame, every face names a card of the catalog being scored, every image is
there.

**How a scene becomes a row.** One scene is one row, in memory only: no manifest is written.

* The frame is *exhaustively boxed* (``completeness`` ``exhaustive-boxes``): the generator
  labels every card it draws, so anything without a box is background. Hover events are
  therefore the centers of the boxes, by the rule ``cardvision.evaluate.events`` has for such
  rows. No pointer position is made up here.
* A face becomes a ``face`` box named by its ``cardId``, and a back a ``back`` box. The
  outline is the generator's ``corners``, which is the whole card, any sleeve included, and the
  first corner is the card's own top-left (``oriented``). ``readable`` is not recorded: nobody
  judged it.
* **A card that is mostly hidden, or tiny, is an ignore region**, as a mostly covered or
  cut-off card is in a manifest: it gives no hover event and no crop, and a detection on it is
  neither a card found nor a false positive. The rule uses the generator's own figures and
  limits. A face is an ignore region exactly when the generator marked it ``ignore``: less
  than ``visibleMin`` of its area shows (``lowvis``), or its short side is under
  ``shortSideMinPx`` (``small``). The generator marks no back, so a back is an ignore region
  when its ``visible`` or its ``short`` is under the same two limits, which ``manifest.json``
  records (a half, and 20 px).
* Everything else the generator draws (distractors, dice, hands, broadcast graphics) has no
  box. It is background, so a detection on it is a false positive and no hover event lies on
  it.
* All scenes of a folder are **one recording group**: they come from one generator, one seed
  and one set of card images. With one group no rate has a group bootstrap interval, and the
  report says so.
* The source type is ``generated``. That is not a source type of the manifest format, whose
  two families are footage of a table and of a client's screen, and a scene is neither. The
  row is of platform ``local`` under the name ``<split>-seed<seed>``, and its time is the
  scene's index: a scene is not a moment of a video. ``role``, ``trained_on`` and ``credit``
  are filled in because a row has them, and nothing on this path reads them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cardvision.composites.dataset import FORMAT, FORMAT_VERSION, LABELS_NAME, MANIFEST_NAME, read_labels
from cardvision.composites.validate import label_problems
from cardvision.core.catalog import Catalog
from cardvision.data.extract import RATIO_TOLERANCE, FrameMissing, FrameStore
from cardvision.data.schema import LOCAL_PLATFORM, Box, Row, RowKey
from cardvision.evaluate.config import file_sha256
from cardvision.evaluate.rows import Account

SOURCE_TYPE = "generated"
GROUP = "generated-scenes"
CREDIT = "cardvision make-composites"
LISTED_PROBLEMS = 3

# What the report says about such rows, after naming them as generated scenes and not footage.
NOTE = (
    "Generated scenes are a harder and a different test from a real table: the cards are small, turned, overlapped "
    "and blurred, and the table is drawn by code. Figures on them say whether the tools run and roughly where the "
    "models stand. They do not say how the models do on real video."
)


class ScenesError(ValueError):
    """A folder cannot be scored as generated scenes."""


class SceneFrames(FrameStore):
    """The images of a folder of generated scenes, looked up by the row of each scene."""

    def __init__(self, folder: str | Path, files: dict[RowKey, str], tolerance: float = RATIO_TOLERANCE) -> None:
        self.folder = Path(folder)
        self.tolerance = tolerance
        self._files = dict(files)

    def path(self, row: Row) -> Path:
        name = self._files.get(row.key)
        if name is None or not (self.folder / name).is_file():
            raise FrameMissing(f"{row.platform} {row.video_id} t={row.t}: no scene image in {self.folder}")
        return self.folder / name


def _run_record(folder: Path) -> dict[str, Any]:
    """The generator's ``manifest.json`` of a folder. Raises ``ScenesError`` when the folder is
    not one the generator wrote, or its labels are no longer the ones it wrote."""
    refusal = f"{folder}: not a folder of generated scenes that cardvision make-composites wrote"
    if not folder.is_dir():
        raise ScenesError(f"{folder}: no such folder of generated scenes")
    if not (folder / MANIFEST_NAME).is_file() or not (folder / LABELS_NAME).is_file():
        raise ScenesError(f"{refusal}: it has no {MANIFEST_NAME} and {LABELS_NAME}")
    try:
        record = json.loads((folder / MANIFEST_NAME).read_text(encoding="utf-8"))
    except ValueError:
        record = None
    if not isinstance(record, dict) or record.get("format") != FORMAT:
        raise ScenesError(f'{refusal}: its {MANIFEST_NAME} does not have the format "{FORMAT}"')
    if record.get("formatVersion") != FORMAT_VERSION:
        raise ScenesError(f"{folder}: its {MANIFEST_NAME} has format version {record.get('formatVersion')!r}, and this "
                          f"package reads version {FORMAT_VERSION}")
    if file_sha256(folder / LABELS_NAME) != record.get("labelsSha256"):
        raise ScenesError(f"{folder}: {LABELS_NAME} is not the file the generator wrote: its SHA-256 differs from the "
                          f"one {MANIFEST_NAME} records. Generate the scenes again")
    return record


def scene_box(found: dict[str, Any], width: int, height: int, visible_min: float, short_min: float) -> Box | None:
    """The box of one labeled object of a scene, or ``None`` for an object that is not a card
    (see the module's rules)."""
    kind = found["kind"]
    if kind not in ("face", "back"):
        return None
    quad = tuple((float(x) / width, float(y) / height) for x, y in found["corners"])
    if kind == "back":
        hidden = found["visible"] < visible_min or found["short"] < short_min
        return Box(kind="back", quad=quad, ignore=hidden, oriented=True, id=str(found["id"]), origin="generated")
    return Box(kind="face", quad=quad, ignore=found["ignore"] is not None, oriented=True, card_id=found["cardId"],
               identity="named", readable=None, id=str(found["id"]), origin="generated")


def read_scenes(folder: str | Path, catalog: Catalog) -> tuple[list[Row], Account, SceneFrames]:
    """The rows, the account and the frames of a folder of generated scenes.

    ``catalog`` is the catalog of the configuration being scored: a face of a card it does
    not hold is a label problem. Raises ``ScenesError`` saying why when the folder cannot be
    scored (see the module's description).
    """
    folder = Path(folder)
    record = _run_record(folder)
    scenes = read_labels(folder)
    try:
        problems = label_problems(folder, scenes, catalog)
    except (KeyError, TypeError, ValueError) as error:
        problems = [f"a scene's labels cannot be read ({type(error).__name__}: {error})"]
    if problems:
        more = f"; and {len(problems) - LISTED_PROBLEMS} more" if len(problems) > LISTED_PROBLEMS else ""
        raise ScenesError(f"{folder}: the labels fail the check of make-composites --check, with {len(problems)} "
                          f"problem(s): {'; '.join(problems[:LISTED_PROBLEMS])}{more}")
    split, seed = record["split"], record["seed"]
    visible_min = float(record["parameters"]["visibleMin"])
    short_min = float(record["parameters"]["shortSideMinPx"])
    rows: list[Row] = []
    files: dict[RowKey, str] = {}
    counts = {"faces": 0, "backs": 0, "ignoredFaces": 0, "ignoredBacks": 0}
    for scene in scenes:
        width, height = int(scene["width"]), int(scene["height"])
        boxes = [box for box in (scene_box(found, width, height, visible_min, short_min) for found in scene["objects"])
                 if box is not None]
        for box in boxes:
            counts["faces" if box.kind == "face" else "backs"] += 1
            if box.ignore:
                counts["ignoredFaces" if box.kind == "face" else "ignoredBacks"] += 1
        row = Row(platform=LOCAL_PLATFORM, video_id=f"{split}-seed{seed}", t=float(scene["index"]), frame_width=width,
                  frame_height=height, source_type=SOURCE_TYPE, group=GROUP, role="extra", trained_on=False,
                  completeness="exhaustive-boxes", credit=CREDIT, boxes=tuple(boxes))
        rows.append(row)
        files[row.key] = scene["file"]
    account = Account(role=None, read=len(rows))
    account.manifests.append({"file": LABELS_NAME, "sha256": record["labelsSha256"], "rows": len(rows)})
    account.scenes = {"generator": "cardvision make-composites", "format": FORMAT, "formatVersion": FORMAT_VERSION,
                      "split": split, "seed": seed, "scenes": len(rows), "visibleMin": visible_min,
                      "shortSideMinPx": short_min, "boxes": counts, "note": NOTE}
    return rows, account, SceneFrames(folder, files)
