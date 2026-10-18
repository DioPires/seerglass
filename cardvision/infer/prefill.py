# SPDX-License-Identifier: Apache-2.0
"""Write detections as a prefill file for the labeling tool.

Labeling goes faster when a person corrects a model's guesses than when they start from a
blank frame. A prefill file is JSON lines, one object per frame::

    {"t": 44.017, "name": "f00045.jpg", "frameW": 1920, "frameH": 1080, "ms": 83.1,
     "boxes": [{"box": [x, y, w, h], "cardId": "ABC-001", "sim": 0.81, "margin": 0.12,
                "top5": [{"id": "ABC-001", "sim": 0.81}, ...], "rot": 0, "flag": null}]}

* ``name`` is the frame's file name and ``t`` its time in the video in seconds;
* ``box`` is in the pixel space ``frameW`` x ``frameH``, which here is the frame's own size,
  rounded to three decimals;
* ``cardId`` is the named card, or null when the recognizer named nothing;
* ``top5`` lists the five best bank rows either way, so the right card is usually one click
  away;
* ``rot`` is the chosen orientation in quarter turns counter-clockwise;
* ``flag`` says why nothing was named: ``"below-tau"`` (the score was under the threshold) or
  ``"low-margin"`` (another card scored too close); null for a named card.

Only faces are written. The format has no field for a card back, so backs are left for the
person labeling to draw.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from cardvision.infer.pipeline import Card, FrameResult, reported_box

FLAG_BELOW_TAU = "below-tau"
FLAG_LOW_MARGIN = "low-margin"


def prefill_flag(card: Card) -> str | None:
    """Why a face was not named, or ``None`` when it was."""
    found = card.recognition
    if found is None or found.accepted:
        return None
    return FLAG_LOW_MARGIN if found.tau_passed else FLAG_BELOW_TAU


def prefill_box(card: Card) -> dict[str, Any]:
    """One face as a prefill box."""
    found = card.recognition
    if found is None:
        raise ValueError("only a recognized face can be written as a prefill box")
    return {
        "box": reported_box(card.box),
        "cardId": found.card_id,
        "sim": round(found.similarity, 4),
        "margin": round(found.margin, 4),
        "top5": [{"id": row.card_id, "sim": round(row.similarity, 4)} for row in found.rows[:5]],
        "rot": found.look,
        "flag": prefill_flag(card),
    }


def prefill_record(result: FrameResult) -> dict[str, Any]:
    """One frame as a prefill object."""
    return {
        "t": result.t,
        "name": result.name,
        "frameW": result.frame_width,
        "frameH": result.frame_height,
        "ms": result.ms,
        "boxes": [prefill_box(card) for card in result.faces],
    }


def write_prefill(path: str | Path, results: Iterable[FrameResult]) -> int:
    """Write one line per frame to ``path``. Returns the number of frames written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(prefill_record(result)) + "\n")
            count += 1
    return count
