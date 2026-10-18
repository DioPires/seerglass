# SPDX-License-Identifier: Apache-2.0
"""A stand-in for the person at the labeling tool, for toy checks only.

Labeling is a person's work. The guides and the tests still need labels to run the route from a
recording to a manifest on the toy game, so ``label_toy_scene`` plays the person: it boxes and
names every card of the toy scene (``test_infer_toy_models.SCENE_CARDS``) on every frame of a
sample, and confirms each frame, through the labeling tool's own save path. The answers are the
scene's own, which the test suite knows because it drew the scene. Nothing here can label a
real frame.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from test_infer_toy_models import EXPECTED_BOXES, EXPECTED_CLASSES, SCENE_CARDS, SCENE_SIZE

LABELER = Path(__file__).resolve().parents[1] / "labeler"

# The card that hangs over the frame's right edge: half of it is out of frame, so it is an
# ignore region around the part that shows (labeler/FORMAT.md).
CUT_BY_THE_EDGE = "TOY-005"


def scene_targets() -> list[dict]:
    """The toy scene's cards as the labeling tool stores them: six faces with their names and
    one card back, each an upright box in pixels of the scene."""
    names = iter(card_id for card_id, _x, _y, _turns in SCENE_CARDS)
    targets = []
    for number, ((x, y, w, h), cls) in enumerate(zip(EXPECTED_BOXES, EXPECTED_CLASSES), start=1):
        card_id = next(names) if cls == "face" else None
        target = {"id": f"B{number:03d}", "corners": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                  "estimated": [False] * 4, "cornerSource": "box", "outcome": "card" if cls == "face" else "card-back",
                  "identity": card_id, "identitySource": "current-frame" if cls == "face" else None,
                  "eligible": "yes" if cls == "face" else "no", "conditions": {}, "labelSource": "manual"}
        if card_id == CUT_BY_THE_EDGE:
            target.update(ignore=True, eligible="no")
        targets.append(target)
    return targets


def label_toy_scene(sample: str | Path, catalog: str | Path, source: str = "physical", physical_group: str = "phone",
                    role: str = "test") -> int:
    """Label and confirm every frame of ``sample`` as the toy scene. Returns the number of frames.

    The labels are written beside the sample, in ``labels/``, as the tool itself writes them.
    ``source`` and ``physical_group`` are the recording tags a person sets in the tool; ``role``
    is the role of the label sets (``test``: made without any model's help).
    """
    sample = Path(sample)
    sys.path.insert(0, str(LABELER))
    try:
        import serve

        serve.configure(serve.build_parser().parse_args([
            "--mode", "box", "--role", role, "--sample", str(sample), "--catalog", str(catalog)]))
        width, height = SCENE_SIZE
        frames = 0
        for asset in json.loads(sample.read_text(encoding="utf-8"))["assets"]:
            for record in asset["frames"]:
                row = {"name": record["name"], "t": record["t"], "frameW": width, "frameH": height, "done": True,
                       "targets": scene_targets()}
                status, saved = serve.save_frame(asset["assetId"], row)
                if status != 200:
                    raise RuntimeError(f"{asset['assetId']}/{record['name']}: the labeling tool refused the frame: {saved}")
                frames += 1
            tags = {"source": source, **({"physicalGroup": physical_group} if source == "physical" else {})}
            status, saved = serve.save_meta(asset["assetId"], {"recording": tags})
            if status != 200:
                raise RuntimeError(f"{asset['assetId']}: the labeling tool refused the recording tags: {saved}")
        return frames
    finally:
        sys.path.remove(str(LABELER))
