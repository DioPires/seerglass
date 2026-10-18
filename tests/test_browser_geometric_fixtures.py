# SPDX-License-Identifier: Apache-2.0
"""The fixtures that pin the browser runtime's geometric stage to this package's.

``browser/test/fixtures/geometric.json`` holds inputs and the answers ``cardvision.geometric``
and the recognizer's stage seam give for them. ``browser/test/geometric.test.ts`` computes the
same answers in TypeScript and compares them with the file; this module computes them again in
Python and compares them with the file, so the file cannot drift from either side unnoticed.

Every input is generated from arithmetic: the "cards" are rectangles of colours drawn from a
xorshift stream (``card_rgb``), the "crops" are those cards scaled by nearest neighbour onto a
background of coloured blocks (``scene_rgb``). No image, footage or model is involved, and the
same two functions are written in TypeScript in the browser's test.

Sections:

- ``config``: settings documents and what ``parse_geometric`` makes of them: the full settings
  (``to_json``) or the list of errors.
- ``round``: Python's ``round(x, 6)`` on values that include ties.
- ``fusion``: fused scores for inlier counts around the floor and the cap, and the median
  baseline, as the hexadecimal bytes of the doubles.
- ``vote``: made-up nearest-neighbour pairs and what ``GeometricStage.votes`` counts from them
  (the ratio test and its exemptions for one card, one identity and one image).
- ``seam``: rows of cosines, stage results, and what ``Recognizer._geometric_recognition``
  answers, re-ranking and boosting.
- ``scenes``: a descriptor index of the toy cards and what ``GeometricStage.verify`` finds on
  each crop, with the crop's own keypoints. Keypoints depend on the OpenCV build; the file
  records the version it was written with, and the check of this section runs only on that
  version.

To write the file again after a deliberate change::

    python tests/test_browser_geometric_fixtures.py --write
"""

from __future__ import annotations

import base64
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cardvision.geometric.config import GeometricConfigError, parse_geometric
from cardvision.geometric.stage import GeometricStage, StageResult, Verified
from cardvision.recognize.accept import AcceptRule, Choice
from cardvision.recognize.looks import Look, identity_columns, score_look

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "browser" / "test" / "fixtures" / "geometric.json"


class Stream:
    """A xorshift32 stream of numbers in [0, 1) with 24 bits each (as in test_browser_fixtures)."""

    def __init__(self, seed: int) -> None:
        self.s = seed % 2**32 or 1

    def unit(self) -> float:
        s = self.s
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= s >> 17
        s ^= (s << 5) & 0xFFFFFFFF
        self.s = s
        return (s >> 8) / 2**24

    def below(self, n: int) -> int:
        return int(self.unit() * n)


def hex64(value: float) -> str:
    """A double's bytes, big-endian, as hexadecimal text."""
    return struct.pack(">d", float(value)).hex()


# --------------------------------------------------------------------------------- settings
PASSED = {"candidates": {"embeddingTopK": 10, "voteTopN": 3}, "crop": {"longSide": 160, "maxFeatures": 500},
          "ransac": {"model": "affine"}, "fusion": {"weight": 1.5, "minInliers": 15, "fullInliers": 50},
          "index": {"path": "index/manifest.json", "sha256": "0" * 64}}

CONFIG_DOCS = [
    {},
    PASSED,
    {"seed": 3, "features": {"type": "sift", "params": {"sigma": 2}}, "gate": {"belowAccept": True,
                                                                              "sources": ["physical/*"]}},
    {"features": {"type": "akaze", "mask": [[0, 0.9, 1, 1]]}, "fusion": {"baseline": "median", "reorder": False}},
    {"fusion": {"wieght": 1}, "ransac": {"model": "conic"}, "features": {"type": "orb", "params": {"threshold": 1}},
     "colour": 1},
    {"ransac": {"method": "usac", "model": "similarity", "minAreaRatio": 5, "maxAreaRatio": 4}},
    {"candidates": {"embeddingTopK": 0, "voteTopN": 0}, "vote": {"ratio": 0, "matcher": "kd"}, "match": {"ratio": 1.5}},
    {"index": {"path": "", "sha256": "ABC"}, "seed": -1},
    {"index": [], "crop": [], "gate": {"belowAccept": 1, "sources": []}},
    {"features": {"type": "surf"}},
    {"features": {"mask": [[0.5, 0, 0.2, 1]], "params": {"nLevels": 0, "scaleFactor": True}, "maxFeatures": 0}},
    {"ransac": {"threshold": 0, "confidence": 1, "maxIters": 0}, "fusion": {"weight": -1, "penalty": "x",
                                                                            "minInliers": -1, "fullInliers": 0}},
    {"index": {"path": "a.json", "other": 1}},
    [],
]


def section_config() -> list[dict]:
    out = []
    for doc in CONFIG_DOCS:
        try:
            config = parse_geometric(doc)
        except GeometricConfigError as error:
            out.append({"doc": doc, "errors": error.errors})
            continue
        out.append({"doc": doc, "settings": config.to_json(),
                    "indexPath": None if config.index_path is None else str(config.index_path),
                    "indexSha256": config.index_sha256})
    return out


# ----------------------------------------------------------------------------------- round
ROUND_VALUES = [0.0078125, -0.0078125, 0.1234565, 0.6200005, 1.0000005, 2.5e-7, 1.5e-6, 0.30000000000000004,
                1.234567890123, -0.000000499, 0.7071067811865476, 123.4567895, 1e-7, 0.62, 1.71, 1 / 3, 3.0000005,
                0.4999995, 1.9999995]


def section_round() -> list[dict]:
    return [{"x": hex64(x), "rounded": hex64(round(x, 6))} for x in ROUND_VALUES]


# ---------------------------------------------------------------------------------- fusion
def section_fusion() -> dict:
    stream = Stream(41)
    fusion = parse_geometric(PASSED)["fusion"]
    stage = SimpleNamespace(config=parse_geometric(PASSED))
    cases = []
    for inliers in (0, 1, 14, 15, 16, 30, 49, 50, 51, 200):
        for _ in range(3):
            cosine = stream.unit() * 1.2 - 0.2
            evidence = max(0.0, inliers - 0.0)
            cases.append({"cosine": hex64(cosine), "inliers": inliers,
                          "fused": hex64(GeometricStage.fused_score(stage, cosine, evidence))})
    median_cfg = parse_geometric({**PASSED, "fusion": {**PASSED["fusion"], "baseline": "median", "penalty": 0.25}})
    median_stage = SimpleNamespace(config=median_cfg)
    medians = []
    for inliers in ([3, 20, 60], [0, 15, 16, 40], [7], [10, 11]):
        baseline = float(np.median(inliers))
        cosine = stream.unit()
        medians.append({"inliers": inliers, "cosine": hex64(cosine), "baseline": hex64(baseline),
                        "evidence": [hex64(max(0.0, n - baseline)) for n in inliers],
                        "fused": [hex64(GeometricStage.fused_score(median_stage, cosine, max(0.0, n - baseline)))
                                  for n in inliers]})
    return {"fusion": fusion, "cases": cases, "median": {"fusion": median_cfg["fusion"], "cases": medians}}


# ------------------------------------------------------------------------------------ vote
class _Match:
    def __init__(self, row: int, distance: int) -> None:
        self.imgIdx, self.trainIdx, self.distance = 0, row, float(distance)


class _Matcher:
    def __init__(self, pairs) -> None:
        self.pairs = pairs

    def knnMatch(self, _descriptors, k: int):  # noqa: N802 (OpenCV's name)
        return [[_Match(*m) for m in pair] for pair in self.pairs]


def section_vote() -> dict:
    """Card positions 0..5 with rows; positions 1 and 2 one identity, 4 and 5 one image."""
    ids = ["V-0", "V-1", "V-2", "V-3", "V-4", "V-5"]
    row_card = [0, 0, 1, 1, 2, 3, 3, 4, 5, 5]
    identities = {"V-1": "same", "V-2": "same"}
    images = {"V-0": "a" * 16, "V-1": "b" * 16, "V-2": "c" * 16, "V-3": "d" * 16, "V-4": "e" * 16, "V-5": "e" * 16}
    stream = Stream(7)
    pairs = []
    for _ in range(80):
        best = stream.below(len(row_card))
        second = stream.below(len(row_card))
        d_best = stream.below(40)
        d_second = d_best + [0, 0, 1, 2, 5, 9, 20][stream.below(7)]
        pairs.append([(best, d_best), (second, d_second)])
    pairs.append([(3, 5)])  # a query with one neighbour: no ratio test
    pairs.append([(6, 0), (2, 0)])  # equal distances between rivals: refused
    pairs.append([(0, 10), (1, 10)])  # equal distances within one card: counted
    stage = GeometricStage.__new__(GeometricStage)
    stage.config = parse_geometric({"vote": {"ratio": 0.8}})
    stage.index = SimpleNamespace(ids=ids, feature_count=lambda: len(row_card))
    stage._identity = [identities.get(card_id, card_id) for card_id in ids]
    stage._image = [images[card_id] for card_id in ids]
    stage._vote_matcher = _Matcher(pairs)
    stage._vote_card = np.array(row_card)
    stage._vote_starts = np.array([0])
    counts = stage.votes(_Sized(len(pairs)))
    return {"ids": ids, "rowCard": row_card, "identities": identities, "images": images, "ratio": 0.8,
            "pairs": pairs, "votes": counts}


class _Sized:
    def __init__(self, n: int) -> None:
        self.n, self.descriptors = n, None

    def __len__(self) -> int:
        return self.n


# ------------------------------------------------------------------------------------ seam
def _seam_case(reorder: bool, cosines: list[float], ids: list[str], identities: dict[str, str],
               verified: list[tuple[str, int]], rule: AcceptRule) -> dict:
    from cardvision.recognize.recognizer import Recognizer

    config = parse_geometric({**PASSED, "fusion": {**PASSED["fusion"], "reorder": reorder}})
    stage = SimpleNamespace(config=config)
    row = np.array(cosines, dtype=np.float32).astype(np.float64)
    columns: dict[str, list[int]] = {}
    for column, card_id in enumerate(ids):
        columns.setdefault(card_id, []).append(column)
    best = {card_id: float(row[cols].max()) for card_id, cols in columns.items()}
    candidates = tuple(Verified(card_id, best[card_id], "embedding", 0, inliers, inliers, float(inliers),
                                GeometricStage.fused_score(stage, best[card_id], float(inliers)))
                       for card_id, inliers in verified)
    stage.verify = lambda crop, ranked, known=None: StageResult(candidates, 0)
    recognizer = Recognizer.__new__(Recognizer)
    recognizer.ids = ids
    recognizer.rule = rule
    recognizer.column_identity = identity_columns(ids, {card_id: identities.get(card_id, card_id) for card_id in ids})
    recognizer._names = ids
    recognizer._columns = columns
    recognizer.geometric = stage
    look = score_look(row, recognizer.column_identity)
    choice = Choice(orientation=2, look=look, retried=True, tau_passed=rule.passes_tau(look), accepted=rule.accepts(look))
    result = recognizer._geometric_recognition(choice, row, np.zeros((4, 4, 3), np.uint8))
    return {"reorder": reorder, "ids": ids, "identities": identities, "cosines": [hex64(c) for c in row],
            "verified": [{"cardId": card_id, "inliers": inliers} for card_id, inliers in verified],
            "rule": {"tau": rule.tau, "delta": rule.delta},
            "expected": {"topId": result.top_id, "similarity": hex64(result.similarity), "margin": hex64(result.margin),
                         "tauPassed": result.tau_passed, "accepted": result.accepted, "look": result.look,
                         "geometric": result.geometric.to_json()}}


def section_seam() -> list[dict]:
    ids = ["S-1", "S-1A", "S-2", "S-3", "S-4", "S-2"]
    identities = {"S-1A": "S-1"}
    rule = AcceptRule(tau=0.62, delta=0.05)
    out = []
    for reorder in (True, False):
        out.append(_seam_case(reorder, [0.71, 0.70, 0.69, 0.40, 0.2, 0.55], ids, identities, [("S-2", 51), ("S-1", 14)], rule))
        out.append(_seam_case(reorder, [0.55, 0.58, 0.50, 0.49, 0.1, 0.52], ids, identities, [("S-1A", 15), ("S-3", 50)], rule))
        out.append(_seam_case(reorder, [0.30, 0.20, 0.31, 0.29, 0.28, 0.27], ids, identities, [("S-4", 200), ("S-2", 0)], rule))
        out.append(_seam_case(reorder, [0.66, 0.66, 0.60, 0.66, 0.1, 0.1], ids, identities, [("S-3", 16)], rule))
        out.append(_seam_case(reorder, [0.60, 0.59, 0.45, 0.40, 0.1, 0.3], ids, identities, [("S-1A", 20), ("S-3", 3)], rule))
    return out


# ---------------------------------------------------------------------------------- scenes
CARD_W, CARD_H = 120, 168


def card_rgb(seed: int) -> np.ndarray:
    """A toy card: a background colour and 40 rectangles of colours from a xorshift stream."""
    stream = Stream(seed)
    image = np.empty((CARD_H, CARD_W, 3), np.uint8)
    image[:, :] = [stream.below(256) for _ in range(3)]
    for _ in range(40):
        x0, y0 = stream.below(CARD_W), stream.below(CARD_H)
        w, h = 4 + stream.below(CARD_W // 3), 4 + stream.below(CARD_H // 3)
        image[y0:y0 + h, x0:x0 + w] = [stream.below(256) for _ in range(3)]
    return image


def scene_rgb(card: np.ndarray | None, size: tuple[int, int], place: tuple[int, int, int, int], seed: int,
              turned: bool) -> np.ndarray:
    """A crop of ``size`` (width, height): 8-pixel blocks of colours from a stream, and the card
    scaled by nearest neighbour to ``place`` (x, y, width, height), turned half a turn if asked."""
    width, height = size
    stream = Stream(seed)
    image = np.empty((height, width, 3), np.uint8)
    for by in range(0, height, 8):
        for bx in range(0, width, 8):
            image[by:by + 8, bx:bx + 8] = [stream.below(256) for _ in range(3)]
    if card is not None:
        ox, oy, sw, sh = place
        for y in range(max(0, oy), min(height, oy + sh)):
            for x in range(max(0, ox), min(width, ox + sw)):
                sx, sy = (x - ox) * CARD_W // sw, (y - oy) * CARD_H // sh
                if turned:
                    sx, sy = CARD_W - 1 - sx, CARD_H - 1 - sy
                image[y, x] = card[sy, sx]
    return image


SCENE_CONFIG = {"candidates": {"embeddingTopK": 3, "voteTopN": 3}, "crop": {"longSide": 160, "maxFeatures": 500},
                "features": {"artLongSide": 200, "maxFeatures": 300},
                "ransac": {"model": "affine"}, "fusion": {"weight": 1.5, "minInliers": 15, "fullInliers": 50}}
CARD_IDS = [f"TOY-{i:03d}" for i in range(1, 13)]
SCENES = [
    {"card": 0, "size": [150, 200], "place": [10, 12, 130, 182], "seed": 101, "turned": False},
    {"card": 3, "size": [260, 340], "place": [8, 6, 240, 320], "seed": 102, "turned": False},
    {"card": 5, "size": [100, 130], "place": [4, 4, 90, 120], "seed": 103, "turned": True},
    {"card": 7, "size": [114, 160], "place": [0, 0, 114, 160], "seed": 104, "turned": False},
    {"card": None, "size": [140, 190], "place": [0, 0, 0, 0], "seed": 105, "turned": False},
    {"card": 10, "size": [180, 220], "place": [20, 10, 150, 200], "seed": 106, "turned": True},
    {"card": 1, "size": [200, 150], "place": [40, 0, 107, 150], "seed": 107, "turned": False},
]


def section_scenes() -> dict | None:
    try:
        import cv2
    except ImportError:
        return None
    from cardvision.geometric.features import extract
    from cardvision.geometric.index import DescriptorIndex

    config = parse_geometric(SCENE_CONFIG)
    features = config.features
    cards, art_versions, packed = {}, {}, []
    for position, card_id in enumerate(CARD_IDS):
        found = extract(card_rgb(1000 + position), "orb", features["artLongSide"], features["maxFeatures"],
                        features["params"])
        cards[card_id] = found
        art_versions[card_id] = f"{position:016x}" if position != 11 else f"{10:016x}"  # 011 and 012: one image
        packed.append({"cardId": card_id, "artV": art_versions[card_id], "width": found.size[0],
                       "height": found.size[1], "n": len(found), "xy": base64.b64encode(found.xy.astype("<f4").tobytes()).decode(),
                       "desc": base64.b64encode(found.descriptors.tobytes()).decode()})
    index = DescriptorIndex(ids=list(CARD_IDS), cards=cards, features=dict(features), doc={},
                            art_versions=art_versions)
    identities = {"TOY-002": "group:TOY-002", "TOY-003": "group:TOY-002"}
    stage = GeometricStage(config, index, identities)
    known = [card_id for card_id in CARD_IDS if card_id != "TOY-010"]
    stream = Stream(77)
    scenes = []
    for scene in SCENES:
        card = None if scene["card"] is None else card_rgb(1000 + scene["card"])
        crop = scene_rgb(card, tuple(scene["size"]), tuple(scene["place"]), scene["seed"], scene["turned"])
        ranked = [[card_id, stream.unit()] for card_id in CARD_IDS]
        ranked.append([CARD_IDS[0], 0.99])  # a card given twice keeps its first pair
        result = stage.verify(crop, [tuple(pair) for pair in ranked], known=set(known))
        crop_features = stage.crop_features(crop)
        scenes.append({**scene, "ranked": ranked, "cropKeypoints": result.crop_keypoints,
                       "cropSize": list(crop_features.size),
                       "cropXy": base64.b64encode(crop_features.xy.astype("<f4").tobytes()).decode(),
                       "cropDesc": base64.b64encode(crop_features.descriptors.tobytes()).decode(),
                       "candidates": [{"cardId": v.card_id, "cosine": hex64(v.cosine), "foundBy": v.found_by,
                                       "votes": v.votes, "matches": v.matches, "inliers": v.inliers,
                                       "evidence": hex64(v.evidence), "fused": hex64(v.fused), "indexed": v.indexed}
                                      for v in result.candidates]})
    return {"opencv": cv2.__version__, "config": SCENE_CONFIG, "settings": config.to_json(), "identities": identities,
            "known": known, "cards": packed, "scenes": scenes}


def build() -> dict:
    return {"config": section_config(), "round": section_round(), "fusion": section_fusion(), "vote": section_vote(),
            "seam": section_seam(), "scenes": section_scenes()}


def write() -> None:
    FIXTURE.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {FIXTURE.relative_to(ROOT)}")


# ---------------------------------------------------------------------------------- checks
@pytest.mark.parametrize("section", ["config", "round", "fusion", "vote", "seam"])
def test_the_fixture_holds_what_the_python_code_computes_now(section):
    stored = json.loads(FIXTURE.read_text(encoding="utf-8"))
    current = json.loads(json.dumps(build_section(section)))
    assert stored[section] == current, "run python tests/test_browser_geometric_fixtures.py --write"


def test_the_scenes_hold_what_this_opencv_computes_now():
    cv2 = pytest.importorskip("cv2")
    stored = json.loads(FIXTURE.read_text(encoding="utf-8"))["scenes"]
    if stored is None or stored["opencv"] != cv2.__version__:
        pytest.skip(f"the scenes were written with OpenCV {stored and stored['opencv']}; keypoints differ between builds")
    assert stored == json.loads(json.dumps(section_scenes()))


def build_section(section: str):
    return {"config": section_config, "round": section_round, "fusion": section_fusion, "vote": section_vote,
            "seam": section_seam}[section]()


if __name__ == "__main__":
    if sys.argv[1:] == ["--write"]:
        write()
    else:
        print(__doc__)
