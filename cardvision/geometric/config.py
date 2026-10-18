# SPDX-License-Identifier: Apache-2.0
"""The geometric stage's settings: one JSON file, every setting in it.

Nothing about the stage is fixed in code. The file names the descriptor index and its SHA-256,
and holds seven blocks; a block or a key left out takes the default below, and the loaded
settings (``GeometricConfig.to_json``) are always written out in full, so a report states
exactly what ran. Any key not listed here is an error, so that a misspelled setting cannot be
silently ignored::

    {
     "index": {"path": "bank.features.json", "sha256": "<64 hex digits>"},
     "seed": 0,
     "features": {"type": "orb", "maxFeatures": 300, "artLongSide": 320, "mask": [],
                  "params": {}},
     "crop": {"longSide": 320, "maxFeatures": 500},
     "candidates": {"embeddingTopK": 5, "voteTopN": 0},
     "vote": {"matcher": "bf", "ratio": 0.8, "lshTables": 12, "lshKeySize": 20, "lshProbes": 2,
              "kdTrees": 4, "checks": 64},
     "match": {"ratio": 0.8},
     "ransac": {"model": "homography", "method": "ransac", "threshold": 6.0, "maxIters": 2000,
                "confidence": 0.995, "minAreaRatio": 0.25, "maxAreaRatio": 4.0},
     "fusion": {"weight": 0.1, "minInliers": 8, "fullInliers": 30, "penalty": 0.0,
                "baseline": "none", "reorder": true},
     "gate": {"belowAccept": false, "sources": null}
    }

**features** describe how keypoints are taken from the card images, and so belong to the
index as much as to the stage: the index records the block it was built with, and the stage
refuses an index whose block differs. ``type`` is ``orb``, ``akaze`` or ``sift`` (all from
OpenCV; ``akaze`` only where the installed OpenCV has it: OpenCV 5 moved it out of the main
package). ``params`` are the detector's own settings, by type (``FEATURE_PARAMS``). ``mask``
lists rectangles ``[x0, y0, x1, y1]``, as fractions of a card image's width and height, in
which no keypoint is taken from the card images: for a band printed over every image (a
watermark) that a card on a table does not have.

**crop**: the crop is scaled so that its longer side is ``longSide`` pixels, and at most
``maxFeatures`` keypoints are taken from it.

**candidates**: which cards are verified for a crop: the recognizer's ``embeddingTopK`` best
cards (card ids, each by its best bank row: printings of one name count one each, even under
the default rule that makes them one identity), together with the ``voteTopN`` cards that collect the most votes when every
keypoint of the crop votes for the card of its nearest descriptor in the whole index
(``vote``: ``matcher`` ``bf``, exhaustive, or ``flann``, approximate: OpenCV's FLANN with
LSH tables for binary descriptors, ``lshTables`` tables of ``lshKeySize`` bits probed at
``lshProbes`` levels, or ``kdTrees`` randomized k-d trees for SIFT's, searched with ``checks``
leaf checks; ``ratio``, the nearest-neighbour ratio a vote must pass, unless the two nearest
descriptors belong to one card, to cards of one identity, or to cards taken from one image).
``voteTopN`` 0 turns the vote off. The vote compares every crop keypoint with every keypoint of
the catalog, so its cost grows with the catalog: it is the expensive option.

**match** and **ransac**: each candidate's descriptors are matched with the crop's (``ratio``
test), and a model is fitted to the matched points by OpenCV: ``homography``
(``findHomography``), ``affine`` (``estimateAffine2D``) or ``similarity``
(``estimateAffinePartial2D``), with ``method`` ``ransac`` or ``usac`` (MAGSAC++, not for
``similarity``), the reprojection ``threshold`` in pixels of the scaled crop, ``maxIters`` and
``confidence``. A fitted model that maps the card image onto an area of the crop outside
``minAreaRatio`` to ``maxAreaRatio`` times the crop's area, or onto a shape that is not convex,
counts as no fit (a ``maxAreaRatio`` of 0 skips the check). The matches the model explains are
the candidate's inliers.

**fusion**: a verified candidate's *evidence* is its inlier count, less the median inlier
count of the crop's verified candidates when ``baseline`` is ``median`` (cards of one game
share a frame, a title bar and printed symbols, which give every candidate some inliers; the
median takes that common share out), and never below 0. Its fused score is its cosine plus
``weight`` times ``min(evidence, fullInliers) / fullInliers`` when the evidence is at least
``minInliers``, and its cosine minus ``penalty`` when it is less. A card that was not verified
keeps its cosine.
With ``reorder`` true the best fused score names the crop (a *re-ranker*); with it false the
identity of the recognizer's best card stays the answer, named by that identity's best fused
printing, and only its score and margin are fused (an *accept booster*). Either way the accept
rule then judges the fused score and margin.

**gate** says where the stage runs at all; both are off by default. With ``belowAccept`` true
it runs only on crops whose embedding look the accept rule did not accept (the score under
``tau`` or the margin under ``delta``): an accepted crop keeps the embedding's answer exactly.
``sources`` is a list of source types or ``fnmatch`` patterns (``["physical/*"]``): the stage
runs only on crops whose source type matches one. The source type is a manifest row's
``source_type`` (``generated`` for scenes from ``make-composites``); where it is not known,
as for ``cardvision infer`` on a file, the stage runs only when there is no ``sources`` list.
A crop the stage did not run on is reported as if there were no stage: no ``geometric`` field.

**The accept rule changes meaning.** ``tau`` and ``delta`` were calibrated on cosines. A fused
score is not a cosine (it can exceed 1), so with the stage on they no longer mean what
``cardvision calibrate`` measured, and in practice ``minInliers`` with ``weight`` becomes the
accept rule: a candidate with enough inliers is lifted over ``tau`` almost whatever its cosine.
Choose the fusion settings on data you do not test on, under a bar on wrong names written down
first and measured on real negatives (pointer positions off the cards, card backs), and check
them with ``cardvision evaluate compare --other-geometric``. ``cardvision calibrate`` does not
run the stage and says nothing about it.

**seed** seeds OpenCV's random number generator before each crop and before the approximate
matcher of the vote is trained (FLANN draws its LSH tables from it), so the stage gives the same
answer for the same crop, settings, index and OpenCV version. OpenCV's RANSAC fits draw from a
generator of their own with a fixed seed, so they repeat whatever this seed is.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FEATURE_TYPES = ("orb", "akaze", "sift")
BASELINES = ("none", "median")
MATCHERS = ("bf", "flann")
MODELS = ("homography", "affine", "similarity")
METHODS = ("ransac", "usac")

# The detector settings each feature type takes, with their defaults (OpenCV's names, in
# camel case). Anything else under "params" is an error.
FEATURE_PARAMS: dict[str, dict[str, Any]] = {
    "orb": {"scaleFactor": 1.2, "nLevels": 8, "edgeThreshold": 15, "patchSize": 15, "fastThreshold": 10},
    "akaze": {"threshold": 0.001, "nOctaves": 4, "nOctaveLayers": 4},
    "sift": {"nOctaveLayers": 3, "contrastThreshold": 0.04, "edgeThreshold": 10.0, "sigma": 1.6},
}

DEFAULTS: dict[str, Any] = {
    "seed": 0,
    "features": {"type": "orb", "maxFeatures": 300, "artLongSide": 320, "mask": [], "params": {}},
    "crop": {"longSide": 320, "maxFeatures": 500},
    "candidates": {"embeddingTopK": 5, "voteTopN": 0},
    "vote": {"matcher": "bf", "ratio": 0.8, "lshTables": 12, "lshKeySize": 20, "lshProbes": 2, "kdTrees": 4,
             "checks": 64},
    "match": {"ratio": 0.8},
    "ransac": {"model": "homography", "method": "ransac", "threshold": 6.0, "maxIters": 2000, "confidence": 0.995,
               "minAreaRatio": 0.25, "maxAreaRatio": 4.0},
    "fusion": {"weight": 0.1, "minInliers": 8, "fullInliers": 30, "penalty": 0.0, "baseline": "none",
               "reorder": True},
    "gate": {"belowAccept": False, "sources": None},
}
_BLOCKS = ("features", "crop", "candidates", "vote", "match", "ransac", "fusion", "gate")
_TOP_KEYS = ("index", "seed", *_BLOCKS)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class GeometricConfigError(ValueError):
    """A geometric settings file is not valid. ``errors`` lists every reason."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def _int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True)
class GeometricConfig:
    """Loaded settings. ``settings`` is the full JSON object, defaults filled in, without the
    index; ``index_path`` and ``index_sha256`` are where the index is and what its bytes must
    hash to (``None`` when the file does not say)."""

    settings: Mapping[str, Any]
    index_path: Path | None = None
    index_sha256: str | None = None

    def __getitem__(self, block: str) -> Any:
        return self.settings[block]

    @property
    def seed(self) -> int:
        return int(self.settings["seed"])

    @property
    def features(self) -> dict[str, Any]:
        return self.settings["features"]

    def to_json(self) -> dict[str, Any]:
        """The full settings, as a report records them (the index is reported by its hash)."""
        return copy.deepcopy(dict(self.settings))


def _merge(doc: Mapping[str, Any], errors: list[str]) -> dict[str, Any]:
    settings = copy.deepcopy(DEFAULTS)
    if "seed" in doc:
        settings["seed"] = doc["seed"]
    for block in _BLOCKS:
        given = doc.get(block, {})
        if not isinstance(given, Mapping):
            errors.append(f"{block} must be an object")
            continue
        unknown = sorted(set(given) - set(DEFAULTS[block]))
        if unknown:
            errors.append(f"{block} has unknown key(s) {unknown} (expected {sorted(DEFAULTS[block])})")
        settings[block].update({key: copy.deepcopy(value) for key, value in given.items() if key in DEFAULTS[block]})
    return settings


def _check_features(features: Mapping[str, Any], errors: list[str]) -> None:
    kind = features["type"]
    if kind not in FEATURE_TYPES:
        errors.append(f"features.type must be one of {list(FEATURE_TYPES)}, got {kind!r}")
        return
    for key in ("maxFeatures", "artLongSide"):
        if not _int(features[key]) or features[key] < 1:
            errors.append(f"features.{key} must be a whole number of at least 1")
    mask = features["mask"]
    if not isinstance(mask, list) or not all(
            isinstance(rect, list) and len(rect) == 4 and all(_number(v) and 0 <= v <= 1 for v in rect)
            and rect[0] < rect[2] and rect[1] < rect[3] for rect in mask):
        errors.append("features.mask must be a list of [x0, y0, x1, y1] fractions from 0 to 1, x0 < x1 and y0 < y1")
    params = features["params"]
    if not isinstance(params, Mapping):
        errors.append("features.params must be an object")
        return
    unknown = sorted(set(params) - set(FEATURE_PARAMS[kind]))
    if unknown:
        errors.append(f"features.params has unknown key(s) {unknown} for {kind} (expected {sorted(FEATURE_PARAMS[kind])})")
    full = {**FEATURE_PARAMS[kind], **{k: v for k, v in params.items() if k in FEATURE_PARAMS[kind]}}
    for key, default in FEATURE_PARAMS[kind].items():
        if (_int(default) and not _int(full[key])) or not _number(full[key]) or full[key] <= 0:
            errors.append(f"features.params.{key} must be a {'whole ' if _int(default) else ''}number above 0")
    features["params"] = full


def _check(settings: dict[str, Any], errors: list[str]) -> None:
    if not _int(settings["seed"]) or settings["seed"] < 0:
        errors.append("seed must be a whole number of 0 or more")
    _check_features(settings["features"], errors)
    crop, candidates, vote = settings["crop"], settings["candidates"], settings["vote"]
    match, ransac, fusion = settings["match"], settings["ransac"], settings["fusion"]
    for where, block, key, low in (("crop", crop, "longSide", 1), ("crop", crop, "maxFeatures", 1),
                                   ("candidates", candidates, "embeddingTopK", 0),
                                   ("candidates", candidates, "voteTopN", 0), ("ransac", ransac, "maxIters", 1),
                                   ("vote", vote, "lshTables", 1), ("vote", vote, "lshKeySize", 1),
                                   ("vote", vote, "lshProbes", 0), ("vote", vote, "kdTrees", 1),
                                   ("vote", vote, "checks", 1),
                                   ("fusion", fusion, "minInliers", 0), ("fusion", fusion, "fullInliers", 1)):
        if not _int(block[key]) or block[key] < low:
            errors.append(f"{where}.{key} must be a whole number of at least {low}")
    if _int(candidates["embeddingTopK"]) and _int(candidates["voteTopN"]) \
            and candidates["embeddingTopK"] + candidates["voteTopN"] == 0:
        errors.append("candidates: embeddingTopK and voteTopN cannot both be 0")
    if vote["matcher"] not in MATCHERS:
        errors.append(f"vote.matcher must be one of {list(MATCHERS)}")
    for where, block in (("vote", vote), ("match", match)):
        if not _number(block["ratio"]) or not 0 < block["ratio"] <= 1:
            errors.append(f"{where}.ratio must be a number above 0 and at most 1")
    if ransac["model"] not in MODELS:
        errors.append(f"ransac.model must be one of {list(MODELS)}")
    if ransac["method"] not in METHODS:
        errors.append(f"ransac.method must be one of {list(METHODS)}")
    elif ransac["method"] == "usac" and ransac["model"] == "similarity":
        errors.append('ransac.method "usac" is not available for the "similarity" model in OpenCV; use "ransac"')
    if not _number(ransac["threshold"]) or ransac["threshold"] <= 0:
        errors.append("ransac.threshold must be a number above 0")
    if not _number(ransac["confidence"]) or not 0 < ransac["confidence"] < 1:
        errors.append("ransac.confidence must be a number above 0 and below 1")
    low, high = ransac["minAreaRatio"], ransac["maxAreaRatio"]
    if not _number(low) or not _number(high) or low < 0 or high < 0 or (high and low > high):
        errors.append("ransac.minAreaRatio and maxAreaRatio must be numbers of 0 or more, the first at most the second")
    for key in ("weight", "penalty"):
        if not _number(fusion[key]) or fusion[key] < 0:
            errors.append(f"fusion.{key} must be a number of 0 or more")
    gate = settings["gate"]
    if not isinstance(gate["belowAccept"], bool):
        errors.append("gate.belowAccept must be true or false")
    sources = gate["sources"]
    if sources is not None and (not isinstance(sources, list) or not sources
                                or not all(isinstance(item, str) and item for item in sources)):
        errors.append('gate.sources must be null or a non-empty list of source types or patterns, e.g. ["physical/*"]')
    if fusion["baseline"] not in BASELINES:
        errors.append(f"fusion.baseline must be one of {list(BASELINES)}")
    if not isinstance(fusion["reorder"], bool):
        errors.append("fusion.reorder must be true or false")


def parse_geometric(doc: Any, base: str | Path | None = None, source: str = "geometric settings") -> GeometricConfig:
    """Read settings from a parsed document. ``base`` is the folder a relative index path is
    resolved against. Raises ``GeometricConfigError`` listing every problem."""
    if not isinstance(doc, Mapping):
        raise GeometricConfigError([f"{source}: must be a JSON object"])
    errors: list[str] = []
    unknown = sorted(set(doc) - set(_TOP_KEYS))
    if unknown:
        errors.append(f"unknown key(s) {unknown} (expected {list(_TOP_KEYS)})")
    index = doc.get("index", {})
    path = sha = None
    if not isinstance(index, Mapping) or set(index) - {"path", "sha256"}:
        errors.append('index must be an object with the keys "path" and "sha256"')
    else:
        if index.get("path") is not None:
            if not isinstance(index["path"], str) or not index["path"]:
                errors.append("index.path must be a file path")
            else:
                path = Path(index["path"]) if base is None else Path(base) / index["path"]
        if index.get("sha256") is not None:
            if not isinstance(index["sha256"], str) or not _SHA256.match(index["sha256"]):
                errors.append("index.sha256 must be 64 lowercase hex digits")
            else:
                sha = index["sha256"]
    settings = _merge(doc, errors)
    _check(settings, errors)
    if errors:
        raise GeometricConfigError([f"{source}: {error}" for error in errors])
    return GeometricConfig(settings=settings, index_path=path, index_sha256=sha)


def load_geometric(path: str | Path) -> GeometricConfig:
    """Read a settings file; a relative index path is resolved against the file's folder."""
    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path.name}: not valid JSON: {error}") from None
    return parse_geometric(doc, base=path.parent, source=path.name)
