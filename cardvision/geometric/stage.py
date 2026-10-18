# SPDX-License-Identifier: Apache-2.0
"""Verify a crop's candidate cards by fitting a geometric model to matched local features.

``GeometricStage.verify`` takes one crop, the recognizer's ranked candidates with their
cosines, and returns every verified candidate with its evidence and its fused score:

1. keypoints are taken from the crop, scaled to ``crop.longSide`` (``features``);
2. the candidates are the ``embeddingTopK`` best of the ranked cards, together with the
   ``voteTopN`` cards whose index descriptors are nearest to the most crop descriptors;
3. each candidate's descriptors are matched with the crop's (ratio test), a model is fitted to
   the matched points with RANSAC, and the inliers are counted;
4. the inlier count is fused with the cosine (``fused_score``).

The settings are ``cardvision.geometric.config``'s. OpenCV's random number generator is seeded
with ``seed`` before each crop and before the vote's approximate matcher is trained, so the same
crop, settings, index and OpenCV version give the same result (RANSAC uses OpenCV's own
fixed-seed generator).

The stage sees one look of the crop: the one the recognizer's orientation rule chose on the
cosines. Local features do not care how the crop is turned, so the stage needs no turned look,
but it cannot change which look's ranking it is given.

The stage does not decide anything itself. ``cardvision.recognize.Recognizer`` holds one when
asked to, and turns the fused scores into its answer under the accept rule.
"""

from __future__ import annotations

import fnmatch
import hashlib
import time
from collections.abc import Container, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cardvision.geometric.config import GeometricConfig, load_geometric
from cardvision.geometric.features import Features, cv2_module, extract, extract_grey, is_binary, norm, scaled_grey
from cardvision.geometric.index import DescriptorIndex, bank_mismatch, load_index, same_features

EMBEDDING, VOTE, BOTH = "embedding", "vote", "both"
# The most descriptor rows given to an OpenCV matcher as one training image (its limit is 2**18).
VOTE_CHUNK = 1 << 17


@dataclass(frozen=True)
class Verified:
    """One verified candidate.

    Attributes:
        card_id: the card.
        cosine: its cosine as the caller gave it; for a card the vote added that the caller did
            not rank, the lowest cosine the caller gave.
        found_by: ``embedding``, ``vote`` or ``both``.
        votes: crop keypoints that voted for it (0 when the vote is off).
        matches: matched keypoints that passed the ratio test.
        inliers: matches the fitted model explains (0 with no fit).
        evidence: the inliers less the baseline (``fusion.baseline``), at least 0.
        fused: the fused score.
        indexed: whether the index has the card; a card it lacks keeps its cosine.
    """

    card_id: str
    cosine: float
    found_by: str
    votes: int
    matches: int
    inliers: int
    evidence: float
    fused: float
    indexed: bool = True

    def to_json(self) -> dict[str, Any]:
        return {"cardId": self.card_id, "cosine": round(self.cosine, 6), "foundBy": self.found_by,
                "votes": self.votes, "matches": self.matches, "inliers": self.inliers,
                "evidence": round(self.evidence, 6), "fused": round(self.fused, 6)}


@dataclass(frozen=True)
class StageResult:
    """What the stage found for one crop: the verified candidates in the order they were
    chosen (embedding rank, then vote rank), the crop's keypoint count, and the time taken.
    ``debug`` is the stage's record of its inputs when it was asked for one (``debug_record``),
    else ``None``; it is not part of the result's identity."""

    candidates: tuple[Verified, ...]
    crop_keypoints: int
    ms: float = 0.0
    debug: dict[str, Any] | None = field(default=None, compare=False)

    def fused(self) -> dict[str, float]:
        """Card id to fused score, for the verified candidates."""
        return {found.card_id: found.fused for found in self.candidates}


@dataclass(frozen=True)
class GeometricOutcome:
    """The stage's part in one ``Recognition``: what the recognizer would have answered without
    it (``top_id``, ``similarity``, ``margin``, ``accepted``) and the verified candidates."""

    top_id: str
    similarity: float
    margin: float
    accepted: bool
    result: StageResult

    def to_json(self) -> dict[str, Any]:
        out = {"before": {"topId": self.top_id, "similarity": round(self.similarity, 6),
                          "margin": round(self.margin, 6), "accepted": self.accepted},
               "candidates": [found.to_json() for found in self.result.candidates]}
        if self.result.debug is not None:
            out["debug"] = self.result.debug
        return out


# A keypoint row's id in a debug record: this many hexadecimal digits of its SHA-256.
ROW_HASH_DIGITS = 16


def row_hashes(found: Features) -> list[str]:
    """One id per keypoint row of ``found``, in its order: the SHA-256 of the row's position as two
    little-endian float32 and its descriptor bytes, to ``ROW_HASH_DIGITS`` hexadecimal digits.
    Two builds of OpenCV took the same keypoint exactly when the ids are equal."""
    xy = np.ascontiguousarray(found.xy, dtype="<f4")
    descriptors = np.ascontiguousarray(found.descriptors)
    return [hashlib.sha256(xy[i].tobytes() + descriptors[i].tobytes()).hexdigest()[:ROW_HASH_DIGITS]
            for i in range(len(found))]


def resize_direction(width: int, height: int, long_side: int) -> str:
    """How ``scaled_grey`` brings a ``width`` x ``height`` image to ``long_side``: ``up``
    (bilinear), ``down`` (area averaging) or ``none``."""
    scale = long_side / max(height, width)
    return "none" if scale == 1 else "down" if scale < 1 else "up"


def debug_record(crop: np.ndarray, grey: np.ndarray, found: Features, votes: Mapping[str, int],
                 long_side: int) -> dict[str, Any]:
    """What a debug run records about one crop: the post-resize grey image the keypoints were
    taken from (its SHA-256 and size), the resize direction, the source crop's size, every
    keypoint row's id (``row_hashes``) and the whole vote, card id to count, before any card is
    left out of it. The browser runtime records the same fields, so the two can be compared
    crop by crop."""
    height, width = crop.shape[:2]
    return {"grey": {"sha256": hashlib.sha256(np.ascontiguousarray(grey).tobytes()).hexdigest(),
                     "width": int(grey.shape[1]), "height": int(grey.shape[0])},
            "resize": resize_direction(width, height, long_side), "source": [int(width), int(height)],
            "keypoints": len(found), "rows": row_hashes(found),
            "votes": {card_id: int(votes[card_id]) for card_id in sorted(votes)}}


def _corners(size: tuple[int, int]) -> np.ndarray:
    width, height = size
    return np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)


class GeometricStage:
    """The settings, the index, and the matchers built from them."""

    def __init__(self, config: GeometricConfig, index: DescriptorIndex,
                 identities: Mapping[str, str] | None = None, debug: bool = False) -> None:
        differ = same_features(config.features, index.features)
        if differ:
            raise ValueError(f"the index was built with other features settings (differing: {', '.join(differ)}); "
                             "build it again with cardvision build-features")
        self.config = config
        self.index = index
        # With debug on, each StageResult carries debug_record's account of the crop. It decides
        # nothing: the keypoints, the vote and every fit are the same with it on and off.
        self.debug = debug
        self.kind = index.kind
        self._norm = norm(self.kind)
        self._vote_matcher: Any = None
        self._vote_card: np.ndarray | None = None
        self._vote_starts: np.ndarray | None = None
        self._position = {card_id: position for position, card_id in enumerate(index.ids)}
        # Two index cards whose descriptors may both be a crop keypoint's nearest without
        # counting as rivals in the vote: one identity (when known), or one image (artV).
        identities = identities or {}
        self._identity = [identities.get(card_id, card_id) for card_id in index.ids]
        self._image = [index.art_versions.get(card_id, card_id) for card_id in index.ids]

    @classmethod
    def open(cls, path: str | Path, bank: str | Path | None = None,
             identities: Mapping[str, str] | None = None) -> GeometricStage:
        """Load a settings file and the index it names, after checking the index's SHA-256.

        ``bank`` is the bank file the stage will serve: an index that does not serve it is
        refused (``cardvision.geometric.index.bank_mismatch``). ``identities`` maps card id to
        identity key (``Catalog.identities()``), for the vote's ratio test."""
        from cardvision.evaluate.config import file_sha256

        config = load_geometric(path)
        if config.index_path is None or config.index_sha256 is None:
            raise ValueError(f"{Path(path).name}: index.path and index.sha256 are both needed "
                             "(cardvision build-features --pin writes them)")
        if not config.index_path.is_file():
            raise FileNotFoundError(f"{config.index_path}: no such descriptor index file")
        actual = file_sha256(config.index_path)
        if actual != config.index_sha256:
            raise ValueError(f"{config.index_path.name}: its SHA-256 is {actual}, not the {config.index_sha256} "
                             f"that {Path(path).name} pins")
        index = load_index(config.index_path)
        if bank is not None:
            problem = bank_mismatch(index, bank)
            if problem is not None:
                raise ValueError(f"{config.index_path.name}: {problem}")
        return cls(config, index, identities)

    def applies(self, source_type: str | None, accepted: bool) -> bool:
        """Whether the stage runs on a crop from ``source_type`` (``None`` when unknown) whose
        embedding look the accept rule ``accepted`` or not (the ``gate`` settings)."""
        gate = self.config["gate"]
        if gate["belowAccept"] and accepted:
            return False
        sources = gate["sources"]
        if sources is None:
            return True
        return source_type is not None and any(fnmatch.fnmatchcase(source_type, pattern) for pattern in sources)

    def opencv_note(self) -> str | None:
        """A warning when the index's keypoints were taken by another OpenCV version than the
        one running now (a version may take slightly different keypoints), else ``None``."""
        built, running = self.index.doc.get("opencv"), cv2_module().__version__
        if built == running:
            return None
        return (f"the descriptor index was built with OpenCV {built or 'of an unrecorded version'} and OpenCV "
                f"{running} is running; keypoints may differ slightly from the index's")

    # ----------------------------------------------------------------------------- scoring
    def fused_score(self, cosine: float, evidence: float) -> float:
        """A verified candidate's fused score from its evidence (see ``cardvision.geometric.config``)."""
        fusion = self.config["fusion"]
        if evidence >= fusion["minInliers"]:
            return cosine + fusion["weight"] * min(evidence, fusion["fullInliers"]) / fusion["fullInliers"]
        return cosine - fusion["penalty"]

    def crop_features(self, crop: np.ndarray) -> Features:
        features, crop_settings = self.config.features, self.config["crop"]
        return extract(crop, self.kind, crop_settings["longSide"], crop_settings["maxFeatures"], features["params"])

    def _matcher(self) -> Any:
        return cv2_module().BFMatcher(self._norm, crossCheck=False)

    def _build_vote(self) -> None:
        """The matcher over every index descriptor. The descriptors are added in chunks of at most
        ``VOTE_CHUNK`` rows, because OpenCV's matchers limit the rows of one training image; a
        match's chunk (``imgIdx``) and row (``trainIdx``) give its position in the whole index."""
        cv2 = cv2_module()
        vote = self.config["vote"]
        descriptors = [self.index.cards[card_id].descriptors for card_id in self.index.ids]
        position = np.concatenate([np.full(len(d), i, dtype=np.int64) for i, d in enumerate(descriptors)] or
                                  [np.zeros(0, np.int64)])
        stacked = np.concatenate(descriptors) if descriptors else np.zeros((0, 1), np.uint8)
        if vote["matcher"] == "bf":
            matcher = cv2.BFMatcher(self._norm, crossCheck=False)
        elif is_binary(self.kind):
            matcher = cv2.FlannBasedMatcher(
                {"algorithm": 6, "table_number": vote["lshTables"], "key_size": vote["lshKeySize"],
                 "multi_probe_level": vote["lshProbes"]}, {"checks": vote["checks"]})
        else:
            matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": vote["kdTrees"]}, {"checks": vote["checks"]})
        starts = list(range(0, len(stacked), VOTE_CHUNK)) or [0]
        matcher.add([np.ascontiguousarray(stacked[start:start + VOTE_CHUNK]) for start in starts])
        cv2.setRNGSeed(self.config.seed)  # the approximate matcher's tables are drawn when it is trained
        matcher.train()
        self._vote_matcher, self._vote_card, self._vote_starts = matcher, position, np.array(starts, dtype=np.int64)

    def votes(self, crop: Features) -> dict[str, int]:
        """Card id to the number of crop keypoints whose nearest index descriptor is that
        card's and that pass the ratio test. The test is waived when the two nearest
        descriptors belong to one card, to two cards of one identity, or to two cards taken from
        one image: such neighbours are not rivals."""
        if len(crop) < 1 or self.index.feature_count() == 0:
            return {}
        if self._vote_matcher is None:
            self._build_vote()
        ratio = self.config["vote"]["ratio"]
        counts: dict[int, int] = {}

        def card_of(match: Any) -> int:
            return int(self._vote_card[self._vote_starts[match.imgIdx] + match.trainIdx])

        for pair in self._vote_matcher.knnMatch(crop.descriptors, k=2):
            if not pair:
                continue
            best = pair[0]
            position = card_of(best)
            if len(pair) > 1:
                second = pair[1]
                other = card_of(second)
                rivals = (other != position and self._identity[other] != self._identity[position]
                          and self._image[other] != self._image[position])
                if rivals and best.distance >= ratio * second.distance:
                    continue
            counts[position] = counts.get(position, 0) + 1
        return {self.index.ids[position]: count for position, count in counts.items()}

    def _fit(self, src: np.ndarray, dst: np.ndarray, crop_size: tuple[int, int], card_size: tuple[int, int]) -> int:
        """Inliers of the model fitted to matched points ``src`` (card) -> ``dst`` (crop)."""
        cv2 = cv2_module()
        ransac = self.config["ransac"]
        model = ransac["model"]
        needed = {"homography": 4, "affine": 3, "similarity": 2}[model]
        if len(src) < needed:
            return 0
        method = cv2.RANSAC if ransac["method"] == "ransac" else cv2.USAC_MAGSAC
        threshold, iters, confidence = float(ransac["threshold"]), int(ransac["maxIters"]), float(ransac["confidence"])
        if model == "homography":
            matrix, mask = cv2.findHomography(src, dst, method, threshold, maxIters=iters, confidence=confidence)
        elif model == "affine":
            matrix, mask = cv2.estimateAffine2D(src, dst, method=method, ransacReprojThreshold=threshold,
                                                maxIters=iters, confidence=confidence)
        else:
            matrix, mask = cv2.estimateAffinePartial2D(src, dst, method=method, ransacReprojThreshold=threshold,
                                                       maxIters=iters, confidence=confidence)
        if matrix is None or mask is None:
            return 0
        if ransac["maxAreaRatio"] > 0 and not self._plausible(matrix, crop_size, card_size):
            return 0
        return int(np.count_nonzero(mask))

    def _plausible(self, matrix: np.ndarray, crop_size: tuple[int, int], card_size: tuple[int, int]) -> bool:
        """Whether the model maps the card image onto a convex shape of a plausible area."""
        cv2 = cv2_module()
        corners = _corners(card_size).reshape(-1, 1, 2)
        if matrix.shape == (3, 3):
            mapped = cv2.perspectiveTransform(corners, matrix)
        else:
            mapped = cv2.transform(corners, matrix)
        if not np.isfinite(mapped).all():
            return False
        mapped = mapped.reshape(-1, 2).astype(np.float32)
        if not cv2.isContourConvex(mapped):
            return False
        ratio = abs(cv2.contourArea(mapped)) / float(crop_size[0] * crop_size[1])
        ransac = self.config["ransac"]
        return ransac["minAreaRatio"] <= ratio <= ransac["maxAreaRatio"]

    def inliers(self, crop: Features, card_id: str) -> tuple[int, int]:
        """``(matches, inliers)`` of one card against a crop's keypoints."""
        card = self.index.cards[card_id]
        if len(crop) < 2 or len(card) < 2:
            return 0, 0
        ratio = self.config["match"]["ratio"]
        good = [pair[0] for pair in self._matcher().knnMatch(crop.descriptors, card.descriptors, k=2)
                if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]
        if not good:
            return 0, 0
        src = np.float32([card.xy[m.trainIdx] for m in good]).reshape(-1, 1, 2)
        dst = np.float32([crop.xy[m.queryIdx] for m in good]).reshape(-1, 1, 2)
        return len(good), self._fit(src, dst, crop.size, card.size)

    def verify(self, crop: np.ndarray, ranked: Sequence[tuple[str, float]],
               known: Container[str] | None = None) -> StageResult:
        """Verify one ``(H, W, 3)`` uint8 RGB crop.

        ``ranked`` is the recognizer's ``(card id, cosine)`` pairs; they are ranked here by
        cosine, best first (in the given order on a tie), and a card given twice keeps its
        first pair. Pass the recognizer's full ranking or its top few: a card the vote adds
        that ``ranked`` lacks takes the lowest cosine given. ``known``, when given, holds the
        cards the vote may add (a recognizer passes the cards its bank has, since it cannot name
        any other).
        """
        started = time.perf_counter()
        cv2 = cv2_module()
        cv2.setRNGSeed(self.config.seed)
        seen: dict[str, float] = {}
        for card_id, cosine in ranked:
            seen.setdefault(card_id, float(cosine))
        order = sorted(seen, key=lambda card_id: -seen[card_id])  # stable: ties keep the given order
        floor = min(seen.values()) if seen else 0.0
        candidates = self.config["candidates"]
        embedded = order[:candidates["embeddingTopK"]]
        if self.debug:
            grey = scaled_grey(crop, self.config["crop"]["longSide"])
            found = extract_grey(grey, self.kind, self.config["crop"]["maxFeatures"], self.config.features["params"])
        else:
            found = self.crop_features(crop)
        votes = self.votes(found) if candidates["voteTopN"] > 0 else {}
        voted = sorted((card_id for card_id in votes if known is None or card_id in known),
                       key=lambda card_id: (-votes[card_id], self._position[card_id]))
        voted = voted[:candidates["voteTopN"]]
        chosen = list(dict.fromkeys([*embedded, *voted]))
        fits = {card_id: self.inliers(found, card_id) for card_id in chosen if card_id in self.index.cards}
        baseline = 0.0
        if self.config["fusion"]["baseline"] == "median" and fits:
            baseline = float(np.median([inliers for _, inliers in fits.values()]))
        results = []
        for card_id in chosen:
            by = BOTH if card_id in embedded and card_id in voted else EMBEDDING if card_id in embedded else VOTE
            cosine = seen.get(card_id, floor)
            if card_id not in fits:
                results.append(Verified(card_id, cosine, by, 0, 0, 0, 0.0, cosine, indexed=False))
                continue
            matches, inliers = fits[card_id]
            evidence = max(0.0, inliers - baseline)
            results.append(Verified(card_id, cosine, by, votes.get(card_id, 0), matches, inliers, evidence,
                                    self.fused_score(cosine, evidence)))
        ms = round((time.perf_counter() - started) * 1000, 3)
        debug = debug_record(crop, grey, found, votes, self.config["crop"]["longSide"]) if self.debug else None
        if debug is not None:
            debug["ms"] = ms
        return StageResult(tuple(results), len(found), ms, debug)
