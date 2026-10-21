# SPDX-License-Identifier: Apache-2.0
"""Crowding: how naming degrades as the bank grows.

A bank row is computed from its own card's image alone, so one embedding of every printing
serves banks of every scope: a bank is a subset of the rows. ``embed`` computes, for each
catalog entry with art,

* its bank row, exactly as ``cardvision.bank.build`` makes one: views 0 to 15 of the image
  (``cardvision.bank.views``), preprocessed with ``full128``, embedded, scaled to unit length,
  averaged, scaled again and put through the fp16 round trip; and
* its *query views*, which no bank row has seen, in two families:

  - ``bank family``: views 16 to 19 of the same image from the bank's own synthesizer. Views 16,
    17 and 18 are downscaled to a short side of 85 to 125 pixels (``small``); view 19, like
    every fourth view, to 200 to 340 pixels (``enlarged``);
  - ``stream`` family: two views of a harsher synthesizer (``stream_view``) sized as cards are
    on a stream: a short side of 40 to 70 pixels, the box off by up to 8 % on each edge, more
    blur and stronger compression.

  Each query is embedded in its four orientations (the tensor turned, as the runtime turns
  it), so that the runtime's orientation rule (``chooseLook`` in
  ``browser/src/recognition.ts``) can be applied as it is.

``run_crowding`` (the ``crowding`` command) then scores fixed query sets against nested banks
that all hold them (``LADDERS``), so that what changes from one bank to the next is the bank
alone, under three conditions: the own row in the bank (an upper bound), the own row left out
(another scan of what else the bank holds of the card) and the open set (cards the bank does not
hold). Each is reported at the accept rule, with the runtime's rotation retry, over a grid of
``tau`` and ``delta``, and paired against the ladder's first bank (McNemar, and a bootstrap
clustered by illustration). ``run_crowding`` documents the report.

What it is not: a measure of accuracy on video. A query is a degraded copy of a catalog image,
with no detector box error, no perspective, glare, sleeve, overlay or occlusion. What the
figures show is the *change* from a small bank to a large one: crowding.

Pooling is done by ``fast_full128``, which gives the same tensor as
``cardvision.core.preprocess.full128`` bit for bit (the cell sums are sums of whole numbers,
exact in float64 in any order) in a fraction of the time.
"""

from __future__ import annotations

import io
import json
import math
import time
import zlib
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from cardvision.bank.build import decode_art, rows_from_views
from cardvision.bank.views import VIEWS_PER_CARD, synthesize_view
from cardvision.core.catalog import Catalog
from cardvision.core.preprocess import (
    FULL128_SIZE,
    _nearest_filled_index,
    normalize,
    rot4,
)

QUERY_VIEWS = (16, 17, 18, 19)
STREAM_VIEWS = 2
# The query kinds, in the order the embeddings store them: view 16, 17, 18, 19, stream 0, stream 1.
QUERY_KINDS = ("small", "small", "small", "enlarged", "stream", "stream")
STREAM_SEED = 4242
SHARD = 2000
DEFAULT_TAU = 0.62
DEFAULT_DELTA = 0.05
NO_OTHER = -2.0


# ------------------------------------------------------------------------------------------- pooling

def _pool_v1_fast(rgb: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    h, w, _ = rgb.shape
    ys = np.minimum(out_h - 1, (np.arange(h) * out_h) // h)
    xs = np.minimum(out_w - 1, (np.arange(w) * out_w) // w)
    cell = (ys[:, None] * out_w + xs[None, :]).ravel()
    flat = rgb.reshape(-1, 3)
    grid = np.stack([np.bincount(cell, weights=flat[:, c], minlength=out_h * out_w) for c in range(3)], axis=1)
    counts = np.bincount(cell, minlength=out_h * out_w)
    return (grid / np.maximum(counts, 1)[:, None]).reshape(out_h, out_w, 3)


def fast_full128(rgb: np.ndarray) -> np.ndarray:
    """``cardvision.core.preprocess.full128``, the same values, faster (bincount for the cell sums)."""
    rgb = np.ascontiguousarray(rgb)
    pooled = _pool_v1_fast(rgb, FULL128_SIZE, FULL128_SIZE)
    h, w = rgb.shape[:2]
    if h < FULL128_SIZE:
        pooled = pooled[_nearest_filled_index(h, FULL128_SIZE)]
    if w < FULL128_SIZE:
        pooled = pooled[:, _nearest_filled_index(w, FULL128_SIZE)]
    return normalize(pooled)


# ------------------------------------------------------------------------------------------- embedding

def stream_view(art: np.ndarray, card_id: str, number: int) -> np.ndarray:
    """A query view sized and damaged as a card on a stream is: the bank synthesizer's steps
    (``cardvision.bank.views``) with harsher amounts. Edge jitter up to 8 % of a side, a short
    side of 40 to 70 pixels, gamma 0.8 to 1.2, gains 0.92 to 1.08, saturation 0.8 to 1.2, blur
    0.4 to 1.2 pixels, JPEG quality 25 to 65. Seeded by the card id and ``number``, apart from
    the bank's views."""
    rng = np.random.default_rng([STREAM_SEED, zlib.crc32(card_id.encode()), number])
    art_h, art_w = art.shape[:2]
    jitter = rng.uniform(-0.08, 0.08, size=4)
    gray = int(rng.integers(20, 91))
    left, top = int(round(jitter[0] * art_w)), int(round(jitter[1] * art_h))
    right, bottom = int(round(jitter[2] * art_w)), int(round(jitter[3] * art_h))
    x0, y0, x1, y1 = left, top, art_w - right, art_h - bottom
    canvas = np.full((y1 - y0, x1 - x0, 3), gray, dtype=np.uint8)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(art_w, x1), min(art_h, y1)
    canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = art[sy0:sy1, sx0:sx1]
    short = rng.uniform(40, 70)
    h, w = canvas.shape[:2]
    k = short / min(h, w)
    image = Image.fromarray(canvas).resize((max(8, int(round(w * k))), max(8, int(round(h * k)))), Image.BILINEAR)
    x = np.asarray(image, dtype=np.float32) / 255.0
    x = np.power(x, rng.uniform(0.8, 1.2)) * rng.uniform(0.92, 1.08, size=3)[None, None, :]
    luma = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])[..., None]
    x = luma + rng.uniform(0.8, 1.2) * (x - luma)
    x = np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)
    image = Image.fromarray(x).filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.2)))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=int(rng.integers(25, 66)), subsampling=2)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return np.array(decoded.convert("RGB"))


def card_tensors(art: np.ndarray, card_id: str) -> tuple[np.ndarray, np.ndarray]:
    """The model inputs of one card: its 16 bank views ``(16, 3, S, S)``, and its query views in
    four orientations each ``(Q * 4, 3, S, S)``, in ``QUERY_KINDS`` order."""
    views = [art, *(synthesize_view(art, card_id, view) for view in range(1, VIEWS_PER_CARD))]
    queries = [synthesize_view(art, card_id, view) for view in QUERY_VIEWS]
    queries += [stream_view(art, card_id, number) for number in range(STREAM_VIEWS)]
    bank = np.stack([fast_full128(view) for view in views])
    looks = np.concatenate([rot4(fast_full128(view)) for view in queries])
    return bank, looks


def embed_card_and_queries(session: Any, art: np.ndarray, card_id: str) -> tuple[np.ndarray, np.ndarray]:
    """One card's bank row ``(D,)`` and its unit query embeddings ``(Q, 4, D)`` (Q query views,
    four orientations each; orientation 0 is upright)."""
    name = session.get_inputs()[0].name
    bank, looks = card_tensors(art, card_id)
    out = session.run(None, {name: np.concatenate([bank, looks])})[0].astype(np.float32)
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    row = rows_from_views(out[None, :VIEWS_PER_CARD])[0]
    return row, out[VIEWS_PER_CARD:].reshape(len(QUERY_KINDS), 4, -1)


_SESSION: Any = None


def _init_worker(recognizer: str) -> None:
    global _SESSION
    from cardvision.core.onnx import create_session

    _SESSION = create_session(recognizer, intra_op_threads=1)


def _embed_one(item: tuple[str, str]) -> tuple[str, np.ndarray | None, np.ndarray | None, str | None]:
    card_id, path = item
    try:
        art = decode_art(Path(path).read_bytes())
        row, queries = embed_card_and_queries(_SESSION, art, card_id)
        return card_id, row, queries, None
    except Exception as error:  # an unreadable image is reported, not fatal
        return card_id, None, None, f"{type(error).__name__}: {error}"


def embedded_ids(out_dir: Path) -> set[str]:
    done: set[str] = set()
    for path in sorted(out_dir.glob("shard-*.json")):
        done.update(json.loads(path.read_text(encoding="utf-8"))["ids"])
    return done


def embed(catalog: Catalog, art: str | Path, recognizer: str | Path, out_dir: str | Path, *,
          ids: Iterable[str] | None = None, workers: int = 4,
          progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    """Embed the entries whose image is in ``art`` and that ``out_dir`` does not hold yet.

    Writes shards of ``SHARD`` cards: ``shard-NNNNN.json`` (ids, failures) beside
    ``shard-NNNNN.rows.f32`` (``(n, D)`` float32 bank rows) and ``shard-NNNNN.queries.f32``
    (``(n, Q, 4, D)``: query views by four orientations). Running it again embeds only what is new, so it can follow a running
    image download.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = embedded_ids(out_dir)
    wanted = list(ids) if ids is not None else [card.id for card in catalog if card.has_art]
    todo = [(card_id, str(Path(art) / catalog.card(card_id).art)) for card_id in wanted
            if card_id not in done and (Path(art) / catalog.card(card_id).art).is_file()]
    shard_number = len(list(out_dir.glob("shard-*.json")))
    started = time.monotonic()
    written = 0
    failures: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(str(recognizer),)) as pool:
        for start in range(0, len(todo), SHARD):
            batch = todo[start:start + SHARD]
            results = list(pool.map(_embed_one, batch, chunksize=16))
            good = [(card_id, row, queries) for card_id, row, queries, error in results if error is None]
            failed = {card_id: error for card_id, _, _, error in results if error is not None}
            failures.update(failed)
            stem = out_dir / f"shard-{shard_number:05d}"
            if good:
                np.stack([row for _, row, _ in good]).astype(np.float32).tofile(f"{stem}.rows.f32")
                np.stack([q for _, _, q in good]).astype(np.float32).tofile(f"{stem}.queries.f32")
            meta = {"ids": [card_id for card_id, _, _ in good], "failed": failed,
                    "dim": int(good[0][1].shape[0]) if good else None, "queryKinds": list(QUERY_KINDS),
                    "orientations": 4, "recognizer": Path(recognizer).name}
            partial = Path(f"{stem}.json.partial")
            partial.write_text(json.dumps(meta) + "\n", encoding="utf-8")
            partial.replace(f"{stem}.json")
            shard_number += 1
            written += len(good)
            if progress is not None:
                progress(start + len(batch), len(todo))
    return {"embedded": written, "failed": failures, "already": len(done), "notYetFetched":
            len(wanted) - len(done) - len(todo), "seconds": round(time.monotonic() - started, 1)}


def load_embeddings(out_dir: str | Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Every embedded card: ids, bank rows ``(N, D)`` and queries ``(N, Q, 4, D)``, float32."""
    ids: list[str] = []
    rows: list[np.ndarray] = []
    queries: list[np.ndarray] = []
    for path in sorted(Path(out_dir).glob("shard-*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        if not meta["ids"]:
            continue
        n, dim, q = len(meta["ids"]), meta["dim"], len(meta["queryKinds"])
        stem = str(path)[: -len(".json")]
        rows.append(np.fromfile(f"{stem}.rows.f32", dtype=np.float32).reshape(n, dim))
        queries.append(np.fromfile(f"{stem}.queries.f32", dtype=np.float32).reshape(n, q, meta["orientations"], dim))
        ids += meta["ids"]
    if not ids:
        raise ValueError(f"no embeddings in {out_dir}")
    return ids, np.concatenate(rows), np.concatenate(queries)


# ------------------------------------------------------------------------------------------- scoring

TAU_GRID = tuple(round(0.50 + 0.02 * i, 2) for i in range(21))  # 0.50 to 0.90
DELTA_GRID = tuple(round(0.01 * i, 2) for i in range(16))  # 0.00 to 0.15
FAMILIES = {"bank": (0, 1, 2, 3), "stream": (4, 5)}


class Bank:
    """A bank: some of the embedded rows, with each row's identity as an integer.

    ``members`` are positions in the embeddings. ``identity_index`` maps an identity key (from
    the catalog) to its integer; a key of no row has none."""

    def __init__(self, members: Sequence[int], keys: Sequence[str], rows: np.ndarray) -> None:
        self.members = np.asarray(members, np.int64)
        member_keys = [keys[m] for m in self.members]
        order = sorted(range(len(member_keys)), key=lambda i: member_keys[i])
        self.members = self.members[order]  # rows sorted by identity: each identity is one slice
        member_keys = [member_keys[i] for i in order]
        self.identity_index: dict[str, int] = {}
        self.ident = np.array([self.identity_index.setdefault(k, len(self.identity_index)) for k in member_keys])
        self.starts = np.flatnonzero(np.r_[True, np.diff(self.ident) != 0])
        self.rows = np.ascontiguousarray(rows[self.members])
        self.position = {int(m): i for i, m in enumerate(self.members)}

    def __len__(self) -> int:
        return len(self.members)


def score_looks(bank: Bank, looks: np.ndarray, own: np.ndarray | None = None,
                block: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each look's best bank row (a position in ``bank``), its score, and the best score of a
    row of another identity (``NO_OTHER`` when there is none): ``topAndOther`` of
    ``browser/src/recognizer.ts``, in float32.

    ``own``, when given, holds for each look a bank position to leave out (or -1): the look's
    own row, for the leave-own-row-out test.
    """
    m = len(looks)
    best_row = np.empty(m, np.int64)
    best = np.empty(m, np.float32)
    other = np.empty(m, np.float32)
    single = len(bank.starts) == 1
    for start in range(0, m, block):
        s = looks[start:start + block] @ bank.rows.T
        if own is not None:
            mine = own[start:start + block]
            hit = np.flatnonzero(mine >= 0)
            s[hit, mine[hit]] = -np.inf
        b_row = np.argmax(s, axis=1)
        b = s[np.arange(len(s)), b_row]
        if single:
            o = np.full(len(s), NO_OTHER, np.float32)
        else:
            per_identity = np.maximum.reduceat(s, bank.starts, axis=1)
            per_identity[np.arange(len(s)), bank.ident[b_row]] = -np.inf
            o = per_identity.max(axis=1)
            o[~np.isfinite(o)] = NO_OTHER
        best_row[start:start + len(s)], best[start:start + len(s)], other[start:start + len(s)] = b_row, b, o
    return best_row, best, other


def choose_looks(best: np.ndarray, margin: np.ndarray, tau: float, delta: float,
                 retry: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """The runtime's orientation rule (``chooseLook``) over ``(n, 4)`` looks: which look each
    query settles on, and whether it names a card. Upright when it is accepted; otherwise the
    accepted look with the highest score (the lower orientation on a tie); none named when no
    look is accepted."""
    accepted = (best >= tau) & (margin >= delta)
    if not retry:
        return np.zeros(len(best), np.int64), accepted[:, 0]
    masked = np.where(accepted, best, -np.inf)
    chosen = np.where(accepted[:, 0], 0, np.argmax(masked, axis=1))
    return chosen, accepted.any(axis=1)


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """McNemar's test of paired outcomes (True = right): how often only one side was right, and
    the two-sided p-value (exact binomial up to 2,000 discordant pairs, else the normal
    approximation with continuity correction)."""
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n = only_a + only_b
    if n == 0:
        p = 1.0
    elif n <= 2000:
        k = min(only_a, only_b)
        log_terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
        top = max(log_terms)
        p = min(1.0, 2 * math.exp(top) * sum(math.exp(t - top) for t in log_terms))
    else:
        z = (abs(only_a - only_b) - 1) / math.sqrt(n)
        p = math.erfc(z / math.sqrt(2))
    return {"onlyFirstRight": only_a, "onlySecondRight": only_b, "p": p}


def cluster_bootstrap(a: np.ndarray, b: np.ndarray, clusters: np.ndarray, reps: int = 1000,
                      seed: int = 0) -> dict[str, float]:
    """The difference of two paired rates (``b`` minus ``a``), with a 95 % interval from a
    bootstrap that resamples clusters (illustrations), since the views of one picture are not
    independent."""
    _, cluster = np.unique(clusters, return_inverse=True)
    k = cluster.max() + 1
    sum_a = np.bincount(cluster, weights=a.astype(float), minlength=k)
    sum_b = np.bincount(cluster, weights=b.astype(float), minlength=k)
    count = np.bincount(cluster, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    for r in range(reps):
        w = np.bincount(rng.integers(0, k, k), minlength=k)
        draws[r] = (w @ sum_b - w @ sum_a) / (w @ count)
    return {"difference": float((sum_b.sum() - sum_a.sum()) / count.sum()),
            "low": float(np.percentile(draws, 2.5)), "high": float(np.percentile(draws, 97.5))}


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {}
    points = (1, 5, 10, 25, 50, 75, 90, 95, 99)
    return {f"p{p}": round(float(v), 4) for p, v in zip(points, np.percentile(values, points))}


def _rate(numerator: np.ndarray | int, denominator: np.ndarray | int) -> float | None:
    numerator, denominator = int(np.sum(numerator)), int(np.sum(denominator))
    return round(numerator / denominator, 4) if denominator else None


class Labels:
    """What the measurements need to know of every embedded row: its identity key (the
    catalog's identity rule), and as integers its illustration and its look
    (``cardvision.mtg.looks``); and whether Scryfall marks its image lowres."""

    def __init__(self, catalog: Catalog, ids: Sequence[str]) -> None:
        from cardvision.mtg.looks import look_key

        cards = [catalog.card(card_id) for card_id in ids]
        self.keys = [catalog.identity_of(card_id) for card_id in ids]
        art: dict[str, int] = {}
        looks: dict[tuple, int] = {}
        self.art = np.array([art.setdefault(card.extra.get("illustration_id") or card.id, len(art)) for card in cards])
        self.look = np.array([looks.setdefault(look_key(card), len(looks)) for card in cards])
        self.lowres = np.array([card.extra.get("image_status") == "lowres" for card in cards], bool)


# Where a query's card stands in a bank, apart from the query's own row: another printing of the
# same look, of the same illustration in another frame, only printings of other illustrations,
# no other row (``alone``), or the card is not in the bank at all (``absent``).
BUCKETS = ("sameLook", "sameArt", "otherArt", "alone", "absent")


def _card_counts(bank: Bank, labels: Labels) -> tuple[Counter, Counter]:
    """Rows per (identity, look) and per (identity, illustration) in the bank (kept on it)."""
    cached = getattr(bank, "_counts", None)
    if cached is None:
        ident = bank.ident.tolist()
        cached = (Counter(zip(ident, labels.look[bank.members].tolist())),
                  Counter(zip(ident, labels.art[bank.members].tolist())))
        bank._counts = cached
    return cached


class Looks:
    """One query set in one family, scored against one bank: per query (one view of a card),
    the best row, score and margin of each of its four orientations, whether that row is of the
    right card, and where the card stands in the bank (``BUCKETS``).

    ``cards`` are positions in the embeddings. With ``leave_own_out``, each query's own row is
    left out of the bank: the query is then another scan of its card's other printings."""

    def __init__(self, bank: Bank, queries: np.ndarray, cards: np.ndarray, labels: Labels, family: str,
                 leave_own_out: bool = False) -> None:
        kinds = FAMILIES[family]
        looks = queries[cards][:, kinds]  # (cards, kinds, 4, D)
        self.card = np.repeat(cards, len(kinds))
        own_at = np.array([bank.position.get(int(c), -1) for c in self.card])
        best_row, best, other = score_looks(bank, looks.reshape(-1, looks.shape[-1]),
                                            np.repeat(own_at, 4) if leave_own_out else None)
        self.best_row = best_row.reshape(-1, 4)
        self.best = best.reshape(-1, 4)
        self.margin = (best - other).reshape(-1, 4)
        truth = np.array([bank.identity_index.get(labels.keys[c], -1) for c in self.card])
        self.right = bank.ident[self.best_row] == truth[:, None]
        self.same_art = labels.art[bank.members][self.best_row] == labels.art[self.card][:, None]
        by_look, by_art = _card_counts(bank, labels)
        rows_of = np.diff(np.r_[bank.starts, len(bank)])
        mine = (own_at >= 0).astype(int)  # the own row, not counted below
        other_rows = np.where(truth >= 0, rows_of[np.maximum(truth, 0)], 0) - mine
        same_look = np.array([by_look[(t, l)] for t, l in zip(truth.tolist(), labels.look[self.card].tolist())]) - mine
        same_art = np.array([by_art[(t, a)] for t, a in zip(truth.tolist(), labels.art[self.card].tolist())]) - mine
        self.bucket = np.select([truth < 0, same_look > 0, same_art > 0, other_rows > 0],
                                ["absent", "sameLook", "sameArt", "otherArt"], "alone")

    def rule(self, tau: float, delta: float, retry: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per query: the orientation the rule settles on, whether a card is named, and whether
        it is the right one."""
        chosen, named = choose_looks(self.best, self.margin, tau, delta, retry)
        right = self.right[np.arange(len(chosen)), chosen]
        return chosen, named, named & right

    def summary(self, tau: float, delta: float, mask: np.ndarray | None = None) -> dict[str, Any]:
        """The figures at one accept rule, over the queries of ``mask`` (all by default)."""
        mask = np.ones(len(self.best), bool) if mask is None else mask
        chosen, named, named_right = self.rule(tau, delta)
        _, upright_named, upright_named_right = self.rule(tau, delta, retry=False)
        pick = np.arange(len(chosen))
        named_wrong = named & ~named_right
        wrong_same_art = named_wrong & self.same_art[pick, chosen]
        m = mask
        top1 = self.right[m, 0]
        margin = self.margin[m, 0]
        return {
            "queries": int(m.sum()),
            "top1Upright": _rate(top1, len(top1)),
            "named": _rate(named[m], len(top1)),
            "namedRight": _rate(named_right[m], len(top1)),
            "wrongOfShown": _rate(named_wrong[m], named[m]),
            "wrongSharingArt": int(wrong_same_art[m].sum()),
            "namedRightUprightOnly": _rate(upright_named_right[m], len(top1)),
            "wrongOfShownUprightOnly": _rate(upright_named[m] & ~upright_named_right[m], upright_named[m]),
            "uprightScoreBelowTau": _rate(self.best[m, 0] < tau, len(top1)),
            "uprightMarginBelowDelta": _rate(margin < delta, len(top1)),
            "score": _percentiles(self.best[m, 0]),
            "margin": _percentiles(margin),
            "marginWhenRight": _percentiles(margin[top1]),
            "marginWhenWrong": _percentiles(margin[~top1]),
        }

    def outcome(self, tau: float, delta: float) -> np.ndarray:
        """Per query, named and right at the rule: the paired outcome of two banks."""
        return self.rule(tau, delta)[2]

    def grid(self, mask: np.ndarray | None = None, open_set: Looks | None = None) -> dict[str, Any] | None:
        """Named right and wrong of shown over ``TAU_GRID`` by ``DELTA_GRID``, with the
        orientation rule; the best point holding wrong of shown to 1, 2 and 5 %; and the delta
        that does it at the default tau. With ``open_set`` (the same bank's scores of cards it
        does not hold), the share of those it names at each point too, and the best points that
        also hold that share to the same limit (``bestWithOpenSet``)."""
        mask = np.ones(len(self.best), bool) if mask is None else mask
        if not mask.any():
            return None
        open_set = open_set if open_set is not None and len(open_set.best) else None
        shape = (len(TAU_GRID), len(DELTA_GRID))
        named_right, wrong = np.zeros(shape), np.zeros(shape)
        open_named = np.zeros(shape)
        for i, tau in enumerate(TAU_GRID):
            for j, delta in enumerate(DELTA_GRID):
                _, named, right = self.rule(tau, delta)
                named, right = named[mask], right[mask]
                named_right[i, j] = right.mean()
                wrong[i, j] = (named & ~right).sum() / max(1, named.sum())
                if open_set is not None:
                    open_named[i, j] = open_set.rule(tau, delta)[1].mean()
        default_row = TAU_GRID.index(DEFAULT_TAU)

        def best(ok: np.ndarray) -> dict[str, Any] | None:
            if not ok.any():
                return None
            i, j = np.unravel_index(np.argmax(np.where(ok, named_right, -1)), shape)
            point = {"tau": TAU_GRID[i], "delta": DELTA_GRID[j], "namedRight": round(float(named_right[i, j]), 4),
                     "wrongOfShown": round(float(wrong[i, j]), 4)}
            if open_set is not None:
                point["openSetNamed"] = round(float(open_named[i, j]), 4)
            at_default = np.flatnonzero(ok[default_row])
            point["deltaAtDefaultTau"] = DELTA_GRID[at_default[0]] if len(at_default) else None
            return point

        limits = (0.01, 0.02, 0.05)
        out = {"queries": int(mask.sum()), "tau": list(TAU_GRID), "delta": list(DELTA_GRID),
               "namedRight": np.round(named_right, 4).tolist(), "wrongOfShown": np.round(wrong, 4).tolist(),
               "best": {f"{limit:.2f}": best(wrong <= limit) for limit in limits}}
        if open_set is not None:
            out["openSetNamed"] = np.round(open_named, 4).tolist()
            out["bestWithOpenSet"] = {f"{limit:.2f}": best((wrong <= limit) & (open_named <= limit)) for limit in limits}
        return out


def printing_figures(bank: Bank, queries: np.ndarray, query_members: np.ndarray, labels: Labels,
                     family: str) -> dict[str, Any]:
    """Within the right card, upright looks only: is the best of the card's rows the query's own
    printing, or a printing of the same look? Over the queries whose card has two or more rows
    in the bank.

    ``ownRowIn`` scores the query against all of its card's rows: the exact printing is then an
    upper bound, since the query is a copy of its own row's scan. ``ownRowOut`` leaves the own
    row out, and asks, of the queries whose look has another printing in the bank, how often the
    best remaining row is of that look: another scan of the same look, the realistic case."""
    kinds = list(FAMILIES[family])
    ends = np.r_[bank.starts[1:], len(bank)]
    exact = same_look = n_in = 0
    twin_found = n_out = 0
    for card in query_members:
        at = bank.position.get(int(card))
        if at is None:
            continue
        ident = bank.ident[at]
        start, end = bank.starts[ident], ends[ident]
        if end - start < 2:
            continue
        sims = queries[card][kinds, 0] @ bank.rows[start:end].T  # (k, rows of the card)
        own = at - start
        twin = labels.look[bank.members[start:end]] == labels.look[card]
        best = np.argmax(sims, axis=1)
        n_in += len(best)
        exact += int((best == own).sum())
        same_look += int(twin[best].sum())
        twin[own] = False
        if twin.any():
            sims[:, own] = -np.inf
            n_out += len(best)
            twin_found += int(twin[np.argmax(sims, axis=1)].sum())
    return {"ownRowIn": {"queries": n_in, "exactPrinting": _rate(exact, n_in), "sameLook": _rate(same_look, n_in)},
            "ownRowOut": {"queriesWithTwin": n_out, "twinFound": _rate(twin_found, n_out)}}


# ------------------------------------------------------------------------------------------- experiments

# The ladders: one fixed query set, scored against nested banks that all hold it, so that what
# changes from one bank to the next is the bank alone. ``control`` is a bank of the size of that
# scope, made of the query set and random other printings: is a step's loss crowding by any
# printings, or by those of the scope?
LADDERS: dict[str, dict[str, Any]] = {
    "arena": {"queries": "standard-arena", "banks": ("standard-arena", "arena", "all"), "control": "arena"},
    "paper": {"queries": "standard", "banks": ("standard", "pioneer", "modern", "all"), "control": "pioneer"},
}
RANDOM_LADDER = (5000, 30000)  # random banks, each holding the smaller, then every row; queries: the first
OPEN_SET_CAP = 6000  # cards scored in an open-set test, at most (a random sample)
MAIN_FAMILY = "stream"
WITHIN = 0.05  # the decision rule's tolerance: named right within 5 points of the first bank's
WRONG_LIMIT = "0.02"  # ... at the best point that holds wrong of shown to 2 %


def _sample(values: Sequence[int], size: int, rng: np.random.Generator) -> list[int]:
    values = list(values)
    if len(values) <= size:
        return sorted(values)
    return sorted(int(v) for v in rng.choice(values, size=size, replace=False))


def _subset(looks: Looks, mask: np.ndarray) -> Looks:
    """The same scores, for the queries of ``mask`` only."""
    out = Looks.__new__(Looks)
    for name, value in vars(looks).items():
        setattr(out, name, value[mask])
    return out


def _pair(a: Looks, b: Looks, a_point: Mapping[str, Any] | None, b_point: Mapping[str, Any] | None,
          clusters: np.ndarray, tau: float, delta: float) -> dict[str, Any]:
    """Bank ``b`` against bank ``a`` on the same queries: top-1 and named right at the default
    rule, at the first bank's operating point for both, and with each bank at its own."""

    def compare(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
        return {**mcnemar(first, second), **cluster_bootstrap(first, second, clusters)}

    out = {"top1Upright": mcnemar(a.right[:, 0], b.right[:, 0]),
           "namedRightAtRule": compare(a.outcome(tau, delta), b.outcome(tau, delta))}
    if a_point:
        out["namedRightAtFirstBankPoint"] = compare(a.outcome(a_point["tau"], a_point["delta"]),
                                                    b.outcome(a_point["tau"], a_point["delta"]))
    if a_point and b_point:
        out["namedRightAtOwnPoint"] = compare(a.outcome(a_point["tau"], a_point["delta"]),
                                              b.outcome(b_point["tau"], b_point["delta"]))
    return out


def _point(grid: Mapping[str, Any] | None, open_set: bool = True) -> dict[str, Any] | None:
    """A bank's operating point: the best point of its grid that holds wrong of shown to
    ``WRONG_LIMIT``, and, when the bank was given an open set (``open_set``), the share of it
    named to the same limit."""
    if not grid:
        return None
    if open_set and "bestWithOpenSet" in grid:
        return grid["bestWithOpenSet"][WRONG_LIMIT]
    return grid["best"][WRONG_LIMIT]


def _decide(banks: Mapping[str, Any], order: Sequence[str], condition: str) -> dict[str, Any]:
    """The decision rule over one ladder's banks: the largest bank (every smaller one passing
    too) whose named right at its operating point (``_point``) is within ``WITHIN`` of the first
    bank's, under one condition (``ownRowOut`` or ``ownRowIn``). Each step also gives the bank's
    named right at the first bank's operating point (``atFirstBankPoint``): the same rule for
    every bank, the loss crowding alone causes."""
    grid_of = (lambda e: e["ownRowOut"]["twinSet"]["grid"]) if condition == "ownRowOut" else (lambda e: e["ownRowIn"]["grid"])
    first_point = _point(grid_of(banks[order[0]]))
    if first_point is None:
        return {"chosen": None, "why": f"the first bank has no point holding wrong of shown to {WRONG_LIMIT}"}
    steps = {}
    chosen, failed = order[0], False
    for name in order:
        point = _point(grid_of(banks[name]))
        loss = None if point is None else round(first_point["namedRight"] - point["namedRight"], 4)
        steps[name] = {"point": point, "lossAgainstFirst": loss}
        if condition == "ownRowOut":
            steps[name]["atFirstBankPoint"] = banks[name]["ownRowOut"]["twinSet"].get("atFirstBankPoint")
        if name.endswith("-size"):  # a control bank
            continue
        failed = failed or loss is None or loss > WITHIN  # a bank counts only if every smaller one does
        if not failed:
            chosen = name
    open_named = first_point.get("openSetNamed")
    return {"chosen": chosen, "steps": steps, "firstBankOpenSetNamed": open_named,
            "openSetOk": None if open_named is None else open_named <= float(WRONG_LIMIT)}


def run_crowding(catalog: Catalog, ids: Sequence[str], rows: np.ndarray, queries: np.ndarray, *,
                 tau: float = DEFAULT_TAU, delta: float = DEFAULT_DELTA, seed: int = 0,
                 ladders: Mapping[str, Mapping[str, Any]] | None = None, random_ladder: Sequence[int] = RANDOM_LADDER,
                 families: Sequence[str] = tuple(FAMILIES),
                 progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Every measurement of the crowding report, from the embeddings (``load_embeddings``).

    Per ladder, family and bank, three conditions:

    ``ownRowIn``
        the queries against the bank as it is: their own row is in it. An upper bound, since a
        query is a copy of its own row's scan;
    ``ownRowOut``
        the own row left out: the query is another scan of what else the bank holds of its card.
        By bucket (``BUCKETS``), and on the *twin set*, the queries whose card has another
        printing of the same look in the first bank (so in every bank of the ladder): the
        realistic case, paired across the banks. The decision rule is applied to it;
    ``openSet``
        cards of the next bank's scope that this bank does not hold: how many it names, all
        wrongly.
    """
    from cardvision.mtg.looks import scope_ids

    say = progress or (lambda message: None)
    ladders = LADDERS if ladders is None else ladders
    rng = np.random.default_rng(seed)
    position = {card_id: i for i, card_id in enumerate(ids)}
    labels = Labels(catalog, ids)

    def members(scope: str) -> list[int]:
        return [position[card_id] for card_id in scope_ids(catalog, scope) if card_id in position]

    report: dict[str, Any] = {
        "embedded": len(ids), "catalogWithArt": sum(1 for card in catalog if card.has_art), "tau": tau, "delta": delta,
        "families": {family: [QUERY_KINDS[k] for k in kinds] for family, kinds in FAMILIES.items()},
        "mainFamily": MAIN_FAMILY, "wrongLimit": float(WRONG_LIMIT), "within": WITHIN, "decisionRule": (
            f"the largest bank (every smaller one passing too) whose named right on the twin set, own row out, at its "
            f"operating point is within {WITHIN:.2f} of the first bank's ({MAIN_FAMILY} family, rotation retry on). "
            f"A bank's operating point is the tau and delta that name the most of the twin set while holding wrong "
            f"of shown to {WRONG_LIMIT} and, when the bank has an open set, the share of it named to {WRONG_LIMIT}"),
        "scopeRows": {}, "ladders": {},
    }

    def ladder(name: str, query_cards: list[int], banks: list[tuple[str, list[int]]]) -> dict[str, Any]:
        out: dict[str, Any] = {"queryCards": len(query_cards), "banks": {}, "families": {}}
        q = np.array(query_cards, np.int64)
        built = []
        ranked = [(bank_name, m) for bank_name, m in banks if not bank_name.endswith("-size")]  # not the control
        for bank_name, bank_members in banks:
            bank = Bank(sorted(set(bank_members) | set(query_cards)), labels.keys, rows)
            open_cards: list[int] = []
            later = [m for n, m in ranked[[n for n, _ in ranked].index(bank_name) + 1:]] if bank_name in dict(ranked) else []
            if later:
                open_cards = _sample([c for c in later[0] if labels.keys[c] not in bank.identity_index], OPEN_SET_CAP, rng)
            built.append((bank_name, bank, open_cards))
            out["banks"][bank_name] = {"rows": len(bank), "identities": len(bank.starts),
                                       "addedQueryCards": len(set(query_cards) - set(bank_members)),
                                       "openSetCards": len(open_cards)}
        for family in families:
            say(f"{name}: {family}")
            scored = []
            for bank_name, bank, open_cards in built:
                inside = Looks(bank, queries, q, labels, family)
                outside = Looks(bank, queries, q, labels, family, leave_own_out=True)
                open_looks = None
                if open_cards:
                    open_looks = Looks(bank, queries, np.array(open_cards, np.int64), labels, family)
                    open_looks = _subset(open_looks, open_looks.bucket == "absent")
                scored.append((bank_name, bank, inside, outside, open_looks))
            twin = scored[0][3].bucket == "sameLook"
            fam: dict[str, Any] = {"twinSetQueries": int(twin.sum()), "banks": {}}
            for bank_name, bank, inside, outside, open_looks in scored:
                low = labels.lowres[inside.card]
                twin_grid = outside.grid(twin, open_looks)
                entry: dict[str, Any] = {
                    "ownRowIn": {"atRule": inside.summary(tau, delta), "grid": inside.grid(),
                                 "lowres": {"lowres": inside.summary(tau, delta, low),
                                            "notLowres": inside.summary(tau, delta, ~low)}},
                    "ownRowOut": {"buckets": {b: outside.summary(tau, delta, outside.bucket == b)
                                              for b in BUCKETS if (outside.bucket == b).any()},
                                  "twinSet": {"atRule": outside.summary(tau, delta, twin), "grid": twin_grid,
                                              "lowres": {"lowres": outside.summary(tau, delta, twin & low),
                                                         "notLowres": outside.summary(tau, delta, twin & ~low)}}},
                }
                if open_looks is not None:
                    entry["openSet"] = {"cards": len(set(open_looks.card.tolist())), "queries": len(open_looks.best),
                                        "namedAtRule": _rate(open_looks.rule(tau, delta)[1], len(open_looks.best)),
                                        "namedUprightOnlyAtRule": _rate(open_looks.rule(tau, delta, retry=False)[1],
                                                                        len(open_looks.best)),
                                        "score": _percentiles(open_looks.best[:, 0]),
                                        "margin": _percentiles(open_looks.margin[:, 0])}
                if not bank_name.endswith("-size"):
                    entry["printing"] = printing_figures(bank, queries, q, labels, family)
                fam["banks"][bank_name] = entry
                at = entry["ownRowOut"]["twinSet"]
                say(f"  {bank_name}: own row in: top-1 {entry['ownRowIn']['atRule']['top1Upright']}, named right "
                    f"{entry['ownRowIn']['atRule']['namedRight']}; own row out, twin set: top-1 {at['atRule']['top1Upright']}, "
                    f"named right {at['atRule']['namedRight']} (wrong of shown {at['atRule']['wrongOfShown']}); "
                    f"operating point: {_point(at['grid'])}; open set named at the rule "
                    f"{entry.get('openSet', {}).get('namedAtRule')}")
            first_name, _, first_in, first_out, _ = scored[0]
            clusters_in = labels.art[first_in.card]
            first_twin = _subset(first_out, twin)
            first_point = _point(fam["banks"][first_name]["ownRowOut"]["twinSet"]["grid"])
            for bank_name, _, _, outside, open_looks in scored:
                at = fam["banks"][bank_name]["ownRowOut"]["twinSet"]
                if first_point:
                    at["atFirstBankPoint"] = {"tau": first_point["tau"], "delta": first_point["delta"],
                                              **outside.summary(first_point["tau"], first_point["delta"], twin)}
                    if open_looks is not None:
                        at["atFirstBankPoint"]["openSetNamed"] = _rate(
                            open_looks.rule(first_point["tau"], first_point["delta"])[1], len(open_looks.best))
            for bank_name, _, inside, outside, _ in scored[1:]:
                e, f = fam["banks"][bank_name], fam["banks"][first_name]
                fam["banks"][bank_name]["againstFirst"] = {
                    "ownRowIn": _pair(first_in, inside, _point(f["ownRowIn"]["grid"], False),
                                      _point(e["ownRowIn"]["grid"], False), clusters_in, tau, delta)}
                if twin.any():
                    fam["banks"][bank_name]["againstFirst"]["ownRowOutTwinSet"] = _pair(
                        first_twin, _subset(outside, twin), _point(f["ownRowOut"]["twinSet"]["grid"]),
                        _point(e["ownRowOut"]["twinSet"]["grid"]), clusters_in[twin], tau, delta)
            order = [bank_name for bank_name, *_ in scored]
            fam["decision"] = {"ownRowOut": _decide(fam["banks"], order, "ownRowOut"),
                               "ownRowIn": _decide(fam["banks"], order, "ownRowIn")}
            out["families"][family] = fam
        return out

    for name, spec in ladders.items():
        query_cards = members(spec["queries"])
        banks = [(scope, members(scope)) for scope in spec["banks"]]
        for scope, m in banks:
            report["scopeRows"][scope] = {"embedded": len(m), "withArt": len(scope_ids(catalog, scope))}
        size = len(members(spec["control"]))
        others = sorted(set(range(len(ids))) - set(query_cards))
        control = sorted(set(query_cards) | set(_sample(others, max(0, size - len(query_cards)), rng)))
        banks.insert(spec["banks"].index(spec["control"]) + 1, (f"random-{spec['control']}-size", control))
        report["ladders"][name] = {"queries": spec["queries"], **ladder(name, query_cards, banks)}

    if random_ladder:
        order = rng.permutation(len(ids))
        query_cards = sorted(order[:random_ladder[0]].tolist())
        banks = [(f"random-{size // 1000}k", sorted(order[:size].tolist())) for size in random_ladder if size < len(ids)]
        banks.append(("all", list(range(len(ids)))))
        report["ladders"]["random"] = {"queries": banks[0][0], **ladder("random", query_cards, banks)}
    return report
