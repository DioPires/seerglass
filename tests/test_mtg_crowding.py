# SPDX-License-Identifier: Apache-2.0
"""The crowding measurements (``cardvision.mtg.crowding``) on small made-up embeddings: the fast
pooling, the exact best-and-other scoring, the orientation rule, the paired statistics and a
whole run of the report."""

from __future__ import annotations

import numpy as np
import pytest

from cardvision.core.catalog import parse_catalog
from cardvision.core.preprocess import full128
from cardvision.mtg import crowding as C


@pytest.mark.parametrize("shape", [(680, 488), (97, 61), (128, 128), (40, 300), (1, 1)])
def test_fast_full128_gives_the_same_tensor_as_full128(shape):
    rgb = np.random.default_rng(1).integers(0, 256, size=(*shape, 3), dtype=np.uint8)
    np.testing.assert_array_equal(C.fast_full128(rgb), full128(rgb))


def _unit(x):
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


def _brute(rows, ident, looks, own=None):
    s = looks.astype(np.float64) @ rows.astype(np.float64).T
    if own is not None:
        for i, o in enumerate(own):
            if o >= 0:
                s[i, o] = -np.inf
    best_row = s.argmax(axis=1)
    other = np.array([max((v for v, k in zip(r, ident) if k != ident[b] and np.isfinite(v)), default=C.NO_OTHER)
                      for r, b in zip(s, best_row)])
    return best_row, s[np.arange(len(s)), best_row], other


def test_score_looks_matches_a_plain_scan_and_leaves_the_own_row_out():
    rng = np.random.default_rng(2)
    rows = _unit(rng.standard_normal((40, 16)))
    keys = [f"k{i % 7}" for i in range(40)]
    bank = C.Bank(range(40), keys, rows)
    looks = _unit(rows[[3, 9, 17, 30]] + 0.1 * rng.standard_normal((4, 16)))
    best_row, best, other = C.score_looks(bank, looks, block=3)
    want_row, want_best, want_other = _brute(bank.rows, bank.ident, looks)
    np.testing.assert_array_equal(best_row, want_row)
    np.testing.assert_allclose(best, want_best, atol=1e-6)
    np.testing.assert_allclose(other, want_other, atol=1e-6)
    own = np.array([bank.position[m] for m in (3, 9, 17, 30)])
    best_row, best, other = C.score_looks(bank, looks, own)
    assert not np.any(best_row == own)
    want_row, want_best, want_other = _brute(bank.rows, bank.ident, looks, own)
    np.testing.assert_array_equal(best_row, want_row)
    np.testing.assert_allclose(other, want_other, atol=1e-6)


def test_a_bank_of_one_card_has_no_other_score():
    rows = _unit(np.random.default_rng(3).standard_normal((3, 8)))
    bank = C.Bank([0, 1, 2], ["a", "a", "a"], rows)
    _, _, other = C.score_looks(bank, rows)
    assert np.all(other == C.NO_OTHER)


def test_choose_looks_follows_the_runtime_rule():
    best = np.array([[0.70, 0.90, 0.80, 0.10],   # upright accepted: kept although another scores higher
                     [0.50, 0.70, 0.80, 0.80],   # upright refused: the best accepted, the lower on a tie
                     [0.50, 0.60, 0.61, 0.40]])  # nothing accepted
    margin = np.full_like(best, 0.10)
    chosen, named = C.choose_looks(best, margin, 0.62, 0.05)
    assert chosen[:2].tolist() == [0, 2] and named.tolist() == [True, True, False]
    chosen, named = C.choose_looks(best, margin, 0.62, 0.05, retry=False)
    assert chosen.tolist() == [0, 0, 0] and named.tolist() == [True, False, False]


def test_mcnemar_and_the_cluster_bootstrap():
    a = np.array([True] * 10 + [False] * 6)
    b = np.array([True] * 16)
    result = C.mcnemar(a, b)
    assert (result["onlyFirstRight"], result["onlySecondRight"]) == (0, 6)
    assert result["p"] == pytest.approx(2 * 0.5 ** 6)
    assert C.mcnemar(a, a)["p"] == 1.0
    same = C.cluster_bootstrap(a, a, np.arange(16) // 2, reps=50)
    assert same == {"difference": 0.0, "low": 0.0, "high": 0.0}
    better = C.cluster_bootstrap(a, b, np.arange(16) // 2, reps=200)
    assert better["difference"] == pytest.approx(6 / 16) and better["low"] > 0


def _entry(card_id, name, oracle, illustration, *, legal, games, printing_of=None, frame="2015"):
    return {"id": card_id, "name": name, "kind": "Creature", "printingOf": printing_of, "art": card_id.replace(":", "-") + ".jpg",
            "extra": {"oracle_id": oracle, "illustration_id": illustration, "frame": frame, "border_color": "black",
                      "full_art": False, "textless": False, "frame_effects": [], "legal": legal, "games": games,
                      "image_status": "highres_scan"}}


def _small_world():
    """Six cards: four Standard ones on Arena, two of them with a second printing of the same
    look (``b``) or another picture (``d``), one Arena card outside Standard, one paper card."""
    std, arena = ["standard", "pioneer", "modern"], ["arena", "paper"]
    cards = [
        _entry("s:1", "A", "oa", "ia", legal=std, games=arena),
        _entry("s:2", "B", "ob", "ib", legal=std, games=arena),
        _entry("t:2", "B", "ob", "ib", legal=std, games=arena, printing_of="s:2"),
        _entry("s:3", "C", "oc", "ic", legal=std, games=arena),
        _entry("s:4", "D", "od", "id", legal=std, games=arena),
        _entry("t:4", "D", "od", "id2", legal=std, games=arena, printing_of="s:4"),
        _entry("h:1", "E", "oe", "ie", legal=["modern"], games=["arena"]),
        _entry("p:1", "F", "of", "if", legal=["modern"], games=["paper"]),
    ]
    catalog = parse_catalog({"game": "mtg", "locale": "en", "identity": "printing", "cards": cards})
    ids = [card["id"] for card in cards]
    rng = np.random.default_rng(4)
    base = {"ia": 0, "ib": 1, "ic": 2, "id": 3, "id2": 4, "ie": 5, "if": 6}
    centres = _unit(rng.standard_normal((7, 32)))
    rows = _unit(np.stack([centres[base[card["extra"]["illustration_id"]]] + 0.05 * rng.standard_normal(32)
                           for card in cards]))
    queries = np.empty((len(ids), len(C.QUERY_KINDS), 4, 32), np.float32)
    for i in range(len(ids)):
        queries[i, :, 0] = _unit(rows[i] + 0.05 * rng.standard_normal((len(C.QUERY_KINDS), 32)))
        queries[i, :, 1:] = _unit(rng.standard_normal((len(C.QUERY_KINDS), 3, 32)))
    return catalog, ids, rows, queries


def test_the_report_on_a_small_world():
    catalog, ids, rows, queries = _small_world()
    ladders = {"arena": {"queries": "standard-arena", "banks": ("standard-arena", "arena", "all"), "control": "arena"}}
    report = C.run_crowding(catalog, ids, rows, queries, ladders=ladders, random_ladder=(), families=("stream",))
    ladder = report["ladders"]["arena"]
    assert ladder["queryCards"] == 6
    assert list(ladder["banks"]) == ["standard-arena", "arena", "random-arena-size", "all"]
    assert ladder["banks"]["standard-arena"]["openSetCards"] == 1  # E, on Arena outside Standard
    assert ladder["banks"]["arena"]["openSetCards"] == 1  # F, paper only
    family = ladder["families"]["stream"]
    first = family["banks"]["standard-arena"]
    assert first["ownRowIn"]["atRule"]["top1Upright"] == 1.0
    buckets = first["ownRowOut"]["buckets"]
    # Own row out: B has its twin, D only another picture, A and C nothing at all.
    assert buckets["sameLook"]["queries"] == 4 and buckets["otherArt"]["queries"] == 4
    assert buckets["alone"]["queries"] == 4
    assert buckets["sameLook"]["top1Upright"] == 1.0
    assert family["twinSetQueries"] == 4
    assert first["openSet"]["cards"] == 1
    assert first["printing"]["ownRowOut"]["twinFound"] == 1.0
    assert set(family["decision"]) == {"ownRowOut", "ownRowIn"}
    assert family["decision"]["ownRowOut"]["chosen"] in ("standard-arena", "arena", "all")
    assert "ownRowOutTwinSet" in family["banks"]["all"]["againstFirst"]
