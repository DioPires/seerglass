# SPDX-License-Identifier: Apache-2.0
"""An epoch's make-up, the draws, the pair batch sampler and the per-row chain, on rows made
up in this file. Skipped without PyTorch."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cardvision.composites.cards import KindSettings  # noqa: E402
from cardvision.core.catalog import parse_catalog  # noqa: E402
from cardvision.train_recognizer import dataset as D  # noqa: E402
from cardvision.train_recognizer.config import ConfigError, parse_config  # noqa: E402
from cardvision.train_recognizer.losses import NEGATIVE, NEGATIVE_BACK  # noqa: E402
from cardvision.train_recognizer.sources import Stream  # noqa: E402
from helpers import pattern  # noqa: E402

# Ten cards: C0 and C1 share a name (C1 is an alternate printing), C8 and C9 are sites.
CARDS = [{"id": f"C{i}", "name": "Twin" if i < 2 else f"Card {i}", "kind": "site" if i >= 8 else "unit",
          "printingOf": "C0" if i == 1 else None, "art": f"C{i}.png"} for i in range(10)]
CATALOG = parse_catalog({"game": "made-up", "locale": "en", "cards": CARDS,
                         "kinds": {"boost": {"shareKind": "site", "share": 0.5}}})
ART = {card["id"]: pattern(90, 64)[:, ::(-1 if i % 2 else 1)].copy() + np.uint8(i) for i, card in enumerate(CARDS)}
PATCHES = [pattern(120, 120), pattern(120, 120)[::-1].copy()]
IMAGE = pattern(162, 115)
INNER = [25, 36, 64, 90]


def rows(ids, **extra):
    return [{"image": IMAGE, "inner": INNER, "card_id": card_id, "group": "g", **extra} for card_id in ids]


def streams_of(**pools) -> dict[str, Stream]:
    settings = {"base": ("physical", False), "comp": ("plain", False), "real": ("plain", False),
                "neg_a": ("plain", True), "neg_b": ("plain", True), "each": ("physical", False)}
    return {name: Stream(name, pool, *settings[name]) for name, pool in pools.items()}


def config(segments, base=100, batch=16, **over):
    return parse_config({
        "seed": 7, "batch": batch, "workers": 0, "data": {"catalog": "c.json", "art": "art"}, "epoch": {"base": base},
        "streams": {"base": {"chain": "physical"}, "comp": {}, "real": {}, "each": {"chain": "physical"},
                    "neg_a": {"negative": True}, "neg_b": {"negative": True}},
        "segment": segments, **over})


POOLS = streams_of(base=rows(["C0", "C1", "C2", "C3"] * 25), comp=rows(["C2", "C4", "C8"] * 8),
                   real=rows(["C5", "C3", "C9"] * 2), neg_a=rows([None] * 5, label="back"),
                   neg_b=rows([None] * 3, label="not-card"), each=rows(["C6"] * 2 + ["C7"] * 5))
SEGMENTS = [{"stream": "base", "per_base": 1.0, "draw": "all"}, {"stream": "comp", "per_base": 0.30, "draw": "subset"},
            {"stream": "real", "share": 0.10}, {"streams": ["neg_a", "neg_b"], "share": 0.08}, {"stream": "each", "per_class": 4}]


# ------------------------------------------------------------------------------- composing
def test_an_epoch_is_worked_out_from_the_stated_base_count():
    comp = D.compose(config(SEGMENTS), POOLS, CATALOG)
    # 100 base rows; 0.30 x 100 = 30 composites, but the stream has only 24;
    # real: round(0.10 / 0.90 x 124) = 14; negatives: round(0.08 / 0.92 x 138) = 12, halved;
    # 4 rows of each of the 2 cards of the last stream.
    assert [(s.stream, s.n) for s in comp.segments] == [
        ("base", 100), ("comp", 24), ("real", 14), ("neg_a", 6), ("neg_b", 6), ("each", 8)]
    assert len(comp) == comp.n_single == 158 and comp.n_boost == 0 and comp.n_units == 0
    assert comp.report["base"] == 100 and comp.report["rowsPerEpoch"] == 158
    assert D.share_rows(0.10, 124) == 14 and D.share_rows(0.0, 124) == 0
    # The classes: the first stream's cards, then each later stream's new ones.
    assert D.class_list(comp) == ["C0", "C1", "C2", "C3", "C4", "C8", "C5", "C9", "C6", "C7"]


def test_sizes_never_follow_a_stream_s_size_unless_the_config_says_all():
    # The same config over streams of other sizes gives the same counts (short of a subset's cap).
    bigger = streams_of(base=rows(["C0", "C1", "C2", "C3"] * 25), comp=rows(["C2", "C4"] * 200), real=rows(["C5"] * 999),
                        neg_a=rows([None] * 70), neg_b=rows([None] * 1), each=rows(["C6", "C7"] * 50))
    assert [s.n for s in D.compose(config(SEGMENTS), bigger, CATALOG).segments] == [100, 30, 14, 6, 7, 8]
    # "all" with a stated count checks the count against the stream.
    with pytest.raises(ConfigError, match=r"has 100 rows, but its segment with draw = \"all\" states 120"):
        D.compose(config(SEGMENTS, base=120), POOLS, CATALOG)
    free = D.compose(config([{"stream": "base", "draw": "all"}, {"stream": "real", "count": 9}]), POOLS, CATALOG)
    assert [s.n for s in free.segments] == [100, 9]


def test_a_stream_without_rows_gives_none_and_is_reported():
    pools = {name: stream for name, stream in POOLS.items() if name not in ("real", "neg_b")}
    comp = D.compose(config(SEGMENTS), pools, CATALOG)
    assert [(s.stream, s.n) for s in comp.segments] == [("base", 100), ("comp", 24), ("neg_a", 11), ("each", 8)]
    assert comp.report["emptyStreams"] == ["real", "neg_b"]


def test_boost_adds_alternate_printings_and_brings_a_kind_to_its_share():
    cfg = config(SEGMENTS, components={"boost": True}, boost={"alternate_printings": 1.0, "share_kind": "site", "share": 0.25})
    comp = D.compose(cfg, POOLS, CATALOG)
    # Expected rows before the boost: positives 100 + 24 + 14 + 8 = 146.
    # Alternate printings (C1): a quarter of the base stream, 25 rows -> 25 more.
    # Sites (C8, C9): a third of the composites and of the real crops, 8 + 14/3 = 12.67 rows;
    # to reach 0.25 of 146 + 25 = 171 rows: ceil((0.25 x 171 - 12.67) / 0.75) = 41.
    (alternates, n_alternates), (sites, n_sites) = comp.boost_pools
    assert n_alternates == 25 and {row["card_id"] for _, row in alternates} == {"C1"}
    assert n_sites == math.ceil((0.25 * 171 - (8 + 14 / 3)) / 0.75) == 41
    assert {row["card_id"] for _, row in sites} == {"C8", "C9"} and {name for name, _ in sites} == {"comp", "real"}
    assert comp.n_boost == 66 and len(comp) == 158 + 66
    # Left out of the config, the kind and its share come from the catalog's kinds.boost.
    from_catalog = D.compose(config(SEGMENTS, components={"boost": True}), POOLS, CATALOG)
    assert from_catalog.report["boost"]["kind"] == "site" and from_catalog.report["boost"]["kindShareTarget"] == 0.5


def test_pairs_take_the_last_indices_and_fill_every_batch():
    cfg = config(SEGMENTS, components={"pairs": True},
                 pairs={"per_batch": 2, "names": [["Twin", "Card 2"], ["Card 4", "Card 5"], ["Card 3", "No Such Card"]]})
    comp = D.compose(cfg, POOLS, CATALOG)
    assert comp.pair_list == [("Twin", "Card 2"), ("Card 4", "Card 5")] and comp.report["pairs"]["dropped"] == [["Card 3", "No Such Card"]]
    # 158 single rows in batches of 16 - 2 x 2 = 12 singles: 13 batches, 2 pairs each.
    assert comp.per_batch == 2 and comp.n_units == 26 and len(comp) == 158 + 52
    assert {row["card_id"] for _, row in comp.pair_pools["Twin"]} == {"C0", "C1"}    # a name's pool holds every printing


# --------------------------------------------------------------------------------- drawing
def test_a_balanced_draw_uses_every_row_equally_often():
    for pool, n in ((7, 3), (7, 7), (7, 20), (1, 5)):
        drawn = D.balanced_draw(np.random.default_rng(1), pool, n)
        counts = Counter(int(i) for i in drawn)
        assert len(drawn) == n and max(counts.values()) - (min(counts.values()) if len(counts) == pool else 0) <= 1
    assert len(set(D.subset_draw(np.random.default_rng(1), 30, 30).tolist())) == 30
    picked = D.per_class_draw(np.random.default_rng(1), {"b": [5, 6], "a": [0, 1, 2]}, 4)   # cards in sorted order
    assert set(picked[:4]) <= {0, 1, 2} and set(picked[4:]) == {5, 6} and Counter(picked[4:]) == {5: 2, 6: 2}


def test_the_pair_batch_sampler_keeps_both_rows_of_a_pair_together():
    sampler = D.PairBatchSampler(n_single=158, n_units=26, per_batch=2, batch=16, seed=7)
    assert len(sampler) == 13
    batches = list(sampler)
    assert len(batches) == 13 and all(len(batch) == 16 for batch in batches)
    singles = [i for batch in batches for i in batch if i < 158]
    assert len(singles) == len(set(singles)) == 13 * 12
    units = []
    for batch in batches:
        pair_rows = sorted(i - 158 for i in batch if i >= 158)
        assert len(pair_rows) == 4 and all(a % 2 == 0 and b == a + 1 for a, b in zip(pair_rows[::2], pair_rows[1::2]))
        units += [a // 2 for a in pair_rows[::2]]
    assert sorted(units) == list(range(26))
    assert batches == list(sampler)                                               # the same epoch again
    sampler.set_epoch(1)
    assert batches != list(sampler)
    assert list(D.PairBatchSampler(158, 26, 2, 16, seed=8)) != batches
    plain = D.PairBatchSampler(n_single=50, n_units=0, per_batch=0, batch=16, seed=7)
    assert len(plain) == 3 and sorted(i for batch in plain for i in batch) != list(range(50))
    assert len({i for batch in plain for i in batch}) == 48


def test_the_sampler_s_seed_is_the_training_seed_the_offset_and_the_epoch():
    import torch

    def expected(seed, offset, epoch, n_single=50, batch=16):
        g = torch.Generator().manual_seed(seed * 1000 + offset + epoch)
        singles = torch.randperm(n_single, generator=g).tolist()
        return [[singles[b * batch:(b + 1) * batch][p] for p in torch.randperm(batch, generator=g).tolist()]
                for b in range(n_single // batch)]

    for offset in (0, 5, 123456):
        sampler = D.PairBatchSampler(n_single=50, n_units=0, per_batch=0, batch=16, seed=7, offset=offset)
        assert list(sampler) == expected(7, offset, 0)
        sampler.set_epoch(3)
        assert list(sampler) == expected(7, offset, 3)
    # Left out, the offset is the one derived from its name, which is also the config's default.
    default = config(SEGMENTS)["seeds"]["sampler_offset"]
    assert D.DEFAULT_SAMPLER_OFFSET == default
    assert list(D.PairBatchSampler(50, 0, 0, 16, seed=7)) == expected(7, default, 0)
    assert list(D.PairBatchSampler(50, 0, 0, 16, seed=7, offset=default + 1)) != expected(7, default, 0)


# ---------------------------------------------------------------------------------- the rows
OFF = {"occluders": False, "glare": False, "low_light": False, "rotation": False}


def training_set(components=None, pools=POOLS, segments=SEGMENTS, **over) -> D.TrainingSet:
    """A training set with the augmentation components off unless ``components`` says otherwise."""
    cfg = config(segments, components={**OFF, **(components or {})}, **over)
    comp = D.compose(cfg, pools, CATALOG)
    return D.TrainingSet(comp, pools, ART, PATCHES, D.class_list(comp), CATALOG, cfg, KindSettings())


TAGGED = {"subset": 3, "balanced": [4], "negative": 5, "per_class": [6, 1], "boost": [6, 2], "pairs": [6, 3]}


def test_the_per_epoch_draws_are_seeded_by_the_config_s_tags():
    """Each draw is the documented one: a generator seeded with the training seed, the tag of
    the kind of draw and the epoch. A run made with other tags is repeated by stating them."""
    both = {"components": {"boost": True, "pairs": True},
            "boost": {"alternate_printings": 1.0, "share_kind": "site", "share": 0.25},
            "pairs": {"per_batch": 2, "names": [["Twin", "Card 2"], ["Card 4", "Card 5"]]}}
    data = training_set(seeds=TAGGED, **both)
    comp = data.composition
    by_stream = {segment.stream: index for index, segment in enumerate(comp.segments)}
    for epoch in (0, 1, 5):
        data.set_epoch(epoch)
        selection = {name: data.selection[index] for name, index in by_stream.items()}
        assert selection["base"] is None                                    # draw = "all": nothing is drawn
        assert selection["comp"].tolist() == D.subset_draw(np.random.default_rng([7, 3, epoch]), 24, 24).tolist()
        assert selection["real"].tolist() == D.balanced_draw(np.random.default_rng([7, 4, epoch]), 6, 14).tolist()
        negatives = np.random.default_rng([7, 5, epoch])                    # one generator for the segment's two streams
        assert selection["neg_a"].tolist() == D.balanced_draw(negatives, 5, 6).tolist()
        assert selection["neg_b"].tolist() == D.balanced_draw(negatives, 3, 6).tolist()
        by_class = comp.segments[by_stream["each"]].by_class
        assert selection["each"] == D.per_class_draw(np.random.default_rng([7, 6, 1, epoch]), by_class, 4)
        boost = np.random.default_rng([7, 6, 2, epoch])
        assert data.boost_rows == [pool[int(i)] for pool, n in comp.boost_pools if n for i in D.balanced_draw(boost, len(pool), n)]
        pairs = np.random.default_rng([7, 6, 3, epoch])
        picks = pairs.integers(0, len(comp.pair_list), comp.n_units)
        expected = []
        for unit in range(comp.n_units):
            for name in comp.pair_list[int(picks[unit])]:
                pool = comp.pair_pools[name]
                expected.append(pool[int(pairs.integers(0, len(pool)))])
        assert data.pair_rows == expected and len(expected) == 2 * comp.n_units > 0

    # Other tags, other draws; the same tags stated again, the same draws.
    def drawn(dataset):
        dataset.set_epoch(2)
        return ([None if s is None else list(map(int, s)) for s in dataset.selection],
                [id(row) for _, row in dataset.boost_rows], [id(row) for _, row in dataset.pair_rows])

    assert drawn(training_set(seeds=TAGGED, **both)) == drawn(data)
    assert drawn(training_set(**both)) != drawn(data)                       # the defaults are other tags
    for name in TAGGED:
        changed = drawn(training_set(seeds={**TAGGED, name: 99}, **both))
        assert changed != drawn(data), name
    # A second table that draws the same way gets a stream of its own.
    twice = training_set(segments=[*SEGMENTS, {"stream": "real", "count": 14}], seeds=TAGGED)
    twice.set_epoch(1)
    first, second = (twice.selection[i].tolist() for i, s in enumerate(twice.composition.segments) if s.stream == "real")
    assert first == D.balanced_draw(np.random.default_rng([7, 4, 1]), 6, 14).tolist()
    assert second == D.balanced_draw(np.random.default_rng([7, 4, 1, 1]), 6, 14).tolist() != first


def test_an_index_stands_for_one_row_of_one_stream_per_epoch():
    data = training_set()
    assert len(data) == 158
    names = [data.source_of(i)[0] for i in range(len(data))]
    assert names == ["base"] * 100 + ["comp"] * 24 + ["real"] * 14 + ["neg_a"] * 6 + ["neg_b"] * 6 + ["each"] * 8
    assert [data.source_of(i)[1] for i in range(100)] == POOLS["base"].rows       # "all": every row once, in order
    composites = [id(data.source_of(i)[1]) for i in range(100, 124)]
    assert len(set(composites)) == 24                                             # a subset has no repeats
    real = Counter(id(data.source_of(i)[1]) for i in range(124, 138))
    assert len(real) == 6 and set(real.values()) <= {2, 3}                        # balanced: 14 rows over 6
    each = Counter(data.source_of(i)[1]["card_id"] for i in range(150, 158))
    assert each == {"C6": 4, "C7": 4}
    first = [id(data.source_of(i)[1]) for i in range(len(data))]
    data.set_epoch(1)
    second = [id(data.source_of(i)[1]) for i in range(len(data))]
    data.set_epoch(0)
    assert first[:100] == second[:100] and first[100:] != second[100:]
    assert first == [id(data.source_of(i)[1]) for i in range(len(data))]          # an epoch is reproducible


def test_an_example_is_a_pair_of_tensors_and_a_label_fixed_by_seed_epoch_and_index():
    data = training_set()
    crop, art, label = data[3]
    assert crop.shape == art.shape == (3, 128, 128) and crop.dtype == art.dtype == torch.float32
    assert -1.0 <= float(crop.min()) and float(crop.max()) <= 1.0
    assert label == D.class_list(data.composition).index(POOLS["base"].rows[3]["card_id"])
    again = data[3]
    assert torch.equal(crop, again[0]) and torch.equal(art, again[1]) and label == again[2]
    assert not torch.equal(crop, data[7][0])                                      # same card and picture, another draw
    data.set_epoch(1)
    assert not torch.equal(crop, data[3][0])
    data.set_epoch(0)
    assert torch.equal(crop, data[3][0])
    other_seed = training_set(seed=8)
    assert not torch.equal(crop, other_seed[3][0])


def test_negatives_carry_no_clean_image_and_backs_are_marked():
    data = training_set()
    for index, expected in ((138, NEGATIVE_BACK), (144, NEGATIVE)):
        crop, art, label = data[index]
        assert label == expected and not art.any() and crop.shape == (3, 128, 128)
    image, art, label = data.example(138, tensors=False)
    assert art is None and image.dtype == np.uint8 and image.ndim == 3


def test_a_component_draws_from_its_own_stream_and_leaves_the_rest_of_the_row_alone():
    three = ("occluders", "glare", "low_light")
    plain = training_set()
    silent = training_set(components={name: True for name in (*three, "rotation")},
                          augment={name: {"p": 0.0} for name in (*three, "rotation")})
    loud = training_set(components={name: True for name in three}, augment={name: {"p": 1.0} for name in three})
    differ = 0
    for index in (0, 50, 101, 130, 139, 152):
        a, b, c = plain.example(index, tensors=False), silent.example(index, tensors=False), loud.example(index, tensors=False)
        # Switched on with probability 0, the components change nothing: the base chain's draws are untouched.
        assert np.array_equal(a[0], b[0]) and a[2] == b[2] and (a[1] is None or np.array_equal(a[1], b[1]))
        # Switched on for every row, they change the crop and nothing else: same size, same clean image.
        assert a[0].shape == c[0].shape and (c[1] is None or np.array_equal(a[1], c[1]))
        differ += not np.array_equal(a[0], c[0])
    assert differ == 6
    info: dict = {}
    loud.example(0, tensors=False, info=info)
    assert set(info) == {"glare", "occluders", "low_light"} and all(value is not None for value in info.values())
    info = {}
    loud.example(139, tensors=False, info=info)                                    # a negative: no card laid over it
    assert info["occluders"]["kind"] != "card"
    # Rotation is applied before the crop is cut.
    turned = training_set(components={"rotation": True}, augment={"rotation": {"p": 1.0}})
    info = {}
    crop, _, _ = turned.example(101, tensors=False, info=info)
    assert abs(info["rotation"]) >= 15.0 and not np.array_equal(crop, plain.example(101, tensors=False)[0])


def test_card_images_become_crops_on_a_background():
    pools = {"base": Stream("base", [{"image": None, "from_art": True, "inner": [0, 0, 64, 90], "card_id": c, "group": "art"}
                                     for c in ("C0", "C2", "C8")], "physical", False, None, 0.4)}
    data = training_set(pools=pools, segments=[{"stream": "base", "count": 12}], batch=4,
                        augment={"physical": {"p_phys": 0.0}, "crop": {"frame_jitter": False, "bg_swap": False, "cutout": False,
                                                                       "res_jitter": False, "photometric": False, "jpeg": False}})
    data.physical = {**data.physical, "p_motion": 0.0}
    for index in range(12):
        crop, art, label = data.example(index, tensors=False)
        card_id = data.source_of(index)[1]["card_id"]
        assert np.array_equal(crop, ART[card_id]) and np.array_equal(art, ART[card_id])   # with every augmentation off: the card itself
        assert label == ["C0", "C2", "C8"].index(card_id)
    assert Counter(data.source_of(i)[1]["card_id"] for i in range(12)) == {"C0": 4, "C2": 4, "C8": 4}


def test_a_row_whose_box_lies_outside_its_image_is_refused():
    pools = streams_of(base=[{"image": IMAGE, "inner": [500, 500, 20, 20], "card_id": "C0", "group": "g"}] * 4)
    data = training_set(pools=pools, segments=[{"stream": "base", "draw": "all"}], batch=4)
    with pytest.raises(ValueError, match="lies outside its image"):
        data[0]


def test_the_loader_serves_the_sampler_s_batches_and_follows_the_epoch():
    cfg = config(SEGMENTS, components={"pairs": True}, pairs={"per_batch": 2, "names": [["Twin", "Card 2"]]})
    comp = D.compose(cfg, POOLS, CATALOG)
    data = D.TrainingSet(comp, POOLS, ART, PATCHES, D.class_list(comp), CATALOG, cfg, KindSettings())
    sampler = D.PairBatchSampler(comp.n_single, comp.n_units, comp.per_batch, 16, seed=7)
    loader = D.make_loader(data, sampler, workers=0)
    crops, arts, labels = next(iter(loader))
    assert crops.shape == arts.shape == (16, 3, 128, 128) and labels.shape == (16,)
    first = next(iter(sampler))
    assert labels.tolist() == [data[i][2] for i in first]
    names = {CATALOG.name_of(data.source_of(i)[1]["card_id"]) for i in first if i >= comp.n_single}
    assert names == {"Twin", "Card 2"}


def test_loader_processes_follow_the_epoch():
    """With loader processes, each process keeps its own copy of the dataset for the whole run;
    the epoch reaches them through shared memory."""
    cfg = config(SEGMENTS, components={**OFF, "pairs": True}, pairs={"per_batch": 2, "names": [["Twin", "Card 2"]]})
    comp = D.compose(cfg, POOLS, CATALOG)
    data = D.TrainingSet(comp, POOLS, ART, PATCHES, D.class_list(comp), CATALOG, cfg, KindSettings())
    sampler = D.PairBatchSampler(comp.n_single, comp.n_units, comp.per_batch, 16, seed=7)
    loader = D.make_loader(data, sampler, workers=2)
    seen = []
    for epoch in range(2):
        data.set_epoch(epoch)
        sampler.set_epoch(epoch)
        first = next(iter(sampler))
        crops, arts, labels = next(iter(loader))
        for position, index in enumerate(first):
            crop, art, label = data[index]                                        # the main process's copy, at this epoch
            assert torch.equal(crop, crops[position]) and torch.equal(art, arts[position]) and label == int(labels[position])
        seen.append(crops)
    assert seen[0].shape == seen[1].shape and not torch.equal(seen[0], seen[1])
    del loader
