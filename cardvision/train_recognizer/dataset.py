# SPDX-License-Identifier: Apache-2.0
"""What an epoch is made of, how a row becomes a training example, and how batches are drawn.

The epoch
=========

``compose`` turns the config's ``[[segment]]`` list into row counts. A segment's rows are a
``count``, a multiple of the stated base count (``per_base``), a share of the epoch so far
(``share``), or so many per card (``per_class``); ``balanced_draw`` and its relatives say which
rows of the stream are used in a given epoch. Two components add rows after the segments:

``boost``
    Extra rows of alternate printings (cards whose catalog entry has ``printingOf``, under
    either identity rule), as a multiple of the rows they are expected to have anyway, and
    extra rows of one card kind, enough to bring it to a stated share of the positive rows.
``pairs``
    Pairs of rows of two easily confused card names. ``PairBatchSampler`` puts a fixed number
    of pairs in every batch, both rows of a pair together, so that the model always has the
    confusable card at hand as a negative. The config gives the pairs by name, so under the
    ``printing`` identity rule a name stands for every card that has it.

Neither component decides which cards are the same. Every card id is a class of its own
(``class_list``); which classes are one identity is the catalog's rule, and it enters
training in one place, the same-name mask of the losses (``cardvision.train_recognizer.loop``).

Every per-epoch draw comes from a NumPy generator seeded by the training seed, a tag for the
kind of draw and the epoch, so an epoch is reproducible whatever the number of loader
processes. The tags, and the offset in the batch sampler's seed, are the config's ``[seeds]``
table; left out, each is derived from its name.

A row
=====

``TrainingSet.example`` is the chain, in this order (components in brackets run only when
switched on, each from its own random stream)::

    load the image and the card's box
    physical streams:  physical chain (p_phys of rows), [rotation], frame jitter, background swap
    plain streams:     [rotation], frame jitter
    [glare]  [occluders]  cutout  resolution jitter  photometric  [low light]  motion blur  JPEG
    a negative: a random quarter turn, and no clean image
    a positive: its card's clean image through ``augment_art``

Both images then go through the preprocessing contract. A negative's label is ``NEGATIVE``, or
``NEGATIVE_BACK`` when its ``label`` marks a card back.

Loader processes keep their copy of the dataset from one epoch to the next, so the epoch number
lives in shared memory: ``set_epoch`` writes it in the main process and every process reads it
before serving a row.
"""

from __future__ import annotations

import math
import multiprocessing
import random
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from cardvision.augment import components as C
from cardvision.augment.crop import (
    CropAugment,
    augment_art,
    background_swap,
    center_mask,
    cutout,
    frame_jitter,
    jpeg,
    on_surface,
    photometric,
    resolution_jitter,
)
from cardvision.augment.physical import motion_blur, physical_chain
from cardvision.composites.cards import KindSettings
from cardvision.core.catalog import Catalog
from cardvision.core.preprocess import input_size, preprocessor
from cardvision.train_recognizer.config import AUGMENT_COMPONENTS, DEFAULTS, SAMPLER_OFFSET, ConfigError
from cardvision.train_recognizer.losses import NEGATIVE, NEGATIVE_BACK
from cardvision.train_recognizer.sources import Row, Stream, load_image, whole_inner

DEFAULT_SAMPLER_OFFSET: int = DEFAULTS["seeds"][SAMPLER_OFFSET]


def draw_tag(cfg: Mapping[str, Any], name: str) -> tuple[int, ...]:
    """The tag of one kind of per-epoch draw, from the config's ``[seeds]`` table. With the
    training seed and the epoch it decides which rows an epoch uses."""
    return tuple(cfg["seeds"][name])


# ------------------------------------------------------------------------------- drawing
def balanced_draw(rng: np.random.Generator, n_pool: int, n: int) -> np.ndarray:
    """``n`` indices into a pool of ``n_pool`` rows, every row the same number of times give or
    take one: whole random orders of the pool, one after another, cut off at ``n``."""
    reps = int(math.ceil(n / n_pool))
    return np.concatenate([rng.permutation(n_pool) for _ in range(reps)])[:n]


def subset_draw(rng: np.random.Generator, n_pool: int, n: int) -> np.ndarray:
    """``n`` different indices into a pool of ``n_pool`` rows."""
    return rng.choice(n_pool, n, replace=False)


def per_class_draw(rng: np.random.Generator, by_class: Mapping[str, list[int]], per_class: int) -> list[int]:
    """``per_class`` rows of every card, balanced within the card, cards in sorted order."""
    chosen: list[int] = []
    for card_id in sorted(by_class):
        pool = by_class[card_id]
        chosen += [pool[int(i)] for i in balanced_draw(rng, len(pool), per_class)]
    return chosen


def share_rows(share: float, rows_so_far: int) -> int:
    """How many rows make a segment the fraction ``share`` of the epoch up to and including
    it."""
    return int(round(share / (1 - share) * rows_so_far)) if share > 0 else 0


# --------------------------------------------------------------------------- composition
@dataclass
class Segment:
    """One stretch of an epoch's rows: ``n`` rows of one stream.

    ``group`` numbers the segments that share one random stream (the streams of one
    ``[[segment]]`` table). ``tag`` is that stream's tag, and ``repeat`` tells apart the
    streams of several tables that draw the same way (0 for the first)."""

    stream: str
    rows: list[Row]
    n: int
    draw: str
    negative: bool
    tag: tuple[int, ...]
    group: int
    repeat: int = 0
    per_class: int = 0
    by_class: dict[str, list[int]] = field(default_factory=dict)


@dataclass
class Composition:
    """An epoch's make-up, as ``compose`` works it out.

    Attributes:
        segments: the segments, in index order.
        boost_pools: ``[(rows, n)]`` for the boost component: ``n`` extra rows per epoch drawn
            evenly from ``rows``, a list of ``(stream name, row)``.
        pair_list: the pairs of card names that have rows on both sides.
        pair_pools: card name to its ``(stream name, row)`` list.
        n_units: pairs per epoch; they take the last ``2 * n_units`` indices.
        per_batch: pairs per batch (0 with the component off).
        report: the counts, for the log and the run record.
    """

    segments: list[Segment]
    boost_pools: list[tuple[list[tuple[str, Row]], int]] = field(default_factory=list)
    pair_list: list[tuple[str, str]] = field(default_factory=list)
    pair_pools: dict[str, list[tuple[str, Row]]] = field(default_factory=dict)
    n_units: int = 0
    per_batch: int = 0
    report: dict[str, Any] = field(default_factory=dict)

    @property
    def n_segment_rows(self) -> int:
        return sum(segment.n for segment in self.segments)

    @property
    def n_boost(self) -> int:
        return sum(n for _, n in self.boost_pools)

    @property
    def n_single(self) -> int:
        """Rows that are not part of a pair."""
        return self.n_segment_rows + self.n_boost

    def __len__(self) -> int:
        return self.n_single + 2 * self.n_units


def compose(cfg: Mapping[str, Any], streams: Mapping[str, Stream], catalog: Catalog) -> Composition:
    """Work out an epoch from the config and the streams' sizes.

    A segment whose stream has no rows gives none. A ``subset`` segment gives at most the
    whole stream. Raises ``ConfigError`` when ``draw = "all"`` states a count other than the
    stream's size.
    """
    base = cfg["epoch"]["base"]
    segments: list[Segment] = []
    used: Counter[tuple[int, ...]] = Counter()
    rows_so_far = 0
    report: dict[str, Any] = {"base": base, "segments": []}
    for group, spec in enumerate(cfg["segment"]):
        names = [name for name in spec["streams"] if name in streams and streams[name].rows]
        if "count" in spec:
            wanted = int(spec["count"])
        elif "per_base" in spec:
            wanted = int(round(spec["per_base"] * base))
        elif "share" in spec:
            wanted = share_rows(spec["share"], rows_so_far)
        else:
            wanted = None
        draw = spec["draw"]
        tag = () if draw == "all" else draw_tag(cfg, "negative" if draw == "balanced" and spec["negative"] else draw)
        repeat = 0
        if draw != "all" and names:
            repeat = used[tag]
            used[tag] += 1
        for position, name in enumerate(names):
            rows = streams[name].rows
            segment = Segment(stream=name, rows=rows, n=0, draw=draw, negative=spec["negative"], tag=tag, group=group,
                              repeat=repeat)
            if draw == "all":
                if wanted is not None and wanted != len(rows):
                    raise ConfigError(
                        f'stream {name!r} has {len(rows)} rows, but its segment with draw = "all" states {wanted}; '
                        f"state the stream's size (with per_base = 1.0, set [epoch] base = {len(rows)})"
                    )
                segment.n = len(rows)
            elif draw == "per_class":
                for index, row in enumerate(rows):
                    segment.by_class.setdefault(row["card_id"], []).append(index)
                segment.per_class = int(spec["per_class"])
                segment.n = segment.per_class * len(segment.by_class)
            else:
                # Streams that share a segment split its rows evenly; the last takes the remainder.
                part = wanted // len(names) if position < len(names) - 1 else wanted - (wanted // len(names)) * (len(names) - 1)
                segment.n = min(part, len(rows)) if draw == "subset" else part
            segments.append(segment)
            report["segments"].append({"stream": name, "pool": len(rows), "rows": segment.n, "draw": draw})
        missing = [name for name in spec["streams"] if name not in names]
        if missing:
            report.setdefault("emptyStreams", []).extend(missing)
        rows_so_far = sum(segment.n for segment in segments)

    composition = Composition(segments=segments, report=report)
    positives = [segment for segment in segments if not segment.negative]
    pools = [(segment.stream, row) for segment in _unique_pools(positives) for row in segment.rows]
    kind_of = {card.id: card.kind for card in catalog}

    if cfg["components"]["boost"]:
        boost = cfg["boost"]
        share_kind, share = boost["share_kind"], boost["share"]
        if share_kind is None or share is None:
            from_catalog = catalog.kinds.get("boost") or {}
            share_kind = share_kind if share_kind is not None else from_catalog.get("shareKind")
            share = share if share is not None else from_catalog.get("share")
        alternates = {card.id for card in catalog if card.printing_of is not None}
        expected: Counter[str] = Counter()
        for segment in positives:
            if segment.draw == "per_class":
                ids = list(segment.by_class)
                expected["kind"] += segment.per_class * sum(kind_of.get(c) == share_kind for c in ids)
                expected["alternate"] += segment.per_class * sum(c in alternates for c in ids)
            else:
                size = max(len(segment.rows), 1)
                expected["kind"] += sum(kind_of.get(r["card_id"]) == share_kind for r in segment.rows) / size * segment.n
                expected["alternate"] += sum(r["card_id"] in alternates for r in segment.rows) / size * segment.n
            expected["positive"] += segment.n
        alternate_pool = [p for p in pools if p[1]["card_id"] in alternates]
        n_alternate = int(round(boost["alternate_printings"] * expected["alternate"])) if alternate_pool else 0
        kind_pool = [p for p in pools if kind_of.get(p[1]["card_id"]) == share_kind] if share_kind is not None else []
        n_kind = 0
        if kind_pool and share:
            after = expected["positive"] + n_alternate
            n_kind = max(0, int(math.ceil((share * after - expected["kind"]) / (1 - share))))
        composition.boost_pools = [(alternate_pool, n_alternate), (kind_pool, n_kind)]
        report["boost"] = {"alternatePrintings": n_alternate, "alternatePool": len(alternate_pool),
                           "kind": share_kind, "kindRows": n_kind, "kindPool": len(kind_pool),
                           "kindShareBefore": round(expected["kind"] / max(expected["positive"], 1), 4),
                           "kindShareTarget": share}

    if cfg["components"]["pairs"]:
        per_batch = cfg["pairs"]["per_batch"]
        by_name: dict[str, list[tuple[str, Row]]] = {}
        for pool in pools:
            by_name.setdefault(catalog.name_of(pool[1]["card_id"]), []).append(pool)
        pair_list = [(a, b) for a, b in cfg["pairs"]["names"] if by_name.get(a) and by_name.get(b)]
        composition.pair_list = pair_list
        composition.pair_pools = {name: by_name[name] for pair in pair_list for name in pair}
        if pair_list:
            composition.per_batch = per_batch
            composition.n_units = per_batch * (composition.n_single // (cfg["batch"] - 2 * per_batch))
        report["pairs"] = {"pairs": [list(pair) for pair in pair_list], "perBatch": composition.per_batch,
                           "dropped": [list(pair) for pair in cfg["pairs"]["names"] if tuple(pair) not in pair_list],
                           "poolRows": {name: len(rows) for name, rows in composition.pair_pools.items()},
                           "perEpoch": composition.n_units}
    report.update({"rowsPerEpoch": len(composition), "boostRows": composition.n_boost, "pairRows": 2 * composition.n_units})
    return composition


def _unique_pools(segments: list[Segment]) -> list[Segment]:
    """One segment per stream, in order of first use: a stream used by two segments is one
    pool of rows."""
    seen: set[str] = set()
    out = []
    for segment in segments:
        if segment.stream not in seen:
            seen.add(segment.stream)
            out.append(segment)
    return out


def class_list(composition: Composition) -> list[str]:
    """The training cards, one class each: the first positive stream's card ids in sorted
    order, then each later stream's new ids in sorted order."""
    cards: list[str] = []
    known: set[str] = set()
    for segment in _unique_pools([s for s in composition.segments if not s.negative]):
        new = sorted({row["card_id"] for row in segment.rows} - known)
        cards += new
        known.update(new)
    return cards


# ------------------------------------------------------------------------------ dataset
class TrainingSet(torch.utils.data.Dataset):
    """The rows of one epoch, served as ``(crop tensor, clean-image tensor, label)``.

    Args:
        composition: the epoch's make-up.
        streams: the streams, by name.
        art: card id to card image.
        patches: background patches.
        classes: the training cards; a card's label is its position.
        catalog: for each card's kind.
        cfg: the resolved config.
        kinds: the game's kind settings, for the physical chain.
    """

    def __init__(self, composition: Composition, streams: Mapping[str, Stream], art: Mapping[str, np.ndarray],
                 patches: list[np.ndarray], classes: list[str], catalog: Catalog, cfg: Mapping[str, Any],
                 kinds: KindSettings | None = None) -> None:
        self.composition = composition
        self.streams = dict(streams)
        self.art = dict(art)
        self.art_ids = sorted(self.art)
        self.patches = patches
        self.label_of = {card_id: index for index, card_id in enumerate(classes)}
        self.kind_of = {card.id: card.kind for card in catalog}
        self.kinds = kinds if kinds is not None else KindSettings()
        self.seed = cfg["seed"]
        self.boost_tag, self.pairs_tag = draw_tag(cfg, "boost"), draw_tag(cfg, "pairs")
        self.back_labels = frozenset(cfg["loss"]["back_labels"])
        self.hooks = {name: bool(cfg["components"][name]) for name in AUGMENT_COMPONENTS}
        self.settings = {name: cfg["augment"][name] for name in AUGMENT_COMPONENTS}
        self.physical = cfg["augment"]["physical"]
        crop = cfg["augment"]["crop"]
        physical_jpeg = {"jpeg_p": self.physical["jpeg_p"], "jpeg_q": self.physical["jpeg_q"]}
        # Three settings of the plain chain: crops of a physical stream, the other crops (no
        # background swap), and clean card images (the plain chain's own JPEG settings).
        self.physical_aug = CropAugment(**{**crop, **physical_jpeg})
        self.plain_aug = CropAugment(**{**crop, **physical_jpeg, "bg_swap": False})
        self.art_aug = CropAugment(**crop)
        self.prepare = preprocessor(cfg["contract"])
        size = input_size(cfg["contract"])
        self.zero_art = torch.zeros(3, size, size)
        self.offsets = np.cumsum([0] + [segment.n for segment in composition.segments])
        self._shared_epoch = multiprocessing.RawValue("q", 0)
        self.epoch = -1
        self.set_epoch(0)

    # ------------------------------------------------------------------------- epochs
    def _plan(self, epoch: int) -> None:
        comp = self.composition
        self.selection: list[Any] = [None] * len(comp.segments)
        generators: dict[int, np.random.Generator] = {}
        for index, segment in enumerate(comp.segments):
            if segment.draw == "all" or segment.n == 0:
                continue
            if segment.group not in generators:
                key = [self.seed, *segment.tag, epoch] + ([segment.repeat] if segment.repeat else [])
                generators[segment.group] = np.random.default_rng(key)
            rng = generators[segment.group]
            if segment.draw == "subset":
                self.selection[index] = subset_draw(rng, len(segment.rows), segment.n)
            elif segment.draw == "per_class":
                self.selection[index] = per_class_draw(rng, segment.by_class, segment.per_class)
            else:
                self.selection[index] = balanced_draw(rng, len(segment.rows), segment.n)
        self.boost_rows: list[tuple[str, Row]] = []
        if comp.n_boost:
            rng = np.random.default_rng([self.seed, *self.boost_tag, epoch])
            self.boost_rows = [pool[int(i)] for pool, n in comp.boost_pools if n for i in balanced_draw(rng, len(pool), n)]
        self.pair_rows: list[tuple[str, Row]] = []
        if comp.n_units:
            rng = np.random.default_rng([self.seed, *self.pairs_tag, epoch])
            picks = rng.integers(0, len(comp.pair_list), comp.n_units)
            for unit in range(comp.n_units):
                for name in comp.pair_list[int(picks[unit])]:
                    pool = comp.pair_pools[name]
                    self.pair_rows.append(pool[int(rng.integers(0, len(pool)))])
        self.epoch = epoch

    def set_epoch(self, epoch: int) -> None:
        """Move to an epoch (in the main process); loader processes follow."""
        self._plan(epoch)
        self._shared_epoch.value = epoch

    def _sync_epoch(self) -> None:
        epoch = self._shared_epoch.value
        if epoch != self.epoch:
            self._plan(epoch)

    def __len__(self) -> int:
        return len(self.composition)

    # -------------------------------------------------------------------------- items
    def source_of(self, index: int) -> tuple[str, Row]:
        """The stream and row that an index of the current epoch stands for."""
        comp = self.composition
        if index < comp.n_segment_rows:
            position = int(np.searchsorted(self.offsets, index, side="right")) - 1
            segment = comp.segments[position]
            j = index - int(self.offsets[position])
            chosen = self.selection[position]
            return segment.stream, segment.rows[j if chosen is None else int(chosen[j])]
        j = index - comp.n_segment_rows
        if j < comp.n_boost:
            return self.boost_rows[j]
        return self.pair_rows[j - comp.n_boost]

    def example(self, index: int, tensors: bool = True, info: dict | None = None):
        """The training example at an index of the current epoch.

        Returns ``(crop, clean image, label)``: tensors made by the preprocessing contract, or,
        with ``tensors=False``, the uint8 images before it (the clean image is None for a
        negative). ``info``, when given, receives what each component drew.
        """
        self._sync_epoch()
        name, row = self.source_of(index)
        stream = self.streams[name]
        info = info if info is not None else {}
        rng = random.Random(C.row_seed(self.seed, self.epoch, index))
        if row.get("from_art"):
            surface = C.stream_for(self.seed, self.epoch, index, "surface")
            patch = self.patches[surface.randrange(len(self.patches))] if self.patches else None
            img, inner = on_surface(self.art[row["card_id"]], patch, stream.surface_margin, surface)
        else:
            img = load_image(row, stream.root)
            inner = whole_inner(row["inner"], img.shape[1], img.shape[0])
        if inner[2] < 1 or inner[3] < 1:
            raise ValueError(f"stream {name!r}: a row's inner box {row['inner']} lies outside its image")
        negative = stream.negative
        if stream.chain == "physical":
            if rng.random() < self.physical["p_phys"]:
                img, inner = physical_chain(img, inner, self.kind_of.get(row["card_id"]), rng, self.kinds, self.physical)
            patch = self.patches[rng.randrange(len(self.patches))] if self.patches else None
            if self.hooks["rotation"]:
                img, inner, info["rotation"] = C.rotation(
                    img, inner, C.stream_for(self.seed, self.epoch, index, "rotation"), self.settings["rotation"])
            c = frame_jitter(img, inner, rng, self.physical_aug)
            c = background_swap(c, patch, rng, self.physical_aug)
            aug = self.physical_aug
        else:
            if self.hooks["rotation"]:
                img, inner, info["rotation"] = C.rotation(
                    img, inner, C.stream_for(self.seed, self.epoch, index, "rotation"), self.settings["rotation"])
            c = frame_jitter(img, inner, rng, self.plain_aug)
            aug = self.plain_aug
        if self.hooks["glare"]:
            c, info["glare"] = C.glare(c, C.stream_for(self.seed, self.epoch, index, "glare"), self.settings["glare"])
        if self.hooks["occluders"]:
            own = C.stream_for(self.seed, self.epoch, index, "occluders")
            other = self.art[self.art_ids[own.randrange(len(self.art_ids))]]
            c, info["occluders"] = C.occluders(c, own, self.settings["occluders"], other_art=other, negative=negative)
        c = cutout(c, rng, aug)
        c = resolution_jitter(c, rng, aug)
        c = photometric(c, rng, aug)
        if self.hooks["low_light"]:
            c, info["low_light"] = C.low_light(
                c, C.stream_for(self.seed, self.epoch, index, "low_light"), self.settings["low_light"])
        c = motion_blur(c, rng, self.physical)
        c = jpeg(c, rng, aug)
        c = center_mask(c, aug)
        if negative:
            c = np.ascontiguousarray(np.rot90(c, rng.randrange(4)))
            label = NEGATIVE_BACK if row.get("label") in self.back_labels else NEGATIVE
            if not tensors:
                return c, None, label
            return torch.from_numpy(self.prepare(c)), self.zero_art, label
        a = augment_art(self.art[row["card_id"]], rng, self.art_aug)
        label = self.label_of[row["card_id"]]
        if not tensors:
            return c, a, label
        return torch.from_numpy(self.prepare(c)), torch.from_numpy(self.prepare(a)), label

    def __getitem__(self, index: int):
        return self.example(index)


# ------------------------------------------------------------------------------ batches
class PairBatchSampler(torch.utils.data.Sampler):
    """Batches of ``batch - 2 * per_batch`` single rows and ``per_batch`` pairs.

    Single rows are the indices below ``n_single``; pair ``u`` is the two indices
    ``n_single + 2u`` and ``n_single + 2u + 1``, which always land in the same batch. The
    singles, the pairs and the order inside each batch are shuffled by a generator seeded with
    ``seed * 1000 + offset + epoch``, where ``offset`` is the config's ``[seeds]
    sampler_offset``. Rows that do not fill a batch are left out of the epoch. With
    ``per_batch`` 0 this is plain shuffled batching.
    """

    def __init__(self, n_single: int, n_units: int, per_batch: int, batch: int, seed: int,
                 offset: int = DEFAULT_SAMPLER_OFFSET) -> None:
        self.n_single, self.n_units, self.P, self.batch, self.seed = n_single, n_units, per_batch, batch, seed
        self.offset = offset
        self.epoch = 0
        self.nb = min(n_single // (batch - 2 * per_batch), n_units // per_batch) if per_batch else n_single // batch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.nb

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 1000 + self.offset + self.epoch)
        singles = torch.randperm(self.n_single, generator=g).tolist()
        units = torch.randperm(self.n_units, generator=g).tolist() if self.P else []
        k = self.batch - 2 * self.P
        for b in range(self.nb):
            idx = singles[b * k:(b + 1) * k]
            for u in units[b * self.P:(b + 1) * self.P]:
                idx += [self.n_single + 2 * u, self.n_single + 2 * u + 1]
            perm = torch.randperm(len(idx), generator=g).tolist()
            yield [idx[p] for p in perm]


def make_loader(dataset: TrainingSet, sampler: PairBatchSampler, workers: int) -> torch.utils.data.DataLoader:
    """The training loader: the batch sampler's batches, with loader processes that live
    through the whole run."""
    return torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=workers,
                                       persistent_workers=workers > 0)
