# SPDX-License-Identifier: Apache-2.0
"""The recognizer trainer's configuration: one TOML file over the defaults below.

A config file states what differs from ``DEFAULTS``; a key that ``DEFAULTS`` does not have is
an error, so a misspelled key cannot be silently ignored. Paths are resolved against the folder
of the config file. ``default.toml``, beside this module, writes the defaults out as a file to
copy; ``games/riftbound/train-recognizer.toml`` is a full recipe.

Top level
    ``seed`` (weights, batch order, augmentation and the per-epoch draws), ``data_seed`` (the
    validation split and the sample of unseen crops; the training seed when left out),
    ``epochs``, ``batch``, ``workers`` (loader processes) and ``contract`` (the preprocessing
    contract of ``cardvision.core.preprocess``).
``[model]``
    ``init``, the encoder's starting weights, and ``dim``, the embedding width. ``init`` is
    ``"imagenet"`` (the default: pretrained weights), ``"random"``, or the path of a
    recognizer checkpoint to start from: a ``best.pt`` or ``last.pt`` that this trainer wrote,
    or the ``.safetensors`` file that ``cardvision export recognizer`` wrote. A value is read
    as a path when it ends in ``.pt`` or ``.safetensors``; any other value must be one of the
    two words. Only the encoder's weights are taken from the file, and they must fit this
    config's ``dim`` exactly (``cardvision.train_recognizer.model`` describes all three).
``[optimizer]``
    AdamW's ``lr`` and ``weight_decay``, and ``warmup_epochs``: the learning rate rises in a
    straight line over that many epochs and then follows half a cosine down to zero.
``[loss]``
    ``arcface_scale``, ``arcface_margin``, ``arcface_weight``; ``contrastive_temperature``,
    ``contrastive_weight``; ``negative_weight`` and ``negative_margin`` for the hinge on
    negatives; ``back_weight`` and ``back_margin`` for negatives whose ``label`` is one of
    ``back_labels`` (set ``back_margin`` to a negative number to treat every negative alike);
    ``bank_refresh_steps``, how often the memory bank of clean-image embeddings is recomputed.
``[data]``
    ``catalog``; ``kinds`` (a kinds file to use in place of the catalog's ``kinds``);
    ``art`` (the folder of card images); ``manifests`` and ``frames`` (label manifests and the
    folder ``cardvision fetch`` wrote), ``roles`` (which rows are used) and ``sources`` (the
    game's sources file, for the hand-row rule); ``composites`` (a ``make-composites`` output
    folder); ``textures`` (a ``fetch-textures`` folder, for background patches); ``margin``
    (the room around a card in a cut crop, as a fraction of its width and height on each
    side); ``extra_sources`` (plug-in streams, see ``cardvision.train_recognizer.sources``).
``[background_patches]``
    ``count`` patches of ``size`` px square are made once, ``texture_share`` of them from the
    textures folder when there is one and the rest drawn procedurally. ``folder`` (optional,
    a path relative to the config file) supplies the patches instead: every .png, .jpg, .jpeg
    or .webp file in it (not its subfolders), taken in byte-wise order of file name, decoded
    to RGB and not resized. Each must be exactly ``size`` x ``size`` px. ``count``, when given,
    must equal the number of files; when left out it is that number. ``texture_share`` is
    then unused.
``[validation]``
    ``groups`` (recording groups whose real crops are held out) or ``group_share`` (that share
    of the groups, chosen by ``data_seed``); ``unseen_cards`` (card ids kept out of training
    altogether) or ``unseen_pick`` (that many cards chosen by ``data_seed``);
    ``unseen_count`` (how many unseen crops are scored); ``composites`` (another
    ``make-composites`` folder to take the unseen crops from; the training one when left out);
    ``every`` (the grid: every that many epochs and the last one); ``at_start`` (also score
    the encoder before training); ``log_unseen_every_epoch`` (score the unseen set on the
    other epochs too, for the curve only); ``source`` (a plug-in that replaces both sets).
``[epoch]``
    ``base``: a number of rows. Every ``per_base`` below is a multiple of it. It has no
    default: an epoch's size is stated, never taken from the size of some stream.
``[streams.NAME]``
    How rows of a stream are treated: ``chain`` (``"physical"``: the physical chain and the
    background swap; ``"plain"``: neither), ``negative`` (its rows show no card face), and for
    card images ``surface_margin`` (the background placed around the image).

    ``[streams.composite_positives]`` also says which faces of a composite scene become crops
    (``cardvision.train_recognizer.sources.composite_crops``). A face that the scene's labels
    mark positive and that names a card is cut when all three hold:

    * ``min_visible`` (default 0.90): its ``visible`` share is at least this;
    * ``min_short_side`` (default 48): its short side, ``short``, is at least this many px;
    * ``max_tilt`` (default 15): its ``angle`` is at most this many degrees from the nearest
      quarter turn (0, 90, 180 or 270), so that the crop, turned upright by quarter turns,
      shows the card nearly straight.

    The same three decide the composite crops of the unseen validation set. Set them to 0, 0
    and 45 to cut every positive face.
``[[segment]]``
    The epoch, in order. Each segment names a ``stream`` (or, for negatives, ``streams``, which
    share its rows evenly), how many rows it gives per epoch and how they are drawn:

    * ``count = N`` rows, or ``per_base = X`` (``round(X * base)``), or ``share = F`` (as many
      rows as make the segment the fraction ``F`` of the epoch up to and including it:
      ``round(F / (1 - F) * rows so far)``), or ``per_class = K`` (``K`` rows of every card);
    * ``draw = "all"`` (every row once, in order; a stated count must equal the stream's
      size), ``"subset"`` (without repeats; at most the whole stream), ``"balanced"`` (every row
      the same number of times, give or take one) or ``"per_class"`` (with ``per_class``).
``[components]``
    Switches: ``occluders``, ``glare``, ``low_light``, ``rotation`` (augmentations on every
    row), ``printing_mask`` (same-name masking in the losses), ``pairs`` and ``boost``.
``[pairs]``
    ``names``: pairs of card names that are easily confused; ``per_batch`` such pairs are put
    in every batch, one row of each name. A pair is named by card name, so two different cards
    of one name cannot be asked for as a pair under the ``printing`` identity rule.
``[boost]``
    ``alternate_printings``: extra rows of alternate printings, as a multiple of the rows they
    have anyway; ``share_kind`` and ``share``: extra rows of that card kind until it makes up
    that share of the positive rows. Left out, the kind and share come from the catalog's
    ``kinds.boost``.
``[augment.crop]``, ``[augment.physical]``, ``[augment.occluders]``, ``[augment.glare]``, ``[augment.low_light]``, ``[augment.rotation]``
    Overrides of the augmentation settings (``cardvision.augment``).
``[seeds]``
    The tags that keep the trainer's random streams apart. Which rows an epoch uses is drawn
    from a generator seeded with the training seed, the tag of the kind of draw and the epoch.
    There is one tag per kind of draw: ``subset`` and ``balanced`` (segments drawn that way),
    ``negative`` (balanced segments of negative streams), ``per_class``, ``boost`` and
    ``pairs``. A tag is a whole number, or a short list of whole numbers. ``sampler_offset``
    is a whole number: the batch sampler shuffles with a generator seeded with
    ``seed * 1000 + sampler_offset + epoch``.

    Left out, each of these is derived from its own name (``default_tag``), which keeps the
    streams apart and means nothing else. They are settings so that a run made with other
    tags can be repeated draw for draw: put that run's values in a config of your own, as
    with the detector trainer's ``[loader] salt``. Changing a tag changes which rows each
    epoch draws and nothing about what is trained on.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
from pathlib import Path
from typing import Any

from cardvision.augment.components import DEFAULTS as COMPONENT_DEFAULTS
from cardvision.augment.crop import CropAugment
from cardvision.augment.physical import PHYSICAL
from cardvision.core.preprocess import CONTRACTS, FULL128

CHAINS = ("physical", "plain")
# The starting weights that are named by a word; any other ``init`` is a checkpoint's path.
INIT_WORDS = ("imagenet", "random")
CHECKPOINT_SUFFIXES = (".pt", ".safetensors")
DRAWS = ("all", "subset", "balanced", "per_class")
COUNT_KEYS = ("count", "per_base", "share", "per_class")
AUGMENT_COMPONENTS = ("occluders", "glare", "low_light", "rotation")
COMPONENTS = (*AUGMENT_COMPONENTS, "printing_mask", "pairs", "boost")

# The stream names the trainer fills itself. Any other name must come from a plug-in.
ART = "art"
COMPOSITE_POSITIVES = "composite_positives"
REAL_POSITIVES = "real_positives"
REAL_NEGATIVES = "real_negatives"
BUILT_IN_STREAMS = (ART, COMPOSITE_POSITIVES, REAL_POSITIVES, REAL_NEGATIVES)

# Which positive faces of a composite scene become crops: the keys of
# ``[streams.composite_positives]`` that say so, and their defaults (see the module docstring).
COMPOSITE_SELECTION: dict[str, Any] = {"min_visible": 0.90, "min_short_side": 48, "max_tilt": 15.0}

# The kinds of per-epoch draw, each with a random stream of its own (see ``[seeds]``).
DRAW_TAGS = ("subset", "balanced", "negative", "per_class", "boost", "pairs")
SAMPLER_OFFSET = "sampler_offset"


def default_tag(name: str) -> int:
    """The tag a random stream has when the config states none: the first four bytes of the
    SHA-256 of its name, read little-endian. Different names give different streams, and the
    number means nothing else."""
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "little")


DEFAULTS: dict[str, Any] = {
    "seed": 0,
    "data_seed": None,
    "epochs": 34,
    "batch": 256,
    "workers": 4,
    "contract": FULL128,
    "model": {"init": "imagenet", "dim": 128},
    "optimizer": {"lr": 3e-4, "weight_decay": 1e-4, "warmup_epochs": 3},
    "loss": {
        "arcface_scale": 32.0, "arcface_margin": 0.2, "arcface_weight": 1.0,
        "contrastive_temperature": 0.07, "contrastive_weight": 0.5,
        "negative_weight": 1.0, "negative_margin": 0.55,
        "back_weight": 4.0, "back_margin": 0.50, "back_labels": ["back"],
        "bank_refresh_steps": 50,
    },
    "data": {
        "catalog": None, "kinds": None, "sources": None, "art": None,
        "manifests": [], "frames": None, "roles": ["train"],
        "composites": None, "textures": None, "margin": 0.4, "extra_sources": [],
    },
    "background_patches": {"count": 256, "size": 320, "texture_share": 0.5, "folder": None},
    "validation": {
        "groups": [], "group_share": 0.10, "unseen_cards": [], "unseen_pick": 0, "unseen_count": 400,
        "composites": None, "every": 5, "at_start": True, "log_unseen_every_epoch": False, "source": None,
    },
    "epoch": {"base": None},
    "streams": {
        ART: {"chain": "physical", "negative": False, "surface_margin": 0.4},
        COMPOSITE_POSITIVES: {"chain": "physical", "negative": False, **COMPOSITE_SELECTION},
        REAL_POSITIVES: {"chain": "plain", "negative": False},
        REAL_NEGATIVES: {"chain": "plain", "negative": True},
    },
    "segment": [
        {"stream": ART, "per_base": 1.0, "draw": "balanced"},
        {"stream": COMPOSITE_POSITIVES, "per_base": 0.30, "draw": "subset"},
        {"stream": REAL_POSITIVES, "share": 0.10, "draw": "balanced"},
        {"streams": [REAL_NEGATIVES], "share": 0.08, "draw": "balanced"},
    ],
    "components": {"occluders": True, "glare": True, "low_light": True, "rotation": False,
                   "printing_mask": True, "pairs": False, "boost": False},
    "pairs": {"per_batch": 4, "names": []},
    "boost": {"alternate_printings": 1.0, "share_kind": None, "share": None},
    "seeds": {**{name: [default_tag(name)] for name in DRAW_TAGS}, SAMPLER_OFFSET: default_tag(SAMPLER_OFFSET)},
    "augment": {
        "crop": {},
        "physical": {},
        **{name: {} for name in AUGMENT_COMPONENTS},
    },
}

_STREAM_KEYS = {"chain", "negative", "surface_margin"}
_SEGMENT_KEYS = {"stream", "streams", "draw", *COUNT_KEYS}
_PATH_KEYS = (("data", "catalog"), ("data", "kinds"), ("data", "sources"), ("data", "art"), ("data", "frames"),
              ("data", "composites"), ("data", "textures"), ("validation", "composites"),
              ("background_patches", "folder"))


class ConfigError(ValueError):
    """The config does not have the documented shape. The message says where."""


def is_checkpoint(init: Any) -> bool:
    """Whether an ``init`` value names a checkpoint file: text that ends in ``.pt`` or
    ``.safetensors``. Nothing is looked up on disk."""
    return isinstance(init, str) and init not in INIT_WORDS and init.endswith(CHECKPOINT_SUFFIXES)


def check_init(init: Any, where: str = "[model] init") -> None:
    """Raise ``ConfigError`` unless ``init`` is one of the two words or a checkpoint's path."""
    if init not in INIT_WORDS and not is_checkpoint(init):
        raise ConfigError(f'{where} must be "imagenet", "random" or the path of a checkpoint to start from '
                          f'(a file ending in {" or ".join(CHECKPOINT_SUFFIXES)}), got {init!r}')


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        import tomllib
    except ImportError:  # Python 3.10
        try:
            import tomli as tomllib
        except ImportError:
            raise RuntimeError(
                'reading a TOML config on Python 3.10 needs tomli, part of an optional dependency group: '
                'python -m pip install -e ".[train]" in the repository folder'
            ) from None
    with open(path, "rb") as handle:
        try:
            return tomllib.load(handle)
        except tomllib.TOMLDecodeError as error:
            raise ConfigError(f"{path.name}: not valid TOML: {error}") from None


def _merge(base: dict[str, Any], over: dict[str, Any], where: str, free: bool = False) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in over.items():
        if key not in base and not free:
            raise ConfigError(f"{where}: unknown key {key!r} (known: {', '.join(sorted(base))})")
        if isinstance(base.get(key), dict) and key not in ("streams", "augment"):
            if not isinstance(value, dict):
                raise ConfigError(f"{where}: {key!r} must be a table")
            out[key] = _merge(base[key], value, f"{where}[{key}]")
        else:
            out[key] = copy.deepcopy(value)
    return out


def _number(value: Any, where: str, low: float | None = None, high: float | None = None, whole: bool = False) -> None:
    ok = isinstance(value, int if whole else (int, float)) and not isinstance(value, bool)
    if ok and low is not None:
        ok = value >= low
    if ok and high is not None:
        ok = value <= high
    if not ok:
        kind = "a whole number" if whole else "a number"
        span = "" if low is None and high is None else f" from {low} to {high}" if high is not None else f" of at least {low}"
        raise ConfigError(f"{where} must be {kind}{span}, got {value!r}")


def _text_list(value: Any, where: str) -> None:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ConfigError(f"{where} must be a list of strings")


def _check_streams(streams: Any, defaults: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(streams, dict):
        raise ConfigError("[streams] must be a table of tables")
    out = copy.deepcopy(defaults)
    for name, settings in streams.items():
        if not isinstance(settings, dict):
            raise ConfigError(f"[streams.{name}] must be a table")
        known = _STREAM_KEYS | set(COMPOSITE_SELECTION) if name == COMPOSITE_POSITIVES else _STREAM_KEYS
        unknown = sorted(set(settings) - known)
        if unknown:
            raise ConfigError(f"[streams.{name}]: unknown key(s) {unknown} (known: {sorted(known)})")
        merged = {"chain": "plain", "negative": False, **out.get(name, {}), **settings}
        if merged["chain"] not in CHAINS:
            raise ConfigError(f'[streams.{name}]: "chain" must be one of {", ".join(CHAINS)}')
        if not isinstance(merged["negative"], bool):
            raise ConfigError(f'[streams.{name}]: "negative" must be true or false')
        if "surface_margin" in merged:
            _number(merged["surface_margin"], f"[streams.{name}] surface_margin", 0, 2)
        if name == COMPOSITE_POSITIVES:
            _number(merged["min_visible"], f"[streams.{name}] min_visible", 0, 1)
            _number(merged["min_short_side"], f"[streams.{name}] min_short_side", 0)
            _number(merged["max_tilt"], f"[streams.{name}] max_tilt", 0, 45)
        out[name] = merged
    return out


def _check_segments(segments: Any, streams: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(segments, list) or not segments:
        raise ConfigError("[[segment]]: at least one segment is needed")
    out = []
    for position, segment in enumerate(segments):
        where = f"[[segment]] number {position + 1}"
        if not isinstance(segment, dict):
            raise ConfigError(f"{where} must be a table")
        unknown = sorted(set(segment) - _SEGMENT_KEYS)
        if unknown:
            raise ConfigError(f"{where}: unknown key(s) {unknown} (known: {sorted(_SEGMENT_KEYS)})")
        if ("stream" in segment) == ("streams" in segment):
            raise ConfigError(f'{where}: give either "stream" or "streams"')
        names = [segment["stream"]] if "stream" in segment else segment["streams"]
        _text_list(names, f"{where}: the stream names")
        if not names:
            raise ConfigError(f'{where}: "streams" names no stream')
        for name in names:
            if name not in streams:
                raise ConfigError(f"{where}: stream {name!r} has no [streams.{name}] table")
        kinds = {bool(streams[name]["negative"]) for name in names}
        if len(kinds) != 1:
            raise ConfigError(f"{where}: positive and negative streams cannot share a segment")
        counts = [key for key in COUNT_KEYS if key in segment]
        draw = segment.get("draw", "per_class" if counts == ["per_class"] else "balanced")
        if draw not in DRAWS:
            raise ConfigError(f'{where}: "draw" must be one of {", ".join(DRAWS)}')
        if len(counts) > 1 or (not counts and draw != "all"):
            raise ConfigError(f'{where}: give exactly one of {", ".join(COUNT_KEYS)} (only draw = "all" may give none)')
        if (draw == "per_class") != (counts == ["per_class"]):
            raise ConfigError(f'{where}: "per_class" and draw = "per_class" go together')
        if counts:
            key = counts[0]
            if key in ("count", "per_class"):
                _number(segment[key], f"{where}: {key}", 0, whole=True)
            elif key == "per_base":
                _number(segment[key], f"{where}: per_base", 0)
            else:
                _number(segment[key], f"{where}: share", 0, 0.95)
        if len(names) > 1 and draw not in ("balanced",):
            raise ConfigError(f'{where}: several streams can only share a "balanced" segment')
        out.append({"streams": list(names), "draw": draw, "negative": kinds.pop(),
                    **{key: segment[key] for key in counts}})
    return out


def _check_augment(augment: Any) -> dict[str, dict[str, Any]]:
    known = {"crop": {field.name: field.default for field in dataclasses.fields(CropAugment)},
             "physical": dict(PHYSICAL), **{name: dict(COMPONENT_DEFAULTS[name]) for name in AUGMENT_COMPONENTS}}
    if not isinstance(augment, dict):
        raise ConfigError("[augment] must be a table of tables")
    out: dict[str, dict[str, Any]] = {}
    for name, defaults in known.items():
        over = augment.get(name, {})
        if not isinstance(over, dict):
            raise ConfigError(f"[augment.{name}] must be a table")
        unknown = sorted(set(over) - set(defaults))
        if unknown:
            raise ConfigError(f"[augment.{name}]: unknown key(s) {unknown} (known: {sorted(defaults)})")
        merged = {**defaults, **over}
        # TOML has no tuples: a range arrives as a two-item list.
        out[name] = {key: (tuple(value) if isinstance(defaults[key], tuple) else value) for key, value in merged.items()}
    unknown = sorted(set(augment) - set(known))
    if unknown:
        raise ConfigError(f"[augment]: unknown table(s) {unknown} (known: {sorted(known)})")
    return out


def parse_config(doc: dict[str, Any], folder: str | Path = ".", source: str = "config") -> dict[str, Any]:
    """Check a parsed config document and return it merged over ``DEFAULTS``, with its paths
    resolved against ``folder``. Raises ``ConfigError`` at the first problem."""
    if not isinstance(doc, dict):
        raise ConfigError(f"{source}: the config must be a table")
    cfg = _merge(DEFAULTS, doc, source)
    folder = Path(folder)

    _number(cfg["seed"], "seed", 0, whole=True)
    if cfg["data_seed"] is None:
        cfg["data_seed"] = cfg["seed"]
    _number(cfg["data_seed"], "data_seed", 0, whole=True)
    _number(cfg["epochs"], "epochs", 1, whole=True)
    _number(cfg["batch"], "batch", 2, whole=True)
    _number(cfg["workers"], "workers", 0, whole=True)
    if cfg["contract"] not in CONTRACTS:
        raise ConfigError(f'contract must be one of {", ".join(sorted(CONTRACTS))}, got {cfg["contract"]!r}')
    check_init(cfg["model"]["init"])
    if is_checkpoint(cfg["model"]["init"]) and not Path(cfg["model"]["init"]).is_absolute():
        cfg["model"]["init"] = str((folder / cfg["model"]["init"]).resolve())
    _number(cfg["model"]["dim"], "[model] dim", 2, whole=True)
    _number(cfg["optimizer"]["lr"], "[optimizer] lr", 0)
    _number(cfg["optimizer"]["weight_decay"], "[optimizer] weight_decay", 0)
    _number(cfg["optimizer"]["warmup_epochs"], "[optimizer] warmup_epochs", 0)
    loss = cfg["loss"]
    for key in ("arcface_scale", "arcface_margin", "arcface_weight", "contrastive_weight", "negative_weight",
                "negative_margin", "back_weight", "back_margin"):
        _number(loss[key], f"[loss] {key}")
    _number(loss["contrastive_temperature"], "[loss] contrastive_temperature", 1e-6)
    _number(loss["bank_refresh_steps"], "[loss] bank_refresh_steps", 1, whole=True)
    _text_list(loss["back_labels"], "[loss] back_labels")

    data = cfg["data"]
    for table, key in _PATH_KEYS:
        value = cfg[table][key]
        if value is not None:
            if not isinstance(value, str) or not value:
                raise ConfigError(f"[{table}] {key} must be a path")
            cfg[table][key] = str((folder / value).resolve()) if not Path(value).is_absolute() else value
    _text_list(data["manifests"], "[data] manifests")
    data["manifests"] = [str((folder / item).resolve()) if not Path(item).is_absolute() else item
                         for item in data["manifests"]]
    _text_list(data["roles"], "[data] roles")
    _text_list(data["extra_sources"], "[data] extra_sources")
    _number(data["margin"], "[data] margin", 0, 2)
    if data["catalog"] is None or data["art"] is None:
        raise ConfigError("[data] catalog and art are required")
    if data["manifests"] and data["frames"] is None:
        raise ConfigError("[data] frames is required with manifests")

    patches = cfg["background_patches"]
    if patches["folder"] is not None and "count" not in doc.get("background_patches", {}):
        patches["count"] = None          # with a folder, an absent count is the number of files
    else:
        _number(patches["count"], "[background_patches] count", 0, whole=True)
    _number(patches["size"], "[background_patches] size", 16, whole=True)
    _number(patches["texture_share"], "[background_patches] texture_share", 0, 1)

    val = cfg["validation"]
    _text_list(val["groups"], "[validation] groups")
    _number(val["group_share"], "[validation] group_share", 0, 0.9)
    _text_list(val["unseen_cards"], "[validation] unseen_cards")
    _number(val["unseen_pick"], "[validation] unseen_pick", 0, whole=True)
    if val["unseen_cards"] and val["unseen_pick"]:
        raise ConfigError("[validation]: give unseen_cards or unseen_pick, not both")
    _number(val["unseen_count"], "[validation] unseen_count", 0, whole=True)
    _number(val["every"], "[validation] every", 1, whole=True)
    for key in ("at_start", "log_unseen_every_epoch"):
        if not isinstance(val[key], bool):
            raise ConfigError(f"[validation] {key} must be true or false")
    if val["source"] is not None and not isinstance(val["source"], str):
        raise ConfigError('[validation] source must be "package.module:function"')

    if cfg["epoch"]["base"] is None:
        raise ConfigError("[epoch] base is required: the number of rows the per_base sizes are multiples of")
    _number(cfg["epoch"]["base"], "[epoch] base", 1, whole=True)

    cfg["streams"] = _check_streams(doc.get("streams", {}), DEFAULTS["streams"])
    cfg["segment"] = _check_segments(cfg["segment"], cfg["streams"])
    cfg["augment"] = _check_augment(doc.get("augment", {}))

    for name in COMPONENTS:
        if not isinstance(cfg["components"][name], bool):
            raise ConfigError(f"[components] {name} must be true or false")
    pairs = cfg["pairs"]
    _number(pairs["per_batch"], "[pairs] per_batch", 1, whole=True)
    if not isinstance(pairs["names"], list) or not all(
            isinstance(pair, list) and len(pair) == 2 and all(isinstance(name, str) and name for name in pair)
            for pair in pairs["names"]):
        raise ConfigError("[pairs] names must be a list of pairs of card names")
    if cfg["components"]["pairs"] and cfg["batch"] <= 2 * pairs["per_batch"]:
        raise ConfigError("[pairs] per_batch pairs do not fit in a batch")
    boost = cfg["boost"]
    _number(boost["alternate_printings"], "[boost] alternate_printings", 0)
    if boost["share"] is not None:
        _number(boost["share"], "[boost] share", 0, 0.95)
    if boost["share_kind"] is not None and not isinstance(boost["share_kind"], str):
        raise ConfigError("[boost] share_kind must be a kind name")
    seeds = cfg["seeds"]
    for name in DRAW_TAGS:
        tag = seeds[name] if isinstance(seeds[name], list) else [seeds[name]]
        if not tag:
            raise ConfigError(f"[seeds] {name} must be a whole number or a list of whole numbers, got an empty list")
        for part in tag:
            _number(part, f"[seeds] {name}", 0, whole=True)
        seeds[name] = list(tag)
    _number(seeds[SAMPLER_OFFSET], f"[seeds] {SAMPLER_OFFSET}", 0, whole=True)
    return cfg


def load_config(path: str | Path) -> dict[str, Any]:
    """Read and check a config file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path}: no such config file")
    return parse_config(_load_toml(path), path.parent, path.name)


def plain(value: Any) -> Any:
    """A config value as JSON-ready data: tuples become lists."""
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value
