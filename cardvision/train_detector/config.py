# SPDX-License-Identifier: Apache-2.0
"""The detector trainer's config file: one TOML or JSON file that states a whole recipe.

Every key is optional except the streams the mix draws from; a key left out takes the value
shown, which is the value in ``default.toml`` beside this module. Unknown keys are refused.

::

    seed = 1337              # fixes the weights' initialization and every tile
    steps = 500              # optimizer steps
    batch_size = 16          # training images per step
    store_width = 704        # the built-in streams' frames are resized to this width when read
    extra_sources = []       # plug-in streams, each "package.module:function"
    # init = "runs/other-game/last.pth"   # the weights to start from; see below

    [streams.real]           # labeled frames of public videos
    manifest = "train.manifest.jsonl"     # one path or a list of paths
    frames = "frames/"                    # the folder "cardvision fetch" wrote
    roles = ["train"]                     # which rows are used
    # trained_on = true                   # keep only rows with this value (default: either)
    # sources = "sources.json"            # the game's sources file, for its hand-row setting
    aspect_range = [0.55, 0.90]           # the label conversion's aspect filter; false: off

    [streams.composites]     # scenes written by "cardvision make-composites"
    folder = "composites/"

    [streams.negatives]      # your own images that show no cards
    folder = "negatives/"

    # Any stream, built-in or plug-in, may be given "degrade": the probability that a frame
    # drawn from it is blurred and re-compressed before use (default 0).
    # [streams.NAME]
    # degrade = 0.6

    [mix]
    # The first tile of each training image, by position in the batch. The tiles add up to
    # batch_size.
    primary = [
        { stream = "real", tiles = 7 },
        { stream = "negatives", tiles = 3 },
        { stream = "composites", tiles = 6 },
    ]
    # Where each of a mosaic's three further tiles comes from. The shares add up to 1.
    mosaic_extra = [
        { stream = "real", share = 0.4444 },
        { stream = "composites", share = 0.3704 },
        { stream = "negatives", share = 0.1852 },
    ]

    [augment]
    mosaic = 0.5             # probability that an image is a four-tile mosaic
    degrees = 10.0           # the mosaic's random rotation, up to this either way
    translate = 0.1          # its shift, as a fraction of the canvas, either way
    scale = [0.5, 1.5]       # its zoom
    shear = 2.0              # its shear in degrees, either way
    flip = 0.5               # probability of a left-right flip
    no_augment_share = 0.1   # the last share of the steps: no mosaic, and the L1 box loss on
    max_targets = 200        # faces and backs kept per image
    max_ignore = 480         # ignore regions kept per image

    [optimizer]
    lr = 0.001               # the peak learning rate
    momentum = 0.9
    weight_decay = 0.0005
    warmup_steps = 30        # linear rise to the peak
    final_lr_ratio = 0.05    # the cosine falls to this share of the peak, then holds
    ema_decay = 0.9998       # the averaged weights' decay

    [loss]
    ignore_overlap = 0.5     # see cardvision.train_detector.loss

    [loader]
    workers = 4              # processes that prepare tiles; the tiles do not depend on it
    threads = 8              # PyTorch's CPU threads
    salt = "tiles"           # mixed into every tile's seed; another value, another sequence

Paths are read relative to the directory the command is run from.

``init`` names the starting weights: the path of a detector checkpoint, or ``"random"`` for
untrained weights. Left out, as in the default file, the start is the public pretrained
checkpoint, which is the documented one. A checkpoint is a ``last.pth`` that this trainer
wrote, or the ``.safetensors`` file that ``cardvision export detector`` wrote; only weights
are read from it (``cardvision.train_detector.model`` has the rules). ``--init`` on the
command line names the same thing and takes the config's place.

**The mix is stated in counts and shares, never against the size of a stream**: a stream with
ten frames and one with ten thousand contribute exactly what the mix says, each drawn from
uniformly with replacement.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SHARE_TOLERANCE = 1e-6
BUILT_IN_STREAMS = ("real", "composites", "negatives")


class ConfigError(ValueError):
    """The config file does not have the documented shape. The message says where."""


@dataclass(frozen=True)
class Augment:
    """The ``[augment]`` table."""

    mosaic: float = 0.5
    degrees: float = 10.0
    translate: float = 0.1
    scale: tuple[float, float] = (0.5, 1.5)
    shear: float = 2.0
    flip: float = 0.5
    no_augment_share: float = 0.1
    max_targets: int = 200
    max_ignore: int = 480


@dataclass(frozen=True)
class Optimizer:
    """The ``[optimizer]`` table."""

    lr: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.0005
    warmup_steps: int = 30
    final_lr_ratio: float = 0.05
    ema_decay: float = 0.9998


@dataclass(frozen=True)
class Loader:
    """The ``[loader]`` table."""

    workers: int = 4
    threads: int = 8
    salt: str = "tiles"


DEFAULT_PRIMARY = (("real", 7), ("negatives", 3), ("composites", 6))
DEFAULT_MOSAIC_EXTRA = (("real", 0.4444), ("composites", 0.3704), ("negatives", 0.1852))


@dataclass(frozen=True)
class Config:
    """A parsed config file.

    Attributes:
        init: the weights to start from, as written: a checkpoint's path or ``"random"``;
            ``None`` for the public pretrained checkpoint.
        streams: the ``[streams.NAME]`` tables as written, by stream name.
        primary: ``(stream, tiles)`` in batch order.
        mosaic_extra: ``(stream, share)`` in the order the shares are stacked.
        raw: the whole document as read, which plug-in streams are handed.
    """

    seed: int = 1337
    steps: int = 500
    batch_size: int = 16
    store_width: int = 704
    init: str | None = None
    extra_sources: tuple[str, ...] = ()
    streams: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    primary: tuple[tuple[str, int], ...] = DEFAULT_PRIMARY
    mosaic_extra: tuple[tuple[str, float], ...] = DEFAULT_MOSAIC_EXTRA
    augment: Augment = Augment()
    optimizer: Optimizer = Optimizer()
    ignore_overlap: float = 0.5
    loader: Loader = Loader()
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False)

    @property
    def no_augment_from(self) -> int:
        """The first step (counted from 0) of the closing phase without mosaic."""
        return self.steps - round(self.augment.no_augment_share * self.steps)

    def stream_names(self) -> list[str]:
        """Every stream the mix draws from, in order of first mention."""
        return list(dict.fromkeys([name for name, _ in self.primary] + [name for name, _ in self.mosaic_extra]))

    def primary_order(self) -> list[str]:
        """The stream of each batch position's first tile."""
        return [name for name, tiles in self.primary for _ in range(tiles)]

    def with_changes(self, **changes: Any) -> Config:
        """A copy with some top-level values replaced, checked again."""
        values = {name: getattr(self, name) for name in self.__dataclass_fields__}
        values.update(changes)
        config = Config(**values)
        _check(config, "config")
        return config

    def to_json(self) -> dict[str, Any]:
        """The recipe as plain values, every default filled in, for the run record. ``init``
        is there when the config states it."""
        return {
            "seed": self.seed,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "store_width": self.store_width,
            **({} if self.init is None else {"init": self.init}),
            "extra_sources": list(self.extra_sources),
            "streams": {name: dict(options) for name, options in self.streams.items()},
            "mix": {
                "primary": [{"stream": name, "tiles": tiles} for name, tiles in self.primary],
                "mosaic_extra": [{"stream": name, "share": share} for name, share in self.mosaic_extra],
            },
            "augment": {**asdict(self.augment), "scale": list(self.augment.scale)},
            "optimizer": asdict(self.optimizer),
            "loss": {"ignore_overlap": self.ignore_overlap},
            "loader": asdict(self.loader),
        }


# ----------------------------------------------------------------------------------- reading
def read_document(path: str | Path) -> dict[str, Any]:
    """Read a ``.toml`` or ``.json`` config file into a dictionary."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path}: no such config file")
    if path.suffix.lower() == ".json":
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as error:
            raise ConfigError(f"{path}: not valid JSON ({error})") from None
    elif path.suffix.lower() == ".toml":
        try:
            import tomllib
        except ImportError:  # Python 3.10
            try:
                import tomli as tomllib
            except ImportError:
                raise ConfigError(
                    f"{path}: reading TOML on this Python needs the tomli package (pip install tomli), "
                    "or write the config as JSON"
                ) from None
        try:
            doc = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as error:
            raise ConfigError(f"{path}: not valid TOML ({error})") from None
    else:
        raise ConfigError(f"{path}: a config file ends in .toml or .json")
    if not isinstance(doc, dict):
        raise ConfigError(f"{path}: the config must be a table of keys")
    return doc


def _number(value: Any, where: str, low: float | None = None, high: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where} must be a number")
    if (low is not None and value < low) or (high is not None and value > high):
        raise ConfigError(f"{where} must be between {low} and {high}, got {value}")
    return float(value)


def _whole(value: Any, where: str, low: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise ConfigError(f"{where} must be a whole number of at least {low}")
    return value


def _table(doc: Mapping[str, Any], key: str, where: str) -> dict[str, Any]:
    value = doc.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: [{key}] must be a table")
    return value


def _no_unknown(table: Mapping[str, Any], known: Any, where: str) -> None:
    unknown = sorted(set(table) - set(known))
    if unknown:
        raise ConfigError(f"{where}: unknown key {', '.join(unknown)}")


def _mix_list(value: Any, amount: str, where: str) -> list[tuple[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{where} must be a list of {{ stream, {amount} }} entries")
    out = []
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != {"stream", amount} or not isinstance(entry["stream"], str):
            raise ConfigError(f"{where}: each entry is {{ stream = \"NAME\", {amount} = ... }}")
        out.append((entry["stream"], entry[amount]))
    names = [name for name, _ in out]
    if len(set(names)) != len(names):
        raise ConfigError(f"{where}: a stream is listed twice")
    return out


def _check(config: Config, where: str) -> None:
    _whole(config.seed, f"{where}: seed")
    _whole(config.steps, f"{where}: steps", 1)
    _whole(config.batch_size, f"{where}: batch_size", 1)
    _whole(config.store_width, f"{where}: store_width", 32)
    if config.init is not None and (not isinstance(config.init, str) or not config.init.strip()):
        raise ConfigError(f'{where}: init must be the path of a checkpoint to start from, or "random"')
    total = sum(tiles for _, tiles in config.primary)
    if total != config.batch_size:
        raise ConfigError(
            f"{where}: [mix] primary adds up to {total} tiles, and batch_size is {config.batch_size}. Every "
            "position in the batch needs a stream: if an optional stream was left out, share its tiles among "
            "the others (default.toml in cardvision/train_detector shows the released recipe treated this way)."
        )
    shares = sum(share for _, share in config.mosaic_extra)
    if abs(shares - 1.0) > SHARE_TOLERANCE:
        raise ConfigError(f"{where}: [mix] mosaic_extra adds up to {shares:.6g}, and the shares must add up to 1")
    if config.optimizer.warmup_steps < 1:
        raise ConfigError(f"{where}: [optimizer] warmup_steps must be at least 1")


def parse_config(doc: Mapping[str, Any], where: str = "config") -> Config:
    """Check a config document and return it as a ``Config``. Raises ``ConfigError``."""
    if not isinstance(doc, Mapping):
        raise ConfigError(f"{where}: the config must be a table of keys")
    _no_unknown(doc, {"seed", "steps", "batch_size", "store_width", "init", "extra_sources", "streams", "mix",
                      "augment", "optimizer", "loss", "loader"}, where)
    defaults = Config()
    extra = doc.get("extra_sources", [])
    if not isinstance(extra, list) or not all(isinstance(item, str) and ":" in item for item in extra):
        raise ConfigError(f'{where}: extra_sources must be a list of "package.module:function" strings')

    streams = _table(doc, "streams", where)
    for name, options in streams.items():
        if not isinstance(options, dict):
            raise ConfigError(f"{where}: [streams.{name}] must be a table")
        if "degrade" in options:
            _number(options["degrade"], f"{where}: [streams.{name}] degrade", 0.0, 1.0)

    mix = _table(doc, "mix", where)
    _no_unknown(mix, {"primary", "mosaic_extra"}, f"{where}: [mix]")
    primary = defaults.primary
    if "primary" in mix:
        primary = tuple((name, _whole(tiles, f"{where}: [mix] primary tiles of {name}", 1))
                        for name, tiles in _mix_list(mix["primary"], "tiles", f"{where}: [mix] primary"))
    mosaic_extra = defaults.mosaic_extra
    if "mosaic_extra" in mix:
        mosaic_extra = tuple((name, _number(share, f"{where}: [mix] mosaic_extra share of {name}", 0.0, 1.0))
                             for name, share in _mix_list(mix["mosaic_extra"], "share", f"{where}: [mix] mosaic_extra"))

    table = _table(doc, "augment", where)
    _no_unknown(table, Augment.__dataclass_fields__, f"{where}: [augment]")
    d = defaults.augment
    scale = table.get("scale", list(d.scale))
    if not isinstance(scale, list) or len(scale) != 2:
        raise ConfigError(f"{where}: [augment] scale must be [smallest, largest]")
    low = _number(scale[0], f"{where}: [augment] scale", 0.01)
    high = _number(scale[1], f"{where}: [augment] scale", low)
    augment = Augment(
        mosaic=_number(table.get("mosaic", d.mosaic), f"{where}: [augment] mosaic", 0.0, 1.0),
        degrees=_number(table.get("degrees", d.degrees), f"{where}: [augment] degrees", 0.0, 180.0),
        translate=_number(table.get("translate", d.translate), f"{where}: [augment] translate", 0.0, 1.0),
        scale=(low, high),
        shear=_number(table.get("shear", d.shear), f"{where}: [augment] shear", 0.0, 89.0),
        flip=_number(table.get("flip", d.flip), f"{where}: [augment] flip", 0.0, 1.0),
        no_augment_share=_number(table.get("no_augment_share", d.no_augment_share),
                                 f"{where}: [augment] no_augment_share", 0.0, 1.0),
        max_targets=_whole(table.get("max_targets", d.max_targets), f"{where}: [augment] max_targets", 1),
        max_ignore=_whole(table.get("max_ignore", d.max_ignore), f"{where}: [augment] max_ignore", 1),
    )

    table = _table(doc, "optimizer", where)
    _no_unknown(table, Optimizer.__dataclass_fields__, f"{where}: [optimizer]")
    d = defaults.optimizer
    optimizer = Optimizer(
        lr=_number(table.get("lr", d.lr), f"{where}: [optimizer] lr", 0.0),
        momentum=_number(table.get("momentum", d.momentum), f"{where}: [optimizer] momentum", 0.0, 1.0),
        weight_decay=_number(table.get("weight_decay", d.weight_decay), f"{where}: [optimizer] weight_decay", 0.0),
        warmup_steps=_whole(table.get("warmup_steps", d.warmup_steps), f"{where}: [optimizer] warmup_steps", 1),
        final_lr_ratio=_number(table.get("final_lr_ratio", d.final_lr_ratio),
                               f"{where}: [optimizer] final_lr_ratio", 0.0, 1.0),
        ema_decay=_number(table.get("ema_decay", d.ema_decay), f"{where}: [optimizer] ema_decay", 0.0, 1.0),
    )

    table = _table(doc, "loss", where)
    _no_unknown(table, {"ignore_overlap"}, f"{where}: [loss]")
    ignore_overlap = _number(table.get("ignore_overlap", defaults.ignore_overlap),
                             f"{where}: [loss] ignore_overlap", 0.0, 1.0)

    table = _table(doc, "loader", where)
    _no_unknown(table, Loader.__dataclass_fields__, f"{where}: [loader]")
    d = defaults.loader
    salt = table.get("salt", d.salt)
    if not isinstance(salt, str):
        raise ConfigError(f"{where}: [loader] salt must be a string")
    loader = Loader(workers=_whole(table.get("workers", d.workers), f"{where}: [loader] workers"),
                    threads=_whole(table.get("threads", d.threads), f"{where}: [loader] threads", 1), salt=salt)

    config = Config(
        seed=doc.get("seed", defaults.seed),
        steps=doc.get("steps", defaults.steps),
        batch_size=doc.get("batch_size", defaults.batch_size),
        store_width=doc.get("store_width", defaults.store_width),
        init=doc.get("init", defaults.init),
        extra_sources=tuple(extra),
        streams={name: dict(options) for name, options in streams.items()},
        primary=primary,
        mosaic_extra=mosaic_extra,
        augment=augment,
        optimizer=optimizer,
        ignore_overlap=ignore_overlap,
        loader=loader,
        raw=dict(doc),
    )
    _check(config, where)
    return config


def load_config(path: str | Path) -> Config:
    """Read and check a config file."""
    return parse_config(read_document(path), str(path))
