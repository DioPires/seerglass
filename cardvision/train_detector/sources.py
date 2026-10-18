# SPDX-License-Identifier: Apache-2.0
"""Where the detector trainer's frames come from: named streams behind one interface.

The trainer never opens a data file itself. It asks this module for *streams*, and builds each
training image out of tiles drawn from them in the proportions the config's ``[mix]`` states.
Nothing else in the trainer knows what a stream holds or where it came from.

The stream format
=================

A stream is a ``Stream``: a name, an indexable of frames and, for each frame, its targets.

``frames``
    Anything with ``len()`` and ``[i]``. ``frames[i]`` is one frame as a ``(H, W, 3)`` uint8
    array in red-green-blue order, any size. Frames may differ in size.
``targets``
    Anything with ``len()`` and ``[i]``, as long as ``frames``. ``targets[i]`` is a list with
    one dictionary per target of frame ``i``::

        {"cx": 0.412, "cy": 0.630, "w": 0.081, "h": 0.197, "k": 0}

    ``cx``, ``cy``: the center of the target's upright box; ``w``, ``h``: its width and height.
    All four are fractions of the frame's width and height, so they hold at any resolution.
    ``k`` is what the target is: ``0`` a card face, ``1`` a card back, ``2`` an ignore region
    (see ``cardvision.train_detector.loss``). An empty list is a frame with nothing in it.
    **A frame must be labeled completely**: a card that has no target is learned as background.
    Mark what you cannot or will not label as an ignore region.
``degrade``
    The probability that a frame drawn from the stream is blurred and re-compressed before
    use (``cardvision.train_detector.augment.degrade``); 0 by default. It suits frames that
    are cleaner than the video the detector will see. ``[streams.NAME] degrade`` in the config
    overrides it.
``record``
    A JSON-ready dictionary of whatever identifies the stream's contents (counts, file
    hashes). It is copied into the run record.

A frame is fetched once for every tile that uses it, in loader worker processes when the
config asks for workers. A stream is copied into each worker, so it should hold file paths
and open files lazily; ``NpyFrames`` and ``ImageFrames`` below do.

The built-in streams
====================

``real``
    Labeled frames of public videos: the rows of a label manifest that are boxed
    exhaustively, with their frames from the folder ``cardvision fetch`` wrote. Targets come
    from ``cardvision.data.detector_targets``. Rows whose frame is absent are left out and
    counted.
``composites``
    A folder written by ``cardvision make-composites``. A face the generator marks positive is
    a face; any other face is an ignore region; a back at least half visible is a back and a
    more hidden one an ignore region; everything else in a scene is background.
``negatives``
    A folder of your own images that show no cards: the images directly in it, by name.

Each is built only if the mix names it.

Adding a stream
===============

List a function in the config, ``extra_sources = ["package.module:function"]``. It is called
once, with the whole config document as a dictionary (so it can read its own settings from a
table of its choosing, for example ``[streams.NAME]``), and returns one ``Stream`` or several::

    from cardvision.train_detector.sources import Stream

    def my_frames(config):
        frames = [...]                      # (H, W, 3) uint8 RGB arrays, or a lazy reader
        targets = [[{"cx": .5, "cy": .5, "w": .1, "h": .2, "k": 0}], ...]
        return Stream("mine", frames, targets, record={"frames": len(frames)})

Then give the stream tiles in ``[mix]`` by its name. A plug-in stream is used exactly like a
built-in one; its name may not be one of theirs.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cardvision.core.constants import DETECTOR_BACK_CLASS, DETECTOR_FACE_CLASS
from cardvision.data.targets import DETECTOR_IGNORE_CODE
from cardvision.train_detector.config import BUILT_IN_STREAMS, Config, ConfigError

FACE = DETECTOR_FACE_CLASS
BACK = DETECTOR_BACK_CLASS
IGNORE = DETECTOR_IGNORE_CODE
TARGET_CODES = (FACE, BACK, IGNORE)
TARGET_KEYS = ("cx", "cy", "w", "h", "k")

# A back in a generated scene counts as a back when at least this share of it is visible.
BACK_VISIBLE_MIN = 0.5
# Decoded frames of the "real" stream kept in memory per process.
REAL_CACHE_FRAMES = 256


class SourceError(ValueError):
    """A stream could not be built or does not have the documented format."""


@dataclass(frozen=True)
class Stream:
    """One named source of training frames. See the module docstring for the format."""

    name: str
    frames: Any
    targets: Any
    degrade: float = 0.0
    record: Mapping[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.targets)


# ------------------------------------------------------------------------------ frame readers
class ImageFrames:
    """Image files read on demand as RGB arrays, resized to one width.

    Args:
        paths: the files, in order.
        width: every frame is resized to this width with Pillow's bilinear filter, keeping
            its shape. ``None`` leaves frames as they are.
        draft: let the JPEG decoder scale a large file down by a power of two before the
            resize. This is faster and gives slightly different pixels from a full decode.
        cache: how many decoded frames to keep in memory in each process.
    """

    def __init__(self, paths: Sequence[str | Path], width: int | None = None, draft: bool = False,
                 cache: int = 0) -> None:
        self.paths = [str(path) for path in paths]
        self.width = width
        self.draft = draft
        self.cache = cache
        self._kept: OrderedDict[int, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self.paths)

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_kept": OrderedDict()}

    def _read(self, index: int) -> np.ndarray:
        with Image.open(self.paths[index]) as image:
            if self.width is None:
                return np.asarray(image.convert("RGB"))
            height = round(image.height * self.width / image.width)
            if self.draft:
                image.draft("RGB", (self.width, height))
            return np.asarray(image.convert("RGB").resize((self.width, height), Image.BILINEAR))

    def __getitem__(self, index: int) -> np.ndarray:
        if not self.cache:
            return self._read(index)
        if index not in self._kept:
            self._kept[index] = self._read(index)
            if len(self._kept) > self.cache:
                self._kept.popitem(last=False)
        return self._kept[index]


class NpyFrames:
    """Frames stored together in one ``.npy`` file of shape ``(N, H, W, 3)``, uint8, RGB.

    The file is memory-mapped on first use in each process, so a stream that holds one of
    these can be handed to loader workers at no cost.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._array: np.ndarray | None = None

    @property
    def array(self) -> np.ndarray:
        if self._array is None:
            self._array = np.load(self.path, mmap_mode="r")
        return self._array

    def __getstate__(self) -> dict[str, Any]:
        return {"path": self.path, "_array": None}

    def __len__(self) -> int:
        return len(self.array)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.array[index]


class Subset:
    """Some items of an indexable, in a given order."""

    def __init__(self, items: Any, indices: Iterable[int]) -> None:
        self.items = items
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        return self.items[self.indices[index]]


class Concat:
    """Several indexables presented as one, end to end."""

    def __init__(self, parts: Sequence[Any]) -> None:
        self.parts = list(parts)
        self.offsets = np.cumsum([0] + [len(part) for part in self.parts])

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, index: int) -> Any:
        part = int(np.searchsorted(self.offsets, index, side="right") - 1)
        return self.parts[part][index - int(self.offsets[part])]


# ------------------------------------------------------------------------------------ hashes
def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def files_sha256(paths: Iterable[str | Path]) -> str:
    """One SHA-256 over a list of files: each file's name and the hash of its bytes, in the
    order given. Folders above the files do not enter into it."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(f"{Path(path).name}\t{file_sha256(path)}\n".encode())
    return digest.hexdigest()


# ------------------------------------------------------------------------- label conversions
def composite_targets(scene: Mapping[str, Any]) -> list[dict[str, float | int]]:
    """The targets of one generated scene (a line of a composites folder's ``labels.jsonl``).

    A face is a face when the generator marked it ``positive`` and an ignore region otherwise
    (too hidden or too small to ask for). A back is a back when at least half of it is
    visible and it carries no ``ignore`` mark, and an ignore region otherwise. Distractors,
    dice, hands and broadcast graphics give no target: they are background.
    """
    targets = []
    for item in scene["objects"]:
        if item["kind"] == "face":
            kind = FACE if item["positive"] else IGNORE
        elif item["kind"] == "back":
            kind = BACK if (item.get("visible") or 0) >= BACK_VISIBLE_MIN and not item.get("ignore") else IGNORE
        else:
            continue
        targets.append({"cx": item["cx"], "cy": item["cy"], "w": item["w"], "h": item["h"], "k": kind})
    return targets


def count_targets(targets: Iterable[Iterable[Mapping[str, Any]]]) -> dict[str, int]:
    """How many faces, backs and ignore regions a stream's targets hold."""
    counts = {FACE: 0, BACK: 0, IGNORE: 0}
    for frame in targets:
        for target in frame:
            counts[int(target["k"])] += 1
    return {"faces": counts[FACE], "backs": counts[BACK], "ignoreRegions": counts[IGNORE]}


# ---------------------------------------------------------------------------- built-in streams
def _options(config: Config, name: str, required: Sequence[str], optional: Sequence[str]) -> dict[str, Any]:
    options = dict(config.streams.get(name, {}))
    missing = [key for key in required if key not in options]
    if missing:
        raise ConfigError(f"[streams.{name}] needs {', '.join(missing)}: the mix draws tiles from this stream")
    unknown = sorted(set(options) - set(required) - set(optional) - {"degrade"})
    if unknown:
        raise ConfigError(f"[streams.{name}]: unknown key {', '.join(unknown)}")
    return options


def real_stream(config: Config) -> Stream:
    """The ``real`` stream: exhaustively boxed manifest rows and their fetched frames."""
    from cardvision.data.extract import FrameMissing, FrameStore, frame_check
    from cardvision.data.manifest import iter_rows, read_manifest
    from cardvision.data.sources import load_source_config
    from cardvision.data.targets import DETECTOR_CODES, TargetRules, detector_targets

    options = _options(config, "real", ("manifest", "frames"), ("roles", "trained_on", "sources", "aspect_range"))
    manifests = [options["manifest"]] if isinstance(options["manifest"], str) else list(options["manifest"])
    roles = options.get("roles", ["train"])
    trained_on = options.get("trained_on")
    aspect = options.get("aspect_range", list(TargetRules().aspect_range))
    if aspect is not False and not (isinstance(aspect, list) and len(aspect) == 2
                                    and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in aspect)):
        raise ConfigError("[streams.real] aspect_range is [lowest, highest], or false for no filter")
    if not isinstance(roles, list) or not (trained_on is None or isinstance(trained_on, bool)):
        raise ConfigError("[streams.real] roles is a list of roles and trained_on is true or false")
    settings: dict[str, Any] = {"aspect_range": None if aspect is False else (float(aspect[0]), float(aspect[1]))}
    if options.get("sources"):
        rules = TargetRules.for_sources(load_source_config(options["sources"]), **settings)
    else:
        rules = TargetRules(**settings)

    rows = []
    for manifest in manifests:
        rows.extend(iter_rows(read_manifest(manifest), role=roles, completeness="exhaustive-boxes",
                              trained_on=trained_on))
    store = FrameStore(options["frames"])
    paths, targets, missing = [], [], 0
    for row in rows:
        try:
            path = store.path(row)
        except FrameMissing:
            missing += 1
            continue
        # The same test of the frame's shape that the store applies when it loads a frame; only
        # the file's header is read here.
        with Image.open(path) as image:
            if frame_check(row, image.width, image.height, store.tolerance) is not None:
                missing += 1
                continue
        paths.append(path)
        targets.append([target.as_dict(DETECTOR_CODES) for target in detector_targets(row, rules)])
    if not paths:
        raise SourceError(
            f"the real stream is empty: {len(rows)} matching row(s) in {', '.join(map(str, manifests))}, "
            f"{missing} of them without a usable frame in {options['frames']}"
        )
    record = {
        "manifests": [{"file": Path(m).name, "sha256": file_sha256(m)} for m in manifests],
        "roles": list(roles), "trainedOn": trained_on,
        "sources": None if not options.get("sources") else {"file": Path(options["sources"]).name,
                                                           "sha256": file_sha256(options["sources"])},
        "aspectRange": None if rules.aspect_range is None else list(rules.aspect_range),
        "handRowSourceTypes": list(rules.hand_row_source_types), "handRowMinCenterY": rules.hand_row_min_center_y,
        "rows": len(rows), "frames": len(paths), "framesMissing": missing,
        "framesSha256": files_sha256(paths), **count_targets(targets),
    }
    frames = ImageFrames(paths, width=config.store_width, cache=REAL_CACHE_FRAMES)
    return Stream("real", frames, targets, record=record)


def composites_stream(config: Config) -> Stream:
    """The ``composites`` stream: a folder written by ``cardvision make-composites``."""
    from cardvision.composites.dataset import LABELS_NAME, MANIFEST_NAME, read_labels

    options = _options(config, "composites", ("folder",), ())
    folder = Path(options["folder"])
    if not (folder / LABELS_NAME).is_file():
        raise SourceError(f"{folder}: no {LABELS_NAME}; the composites stream reads a folder written by "
                          "cardvision make-composites")
    scenes = read_labels(folder)
    if not scenes:
        raise SourceError(f"{folder}: {LABELS_NAME} lists no scene")
    paths = [folder / scene["file"] for scene in scenes]
    absent = [path.name for path in paths if not path.is_file()]
    if absent:
        raise SourceError(f"{folder}: {len(absent)} scene image(s) are absent, for example {absent[0]}")
    targets = [composite_targets(scene) for scene in scenes]
    record: dict[str, Any] = {"scenes": len(scenes), "labelsSha256": file_sha256(folder / LABELS_NAME),
                              **count_targets(targets)}
    if (folder / MANIFEST_NAME).is_file():
        generated = json.loads((folder / MANIFEST_NAME).read_text(encoding="utf-8"))
        record["generator"] = {key: generated.get(key) for key in ("format", "formatVersion", "split", "seed")}
        record["manifestSha256"] = file_sha256(folder / MANIFEST_NAME)
    frames = ImageFrames(paths, width=config.store_width, draft=True)
    return Stream("composites", frames, targets, record=record)


def negatives_stream(config: Config) -> Stream:
    """The ``negatives`` stream: the images directly in a folder, none of which shows a card."""
    from cardvision.infer.frames import IMAGE_SUFFIXES

    options = _options(config, "negatives", ("folder",), ())
    folder = Path(options["folder"])
    if not folder.is_dir():
        raise SourceError(f"{folder}: no such folder of card-free images")
    paths = sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise SourceError(f"{folder}: the folder holds no image files")
    record = {"frames": len(paths), "framesSha256": files_sha256(paths)}
    return Stream("negatives", ImageFrames(paths, width=config.store_width), [[] for _ in paths], record=record)


BUILDERS: dict[str, Callable[[Config], Stream]] = {
    "real": real_stream,
    "composites": composites_stream,
    "negatives": negatives_stream,
}
assert tuple(BUILDERS) == BUILT_IN_STREAMS


# ------------------------------------------------------------------------------------ plug-ins
def check_stream(stream: Any) -> Stream:
    """Refuse a stream that does not have the documented format. Looks at every target and at
    the first frame."""
    if not isinstance(stream, Stream):
        raise SourceError(f"a stream function must return Stream objects, got {type(stream).__name__}")
    name = stream.name
    if not isinstance(name, str) or not name:
        raise SourceError("a stream needs a name")
    try:
        frames, targets = len(stream.frames), len(stream.targets)
    except TypeError:
        raise SourceError(f'stream "{name}": frames and targets must both have a length') from None
    if frames != targets or frames == 0:
        raise SourceError(f'stream "{name}": {frames} frame(s) and {targets} target list(s); they must be equal '
                          "in number and not empty")
    if not 0.0 <= float(stream.degrade) <= 1.0:
        raise SourceError(f'stream "{name}": degrade is a probability, got {stream.degrade}')
    first = np.asarray(stream.frames[0])
    if first.ndim != 3 or first.shape[2] != 3 or first.dtype != np.uint8 or 0 in first.shape:
        raise SourceError(f'stream "{name}": a frame is an (H, W, 3) uint8 RGB array, got shape {first.shape} '
                          f"and dtype {first.dtype}")
    for index in range(targets):
        for target in stream.targets[index]:
            if not isinstance(target, Mapping) or not all(key in target for key in TARGET_KEYS):
                raise SourceError(f'stream "{name}", frame {index}: a target is a dictionary with the keys '
                                  f"{', '.join(TARGET_KEYS)}")
            if target["k"] not in TARGET_CODES:
                raise SourceError(f'stream "{name}", frame {index}: k is 0 (face), 1 (back) or 2 (ignore region), '
                                  f"got {target['k']!r}")
            box = [target[key] for key in TARGET_KEYS[:4]]
            if not all(isinstance(value, (int, float)) and np.isfinite(value) for value in box) \
                    or target["w"] <= 0 or target["h"] <= 0:
                raise SourceError(f'stream "{name}", frame {index}: cx, cy, w and h are finite fractions of the '
                                  f"frame with a positive width and height, got {box}")
    return stream


def extra_streams(config: Config) -> list[Stream]:
    """Call the config's ``extra_sources`` functions and collect the streams they return."""
    from cardvision.export.checkpoint import load_factory

    streams: list[Stream] = []
    for spec in config.extra_sources:
        try:
            function = load_factory(spec)
        except ValueError as error:
            raise SourceError(f"extra_sources: {error}") from None
        returned = function(dict(config.raw))
        for stream in [returned] if isinstance(returned, Stream) else list(returned or []):
            check_stream(stream)
            if stream.name in BUILT_IN_STREAMS:
                raise SourceError(f'extra_sources: {spec} returned a stream named "{stream.name}", which is the '
                                  "name of a built-in stream")
            if any(stream.name == other.name for other in streams):
                raise SourceError(f'extra_sources: two streams are named "{stream.name}"')
            streams.append(stream)
    return streams


def build_streams(config: Config) -> dict[str, Stream]:
    """Every stream the config's mix draws from, by name, in the mix's order.

    Built-in streams are built from their ``[streams.NAME]`` tables; the others must come from
    ``extra_sources``. A ``degrade`` set in a stream's table replaces the stream's own.
    """
    extra = {stream.name: stream for stream in extra_streams(config)}
    streams: dict[str, Stream] = {}
    for name in config.stream_names():
        if name in BUILDERS:
            stream = check_stream(BUILDERS[name](config))
        elif name in extra:
            stream = extra[name]
        else:
            known = ", ".join([*BUILT_IN_STREAMS, *extra])
            raise ConfigError(f'[mix] names the stream "{name}", which is not built in and was not returned by an '
                              f"extra_sources function (known streams: {known})")
        degrade = config.streams.get(name, {}).get("degrade")
        streams[name] = stream if degrade is None else replace(stream, degrade=float(degrade))
    unused = sorted(set(extra) - set(streams))
    if unused:
        raise ConfigError(f"extra_sources returned {', '.join(unused)}, which [mix] gives no tiles")
    return streams


def stream_records(streams: Mapping[str, Stream]) -> dict[str, Any]:
    """What goes into the run record about each stream."""
    return {name: {"frames": len(stream), "degrade": stream.degrade, **dict(stream.record)}
            for name, stream in streams.items()}
