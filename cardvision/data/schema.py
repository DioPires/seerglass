# SPDX-License-Identifier: Apache-2.0
"""The label manifest: one JSON object per labeled video frame, one object per line.

This module is the format's definition in code. ``ROW_SCHEMA`` is a JSON Schema for one row
(``data/manifest.schema.json`` is a copy of it, kept equal by a test); ``Row``, ``Box`` and
``Point`` are the same thing as Python objects; ``row_problems`` says what is wrong with a
candidate row. ``data/SCHEMA.md`` explains every field in prose.

A manifest holds labels and the address of the frame they describe: a platform, a video id and
a time. It never holds pixels.
"""

from __future__ import annotations

import datetime
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from cardvision.core.catalog import CARD_ID_PATTERN  # the card-id rule is stated once, with the catalog
from cardvision.data import geometry
from cardvision.data.dhash import DHASH_PATTERN

MANIFEST_VERSION = 1

# Where a video is hosted. A row may also name a recording of your own, which is on no platform:
# its platform is LOCAL_PLATFORM and its video id a name you choose for it. ROW_PLATFORMS is every
# value a row's ``platform`` may have.
PLATFORMS = ("youtube", "twitch")
LOCAL_PLATFORM = "local"
ROW_PLATFORMS = (*PLATFORMS, LOCAL_PLATFORM)
ROLES = ("train", "evaluation", "extra")
COMPLETENESS = ("exhaustive-boxes", "points")
KINDS = ("face", "back", "unreadable", "not-card")
IDENTITIES = ("named", "unknown", "unlabeled")
READABLE = ("yes", "maybe", "no")
SHAPES = ("rotated-box", "quad")

# Where the footage comes from. The two families are fixed; a value may add one detail after a
# slash. The details below are built in; a game adds its own (its digital clients, for
# instance) through the ``source_types`` argument of the validator.
SOURCE_FAMILIES = ("physical", "digital")
SOURCE_TYPES = ("physical", "physical/tournament", "physical/webcam", "physical/phone", "digital")
SOURCE_TYPE_PATTERN = r"^(physical|digital)(/[a-z0-9]+(-[a-z0-9]+)*)?$"

# A quad's corner may lie outside the frame when a card runs past the frame's edge, so its
# coordinates are allowed this much room around the 0 to 1 range. A point is always inside.
QUAD_COORDINATE_RANGE = (-1.0, 2.0)

# The video id of a local recording is a name: 1 to 64 letters, digits, underscores and hyphens,
# the first a letter or a digit. It names no file; a file is given for it when frames are wanted.
LOCAL_VIDEO_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"
_VIDEO_ID_PATTERNS = {"youtube": r"^[A-Za-z0-9_-]{11}$", "twitch": r"^[0-9]{6,12}$", LOCAL_PLATFORM: LOCAL_VIDEO_ID_PATTERN}
_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$"
_SLUG_PATTERN = r"^[a-z][a-z0-9-]{0,31}$"
_GROUP_PATTERN = r"^[a-z0-9][a-z0-9-]{1,39}$"
_CREDIT_PATTERN = r"^[^\s\u0000-\u001f\u007f]([^\u0000-\u001f\u007f]*[^\s\u0000-\u001f\u007f])?$"
_DATE_PATTERN = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"

_ATTRIBUTES_SCHEMA: dict[str, Any] = {
    "description": "Further labeled facts, as short machine-readable values. Never free text.",
    "type": "object",
    "minProperties": 1,
    "maxProperties": 16,
    "propertyNames": {"pattern": r"^[a-z][a-z0-9_]{0,31}$"},
    "additionalProperties": {
        "anyOf": [
            {"type": "string", "pattern": r"^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$"},
            {"type": "number"},
            {"type": "boolean"},
        ]
    },
}

# A row's attributes also have one key with a fixed form: the fingerprint of the labeled frame.
# The generic value rule would let 16 characters through, but not insist on lower-case hex.
_ROW_ATTRIBUTES_SCHEMA: dict[str, Any] = {
    **_ATTRIBUTES_SCHEMA,
    "properties": {
        "frame_dhash": {
            "description": "The fingerprint of the frame the labels were made on: 16 lower-case "
                           "hexadecimal digits, defined in data/SCHEMA.md.",
            "type": "string",
            "pattern": DHASH_PATTERN,
        }
    },
}

_IDENTITY_PROPERTIES: dict[str, Any] = {
    "card_id": {
        "description": "The catalog id of the card, or null when it was not named. An id has no "
                       "whitespace and no control character in it.",
        "type": ["string", "null"],
        "minLength": 1,
        "maxLength": 64,
        "pattern": CARD_ID_PATTERN,
    },
    "identity": {
        "description": "named: card_id says which card. unknown: a person looked and could not "
                       "name it. unlabeled: nobody was asked to name it.",
        "enum": list(IDENTITIES),
    },
    "readable": {
        "description": "Whether the card can be named from this frame alone (yes, maybe, no), "
                       "or null when that was not recorded.",
        "enum": [*READABLE, None],
    },
}

# A face carries the three identity fields; anything else carries none of them. card_id is a
# string exactly when identity is "named".
_IDENTITY_RULES: list[dict[str, Any]] = [
    {
        "$comment": "a face carries card_id, identity and readable; any other kind carries none of them",
        "if": {"properties": {"kind": {"const": "face"}}, "required": ["kind"]},
        "then": {"required": ["card_id", "identity", "readable"]},
        "else": {"properties": {"card_id": False, "identity": False, "readable": False}},
    },
    {
        "$comment": 'card_id is a catalog id exactly when identity is "named"',
        "if": {"properties": {"identity": {"const": "named"}}, "required": ["identity"]},
        "then": {"properties": {"card_id": {"type": "string"}}},
        "else": {"properties": {"card_id": {"type": "null"}}},
    },
]

_KIND_SCHEMA: dict[str, Any] = {
    "description": "What is there: a card face, a card back, something that cannot be told "
                   "apart (unreadable), or not a card at all (not-card).",
    "enum": list(KINDS),
}

_ORIGIN_SCHEMA: dict[str, Any] = {
    "description": "How the label came about, as a short tag (for example manual, prefill, seeded).",
    "type": "string",
    "pattern": _SLUG_PATTERN,
}

_LABEL_ID_SCHEMA: dict[str, Any] = {
    "description": "A name for this label, unique among its kind within the row.",
    "type": "string",
    "pattern": _ID_PATTERN,
}

BOX_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "ignore", "quad", "oriented", "shape"],
    "properties": {
        "id": _LABEL_ID_SCHEMA,
        "kind": _KIND_SCHEMA,
        "ignore": {
            "description": "True for an ignore region: training neither rewards nor penalizes "
                           "it and scoring does not count it.",
            "type": "boolean",
        },
        "quad": {
            "description": "Four corners as [x, y] fractions of the frame, clockwise on screen.",
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "number", "minimum": QUAD_COORDINATE_RANGE[0], "maximum": QUAD_COORDINATE_RANGE[1]},
            },
        },
        "oriented": {
            "description": "True when the first corner is the card's own top-left, so the first "
                           "edge is the card's top.",
            "type": "boolean",
        },
        "shape": {
            "description": "rotated-box: a rectangle fitted to the card, possibly turned. "
                           "quad: four corners placed one by one.",
            "enum": list(SHAPES),
        },
        **_IDENTITY_PROPERTIES,
        "origin": _ORIGIN_SCHEMA,
        "attributes": _ATTRIBUTES_SCHEMA,
    },
    "allOf": _IDENTITY_RULES,
}

POINT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["x", "y", "kind"],
    "properties": {
        "id": _LABEL_ID_SCHEMA,
        "x": {"description": "Fraction of the frame's width.", "type": "number", "minimum": 0, "maximum": 1},
        "y": {"description": "Fraction of the frame's height.", "type": "number", "minimum": 0, "maximum": 1},
        "kind": _KIND_SCHEMA,
        **_IDENTITY_PROPERTIES,
        "origin": _ORIGIN_SCHEMA,
        "attributes": _ATTRIBUTES_SCHEMA,
    },
    "allOf": _IDENTITY_RULES,
}

ROW_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Label manifest row",
    "description": "One labeled video frame. A manifest is a JSON Lines file of these.",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "manifest_version", "platform", "video_id", "t", "frame_width", "frame_height", "source_type",
        "group", "role", "trained_on", "completeness", "credit", "reachable", "reachable_checked",
    ],
    "properties": {
        "manifest_version": {"description": "Version of this format.", "const": MANIFEST_VERSION},
        "platform": {
            "description": "Where the video is hosted, or local for a recording of your own that is on no platform.",
            "enum": list(ROW_PLATFORMS),
        },
        "video_id": {
            "description": "The platform's id of the video, which identifies the video's channel. "
                           "For a local recording, a name chosen by whoever made it.",
            "type": "string",
            "pattern": r"^[A-Za-z0-9_-]{1,64}$",
        },
        "t": {
            "description": "A time in seconds, from the start of the video, at which the "
                           "labeled frame is on screen.",
            "type": "number",
            "minimum": 0,
            "maximum": 400000,
        },
        "frame_width": {
            "description": "Width in pixels of the frame the labels were made on.",
            "type": "integer",
            "minimum": 16,
            "maximum": 16384,
        },
        "frame_height": {
            "description": "Height in pixels of the frame the labels were made on.",
            "type": "integer",
            "minimum": 16,
            "maximum": 16384,
        },
        "source_type": {
            "description": "The kind of footage: physical or digital, optionally with a detail after a slash.",
            "type": "string",
            "pattern": SOURCE_TYPE_PATTERN,
        },
        "group": {
            "description": "Pseudonymous id of the recording group: videos that share an origin "
                           "(a channel, an event) share a group and are never split across roles.",
            "type": "string",
            "pattern": _GROUP_PATTERN,
        },
        "role": {
            "description": "train: published as training data. evaluation: part of the public "
                           "evaluation set, never trained on. extra: published, in neither set.",
            "enum": list(ROLES),
        },
        "trained_on": {
            "description": "True only when this frame was among the frames the released models were trained on.",
            "type": "boolean",
        },
        "completeness": {
            "description": "exhaustive-boxes: every card in the frame has a box. points: only "
                           "the listed points are labeled.",
            "enum": list(COMPLETENESS),
        },
        "credit": {
            "description": "The credit chosen by whoever contributed the labels.",
            "type": "string",
            "minLength": 1,
            "maxLength": 80,
            "pattern": _CREDIT_PATTERN,
        },
        "reachable": {
            "description": "Whether the video could be found when last checked; null when never checked, "
                           "and always null for a local recording.",
            "type": ["boolean", "null"],
        },
        "reachable_checked": {
            "description": "Date of that check (YYYY-MM-DD), or null when never checked.",
            "type": ["string", "null"],
            "pattern": _DATE_PATTERN,
        },
        "prefill_offered": {
            "description": "Whether a model's proposed boxes were shown to the labeler on this frame.",
            "type": "boolean",
        },
        "attributes": _ROW_ATTRIBUTES_SCHEMA,
        "boxes": {
            "description": "The frame's boxes. Under exhaustive-boxes an empty list means a frame with no card.",
            "type": "array",
            "maxItems": 2000,
            "items": {"$ref": "#/$defs/box"},
        },
        "points": {
            "description": "Labeled points of the frame.",
            "type": "array",
            "maxItems": 2000,
            "items": {"$ref": "#/$defs/point"},
        },
    },
    "allOf": [
        {
            "$comment": "an exhaustive-boxes row has boxes; a points row has at least one point and no box",
            "if": {"properties": {"completeness": {"const": "exhaustive-boxes"}}, "required": ["completeness"]},
            "then": {"required": ["boxes"]},
            "else": {
                "required": ["points"],
                "properties": {"points": {"minItems": 1}, "boxes": {"maxItems": 0}},
            },
        },
        {
            "$comment": "a YouTube video id has 11 characters",
            "if": {"properties": {"platform": {"const": "youtube"}}, "required": ["platform"]},
            "then": {"properties": {"video_id": {"pattern": _VIDEO_ID_PATTERNS["youtube"]}}},
        },
        {
            "$comment": "a Twitch video id is a number",
            "if": {"properties": {"platform": {"const": "twitch"}}, "required": ["platform"]},
            "then": {"properties": {"video_id": {"pattern": _VIDEO_ID_PATTERNS["twitch"]}}},
        },
        {
            "$comment": "a local recording's video id starts with a letter or a digit, and nobody "
                        "else can check that it is reachable",
            "if": {"properties": {"platform": {"const": LOCAL_PLATFORM}}, "required": ["platform"]},
            "then": {"properties": {"video_id": {"pattern": _VIDEO_ID_PATTERNS[LOCAL_PLATFORM]},
                                    "reachable": {"type": "null"}}},
        },
        {
            "$comment": "reachable and reachable_checked are both null or both set",
            "if": {"properties": {"reachable": {"type": "null"}}, "required": ["reachable"]},
            "then": {"properties": {"reachable_checked": {"type": "null"}}},
            "else": {"properties": {"reachable_checked": {"type": "string"}}},
        },
        {
            "$comment": "only a train row can have trained the released models",
            "if": {"properties": {"trained_on": {"const": True}}, "required": ["trained_on"]},
            "then": {"properties": {"role": {"const": "train"}}},
        },
    ],
    "$defs": {"box": BOX_SCHEMA, "point": POINT_SCHEMA},
}


def schema_json() -> str:
    """``ROW_SCHEMA`` as the text of ``data/manifest.schema.json``."""
    return json.dumps(ROW_SCHEMA, indent=2, ensure_ascii=False) + "\n"


# ----------------------------------------------------------------------- structural checking
# A checker for the part of JSON Schema that ROW_SCHEMA uses, so that reading a manifest needs
# no library beyond the standard one. A test compares it with a full implementation.

_TYPE_NAMES = ("null", "boolean", "integer", "number", "string", "array", "object")


def fits_pattern(pattern: str, text: str) -> bool:
    """Whether ``text`` fits a JSON Schema ``pattern``.

    A JSON Schema pattern may match anywhere in the text, and by the rules of JSON Schema its
    ``$`` is the end of the text. Python's ``$`` also matches before a line break that ends
    the text, which would let ``"EXAMPLE0001\n"`` pass as a video id. So a ``$`` that ends the
    pattern is read as the end of the text and nothing else.
    """
    if pattern.endswith("$") and not pattern.endswith("\\$"):
        pattern = pattern[:-1] + "\\Z"
    return re.search(pattern, text) is not None


def _has_type(value: Any, name: str) -> bool:
    if name == "null":
        return value is None
    if name == "boolean":
        return isinstance(value, bool)
    if name == "string":
        return isinstance(value, str)
    if name == "array":
        return isinstance(value, list)
    if name == "object":
        return isinstance(value, dict)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, float) and not math.isfinite(value):
        return False
    return name == "number" or isinstance(value, int) or float(value).is_integer()


def _same(a: Any, b: Any) -> bool:
    """Equality as JSON sees it: true is not 1."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    return a == b


def _where(path: str, key: str | int) -> str:
    if isinstance(key, int):
        return f"{path}[{key}]"
    return f"{path}.{key}" if path else key


def _resolve(schema: Any, root: Mapping[str, Any]) -> Any:
    while isinstance(schema, dict) and "$ref" in schema:
        node: Any = root
        for part in schema["$ref"].removeprefix("#/").split("/"):
            node = node[part]
        schema = node
    return schema


def schema_errors(value: Any, schema: Any, root: Mapping[str, Any] | None = None, path: str = "") -> list[str]:
    """What is wrong with ``value`` under ``schema``; an empty list when nothing is.

    Supports the keywords ``ROW_SCHEMA`` uses and no others.
    """
    root = schema if root is None else root
    schema = _resolve(schema, root)
    here = path or "row"
    if schema is True:
        return []
    if schema is False:
        return [f"{here}: is not allowed here"]
    errors: list[str] = []

    if "const" in schema and not _same(value, schema["const"]):
        errors.append(f"{here}: must be {json.dumps(schema['const'])}, got {json.dumps(value)}")
    if "enum" in schema and not any(_same(value, option) for option in schema["enum"]):
        options = ", ".join(json.dumps(option) for option in schema["enum"])
        errors.append(f"{here}: must be one of {options}; got {json.dumps(value)}")
    if "type" in schema:
        names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_has_type(value, name) for name in names):
            errors.append(f"{here}: must be {' or '.join(names)}, got {json.dumps(value)}")
            return errors

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{here}: must have at least {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{here}: must have at most {schema['maxLength']} characters")
        if "pattern" in schema and not fits_pattern(schema["pattern"], value):
            errors.append(f"{here}: {json.dumps(value)} does not have the required form ({schema['pattern']})")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{here}: {value} is below the minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{here}: {value} is above the maximum {schema['maximum']}")
    elif isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{here}: must have at least {schema['minItems']} items, has {len(value)}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{here}: must have at most {schema['maxItems']} items, has {len(value)}")
        if "items" in schema:
            for index, item in enumerate(value):
                errors.extend(schema_errors(item, schema["items"], root, _where(path, index)))
    elif isinstance(value, dict):
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f'{here}: missing "{name}"')
        if "minProperties" in schema and len(value) < schema["minProperties"]:
            errors.append(f"{here}: must have at least {schema['minProperties']} entries")
        if "maxProperties" in schema and len(value) > schema["maxProperties"]:
            errors.append(f"{here}: must have at most {schema['maxProperties']} entries")
        for name, item in value.items():
            if "propertyNames" in schema:
                errors.extend(schema_errors(name, schema["propertyNames"], root, _where(path, name) + " (the key)"))
            if name in properties:
                errors.extend(schema_errors(item, properties[name], root, _where(path, name)))
            elif "additionalProperties" in schema:
                extra = schema["additionalProperties"]
                if extra is False:
                    errors.append(f'{here}: unknown field "{name}"')
                else:
                    errors.extend(schema_errors(item, extra, root, _where(path, name)))

    if "anyOf" in schema and not any(not schema_errors(value, option, root, path) for option in schema["anyOf"]):
        errors.append(f"{here}: {json.dumps(value)} is not an accepted value")
    for rule in schema.get("allOf", []):
        errors.extend(schema_errors(value, rule, root, path))
    if "if" in schema:
        branch = "then" if not schema_errors(value, schema["if"], root, path) else "else"
        if branch in schema:
            note = f" ({schema['$comment']})" if "$comment" in schema else ""
            errors.extend(message + note for message in schema_errors(value, schema[branch], root, path))
    return errors


# ------------------------------------------------------------------------------- the objects
Quad = tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]


def _attributes_of(value: Any) -> dict[str, Any]:
    return dict(value) if value else {}


@dataclass(frozen=True)
class Box:
    """One box of a frame. See ``data/SCHEMA.md`` for the meaning of each field.

    ``card_id``, ``identity`` and ``readable`` are used on faces only; on any other kind they
    are ``None`` and are not written.
    """

    kind: str
    quad: Quad
    ignore: bool = False
    oriented: bool = False
    shape: str = "quad"
    card_id: str | None = None
    identity: str | None = None
    readable: str | None = None
    id: str | None = None
    origin: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_face(self) -> bool:
        return self.kind == "face"

    def pixel_quad(self, width: float, height: float):
        """The four corners as a ``(4, 2)`` array of pixels in a frame of the given size."""
        return geometry.to_pixels(self.quad, width, height)

    def pixel_box(self, width: float, height: float, clip: bool = True) -> geometry.PixelBox:
        """The upright box ``(x, y, width, height)`` around the corners, in pixels of a frame
        of the given size, clipped to the frame unless ``clip`` is false."""
        corners = self.pixel_quad(width, height)
        return geometry.bounding_box(corners, width, height) if clip else geometry.bounding_box(corners)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.id is not None:
            out["id"] = self.id
        out["kind"] = self.kind
        out["ignore"] = self.ignore
        out["quad"] = [[x, y] for x, y in self.quad]
        out["oriented"] = self.oriented
        out["shape"] = self.shape
        if self.kind == "face":
            out["card_id"] = self.card_id
            out["identity"] = self.identity
            out["readable"] = self.readable
        if self.origin is not None:
            out["origin"] = self.origin
        if self.attributes:
            out["attributes"] = dict(self.attributes)
        return out

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Box:
        """Build a box from a JSON object that has already passed ``row_problems``."""
        quad = tuple((float(x), float(y)) for x, y in doc["quad"])
        return cls(kind=doc["kind"], quad=quad, ignore=doc["ignore"], oriented=doc["oriented"],  # type: ignore[arg-type]
                   shape=doc["shape"], card_id=doc.get("card_id"), identity=doc.get("identity"),
                   readable=doc.get("readable"), id=doc.get("id"), origin=doc.get("origin"),
                   attributes=_attributes_of(doc.get("attributes")))


@dataclass(frozen=True)
class Point:
    """One labeled point of a frame: what is under it, and which card when it is a face."""

    x: float
    y: float
    kind: str
    card_id: str | None = None
    identity: str | None = None
    readable: str | None = None
    id: str | None = None
    origin: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_face(self) -> bool:
        return self.kind == "face"

    def pixel(self, width: float, height: float) -> tuple[float, float]:
        """The point in pixels of a frame of the given size."""
        return self.x * float(width), self.y * float(height)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.id is not None:
            out["id"] = self.id
        out["x"] = self.x
        out["y"] = self.y
        out["kind"] = self.kind
        if self.kind == "face":
            out["card_id"] = self.card_id
            out["identity"] = self.identity
            out["readable"] = self.readable
        if self.origin is not None:
            out["origin"] = self.origin
        if self.attributes:
            out["attributes"] = dict(self.attributes)
        return out

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Point:
        """Build a point from a JSON object that has already passed ``row_problems``."""
        return cls(x=float(doc["x"]), y=float(doc["y"]), kind=doc["kind"], card_id=doc.get("card_id"),
                   identity=doc.get("identity"), readable=doc.get("readable"), id=doc.get("id"),
                   origin=doc.get("origin"), attributes=_attributes_of(doc.get("attributes")))


RowKey = tuple[str, str, float]


@dataclass(frozen=True)
class Row:
    """One labeled frame. See ``data/SCHEMA.md`` for the meaning of each field.

    ``boxes`` is ``None`` when the row has no ``boxes`` field (a points row) and a tuple, which
    may be empty, when it has one; likewise ``points``.
    """

    platform: str
    video_id: str
    t: float
    frame_width: int
    frame_height: int
    source_type: str
    group: str
    role: str
    trained_on: bool
    completeness: str
    credit: str
    reachable: bool | None = None
    reachable_checked: str | None = None
    boxes: tuple[Box, ...] | None = None
    points: tuple[Point, ...] | None = None
    prefill_offered: bool | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    manifest_version: int = MANIFEST_VERSION

    @property
    def key(self) -> RowKey:
        """What makes a row unique: platform, video id and time."""
        return self.platform, self.video_id, self.t

    @property
    def frame_dhash(self) -> str | None:
        """The fingerprint of the labeled frame (``attributes["frame_dhash"]``), when the row
        has one."""
        return self.attributes.get("frame_dhash")

    @property
    def source_family(self) -> str:
        """``physical`` or ``digital``: the part of ``source_type`` before any slash."""
        return source_family(self.source_type)

    def pixel_boxes(self, width: float, height: float, clip: bool = True) -> list[geometry.PixelBox]:
        """Every box of the row as an upright ``(x, y, width, height)`` in pixels of a frame of
        the given size, in the order of ``boxes``; empty for a row without boxes. Boxes are
        clipped to the frame unless ``clip`` is false."""
        return [box.pixel_box(width, height, clip) for box in self.boxes or ()]

    def pixel_points(self, width: float, height: float) -> list[tuple[float, float]]:
        """Every point of the row in pixels of a frame of the given size, in order."""
        return [point.pixel(width, height) for point in self.points or ()]

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "manifest_version": self.manifest_version,
            "platform": self.platform,
            "video_id": self.video_id,
            "t": self.t,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "source_type": self.source_type,
            "group": self.group,
            "role": self.role,
            "trained_on": self.trained_on,
            "completeness": self.completeness,
            "credit": self.credit,
            "reachable": self.reachable,
            "reachable_checked": self.reachable_checked,
        }
        if self.prefill_offered is not None:
            out["prefill_offered"] = self.prefill_offered
        if self.attributes:
            out["attributes"] = dict(self.attributes)
        if self.boxes is not None:
            out["boxes"] = [box.to_json() for box in self.boxes]
        if self.points is not None:
            out["points"] = [point.to_json() for point in self.points]
        return out

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Row:
        """Build a row from a JSON object that has already passed ``row_problems``."""
        boxes = doc.get("boxes")
        points = doc.get("points")
        return cls(
            platform=doc["platform"], video_id=doc["video_id"], t=float(doc["t"]),
            frame_width=int(doc["frame_width"]), frame_height=int(doc["frame_height"]),
            source_type=doc["source_type"], group=doc["group"], role=doc["role"],
            trained_on=doc["trained_on"], completeness=doc["completeness"], credit=doc["credit"],
            reachable=doc["reachable"], reachable_checked=doc["reachable_checked"],
            boxes=None if boxes is None else tuple(Box.from_json(box) for box in boxes),
            points=None if points is None else tuple(Point.from_json(point) for point in points),
            prefill_offered=doc.get("prefill_offered"), attributes=_attributes_of(doc.get("attributes")),
            manifest_version=int(doc["manifest_version"]),
        )


def source_family(source_type: str) -> str:
    """``physical`` or ``digital``: the part of a source type before any slash."""
    return source_type.split("/", 1)[0]


# ----------------------------------------------------------------------------- row checking
def _quad_problems(quad: Sequence[Sequence[float]], width: int, height: int, where: str) -> list[str]:
    corners = geometry.to_pixels(quad, width, height)
    if not geometry.is_convex(corners):
        return [f"{where}.quad: the four corners do not form a convex outline"]
    problems = []
    if geometry.signed_area(corners) < 0:
        problems.append(f"{where}.quad: the corners run counter-clockwise; list them clockwise as seen on screen")
    if geometry.short_side(corners) < 1.0:
        problems.append(f"{where}.quad: an edge is shorter than one pixel of the labeled frame")
    _, _, clipped_width, clipped_height = geometry.bounding_box(corners, width, height)
    if clipped_width <= 0 or clipped_height <= 0:
        problems.append(f"{where}.quad: the box lies wholly outside the frame")
    return problems


def _label_id_problems(labels: Sequence[Mapping[str, Any]], where: str) -> list[str]:
    seen: set[str] = set()
    problems = []
    for index, label in enumerate(labels):
        label_id = label.get("id")
        if label_id is None:
            continue
        if label_id in seen:
            problems.append(f'{where}[{index}].id: "{label_id}" is used twice in this row')
        seen.add(label_id)
    return problems


def detail_problems(doc: Mapping[str, Any]) -> list[str]:
    """What is wrong with a row that already fits ``ROW_SCHEMA``, under the rules a schema
    cannot state: a real calendar date, a time given to the millisecond at most, the shape of
    each quad, and no label repeated within the row."""
    problems: list[str] = []
    if round(float(doc["t"]), 3) != float(doc["t"]):
        problems.append(f"t: {doc['t']} has more than three decimals (milliseconds are the finest unit)")
    checked = doc["reachable_checked"]
    if checked is not None:
        try:
            datetime.date.fromisoformat(checked)
        except ValueError:
            problems.append(f'reachable_checked: "{checked}" is not a calendar date')
    width, height = int(doc["frame_width"]), int(doc["frame_height"])
    boxes = doc.get("boxes") or []
    points = doc.get("points") or []
    seen_boxes: set[tuple] = set()
    for index, box in enumerate(boxes):
        where = f"boxes[{index}]"
        problems.extend(_quad_problems(box["quad"], width, height, where))
        fingerprint = (box["kind"], box["ignore"], tuple(tuple(corner) for corner in box["quad"]))
        if fingerprint in seen_boxes:
            problems.append(f"{where}: repeats an earlier box of this row exactly")
        seen_boxes.add(fingerprint)
    seen_points: set[tuple[float, float]] = set()
    for index, point in enumerate(points):
        position = (float(point["x"]), float(point["y"]))
        if position in seen_points:
            problems.append(f"points[{index}]: repeats the position of an earlier point of this row")
        seen_points.add(position)
    problems.extend(_label_id_problems(boxes, "boxes"))
    problems.extend(_label_id_problems(points, "points"))
    return problems


def row_problems(doc: Any) -> list[str]:
    """Everything wrong with one candidate row, taken on its own; empty when it is valid.

    The JSON Schema is checked first and, when the row fits it, the rules of
    ``detail_problems``. Rules that involve other rows, a catalog or the list of source types
    are the validator's (``cardvision.data.validate``).
    """
    return schema_errors(doc, ROW_SCHEMA) or detail_problems(doc)


class ManifestError(ValueError):
    """A manifest, or a row offered as one, does not have the documented form."""


def parse_row(doc: Any) -> Row:
    """A ``Row`` from a JSON object. Raises ``ManifestError`` naming every problem found."""
    problems = row_problems(doc)
    if problems:
        raise ManifestError("; ".join(problems))
    return Row.from_json(doc)
