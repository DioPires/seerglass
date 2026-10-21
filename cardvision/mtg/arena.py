# SPDX-License-Identifier: Apache-2.0
"""The Arena catalog: what a bank for MTG Arena footage holds.

MTG Arena draws a card in two ways. In the hand, on the stack and in the inspect view it draws
the whole card (in the hand, only its top part shows: the rest is below the screen's edge). On
the battlefield it draws a permanent as a *tile*: a nearly square box of the card's art, with
its name and, for a creature, its power and toughness, and no rules text. A tapped permanent
tilts by a few degrees, fades and carries a large tap mark; it never turns a quarter turn, which
is why the accept rule for Arena has the rotation retry off (``games/mtg/accept.json``).

A bank row made from the whole card does not match a tile: on synthetic tiles the whole-card
rows named 2.7 % right, and 40 % of the names they gave were wrong (the Phase 1 review). So the
Arena catalog holds, besides each face of the Arena pool, a *tile entry* for each permanent:

* **Faces.** Every entry of the full catalog (``cardvision.mtg.catalog``) whose printing is on
  Arena (``extra.games`` names ``arena``).
* **Tile entries**, ``<face id>:art``, one per face of a permanent kind (``PERMANENT_KINDS``)
  that has art. Its picture is a square cut from the face's art box (``tile_image``), written
  under ``tiles/`` in the image folder by ``render_tiles``. Its ``printingOf`` is the identity's
  representative, so a tile and its card are one identity and never narrow each other's margin,
  and ``extra.artOf`` names the face. Instants and sorceries never lie on the battlefield and
  get none: a tile row for them could only ever win wrongly.
* **Tokens** (``token_entries``), from the token sets of the sets the pool's printings come
  from. Scryfall marks almost none of them as being on Arena, so ``games`` is not read for them.
  Each is marked ``"token": true``. Tokens are grouped by name: a token named like a card of the
  pool (a token copy, an eternalized creature) joins that card's identity; any other token is
  named ``"<name> token"``, and all tokens of one name are one identity. Permanent tokens get
  tile entries too. Emblems are not cards on Arena (they are drawn as boxes) and stay out.

The full catalog's representatives are mostly paper printings, so each identity's
representative is chosen again among the entries kept, by the full catalog's own order (its
earliest printing, front face with art first). ``extra`` is cut to the fields the Arena tools
read (``SLIM_EXTRA``).

The tile's shape is provisional (``TILE_SHAPE``): a square cut from the middle of the art box.
Arena's own tile crop, its name strip and its power and toughness box are to be measured on
real footage, and the tiles drawn again in that shape.
"""

from __future__ import annotations

import io
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cardvision.core.catalog import parse_catalog
from cardvision.mtg.catalog import _faces, _order_key, card_id, kind_of

SOURCE_TYPE = "digital/mtg-arena"
PERMANENT_KINDS = frozenset({"Creature", "Planeswalker", "Battle", "Land", "Artifact", "Enchantment"})
TOKEN_LAYOUTS = frozenset({"token", "double_faced_token"})
TILE_SUFFIX = ":art"
TILE_FOLDER = "tiles"
TILE_SHAPE = "square-art-v0"
# The fields of ``extra`` an Arena catalog keeps.
SLIM_EXTRA = ("set", "collector_number", "face", "oracle_id", "illustration_id", "arena_id", "frame",
              "frame_effects", "border_color", "full_art", "layout", "released_at", "digital", "legal")
# The art box of a ``normal`` image (488 x 680 px), per frame: left, top, right, bottom, as
# fractions of the image. Measured by eye on one card of each frame.
ART_BOXES = {
    "1993": (0.11, 0.095, 0.89, 0.545),
    "1997": (0.11, 0.095, 0.89, 0.545),
    "2003": (0.08, 0.11, 0.92, 0.555),
    "2015": (0.075, 0.11, 0.925, 0.555),
    "future": (0.08, 0.11, 0.92, 0.555),
}
# Full-art and borderless faces: the art runs to the edges, behind the text.
FULL_ART_BOX = (0.05, 0.11, 0.95, 0.60)
TILE_QUALITY = 92


@dataclass
class ArenaCatalog:
    """The result of ``arena_catalog``: the catalog document (``cardvision.core.catalog``
    format), and what was kept and added, for the report."""

    catalog: dict[str, Any]
    counts: dict[str, Any] = field(default_factory=dict)


def tile_id(face_id: str) -> str:
    """The id of a face's tile entry."""
    return face_id + TILE_SUFFIX


def is_tile(entry: Mapping[str, Any]) -> bool:
    return "artOf" in (entry.get("extra") or {})


def art_box(extra: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """The art box of a face's ``normal`` image, as fractions of its width and height."""
    if extra.get("full_art") or extra.get("border_color") == "borderless":
        return FULL_ART_BOX
    return ART_BOXES.get(extra.get("frame") or "2015", ART_BOXES["2015"])


def tile_image(image, extra: Mapping[str, Any]):
    """A tile's picture: the largest square in the middle of the face's art box.

    ``image`` is a Pillow image of the face's ``normal`` scan; the result is a new image.
    """
    left, top, right, bottom = art_box(extra)
    width, height = image.size
    box = (left * width, top * height, right * width, bottom * height)
    side = min(box[2] - box[0], box[3] - box[1])
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    square = (round(cx - side / 2), round(cy - side / 2), round(cx + side / 2), round(cy + side / 2))
    return image.convert("RGB").crop(square)


def _slim(extra: Mapping[str, Any]) -> dict[str, Any]:
    return {key: extra[key] for key in SLIM_EXTRA if key in extra}


def parent_set(card: Mapping[str, Any]) -> str:
    """The set a token belongs with. Card objects do not carry the token set's parent (that is a
    field of Scryfall's set objects), but token sets are named after their parent with a ``t``
    in front (``tdsk`` for ``dsk``); a token printed in its parent set keeps that set."""
    code = card.get("set") or ""
    return code[1:] if card.get("set_type") == "token" and code.startswith("t") else code


def token_entries(cards: Iterable[Mapping[str, Any]], sets: Iterable[str]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """The token faces whose set (``parent_set``) is one of ``sets``, as catalog entries
    (``printingOf`` not yet set), and their image URLs. Only permanents: Scryfall files helper
    cards (The Monarch, Day and Night, checklists) as tokens too."""
    parents = set(sets)
    entries: list[dict[str, Any]] = []
    images: dict[str, str] = {}
    for card in cards:
        if card.get("layout") not in TOKEN_LAYOUTS or card.get("lang") != "en":
            continue
        if parent_set(card) not in parents:
            continue
        for face_number, face in _faces(card):
            kind = kind_of(face.get("type_line") or card.get("type_line"))
            if kind not in PERMANENT_KINDS:  # helper cards: The Monarch, Day and Night, Copy, checklists
                continue
            entry_id = card_id(card["set"], card["collector_number"], face_number)
            uris = face.get("image_uris") or {}
            has_art = bool(uris.get("normal")) and card.get("image_status") not in ("missing", "placeholder")
            extra = _slim({
                "set": card["set"], "collector_number": card["collector_number"], "face": face_number,
                "oracle_id": face.get("oracle_id") or card.get("oracle_id"),
                "illustration_id": face.get("illustration_id") or (card.get("illustration_id") if face is card else None),
                "arena_id": card.get("arena_id"), "frame": card.get("frame"),
                "frame_effects": sorted(card.get("frame_effects") or ()), "border_color": card.get("border_color"),
                "full_art": bool(card.get("full_art")), "layout": card.get("layout"),
                "released_at": card.get("released_at"), "digital": bool(card.get("digital")), "legal": [],
            })
            entries.append({
                "id": entry_id,
                "name": face.get("name") if face is not card else card["name"],
                "kind": kind,
                "printingOf": None,
                "art": f"{card['id']}{'' if face_number == 0 else '-b'}.jpg" if has_art else None,
                "token": True,
                "extra": extra,
            })
            if has_art:
                images[entry_id] = uris["normal"]
    return entries, images


def arena_catalog(full: Mapping[str, Any], tokens: Iterable[Mapping[str, Any]] = (), *,
                  tiles: bool = True) -> ArenaCatalog:
    """The Arena catalog, from the full catalog document and the token entries of
    ``token_entries``.

    The result is checked with the package's own loader before it is returned.
    """
    counts: Counter = Counter()
    faces = [entry for entry in full["cards"] if "arena" in entry["extra"].get("games", ())]
    counts["faces"] = len(faces)

    # key_of: an entry's identity. A face keeps the full catalog's (its representative's id).
    entries: list[dict[str, Any]] = []
    key_of: dict[str, str] = {}
    for entry in faces:
        entries.append({**entry, "printingOf": None, "extra": _slim(entry["extra"])})
        key_of[entry["id"]] = entry.get("printingOf") or entry["id"]

    # Tokens: a card's name joins that card's identity, any other name is an identity of its own.
    identity_of_name: dict[str, str] = {}
    for entry in sorted(entries, key=_order_key):
        identity_of_name.setdefault(entry["name"], key_of[entry["id"]])
    for token in tokens:
        if token["id"] in key_of:
            counts["tokens-skipped-duplicate-id"] += 1
            continue
        token = {**token, "printingOf": None, "token": True}
        if token["name"] in identity_of_name:
            key_of[token["id"]] = identity_of_name[token["name"]]
            counts["tokens-joining-a-card"] += 1
        else:
            token["name"] = f"{token['name']} token"
            key_of[token["id"]] = "token:" + token["name"]
        entries.append(token)
        counts["token-faces"] += 1
    counts["token-identities"] = len({key for key in key_of.values() if key.startswith("token:")})

    # Each identity's representative, chosen again among what is kept.
    members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        members[key_of[entry["id"]]].append(entry)
    representative_of: dict[str, str] = {}
    for key, group in members.items():
        group.sort(key=_order_key)
        representative_of[key] = group[0]["id"]
        for entry in group[1:]:
            entry["printingOf"] = group[0]["id"]

    if tiles:
        added = []
        for entry in entries:
            if entry["kind"] not in PERMANENT_KINDS or entry["art"] is None:
                continue
            added.append({
                "id": tile_id(entry["id"]),
                "name": entry["name"],
                "kind": entry["kind"],
                "printingOf": representative_of[key_of[entry["id"]]],
                "art": f"{TILE_FOLDER}/{entry['art']}",
                **({"token": True} if entry.get("token") else {}),
                "extra": {**entry["extra"], "artOf": entry["id"], "tile": TILE_SHAPE},
            })
        entries.extend(added)
        counts["tile-entries"] = len(added)

    entries.sort(key=lambda entry: entry["id"])
    document = {"game": full.get("game", "mtg"), "locale": full.get("locale", "en"), "identity": "printing",
                "cards": entries}
    parse_catalog(document, source="mtg arena catalog")
    counts["entries"] = len(entries)
    counts["identities"] = len(members)
    return ArenaCatalog(catalog=document, counts=dict(sorted(counts.items())))


@dataclass
class TileReport:
    written: int = 0
    present: int = 0
    missing_source: list[str] = field(default_factory=list)


def render_tiles(catalog: Mapping[str, Any], art_dir: str | Path, *, overwrite: bool = False) -> TileReport:
    """Write the picture of every tile entry of ``catalog`` (a document), cut from its face's
    image in ``art_dir``, to the tile entry's ``art`` path in the same folder."""
    from PIL import Image

    art_dir = Path(art_dir)
    by_id = {entry["id"]: entry for entry in catalog["cards"]}
    report = TileReport()
    for entry in catalog["cards"]:
        if not is_tile(entry):
            continue
        out = art_dir / entry["art"]
        if out.is_file() and not overwrite:
            report.present += 1
            continue
        face = by_id[entry["extra"]["artOf"]]
        source = art_dir / face["art"]
        if not source.is_file():
            report.missing_source.append(entry["id"])
            continue
        with Image.open(source) as image:
            tile = tile_image(image, face["extra"])
        out.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        tile.save(buffer, format="JPEG", quality=TILE_QUALITY)
        partial = out.with_name(out.name + ".partial")
        partial.write_bytes(buffer.getvalue())
        partial.replace(out)
        report.written += 1
    return report
