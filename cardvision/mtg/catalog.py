# SPDX-License-Identifier: Apache-2.0
"""The Magic catalog, built from Scryfall's ``default_cards`` bulk file.

What goes in
------------
Every English printing that exists on paper or in MTG Arena (``games`` names ``paper`` or
``arena``), one catalog entry per printed face that has its own image:

* a card with one image (Scryfall's top-level ``image_uris``: a normal card, and also split,
  flip, adventure and meld cards, whose faces share one picture) is one entry;
* a card whose faces carry the images (``card_faces[].image_uris``: transform, modal
  double-faced and reversible cards) is one entry per face.

What is left out, and why, is listed in the exclusions file (``build_catalog`` returns it):
objects that are not cards of a deck (tokens, emblems, art-series cards, Jumpstart front cards,
planes, schemes, vanguards, substitute cards, Theros hero cards, sticker sheets), oversized
cards, and printings that exist only on MTGO or not in English. A printing whose image Scryfall marks ``missing`` or ``placeholder`` stays in the
catalog with ``"art": null``, so that its id is known and no bank row is built from a stand-in
picture.

Ids
---
``set:collector_number`` for the front face, and ``set:collector_number:b`` for the back face
(``:c`` and so on after it). Over Scryfall's data the pair is unique, short (14 characters at
most) and readable on a label; a collector number may hold ``-``, ``★`` or ``†`` but never a
colon, so the face suffix cannot collide with one.

Identity
--------
The catalog states ``"identity": "printing"``: names do not decide what is the same card, the
``printingOf`` links do. The identity of a printing is its Oracle card (``oracle_id``; a
reversible card's faces carry theirs per face), with one correction. Some printings share their
illustration with a printing of a *different* Oracle card: an Alchemy card rebalanced for Arena
(its name starts with ``A-``, or it is a digital card drawn from a paper card's art), an Un-set
card printed in several variants of one picture, and a handful of data oddities. A recognizer
sees the same picture under two names there, and the accept rule's margin between them is zero
by construction. Such a printing is made an *alias*: a printing of the Oracle card that owns the
illustration (the card of its earliest printing), with its own Oracle card kept in
``extra.aliasOf``, when the two can be on screen together, that is when the alias and a
printing of the owner with that illustration share an environment (paper or Arena). Only the
printings that share the picture move; the card's other printings keep their own identity (a
Snow-Covered Island that reuses one Island's art is an alias, the other Snow-Covered Islands are
not). A printing that shares a picture with a card it never meets (an Arena-only Alchemy card
drawn from a paper-only card's art) keeps its identity and names the owner in
``extra.sharesArtWith``: a bank of one environment never holds both, and a bank of both reports
their confusion as what it is. An ``A-`` card's printings are aliases of the card it
rebalances.

Each identity has one representative: its earliest printing (by release date, then paper before
digital, then by id), preferring a front face with an image. Every other entry of the identity
has ``printingOf`` set to it, and the representative has none, which is the shape the
``"printing"`` rule requires (``cardvision.core.catalog``).

``extra``
---------
Per entry: ``set``, ``collector_number``, ``scryfall_id``, ``face`` (0 for the front),
``oracle_id``, ``illustration_id``, ``frame``, ``frame_effects``, ``border_color``,
``full_art``, ``textless``, ``promo_types``, ``finishes``, ``foilOnly``, ``lang``,
``arena_id``, ``mtgo_id``, ``layout``, ``games``, ``set_type``, ``digital``,
``released_at``, ``image_status``, ``legal`` (the formats of ``FORMATS`` the card is legal
in), ``aliasOf`` (the printing's own Oracle id) on an alias, and ``sharesArtWith`` (the Oracle
id of the card whose picture it shares) on a printing that shares one across environments. ``foilOnly`` is true when ``finishes`` has
no ``nonfoil``: a foil-only or etched-only printing, the one case where the finish is reported.

The image addresses are not in the catalog (they would add about a tenth to its size and go
stale): ``build_catalog`` returns them as a separate map from id to the ``normal`` image's URL,
which the image fetcher reads. Each entry's ``art`` is the file name the fetcher writes:
``<scryfall id>.jpg``, or ``<scryfall id>-b.jpg`` for a back face.

A release's catalog
-------------------
``extra`` is for the tools that build catalogs and banks, and is about nine tenths of the file;
the browser runtime reads none of it, and holds every entry in memory. ``release_catalog`` cuts
each entry to ``RELEASE_KEYS``. Ids, names, ``printingOf`` and ``art`` stay as they are, so the
identities do not change, and a bank built with the full catalog is valid with the cut one.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from cardvision.core.catalog import parse_catalog

GAME = "mtg"
LOCALE = "en"

# Layouts that are not cards of a deck. Each is left out, with its layout as the reason.
NOT_DECK_LAYOUTS = frozenset({
    "token", "double_faced_token", "emblem", "art_series", "front_card", "vanguard", "planar", "scheme",
})
# Type lines of objects that are not cards of a deck: substitute and helper cards, Theros hero
# cards, Unfinity sticker sheets. Scryfall files them with layout "normal".
NOT_DECK_TYPES = frozenset({"Card", "Hero", "Stickers"})
# Where a printing must exist to be kept.
GAMES_KEPT = frozenset({"paper", "arena"})
# Scryfall image states that are not a picture of the card: kept with "art": null.
NO_PICTURE = frozenset({"missing", "placeholder"})
# The formats whose legality ``extra.legal`` records, and that the scopes are built from.
FORMATS = ("standard", "pioneer", "modern", "legacy", "vintage", "pauper", "commander", "historic", "timeless",
           "alchemy", "brawl", "standardbrawl")
# The kind is the card's first type in this order of precedence.
KIND_ORDER = ("Creature", "Planeswalker", "Battle", "Land", "Artifact", "Enchantment", "Instant", "Sorcery",
              "Kindred", "Tribal", "Conspiracy", "Dungeon")
FACE_SUFFIXES = "bcdefgh"

# The keys of an entry that a release's catalog keeps: the ones the runtime reads.
RELEASE_KEYS = ("id", "name", "kind", "printingOf", "art", "token", "facets")


@dataclass
class CatalogBuild:
    """The result of ``build_catalog``.

    Attributes:
        catalog: the catalog document (``cardvision.core.catalog`` format).
        images: entry id to its ``normal`` image's URL, for the entries with art.
        exclusions: one record per Scryfall printing left out, with its reason.
        aliases: Oracle id of an illustration's owner to the Oracle ids of the printings made its
            aliases.
        counts: what was read and kept, for the report.
    """

    catalog: dict[str, Any]
    images: dict[str, str]
    exclusions: list[dict[str, Any]] = field(default_factory=list)
    aliases: dict[str, list[str]] = field(default_factory=dict)
    counts: dict[str, Any] = field(default_factory=dict)


def card_id(set_code: str, collector_number: str, face: int = 0) -> str:
    """The catalog id of a printed face: ``set:cn``, with ``:b``, ``:c``... after the front."""
    base = f"{set_code}:{collector_number}"
    return base if face == 0 else f"{base}:{FACE_SUFFIXES[face - 1]}"


def kind_of(type_line: str | None) -> str | None:
    """The card's kind: its first card type in ``KIND_ORDER`` (front face of a two-part type line)."""
    if not type_line:
        return None
    front = type_line.split("//")[0]
    words = front.replace("—", " ").split()
    for kind in KIND_ORDER:
        if kind in words:
            return "Kindred" if kind == "Tribal" else kind
    return None


def exclusion_reason(card: Mapping[str, Any]) -> str | None:
    """Why a Scryfall printing is left out of the catalog, or ``None`` when it is kept."""
    if card.get("lang") != LOCALE:
        return "not-english"
    if not GAMES_KEPT & set(card.get("games") or ()):
        return "not-paper-or-arena"
    layout = card.get("layout")
    if layout in NOT_DECK_LAYOUTS:
        return f"layout:{layout}"
    if card.get("set_type") == "token":
        return "set_type:token"
    if card.get("type_line") in NOT_DECK_TYPES:
        return f"type:{card['type_line']}"
    if card.get("oversized"):
        return "oversized"
    return None


def _faces(card: Mapping[str, Any]) -> list[tuple[int, Mapping[str, Any]]]:
    """The printed faces with their own image, as (face number, the object that holds it):
    the card itself when it has top-level images, else each face that has images."""
    if card.get("image_uris"):
        return [(0, card)]
    faces = [(i, face) for i, face in enumerate(card.get("card_faces") or ()) if face.get("image_uris")]
    return faces or [(0, card)]  # no image anywhere: one entry, without art


def _oracle_of(card: Mapping[str, Any], face: Mapping[str, Any]) -> str | None:
    return face.get("oracle_id") or card.get("oracle_id") or next(
        (f.get("oracle_id") for f in card.get("card_faces") or () if f.get("oracle_id")), None)


def _illustration_of(card: Mapping[str, Any], face: Mapping[str, Any]) -> str | None:
    return face.get("illustration_id") or (card.get("illustration_id") if face is card else None)


def _order_key(entry: Mapping[str, Any]) -> tuple:
    extra = entry["extra"]
    return (extra["face"] != 0, entry["art"] is None, extra.get("released_at") or "9999", bool(extra.get("digital")),
            entry["id"])


def build_catalog(cards: Iterable[Mapping[str, Any]]) -> CatalogBuild:
    """Build the catalog from Scryfall card objects (a ``default_cards`` bulk file's contents).

    The catalog is checked with the package's own loader before it is returned.
    """
    entries: list[dict[str, Any]] = []
    images: dict[str, str] = {}
    exclusions: list[dict[str, Any]] = []
    counts: Counter = Counter()
    names_by_oracle: dict[str, str] = {}
    for card in cards:
        counts["read"] += 1
        reason = exclusion_reason(card)
        if reason is not None:
            counts[f"excluded:{reason}"] += 1
            if reason not in ("not-english", "not-paper-or-arena"):
                exclusions.append({"id": card_id(card["set"], card["collector_number"]), "name": card.get("name"),
                                   "scryfall_id": card.get("id"), "reason": reason})
            continue
        counts["printings"] += 1
        legal = [fmt for fmt in FORMATS if (card.get("legalities") or {}).get(fmt) in ("legal", "restricted")]
        for face_number, face in _faces(card):
            entry_id = card_id(card["set"], card["collector_number"], face_number)
            uris = face.get("image_uris") or {}
            status = card.get("image_status")
            has_art = bool(uris.get("normal")) and status not in NO_PICTURE
            oracle = _oracle_of(card, face)
            name = face.get("name") if face is not card else card["name"]
            if oracle and face_number == 0 and face is card:
                names_by_oracle.setdefault(oracle, card["name"])
            extra = {
                "set": card["set"],
                "collector_number": card["collector_number"],
                "scryfall_id": card["id"],
                "face": face_number,
                "oracle_id": oracle,
                "illustration_id": _illustration_of(card, face),
                "frame": card.get("frame"),
                "frame_effects": sorted(card.get("frame_effects") or ()),
                "border_color": card.get("border_color"),
                "full_art": bool(card.get("full_art")),
                "textless": bool(card.get("textless")),
                "promo_types": sorted(card.get("promo_types") or ()),
                "finishes": list(card.get("finishes") or ()),
                "foilOnly": bool(card.get("finishes")) and "nonfoil" not in card["finishes"],
                "lang": card.get("lang"),
                "arena_id": card.get("arena_id"),
                "mtgo_id": card.get("mtgo_id"),
                "layout": card.get("layout"),
                "games": sorted(card.get("games") or ()),
                "set_type": card.get("set_type"),
                "digital": bool(card.get("digital")),
                "released_at": card.get("released_at"),
                "image_status": status,
                "legal": legal,
            }
            entry = {
                "id": entry_id,
                "name": name,
                "kind": kind_of(face.get("type_line") or card.get("type_line")),
                "printingOf": None,
                "art": f"{card['id']}{'' if face_number == 0 else '-' + FACE_SUFFIXES[face_number - 1]}.jpg"
                if has_art else None,
                "extra": extra,
            }
            if has_art:
                images[entry_id] = uris["normal"]
            else:
                counts["without-art"] += 1
            entries.append(entry)

    # Identity: the Oracle card, except for an alias, which takes the owner of its picture's.
    def own_key(entry: Mapping[str, Any]) -> str:
        return entry["extra"]["oracle_id"] or f"id:{entry['id']}"

    key_of = {entry["id"]: own_key(entry) for entry in entries}
    by_illustration: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        if entry["extra"]["illustration_id"]:
            by_illustration[entry["extra"]["illustration_id"]].append(entry)
    for shared in by_illustration.values():
        if len({own_key(entry) for entry in shared}) < 2:
            continue
        owner = own_key(min(shared, key=_order_key))
        owner_games = set().union(*(set(e["extra"]["games"]) for e in shared if own_key(e) == owner)) & GAMES_KEPT
        for entry in shared:
            if own_key(entry) == owner:
                continue
            if set(entry["extra"]["games"]) & owner_games:
                key_of[entry["id"]] = owner
            else:  # never on screen beside the owner's printing: it keeps its own identity
                entry["extra"]["sharesArtWith"] = owner
    oracle_by_name = {name: oracle for oracle, name in names_by_oracle.items()}
    for entry in entries:
        base = oracle_by_name.get(entry["name"][2:]) if entry["name"].startswith("A-") else None
        if base is not None and base != own_key(entry):
            key_of[entry["id"]] = base

    members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        members[key_of[entry["id"]]].append(entry)
    aliases: dict[str, set[str]] = defaultdict(set)
    for key, group in members.items():
        group.sort(key=_order_key)
        representative = group[0]
        for entry in group:
            if own_key(entry) != key:
                entry["extra"]["aliasOf"] = own_key(entry)
                aliases[key].add(own_key(entry))
            if entry is not representative:
                entry["printingOf"] = representative["id"]

    entries.sort(key=lambda entry: entry["id"])
    document = {"game": GAME, "locale": LOCALE, "identity": "printing", "cards": entries}
    parse_catalog(document, source="mtg catalog")  # the package's own check, before anyone reads it
    counts["entries"] = len(entries)
    counts["identities"] = len(members)
    counts["alias-printings"] = sum(1 for entry in entries if "aliasOf" in entry["extra"])
    counts["back-faces"] = sum(1 for entry in entries if entry["extra"]["face"] > 0)
    return CatalogBuild(catalog=document, images=images, exclusions=exclusions,
                        aliases={key: sorted(values) for key, values in sorted(aliases.items())},
                        counts=dict(sorted(counts.items())))


def release_catalog(document: Mapping[str, Any]) -> dict[str, Any]:
    """The catalog a release ships: ``document`` with each entry cut to ``RELEASE_KEYS``.

    The result is checked with the package's own loader before it is returned.
    """
    cards = [{key: entry[key] for key in RELEASE_KEYS if key in entry} for entry in document["cards"]]
    result = {**document, "cards": cards}
    parse_catalog(result, source="mtg release catalog")
    return result
