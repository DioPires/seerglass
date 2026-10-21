# SPDX-License-Identifier: Apache-2.0
"""Look groups: the printings a picture cannot tell apart.

Two printings *look alike* when they share the illustration and every large feature of the
frame that a viewer sees at stream size. Their difference, if any, is then the set symbol, the
collector line or a promo stamp: a few pixels in a video frame. The look key is

    (illustration_id, frame, border_color, full_art, textless, frame effects that change the picture)

and a *look group* is the set of printings of one scope that share a key. A printing whose look
group holds only itself has a *visual identity*: a model that sees its picture well enough can
name that exact printing. A printing in a group of two or more has a *visual twin*.

Three keys are measured, so that the effect of the choice is visible:

``all`` (the default)
    every frame effect Scryfall records. Where printings of one illustration and frame differ
    in a frame effect, the difference is drawn on the card (an extended-art or showcase frame,
    an inverted or etched frame, a legendary crown or a Nyx background added in a later
    reprint), so this is the key that splits printings as a viewer would;
``visible``
    ``all`` without the small marks that cannot be seen at stream size: the double-faced card's
    corner symbol (``*dfc``) and the ``legendary``, ``tombstone``, ``lesson``, ``draft`` and
    ``spree`` marks. It counts more twins;
``stamps``
    ``all`` with the promo stamps added (``STAMP_PROMOS``: a planeswalker or date stamp, a
    prerelease or promo-pack stamp, a serial number). They are a few pixels on a stream, so this
    key counts printings that differ only in a stamp as different: the fewest twins, and the
    figure for a close-up camera rather than a stream.

A printing with no ``illustration_id`` (Scryfall has none for some old and special cards) is a
look of its own, and is counted apart as ``unknownIllustration``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from statistics import median
from typing import Any

from cardvision.core.catalog import Card, Catalog

SMALL_EFFECTS = frozenset({"legendary", "tombstone", "lesson", "draft", "spree"})
STAMP_PROMOS = frozenset({"stamped", "datestamped", "prerelease", "promopack", "serialized"})
VARIANTS = ("all", "visible", "stamps")


def _effects(effects: Iterable[str], variant: str) -> tuple[str, ...]:
    if variant in ("all", "stamps"):
        return tuple(sorted(effects))
    if variant != "visible":
        raise ValueError(f"unknown look variant {variant!r} (known: {list(VARIANTS)})")
    return tuple(sorted(e for e in effects if e not in SMALL_EFFECTS and not e.endswith("dfc")))


def look_key(card: Card, variant: str = "all") -> tuple:
    """The look key of a catalog entry built by ``cardvision.mtg.catalog``."""
    extra = card.extra
    illustration = extra.get("illustration_id")
    if not illustration:
        return ("unknown", card.id)
    key = (illustration, extra.get("frame"), extra.get("border_color"), bool(extra.get("full_art")),
           bool(extra.get("textless")), _effects(extra.get("frame_effects") or (), variant))
    if variant == "stamps":
        key += (tuple(sorted(STAMP_PROMOS & set(extra.get("promo_types") or ()))),)
    return key


# A scope is a predicate on a catalog entry: which printings a bank of that scope holds.
def _legal(fmt: str) -> Callable[[Card], bool]:
    return lambda card: fmt in card.extra.get("legal", ())


SCOPES: dict[str, tuple[str, Callable[[Card], bool]]] = {
    "standard-arena": ("Arena printings of Standard-legal cards",
                       lambda card: "standard" in card.extra.get("legal", ()) and "arena" in card.extra.get("games", ())),
    "standard": ("every printing of a Standard-legal card", _legal("standard")),
    "arena": ("every Arena printing (the Arena pool)", lambda card: "arena" in card.extra.get("games", ())),
    "pioneer": ("every printing of a Pioneer-legal card", _legal("pioneer")),
    "modern": ("every printing of a Modern-legal card", _legal("modern")),
    "commander": ("every printing of a Commander-legal card", _legal("commander")),
    "all": ("every printing in the catalog", lambda card: True),
}


def scope_ids(catalog: Catalog, scope: str, with_art: bool = True) -> list[str]:
    """The ids of a scope's entries, in catalog order (only those with art, by default)."""
    keep = SCOPES[scope][1]
    return [card.id for card in catalog if keep(card) and (card.has_art or not with_art)]


def _histogram(sizes: Iterable[int]) -> dict[str, int]:
    """Printings by the size of their look group: 1, 2, 3-5, 6-20, 21+."""
    bins = Counter()
    for size in sizes:
        label = "1" if size == 1 else "2" if size == 2 else "3-5" if size <= 5 else "6-20" if size <= 20 else "21+"
        bins[label] += size
    return {label: bins.get(label, 0) for label in ("1", "2", "3-5", "6-20", "21+")}


def look_stats(catalog: Catalog, ids: Iterable[str], variant: str = "all") -> dict[str, Any]:
    """Look-group statistics of a set of catalog entries (one scope)."""
    cards = [catalog.card(card_id) for card_id in ids]
    groups: dict[tuple, list[Card]] = defaultdict(list)
    for card in cards:
        groups[look_key(card, variant)].append(card)
    by_identity: dict[str, list[Card]] = defaultdict(list)
    for card in cards:
        by_identity[catalog.identity_of(card.id)].append(card)
    size_of = {key: len(group) for key, group in groups.items()}
    sizes = sorted(size_of.values())
    twins = sum(size for size in sizes if size >= 2)
    multi = {ident: members for ident, members in by_identity.items() if len(members) >= 2}
    # A card whose printings all have a look of their own, and one whose printings all share one.
    all_distinct = sum(1 for members in multi.values() if all(size_of[look_key(c, variant)] == 1 for c in members))
    one_look = sum(1 for members in multi.values() if len({look_key(c, variant) for c in members}) == 1)
    across = sum(1 for group in groups.values() if len({catalog.identity_of(c.id) for c in group}) > 1)
    illustrations = {card.extra.get("illustration_id") for card in cards if card.extra.get("illustration_id")}
    largest = max(groups.values(), key=len) if groups else []
    return {
        "printings": len(cards),
        "identities": len(by_identity),
        "illustrations": len(illustrations),
        "lookGroups": len(groups),
        "unknownIllustration": sum(1 for card in cards if not card.extra.get("illustration_id")),
        "printingsPerLookGroup": {"mean": round(len(cards) / len(groups), 3) if groups else None,
                                  "median": median(sizes) if sizes else None, "max": sizes[-1] if sizes else None},
        "printingsByLookGroupSize": _histogram(sizes),
        "printingsWithVisualTwin": twins,
        "shareWithVisualTwin": round(twins / len(cards), 4) if cards else None,
        "identitiesWithSeveralPrintings": len(multi),
        "identitiesAllPrintingsDistinct": all_distinct,
        "identitiesOneLookForAll": one_look,
        "lookGroupsAcrossIdentities": across,
        "largestLookGroup": {"size": len(largest), "name": largest[0].name if largest else None,
                             "example": [card.id for card in largest[:5]]},
    }


def look_report(catalog: Catalog, scopes: Iterable[str] = tuple(SCOPES),
                variants: Iterable[str] = VARIANTS) -> dict[str, Any]:
    """``look_stats`` for each scope and variant, over the entries with art."""
    report: dict[str, Any] = {"definition": __doc__.split("\n\n")[1].strip(), "scopes": {}}
    for scope in scopes:
        ids = scope_ids(catalog, scope)
        report["scopes"][scope] = {"description": SCOPES[scope][0],
                                   **{variant: look_stats(catalog, ids, variant) for variant in variants}}
    return report


def markdown_table(report: Mapping[str, Any], variant: str = "all") -> str:
    """The report's main figures as a Markdown table, one row per scope."""
    lines = [("| scope | printings | cards | look groups | printings with a visual twin | cards with one look for all "
              "printings | cards whose printings all look different |"),
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for scope, values in report["scopes"].items():
        v = values[variant]
        multi = v["identitiesWithSeveralPrintings"] or 1
        lines.append(f"| {scope} | {v['printings']:,} | {v['identities']:,} | {v['lookGroups']:,} | "
                     f"{v['printingsWithVisualTwin']:,} ({100 * v['shareWithVisualTwin']:.1f} %) | "
                     f"{v['identitiesOneLookForAll']:,} of {v['identitiesWithSeveralPrintings']:,} "
                     f"({100 * v['identitiesOneLookForAll'] / multi:.1f} %) | "
                     f"{v['identitiesAllPrintingsDistinct']:,} ({100 * v['identitiesAllPrintingsDistinct'] / multi:.1f} %) |")
    return "\n".join(lines)
