# SPDX-License-Identifier: Apache-2.0
"""A toy game in which different cards share a name, for the tests of the identity rule.

It is the toy game of ``toy_game`` with three changes:

* three different cards are all named ``Amber Circle``: ``TOY-001``, ``TOY-007`` and
  ``TOY-013``. They keep their own pictures, so they are told apart by eye and by id only;
* ``TOY-001`` and ``TOY-007`` each get an alternate printing, ``TOY-001A`` and ``TOY-007A``,
  drawn like the toy game's other alternates. With ``TOY-003A`` and ``TOY-P01`` the game has
  four alternate printings;
* ``TOY-016`` is named ``Plum Triangle``, the name of ``TOY-015``, and its ``art`` is the image
  of ``TOY-015``: two different cards that share a name and a picture. A bank cannot tell them
  apart. (The toy bank of ``test_infer_toy_models.ToySetup`` gives them the same row.)

The same cards are written under either identity rule (``namesake_document``). Under ``name``
every ``Amber Circle`` is one identity, five entries of three different cards. Under
``printing`` each card is its own identity and its alternate printing belongs to it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import toy_game

SHARED_NAME = "Amber Circle"
NAMESAKES = ("TOY-001", "TOY-007", "TOY-013")          # three different cards with one name
NEW_ALTERNATES = {"TOY-001A": "TOY-001", "TOY-007A": "TOY-007"}
LOOKALIKE, LOOKALIKE_OF = "TOY-016", "TOY-015"         # share a name, and a bank row's image in ToySetup


def namesake_document(identity: str | None) -> dict:
    """The catalog of the game above. ``identity`` is the rule it states; ``None`` leaves the
    field out, which means ``name``."""
    document = toy_game.catalog_document()
    by_id = {card["id"]: card for card in document["cards"]}
    for card_id in NAMESAKES:
        by_id[card_id]["name"] = SHARED_NAME
    by_id[LOOKALIKE]["name"] = by_id[LOOKALIKE_OF]["name"]
    by_id[LOOKALIKE]["art"] = by_id[LOOKALIKE_OF]["art"]
    for card_id, base in NEW_ALTERNATES.items():
        document["cards"].append({"id": card_id, "name": by_id[base]["name"], "kind": by_id[base]["kind"],
                                  "printingOf": base, "art": f"{card_id}.png"})
    if identity is None:
        return document
    return {"game": document["game"], "locale": document["locale"], "identity": identity,
            **{key: value for key, value in document.items() if key not in ("game", "locale")}}


def write_namesake_game(out_dir: str | Path, identity: str | None, toy_art: str | Path) -> tuple[Path, Path]:
    """Write the catalog and the card images into ``out_dir``. ``toy_art`` is the toy game's
    image folder; the two new alternates are drawn beside copies of its images. Returns
    ``(catalog path, art folder)``."""
    out_dir = Path(out_dir)
    art_dir = out_dir / "art"
    shutil.copytree(toy_art, art_dir)
    drawn = {card_id: (name, kind) for card_id, name, kind, _ in toy_game.CARDS}
    for card_id, base in NEW_ALTERNATES.items():
        name, kind = drawn[base]
        toy_game.draw_card(card_id, name, kind, alternate=True).save(art_dir / f"{card_id}.png")
    catalog_path = out_dir / "catalog.json"
    catalog_path.write_text(json.dumps(namesake_document(identity), indent=2) + "\n", encoding="utf-8")
    return catalog_path, art_dir
