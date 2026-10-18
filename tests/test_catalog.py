# SPDX-License-Identifier: Apache-2.0
"""Loading and validating the catalog file, and the helpers built on it."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import toy_game
from cardvision.core import catalog as C


def minimal() -> dict:
    return {
        "game": "toy",
        "locale": "en",
        "cards": [
            {"id": "AB-001", "name": "Kite", "kind": "unit", "printingOf": None, "art": "AB-001.png"},
            {"id": "AB-001A", "name": "Kite", "kind": "unit", "printingOf": "AB-001", "art": "alt/AB-001A.png"},
            {"id": "AB-002", "name": "Lamp", "kind": None, "printingOf": None, "art": "https://example.org/a b.png"},
        ],
    }


# --------------------------------------------------------------------------- loading

def test_the_toy_catalog_loads(toy):
    catalog = C.load_catalog(toy[0])
    assert catalog.game == "toy" and catalog.locale == "en"
    assert len(catalog) == 20 and len(catalog.cards) == 20
    assert catalog.ids() == [card_id for card_id, *_ in toy_game.CARDS]
    assert [card.id for card in catalog] == catalog.ids()
    assert catalog.kinds == toy_game.KINDS
    first = catalog.card("TOY-001")
    assert first == C.Card(id="TOY-001", name="Amber Circle", kind="unit", printing_of=None, art="TOY-001.png")
    assert catalog.card("TOY-003A").printing_of == "TOY-003"


def test_load_accepts_a_string_path(toy):
    assert C.load_catalog(str(toy[0])).ids() == C.load_catalog(Path(toy[0])).ids()


def test_optional_keys_default_to_null_and_kinds_to_empty():
    doc = minimal()
    del doc["cards"][2]["kind"], doc["cards"][2]["printingOf"]
    catalog = C.parse_catalog(doc)
    assert catalog.card("AB-002").kind is None and catalog.card("AB-002").printing_of is None
    assert catalog.kinds == {}
    assert [card.token for card in catalog] == [False, False, False] and catalog.token_ids() == []


def test_a_card_can_be_marked_as_a_token():
    doc = minimal()
    doc["cards"][0]["token"] = True
    doc["cards"][1]["token"] = False
    catalog = C.parse_catalog(doc)
    assert [card.token for card in catalog] == [True, False, False]
    assert catalog.token_ids() == ["AB-001"]
    # The mark is a fact about the card only: identity still goes by name.
    assert catalog.same_identity("AB-001", "AB-001A")


def test_the_documented_kinds_keys_are_checked_and_others_pass_through():
    doc = minimal()
    doc["kinds"] = {
        "landscape": ["site"],
        "foilProbability": {"unit": 0.1},
        "dice": ["unit"],
        "board": {"leader": "unit", "unit": ["unit", "site"]},
        "boost": {"shareKind": "site", "share": 0.1},
        "somethingElse": {"left": "alone"},
    }
    assert C.parse_catalog(doc).kinds == doc["kinds"]
    for key in ("landscape", "foilProbability", "dice", "board", "boost"):
        assert f"``{key}``" in C.__doc__, key


def test_parsing_does_not_keep_references_into_the_document():
    doc = minimal()
    doc["kinds"] = {"landscape": ["site"], "custom": {"anything": [1, 2]}}
    catalog = C.parse_catalog(doc)
    doc["kinds"]["landscape"].append("changed")
    assert catalog.kinds == {"landscape": ["site"], "custom": {"anything": [1, 2]}}


# --------------------------------------------------------------------------- helpers

def test_id_and_name_helpers():
    catalog = C.parse_catalog(minimal())
    assert "AB-001" in catalog and "ZZ-404" not in catalog
    assert catalog.name_of("AB-001A") == "Kite"
    assert catalog.names() == {"AB-001": "Kite", "AB-001A": "Kite", "AB-002": "Lamp"}
    assert catalog.ids_named("Kite") == ("AB-001", "AB-001A")
    assert catalog.ids_named("Lamp") == ("AB-002",)
    assert catalog.ids_named("kite") == ()  # names are exact
    assert catalog.ids_by_name() == {"Kite": ("AB-001", "AB-001A"), "Lamp": ("AB-002",)}
    assert catalog.same_identity("AB-001", "AB-001A")
    assert not catalog.same_identity("AB-001", "AB-002")


def test_printings_of_one_name_are_one_identity_in_the_toy_game(toy):
    catalog = C.load_catalog(toy[0])
    shared = {name: ids for name, ids in catalog.ids_by_name().items() if len(ids) > 1}
    assert shared == {"Coral Triangle": ("TOY-003", "TOY-003A"), "Plum Diamond": ("TOY-010", "TOY-P01")}
    assert len(catalog.ids_by_name()) == 18


def test_an_unknown_id_is_a_key_error_that_names_it():
    catalog = C.parse_catalog(minimal())
    with pytest.raises(KeyError, match="ZZ-404"):
        catalog.card("ZZ-404")
    with pytest.raises(KeyError, match="ZZ-404"):
        catalog.name_of("ZZ-404")
    with pytest.raises(KeyError, match="ZZ-404"):
        catalog.art_location("ZZ-404", "art")


# --------------------------------------------------------------------------- art

def test_is_url():
    assert C.is_url("https://example.org/a.png") and C.is_url("HTTP://example.org/a.png")
    assert not C.is_url("a.png") and not C.is_url("ftp://example.org/a.png") and not C.is_url("httpx/a.png")


def test_art_resolves_against_a_folder(tmp_path):
    catalog = C.parse_catalog(minimal())
    assert catalog.art_location("AB-001", tmp_path) == str(tmp_path / "AB-001.png")
    assert catalog.art_location("AB-001A", str(tmp_path)) == str(tmp_path / "alt" / "AB-001A.png")


def test_art_resolves_against_a_url_base():
    catalog = C.parse_catalog(minimal())
    assert catalog.art_location("AB-001", "https://cdn.example.org/cards") == "https://cdn.example.org/cards/AB-001.png"
    assert catalog.art_location("AB-001A", "https://cdn.example.org/cards/") == "https://cdn.example.org/cards/alt/AB-001A.png"
    assert C.resolve_art("my card#1.png", "https://cdn.example.org") == "https://cdn.example.org/my%20card%231.png"


def test_art_that_is_a_url_ignores_the_base(tmp_path):
    catalog = C.parse_catalog(minimal())
    url = "https://example.org/a b.png"
    assert catalog.art_location("AB-002") == url
    assert catalog.art_location("AB-002", tmp_path) == url
    assert catalog.art_location("AB-002", "https://cdn.example.org") == url


def test_relative_art_needs_a_base():
    with pytest.raises(ValueError, match="AB-001.png.*base"):
        C.parse_catalog(minimal()).art_location("AB-001")


def test_toy_art_files_exist_where_the_catalog_says(toy):
    catalog_path, art_dir = toy
    catalog = C.load_catalog(catalog_path)
    for card in catalog:
        assert Path(catalog.art_location(card.id, art_dir)).is_file()


# --------------------------------------------------------------------------- cards without art

def test_a_card_without_art_states_art_as_null():
    doc = minimal()
    doc["cards"].append({"id": "AB-003", "name": "Moth", "kind": "unit", "printingOf": None, "art": None})
    catalog = C.parse_catalog(doc)
    card = catalog.card("AB-003")
    assert (card.art, card.has_art, catalog.card("AB-001").has_art) == (None, False, True)
    assert catalog.ids_without_art() == ["AB-003"]
    assert catalog.ids_with_art() == ["AB-001", "AB-001A", "AB-002"]
    assert catalog.ids() == ["AB-001", "AB-001A", "AB-002", "AB-003"] and len(catalog) == 4
    assert catalog.name_of("AB-003") == "Moth" and catalog.identity_of("AB-003") == "Moth"
    with pytest.raises(C.NoArtError, match="'AB-003' has no art"):
        catalog.art_location("AB-003", "art")
    assert issubclass(C.NoArtError, ValueError)


@pytest.mark.parametrize("rule", list(C.IDENTITY_RULES))
def test_a_card_without_art_is_an_identity_and_takes_part_in_printings(rule):
    doc = minimal()
    doc["identity"] = rule
    doc["cards"][0]["art"] = None          # the root without art, its printing with art
    doc["cards"].append({"id": "AB-002P", "name": "Lamp", "printingOf": "AB-002", "art": None})  # a printing without art
    catalog = C.parse_catalog(doc)
    assert catalog.ids_without_art() == ["AB-001", "AB-002P"]
    assert catalog.same_identity("AB-001", "AB-001A") and catalog.same_identity("AB-002", "AB-002P")
    assert catalog.identity_groups() == [["AB-001", "AB-001A"], ["AB-002", "AB-002P"]]
    # a bank holds the cards with art only: the groups of its ids
    assert catalog.identity_groups(catalog.ids_with_art()) == []
    if rule == C.IDENTITY_PRINTING:
        assert catalog.identity_of("AB-001A") == "AB-001" and catalog.identity_anchors() == {"AB-001", "AB-002"}


def test_a_catalog_of_cards_without_art_only_is_a_catalog():
    doc = {"game": "toy", "locale": "en", "cards": [{"id": "AB-009", "name": "Moth", "art": None}]}
    assert C.parse_catalog(doc).ids_with_art() == []


# --------------------------------------------------------------------------- validation failures

def _set(path: list, value):
    def change(doc: dict) -> None:
        target = doc
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return change


def _delete(path: list):
    def change(doc: dict) -> None:
        target = doc
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]
    return change


CARD_FAILURES = [
    # (what is wrong, change to make, fragments the message must contain)
    ("name missing", _delete(["cards", 1, "name"]), ["'AB-001A'", "cards[1]", "missing", "name"]),
    ("art missing", _delete(["cards", 0, "art"]), ["'AB-001'", "cards[0]", "missing", "art"]),
    ("name empty", _set(["cards", 1, "name"], ""), ["'AB-001A'", '"name"']),
    ("name blank", _set(["cards", 1, "name"], "   "), ["'AB-001A'", '"name"']),
    ("name padded", _set(["cards", 1, "name"], "Kite "), ["'AB-001A'", "whitespace"]),
    ("name not text", _set(["cards", 2, "name"], 7), ["'AB-002'", '"name"']),
    ("kind not text", _set(["cards", 2, "kind"], ["unit"]), ["'AB-002'", '"kind"']),
    ("kind empty", _set(["cards", 2, "kind"], ""), ["'AB-002'", '"kind"']),
    ("printingOf not text", _set(["cards", 1, "printingOf"], 1), ["'AB-001A'", '"printingOf"']),
    ("printingOf unknown", _set(["cards", 1, "printingOf"], "AB-999"), ["'AB-001A'", "'AB-999'", "not a card"]),
    ("printingOf itself", _set(["cards", 1, "printingOf"], "AB-001A"), ["'AB-001A'", "itself"]),
    ("art empty", _set(["cards", 0, "art"], ""), ["'AB-001'", '"art"']),
    ("art not text", _set(["cards", 0, "art"], False), ["'AB-001'", '"art"', "or null"]),
    ("art a number", _set(["cards", 0, "art"], 7), ["'AB-001'", '"art"']),
    ("art blank", _set(["cards", 0, "art"], "  "), ["'AB-001'", '"art"']),
    ("art missing on a card that is a printing", _delete(["cards", 1, "art"]), ["'AB-001A'", "missing", "art"]),
    ("art absolute", _set(["cards", 0, "art"], "/srv/art/AB-001.png"), ["'AB-001'", "relative"]),
    ("art on a drive", _set(["cards", 0, "art"], "C:/art/AB-001.png"), ["'AB-001'", "relative"]),
    ("art climbs out", _set(["cards", 0, "art"], "../AB-001.png"), ["'AB-001'", '".."']),
    ("art climbs out later", _set(["cards", 0, "art"], "a/../../AB-001.png"), ["'AB-001'", '".."']),
    ("art backslashes", _set(["cards", 0, "art"], "alt\\AB-001.png"), ["'AB-001'", "forward slashes"]),
    ("art other scheme", _set(["cards", 0, "art"], "ftp://example.org/a.png"), ["'AB-001'", "scheme"]),
    ("unknown card key", _set(["cards", 2, "printingof"], None), ["'AB-002'", "unknown", "printingof"]),
    ("token not a boolean", _set(["cards", 2, "token"], "yes"), ["'AB-002'", '"token"', "true or false"]),
    ("token a number", _set(["cards", 0, "token"], 1), ["'AB-001'", '"token"']),
    ("token null", _set(["cards", 0, "token"], None), ["'AB-001'", '"token"']),
    ("duplicate id", _set(["cards", 2, "id"], "AB-001"), ["'AB-001'", "cards[2]", "duplicate", "cards[0]"]),
    ("id padded", _set(["cards", 2, "id"], " AB-002"), ["cards[2]", "whitespace"]),
    ("id with a space inside", _set(["cards", 2, "id"], "AB 002"), ["cards[2]", "'AB 002'", "no space"]),
    ("id with a line break at the end", _set(["cards", 2, "id"], "AB-002\n"), ["cards[2]", "whitespace"]),
    ("id with a tab inside", _set(["cards", 2, "id"], "AB\t002"), ["cards[2]", "no space"]),
    ("id with a control character", _set(["cards", 2, "id"], "AB\x00002"), ["cards[2]", "control character"]),
    ("id with a no-break space", _set(["cards", 2, "id"], "AB\u00a0002"), ["cards[2]", "no space"]),
    ("id too long", _set(["cards", 2, "id"], "A" * 65), ["cards[2]", "at most 64"]),
    ("id missing", _delete(["cards", 2, "id"]), ["cards[2]", '"id"']),
    ("id not text", _set(["cards", 2, "id"], 2), ["cards[2]", '"id"']),
    ("card not an object", _set(["cards", 1], "AB-001A"), ["cards[1]", "object"]),
]


@pytest.mark.parametrize("change,fragments", [case[1:] for case in CARD_FAILURES], ids=[case[0] for case in CARD_FAILURES])
def test_a_bad_card_is_named_in_the_error(change, fragments):
    doc = minimal()
    change(doc)
    with pytest.raises(C.CatalogError) as raised:
        C.parse_catalog(doc, source="toy.json")
    message = str(raised.value)
    assert message.startswith("toy.json: ")
    for fragment in fragments:
        assert fragment in message, message


TOP_FAILURES = [
    ("not an object", lambda doc: [], ["JSON object"]),
    ("game missing", _delete(["game"]), ['"game"']),
    ("game empty", _set(["game"], ""), ['"game"']),
    ("locale not text", _set(["locale"], 5), ['"locale"']),
    ("cards missing", _delete(["cards"]), ['"cards"']),
    ("cards empty", _set(["cards"], []), ['"cards"', "non-empty"]),
    ("cards not a list", _set(["cards"], {"AB-001": {}}), ['"cards"']),
    ("unknown top key", _set(["card"], []), ["unknown", "card"]),
    ("tokens list at the top", _set(["tokens"], ["Spark"]), ["unknown", "tokens"]),
    ("kinds not an object", _set(["kinds"], ["site"]), ['"kinds"']),
    ("landscape not a list", _set(["kinds"], {"landscape": "site"}), ["kinds.landscape"]),
    ("landscape of numbers", _set(["kinds"], {"landscape": [1]}), ["kinds.landscape"]),
    ("foil not an object", _set(["kinds"], {"foilProbability": 0.1}), ["kinds.foilProbability"]),
    ("foil above one", _set(["kinds"], {"foilProbability": {"unit": 1.5}}), ["kinds.foilProbability", "'unit'"]),
    ("foil a boolean", _set(["kinds"], {"foilProbability": {"unit": True}}), ["kinds.foilProbability", "'unit'"]),
    ("dice not a list", _set(["kinds"], {"dice": "unit"}), ["kinds.dice"]),
    ("dice of numbers", _set(["kinds"], {"dice": [1]}), ["kinds.dice"]),
    ("board not an object", _set(["kinds"], {"board": ["unit"]}), ["kinds.board"]),
    ("board group empty", _set(["kinds"], {"board": {"leader": []}}), ["kinds.board.leader"]),
    ("board group of numbers", _set(["kinds"], {"board": {"unit": [3]}}), ["kinds.board.unit"]),
    ("boost incomplete", _set(["kinds"], {"boost": {"share": 0.1}}), ["kinds.boost"]),
    ("boost extra key", _set(["kinds"], {"boost": {"shareKind": "site", "share": 0.1, "x": 1}}), ["kinds.boost"]),
    ("boost kind empty", _set(["kinds"], {"boost": {"shareKind": "", "share": 0.1}}), ["kinds.boost.shareKind"]),
    ("boost share negative", _set(["kinds"], {"boost": {"shareKind": "site", "share": -0.1}}), ["kinds.boost.share"]),
]


@pytest.mark.parametrize("change,fragments", [case[1:] for case in TOP_FAILURES], ids=[case[0] for case in TOP_FAILURES])
def test_a_bad_document_is_rejected(change, fragments):
    doc = minimal()
    replaced = change(doc)
    with pytest.raises(C.CatalogError) as raised:
        C.parse_catalog(doc if replaced is None else replaced, source="toy.json")
    for fragment in fragments:
        assert fragment in str(raised.value), str(raised.value)


def test_catalog_error_is_a_value_error():
    assert issubclass(C.CatalogError, ValueError)


def test_validation_does_not_change_the_document():
    doc = minimal()
    doc["kinds"] = copy.deepcopy(toy_game.KINDS)
    before = copy.deepcopy(doc)
    C.parse_catalog(doc)
    assert doc == before


def test_a_file_that_is_not_json_is_a_catalog_error(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text('{"game": "toy",', encoding="utf-8")
    with pytest.raises(C.CatalogError, match="broken.json: not valid JSON"):
        C.load_catalog(path)


def test_errors_from_a_file_carry_its_name(tmp_path):
    doc = minimal()
    doc["cards"][1]["name"] = ""
    path = tmp_path / "mine.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(C.CatalogError, match=r"mine\.json: card 'AB-001A' \(cards\[1\]\)"):
        C.load_catalog(path)


def test_a_missing_file_is_reported_by_the_operating_system(tmp_path):
    with pytest.raises(FileNotFoundError):
        C.load_catalog(tmp_path / "absent.json")


def test_the_card_id_rule_is_the_one_the_manifest_format_uses():
    from cardvision.data import schema
    assert C.CARD_ID_PATTERN == schema.CARD_ID_PATTERN
    assert C.CARD_ID_MAX_LENGTH == 64

