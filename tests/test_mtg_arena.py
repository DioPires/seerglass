# SPDX-License-Identifier: Apache-2.0
"""The Arena catalog (``cardvision.mtg.arena``): what it keeps, its representatives, its tile
entries and its tokens, on a small hand-made catalog."""

from __future__ import annotations

import json

import pytest

from cardvision.core.catalog import parse_catalog
from cardvision.mtg import arena as A


def _entry(card_id, name, oracle, *, kind="Creature", printing_of=None, games=("paper", "arena"),
           released="2020-01-01", art=True, frame="2015"):
    extra = {"set": card_id.split(":")[0], "collector_number": card_id.split(":")[1], "face": 0,
             "oracle_id": oracle, "illustration_id": "i-" + oracle, "frame": frame, "frame_effects": [],
             "border_color": "black", "full_art": False, "layout": "normal", "released_at": released,
             "digital": False, "legal": ["standard"], "games": list(games), "promo_types": []}
    return {"id": card_id, "name": name, "kind": kind, "printingOf": printing_of,
            "art": card_id.replace(":", "-") + ".jpg" if art else None, "extra": extra}


@pytest.fixture
def full():
    return {"game": "mtg", "locale": "en", "identity": "printing", "cards": [
        _entry("lea:1", "Kite", "k", games=("paper",), released="1993-08-05"),     # the paper representative
        _entry("dmu:2", "Kite", "k", printing_of="lea:1", released="2022-09-09"),
        _entry("m21:3", "Kite", "k", printing_of="lea:1", released="2020-07-03"),
        _entry("dmu:9", "Bolt", "b", kind="Instant"),
        _entry("dmu:10", "Forest", "f", kind="Land", art=False),
        _entry("lea:4", "Paper Only", "p", games=("paper",)),
    ]}


def _tokens():
    return [
        {"id": "tdmu:1", "name": "Soldier", "kind": "Creature", "printingOf": None, "art": "s1.jpg", "token": True,
         "extra": {"set": "tdmu", "face": 0, "released_at": "2022-09-09"}},
        {"id": "tm21:1", "name": "Soldier", "kind": "Creature", "printingOf": None, "art": "s2.jpg", "token": True,
         "extra": {"set": "tm21", "face": 0, "released_at": "2020-07-03"}},
        {"id": "tdmu:2", "name": "Kite", "kind": "Creature", "printingOf": None, "art": "k.jpg", "token": True,
         "extra": {"set": "tdmu", "face": 0, "released_at": "2022-09-09"}},
    ]


def test_the_pool_and_its_representatives(full):
    result = A.arena_catalog(full, tiles=False)
    cards = {card["id"]: card for card in result.catalog["cards"]}
    assert set(cards) == {"dmu:2", "m21:3", "dmu:9", "dmu:10"}  # paper-only printings are left out
    assert cards["m21:3"]["printingOf"] is None                  # the earliest Arena printing now represents
    assert cards["dmu:2"]["printingOf"] == "m21:3"
    assert "games" not in cards["dmu:2"]["extra"] and cards["dmu:2"]["extra"]["oracle_id"] == "k"
    parse_catalog(result.catalog)


def test_tile_entries_for_permanents_with_art(full):
    result = A.arena_catalog(full)
    cards = {card["id"]: card for card in result.catalog["cards"]}
    assert {card_id for card_id in cards if card_id.endswith(A.TILE_SUFFIX)} == {"dmu:2:art", "m21:3:art"}
    tile = cards["dmu:2:art"]
    assert tile["printingOf"] == "m21:3" and tile["name"] == "Kite"
    assert tile["art"] == "tiles/dmu-2.jpg" and tile["extra"]["artOf"] == "dmu:2"
    assert A.is_tile(tile) and not A.is_tile(cards["dmu:2"])
    assert result.counts["tile-entries"] == 2  # the instant has none, nor the land without art


def test_tokens_group_by_name_and_join_a_card_of_their_name(full):
    result = A.arena_catalog(full, _tokens())
    cards = {card["id"]: card for card in result.catalog["cards"]}
    assert cards["tdmu:1"]["name"] == cards["tm21:1"]["name"] == "Soldier token"
    assert cards["tm21:1"]["printingOf"] is None and cards["tdmu:1"]["printingOf"] == "tm21:1"
    assert cards["tdmu:2"]["name"] == "Kite" and cards["tdmu:2"]["printingOf"] == "m21:3"
    assert cards["tdmu:1:art"]["token"] is True and cards["tdmu:1:art"]["printingOf"] == "tm21:1"
    assert result.counts["token-identities"] == 1 and result.counts["tokens-joining-a-card"] == 1


def test_token_entries_from_scryfall_objects():
    soldier = {"layout": "token", "lang": "en", "set": "tdmu", "set_type": "token", "collector_number": "1",
               "id": "abc", "name": "Soldier", "type_line": "Token Creature — Soldier",
               "image_uris": {"normal": "https://cards.scryfall.io/normal/front/a/b/abc.jpg?1"}}
    monarch = {**soldier, "id": "m", "collector_number": "2", "name": "The Monarch", "type_line": "Card"}
    other = {**soldier, "id": "o", "set": "tneo", "collector_number": "3"}
    japanese = {**soldier, "id": "j", "lang": "ja", "collector_number": "4"}
    entries, images = A.token_entries([soldier, monarch, other, japanese], {"dmu"})
    assert [entry["id"] for entry in entries] == ["tdmu:1"]  # helper cards, other sets and languages stay out
    assert entries[0]["token"] is True and entries[0]["art"] == "abc.jpg" and entries[0]["kind"] == "Creature"
    assert images == {"tdmu:1": soldier["image_uris"]["normal"]}
    assert A.parent_set({"set": "tdmu", "set_type": "token"}) == "dmu"
    assert A.parent_set({"set": "dmu", "set_type": "expansion"}) == "dmu"


def test_the_tile_is_the_name_bar_and_the_art_box():
    from PIL import Image

    image = Image.new("RGB", (488, 680), "white")
    tile = A.tile_image(image, {"frame": "2015", "border_color": "black", "full_art": False})
    width, height = tile.size
    assert (width, height) == (414, 350)  # as wide as the art box, from the name bar's top: about 1.2 to 1
    assert A.tile_image(image, {"frame": "1997"}).size == (380, 347)
    assert A.art_box({"border_color": "borderless"}) == A.FULL_ART_BOX
    assert A.art_box({"frame": "1997"}) == A.ART_BOXES["1997"]


def test_render_tiles_writes_each_tile_once(tmp_path, full):
    from PIL import Image

    result = A.arena_catalog(full)
    Image.new("RGB", (488, 680), "gray").save(tmp_path / "dmu-2.jpg")
    report = A.render_tiles(result.catalog, tmp_path)
    assert report.written == 1 and report.missing_source == ["m21:3:art"]
    assert (tmp_path / "tiles" / "dmu-2.jpg").is_file()
    again = A.render_tiles(result.catalog, tmp_path)
    assert again.written == 0 and again.present == 1


def test_a_base_bank_from_embeddings_is_the_bank_build_bank_makes(tmp_path, toy, tiny_model):
    from cardvision.bank import build as bank_build
    from cardvision.core.catalog import load_catalog
    from cardvision.core.onnx import create_session
    from cardvision.mtg.__main__ import main
    from cardvision.mtg.crowding import embed

    catalog_path, art = toy
    catalog = load_catalog(catalog_path)
    embed(catalog, art, tiny_model, tmp_path / "emb", workers=1)
    base_path = tmp_path / "base.json"
    assert main(["bank-from-embeddings", "--catalog", str(catalog_path), "--art", str(art),
                 "--embeddings", str(tmp_path / "emb"), "--model", "tiny", "--out", str(base_path)]) == 0
    base = json.loads(base_path.read_text(encoding="utf-8"))
    session = create_session(tiny_model)
    extended = bank_build.build_bank(catalog, art, session, base=base)
    assert extended.embedded == [] and extended.reembedded == []  # every row kept, none embedded again
    fresh = bank_build.build_bank(catalog, art, session, model="tiny").bank
    assert [e["vec"] for e in extended.bank["entries"]] == [e["vec"] for e in fresh["entries"]]
    assert [e["artV"] for e in extended.bank["entries"]] == [e["artV"] for e in fresh["entries"]]
