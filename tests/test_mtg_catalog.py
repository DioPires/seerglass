# SPDX-License-Identifier: Apache-2.0
"""The Magic catalog builder (``cardvision.mtg.catalog``) on hand-made Scryfall records: what is
kept, the ids, double-faced cards, the identity rule with its aliases, and the bulk reader."""

from __future__ import annotations

import gzip
import json

import pytest

from cardvision.core.catalog import parse_catalog
from cardvision.mtg import catalog as M
from cardvision.mtg.scryfall import read_cards


def card(name, set_code="abc", cn="1", *, oracle=None, illustration=None, scryfall_id=None, **fields):
    """A Scryfall card object with the fields the builder reads."""
    record = {
        "object": "card", "id": scryfall_id or f"id-{set_code}-{cn}", "oracle_id": oracle or f"o-{name}",
        "name": name, "lang": "en", "released_at": "2020-01-01", "layout": "normal",
        "image_status": "highres_scan", "type_line": "Creature — Elf",
        "image_uris": {"normal": f"https://cards.scryfall.io/normal/front/{set_code}{cn}.jpg?111"},
        "legalities": {"standard": "not_legal", "modern": "legal", "commander": "legal"},
        "games": ["paper", "arena"], "finishes": ["nonfoil", "foil"], "oversized": False, "set": set_code,
        "set_type": "expansion", "collector_number": cn, "digital": False, "illustration_id": illustration or f"i-{name}",
        "border_color": "black", "frame": "2015", "full_art": False, "textless": False,
    }
    record.update(fields)
    return record


def dfc(name_front, name_back, set_code="dfc", cn="5", layout="transform"):
    record = card(f"{name_front} // {name_back}", set_code, cn, layout=layout)
    del record["image_uris"], record["illustration_id"]
    record["card_faces"] = [
        {"name": name_front, "type_line": "Creature — Human", "illustration_id": f"i-{name_front}",
         "image_uris": {"normal": f"https://cards.scryfall.io/normal/front/{cn}.jpg?2"}},
        {"name": name_back, "type_line": "Creature — Horror", "illustration_id": f"i-{name_back}",
         "image_uris": {"normal": f"https://cards.scryfall.io/normal/back/{cn}.jpg?2"}},
    ]
    return record


def build(*records):
    return M.build_catalog(records)


def test_ids_are_set_and_collector_number_and_the_catalog_loads():
    result = build(card("Llanowar Elves", "m19", "314"), card("Llanowar Elves", "dom", "168", released_at="2018-04-27"),
                   card("Forest", "blb", "280★"))
    catalog = parse_catalog(result.catalog)
    assert catalog.ids() == ["blb:280★", "dom:168", "m19:314"]
    assert catalog.identity == "printing" and catalog.game == "mtg"
    assert result.images["m19:314"].startswith("https://cards.scryfall.io/normal/")
    assert catalog.card("m19:314").art == "id-m19-314.jpg"


def test_one_representative_per_oracle_card_the_earliest_printing():
    result = build(card("Llanowar Elves", "m19", "314", released_at="2018-07-13"),
                   card("Llanowar Elves", "dom", "168", released_at="2018-04-27"),
                   card("Llanowar Elves", "lea", "210", released_at="1993-08-05"))
    catalog = parse_catalog(result.catalog)
    assert catalog.card("lea:210").printing_of is None
    assert catalog.card("dom:168").printing_of == "lea:210" and catalog.card("m19:314").printing_of == "lea:210"
    assert catalog.same_identity("dom:168", "m19:314")


def test_cards_that_share_a_name_but_not_an_oracle_card_stay_apart():
    # Under the printing rule a name joins nothing: only the Oracle card does.
    result = build(card("Very Cryptic Command", "ust", "49d", oracle="o1", illustration="i1"),
                   card("Very Cryptic Command", "ust", "49e", oracle="o2", illustration="i2"))
    catalog = parse_catalog(result.catalog)
    assert not catalog.same_identity("ust:49d", "ust:49e")


def test_a_printing_that_reuses_another_cards_illustration_is_an_alias_of_it():
    result = build(
        card("Island", "cst", "374", illustration="shared", released_at="2006-07-21"),
        card("Island", "m19", "265", illustration="island-2"),
        card("Snow-Covered Island", "ice", "371", illustration="shared", released_at="1995-06-03"),
        card("Snow-Covered Island", "khm", "277", illustration="snow-2", released_at="2021-02-05"),
    )
    catalog = parse_catalog(result.catalog)
    # The earliest printing of the picture is the Snow-Covered Island, so the Island that reuses
    # it is the alias; every other printing keeps its own card.
    assert catalog.same_identity("cst:374", "ice:371"), "same picture, so one identity"
    assert catalog.card("cst:374").extra["aliasOf"] == "o-Island"
    assert not catalog.same_identity("m19:265", "khm:277")
    assert catalog.same_identity("m19:265", "m19:265") and not catalog.same_identity("m19:265", "ice:371")
    assert catalog.same_identity("khm:277", "ice:371")
    assert "aliasOf" not in catalog.card("m19:265").extra
    assert result.aliases == {"o-Snow-Covered Island": ["o-Island"]}


def test_a_shared_picture_across_environments_keeps_both_cards():
    # An Arena-only Alchemy card drawn from a paper-only card's art: never on one screen.
    result = build(
        card("Shadowheart, Dark Justiciar", "clb", "146", illustration="shared", released_at="2022-06-10",
             games=["paper"]),
        card("Shadowheart, Sharran Cleric", "hbg", "10", illustration="shared", released_at="2022-07-07",
             games=["arena"], digital=True),
    )
    catalog = parse_catalog(result.catalog)
    assert not catalog.same_identity("hbg:10", "clb:146")
    assert catalog.card("hbg:10").extra["sharesArtWith"] == "o-Shadowheart, Dark Justiciar"
    assert result.aliases == {}


def test_an_alchemy_rebalanced_card_is_an_alias_of_the_card_it_rebalances():
    result = build(card("Alrund's Epiphany", "khm", "38"),
                   card("A-Alrund's Epiphany", "khm", "A-38", illustration="re-drawn", games=["arena"]))
    catalog = parse_catalog(result.catalog)
    assert catalog.same_identity("khm:A-38", "khm:38")
    assert catalog.card("khm:A-38").printing_of == "khm:38"


def test_double_faced_cards_have_an_entry_per_face_of_one_identity():
    result = build(dfc("Delver of Secrets", "Insectile Aberration", "isd", "51"),
                   card("Fire // Ice", "apc", "128", layout="split", type_line="Instant // Instant"))
    catalog = parse_catalog(result.catalog)
    front, back = catalog.card("isd:51"), catalog.card("isd:51:b")
    assert (front.name, back.name) == ("Delver of Secrets", "Insectile Aberration")
    assert back.printing_of == "isd:51" and catalog.same_identity("isd:51", "isd:51:b")
    assert back.art.endswith("-b.jpg") and front.extra["face"] == 0 and back.extra["face"] == 1
    assert result.images["isd:51:b"].endswith("/back/51.jpg?2")
    assert back.extra["illustration_id"] == "i-Insectile Aberration"
    # A split card is one picture: one entry, its full name.
    assert catalog.card("apc:128").name == "Fire // Ice" and "apc:128:b" not in catalog
    assert catalog.card("apc:128").kind == "Instant"


def test_reversible_cards_take_the_oracle_card_from_their_faces():
    record = dfc("Jinnie Fay, Jetmir's Second", "Jinnie Fay, Jetmir's Second", "sld", "1556", layout="reversible_card")
    record["oracle_id"] = None
    for face in record["card_faces"]:
        face["oracle_id"] = "o-jinnie"
    result = build(record, card("Jinnie Fay, Jetmir's Second", "snc", "190", oracle="o-jinnie"))
    catalog = parse_catalog(result.catalog)
    assert catalog.same_identity("sld:1556", "sld:1556:b") and catalog.same_identity("sld:1556", "snc:190")


def test_what_is_left_out_and_why():
    records = [
        card("Goblin", "tm19", "1", layout="token"),
        card("Ajani Emblem", "tm19", "2", layout="emblem"),
        card("Sol Ring", "amh2", "3", layout="art_series"),
        card("Sol Ring", "ocmd", "4", oversized=True),
        card("Double-Faced Substitute Card", "sznr", "5", set_type="token", type_line="Card"),
        card("Sol Ring", "abc", "6", games=["mtgo"]),
        card("Sol Ring", "abc", "7", lang="ja"),
        card("Sol Ring", "abc", "8"),
    ]
    result = build(*records)
    assert parse_catalog(result.catalog).ids() == ["abc:8"]
    reasons = {row["id"]: row["reason"] for row in result.exclusions}
    assert reasons == {"tm19:1": "layout:token", "tm19:2": "layout:emblem", "amh2:3": "layout:art_series",
                       "ocmd:4": "oversized", "sznr:5": "set_type:token"}
    assert result.counts["excluded:not-paper-or-arena"] == 1 and result.counts["excluded:not-english"] == 1


def test_a_placeholder_image_is_a_card_without_art():
    result = build(card("Disa the Restless", "m3c", "148", image_status="placeholder"), card("Disa the Restless", "mh3", "1"))
    catalog = parse_catalog(result.catalog)
    assert catalog.card("m3c:148").art is None and "m3c:148" not in result.images
    assert catalog.card("mh3:1").printing_of is None, "the representative is a printing with art"
    assert catalog.card("m3c:148").printing_of == "mh3:1"


def test_extra_carries_the_fields_and_the_foil_only_flag():
    result = build(card("Sol Ring", "sld", "1011", finishes=["foil"], frame_effects=["showcase", "legendary"],
                        arena_id=None, mtgo_id=999, legalities={"commander": "legal", "vintage": "restricted"}),
                   card("Sol Ring", "c21", "263", finishes=["nonfoil", "etched"]),
                   card("Sol Ring", "cmr", "472", finishes=["etched"]))
    catalog = parse_catalog(result.catalog)
    extra = catalog.card("sld:1011").extra
    for key in ("oracle_id", "illustration_id", "frame", "frame_effects", "border_color", "full_art", "finishes",
                "lang", "arena_id", "mtgo_id", "layout", "games", "set_type", "released_at", "image_status"):
        assert key in extra, key
    assert extra["frame_effects"] == ["legendary", "showcase"] and extra["legal"] == ["vintage", "commander"]
    assert extra["foilOnly"] is True
    assert catalog.card("c21:263").extra["foilOnly"] is False
    assert catalog.card("cmr:472").extra["foilOnly"] is True, "etched-only counts as foil-only"


def test_the_bulk_reader_takes_json_lines_and_arrays_gzipped_or_not(tmp_path):
    records = [card("A", cn="1"), card("B", cn="2")]
    (tmp_path / "cards.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    (tmp_path / "cards.json").write_text(json.dumps(records), encoding="utf-8")
    with gzip.open(tmp_path / "cards.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write("\n".join(json.dumps(r) for r in records))
    for name in ("cards.jsonl", "cards.json", "cards.jsonl.gz"):
        assert [r["name"] for r in read_cards(tmp_path / name)] == ["A", "B"]


def test_the_command_writes_the_catalog_and_its_companions(tmp_path):
    from cardvision.mtg.__main__ import main

    bulk = tmp_path / "bulk.jsonl"
    bulk.write_text("\n".join(json.dumps(r) for r in [card("A", cn="1"), card("Goblin", cn="2", layout="token")]),
                    encoding="utf-8")
    assert main(["build-catalog", "--bulk", str(bulk), "--out", str(tmp_path / "out")]) == 0
    for name in ("catalog.json", "images.json", "exclusions.json", "aliases.json", "catalog-summary.json"):
        assert (tmp_path / "out" / name).is_file(), name
    assert json.loads((tmp_path / "out" / "exclusions.json").read_text(encoding="utf-8"))[0]["reason"] == "layout:token"


def test_a_release_catalog_keeps_what_the_runtime_reads_and_the_same_identities():
    full = build(card("Llanowar Elves", "m19", "314", released_at="2018-07-13"),
                 card("Llanowar Elves", "dom", "168", released_at="2018-04-27"), dfc("Delver of Secrets", "Insectile Aberration"),
                 card("Forest", "blb", "280★", image_status="placeholder")).catalog
    full["cards"].append({"id": "tblb:1", "name": "Rabbit token", "kind": "Creature", "printingOf": None,
                          "art": "id-tblb-1.jpg", "token": True, "extra": {"set": "tblb"}})
    cut = M.release_catalog(full)
    assert [key for key in cut] == ["game", "locale", "identity", "cards"]
    assert all("extra" not in entry for entry in cut["cards"])
    assert [{k: v for k, v in entry.items() if k != "extra"} for entry in full["cards"]] == cut["cards"]
    before, after = parse_catalog(full), parse_catalog(cut)
    assert after.identity_groups(after.ids()) == before.identity_groups(before.ids()) != []
    assert after.card("blb:280★").art is None and after.token_ids() == ["tblb:1"]
    assert "extra" in full["cards"][0]  # the full catalog is left as it was


def test_the_release_catalog_command_writes_the_cut_catalog(tmp_path):
    from cardvision.mtg.__main__ import main

    (tmp_path / "catalog.json").write_text(json.dumps(build(card("A", cn="1"), card("A", "xyz", "2")).catalog),
                                           encoding="utf-8")
    assert main(["release-catalog", "--catalog", str(tmp_path / "catalog.json"), "--out", str(tmp_path / "rel" / "c.json")]) == 0
    written = json.loads((tmp_path / "rel" / "c.json").read_text(encoding="utf-8"))
    assert [set(entry) for entry in written["cards"]] == [{"id", "name", "kind", "printingOf", "art"}] * 2


@pytest.mark.parametrize("type_line, kind", [("Legendary Creature — Elf", "Creature"), ("Basic Land — Forest", "Land"),
                                             ("Artifact Creature — Golem", "Creature"), ("Tribal Instant — Elf", "Instant"), ("Tribal — Elf", "Kindred"),
                                             ("Instant // Sorcery", "Instant"), (None, None)])
def test_the_kind_is_the_first_card_type(type_line, kind):
    assert M.kind_of(type_line) == kind
