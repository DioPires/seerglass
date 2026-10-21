# SPDX-License-Identifier: Apache-2.0
"""Look groups (``cardvision.mtg.looks``): the key, the scopes and the statistics, on a small
hand-made catalog."""

from __future__ import annotations

import pytest

from cardvision.core.catalog import parse_catalog
from cardvision.mtg import looks as L


def _entry(card_id, oracle, illustration, *, printing_of=None, effects=(), promos=(), legal=("standard",),
           games=("paper", "arena"), frame="2015"):
    extra = {"oracle_id": oracle, "illustration_id": illustration, "frame": frame, "border_color": "black",
             "full_art": False, "textless": False, "frame_effects": list(effects), "promo_types": list(promos),
             "legal": list(legal), "games": list(games)}
    return {"id": card_id, "name": oracle.upper(), "kind": None, "printingOf": printing_of,
            "art": card_id.replace(":", "-") + ".jpg", "extra": extra}


@pytest.fixture
def catalog():
    return parse_catalog({"game": "mtg", "locale": "en", "identity": "printing", "cards": [
        _entry("a:1", "a", "ia"),
        _entry("b:1", "a", "ia", printing_of="a:1"),                       # a's twin: same look
        _entry("c:1", "a", "ia", printing_of="a:1", effects=["showcase"]),  # same picture, other frame
        _entry("d:1", "a", "ia", printing_of="a:1", promos=["prerelease"]),  # a stamp only
        _entry("e:1", "a", "ia", printing_of="a:1", effects=["legendary"]),
        _entry("f:1", "f", None, games=("paper",), legal=("modern",)),     # no illustration id
        _entry("g:1", "g", None, games=("paper",), legal=("modern",)),
    ]})


def test_the_look_key(catalog):
    key = {card.id: L.look_key(card) for card in catalog}
    assert key["a:1"] == key["b:1"] == key["d:1"]
    assert key["c:1"] != key["a:1"] and key["e:1"] != key["a:1"]
    assert key["f:1"] != key["g:1"]  # no illustration: a look of its own
    assert L.look_key(catalog.card("e:1"), "visible") == key["a:1"]  # the legendary crown is small
    assert L.look_key(catalog.card("d:1"), "stamps") != L.look_key(catalog.card("a:1"), "stamps")
    with pytest.raises(ValueError):
        L.look_key(catalog.card("a:1"), "nope")


def test_scopes(catalog):
    assert L.scope_ids(catalog, "standard-arena") == ["a:1", "b:1", "c:1", "d:1", "e:1"]
    assert L.scope_ids(catalog, "modern") == ["f:1", "g:1"]
    assert len(L.scope_ids(catalog, "all")) == 7


def test_look_stats(catalog):
    stats = L.look_stats(catalog, L.scope_ids(catalog, "all"))
    assert stats["printings"] == 7 and stats["identities"] == 3 and stats["illustrations"] == 1
    assert stats["lookGroups"] == 5 and stats["unknownIllustration"] == 2
    assert stats["printingsWithVisualTwin"] == 3 and stats["shareWithVisualTwin"] == round(3 / 7, 4)
    assert stats["printingsByLookGroupSize"] == {"1": 4, "2": 0, "3-5": 3, "6-20": 0, "21+": 0}
    assert stats["identitiesWithSeveralPrintings"] == 1
    assert stats["identitiesAllPrintingsDistinct"] == 0 and stats["identitiesOneLookForAll"] == 0
    assert stats["largestLookGroup"]["size"] == 3
    visible = L.look_stats(catalog, L.scope_ids(catalog, "all"), "visible")
    assert visible["printingsWithVisualTwin"] == 4


def test_the_report_and_its_table(catalog):
    report = L.look_report(catalog, scopes=("standard-arena", "all"))
    assert set(report["scopes"]) == {"standard-arena", "all"}
    table = L.markdown_table(report)
    assert table.count("\n") == 3 and "| standard-arena | 5 |" in table
