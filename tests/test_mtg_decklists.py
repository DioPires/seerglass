# SPDX-License-Identifier: Apache-2.0
"""decklist-names: front-face names from deck-list sites made full, nothing else changed."""

from __future__ import annotations

import json

from cardvision.mtg.__main__ import main
from cardvision.mtg.decklists import full_names

CARDS = [
    {"id": "woe:221", "name": "Callous Sell-Sword // Burn Together"},
    {"id": "woe:289", "name": "Elusive Otter // Grove's Bounty"},
    {"id": "dmu:1", "name": "Fire"},
    {"id": "dmu:2", "name": "Fire // Ice"},
    {"id": "x:1", "name": "Twin // One"},
    {"id": "x:2", "name": "Twin // Two"},
    {"id": "fdn:1", "name": "Island"},
]


def test_a_front_face_name_becomes_the_full_name_with_its_count():
    text, rewritten = full_names("3 Callous Sell-Sword\n4x elusive otter\n10 Island\n", CARDS)
    assert text == "3 Callous Sell-Sword // Burn Together\n4x Elusive Otter // Grove's Bounty\n10 Island\n"
    assert rewritten == [("Callous Sell-Sword", "Callous Sell-Sword // Burn Together"),
                         ("elusive otter", "Elusive Otter // Grove's Bounty")]


def test_a_name_of_its_own_an_unclear_front_and_comments_are_kept():
    text = "1 Fire\n2 Twin\n# 3 Callous Sell-Sword\n\nSideboard\n1 Callous Sell-Sword\n"
    fixed, rewritten = full_names(text, CARDS)
    assert fixed == "1 Fire\n2 Twin\n# 3 Callous Sell-Sword\n\nSideboard\n1 Callous Sell-Sword // Burn Together\n"
    assert rewritten == [("Callous Sell-Sword", "Callous Sell-Sword // Burn Together")]


def test_the_command_writes_the_fixed_list(tmp_path, capsys):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"cards": CARDS}))
    raw = tmp_path / "raw.txt"
    raw.write_text("2 Elusive Otter\n")
    out = tmp_path / "lists" / "fixed.txt"
    assert main(["decklist-names", "--catalog", str(catalog), "--in", str(raw), "--out", str(out)]) == 0
    assert out.read_text() == "2 Elusive Otter // Grove's Bounty\n"
    assert "1 name(s) made full" in capsys.readouterr().out
