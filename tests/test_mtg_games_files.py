# SPDX-License-Identifier: Apache-2.0
"""The files under ``games/mtg/``: the two accept rules and the sources file."""

from __future__ import annotations

from pathlib import Path

from cardvision.data.sources import load_source_config
from cardvision.recognize import load_accept

MTG = Path(__file__).resolve().parents[1] / "games" / "mtg"


def test_arena_has_the_rotation_retry_off_and_paper_on():
    arena, paper = load_accept(MTG / "accept.json"), load_accept(MTG / "accept-paper.json")
    assert (arena.tau, arena.delta) == (paper.tau, paper.delta) == (0.72, 0.07)
    assert arena.rotation_retry is False and paper.rotation_retry is True


def test_the_sources_file_adds_arena_and_produced_shows():
    sources = load_source_config(MTG / "sources.json")
    assert {"digital/mtg-arena", "physical/show", "physical/tournament", "physical/webcam"} <= set(sources.source_types)
    assert sources.hand_row_source_types == ("digital/mtg-arena",)
