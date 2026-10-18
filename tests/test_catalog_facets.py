# SPDX-License-Identifier: Apache-2.0
"""A catalog's optional facets (the labeling tool's filters after Type), and the agreement of the
three places that read them: this package (``cardvision.core.catalog``), the browser runtime
(``browser/test/facets.test.ts``) and the labeling tool (``labeler/labelcore.py`` refuses a bad
catalog at start; ``labeler/filters.js`` orders the values). The cases are the ``facets`` section
of the browser's contract file, written by ``tests/test_browser_fixtures.py``."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from cardvision.core import catalog as C

ROOT = Path(__file__).resolve().parents[1]
LABELER = ROOT / "labeler"
FIXTURE = json.loads((ROOT / "browser" / "test" / "fixtures" / "contract.json").read_text(encoding="utf-8"))["facets"]


def _labeler_module(name: str):
    sys.path.insert(0, str(LABELER))
    try:
        return __import__(name)
    finally:
        sys.path.remove(str(LABELER))


def test_the_catalog_exposes_its_facets_and_each_cards_values():
    catalog = C.parse_catalog(FIXTURE["accepted"][0]["doc"])
    assert [(f.key, f.label, f.type, f.values) for f in catalog.facets] == [
        ("color", "Color", "text", ("Red", "Blue", "Green")), ("cost", "Cost", "number", (1, 3, 10)),
        ("set", "Set", "text", ("BETA", "ALPHA")), ("rarity", "Rarity", "text", ())]
    assert catalog.card("GEN-001").facets == {"color": ("Red", "Blue"), "cost": (3,), "set": ("BETA",)}
    assert catalog.card("GEN-004").facets == {}
    assert len({catalog.card("GEN-001"), catalog.card("GEN-002")}) == 2, "a card stays hashable"


def test_a_catalog_without_facets_has_none():
    doc = FIXTURE["accepted"][1]["doc"]
    assert "facets" not in doc
    assert C.parse_catalog(doc).facets == ()


def test_validation_does_not_change_the_document():
    doc = copy.deepcopy(FIXTURE["accepted"][0]["doc"])
    C.parse_catalog(doc)
    assert doc == FIXTURE["accepted"][0]["doc"]


@pytest.mark.parametrize("path", sorted((ROOT / "games").glob("*/catalog.json")), ids=lambda path: path.parent.name)
def test_the_repositorys_catalogs_still_load(path):
    C.load_catalog(path)


def test_the_labeling_tools_toy_game_has_facets():
    toy = _labeler_module("make_toy_data")
    catalog = C.parse_catalog(toy.catalog())
    assert [f.key for f in catalog.facets] == ["color", "cost", "set"]
    assert catalog.facets[1].values == (0, 1, 2, 3, 5)
    assert catalog.card("TOY-009").facets == {"color": ("Coral",), "set": ("TOY",)}, "a site has no cost"
    assert len(catalog) == 13


@pytest.mark.parametrize("case", FIXTURE["accepted"], ids=lambda case: case["what"])
def test_the_labeling_tool_accepts_what_this_package_accepts(case):
    assert _labeler_module("labelcore").facets_problem(case["doc"]) is None


@pytest.mark.parametrize("case", FIXTURE["refused"], ids=lambda case: case["what"])
def test_the_labeling_tool_refuses_what_this_package_refuses(case):
    problem = _labeler_module("labelcore").facets_problem(case["doc"])
    assert problem is not None
    for fragment in case["mentions"]:
        assert fragment in problem, problem


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_the_labeling_tools_filter_rows_hold_the_values_in_this_packages_order():
    script = ("const F = require(process.argv[1]); const cases = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
              "process.stdout.write(JSON.stringify(cases.map((c) => F.chipRows(c.doc.cards, c.doc.facets))));")
    run = subprocess.run(["node", "-e", script, str(LABELER / "filters.js")], input=json.dumps(FIXTURE["accepted"]),
                         capture_output=True, text=True, check=True)
    for case, rows in zip(FIXTURE["accepted"], json.loads(run.stdout)):
        expected = [{"key": f["key"], "label": f["label"], "values": f["values"]} for f in case["facets"] if f["values"]]
        assert [row for row in rows if row["key"] != "kind"] == expected, case["what"]
