# SPDX-License-Identifier: Apache-2.0
"""The identity rule a catalog states, followed from the catalog to every figure.

The game is ``support_identity``'s: three different cards are named ``Amber Circle``, two of
them have an alternate printing, and ``TOY-016`` shares the name of ``TOY-015`` and, in the
toy bank, its picture. The same cards are scored under both rules:

* under ``name`` (stated, or left out) every ``Amber Circle`` is one identity. That is the
  rule for a game where a name is one card, and the wrong rule for this game: naming one card
  for another of the same name counts as right;
* under ``printing`` each card is its own identity, its alternate printing belongs to it, the
  accept rule's margin is measured against a different card of the same name, and such a card
  shown for another is wrong.

Everything is worked out by hand from the toy scene of ``test_infer_toy_models``: one table,
whose boxes the stand-in detector reports whatever the image.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from cardvision import cli
from cardvision.bank import build as BB
from cardvision.bank import release as R
from cardvision.bank.documents import canonical_bytes, read_json
from cardvision.bank.refresh import check_refresh
from cardvision.core import catalog as C
from cardvision.core.bank import IdentityGroups, load_bank, recorded_representatives
from cardvision.core.images import load_rgb
from cardvision.core.onnx import create_session, run_batched
from cardvision.data.extract import FrameStore, frame_file_name
from cardvision.data.manifest import write_manifest
from cardvision.data.schema import Box, Point, Row
from cardvision.evaluate import open_configuration, read_rows, run_hover, spec_from_paths
from cardvision.evaluate.events import row_events, score_row
from cardvision.evaluate.hover import HoverLayer, Proposal
from cardvision.evaluate.scoring import DIFFERENT_WRONG, analyze_hover, run_compare, transition, transitions
from cardvision.infer.render import card_label
from cardvision.recognize import AcceptRule, bank_transposed, similarities
from test_evaluate_browser import header as browser_header
from test_evaluate_browser import line as browser_line
from test_evaluate_browser import write_events
from support_identity import LOOKALIKE, LOOKALIKE_OF, NAMESAKES, SHARED_NAME, namesake_document, write_namesake_game
from test_infer_toy_models import (
    EXPECTED_BOXES,
    LEFT_OUT,
    SCENE_BACK,
    SCENE_SIZE,
    TOY_DELTA,
    TOY_TAU,
    ToySetup,
    build_detector,
    detector_rows,
    save_image,
)
from test_infer_toy_models import build_bank as build_toy_bank
from test_release_document import COMPONENTS, components, reseal, rewrite, write_tiny_detector

W, H = SCENE_SIZE
AMBER_CIRCLES = ["TOY-001", "TOY-001A", "TOY-007", "TOY-007A", "TOY-013"]   # five entries, three different cards


def parse(identity: str | None) -> C.Catalog:
    return C.parse_catalog(namesake_document(identity))


# --------------------------------------------------------------------------- the catalog
def test_a_catalog_states_its_rule_and_states_name_when_it_states_none():
    assert parse(None).identity == parse("name").identity == "name"
    assert parse("printing").identity == "printing"
    assert "identity" not in namesake_document(None)
    assert list(namesake_document("printing"))[:3] == ["game", "locale", "identity"]
    with pytest.raises(C.CatalogError) as refused:
        C.parse_catalog({**namesake_document(None), "identity": "art"}, source="game.json")
    assert str(refused.value) == "game.json: \"identity\" must be one of ['name', 'printing'], not 'art'"
    with pytest.raises(C.CatalogError, match=r'"identity" must be one of'):
        C.parse_catalog({**namesake_document(None), "identity": None})


def test_under_the_name_rule_the_namesakes_are_one_identity():
    for stated in (None, "name"):
        catalog = parse(stated)
        assert {catalog.identity_of(card_id) for card_id in AMBER_CIRCLES} == {SHARED_NAME}
        assert catalog.same_identity("TOY-001", "TOY-007") and catalog.same_identity("TOY-001A", "TOY-013")
        assert catalog.same_identity(LOOKALIKE, LOOKALIKE_OF)
        assert not catalog.same_identity("TOY-001", "TOY-002")
        assert catalog.identity_groups() == [
            AMBER_CIRCLES, ["TOY-003", "TOY-003A"], ["TOY-010", "TOY-P01"], ["TOY-015", "TOY-016"]]
        assert catalog.identities() == catalog.names()
        assert catalog.identity_anchors() == frozenset()


def test_under_the_printing_rule_each_card_is_its_own_identity_with_its_printings():
    catalog = parse("printing")
    # The three Amber Circles are three cards, and a printing is the card it reprints.
    assert [catalog.identity_of(card_id) for card_id in AMBER_CIRCLES] == ["TOY-001", "TOY-001", "TOY-007", "TOY-007", "TOY-013"]
    assert catalog.same_identity("TOY-001", "TOY-001A") and catalog.same_identity("TOY-007A", "TOY-007")
    assert not catalog.same_identity("TOY-001", "TOY-007") and not catalog.same_identity("TOY-001A", "TOY-007A")
    assert not catalog.same_identity("TOY-013", "TOY-001") and not catalog.same_identity(LOOKALIKE, LOOKALIKE_OF)
    assert catalog.identity_groups() == [
        ["TOY-001", "TOY-001A"], ["TOY-003", "TOY-003A"], ["TOY-007", "TOY-007A"], ["TOY-010", "TOY-P01"]]
    # Asked about some ids only: a card without its printing is in no group, and repeats count once.
    assert catalog.identity_groups(["TOY-007A", "TOY-001", "TOY-007", "TOY-007", "TOY-013"]) == [["TOY-007", "TOY-007A"]]
    assert catalog.identity_anchors() == frozenset(card.id for card in catalog if card.printing_of is None)
    assert len(catalog.identity_anchors()) == len(catalog) - 4
    # The names are untouched by the rule: a name still lists every entry that has it.
    assert catalog.ids_named(SHARED_NAME) == (*NAMESAKES, "TOY-001A", "TOY-007A")
    with pytest.raises(KeyError, match="no card 'TOY-999'"):
        catalog.identity_of("TOY-999")


def test_the_printing_rule_refuses_a_printing_of_a_printing_and_the_name_rule_keeps_it():
    document = namesake_document("printing")
    document["cards"].append({"id": "TOY-001B", "name": SHARED_NAME, "kind": "unit", "printingOf": "TOY-001A", "art": "TOY-001A.png"})
    with pytest.raises(C.CatalogError) as refused:
        C.parse_catalog(document, source="game.json")
    message = str(refused.value)
    assert message.startswith("game.json: card 'TOY-001B' (cards[22]): \"printingOf\" names 'TOY-001A', which is itself a printing of 'TOY-001'")
    assert 'under "identity": "printing"' in message
    # Under the name rule printingOf does not enter identity, and a chain loads as it always did.
    chained = C.parse_catalog({**document, "identity": "name"})
    assert chained.same_identity("TOY-001B", "TOY-013") and chained.card("TOY-001B").printing_of == "TOY-001A"
    # A printing of a card that is not there is refused under both rules, by the entry's id.
    for rule in ("name", "printing"):
        missing = namesake_document(rule)
        missing["cards"][-1]["printingOf"] = "TOY-404"
        with pytest.raises(C.CatalogError, match=r"card 'TOY-007A' \(cards\[21\]\): \"printingOf\" names 'TOY-404', which is not a card"):
            C.parse_catalog(missing)


def test_a_catalog_without_the_field_answers_as_it_did_before_the_field_existed(toy):
    """The toy game and the template state no rule. Their identities are their names, and the
    one function every consumer asks gives what comparing names gave."""
    catalog = C.load_catalog(toy[0])
    assert catalog.identity == "name" and catalog.identities() == catalog.names()
    by_name = sorted((sorted(ids) for ids in catalog.ids_by_name().values() if len(ids) > 1), key=lambda group: group[0])
    assert catalog.identity_groups() == by_name == [["TOY-003", "TOY-003A"], ["TOY-010", "TOY-P01"]]
    assert all(catalog.same_identity(a, b) == (catalog.name_of(a) == catalog.name_of(b))
               for a in catalog.ids() for b in catalog.ids())
    assert catalog.identity_description == "printings that share a catalog name are one identity"


# ---------------------------------------------------------- the groups file and the release
@pytest.fixture(scope="module")
def banks(toy, tiny_model, tmp_path_factory) -> dict:
    """For each rule: the game, a bank of it and its identity groups, as ``build-bank`` writes them."""
    built = {}
    for rule in ("name", "printing"):
        folder = tmp_path_factory.mktemp(f"groups-{rule}")
        catalog_path, art_dir = write_namesake_game(folder, rule, toy[1])
        catalog = C.load_catalog(catalog_path)
        bank = BB.build_bank(catalog, art_dir, create_session(tiny_model), model="tiny").bank
        bank_path, groups_path = BB.write_bank(bank, catalog, folder / "bank.json")
        built[rule] = {"folder": folder, "catalog": catalog, "catalog_path": catalog_path, "art": art_dir,
                       "bank": bank, "groups": read_json(groups_path)}
    return built


def test_the_groups_file_records_the_rule_and_holds_its_groups(banks):
    by_name, by_printing = banks["name"]["groups"], banks["printing"]["groups"]
    assert by_name["rule"] == {"level": "name"} and by_printing["rule"] == {"level": "printing"}
    assert by_name["groups"] == [AMBER_CIRCLES, ["TOY-003", "TOY-003A"], ["TOY-010", "TOY-P01"], ["TOY-015", "TOY-016"]]
    assert by_name["identityCount"] == 22 - 4 - 1 - 1 - 1
    assert by_printing["groups"] == [["TOY-001", "TOY-001A"], ["TOY-003", "TOY-003A"], ["TOY-007", "TOY-007A"], ["TOY-010", "TOY-P01"]]
    assert by_printing["identityCount"] == 22 - 4
    assert by_printing["representative"] == {"TOY-001A": "TOY-001", "TOY-003A": "TOY-003", "TOY-007A": "TOY-007", "TOY-P01": "TOY-010"}
    # The rows are the same bank under both rules: the rule changes which rows are one card, not a vector.
    assert banks["name"]["bank"]["entries"] == banks["printing"]["bank"]["entries"]


def test_under_printing_the_card_the_others_reprint_represents_its_group():
    """An id of the plain form wins under the name rule. Under ``printing`` the catalog says
    which entry the others are printings of, and that entry represents them."""
    document = {"game": "toy", "locale": "en", "identity": "printing", "cards": [
        {"id": "ZZZ-900X", "name": "Lamp", "printingOf": None, "art": "a.png"},
        {"id": "AAA-001", "name": "Lamp", "printingOf": "ZZZ-900X", "art": "b.png"},
        {"id": "AAA-002", "name": "Lamp", "printingOf": "ZZZ-900X", "art": "c.png"},
    ]}
    catalog = C.parse_catalog(document)
    ids = catalog.ids()
    groups = IdentityGroups(catalog.identity_groups(ids), ids, prefer=catalog.identity_anchors())
    assert groups.representative_map() == {"AAA-001": "ZZZ-900X", "AAA-002": "ZZZ-900X"}
    # A bank without that entry falls back on the smallest key among the printings it holds.
    held = ["AAA-002", "AAA-001"]
    partial = IdentityGroups(catalog.identity_groups(held), held, prefer=catalog.identity_anchors())
    assert partial.representative_map() == {"AAA-002": "AAA-001"}
    # The same entries under the name rule: the plain id represents them, as it always did.
    named = C.parse_catalog({**document, "identity": "name"})
    assert IdentityGroups(named.identity_groups(ids), ids, prefer=named.identity_anchors()).representative_map() == {
        "AAA-002": "AAA-001", "ZZZ-900X": "AAA-001"}
    # A groups document is read back with the representatives it records.
    recorded = {"groups": [sorted(ids)], "representative": groups.representative_map()}
    assert recorded_representatives(recorded) == {"ZZZ-900X"}
    assert IdentityGroups(recorded["groups"], ids, prefer=recorded_representatives(recorded)).of("AAA-001") == "ZZZ-900X"


@pytest.fixture(scope="module")
def releases(banks, tiny_model, tmp_path_factory) -> dict[str, Path]:
    """For each rule, a folder with every component of a release of the game and its ``release.json``."""
    roots = {}
    for rule, built in banks.items():
        root = tmp_path_factory.mktemp(f"release-{rule}")
        shutil.copy(built["catalog_path"], root / "catalog.json")
        write_tiny_detector(root / COMPONENTS["detector"])
        shutil.copy(tiny_model, root / COMPONENTS["recognizer"])
        BB.write_bank(built["bank"], built["catalog"], root / "bank.json")
        R.build_release(root / "release.json", **components(root))
        roots[rule] = root
    return roots


def test_a_release_records_the_rule_of_its_catalog_and_verifies(releases, capsys):
    for rule, root in releases.items():
        doc = R.verify_release(root / "release.json")
        assert doc["identity"]["level"] == rule
        assert list(doc["identity"]) == ["level", "groups", "display"]
        assert cli.main(["verify-release", str(root / "release.json")]) == 0
        assert f"release {doc['id']}: ok (toy, 22 cards in the bank, 22 in the catalog" in capsys.readouterr().out
    assert read_json(releases["name"] / "release.json")["id"] != read_json(releases["printing"] / "release.json")["id"]


def test_a_release_verifies_when_the_card_the_others_reprint_has_not_the_smallest_key(banks, tiny_model, tmp_path, capsys):
    """In the game above every card has a smaller key than its printings, so the member with
    the smallest key and the card the catalog names are the same one. Here they are not:
    ``TOY-003A`` is the card and ``TOY-003`` its printing. The groups file records the card as
    the representative, and a release is checked by the catalog's word, not by the keys."""
    document = namesake_document("printing")
    by_id = {card["id"]: card for card in document["cards"]}
    by_id["TOY-003A"]["printingOf"], by_id["TOY-003"]["printingOf"] = None, "TOY-003A"
    root = tmp_path / "release"
    root.mkdir()
    (root / "catalog.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    catalog = C.load_catalog(root / "catalog.json")
    bank = BB.build_bank(catalog, banks["printing"]["art"], create_session(tiny_model), model="tiny").bank
    _, groups_path = BB.write_bank(bank, catalog, root / "bank.json")
    recorded = read_json(groups_path)["representative"]
    assert recorded["TOY-003"] == "TOY-003A" and recorded["TOY-001A"] == "TOY-001"
    # By the keys alone the plain id would stand for the group: the two ways of choosing differ here.
    assert IdentityGroups(catalog.identity_groups(), catalog.ids()).representative_map()["TOY-003A"] == "TOY-003"
    write_tiny_detector(root / COMPONENTS["detector"])
    shutil.copy(tiny_model, root / COMPONENTS["recognizer"])
    R.build_release(root / "release.json", **components(root))
    assert R.verify_release(root / "release.json")["identity"]["level"] == "printing"
    assert cli.main(["verify-release", str(root / "release.json")]) == 0
    assert "ok (toy, 22 cards in the bank, 22 in the catalog" in capsys.readouterr().out
    # A groups file that names the smallest key instead is not this release's.
    groups = read_json(groups_path)
    groups["representative"] = {**{key: value for key, value in recorded.items() if key != "TOY-003"}, "TOY-003A": "TOY-003"}
    groups_path.write_bytes(canonical_bytes(groups))
    assert rewrite(root, lambda doc: reseal(root, "identity.groups", doc)) == [
        "identity.groups: the representative map does not match the groups"]


def test_a_release_whose_rule_is_not_its_catalog_s_is_refused(releases, tmp_path):
    printing = Path(shutil.copytree(releases["printing"], tmp_path / "printing"))
    # The document says name, the catalog and the groups file say printing.
    assert rewrite(printing, lambda doc: doc["identity"].update(level="name")) == [
        'identity.level is "name", and the catalog states the identity rule "printing": a release is used with '
        "the rule its catalog states"]
    assert rewrite(printing, lambda doc: doc["identity"].update(level="art")) == ['identity.level must be "printing"']
    assert rewrite(printing, lambda doc: doc["identity"].update(level="printing")) == []

    # The catalog is swapped for the same cards under the other rule: the level, the groups
    # file's rule and its groups all stop agreeing with it, and each is said.
    (printing / "catalog.json").write_text(json.dumps(namesake_document("name"), indent=2) + "\n", encoding="utf-8")
    assert rewrite(printing, lambda doc: reseal(printing, "catalog", doc)) == [
        'identity.level is "printing", and the catalog states the identity rule "name": a release is used with '
        "the rule its catalog states",
        "identity.groups: the file's groups are not by name",
        "identity.groups: the groups are not the ones the catalog's names give for the bank's ids"]

    # A groups file made under the other rule, in a release whose document and catalog say name.
    named = Path(shutil.copytree(releases["name"], tmp_path / "name"))
    groups = read_json(releases["printing"] / COMPONENTS["identity.groups"])
    (named / COMPONENTS["identity.groups"]).write_bytes(canonical_bytes(groups))
    assert rewrite(named, lambda doc: reseal(named, "identity.groups", doc)) == [
        "identity.groups: the file's groups are not by name",
        "identity.groups: the groups are not the ones the catalog's names give for the bank's ids"]


def test_groups_the_printing_rule_does_not_give_are_refused(releases, tmp_path):
    root = Path(shutil.copytree(releases["printing"], tmp_path / "release"))
    path = root / COMPONENTS["identity.groups"]
    groups = read_json(path)
    # Two different cards of one name, put in one group as the name rule would have it.
    groups["groups"] = [["TOY-001", "TOY-001A", "TOY-007", "TOY-007A"], ["TOY-003", "TOY-003A"], ["TOY-010", "TOY-P01"]]
    groups["representative"] = {"TOY-001A": "TOY-001", "TOY-007": "TOY-001", "TOY-007A": "TOY-001", "TOY-003A": "TOY-003",
                                "TOY-P01": "TOY-010"}
    groups["identityCount"] = 17
    path.write_bytes(canonical_bytes(groups))
    assert rewrite(root, lambda doc: reseal(root, "identity.groups", doc)) == [
        "identity.groups: the groups are not the ones the catalog's identity rule gives for the bank's ids"]


def test_a_refresh_keeps_the_rule(banks):
    """``check-refresh`` reads the two groups files. It accepts a bank compared with itself
    under either rule, and refuses groups made by another rule than the old ones."""
    def later(rule: str) -> dict:
        return {**banks[rule]["bank"], "version": 2}

    def later_groups(rule: str) -> dict:
        return {**banks[rule]["groups"], "version": 2}

    for rule in ("name", "printing"):
        assert check_refresh(banks[rule]["bank"], later(rule), banks[rule]["groups"], later_groups(rule)).ok
    report = check_refresh(banks["name"]["bank"], later("printing"), banks["name"]["groups"], later_groups("printing"))
    assert 'the old identity groups were made by the "name" rule and the new ones by the "printing" rule: a refresh ' \
           "keeps the identity rule of the catalog" in report.errors
    # The split groups are reported too: cards that were one identity must stay one.
    assert any(error.startswith(f"identity group {AMBER_CIRCLES} of the old bank is split") for error in report.errors)


def test_build_bank_says_when_the_base_bank_was_grouped_under_another_rule(banks, tiny_model, tmp_path, capsys):
    """A row does not depend on the rule, so a bank is extended under another rule than its base
    was grouped by. ``check-refresh`` will not take the result for a refresh, and one line of
    the build says so. The rule of the base is read from the groups file beside it."""
    said = ('the base bank\'s identity groups were made by the "{}" rule and this catalog states the "{}" rule: '
            "check-refresh will refuse to treat this bank as a refresh of the base bank")

    def extend(base: Path, rule: str, out: Path) -> str | None:
        """Extend ``base`` with the game's catalog under ``rule``. Returns the line after the
        one that names the groups file, or ``None`` when that is the last line."""
        code = cli.main(["build-bank", "--catalog", str(banks[rule]["catalog_path"]), "--art", str(banks[rule]["art"]),
                         "--recognizer", str(tiny_model), "--base", str(base), "--out", str(out)])
        captured = capsys.readouterr()
        assert code == 0, captured.err
        lines = captured.out.splitlines()
        named = lines.index(f"identity groups: {BB.groups_path_for(out)}")
        assert read_json(BB.groups_path_for(out))["rule"] == {"level": rule}
        return lines[named + 1] if named + 1 < len(lines) else None

    for base_rule, rule in (("name", "printing"), ("printing", "name")):
        base, out = banks[base_rule]["folder"] / "bank.json", tmp_path / f"{base_rule}-to-{rule}.json"
        assert extend(base, rule, out) == said.format(base_rule, rule)
        assert cli.main(["check-refresh", str(base), str(out)]) == 1
        assert (f'the old identity groups were made by the "{base_rule}" rule and the new ones by the "{rule}" rule: a '
                "refresh keeps the identity rule of the catalog") in capsys.readouterr().err
    # The same rule on both sides: nothing is said, and the result is a refresh.
    for rule in ("name", "printing"):
        base, out = banks[rule]["folder"] / "bank.json", tmp_path / f"{rule}-again.json"
        assert not (extend(base, rule, out) or "").startswith("the base bank")
        assert cli.main(["check-refresh", str(base), str(out)]) == 0
        capsys.readouterr()
    # A base bank with no groups file beside it, or with one that records no rule: nothing is known, and nothing is said.
    alone = tmp_path / "alone" / "bank.json"
    alone.parent.mkdir()
    shutil.copy(banks["name"]["folder"] / "bank.json", alone)
    assert not (extend(alone, "printing", tmp_path / "from-alone.json") or "").startswith("the base bank")
    BB.groups_path_for(alone).write_bytes(canonical_bytes({key: value for key, value in banks["name"]["groups"].items() if key != "rule"}))
    assert not (extend(alone, "printing", tmp_path / "from-unrecorded.json") or "").startswith("the base bank")
    BB.groups_path_for(alone).write_text("not JSON", encoding="utf-8")
    assert not (extend(alone, "printing", tmp_path / "from-unreadable.json") or "").startswith("the base bank")


# ------------------------------------------------------------- the accept rule and the scores
# The scene's boxes, by what lies in them (test_infer_toy_models.SCENE_CARDS).
AMBER, CORAL, PLUM, STRIPES = EXPECTED_BOXES[0], EXPECTED_BOXES[1], EXPECTED_BOXES[3], EXPECTED_BOXES[4]


def quad(x, y, w, h):
    return ((x / W, y / H), ((x + w) / W, y / H), ((x + w) / W, (y + h) / H), (x / W, (y + h) / H))


def face(box, card_id, label):
    return Box(kind="face", quad=quad(*box), card_id=card_id, identity="named", readable="yes", id=label)


def row(video, group, boxes, points=()):
    return Row(platform="youtube", video_id=video, t=12.517, frame_width=W, frame_height=H, source_type="digital",
               group=group, role="train", trained_on=False, credit="tests", completeness="exhaustive-boxes",
               boxes=tuple(boxes), points=tuple(points) or None)


# What the table shows is the same in both rows; the labels differ.
#   the card at AMBER is TOY-001, and the pipeline shows TOY-001 under either rule
#   the card at PLUM is TOY-015, whose bank row TOY-016's row equals: one name, two cards
ROWS = [
    row("VIDEOAAAAA1", "g-one", [
        face(AMBER, "TOY-001A", "its-printing"),       # labeled as the alternate printing of the card shown: right
        face(CORAL, "TOY-003A", "coral"),              # another printing: right
        face(PLUM, "TOY-015", "plum"),                 # right when a name is shown
        face(STRIPES, "TOY-017", "stripes"),           # right
        Box(kind="back", quad=quad(*SCENE_BACK), id="back"),
    ]),
    row("VIDEOBBBBB2", "g-two", [
        face(AMBER, "TOY-007", "namesake"),            # a different card of the same name
        face(PLUM, LOOKALIKE, "lookalike"),            # TOY-016: the other card of that name and picture
    ], [Point(x=60 / W, y=60 / H, kind="not-card", id="edge")]),   # on the box of the card at AMBER
]
FACES = 6


class Game:
    """The toy models and the labeled rows for the game under one rule."""

    def __init__(self, folder: Path, rule: str | None, toy_art: Path) -> None:
        self.rule = rule
        catalog_path, art_dir = write_namesake_game(folder / "game", rule, toy_art)
        (folder / "models").mkdir()
        self.toy = ToySetup(folder / "models", catalog_path, art_dir)
        self.manifest = folder / "labels" / "namesakes.manifest.jsonl"
        self.manifest.parent.mkdir()
        write_manifest(self.manifest, ROWS)
        self.frames = folder / "frames"
        self.frames.mkdir()
        for labeled in ROWS:
            save_image(self.frames / frame_file_name(labeled, ".png"), self.toy.frame)
        self.loose = folder / "models" / "accept-loose.json"   # the same tau, and no margin asked
        self.loose.write_text(json.dumps({"tau": TOY_TAU, "delta": 0.0, "rotationRetry": True}), encoding="utf-8")
        self.config = open_configuration(spec_from_paths(self.toy.detector, self.toy.recognizer, self.toy.bank,
                                                         self.toy.catalog, self.toy.accept))

    def arguments(self, *extra: str) -> list[str]:
        return ["--manifest", str(self.manifest), "--frames", str(self.frames), "--role", "train", *self.toy.arguments(), *extra]

    def rows(self):
        return read_rows([self.manifest], "train")


@pytest.fixture(scope="module")
def games(toy, tmp_path_factory) -> dict:
    return {rule: Game(tmp_path_factory.mktemp(f"game-{rule}"), rule, toy[1]) for rule in (None, "name", "printing")}


def faces_of(game: Game) -> dict[str, object]:
    result = game.config.pipeline.analyze(game.toy.frame)
    return {"amber": result.faces[0].recognition, "plum": result.faces[3].recognition,
            "names": [card.recognition.name for card in result.faces]}


def test_the_margin_is_taken_against_a_different_card_of_the_same_name(games):
    """The card at PLUM is TOY-015, and the bank's rows for TOY-015 and TOY-016 are equal. They
    share a name. By name the two rows are one identity and the margin looks past both; by
    printing TOY-016 is another card, the best rival, and the margin is nothing."""
    for rule in (None, "name"):
        plum = faces_of(games[rule])["plum"]
        assert (plum.top_id, plum.accepted, plum.name, plum.card_id) == ("TOY-015", True, "Plum Triangle", "TOY-015")
        assert plum.margin > TOY_DELTA
        assert faces_of(games[rule])["names"] == ["Amber Circle", "Coral Triangle", None, "Plum Triangle", "Amber Stripes", None]
    plum = faces_of(games["printing"])["plum"]
    assert (plum.top_id, plum.tau_passed, plum.accepted, plum.name) == ("TOY-015", True, False, None)
    assert plum.margin == 0.0 and plum.similarity == faces_of(games["name"])["plum"].similarity
    assert faces_of(games["printing"])["names"] == ["Amber Circle", "Coral Triangle", None, None, "Amber Stripes", None]


def test_the_margin_by_hand_under_each_rule(games):
    """For the card at AMBER, from the similarities themselves: the best row is TOY-001, and the
    margin is its score less the best row of another identity. By name every Amber Circle is
    left out of the rivals; by printing only TOY-001's own printing is."""
    for rule, same_card in ((None, AMBER_CIRCLES), ("name", AMBER_CIRCLES), ("printing", ["TOY-001", "TOY-001A"])):
        game = games[rule]
        recognizer = game.config.pipeline.recognizer
        found = faces_of(game)["amber"]
        ids, rows, _ = load_bank(game.toy.bank)
        crop = game.toy.frame[40:376, 40:280]
        scores = similarities(run_batched(recognizer.session, [recognizer.preprocess(crop)]), bank_transposed(rows))[0]
        rivals = [score for card_id, score in zip(ids, scores) if card_id not in same_card]
        assert found.top_id == "TOY-001" and found.accepted
        assert found.margin == pytest.approx(float(scores[ids.index("TOY-001")]) - max(rivals), abs=1e-12)
    # A namesake is a rival by printing and not by name, so the margin by printing is no larger.
    assert faces_of(games["printing"])["amber"].margin <= faces_of(games["name"])["amber"].margin
    # The candidates a recognition carries are of different identities: by printing they may share a name.
    by_printing = [candidate.card_id for candidate in faces_of(games["printing"])["plum"].names]
    assert by_printing[:2] == ["TOY-015", LOOKALIKE]
    assert LOOKALIKE not in [candidate.card_id for candidate in faces_of(games["name"])["plum"].names]


def test_a_same_name_different_card_answer_is_wrong_by_printing_and_right_by_name():
    """The scoring alone, on a hand-built detection: TOY-001 shown on a face labeled TOY-007."""
    labeled = row("VIDEOCCCCC3", "g-three", [face(AMBER, "TOY-007", "namesake"), face(PLUM, "TOY-001A", "its-printing")])
    shown = [Proposal(box=tuple(float(v) for v in box), detector_score=0.9, similarity=1.0, look=0, tau_passed=True,
                      accepted=True, name=SHARED_NAME, card_id="TOY-001", top_name=SHARED_NAME) for box in (AMBER, PLUM)]
    layer = HoverLayer(shown, W, H)
    outcomes = {}
    for rule in (None, "name", "printing"):
        catalog = parse(rule)
        names = catalog.names()
        scored = score_row(labeled, row_events(labeled, names), layer, names, catalog.identities())
        outcomes[rule] = {item.event.label: item.outcome for item in scored}
        assert {item.shown for item in scored} == {SHARED_NAME} and {item.shown_id for item in scored} == {"TOY-001"}
    assert outcomes[None] == outcomes["name"] == {"namesake": "right", "its-printing": "right"}
    assert outcomes["printing"] == {"namesake": "wrong", "its-printing": "right"}


def counts(found: dict) -> tuple[int, int]:
    return found["numerator"], found["denominator"]


def evaluate(game: Game, measure: str, out: Path, capsys, *extra: str) -> dict:
    code = cli.main(["evaluate", measure, *game.arguments("--out", str(out), "--bootstrap", "20", *extra)])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    result = json.loads((out / f"evaluate-{measure}.json").read_text(encoding="utf-8"))
    result["markdown"] = captured.out
    return result


def test_evaluate_hover_under_each_rule(games, capsys, tmp_path):
    for rule in (None, "name"):
        found = evaluate(games[rule], "hover", tmp_path / f"hover-{rule}", capsys)
        overall = found["results"]["overall"]
        # Every labeled face shows the labeled name, the namesake and the lookalike included,
        # and the point on the Amber Circle's box is adjacent: a face of that name is labeled there.
        assert counts(overall["namedRight"]) == (FACES, FACES) and counts(overall["wrongOfShown"]) == (0, FACES)
        assert overall["counts"]["offCardShownAdjacent"] == 1 and overall["counts"]["offCardShownNonAdjacent"] == 0
        assert "- identity: printings that share a catalog name are one identity (no identity groups file was given)" in found["markdown"]
    found = evaluate(games["printing"], "hover", tmp_path / "hover-printing", capsys)
    overall = found["results"]["overall"]
    # Right: the printing, the coral card, the stripes. Wrong: TOY-001 shown on the face labeled
    # TOY-007. Nothing: both faces at PLUM, where two cards of one name score alike.
    assert overall["counts"]["faceRight"] == 3 and overall["counts"]["faceWrong"] == 1 and overall["counts"]["faceNothing"] == 2
    # The point on the box of TOY-001: the face labeled there is TOY-007, another card, so the name is not adjacent.
    assert overall["counts"]["offCardShownAdjacent"] == 0 and overall["counts"]["offCardShownNonAdjacent"] == 1
    assert counts(overall["namedRight"]) == (3, FACES) and counts(overall["wrongOfShown"]) == (2, 5)
    assert ('- identity: a card and the entries whose "printingOf" names it are one identity, and cards that only share a '
            'name are different cards (the catalog states "identity": "printing") (no identity groups file was given)'
            ) in found["markdown"]
    # The definitions speak of names; one more line says how to read them under this rule, and only under it.
    assert found["definitions"]["identity"] == C.IDENTITY_DEFINITION and f"- {C.IDENTITY_DEFINITION}\n" in found["markdown"]
    assert "identity" not in evaluate(games["name"], "hover", tmp_path / "hover-again", capsys)["definitions"]
    events = {json.loads(line)["label"]: json.loads(line) for line in
              (tmp_path / "hover-printing" / "evaluate-hover-events.jsonl").read_text(encoding="utf-8").splitlines()}
    # The listing lets a person tell the two cards apart: same name, two ids.
    assert (events["namesake"]["outcome"], events["namesake"]["truth"], events["namesake"]["truth_id"],
            events["namesake"]["shown"], events["namesake"]["shown_id"]) == ("wrong", SHARED_NAME, "TOY-007", SHARED_NAME, "TOY-001")
    assert (events["its-printing"]["outcome"], events["its-printing"]["truth_id"], events["its-printing"]["shown_id"]) == (
        "right", "TOY-001A", "TOY-001")


def test_evaluate_hover_from_a_browser_s_outcomes_under_each_rule(games, capsys, tmp_path):
    """``evaluate hover --browser-events`` scores outcomes a browser recorded. Here the recorded
    outcomes are this package's own on the same frames, written in the form the scoring page
    writes, so the command must give the figures and the listing ``evaluate hover`` gives:
    under ``printing`` too, where a card of the labeled name is not always the labeled card."""
    events = {}
    for rule, game in games.items():
        store = FrameStore(game.frames)
        lines = [browser_line(labeled, layer=HoverLayer.of(game.config.pipeline.analyze(load_rgb(store.path(labeled))),
                                                           game.config.selection)) for labeled in ROWS]
        events[rule] = write_events(tmp_path / f"events-{rule}.jsonl", browser_header(role="train", identity=rule or "name"), *lines)
        out = tmp_path / f"browser-{rule}"
        code = cli.main(["evaluate", "hover", "--browser-events", str(events[rule]), "--manifest", str(game.manifest), "--catalog",
                         str(game.toy.catalog), "--role", "train", "--out", str(out), "--bootstrap", "20"])
        captured = capsys.readouterr()
        assert code == 0, captured.err
        browser = json.loads((out / "evaluate-hover.json").read_text(encoding="utf-8"))
        python = evaluate(game, "hover", tmp_path / f"python-{rule}", capsys)
        assert browser["results"] == python["results"] and browser["definitions"] == python["definitions"]
        assert (out / "evaluate-hover-events.jsonl").read_text() == (tmp_path / f"python-{rule}" / "evaluate-hover-events.jsonl").read_text()
        found = browser["results"]["overall"]["counts"]
        assert (found["faceRight"], found["faceWrong"], found["faceNothing"]) == ((3, 1, 2) if rule == "printing" else (FACES, 0, 0))
        assert f"- identity: {game.config.catalog.identity_description}\n" in captured.out
    # Outcomes recorded under one rule are not scored under the other: the header says which rule they are of.
    for recorded, scored_by in (("name", "printing"), ("printing", "name")):
        code = cli.main(["evaluate", "hover", "--browser-events", str(events[recorded]), "--manifest", str(games[scored_by].manifest),
                         "--catalog", str(games[scored_by].toy.catalog), "--role", "train"])
        captured = capsys.readouterr()
        assert code == 2 and captured.out == ""
        assert (f'events-{recorded}.jsonl was recorded under the identity rule "{recorded}", and catalog.json states '
                f'"{scored_by}"') in captured.err


def test_evaluate_crops_under_each_rule(games, capsys, tmp_path):
    for rule in (None, "name"):
        overall = evaluate(games[rule], "crops", tmp_path / f"crops-{rule}", capsys)["results"]["overall"]
        assert counts(overall["top1Right"]) == counts(overall["namedRight"]) == (FACES, FACES)
        assert counts(overall["namedWrong"]) == (0, FACES) and counts(overall["rejected"]) == (0, FACES)
    overall = evaluate(games["printing"], "crops", tmp_path / "crops-printing", capsys)["results"]["overall"]
    # The best row for the face labeled TOY-007 is TOY-001, and for the one labeled TOY-016 it is
    # TOY-015: each a different card of the labeled card's name.
    assert counts(overall["top1Right"]) == (4, FACES)
    assert counts(overall["namedRight"]) == (3, FACES) and counts(overall["namedWrong"]) == (1, FACES)
    assert counts(overall["rejected"]) == (2, FACES) and counts(overall["wrongOfNamed"]) == (1, 4)


def test_evaluate_detector_does_not_depend_on_the_rule(games, capsys, tmp_path):
    found = {rule: evaluate(games[rule], "detector", tmp_path / f"detector-{rule}", capsys)["results"] for rule in games}
    assert found[None] == found["name"] == found["printing"]


def test_evaluate_compare_under_each_rule(games, capsys, tmp_path):
    """The candidate asks for no margin, so the card at PLUM is named TOY-015 under every rule."""
    for rule in (None, "name"):
        table = evaluate(games[rule], "compare", tmp_path / f"compare-{rule}", capsys, "--other-accept",
                         str(games[rule].loose))["results"]["overall"]["transitions"]
        # Nothing changes: by name the card at PLUM was named already. The adjacent name counts as nothing.
        assert table["regressions"] == 0 and table["table"] == {"nothing -> nothing": 2, "right -> right": FACES}
    table = evaluate(games["printing"], "compare", tmp_path / "compare-printing", capsys, "--other-accept",
                     str(games["printing"].loose))["results"]["overall"]["transitions"]
    # The face labeled TOY-015 goes from nothing to right. The face labeled TOY-016 goes from
    # nothing to TOY-015: the same name, another card, a regression.
    assert table["table"] == {"nothing -> nothing": 1, "nothing -> right": 1, "nothing -> wrong": 1, "right -> right": 3,
                              "wrong -> wrong": 2}
    assert [(event["key"].rsplit("/", 1)[1], event["transition"], event["truth"], event["candidateShown"])
            for event in table["regressingEvents"]] == [("lookalike", "nothing -> wrong", "Plum Triangle", "Plum Triangle")]


def listing(folder: Path, name: str) -> list[dict]:
    return [json.loads(line) for line in (folder / name).read_text(encoding="utf-8").splitlines()]


def test_the_crop_listing_carries_card_ids_only_where_names_are_shared(games, capsys, tmp_path):
    """Under ``printing`` two different cards can have one name, and the listing of crops gives
    each card's id beside its name. Under ``name`` a line has the fields it always had."""
    for rule in (None, "name"):
        evaluate(games[rule], "crops", tmp_path / f"crops-{rule}", capsys)
        lines = listing(tmp_path / f"crops-{rule}", "evaluate-crops-crops.jsonl")
        assert len(lines) == FACES and all(list(line) == [
            "platform", "video_id", "t", "group", "source_type", "label", "truth", "topName", "name", "similarity", "margin",
            "look"] for line in lines)
    evaluate(games["printing"], "crops", tmp_path / "crops-printing", capsys)
    lines = {line["label"]: line for line in listing(tmp_path / "crops-printing", "evaluate-crops-crops.jsonl")}
    assert list(lines["namesake"]) == ["platform", "video_id", "t", "group", "source_type", "label", "truth", "truthId",
                                       "topName", "topId", "name", "cardId", "similarity", "margin", "look"]

    def cards(label: str) -> tuple:
        return tuple(lines[label][key] for key in ("truth", "truthId", "topName", "topId", "name", "cardId"))

    # The face labeled TOY-007 is named TOY-001: one name three times over, and the ids tell the cards apart.
    assert cards("namesake") == (SHARED_NAME, "TOY-007", SHARED_NAME, "TOY-001", SHARED_NAME, "TOY-001")
    assert cards("its-printing") == (SHARED_NAME, "TOY-001A", SHARED_NAME, "TOY-001", SHARED_NAME, "TOY-001")
    # The best row for the face labeled TOY-016 is TOY-015, and it is not named.
    assert cards("lookalike") == ("Plum Triangle", LOOKALIKE, "Plum Triangle", LOOKALIKE_OF, None, None)
    # Without the ids the lines are the ones the name rule's listing has for these fields.
    by_name = {line["label"]: line for line in listing(tmp_path / "crops-name", "evaluate-crops-crops.jsonl")}
    assert all(by_name[label][key] == lines[label][key] for label in ("namesake", "its-printing", "coral", "stripes")
               for key in ("truth", "topName", "similarity", "look"))


def test_the_regressing_events_carry_card_ids_only_where_names_are_shared(games, capsys, tmp_path):
    """``compare`` lists each regressing event with the labeled card and what each side showed.
    Under ``printing`` the ids stand beside the names, in the result and in the report's line."""
    strict = tmp_path / "accept-strict.json"   # a margin no look has: everything named becomes nothing
    strict.write_text(json.dumps({"tau": TOY_TAU, "delta": 0.99, "rotationRetry": True}), encoding="utf-8")
    for rule in (None, "name"):
        found = evaluate(games[rule], "compare", tmp_path / f"compare-{rule}", capsys, "--other-accept", str(strict))
        events = found["results"]["overall"]["transitions"]["regressingEvents"]
        assert len(events) == FACES and all(list(event) == [
            "key", "group", "source_type", "role", "truth", "transition", "baseShown", "candidateShown"] for event in events)
        namesake = next(event for event in events if event["key"].endswith("/namesake"))
        assert (f"  - `{namesake['key']}` (face, labeled Amber Circle): right -> nothing; base showed Amber Circle, "
                "candidate showed nothing\n") in found["markdown"]
    found = evaluate(games["printing"], "compare", tmp_path / "compare-printing", capsys, "--other-accept",
                     str(games["printing"].loose))
    (event,) = found["results"]["overall"]["transitions"]["regressingEvents"]
    assert list(event) == ["key", "group", "source_type", "role", "truth", "truthId", "transition", "baseShown", "baseShownId",
                           "candidateShown", "candidateShownId"]
    # The face labeled TOY-016 was not named, and is now named TOY-015: the same name, another card.
    assert (event["truth"], event["truthId"], event["baseShown"], event["baseShownId"], event["candidateShown"],
            event["candidateShownId"]) == ("Plum Triangle", LOOKALIKE, None, None, "Plum Triangle", LOOKALIKE_OF)
    assert (f"  - `{event['key']}` (face, labeled Plum Triangle {LOOKALIKE}): nothing -> wrong; base showed nothing, "
            f"candidate showed Plum Triangle {LOOKALIKE_OF}\n") in found["markdown"]
    # The strict candidate under printing: what was shown by the base carries its id, and a back has none to carry.
    found = evaluate(games["printing"], "compare", tmp_path / "compare-printing-strict", capsys, "--other-accept", str(strict))
    events = {event["key"].rsplit("/", 1)[1]: event for event in found["results"]["overall"]["transitions"]["regressingEvents"]}
    assert set(events) == {"its-printing", "coral", "stripes"}
    assert (events["its-printing"]["truthId"], events["its-printing"]["baseShownId"], events["its-printing"]["candidateShownId"]) == (
        "TOY-001A", "TOY-001", None)
    assert (f"  - `{events['its-printing']['key']}` (face, labeled Amber Circle TOY-001A): right -> nothing; base showed "
            "Amber Circle TOY-001, candidate showed nothing\n") in found["markdown"]


def test_two_different_cards_of_one_name_are_two_different_wrong_answers_by_printing():
    """The transition table has a row for an event that is wrong both times under two different
    cards. Two cards of one name are one answer by name and two by printing: what is compared
    is the identity of what was shown, not its name."""
    labeled = row("VIDEODDDDD4", "g-four", [face(AMBER, "TOY-002", "another")])

    def scored(catalog: C.Catalog, card_id: str):
        shown = Proposal(box=tuple(float(v) for v in AMBER), detector_score=0.9, similarity=1.0, look=0, tau_passed=True,
                         accepted=True, name=catalog.name_of(card_id), card_id=card_id, top_name=catalog.name_of(card_id))
        names = catalog.names()
        (found,) = score_row(labeled, row_events(labeled, names), HoverLayer([shown], W, H), names, catalog.identities())
        assert (found.outcome, found.shown, found.shown_id) == ("wrong", SHARED_NAME, card_id)
        return found

    for rule, expected in ((None, "wrong -> wrong"), ("name", "wrong -> wrong"), ("printing", DIFFERENT_WRONG)):
        catalog = parse(rule)
        pair = (scored(catalog, "TOY-001"), scored(catalog, "TOY-007"))   # the base shows one Amber Circle, the candidate another
        assert transition(*pair) == expected
        table = transitions([pair])
        assert table["table"] == {expected: 1} and table["regressions"] == 0
        # A card and its own printing are one wrong answer under every rule.
        assert transition(scored(catalog, "TOY-001"), scored(catalog, "TOY-001A")) == "wrong -> wrong"


def test_what_the_candidate_shows_is_judged_by_the_candidate_s_catalog(games, capsys, tmp_path):
    """A refresh adds a printing the base catalog does not hold: ``TOY-017A``, a printing of
    the card at STRIPES, and the candidate's bank has its row in place of the card's own. The
    candidate then shows ``TOY-017A`` on the face labeled ``TOY-017``. Only the candidate's
    catalog says which card that is, and by it the answer is right, as before."""
    for rule, game in games.items():
        folder = tmp_path / f"candidate-{rule}"
        folder.mkdir()
        document = namesake_document(rule)
        document["cards"].append({"id": "TOY-017A", "name": "Amber Stripes", "kind": "site", "printingOf": "TOY-017",
                                  "art": "TOY-017.png"})
        (folder / "catalog.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        bank = build_toy_bank(folder / "bank.json", game.toy.recognizer, folder / "catalog.json", game.toy.art_dir,
                              leave_out=[LEFT_OUT, "TOY-017"], art_of={"TOY-016": "TOY-015"})
        found = evaluate(game, "compare", folder / "out", capsys, "--other-bank", str(bank), "--other-catalog",
                         str(folder / "catalog.json"))
        events = {event["label"]: event for event in listing(folder / "out", "evaluate-compare-events.jsonl")}
        assert (events["stripes"]["shown_id"], events["stripes"]["candidate"]["shown_id"]) == ("TOY-017", "TOY-017A")
        assert (events["stripes"]["outcome"], events["stripes"]["candidate"]["outcome"]) == ("right", "right")
        table = found["results"]["overall"]["transitions"]
        assert table["regressions"] == 0 and all(name.split(" -> ")[0] == name.split(" -> ")[1] for name in table["table"])
        if rule == "printing":
            assert table["table"] == {"nothing -> nothing": 3, "right -> right": 3, "wrong -> wrong": 2}


def test_a_comparison_of_two_rules_is_refused(games):
    rows, account = games["name"].rows()
    with pytest.raises(ValueError, match='the base catalog states the identity rule "name" and the candidate\'s states "printing"'):
        run_compare(rows, FrameStore(games["name"].frames), games["name"].config, games["printing"].config, account)


def test_calibrate_scores_every_pair_as_evaluate_does_under_each_rule(games, capsys, tmp_path):
    """One frame analysis scored under many rules gives ``evaluate hover``'s events, and the
    command's figures for the pair it started from are ``evaluate hover``'s."""
    for rule, game in games.items():
        rows, account = game.rows()
        analysis = analyze_hover(rows, FrameStore(game.frames), game.config, account)
        for accept in (AcceptRule(TOY_TAU, TOY_DELTA), AcceptRule(TOY_TAU, 0.0), AcceptRule(0.5, 0.3)):
            spec = spec_from_paths(game.toy.detector, game.toy.recognizer, game.toy.bank, game.toy.catalog)
            config = open_configuration(dataclasses.replace(spec, rule=accept))
            rows_again, account_again = game.rows()
            assert analysis.score(accept) == run_hover(rows_again, FrameStore(game.frames), config, account_again)
        out = tmp_path / f"calibrate-{rule}"
        code = cli.main(["calibrate", *game.arguments("--out", str(out), "--bootstrap", "20", "--tau", "0.9:0.99:0.03",
                                                      "--delta", "0:0.02:0.02")])
        assert code == 0, capsys.readouterr().err
        capsys.readouterr()
        calibrated = json.loads((out / "calibrate.json").read_text(encoding="utf-8"))
        assert ("identity" in calibrated["definitions"]) is (rule == "printing")
        started = calibrated["startedFrom"]
        hover = evaluate(game, "hover", tmp_path / f"calibrate-hover-{rule}", capsys)["results"]["overall"]
        assert started["counts"] == hover["counts"]
        assert counts(started["namedRight"]) == ((FACES, FACES) if rule != "printing" else (3, FACES))


# --------------------------------------------- one release, read by infer and by evaluate
@pytest.fixture(scope="module")
def pinned(toy, tmp_path_factory) -> dict:
    """For each rule, a release of the game that names real boxes: the scene's detector with a
    full set of rows, a bank from ``build-bank`` and a release from ``build-release``.

    The bank is built from one view per card, the card's image itself, in place of the usual
    sixteen. The toy recognizer then gives the two cards that share a picture the same row, as
    in the toy bank of the tests above, and every outcome can be worked out by hand."""
    from test_infer_toy_models import SCENE_CARDS, build_recognizer, make_scene

    out = {}
    views, BB.VIEWS_PER_CARD = BB.VIEWS_PER_CARD, 1
    try:
        for rule in ("name", "printing"):
            out[rule] = _pin(rule, tmp_path_factory.mktemp(f"pinned-{rule}"), toy[1], SCENE_CARDS, build_recognizer, make_scene)
    finally:
        BB.VIEWS_PER_CARD = views
    return out


def _pin(rule, folder, toy_art, scene_cards, build_recognizer, make_scene) -> dict:
    catalog_path, art_dir = write_namesake_game(folder, rule, toy_art)
    frame, card_boxes = make_scene(art_dir, scene_cards)
    rows = np.zeros((R.yolox_anchor_count(640, 384), 7), dtype=np.float32)
    real = detector_rows([*card_boxes[:5], SCENE_BACK], [0.95, 0.9, 0.85, 0.8, 0.75, 0.7], [0, 0, 0, 0, 0, 1])
    rows[: len(real)] = real
    build_detector(folder / "detector.onnx", rows)
    build_recognizer(folder / "recognizer.onnx")
    (folder / "accept.json").write_text(json.dumps({"tau": TOY_TAU, "delta": TOY_DELTA, "rotationRetry": True}),
                                        encoding="utf-8")
    image = save_image(folder / "table.png", frame)
    models = ["--detector", str(folder / "detector.onnx"), "--recognizer", str(folder / "recognizer.onnx"), "--bank",
              str(folder / "bank.json"), "--catalog", str(catalog_path), "--accept", str(folder / "accept.json")]
    assert cli.main(["build-bank", "--catalog", str(catalog_path), "--art", str(art_dir), "--recognizer",
                     str(folder / "recognizer.onnx"), "--out", str(folder / "bank.json"), "--model", "toy-recognizer"]) == 0
    assert cli.main(["build-release", *models, "--out", str(folder / "release.json")]) == 0
    return {"folder": folder, "image": image, "models": models, "art": art_dir, "frame": frame}


def infer_lines(capsys, *arguments: str) -> list[dict]:
    capsys.readouterr()
    assert cli.main(["infer", *arguments]) == 0
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def test_infer_from_a_release_and_from_its_files_agree_under_each_rule(pinned, capsys, tmp_path):
    named = {}
    for rule, built in pinned.items():
        release = built["folder"] / "release.json"
        assert read_json(release)["identity"]["level"] == rule
        from_release = infer_lines(capsys, str(built["image"]), "--release", str(release))
        from_files = infer_lines(capsys, str(built["image"]), *built["models"])
        assert from_release == from_files
        (frame,) = from_release
        named[rule] = [(box["name"], box["cardId"]) for box in frame["boxes"] if box["class"] == "face"]
    # The card at PLUM is TOY-015, and the catalog files TOY-015's picture under TOY-016 as well.
    # By name the two rows are one card and it is named; by printing they are two and it is not.
    assert named["name"][0] == named["printing"][0] == (SHARED_NAME, "TOY-001")
    assert named["name"][3] == ("Plum Triangle", LOOKALIKE_OF) and named["printing"][3] == (None, None)
    assert [pair for index, pair in enumerate(named["name"]) if index != 3] == [
        pair for index, pair in enumerate(named["printing"]) if index != 3]


def test_the_pictures_infer_draws_carry_the_id_where_names_are_shared(pinned, capsys, tmp_path, monkeypatch):
    """Under ``printing`` two different cards can show one name, so the annotated frame puts the
    card id beside it. Under ``name`` the label is the name alone, as before."""
    from cardvision.infer import render
    from cardvision.infer.contact_sheet import ArtLibrary, contact_sheet
    from cardvision.infer.pipeline import Pipeline
    from cardvision.recognize import load_accept

    asked = []
    write_annotated = render.write_annotated

    def spy(directory, rgb, result, with_ids=False):
        asked.append(with_ids)
        return write_annotated(directory, rgb, result, with_ids)

    monkeypatch.setattr(render, "write_annotated", spy)
    headings = {}
    for rule, built in pinned.items():
        folder = built["folder"]
        catalog = C.load_catalog(folder / "catalog.json")
        pipeline = Pipeline.open(folder / "detector.onnx", folder / "recognizer.onnx", folder / "bank.json", catalog,
                                 load_accept(folder / "accept.json"))
        result = pipeline.analyze(built["frame"])
        first, plum, back = result.cards[0], result.cards[3], result.cards[5]
        assert card_label(first) == "Amber Circle 1.00" and card_label(first, with_id=True) == "Amber Circle TOY-001 1.00"
        assert card_label(back) == card_label(back, with_id=True) == "back 0.70"
        if rule == "printing":   # not named: the best row is given, and with its id it says which Plum Triangle
            top = plum.recognition
            assert card_label(plum) == f"? (Plum Triangle {top.similarity:.2f})"
            assert card_label(plum, with_id=True) == f"? (Plum Triangle {top.top_id} {top.similarity:.2f})"
        out = tmp_path / f"annotated-{rule}"
        infer_lines(capsys, str(built["image"]), "--release", str(folder / "release.json"), "--annotate", str(out),
                    "--contact-sheet", str(tmp_path / f"sheet-{rule}.png"), "--art", str(built["art"]))
        assert asked.pop() is (rule == "printing")
        assert render.annotate(built["frame"], result).tobytes() != render.annotate(built["frame"], result, with_ids=True).tobytes()
        assert load_rgb(out / "table.jpg").shape == built["frame"].shape and (tmp_path / f"sheet-{rule}.png").is_file()
        headings[rule] = contact_sheet([], ArtLibrary(catalog, None)).tobytes()
    # The sheet's heading over the candidates says "differently named" only where that is the rule.
    assert headings["name"] == contact_sheet([]).tobytes() != headings["printing"]


def test_evaluate_reads_the_rule_from_a_release(pinned, capsys, tmp_path):
    for rule, built in pinned.items():
        manifest = tmp_path / rule / "labels" / "namesakes.manifest.jsonl"
        manifest.parent.mkdir(parents=True)
        write_manifest(manifest, ROWS)
        frames = tmp_path / rule / "frames"
        frames.mkdir()
        for labeled in ROWS:
            save_image(frames / frame_file_name(labeled, ".png"), built["frame"])
        code = cli.main(["evaluate", "hover", "--manifest", str(manifest), "--frames", str(frames), "--role", "train",
                         "--release", str(built["folder"] / "release.json"), "--out", str(tmp_path / rule / "out"), "--bootstrap", "20"])
        captured = capsys.readouterr()
        assert code == 0, captured.err
        overall = json.loads((tmp_path / rule / "out" / "evaluate-hover.json").read_text(encoding="utf-8"))["results"]["overall"]
        # TOY-001 is shown on the face labeled TOY-007, its namesake: right by name, wrong by printing.
        assert overall["counts"]["faceWrong"] == (1 if rule == "printing" else 0)
        line = ("printings that share a catalog name are one identity" if rule == "name" else
                'a card and the entries whose "printingOf" names it are one identity, and cards that only share a name are '
                'different cards (the catalog states "identity": "printing")')
        assert f"- identity: {line}\n" in captured.out

