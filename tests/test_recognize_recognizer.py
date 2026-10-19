# SPDX-License-Identifier: Apache-2.0
"""Looks, margins at name level, and the recognizer end to end on the toy game."""

from __future__ import annotations

import json

import numpy as np
import pytest

from cardvision.core.bank import load_bank
from cardvision.core.catalog import load_catalog
from cardvision.core.images import load_rgb
from cardvision.core.onnx import create_session
from cardvision.core.preprocess import full128
from cardvision.recognize import (
    NAMED,
    NO_OTHER_IDENTITY,
    NOTHING,
    AcceptRule,
    Look,
    Recognizer,
    bank_transposed,
    best_identities,
    best_rows,
    declared_contract,
    name_columns,
    score_look,
    similarities,
    top_and_other,
)
from test_infer_toy_models import build_bank, build_recognizer

# The toy recognizer is crude: different toy cards can resemble each other to 0.96. A strict
# rule keeps the expectations below unambiguous.
RULE = AcceptRule(tau=0.99, delta=0.02)
LEFT_OUT = "TOY-012"


# ----------------------------------------------------------------------------- looks


def test_name_columns_give_one_index_per_name():
    ids = ["A-1", "B-1", "A-1A", "C-1", "B-2"]
    names = {"A-1": "Alpha", "A-1A": "Alpha", "B-1": "Beta", "B-2": "Beta", "C-1": "Gamma", "Z-9": "Unused"}
    columns = name_columns(ids, names)
    assert columns.dtype == np.int64
    assert columns.tolist() == [0, 1, 0, 2, 1]


def test_name_columns_compare_names_exactly():
    columns = name_columns(["A", "B", "C"], {"A": "Kite", "B": "kite", "C": "Kite "})
    assert len(set(columns.tolist())) == 3


def test_name_columns_need_every_bank_card_named():
    with pytest.raises(KeyError, match="'B-1' is not in the catalog"):
        name_columns(["A-1", "B-1"], {"A-1": "Alpha"})


def test_top_and_other_by_hand():
    #              Alpha  Beta  Alpha  Gamma
    row = np.array([0.90, 0.70, 0.88, 0.75])
    columns = np.array([0, 1, 0, 2])
    assert top_and_other(row, columns) == (0, 0.90, 0.75)
    found = score_look(row, columns)
    assert found.top == 0 and found.score == 0.90
    assert found.margin == pytest.approx(0.15)        # against Gamma, not the other Alpha printing


def test_the_margin_ignores_other_printings_of_the_same_name():
    row = np.array([0.90, 0.899, 0.10])
    assert score_look(row, np.array([0, 0, 1])).margin == pytest.approx(0.80)
    assert score_look(row, np.array([0, 1, 2])).margin == pytest.approx(0.001)   # as separate cards


def test_the_first_best_row_wins_a_tie():
    row = np.array([0.5, 0.9, 0.9, 0.2])
    assert top_and_other(row, np.array([0, 1, 2, 3])) == (1, 0.9, 0.9)


def test_a_bank_of_one_identity_has_no_rival():
    top, best, other = top_and_other(np.array([0.3, 0.8]), np.array([0, 0]))
    assert (top, best, other) == (1, 0.8, NO_OTHER_IDENTITY)
    assert score_look(np.array([0.3, 0.8]), np.array([0, 0])).margin == pytest.approx(2.8)


def test_similarities_are_double_precision_dot_products():
    rng = np.random.default_rng(0)
    bank = rng.normal(size=(5, 8)).astype(np.float32)
    looks = rng.normal(size=(3, 4, 8)).astype(np.float32)
    table = similarities(looks, bank_transposed(bank))
    assert table.shape == (3, 4, 5) and table.dtype == np.float64
    expected = float(np.dot(looks[2, 1].astype(np.float64), bank[4].astype(np.float64)))
    assert table[2, 1, 4] == pytest.approx(expected, rel=1e-14)
    assert similarities(looks[0, 0], bank_transposed(bank)).shape == (5,)
    assert similarities(np.zeros((0, 8), dtype=np.float32), bank_transposed(bank)).shape == (0, 5)


def test_best_rows_and_best_identities():
    #              0     1     2     3     4
    row = np.array([0.2, 0.9, 0.8, 0.85, 0.5])
    columns = np.array([7, 3, 3, 4, 7])
    assert best_rows(row, 3) == [1, 3, 2]
    assert best_rows(row, 10) == [1, 3, 2, 4, 0]
    assert best_identities(row, columns, 3) == [1, 3, 4]      # row 2 is another printing of row 1's name
    assert best_identities(row, columns, 10) == [1, 3, 4]


# ----------------------------------------------------------------------------- the recognizer


@pytest.fixture(scope="module")
def setup(tmp_path_factory, toy):
    """The toy recognizer with a bank in which one card is missing and two rows are staged:
    TOY-P01 (a printing of TOY-010) and TOY-016 (a different name from TOY-015) each hold the
    embedding of the other card's image."""
    catalog_path, art_dir = toy
    folder = tmp_path_factory.mktemp("recognizer")
    model = build_recognizer(folder / "recognizer.onnx")
    bank = build_bank(
        folder / "bank.json", model, catalog_path, art_dir,
        leave_out=[LEFT_OUT], art_of={"TOY-P01": "TOY-010", "TOY-016": "TOY-015"},
    )
    catalog = load_catalog(catalog_path)
    return Recognizer.open(model, bank, catalog, RULE), art_dir, model, bank, catalog


def art(art_dir, card_id: str, turns: int = 0) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(load_rgb(art_dir / f"{card_id}.png"), turns))


def test_an_upright_card_is_named_at_the_first_look(setup):
    recognizer, art_dir, *_ = setup
    (found,) = recognizer.recognize([art(art_dir, "TOY-001")])
    assert (found.name, found.card_id, found.top_id, found.top_name) == ("Amber Circle", "TOY-001", "TOY-001", "Amber Circle")
    assert (found.look, found.retried, found.tau_passed, found.accepted) == (0, False, True, True)
    assert found.outcome == NAMED
    assert found.similarity > 0.9999 and found.margin > RULE.delta


@pytest.mark.parametrize(("turns", "look"), [(1, 3), (2, 2), (3, 1)])
@pytest.mark.parametrize("card_id", ["TOY-003", "TOY-008", "TOY-017"])
def test_a_turned_card_is_named_at_the_look_that_turns_it_back(setup, card_id, turns, look):
    recognizer, art_dir, _, _, catalog = setup
    (found,) = recognizer.recognize([art(art_dir, card_id, turns)])
    assert found.retried is True and found.look == look
    assert found.accepted and found.card_id == card_id and found.name == catalog.name_of(card_id)
    assert found.similarity > 0.999


def test_a_card_the_bank_does_not_hold_is_not_named(setup):
    recognizer, art_dir, *_ = setup
    (found,) = recognizer.recognize([art(art_dir, LEFT_OUT)])
    assert (found.name, found.card_id) == (None, None)
    assert found.outcome == NOTHING and found.retried is True
    assert found.tau_passed is False and found.accepted is False
    assert found.top_id != LEFT_OUT and found.top_name            # the best guess is still reported
    assert found.similarity < RULE.tau


def test_two_differently_named_rows_that_look_alike_leave_the_card_unnamed(setup):
    recognizer, art_dir, *_ = setup
    (found,) = recognizer.recognize([art(art_dir, "TOY-015")])
    assert found.top_id in ("TOY-015", "TOY-016")
    assert found.similarity > 0.9999 and abs(found.margin) < 1e-6
    assert found.tau_passed is True and found.accepted is False
    assert (found.name, found.card_id, found.outcome) == (None, None, NOTHING)
    assert found.retried is True and found.look == 0      # no turned look scored higher, so upright stays


def test_two_printings_of_one_name_that_look_alike_do_not_hurt_each_other(setup):
    recognizer, art_dir, *_ = setup
    (found,) = recognizer.recognize([art(art_dir, "TOY-010")])
    assert found.accepted and found.name == "Plum Diamond"
    assert found.card_id in ("TOY-010", "TOY-P01")         # either printing: their rows are the same
    assert found.margin > RULE.delta
    assert {row.card_id for row in found.rows[:2]} == {"TOY-010", "TOY-P01"}
    assert found.names[0].name == "Plum Diamond" and found.names[1].name != "Plum Diamond"


def test_an_alternate_printing_is_reported_by_its_own_id_under_the_shared_name(setup):
    recognizer, art_dir, *_ = setup
    (found,) = recognizer.recognize([art(art_dir, "TOY-003A")])
    assert (found.name, found.card_id) == ("Coral Triangle", "TOY-003A")


def test_candidates_are_sorted_and_names_are_distinct(setup):
    recognizer, art_dir, *_ = setup
    (found,) = recognizer.recognize([art(art_dir, "TOY-001")])
    assert len(found.rows) == 5 and len(found.names) == 3
    assert [row.similarity for row in found.rows] == sorted((row.similarity for row in found.rows), reverse=True)
    assert found.rows[0].card_id == found.top_id and found.rows[0].similarity == found.similarity
    assert len({candidate.name for candidate in found.names}) == 3
    assert found.names[0].card_id == found.top_id
    assert found.margin == pytest.approx(found.similarity - found.names[1].similarity)


def test_the_accept_rule_is_a_parameter(setup):
    _, art_dir, model, bank, catalog = setup
    crop = art(art_dir, LEFT_OUT)
    lenient = Recognizer.open(model, bank, catalog, AcceptRule(tau=0.5, delta=0.0))
    (found,) = lenient.recognize([crop])
    assert found.accepted and found.name == found.top_name and found.retried is False
    strict = Recognizer.open(model, bank, catalog, AcceptRule(tau=0.5, delta=0.9))
    assert strict.recognize([crop])[0].accepted is False


class CountingSession:
    """Passes everything to a real session and records the batch size of every run."""

    def __init__(self, session):
        self._session = session
        self.batches: list[int] = []

    def get_inputs(self):
        return self._session.get_inputs()

    def get_outputs(self):
        return self._session.get_outputs()

    def run(self, outputs, feed):
        self.batches.append(len(next(iter(feed.values()))))
        return self._session.run(outputs, feed)


def test_turned_looks_are_embedded_only_for_crops_that_need_them(setup):
    _, art_dir, model, bank, catalog = setup
    ids, rows, _ = load_bank(bank)
    session = CountingSession(create_session(model))
    recognizer = Recognizer(session, ids, rows, catalog.names(), RULE)

    upright = [art(art_dir, "TOY-001"), art(art_dir, "TOY-005"), art(art_dir, "TOY-017")]
    assert all(found.accepted and not found.retried for found in recognizer.recognize(upright))
    assert session.batches == [3]                       # one run, no turned looks

    session.batches.clear()
    mixed = [art(art_dir, "TOY-001"), art(art_dir, "TOY-003", 1), art(art_dir, "TOY-005"), art(art_dir, LEFT_OUT)]
    found = recognizer.recognize(mixed)
    assert [f.retried for f in found] == [False, True, False, True]
    assert session.batches == [4, 6]                    # four upright looks, then three turns of two crops
    assert [f.name for f in found] == ["Amber Circle", "Coral Triangle", "Plum Stripes", None]


def test_without_the_rotation_retry_no_turned_look_is_embedded(setup):
    _, art_dir, model, bank, catalog = setup
    ids, rows, _ = load_bank(bank)
    mixed = [art(art_dir, "TOY-001"), art(art_dir, "TOY-003", 1), art(art_dir, "TOY-005"), art(art_dir, LEFT_OUT)]

    session = CountingSession(create_session(model))
    with_retry = Recognizer(session, ids, rows, catalog.names(), AcceptRule(RULE.tau, RULE.delta, rotation_retry=True))
    expected = with_retry.recognize(mixed)
    assert session.batches == [4, 6] and [f.look for f in expected] == [0, 3, 0, expected[3].look]

    session = CountingSession(create_session(model))
    no_retry = Recognizer(session, ids, rows, catalog.names(), AcceptRule(RULE.tau, RULE.delta, rotation_retry=False))
    found = no_retry.recognize(mixed)
    assert session.batches == [4]                       # the upright looks only
    assert [(f.look, f.retried) for f in found] == [(0, False)] * 4
    assert [f.name for f in found] == ["Amber Circle", None, "Plum Stripes", None]   # the turned card is not named
    # The upright cards are decided exactly as they are with the retry on.
    assert (found[0], found[2]) == (expected[0], expected[2])
    # A crop that is not named still reports its best upright guess.
    assert found[1].top_id in ids and found[1].accepted is False and found[1].outcome == NOTHING


def test_results_do_not_depend_on_what_else_is_in_the_batch(setup):
    recognizer, art_dir, *_ = setup
    crops = [art(art_dir, "TOY-001"), art(art_dir, "TOY-003", 1), art(art_dir, LEFT_OUT), art(art_dir, "TOY-018", 2)]
    together = recognizer.recognize(crops)
    alone = [recognizer.recognize([crop])[0] for crop in crops]
    for a, b in zip(together, alone):
        assert (a.name, a.card_id, a.top_id, a.look, a.retried, a.accepted) == (b.name, b.card_id, b.top_id, b.look, b.retried, b.accepted)
        assert a.similarity == pytest.approx(b.similarity, abs=1e-6)


def test_the_turn_is_applied_to_the_tensor_not_to_the_crop(setup):
    """Look k of a crop is its preprocessed tensor turned k quarter turns; for a crop that is
    not square this is not the same as preprocessing the turned crop."""
    recognizer, art_dir, model, bank, _ = setup
    crop = art(art_dir, "TOY-003", 1)
    (found,) = recognizer.recognize([crop])
    ids, rows, _ = load_bank(bank)
    session = create_session(model)
    tensor = np.rot90(full128(crop), found.look, axes=(1, 2))
    embedding = session.run(None, {"crops": np.ascontiguousarray(tensor)[None]})[0][0]
    assert found.similarity == pytest.approx(float(embedding.astype(np.float64) @ rows[ids.index("TOY-003")].astype(np.float64)), abs=1e-6)


def test_no_crops_no_results(setup):
    recognizer, *_ = setup
    assert recognizer.recognize([]) == []


def test_a_bank_of_the_wrong_width_is_refused(setup):
    _, _, model, bank, catalog = setup
    ids, rows, _ = load_bank(bank)
    with pytest.raises(ValueError, match="64 values but the model's embeddings have 128"):
        Recognizer(create_session(model), ids, rows[:, :64], catalog.names())
    with pytest.raises(ValueError, match="one row per id"):
        Recognizer(create_session(model), ids[:-1], rows, catalog.names())


def test_a_model_that_does_not_declare_its_width_still_works(setup, tmp_path):
    _, art_dir, _, bank, catalog = setup
    ids, rows, _ = load_bank(bank)
    session = create_session(build_recognizer(tmp_path / "undeclared.onnx", declare_width=False))
    recognizer = Recognizer(session, ids, rows, catalog.names(), RULE)
    assert recognizer.recognize([art(art_dir, "TOY-001")])[0].name == "Amber Circle"
    # A mismatch is still caught: when the session is opened if the runtime can work the width
    # out for itself, otherwise at the first comparison.
    with pytest.raises(ValueError, match="64"):
        Recognizer(session, ids, rows[:, :64], catalog.names(), RULE).recognize([art(art_dir, "TOY-001")])


def test_embeddings_of_the_wrong_width_cannot_be_compared():
    with pytest.raises(ValueError, match="embeddings have 16 values but the bank's rows have 8"):
        similarities(np.zeros((2, 16), dtype=np.float32), bank_transposed(np.zeros((5, 8), dtype=np.float32)))


def test_a_bank_card_missing_from_the_catalog_is_refused(setup):
    _, _, model, bank, catalog = setup
    ids, rows, _ = load_bank(bank)
    names = {card_id: name for card_id, name in catalog.names().items() if card_id != "TOY-004"}
    with pytest.raises(KeyError, match="TOY-004"):
        Recognizer(create_session(model), ids, rows, names)


def test_the_banks_declared_contract_is_honored(setup, tmp_path):
    _, _, model, bank, catalog = setup
    doc = json.loads(bank.read_text())
    assert declared_contract(doc) == "full128"
    assert declared_contract({"preprocessing": {"id": "full128", "size": 128}}) == "full128"
    assert declared_contract({}) is None

    undeclared = tmp_path / "undeclared.json"
    undeclared.write_text(json.dumps({key: value for key, value in doc.items() if key != "preprocessing"}))
    Recognizer.open(model, undeclared, catalog)          # falls back to full128

    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps({**doc, "preprocessing": "something-else"}))
    with pytest.raises(ValueError, match="unknown.json declares the preprocessing contract 'something-else'"):
        Recognizer.open(model, unknown, catalog)
    Recognizer.open(model, unknown, catalog, contract="full128")   # an explicit choice overrides the file


def test_look_is_a_plain_value():
    assert Look(top=3, score=0.5, margin=0.1) == Look(3, 0.5, 0.1)


def test_an_embedding_s_similarities_do_not_depend_on_its_batch():
    """A row of one matrix product can change in its last bits with the number of rows beside
    it (BLAS picks its kernel by shape); similarities() must not, or recognize and looks, which
    embed different batches, can disagree at an exact tie."""
    from cardvision.recognize.looks import similarities
    rng = np.random.default_rng(0)
    bank_t = rng.standard_normal((128, 4000))
    x = rng.standard_normal((96, 128))
    whole = similarities(x, bank_t)
    for i in (0, 2, 50, 95):
        assert np.array_equal(similarities(x[i], bank_t), whole[i])
        for a, b in ((i, i + 1), (max(0, i - 1), min(96, i + 2)), (0, 96)):
            assert np.array_equal(similarities(x[a:b], bank_t)[i - a], whole[i])
