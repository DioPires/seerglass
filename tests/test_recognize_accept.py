# SPDX-License-Identifier: Apache-2.0
"""The accept rule and the orientation rule: the full truth table."""

from __future__ import annotations

import itertools
import json

import pytest

from cardvision.recognize import (
    DEFAULT_DELTA,
    DEFAULT_TAU,
    NAMED,
    NOTHING,
    AcceptError,
    AcceptRule,
    Look,
    accept_errors,
    choose_among,
    choose_look,
    load_accept,
    parse_accept,
)

TAU, DELTA = 0.62, 0.05
RULE = AcceptRule(TAU, DELTA)


def look(score: float, margin: float, top: int = 0) -> Look:
    return Look(top=top, score=score, margin=margin)


# Four kinds of look, by which of the two conditions they meet.
PASS = look(0.90, 0.20)          # score and margin both pass
DOUBT = look(0.90, 0.01)         # score passes, margin fails: two cards look alike
WEAK = look(0.50, 0.30)          # score fails, margin passes: resembles nothing much
NONE = look(0.30, 0.00)          # neither


def test_published_thresholds_are_the_defaults():
    assert (DEFAULT_TAU, DEFAULT_DELTA) == (0.62, 0.05)
    assert AcceptRule() == AcceptRule(0.62, 0.05) == AcceptRule(0.62, 0.05, rotation_retry=True)


# ----------------------------------------------------------------------------- one look


@pytest.mark.parametrize(
    ("score", "margin", "tau_passed", "accepted"),
    [
        (0.61, 0.04, False, False),
        (0.61, 0.05, False, False),
        (0.61, 0.90, False, False),
        (0.62, 0.04, True, False),
        (0.62, 0.05, True, True),       # both thresholds are inclusive
        (0.62, 0.90, True, True),
        (0.99, 0.04, True, False),
        (0.99, 0.05, True, True),
        (0.99, 0.90, True, True),
        (0.99, -0.10, True, False),
        (0.62, 0.049999999, True, False),
        (0.619999999, 0.05, False, False),
    ],
)
def test_one_look(score, margin, tau_passed, accepted):
    candidate = look(score, margin)
    assert RULE.passes_tau(candidate) is tau_passed
    assert RULE.accepts(candidate) is accepted


# ----------------------------------------------------------------------------- four looks


def never():
    raise AssertionError("the turned looks were asked for although the upright look was accepted")


def test_an_accepted_upright_look_wins_without_asking_for_the_others():
    choice = choose_look(PASS, never, RULE)
    assert (choice.orientation, choice.look, choice.retried) == (0, PASS, False)
    assert choice.tau_passed and choice.accepted and choice.outcome == NAMED


def test_an_accepted_upright_look_wins_even_over_a_better_turned_look():
    better = look(0.99, 0.50, top=7)
    choice = choose_among([PASS, better, better, better], RULE)
    assert (choice.orientation, choice.retried, choice.look) == (0, False, PASS)


@pytest.mark.parametrize("upright", [DOUBT, WEAK, NONE])
def test_the_turned_looks_are_asked_for_once_when_upright_is_not_accepted(upright):
    calls = []

    def rotations():
        calls.append(1)
        return [NONE, NONE, NONE]

    choice = choose_look(upright, rotations, RULE)
    assert calls == [1]
    assert choice.retried


# Each row: the four looks (upright first), then the chosen orientation and whether it names.
A = look(0.80, 0.20, top=1)
A_HIGH = look(0.95, 0.20, top=2)
A_LOW = look(0.70, 0.06, top=3)
D_HIGH = look(0.97, 0.01, top=4)   # the highest score of all, but in doubt
W_HIGH = look(0.61, 0.40, top=5)   # just under tau
W_LOW = look(0.40, 0.40, top=6)

TABLE = [
    # upright fails on the margin
    ([DOUBT, A, NONE, NONE], 1, True),
    ([DOUBT, NONE, A, NONE], 2, True),
    ([DOUBT, NONE, NONE, A], 3, True),
    ([DOUBT, A_LOW, A_HIGH, A], 2, True),            # the accepted look with the highest score
    ([DOUBT, A, A, A], 1, True),                     # equal scores: the lower orientation
    ([DOUBT, NONE, A, A], 2, True),
    ([DOUBT, D_HIGH, A_LOW, NONE], 2, True),         # an accepted look beats a higher-scoring doubtful one
    ([D_HIGH, A_LOW, NONE, NONE], 1, True),          # even when the doubtful one is the upright look
    # upright fails on the score
    ([WEAK, A, NONE, NONE], 1, True),
    ([W_HIGH, NONE, NONE, A_LOW], 3, True),
    ([NONE, A_LOW, A, A_HIGH], 3, True),
    # nothing is accepted: the highest score is reported and nothing is named
    ([DOUBT, NONE, NONE, NONE], 0, False),
    ([DOUBT, D_HIGH, NONE, NONE], 1, False),         # strictly higher: the turned look replaces upright
    ([DOUBT, DOUBT, DOUBT, DOUBT], 0, False),        # equal: upright stays
    ([WEAK, WEAK, W_HIGH, W_HIGH], 2, False),        # equal among turned looks: the first
    ([WEAK, W_LOW, W_LOW, W_LOW], 0, False),
    ([NONE, W_LOW, WEAK, W_HIGH], 3, False),
    ([NONE, NONE, NONE, NONE], 0, False),
    ([W_HIGH, DOUBT, WEAK, NONE], 1, False),         # tau passes on the chosen look, margin does not
]


@pytest.mark.parametrize(("looks", "orientation", "named"), TABLE)
def test_orientation_rule_table(looks, orientation, named):
    choice = choose_among(looks, RULE)
    assert choice.orientation == orientation
    assert choice.look is looks[orientation]
    assert choice.retried is True
    assert choice.accepted is named
    assert choice.outcome == (NAMED if named else NOTHING)
    assert choice.tau_passed is (looks[orientation].score >= TAU)


def test_every_combination_of_look_kinds():
    """All 5^4 combinations, against the rule stated in words."""
    kinds = [PASS, look(0.70, 0.07), DOUBT, WEAK, NONE]
    for looks in itertools.product(kinds, repeat=4):
        choice = choose_among(list(looks), RULE)
        accepted = [k for k in range(4) if looks[k].score >= TAU and looks[k].margin >= DELTA]
        if 0 in accepted:
            expected, retried = 0, False
        elif accepted:
            best = max(looks[k].score for k in accepted)
            expected, retried = next(k for k in accepted if looks[k].score == best), True
        else:
            best = max(candidate.score for candidate in looks)
            expected, retried = next(k for k in range(4) if looks[k].score == best), True
        assert (choice.orientation, choice.retried) == (expected, retried), looks
        assert choice.accepted is (expected in accepted), looks
        assert choice.look is looks[expected]


def test_other_thresholds_change_the_decision():
    looks = [look(0.55, 0.03), look(0.60, 0.02), NONE, NONE]
    assert choose_among(looks, AcceptRule(0.62, 0.05)).accepted is False
    assert choose_among(looks, AcceptRule(0.62, 0.05)).orientation == 1
    lenient = choose_among(looks, AcceptRule(0.50, 0.03))
    assert (lenient.orientation, lenient.retried, lenient.accepted) == (0, False, True)
    no_margin = choose_among(looks, AcceptRule(0.60, 0.0))
    assert (no_margin.orientation, no_margin.retried, no_margin.accepted) == (1, True, True)


# ----------------------------------------------------------------------------- no rotation retry


NO_RETRY = AcceptRule(TAU, DELTA, rotation_retry=False)


@pytest.mark.parametrize("upright", [PASS, DOUBT, WEAK, NONE])
def test_without_the_retry_the_turned_looks_are_never_asked_for(upright):
    choice = choose_look(upright, never, NO_RETRY)
    assert (choice.orientation, choice.look, choice.retried) == (0, upright, False)
    assert choice.accepted is RULE.accepts(upright) and choice.tau_passed is RULE.passes_tau(upright)


@pytest.mark.parametrize(("looks", "orientation", "named"), TABLE)
def test_without_the_retry_a_better_turned_look_changes_nothing(looks, orientation, named):
    """Every row of the table names a card, or reports a better guess, at a turned look or not
    at all; with the retry off the answer is the upright look, which none of them accepts."""
    with_retry, without = choose_among(looks, RULE), choose_among(looks, NO_RETRY)
    assert (with_retry.orientation, with_retry.accepted, with_retry.retried) == (orientation, named, True)
    assert (without.orientation, without.look, without.retried, without.accepted) == (0, looks[0], False, False)
    assert without.outcome == NOTHING


def test_the_wrong_number_of_looks_is_refused():
    with pytest.raises(ValueError, match="four looks"):
        choose_among([PASS, PASS, PASS], RULE)
    with pytest.raises(ValueError, match="three turned looks"):
        choose_look(NONE, lambda: [NONE, NONE], RULE)


# ----------------------------------------------------------------------------- the accept file


@pytest.mark.parametrize("retry", [True, False])
def test_accept_file_round_trip(tmp_path, retry):
    path = tmp_path / "accept.json"
    path.write_text(json.dumps({"tau": 0.7, "delta": 0.1, "rotationRetry": retry}))
    rule = load_accept(path)
    assert rule == AcceptRule(0.7, 0.1, rotation_retry=retry) and rule.rotation_retry is retry
    assert rule.to_json() == {"tau": 0.7, "delta": 0.1, "rotationRetry": retry}
    assert parse_accept(rule.to_json()) == rule


def test_whole_numbers_are_read_as_thresholds():
    assert parse_accept({"tau": 1, "delta": 0, "rotationRetry": True}) == AcceptRule(1.0, 0.0)


@pytest.mark.parametrize(
    ("doc", "message"),
    [
        ([0.62, 0.05], "accept.json must be an object with the keys"),
        ({"tau": 0.62, "rotationRetry": True}, "accept.json.delta must be a number from 0 to below 1"),
        ({"delta": 0.05, "rotationRetry": True}, "accept.json.tau must be a number above 0 and at most 1"),
        ({"tau": "0.62", "delta": 0.05, "rotationRetry": True}, "accept.json.tau must be a number"),
        ({"tau": True, "delta": 0.05, "rotationRetry": True}, "accept.json.tau must be a number"),
        ({"tau": 1.5, "delta": 0.05, "rotationRetry": True}, "accept.json.tau must be a number above 0 and at most 1"),
        ({"tau": 0, "delta": 0.05, "rotationRetry": True}, "accept.json.tau must be a number above 0 and at most 1"),
        ({"tau": 0.62, "delta": -0.01, "rotationRetry": True}, "accept.json.delta must be a number from 0 to below 1"),
        ({"tau": 0.62, "delta": 1, "rotationRetry": True}, "accept.json.delta must be a number from 0 to below 1"),
        ({"tau": 0.62, "delta": 0.05}, "accept.json.rotationRetry must be true or false"),
        ({"tau": 0.62, "delta": 0.05, "rotationRetry": 1}, "accept.json.rotationRetry must be true or false"),
        ({"tau": 0.62, "delta": 0.05, "rotationRetry": True, "rotationretry": False}, r"unknown key\(s\) \['rotationretry'\]"),
        ({"tau": 0.62, "delta": 0.05, "rotationRetry": True, "note": "calibrated elsewhere"}, r"unknown key\(s\) \['note'\]"),
    ],
)
def test_bad_accept_documents_are_refused(doc, message):
    with pytest.raises(ValueError, match=message):
        parse_accept(doc, source="accept.json")
    assert accept_errors(doc, where="accept.json") != []


def test_every_problem_of_an_accept_document_is_listed():
    with pytest.raises(AcceptError) as caught:
        parse_accept({"tau": True, "delta": "0.05"})
    assert caught.value.errors == accept_errors({"tau": True, "delta": "0.05"}, where="accept rule") == [
        "accept rule.tau must be a number above 0 and at most 1",
        "accept rule.delta must be a number from 0 to below 1",
        "accept rule.rotationRetry must be true or false",
    ]
    assert accept_errors({"tau": 0.62, "delta": 0.05, "rotationRetry": False}) == []


def test_accept_file_that_is_not_json(tmp_path):
    path = tmp_path / "accept.json"
    path.write_text("{tau: 0.62")
    with pytest.raises(ValueError, match="accept.json: not valid JSON"):
        load_accept(path)
