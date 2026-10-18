# SPDX-License-Identifier: Apache-2.0
"""Two configurations on the same events: the transition table and the paired differences."""

from __future__ import annotations

import numpy as np
import pytest

from cardvision.evaluate.events import (
    ADJACENT,
    BACK,
    FACE,
    FACE_UNREADABLE,
    NOTHING,
    OFF_CARD,
    RIGHT,
    SHOWN,
    WRONG,
    Event,
    Scored,
)
from cardvision.evaluate.scoring import (
    DIFFERENT_WRONG,
    REGRESSIONS,
    compare_results,
    paired_differences,
    reduced,
    transition,
    transitions,
)
from cardvision.evaluate.stats import bootstrap_draws


def event(index, role=FACE, group="g-a", source_type="digital"):
    return Event(platform="youtube", video_id="AAAAAAAAAAA", t=float(index), group=group, source_type=source_type,
                 kind="point", label=f"p{index}", x=0.5, y=0.5, role=role, truth="Alpha" if role == FACE else None)


def pair(index, before, after, role=FACE, group="g-a", source_type="digital", shown=("One", "Two")):
    made = event(index, role, group, source_type)
    names = [name if outcome in (WRONG, ADJACENT, SHOWN) else ("Alpha" if outcome == RIGHT else None)
             for outcome, name in zip((before, after), shown)]
    return Scored(made, before, names[0]), Scored(made, after, names[1])


def test_the_regressions_are_exactly_three_transitions():
    assert REGRESSIONS == {(RIGHT, NOTHING), (RIGHT, WRONG), (NOTHING, WRONG)}


def test_transitions_by_hand():
    pairs = [
        pair(0, RIGHT, RIGHT),
        pair(1, RIGHT, NOTHING),                              # regression
        pair(2, RIGHT, WRONG),                                # regression
        pair(3, NOTHING, WRONG),                              # regression
        pair(4, NOTHING, RIGHT),
        pair(5, WRONG, RIGHT),
        pair(6, WRONG, NOTHING),
        pair(7, WRONG, WRONG, shown=("One", "One")),          # the same wrong name
        pair(8, WRONG, WRONG, shown=("One", "Two")),          # a different wrong name: not a regression
        pair(9, NOTHING, NOTHING, role=BACK),
        pair(10, NOTHING, WRONG, role=BACK),                  # a name appears on a back: regression
        pair(11, ADJACENT, WRONG, role=OFF_CARD),             # adjacent counts as nothing: regression
        pair(12, WRONG, ADJACENT, role=OFF_CARD),             # wrong -> nothing
        pair(13, NOTHING, ADJACENT, role=OFF_CARD),           # nothing -> nothing
        pair(14, NOTHING, SHOWN, role=FACE_UNREADABLE),       # outside every rate: not in the table
    ]
    found = transitions(pairs)
    assert found["events"] == 14
    assert found["table"] == {
        "nothing -> nothing": 2, "nothing -> right": 1, "nothing -> wrong": 3,
        "right -> nothing": 1, "right -> right": 1, "right -> wrong": 1,
        DIFFERENT_WRONG: 1, "wrong -> nothing": 2, "wrong -> right": 1, "wrong -> wrong": 1,
    }
    assert found["regressions"] == 5
    assert [(item["key"].rsplit("/", 1)[1], item["transition"]) for item in found["regressingEvents"]] == [
        ("p1", "right -> nothing"), ("p2", "right -> wrong"), ("p3", "nothing -> wrong"),
        ("p10", "nothing -> wrong"), ("p11", "nothing -> wrong"),
    ]
    assert found["regressingEvents"][1]["baseShown"] == "Alpha" and found["regressingEvents"][1]["candidateShown"] == "Two"
    assert sum(found["table"].values()) == found["events"]


def test_reduction_and_single_transitions():
    before, after = pair(0, ADJACENT, NOTHING, role=OFF_CARD)
    assert reduced(before) == NOTHING and transition(before, after) == "nothing -> nothing"
    assert transition(*pair(1, WRONG, WRONG, shown=("One", "Two"))) == DIFFERENT_WRONG
    assert transition(*pair(2, WRONG, WRONG, shown=("One", "One"))) == "wrong -> wrong"


def test_paired_differences_by_hand():
    # Four groups of two faces. Base names 4 of 8 right and 2 wrong; the candidate 6 of 8 and 1 wrong.
    outcomes = [
        ("g-a", RIGHT, RIGHT), ("g-a", NOTHING, RIGHT),
        ("g-b", RIGHT, RIGHT), ("g-b", WRONG, RIGHT),
        ("g-c", RIGHT, NOTHING), ("g-c", NOTHING, RIGHT),
        ("g-d", RIGHT, RIGHT), ("g-d", WRONG, WRONG),
    ]
    pairs = [pair(index, before, after, group=group) for index, (group, before, after) in enumerate(outcomes)]
    found = paired_differences(pairs, seed=9, replicates=500)
    named, wrong = found["namedRight"], found["wrongOfShown"]
    assert (named["base"], named["candidate"], named["difference"]) == pytest.approx((4 / 8, 6 / 8, 2 / 8))
    assert (wrong["base"], wrong["candidate"]) == pytest.approx((2 / 6, 1 / 7))
    assert wrong["difference"] == pytest.approx(1 / 7 - 2 / 6)

    # The same replicates by hand: groups a, b, c, d are columns 0..3.
    draws = bootstrap_draws(9, 500, 4)
    faces = np.array([2.0, 2.0, 2.0, 2.0])
    right_a, right_b = np.array([1.0, 1.0, 1.0, 1.0]), np.array([2.0, 2.0, 1.0, 1.0])
    wrong_a, wrong_b = np.array([0.0, 1.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0, 1.0])
    by_hand = (right_b - right_a)[draws].sum(axis=1) / faces[draws].sum(axis=1)
    assert named["groupBootstrap95"]["low"] == pytest.approx(np.percentile(by_hand, 2.5))
    assert named["groupBootstrap95"]["high"] == pytest.approx(np.percentile(by_hand, 97.5))
    ratio_a = wrong_a[draws].sum(axis=1) / (right_a + wrong_a)[draws].sum(axis=1)
    ratio_b = wrong_b[draws].sum(axis=1) / (right_b + wrong_b)[draws].sum(axis=1)
    assert wrong["groupBootstrap95"]["low"] == pytest.approx(np.percentile(ratio_b - ratio_a, 2.5))
    assert wrong["groupBootstrap95"]["high"] == pytest.approx(np.percentile(ratio_b - ratio_a, 97.5))
    assert paired_differences(pairs, seed=9, replicates=500) == found


def test_identical_configurations_differ_by_nothing():
    pairs = [pair(index, outcome, outcome, group=group, shown=("One", "One"))
             for index, (group, outcome) in enumerate([("g-a", RIGHT), ("g-a", WRONG), ("g-b", RIGHT), ("g-c", NOTHING)])]
    found = compare_results(pairs, seed=0, replicates=200)["overall"]
    assert found["transitions"]["regressions"] == 0
    assert found["paired"]["namedRight"]["difference"] == 0.0
    assert found["paired"]["namedRight"]["groupBootstrap95"]["low"] == 0.0
    assert found["paired"]["namedRight"]["groupBootstrap95"]["high"] == 0.0
    assert found["base"] == found["candidate"]


def test_a_degenerate_paired_difference_is_marked_in_the_compare_report():
    """Two identical configurations differ by nothing in every resample. The report prints the
    paired interval in points and marks it as descriptive, worded for a difference."""
    from cardvision.evaluate.report import _compare_slice

    groups = [f"g-{letter}" for letter in "abcdef"]
    pairs = [pair(index, outcome, outcome, group=group, shown=("One", "One"))
             for index, (group, outcome) in enumerate((group, outcome) for group in groups
                                                      for outcome in (RIGHT, WRONG, NOTHING))]
    found = compare_results(pairs, seed=0, replicates=200)["overall"]
    assert found["paired"]["namedRight"]["groupBootstrap95"]["groups"] == 6
    (line,) = [line for line in _compare_slice(found) if line.startswith("- **named right, candidate minus base**")]
    assert "group bootstrap 95 % interval +0.00 points to +0.00 points (6 groups" in line
    assert "**Read as descriptive:** the interval is degenerate: the resamples gave one value at both ends" in line
    assert "the difference is exactly this" in line and "the rate is exactly this" not in line


def test_results_are_split_by_source():
    pairs = [pair(0, RIGHT, NOTHING, group="g-a", source_type="digital/client-a"),
             pair(1, NOTHING, RIGHT, group="g-b", source_type="physical/webcam"),
             pair(2, RIGHT, RIGHT, group="g-c", source_type="physical")]
    found = compare_results(pairs, seed=0, replicates=50)
    assert list(found["bySourceType"]) == ["digital/client-a", "physical", "physical/webcam"]
    assert list(found["bySourceFamily"]) == ["digital", "physical"]
    assert found["overall"]["transitions"]["regressions"] == 1
    assert found["bySourceFamily"]["digital"]["transitions"]["regressions"] == 1
    assert found["bySourceFamily"]["physical"]["transitions"]["table"] == {"nothing -> right": 1, "right -> right": 1}
