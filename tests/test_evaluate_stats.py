# SPDX-License-Identifier: Apache-2.0
"""The exact bound and the group bootstrap, against values worked out independently."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cardvision.evaluate.stats import (
    FEW_GROUPS,
    GroupBootstrap,
    binomial_cdf,
    bootstrap_draws,
    caution,
    exact_lower,
    exact_upper,
    rate,
)


def test_binomial_cdf_against_sums_written_out():
    # P(X <= 1) for 3 trials of 0.5: (1 + 3) / 8.
    assert binomial_cdf(1, 3, 0.5) == pytest.approx(0.5)
    # P(X <= 2) for 4 trials of 0.25: 0.75^4 + 4 * 0.25 * 0.75^3 + 6 * 0.25^2 * 0.75^2.
    assert binomial_cdf(2, 4, 0.25) == pytest.approx(0.75**4 + 4 * 0.25 * 0.75**3 + 6 * 0.0625 * 0.5625)
    assert binomial_cdf(-1, 4, 0.25) == 0.0
    assert binomial_cdf(4, 4, 0.25) == 1.0
    assert binomial_cdf(2, 4, 0.0) == 1.0 and binomial_cdf(2, 4, 1.0) == 0.0


def test_exact_upper_bound_has_two_closed_forms():
    # No event in n trials: (1 - p)^n = 0.05, so p = 1 - 0.05^(1/n).
    for n in (1, 10, 149, 199, 2000):
        assert exact_upper(0, n) == pytest.approx(1 - 0.05 ** (1 / n), abs=1e-12)
    # All but one: 1 - p^n = 0.05, so p = 0.95^(1/n).
    for n in (2, 10, 77):
        assert exact_upper(n - 1, n) == pytest.approx(0.95 ** (1 / n), abs=1e-12)


@pytest.mark.parametrize(("x", "n", "expected"), [
    # The 0.95 quantile of a beta(x + 1, n - x) distribution, computed to 30 digits elsewhere.
    (1, 10, 0.394163302436505),
    (3, 20, 0.343663804314282),
    (5, 285, 0.0365319811216745),
    (10, 500, 0.0336882723311445),
    (40, 50, 0.887278386585924),
])
def test_exact_upper_bound_against_known_values(x, n, expected):
    assert exact_upper(x, n) == pytest.approx(expected, abs=1e-10)


def test_exact_bound_edges_and_the_lower_bound():
    assert exact_upper(0, 0) is None and exact_lower(0, 0) is None
    assert exact_upper(7, 7) == 1.0
    assert exact_lower(0, 7) == 0.0
    # The 0.05 quantile of beta(x, n - x + 1).
    assert exact_lower(40, 50) == pytest.approx(0.684403938579986, abs=1e-10)
    assert exact_lower(3, 20) == pytest.approx(0.0421694078857786, abs=1e-10)
    assert exact_lower(40, 50) < 40 / 50 < exact_upper(40, 50)
    with pytest.raises(ValueError):
        exact_upper(5, 4)


def test_another_level_moves_the_bound_the_right_way():
    assert exact_upper(3, 20, 0.975) > exact_upper(3, 20, 0.95) > exact_upper(3, 20, 0.5) > 0


def test_draws_depend_only_on_seed_replicates_and_group_count():
    first = bootstrap_draws(7, 100, 12)
    assert first.shape == (100, 12) and first.min() >= 0 and first.max() <= 11
    np.testing.assert_array_equal(first, bootstrap_draws(7, 100, 12))
    np.testing.assert_array_equal(first, np.random.default_rng(7).integers(0, 12, size=(100, 12)))
    assert not np.array_equal(first, bootstrap_draws(8, 100, 12))


def _interval(per_group_numerator, per_group_denominator, seed=3, replicates=2000):
    groups = [f"g{index:02d}" for index in range(len(per_group_numerator))]
    boot = GroupBootstrap(groups, seed, replicates)
    return boot.ratio(np.array(per_group_numerator, dtype=float), np.array(per_group_denominator, dtype=float))


def test_the_bootstrap_is_deterministic_per_seed():
    numerator, denominator = [3, 0, 5, 1, 2, 0, 4, 1], [10, 8, 12, 9, 10, 7, 11, 10]
    first, again = _interval(numerator, denominator), _interval(numerator, denominator)
    assert first == again
    assert first["low"] < sum(numerator) / sum(denominator) < first["high"]
    assert first["groups"] == 8 and first["replicates"] == 2000 and first["undefinedReplicates"] == 0
    other = _interval(numerator, denominator, seed=4)
    assert (other["low"], other["high"]) != (first["low"], first["high"])


def test_the_interval_is_wider_when_the_events_sit_in_one_group():
    # Ten groups of ten events, ten wrong in all: one in each group, or all ten in one group.
    spread = _interval([1] * 10, [10] * 10)
    concentrated = _interval([10] + [0] * 9, [10] * 10)
    # Spread evenly, every replicate is exactly 10 %: resampling groups changes nothing.
    assert spread["low"] == pytest.approx(0.1) and spread["high"] == pytest.approx(0.1)
    assert concentrated["low"] == 0.0 and concentrated["high"] >= 0.3
    assert concentrated["high"] - concentrated["low"] > spread["high"] - spread["low"]
    # The exact bound cannot tell the two apart: it sees 10 of 100 either way.
    assert exact_upper(10, 100) == pytest.approx(0.163717623275814, abs=1e-10)


def test_the_bootstrap_matches_a_replicate_worked_by_hand():
    boot = GroupBootstrap(["b", "a", "c"], seed=11, replicates=5)
    assert boot.groups == ["a", "b", "c"]
    numerator = boot.per_group(["a", "b", "b", "c"], [1.0, 0.0, 1.0, 1.0])
    denominator = boot.per_group(["a", "b", "b", "c"])
    np.testing.assert_array_equal(numerator, [1, 1, 1])
    np.testing.assert_array_equal(denominator, [1, 2, 1])
    draws = np.random.default_rng(11).integers(0, 3, size=(5, 3))
    by_hand = [numerator[row].sum() / denominator[row].sum() for row in draws]
    found = boot.ratio(numerator, denominator)
    assert found["low"] == pytest.approx(np.percentile(by_hand, 2.5))
    assert found["high"] == pytest.approx(np.percentile(by_hand, 97.5))


def test_replicates_without_a_denominator_are_counted_and_left_out():
    # Only one of four groups has anything to divide by, so many replicates draw none of it.
    found = _interval([1, 0, 0, 0], [2, 0, 0, 0], replicates=500)
    draws = bootstrap_draws(3, 500, 4)
    assert found["undefinedReplicates"] == int((draws != 0).all(axis=1).sum()) > 0
    assert found["low"] == pytest.approx(0.5) and found["high"] == pytest.approx(0.5)
    nothing = _interval([0, 0], [0, 0], replicates=50)
    assert nothing["low"] is None and nothing["undefinedReplicates"] == 50 and "undefinedReason" in nothing


def test_fewer_than_two_groups_give_no_interval():
    boot = GroupBootstrap(["only"], seed=1, replicates=100)
    found = boot.ratio(np.array([3.0]), np.array([4.0]))
    assert found["low"] is None and found["high"] is None and found["groups"] == 1
    assert found["undefinedReason"] == "fewer than two recording groups"
    assert GroupBootstrap([], 1, 100).ratio(np.zeros(0), np.zeros(0))["low"] is None
    with pytest.raises(ValueError):
        GroupBootstrap(["a", "b"], 1, 0)


def test_the_paired_difference_uses_the_same_groups_for_both_ratios():
    boot = GroupBootstrap(["a", "b", "c", "d"], seed=5, replicates=400)
    ones = np.ones(4)
    a, b = np.array([0.0, 1.0, 0.0, 1.0]), np.array([1.0, 1.0, 0.0, 1.0])
    found = boot.ratio_difference(a, ones, b, ones)
    draws = bootstrap_draws(5, 400, 4)
    by_hand = b[draws].mean(axis=1) - a[draws].mean(axis=1)
    assert found["low"] == pytest.approx(np.percentile(by_hand, 2.5))
    assert found["high"] == pytest.approx(np.percentile(by_hand, 97.5))
    # A ratio compared with itself differs by exactly nothing in every replicate.
    same = boot.ratio_difference(a, ones, a, ones)
    assert same["low"] == 0.0 and same["high"] == 0.0


def test_a_rate_carries_its_counts_and_both_intervals():
    boot = GroupBootstrap(["a", "b"], seed=2, replicates=200)
    found = rate(3, 20, bound="upper", bootstrap=boot, per_group=(np.array([1.0, 2.0]), np.array([10.0, 10.0])))
    assert (found["numerator"], found["denominator"], found["value"]) == (3, 20, 0.15)
    assert found["exact95"] == {"side": "upper", "value": exact_upper(3, 20), "note": "assumes independent events"}
    assert found["groupBootstrap95"]["note"] == "resamples whole recording groups"
    lower = rate(17, 20, bound="lower")
    assert lower["exact95"]["value"] == pytest.approx(1 - exact_upper(3, 20)) and "groupBootstrap95" not in lower
    empty = rate(0, 0, bound="upper")
    assert empty["value"] is None and empty["exact95"]["value"] is None
    assert math.isclose(rate(1, 4)["value"], 0.25)


def test_a_degenerate_or_few_group_interval_is_marked_as_descriptive():
    """No error in any group gives 0 to 0 however many groups there are; under FEW_GROUPS groups
    any interval is coarse. Both are flagged, and a sound-looking interval is not."""
    groups = [f"g{i}" for i in range(8)]
    boot = GroupBootstrap(groups, seed=1, replicates=500)
    zero = boot.ratio(np.zeros(8), np.full(8, 20.0))
    assert zero["low"] == zero["high"] == 0.0 and zero["groups"] == 8
    assert "degenerate" in caution(zero) and "does not establish that the rate is exactly this" in caution(zero)
    spread = boot.ratio(np.array([0, 1, 2, 0, 3, 1, 0, 2.0]), np.full(8, 20.0))
    assert spread["low"] < spread["high"] and caution(spread) is None
    few = GroupBootstrap(groups[:FEW_GROUPS - 1], seed=1, replicates=500).ratio(
        np.array([0, 1, 2, 3.0]), np.full(4, 20.0))
    assert few["low"] < few["high"] and f"4 recording groups, fewer than {FEW_GROUPS}" in caution(few)
    enough = GroupBootstrap(groups[:FEW_GROUPS], seed=1, replicates=500).ratio(
        np.array([0, 1, 2, 3, 1.0]), np.full(5, 20.0))
    assert caution(enough) is None
    # A paired difference that is the same in every resample, and no interval at all.
    same = boot.ratio_difference(np.ones(8), np.full(8, 2.0), np.ones(8), np.full(8, 2.0))
    assert "the resamples gave one value at both ends" in caution(same, difference=True)
    assert "the difference is exactly this" in caution(same, difference=True)
    assert caution(GroupBootstrap(["only"], 1, 10).ratio(np.ones(1), np.ones(1))) is None and caution(None) is None


def test_the_report_marks_a_descriptive_interval_and_leaves_a_sound_one_alone():
    from cardvision.evaluate.report import format_rate

    boot = GroupBootstrap([f"g{i}" for i in range(8)], seed=1, replicates=500)
    zero = rate(0, 160, bound="upper", bootstrap=boot, per_group=(np.zeros(8), np.full(8, 20.0)))
    line = format_rate(zero)
    assert "group bootstrap 95 % interval 0.00 % to 0.00 % (8 groups, resamples whole recording groups). " \
           "**Read as descriptive:** the interval is degenerate" in line
    spread = rate(9, 160, bound="upper", bootstrap=boot,
                  per_group=(np.array([0, 1, 2, 0, 3, 1, 0, 2.0]), np.full(8, 20.0)))
    assert "Read as descriptive" not in format_rate(spread)
