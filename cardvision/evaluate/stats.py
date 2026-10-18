# SPDX-License-Identifier: Apache-2.0
"""The two intervals every rate is reported with, and nothing else statistical.

**The exact bound.** A one-sided 95 % Clopper-Pearson bound on a proportion ``x / n``: the
largest ``p`` (for an upper bound) at which seeing ``x`` or fewer events out of ``n`` still has
probability 5 %. It is exact for independent events and is found by bisection on the binomial
distribution, with no approximation to the normal. Events from one video are not independent
(the same table, camera and cards recur), so this bound is too narrow whenever events cluster.
It is printed as a reference and labeled "assumes independent events".

**The group bootstrap.** It accounts for dependence within recordings. Whole recording groups
are resampled with replacement: a replicate draws as many groups as there are, sums each
group's numerator and denominator, and takes the ratio of the sums. The interval is the 2.5th
to 97.5th percentile of the replicates (linear interpolation between order statistics). A
replicate whose denominator is zero has no ratio; such replicates are left out of the
percentiles and their number is reported. With fewer than two groups there is no interval.

With few groups, or sparse errors, the interval can be unstable or misleadingly narrow. When
no group holds an error, no resample can hold one, and the interval is 0 to 0: that shows no
spread, not a zero rate. ``caution`` says when an interval is to be read as descriptive:
when it is degenerate (its two ends are equal) or rests on fewer than ``FEW_GROUPS`` groups.

The draws depend only on the seed, the number of replicates and the number of groups: every
analysis starts a fresh ``numpy.random.default_rng(seed)`` and takes
``integers(0, groups, size=(replicates, groups))``, so a figure does not depend on which other
figures were computed before it. Groups are put in sorted order first.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

DEFAULT_REPLICATES = 10000
DEFAULT_SEED = 0
BOUND_LEVEL = 0.95      # one-sided level of the exact bound
INTERVAL_LEVEL = 0.95   # two-sided level of the bootstrap interval

INDEPENDENCE_NOTE = "assumes independent events"
BOOTSTRAP_NOTE = "resamples whole recording groups"
# Under this many groups an interval is flagged as descriptive. With k groups a resample is one
# of C(2k - 1, k) multisets of groups: 3 for two groups, 10 for three, 35 for four, 126 for
# five. Below five, the 2.5th and 97.5th percentiles are read off a few dozen distinct values at
# most. Five or more groups is no guarantee either: a percentile bootstrap over few clusters
# tends to run narrow, so the flag is a floor, not a line above which an interval is sound.
FEW_GROUPS = 5


# ------------------------------------------------------------------------------ exact bound
def binomial_cdf(k: int, n: int, p: float) -> float:
    """``P(X <= k)`` for ``X`` binomial with ``n`` trials of probability ``p``."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    terms = (
        math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * log_p + (n - i) * log_q)
        for i in range(k + 1)
    )
    return min(1.0, math.fsum(terms))


def exact_upper(x: int, n: int, level: float = BOUND_LEVEL) -> float | None:
    """One-sided exact (Clopper-Pearson) upper bound on a proportion with ``x`` events in ``n``.

    The ``p`` at which ``P(X <= x) = 1 - level``. ``None`` when ``n`` is 0, and 1 when every
    trial was an event.
    """
    if n < 0 or not 0 <= x <= max(n, 0):
        raise ValueError(f"expected 0 <= x <= n, got x={x} and n={n}")
    if n == 0:
        return None
    if x >= n:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(100):
        middle = (low + high) / 2
        if binomial_cdf(x, n, middle) > 1 - level:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def exact_lower(x: int, n: int, level: float = BOUND_LEVEL) -> float | None:
    """One-sided exact lower bound on a proportion: one minus the upper bound on its complement."""
    upper = exact_upper(n - x, n, level) if n else None
    return None if upper is None else 1.0 - upper


def caution(found: dict[str, Any] | None, *, difference: bool = False) -> str | None:
    """Why a bootstrap interval is to be read as descriptive, or ``None``. The report puts it
    after a bold "Read as descriptive" marker.

    A degenerate interval (both ends equal) says that the resamples gave one value at both ends,
    which is no evidence that the rate or the difference is exactly that. An interval over fewer than
    ``FEW_GROUPS`` groups is too coarse to support a reliability claim. ``difference`` words it
    for a paired difference. Nothing when there is no interval: the report says so already.
    """
    if not found or found.get("low") is None:
        return None
    what = "difference" if difference else "rate"
    if found["low"] == found["high"]:
        return (f"the interval is degenerate: the resamples gave one value at both ends. That shows no spread, "
                f"and it does not establish that the {what} is exactly this")
    if found["groups"] < FEW_GROUPS:
        return (f"{found['groups']} recording groups, fewer than {FEW_GROUPS}: the interval can be unstable "
                "or misleadingly narrow. Collect more independent recordings before making a reliability claim")
    return None


# --------------------------------------------------------------------------- group bootstrap
def bootstrap_draws(seed: int, replicates: int, groups: int) -> np.ndarray:
    """The resampling indices: ``(replicates, groups)`` integers in ``[0, groups)``."""
    return np.random.default_rng(int(seed)).integers(0, groups, size=(int(replicates), groups))


class GroupBootstrap:
    """Resampling of one fixed set of recording groups.

    ``groups`` are the group ids that take part, in any order; they are sorted, and group ``g``
    is column ``index[g]`` of every per-group array.
    """

    def __init__(self, groups: Iterable[str], seed: int = DEFAULT_SEED, replicates: int = DEFAULT_REPLICATES) -> None:
        self.groups = sorted(set(groups))
        self.index = {group: position for position, group in enumerate(self.groups)}
        self.seed = int(seed)
        self.replicates = int(replicates)
        if self.replicates < 1:
            raise ValueError(f"the bootstrap needs at least one replicate, got {replicates}")
        self.draws = bootstrap_draws(self.seed, self.replicates, len(self.groups)) if len(self.groups) >= 2 else None

    def per_group(self, groups: Sequence[str], values: Sequence[float] | None = None) -> np.ndarray:
        """Sum ``values`` (1 each when omitted) by group: one number per group, in group order."""
        sums = np.zeros(len(self.groups))
        for position, group in enumerate(groups):
            sums[self.index[group]] += 1.0 if values is None else values[position]
        return sums

    def _replicate_sums(self, per_group: np.ndarray) -> np.ndarray:
        return per_group[self.draws].sum(axis=1)

    def _interval(self, replicates: np.ndarray | None, level: float) -> dict[str, Any]:
        out: dict[str, Any] = {"low": None, "high": None, "groups": len(self.groups), "replicates": self.replicates,
                               "undefinedReplicates": 0, "note": BOOTSTRAP_NOTE}
        if replicates is None:
            out["undefinedReason"] = "fewer than two recording groups"
            return out
        undefined = int(np.isnan(replicates).sum())
        out["undefinedReplicates"] = undefined
        if undefined == len(replicates):
            out["undefinedReason"] = "the denominator is zero in every replicate"
            return out
        tail = round(100 * (1 - level) / 2, 10)  # 2.5 exactly for the default level, not 2.5000000000000022
        out["low"] = float(np.nanpercentile(replicates, tail))
        out["high"] = float(np.nanpercentile(replicates, round(100 - tail, 10)))
        return out

    def _ratios(self, numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        bottom = self._replicate_sums(denominator)
        top = self._replicate_sums(numerator)
        return np.where(bottom > 0, top / np.maximum(bottom, 1e-12), np.nan)

    def ratio(self, numerator: np.ndarray, denominator: np.ndarray, level: float = INTERVAL_LEVEL) -> dict[str, Any]:
        """The interval of ``sum(numerator) / sum(denominator)`` over resampled groups.

        Both arguments hold one number per group (see ``per_group``).
        """
        return self._interval(None if self.draws is None else self._ratios(numerator, denominator), level)

    def ratio_difference(
        self,
        numerator_a: np.ndarray,
        denominator_a: np.ndarray,
        numerator_b: np.ndarray,
        denominator_b: np.ndarray,
        level: float = INTERVAL_LEVEL,
    ) -> dict[str, Any]:
        """The interval of ``ratio b - ratio a``, both ratios taken on the same resampled groups.

        A replicate in which either ratio has a zero denominator is undefined.
        """
        if self.draws is None:
            return self._interval(None, level)
        return self._interval(self._ratios(numerator_b, denominator_b) - self._ratios(numerator_a, denominator_a), level)


def rate(
    numerator: int,
    denominator: int,
    *,
    bound: str | None = None,
    bootstrap: GroupBootstrap | None = None,
    per_group: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """One reported rate: the unreduced counts, the value, and its two intervals.

    ``bound`` is ``"upper"`` for a rate where lower is better and ``"lower"`` where higher is
    better: the exact bound is given on the unfavorable side. ``per_group`` is the numerator
    and the denominator summed by group, for the bootstrap.
    """
    out: dict[str, Any] = {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "value": numerator / denominator if denominator else None,
    }
    if bound is not None:
        value = exact_upper(numerator, denominator) if bound == "upper" else exact_lower(numerator, denominator)
        out["exact95"] = {"side": bound, "value": value, "note": INDEPENDENCE_NOTE}
    if bootstrap is not None and per_group is not None:
        out["groupBootstrap95"] = bootstrap.ratio(*per_group)
    return out
