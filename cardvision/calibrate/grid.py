# SPDX-License-Identifier: Apache-2.0
"""The grid of accept rules, its scoring, and the rule that picks one pair.

**The grid.** ``--tau`` and ``--delta`` each give a list of values (``parse_values``): numbers
separated by commas, where an item may also be a range ``START:STOP:STEP`` that includes its
stop. Every ``tau`` is paired with every ``delta``, and the pair the run started from is added
when the lists do not hold it, so that the current rule is always in the table. Each pair is an
accept rule read by ``cardvision.recognize``'s own reader, with the ``rotationRetry`` of the
rule the run started from.

**Scoring.** Each pair is scored on the same analyzed frames
(``cardvision.evaluate.scoring.HoverAnalysis``), and its figures are the ones
``cardvision.evaluate.events.rates`` gives: the counts, named right and wrong of shown, each
with its group bootstrap interval. That is the function ``cardvision evaluate hover`` reports
its overall figures from. The exact bound of a rate is left out of the grid, because it takes
long to compute for thousands of events; the report gives it for the pair the run started from
and for the recommended pair.

**The pick.** ``recommend`` takes a limit on wrong of shown and returns, among the pairs at or
under it, the pair that names the most readable faces right. Pairs that tie on that are told
apart by the lower wrong of shown, then the higher ``tau``, then the higher ``delta``: of two
pairs that do equally well here, the stricter one is kept. A pair that shows no name at all has
no wrong of shown and is never picked.

``trade_off`` lists the pairs worth choosing between: for each wrong of shown that some pair
reaches, lowest first, the pair ``recommend`` picks at that limit, kept when it names more
faces right than the row before. So ``recommend`` at any limit gives the last row of that list
that is at or under the limit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from cardvision.evaluate.events import rates
from cardvision.evaluate.scoring import HoverAnalysis
from cardvision.recognize import AcceptRule, parse_accept

DEFAULT_TAU = "0.40:0.90:0.02"
DEFAULT_DELTA = "0.00:0.10:0.01"
# Grid values are rounded to this many decimals, so that a range gives 0.62 and not 0.6200000001.
VALUE_DECIMALS = 6
MAX_PAIRS = 10000
# Under this many negative events, the report says that wrong of shown is weakly measured.
MIN_NEGATIVE_EVENTS = 30


def parse_values(text: str, option: str) -> tuple[float, ...]:
    """The values of ``--tau`` or ``--delta``: sorted, each once.

    ``text`` is numbers separated by commas; an item ``START:STOP:STEP`` stands for ``START``,
    ``START + STEP`` and so on, up to and including ``STOP`` when a step lands on it. ``option``
    names the option in the error message. Raises ``ValueError``.
    """
    values: set[float] = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            if ":" in item:
                start, stop, step = (float(part) for part in item.split(":"))
                if not (step > 0 and stop >= start and math.isfinite(stop) and math.isfinite(step)):
                    raise ValueError
                count = math.floor((stop - start) / step + 1e-9) + 1
                if count > MAX_PAIRS:
                    raise ValueError
                values.update(round(start + index * step, VALUE_DECIMALS) for index in range(count))
            else:
                values.add(round(float(item), VALUE_DECIMALS))
        except ValueError:
            raise ValueError(f"{option}: expected numbers separated by commas, each a number or a range "
                             f"START:STOP:STEP with STOP at least START and STEP above 0, got {item!r}") from None
    if not values:
        raise ValueError(f"{option}: no value given")
    return tuple(sorted(values))


def grid_rules(taus: Sequence[float], deltas: Sequence[float], start: AcceptRule) -> list[AcceptRule]:
    """Every pair of the two lists as an accept rule, and the pair of ``start``, in order of
    ``tau`` and then ``delta``. Each keeps ``start``'s ``rotation_retry``.

    Raises ``ValueError`` for a value the accept rule does not allow, and for a grid of more
    than ``MAX_PAIRS`` pairs.
    """
    pairs = {(tau, delta) for tau in taus for delta in deltas} | {(start.tau, start.delta)}
    if len(pairs) > MAX_PAIRS:
        raise ValueError(f"the grid has {len(pairs)} pairs; at most {MAX_PAIRS} are scored. Use a larger step")
    return [parse_accept({"tau": tau, "delta": delta, "rotationRetry": start.rotation_retry},
                         source=f"the pair tau {tau:g}, delta {delta:g}: accept")
            for tau, delta in sorted(pairs)]


@dataclass(frozen=True)
class Point:
    """One pair of the grid and its figures.

    Attributes:
        rule: the accept rule.
        rates: what ``cardvision.evaluate.events.rates`` gives for the events under that rule,
            without the exact bounds.
    """

    rule: AcceptRule
    rates: dict[str, Any]

    @property
    def named_right(self) -> int:
        """How many readable faces are named right."""
        return self.rates["namedRight"]["numerator"]

    @property
    def wrong_of_shown(self) -> float | None:
        """Wrong of shown as a fraction, or ``None`` when no name is shown at all."""
        return self.rates["wrongOfShown"]["value"]


def score_grid(analysis: HoverAnalysis, rules: Sequence[AcceptRule], seed: int, replicates: int) -> list[Point]:
    """The figures of every rule on the same analyzed frames. No model runs here."""
    return [Point(rule, rates(analysis.score(rule), seed, replicates, exact=False)) for rule in rules]


def _preference(point: Point) -> tuple[int, float, float, float]:
    """The order among pairs under one limit: the highest wins."""
    return (point.named_right, -point.wrong_of_shown, point.rule.tau, point.rule.delta)


def recommend(points: Sequence[Point], max_wrong: float) -> Point | None:
    """The pair to use when wrong of shown may be at most ``max_wrong`` (see the module's
    description), or ``None`` when no pair that shows a name is at or under the limit."""
    allowed = [point for point in points if point.wrong_of_shown is not None and point.wrong_of_shown <= max_wrong]
    return max(allowed, key=_preference, default=None)


def trade_off(points: Sequence[Point]) -> list[Point]:
    """The pairs worth choosing between, by rising wrong of shown (see the module's
    description). Every row names more faces right than the row before it."""
    rows: list[Point] = []
    for level in sorted({point.wrong_of_shown for point in points if point.wrong_of_shown is not None}):
        best = recommend(points, level)
        if not rows or best.named_right > rows[-1].named_right:
            rows.append(best)
    return rows


def event_counts(point: Point) -> dict[str, int]:
    """How many events of each kind the figures rest on. The same for every pair of a grid:
    the labels decide it, not the rule."""
    counts = point.rates["counts"]
    backs = counts["backShown"] + counts["backNothing"]
    off_card = counts["offCardShownNonAdjacent"] + counts["offCardShownAdjacent"] + counts["offCardNothing"]
    return {
        "events": point.rates["events"],
        "groups": point.rates["groups"],
        "readableFaces": counts["faceRight"] + counts["faceWrong"] + counts["faceNothing"],
        "backs": backs,
        "offCard": off_card,
        "negatives": backs + off_card,
    }
