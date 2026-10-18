# SPDX-License-Identifier: Apache-2.0
"""Calibrate the accept rule: choose ``tau`` and ``delta`` for one model and one bank.

``tau`` and ``delta`` decide when a match is shown (``cardvision.recognize.accept``). They
belong to a model and its bank, so a retrained model or another game needs its own. This
package scores a grid of pairs on labeled frames and recommends one under a limit that the
user sets.

Each frame is analyzed once (``cardvision.evaluate.scoring.analyze_hover``). Every pair is then
scored on the same looks, with the accept rule, the orientation rule and the hover rule that
``cardvision evaluate hover`` applies, so a pair's figures here are the figures that command
prints for the same rows with that pair as its accept rule.

``grid``
    The values of the grid, every pair's figures, and the rule that picks one pair.
``report``
    The JSON result, the Markdown report, the grid as CSV, and ``accept.json``.
``cli``
    The ``cardvision calibrate`` subcommand.
"""

from cardvision.calibrate.grid import (
    DEFAULT_DELTA,
    DEFAULT_TAU,
    MIN_NEGATIVE_EVENTS,
    Point,
    event_counts,
    grid_rules,
    parse_values,
    recommend,
    score_grid,
    trade_off,
)
from cardvision.calibrate.report import (
    GRID_COLUMNS,
    accept_text,
    build_calibration,
    grid_csv,
    to_markdown,
    write_calibration,
)

__all__ = [
    "DEFAULT_DELTA",
    "DEFAULT_TAU",
    "GRID_COLUMNS",
    "MIN_NEGATIVE_EVENTS",
    "Point",
    "accept_text",
    "build_calibration",
    "event_counts",
    "grid_csv",
    "grid_rules",
    "parse_values",
    "recommend",
    "score_grid",
    "to_markdown",
    "trade_off",
    "write_calibration",
]
