# SPDX-License-Identifier: Apache-2.0
"""The result of a calibration: a JSON document, a Markdown report, the grid as a table, and
the accept rule of the recommended pair.

The report is laid out like an evaluation's (``cardvision.evaluate.report``) and uses its
lines: what was scored, with every hash; what became of every row; the definitions. Then the
grid, the pairs worth choosing between, and the recommendation with what it rests on. The
figures of the recommended pair are printed as ``cardvision evaluate hover`` prints them.

``write_calibration`` writes into a folder:

``calibrate.json``
    The whole result: every pair's counts, rates and bootstrap intervals.
``calibrate.md``
    The report that is also printed.
``calibrate-grid.csv``
    One line per pair (``GRID_COLUMNS``), to sort and filter in a spreadsheet.
``accept.json``
    The recommended pair, in the format ``--accept`` reads, with the ``rotationRetry`` of the
    rule the run started from. Written only when a pair is recommended, and never over a file
    that is already there.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cardvision import __version__
from cardvision.calibrate.grid import MIN_NEGATIVE_EVENTS, Point, event_counts
from cardvision.evaluate.config import Configuration
from cardvision.evaluate.report import (
    HOVER_DEFINITIONS,
    INTERVAL_DEFINITIONS,
    configuration_lines,
    format_rate,
    hover_slice,
    identity_definition,
    input_lines,
    percent,
    titled_slices,
)
from cardvision.evaluate.rows import Account
from cardvision.recognize import AcceptRule

RESULT_KIND = "cardvision-calibration"
RESULT_VERSION = 1
RESULT_NAME = "calibrate.json"
REPORT_NAME = "calibrate.md"
GRID_NAME = "calibrate-grid.csv"
ACCEPT_NAME = "accept.json"

NEGATIVE_DEFINITION = (
    "negative events = the events on card backs and on off-card positions, where no name should be shown"
)
RECOMMENDED_DEFINITION = (
    "recommended pair = among the pairs whose wrong of shown is at or under the limit, the pair that names the "
    "most readable faces right; when pairs tie on that, the lower wrong of shown, then the higher tau, then the "
    "higher delta; a pair that shows no name at all is never recommended"
)
TRADE_OFF_DEFINITION = (
    "trade-off = for each wrong of shown that some pair reaches, lowest first, the pair recommended at that "
    "limit, listed when it names more readable faces right than the row before"
)
FRAGILE_NOTE = (
    "It is the best pair on these rows, and that is a measurement, not a guarantee. Other footage will give "
    "other figures, and with few events one event more or less changes which pair is picked."
)
EVALUATION_ROLE_WARNING = (
    "these are the evaluation rows (--role evaluation). A rule chosen on them fits them, so any figure scored on "
    "these rows afterwards is optimistic. Calibrate on other rows and keep these for the final score"
)

_COUNT_COLUMNS = ("faceRight", "faceWrong", "faceNothing", "backShown", "backNothing", "offCardShownNonAdjacent",
                  "offCardShownAdjacent", "offCardNothing", "faceUnreadableShown", "faceUnreadableNothing",
                  "faceUnnamedShown", "faceUnnamedNothing", "unreadableShown", "unreadableNothing")
GRID_COLUMNS = ("tau", "delta", "rotationRetry", "namedRight", "readableFaces", "namedRightRate", "wrongShown",
                "shown", "wrongOfShownRate", *_COUNT_COLUMNS, "namedRightBootstrapLow", "namedRightBootstrapHigh",
                "wrongOfShownBootstrapLow", "wrongOfShownBootstrapHigh", "tradeOff", "recommended", "startedFrom")


def few_negatives_warning(negatives: int) -> str | None:
    """What to say when the rows hold few negative events, or ``None`` when they hold enough."""
    if negatives >= MIN_NEGATIVE_EVENTS:
        return None
    return (f"wrong of shown is weakly measured: the rows hold {negatives} negative event(s) (card backs and "
            f"positions with no card), fewer than {MIN_NEGATIVE_EVENTS}. A name shown where there is no card can "
            "only be counted on such events")


def _pair(rule: AcceptRule) -> dict[str, float]:
    return {"tau": rule.tau, "delta": rule.delta}


def build_calibration(
    config: Configuration,
    account: Account,
    points: Sequence[Point],
    trade_off: Sequence[Point],
    started: Mapping[str, Any],
    *,
    taus: Sequence[float],
    deltas: Sequence[float],
    max_wrong: float | None,
    recommended: Point | None,
    recommended_results: Mapping[str, Any] | None,
    warnings: Sequence[str],
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    """The JSON document of one calibration.

    ``points`` are every pair's figures, ``trade_off`` the pairs worth choosing between and
    ``started`` the figures of the pair the run started from, with their exact bounds
    (``cardvision.evaluate.events.rates``). ``recommended`` is the pair picked under
    ``max_wrong`` (``None`` without a limit, or when no pair is under it) and
    ``recommended_results`` that pair's figures overall and by source, as
    ``cardvision.evaluate.scoring.hover_results`` gives them.
    """
    start = config.pipeline.recognizer.rule
    return {
        "kind": RESULT_KIND,
        "version": RESULT_VERSION,
        "package": {"name": "cardvision", "version": __version__},
        "configuration": config.describe(),
        "inputs": account.to_json(),
        "bootstrap": {"seed": seed, "replicates": replicates, "unit": "recording group",
                      "interval": "2.5th to 97.5th percentile"},
        "definitions": {**HOVER_DEFINITIONS, **identity_definition(config), "negativeEvents": NEGATIVE_DEFINITION,
                        "tradeOff": TRADE_OFF_DEFINITION, "recommended": RECOMMENDED_DEFINITION, **INTERVAL_DEFINITIONS},
        "grid": {"tau": list(taus), "delta": list(deltas), "rotationRetry": start.rotation_retry,
                 "startedFrom": _pair(start), "pairs": len(points),
                 "pairsShowingNothing": sum(point.wrong_of_shown is None for point in points)},
        "startedFrom": {**_pair(start), **started},
        "events": event_counts(points[0]),
        "points": [{**_pair(point.rule), **point.rates} for point in points],
        "tradeOff": [_pair(point.rule) for point in trade_off],
        "maxWrong": max_wrong,
        "recommended": None if recommended is None else {**recommended.rule.to_json(), "results": recommended_results},
        "warnings": list(warnings),
    }


# ----------------------------------------------------------------------------------- Markdown
def _span(values: Sequence[float]) -> str:
    return f"{len(values)} values from {values[0]:g} to {values[-1]:g}" if len(values) > 1 else f"1 value, {values[0]:g}"


def _retry(on: bool) -> str:
    return "the turned looks retried" if on else "the turned looks not tried (upright looks only)"


def _point(result: Mapping[str, Any], pair: Mapping[str, float]) -> Mapping[str, Any]:
    return next(point for point in result["points"] if (point["tau"], point["delta"]) == (pair["tau"], pair["delta"]))


def _short_rate(found: Mapping[str, Any]) -> str:
    return f"{found['numerator']} / {found['denominator']} = {percent(found['value'])}"


def to_markdown(result: Mapping[str, Any]) -> str:
    """The Markdown report of a calibration document."""
    grid, events, boot = result["grid"], result["events"], result["bootstrap"]
    start, retry = grid["startedFrom"], grid["rotationRetry"]
    lines = ["# Calibration: the accept rule on hover events", "",
             f"cardvision {result['package']['version']}; result format {result['version']}.", "",
             "## What was scored", ""]
    lines += configuration_lines("Configuration", result["configuration"])
    lines += ["The accept rule above is the one the run started from. The grid replaces its tau and delta, and "
              "keeps its setting for the turned looks.", ""]
    lines += input_lines(result["inputs"])
    lines += ["## Definitions", ""]
    lines += [f"- {text}" for text in result["definitions"].values()]
    lines += [f"- bootstrap: {boot['replicates']} replicates, seed {boot['seed']}", ""]

    lines += [
        "## The grid", "",
        f"- tau: {_span(grid['tau'])}",
        f"- delta: {_span(grid['delta'])}",
        f"- pairs scored: {grid['pairs']}, each with {_retry(retry)}. The pair the run started from, tau "
        f"{start['tau']:g} and delta {start['delta']:g}, is one of them",
        "- every frame was analyzed once. Every pair was then scored on the same looks, with the accept rule and "
        "the hover rule that `cardvision evaluate hover` applies",
        f"- events: {events['events']} in {events['groups']} recording groups: {events['readableFaces']} readable "
        f"faces and {events['negatives']} negative events ({events['backs']} card backs, {events['offCard']} "
        "off-card positions)",
        f"- pairs that show no name at all: {grid['pairsShowingNothing']}",
        "",
    ]
    lines += [f"**Warning:** {warning}." for warning in result["warnings"]]
    lines += [""] if result["warnings"] else []

    began = result["startedFrom"]
    lines += ["### The pair the run started from", "",
              f"- tau {start['tau']:g}, delta {start['delta']:g}",
              f"- **named right**: {format_rate(began['namedRight'])}",
              f"- **wrong of shown**: {format_rate(began['wrongOfShown'])}", ""]

    lines += ["### The trade-off", ""]
    if result["tradeOff"]:
        lines += ["The pairs worth choosing between. Each row names more readable faces right than the row above "
                  "it, at a higher wrong of shown. `--max-wrong` picks the last row that is at or under its "
                  "limit. Where several pairs have a row's figures, the row shows the one with the highest tau, "
                  f"then the highest delta. `{GRID_NAME}`, written with `--out`, has every pair.", "",
                  "| tau | delta | Named right | Wrong of shown |", "| --- | --- | --- | --- |"]
        for pair in result["tradeOff"]:
            point = _point(result, pair)
            lines.append(f"| {pair['tau']:g} | {pair['delta']:g} | {_short_rate(point['namedRight'])} | "
                         f"{_short_rate(point['wrongOfShown'])} |")
    else:
        lines += ["No pair of the grid shows any name on these rows, so there is nothing to choose between."]
    lines += ["", "## Recommendation", ""]

    limit, chosen = result["maxWrong"], result["recommended"]
    if limit is None:
        lines += ["No pair is recommended, because no limit was given. The limit is yours to set: look at the "
                  "trade-off above, decide the highest wrong of shown you accept, and run again with "
                  f"`--max-wrong RATE` (a fraction: 0.02 is 2 %). `{ACCEPT_NAME}` is written then, with `--out`."]
    elif chosen is None:
        reached = [_point(result, pair) for pair in result["tradeOff"][:1]]
        lines += [f"- limit: wrong of shown at or under {percent(limit)} (`--max-wrong {limit:g}`)",
                  "- **no pair is recommended**: no pair of the grid that shows a name keeps wrong of shown at or "
                  "under the limit"
                  + (f". The lowest that any pair reaches is {_short_rate(reached[0]['wrongOfShown'])}"
                     if reached else "")]
    else:
        lines += [
            f"- limit: wrong of shown at or under {percent(limit)} (`--max-wrong {limit:g}`)",
            f"- **recommended**: tau {chosen['tau']:g}, delta {chosen['delta']:g}, {_retry(chosen['rotationRetry'])}",
            f"- it rests on {events['events']} events in {events['groups']} recording groups: "
            f"{events['readableFaces']} readable faces and {events['negatives']} negative events",
            f"- {FRAGILE_NOTE}",
            "- These rows chose the pair, so their figures flatter it. Score your evaluation rows once with it, "
            "and report those figures.",
            "",
        ]
        for title, part in titled_slices(chosen["results"]):
            lines += [f"### {title}, with the recommended pair", "", *hover_slice(part), ""]
    return "\n".join(lines).rstrip() + "\n"


# -------------------------------------------------------------------------------------- files
def grid_csv(result: Mapping[str, Any]) -> str:
    """The grid as CSV text: one line per pair, under ``GRID_COLUMNS``. A rate with a zero
    denominator, and an interval that could not be computed, are left empty."""
    def mark(pair: Mapping[str, float], pairs: Sequence[Mapping[str, float]]) -> str:
        return "yes" if any((pair["tau"], pair["delta"]) == (other["tau"], other["delta"]) for other in pairs) else ""

    def blank(value: float | None) -> Any:
        return "" if value is None else value

    grid, chosen = result["grid"], result["recommended"]
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(GRID_COLUMNS)
    for point in result["points"]:
        named, wrong = point["namedRight"], point["wrongOfShown"]
        writer.writerow([
            point["tau"], point["delta"], "true" if grid["rotationRetry"] else "false",
            named["numerator"], named["denominator"], blank(named["value"]),
            wrong["numerator"], wrong["denominator"], blank(wrong["value"]),
            *(point["counts"][name] for name in _COUNT_COLUMNS),
            blank(named["groupBootstrap95"]["low"]), blank(named["groupBootstrap95"]["high"]),
            blank(wrong["groupBootstrap95"]["low"]), blank(wrong["groupBootstrap95"]["high"]),
            mark(point, result["tradeOff"]), mark(point, [chosen] if chosen else []), mark(point, [grid["startedFrom"]]),
        ])
    return out.getvalue()


def accept_text(result: Mapping[str, Any]) -> str | None:
    """The recommended pair as an accept-rule file, or ``None`` when no pair is recommended."""
    chosen = result["recommended"]
    if chosen is None:
        return None
    return json.dumps({key: chosen[key] for key in ("tau", "delta", "rotationRetry")}, indent=1) + "\n"


def refuse_existing_accept(out: str | Path) -> None:
    """Raise ``FileExistsError`` when the folder already holds an accept rule.

    A calibration never replaces an accept rule: not a game's, not a release's, not an
    earlier calibration's. It is checked before any model runs.
    """
    path = Path(out) / ACCEPT_NAME
    if path.exists():
        raise FileExistsError(f"{path} already exists, and calibrate never replaces an accept rule. "
                              "Give --out a folder that holds none")


def write_calibration(out: str | Path, result: Mapping[str, Any]) -> list[Path]:
    """Write the result's files into a folder (see the module's description). Returns the paths
    written. The report, the result and the grid replace earlier ones; the accept rule does not."""
    folder = Path(out)
    refuse_existing_accept(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = [folder / RESULT_NAME, folder / REPORT_NAME, folder / GRID_NAME]
    written[0].write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    written[1].write_text(to_markdown(result), encoding="utf-8")
    written[2].write_text(grid_csv(result), encoding="utf-8")
    accept = accept_text(result)
    if accept is not None:
        (folder / ACCEPT_NAME).write_text(accept, encoding="utf-8")
        written.append(folder / ACCEPT_NAME)
    return written
