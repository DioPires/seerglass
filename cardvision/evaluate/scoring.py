# SPDX-License-Identifier: Apache-2.0
"""Run the hover measure over rows and frames, for one configuration or for two.

**One configuration** (``run_hover``, ``hover_results``): every usable row's frame goes through
the pipeline once, its hover layer is built, and each of the row's events is resolved and
classified (``cardvision.evaluate.events``).

**One configuration under many accept rules** (``analyze_hover``, ``HoverAnalysis``): the same
rows and events, with each frame analyzed once up to the accept rule
(``Pipeline.analyze_looks``). ``HoverAnalysis.score(rule)`` then gives the events ``run_hover``
gives for a configuration that holds that rule, without running a model again:
``cardvision calibrate`` scores a grid of rules this way.

**Two configurations on the same events** (``run_compare``, ``compare_results``): the check to
run after refreshing a bank with new cards, or after retraining a model. The first
configuration is the *base* and the second the *candidate*. Each event has an outcome under
both, reduced to three words: ``right``, ``wrong``, ``nothing``, where an adjacent name on an
off-card position counts as ``nothing`` and only events of the scored roles (readable faces,
backs, off-card positions) take part. The transition table counts events by the pair of words,
with one extra row, ``wrong -> a different wrong name``, for an event that is wrong both times
under two names (two cards of different identities, under the catalog's rule).

Three transitions are **regressions**: ``right -> nothing``, ``right -> wrong`` and
``nothing -> wrong``. Everything else is not, including ``wrong -> a different wrong name``.
Regressing events are listed one by one, with the card ids beside the names where the
catalog lets different cards share a name. Beside the table are the paired differences,
candidate minus base, of the two rates, with intervals from resampling whole recording groups:
both configurations are evaluated on the same resampled groups in every replicate.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cardvision.data.extract import FrameStore
from cardvision.data.schema import Row
from cardvision.evaluate.config import Configuration
from cardvision.evaluate.events import (
    ADJACENT,
    FACE,
    NOTHING,
    RIGHT,
    SCORED_ROLES,
    WRONG,
    Event,
    Scored,
    rates,
    row_events,
    score_row,
)
from cardvision.evaluate.hover import HoverLayer
from cardvision.evaluate.rows import Account, scorable_rows, sliced
from cardvision.evaluate.stats import GroupBootstrap
from cardvision.infer.pipeline import FrameLooks
from cardvision.recognize import AcceptRule

REGRESSIONS = frozenset({(RIGHT, NOTHING), (RIGHT, WRONG), (NOTHING, WRONG)})
DIFFERENT_WRONG = "wrong -> a different wrong name"


def _event_labels(names):
    def labels(row: Row) -> Counter:
        return Counter(event.role for event in row_events(row, names))
    return labels


def source_of(config: Configuration, row: Row) -> dict[str, str]:
    """The row's source type, as a keyword for the pipeline, when the configuration has a
    geometric stage (whose ``gate.sources`` reads it); nothing otherwise."""
    return {"source_type": row.source_type} if config.pipeline.recognizer.geometric is not None else {}


def run_hover(rows: Sequence[Row], frames: FrameStore | str | Path, config: Configuration, account: Account,
              observe: Callable[..., None] | None = None) -> list[Scored]:
    """Score every hover event of ``rows`` under one configuration.

    ``observe``, when given, is called after each frame with ``(row, rgb, result, layer,
    scored)``: the frame, its analysis, its hover layer and its scored events. It only watches
    (``cardvision.evaluate.evidence``)."""
    names, identities = config.catalog.names(), config.catalog.identities()
    scored: list[Scored] = []
    for row, rgb in scorable_rows(rows, frames, account, _event_labels(names)):
        events = row_events(row, names, account.labels_skipped)
        result = config.pipeline.analyze(rgb, **source_of(config, row))
        layer = HoverLayer.of(result, config.selection)
        found = score_row(row, events, layer, names, identities)
        if observe is not None:
            observe(row, rgb, result, layer, found)
        scored.extend(found)
    return scored


def hover_results(scored: Sequence[Scored], seed: int, replicates: int) -> dict[str, Any]:
    """The rates overall, by source type and by source family."""
    return sliced(scored, lambda s: s.event.source_type, lambda part: rates(part, seed, replicates))


# ------------------------------------------------------------------------ many accept rules
@dataclass(frozen=True)
class AnalyzedRow:
    """A usable row, its events, and its frame analyzed up to the accept rule."""

    row: Row
    events: tuple[Event, ...]
    looks: FrameLooks


class HoverAnalysis:
    """The usable rows of a run, each frame analyzed once, to be scored under any accept rule.

    ``score(rule)`` gives what ``run_hover`` gives for the same rows under a configuration that
    holds ``rule``: the frame's result under the rule (``FrameLooks.under``), the hover layer
    of the configuration's selection, and the events scored against it (``score_row``).

    Many rules decide a frame alike. On the way from a frame's looks to its events the rule is
    asked two things about a look and nothing else: whether it reaches ``tau``
    (``AcceptRule.passes_tau``) and whether it is accepted (``AcceptRule.accepts``). So a frame
    is scored once for each distinct list of those answers over its looks, and the result is
    used again for every rule that gives the same answers.
    """

    def __init__(self, config: Configuration, rows: Sequence[AnalyzedRow]) -> None:
        self.rows = list(rows)
        self._names = config.catalog.names()
        self._identities = config.catalog.identities()
        self._selection = config.selection
        self._retry = config.pipeline.recognizer.rule.rotation_retry
        self._looks = [[look for index in sorted(item.looks.looks) for look in item.looks.looks[index]]
                       for item in self.rows]
        self._scored: list[dict[tuple[tuple[bool, bool], ...], list[Scored]]] = [{} for _ in self.rows]

    def score(self, rule: AcceptRule) -> list[Scored]:
        """Every hover event under ``rule``, in the order ``run_hover`` gives them.

        ``rule`` may have any ``tau`` and ``delta``. Its ``rotation_retry`` must be the one the
        frames were analyzed with, which decided the looks that were computed.
        """
        if rule.rotation_retry != self._retry:
            raise ValueError("the frames were analyzed for a rule with another rotation_retry setting")
        scored: list[Scored] = []
        for item, looks, seen in zip(self.rows, self._looks, self._scored):
            answers = tuple((rule.passes_tau(look), rule.accepts(look)) for look in looks)
            if answers not in seen:
                layer = HoverLayer.of(item.looks.under(rule), self._selection)
                seen[answers] = score_row(item.row, item.events, layer, self._names, self._identities)
            scored.extend(seen[answers])
        return scored


def analyze_hover(rows: Sequence[Row], frames: FrameStore | str | Path, config: Configuration, account: Account,
                  progress: Callable[[int], None] | None = None) -> HoverAnalysis:
    """Analyze every usable row's frame once, up to the accept rule (see ``HoverAnalysis``).

    The rows, the events and the account are the ones ``run_hover`` has. ``progress`` is called
    with the number of frames analyzed so far, after each.
    """
    names = config.catalog.names()
    analyzed: list[AnalyzedRow] = []
    for row, rgb in scorable_rows(rows, frames, account, _event_labels(names)):
        events = row_events(row, names, account.labels_skipped)
        analyzed.append(AnalyzedRow(row, tuple(events), config.pipeline.analyze_looks(rgb)))
        if progress is not None:
            progress(len(analyzed))
    return HoverAnalysis(config, analyzed)


# ------------------------------------------------------------------------- two configurations
def run_compare(
    rows: Sequence[Row],
    frames: FrameStore | str | Path,
    base: Configuration,
    candidate: Configuration,
    account: Account,
) -> list[tuple[Scored, Scored]]:
    """Score the same events under two configurations: ``(base, candidate)`` per event.

    The events and their labeled cards come from the base configuration's catalog. What the
    candidate shows gets its identity from the candidate's own catalog, and the two identity
    keys are compared as text, so the two catalogs must state the same identity rule and
    agree on the cards they share: on their names under the ``name`` rule, on their
    ``printingOf`` under the ``printing`` rule. Catalogs that state different rules are
    refused with ``ValueError``.
    """
    if base.catalog.identity != candidate.catalog.identity:
        raise ValueError(
            f'the base catalog states the identity rule "{base.catalog.identity}" and the candidate\'s states '
            f'"{candidate.catalog.identity}": a comparison needs one rule for what counts as the same card')
    names, identities = base.catalog.names(), base.catalog.identities()
    pairs: list[tuple[Scored, Scored]] = []
    for row, rgb in scorable_rows(rows, frames, account, _event_labels(names)):
        events = row_events(row, names, account.labels_skipped)
        both = []
        for config in (base, candidate):
            layer = HoverLayer.of(config.pipeline.analyze(rgb, **source_of(config, row)), config.selection)
            shown_by = (config.catalog.names(), config.catalog.identities())
            both.append(score_row(row, events, layer, names, identities, shown_by))
        pairs.extend(zip(*both))
    return pairs


def reduced(scored: Scored) -> str:
    """An outcome as one of ``right``, ``wrong``, ``nothing`` (adjacent counts as nothing)."""
    return NOTHING if scored.outcome == ADJACENT else scored.outcome


def transition(base: Scored, candidate: Scored) -> str:
    """The transition table's row for one event."""
    before, after = reduced(base), reduced(candidate)
    if before == WRONG and after == WRONG and base.shown_key != candidate.shown_key:
        return DIFFERENT_WRONG
    return f"{before} -> {after}"


def transitions(pairs: Sequence[tuple[Scored, Scored]], with_ids: bool = False) -> dict[str, Any]:
    """The transition table of the scored roles' events, and the regressing events.

    ``with_ids`` puts each card's id beside its name in a regressing event (``truthId``,
    ``baseShownId``, ``candidateShownId``), for a catalog in which different cards share a
    name: the names alone would not say which card was labeled and which was shown.
    """
    def named(key: str, name: str | None, card_id: str | None) -> dict[str, Any]:
        return {key: name, f"{key}Id": card_id} if with_ids else {key: name}

    table: Counter[str] = Counter()
    regressions = []
    events = 0
    for base, candidate in pairs:
        if base.event.role not in SCORED_ROLES:
            continue
        events += 1
        table[transition(base, candidate)] += 1
        if (reduced(base), reduced(candidate)) in REGRESSIONS:
            regressions.append({"key": base.event.key, "group": base.event.group, "source_type": base.event.source_type,
                                "role": base.event.role, **named("truth", base.event.truth, base.event.truth_id),
                                "transition": f"{reduced(base)} -> {reduced(candidate)}",
                                **named("baseShown", base.shown, base.shown_id),
                                **named("candidateShown", candidate.shown, candidate.shown_id)})
    return {"events": events, "table": dict(sorted(table.items())), "regressions": len(regressions),
            "regressingEvents": regressions}


def _is_right(s: Scored) -> float:
    return float(s.event.role == FACE and s.outcome == RIGHT)


def _is_wrong(s: Scored) -> float:
    return float(s.event.role in SCORED_ROLES and s.outcome == WRONG)


def paired_differences(pairs: Sequence[tuple[Scored, Scored]], seed: int, replicates: int) -> dict[str, Any]:
    """Candidate minus base for the two rates, with group-bootstrap intervals."""
    groups = [base.event.group for base, _ in pairs]
    boot = GroupBootstrap(set(groups), seed, replicates)
    faces = boot.per_group(groups, [float(base.event.role == FACE) for base, _ in pairs])
    right_a = boot.per_group(groups, [_is_right(base) for base, _ in pairs])
    right_b = boot.per_group(groups, [_is_right(candidate) for _, candidate in pairs])
    wrong_a = boot.per_group(groups, [_is_wrong(base) for base, _ in pairs])
    wrong_b = boot.per_group(groups, [_is_wrong(candidate) for _, candidate in pairs])

    def ratio(top: float, bottom: float) -> float | None:
        return top / bottom if bottom else None

    def difference(a: float | None, b: float | None) -> float | None:
        return None if a is None or b is None else b - a

    named = boot.ratio(right_b - right_a, faces)
    wrong = boot.ratio_difference(wrong_a, right_a + wrong_a, wrong_b, right_b + wrong_b)
    return {
        "namedRight": {
            "base": ratio(right_a.sum(), faces.sum()), "candidate": ratio(right_b.sum(), faces.sum()),
            "difference": ratio(right_b.sum() - right_a.sum(), faces.sum()), "groupBootstrap95": named,
        },
        "wrongOfShown": {
            "base": ratio(wrong_a.sum(), (right_a + wrong_a).sum()),
            "candidate": ratio(wrong_b.sum(), (right_b + wrong_b).sum()),
            "difference": difference(ratio(wrong_a.sum(), (right_a + wrong_a).sum()),
                                     ratio(wrong_b.sum(), (right_b + wrong_b).sum())),
            "groupBootstrap95": wrong,
        },
    }


def compare_results(pairs: Sequence[tuple[Scored, Scored]], seed: int, replicates: int,
                    with_ids: bool = False) -> dict[str, Any]:
    """Everything the comparison reports, overall and by source. ``with_ids``: see ``transitions``."""
    def measure(part: list[tuple[Scored, Scored]]) -> dict[str, Any]:
        return {
            "transitions": transitions(part, with_ids),
            "paired": paired_differences(part, seed, replicates),
            "base": rates([base for base, _ in part], seed, replicates),
            "candidate": rates([candidate for _, candidate in part], seed, replicates),
        }

    return sliced(pairs, lambda pair: pair[0].event.source_type, measure)
