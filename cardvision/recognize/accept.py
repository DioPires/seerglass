# SPDX-License-Identifier: Apache-2.0
"""The accept rule: when a look is trusted enough to name a card, and which look to use.

One look is **accepted** when its score is at least ``tau`` *and* its margin over the best
different identity is at least ``delta``. ``tau`` refuses crops that resemble nothing in the
bank; ``delta`` refuses crops that resemble two different cards about equally.

A card may lie sideways or upside down, so a crop can be looked at in four orientations (the
upright one and three quarter turns). The orientation rule keeps the common case cheap:

* if the upright look is accepted it is used, and the turned looks are never computed;
* otherwise, when the rule's ``rotationRetry`` is true, the three turned looks are computed and
  all four compete:

  * if any look is accepted, the accepted look with the highest score is used (on equal
    scores, the lower orientation);
  * if none is, the look with the highest score is reported, a turned look replacing the
    upright one only when it is strictly higher. Nothing is named in this case.

With ``rotationRetry`` false the turned looks are never computed: the upright look is the
answer, accepted or not. That suits footage where every card is known to be upright.

The two outcomes of a crop are therefore *named* (a look was accepted, and its best row's card
is the answer) and *nothing* (no look was accepted; the best guess is still reported, so that a
person or a later stage can see what it was).

``tau`` and ``delta`` belong to a model and its bank: they are calibrated on crops of real
footage, and a retrained model or another game needs its own (``cardvision calibrate``).

The rule is stored as a JSON object with exactly three keys, which is both a game's
``accept.json`` and the ``accept`` block of a release document::

    {"tau": 0.62, "delta": 0.05, "rotationRetry": true}

``accept_errors`` is the one check of that object and ``parse_accept`` / ``load_accept`` the one
reader; inference and the release tools both go through them. All three keys are required and
any other key is an error, so that a misspelled setting cannot be silently ignored.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cardvision.recognize.looks import Look

# The values calibrated for the published model and bank.
DEFAULT_TAU = 0.62
DEFAULT_DELTA = 0.05

# The keys of the stored rule, in the order they are written.
ACCEPT_KEYS = ("tau", "delta", "rotationRetry")

NAMED = "named"
NOTHING = "nothing"


@dataclass(frozen=True)
class AcceptRule:
    """The accept rule's settings: the two thresholds, and whether turned looks are tried."""

    tau: float = DEFAULT_TAU
    delta: float = DEFAULT_DELTA
    rotation_retry: bool = True

    def passes_tau(self, look: Look) -> bool:
        """Whether the look's score reaches ``tau`` (the margin is not considered)."""
        return look.score >= self.tau

    def accepts(self, look: Look) -> bool:
        """Whether the look is accepted: score at least ``tau`` and margin at least ``delta``."""
        return look.score >= self.tau and look.margin >= self.delta

    def to_json(self) -> dict[str, Any]:
        """The rule as its stored object (see ``ACCEPT_KEYS``)."""
        return {"tau": self.tau, "delta": self.delta, "rotationRetry": self.rotation_retry}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def accept_errors(accept: Any, where: str = "accept") -> list[str]:
    """Why a stored accept rule is not valid: an empty list when it is.

    ``accept`` is a parsed ``accept.json`` or a release document's ``accept`` block; ``where``
    prefixes every message (pass the file name or the block's name).
    """
    if not isinstance(accept, Mapping):
        return [f"{where} must be an object with the keys {list(ACCEPT_KEYS)}"]
    errors = []
    unknown = sorted(set(accept) - set(ACCEPT_KEYS))
    if unknown:
        errors.append(f"{where} has unknown key(s) {unknown} (expected {list(ACCEPT_KEYS)})")
    tau, delta = accept.get("tau"), accept.get("delta")
    if not _is_number(tau) or not 0 < tau <= 1:
        errors.append(f"{where}.tau must be a number above 0 and at most 1")
    if not _is_number(delta) or not 0 <= delta < 1:
        errors.append(f"{where}.delta must be a number from 0 to below 1")
    if not isinstance(accept.get("rotationRetry"), bool):
        errors.append(f"{where}.rotationRetry must be true or false")
    return errors


class AcceptError(ValueError):
    """A stored accept rule is not valid. ``errors`` lists every reason."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def parse_accept(doc: Any, source: str = "accept rule") -> AcceptRule:
    """Read the rule from a parsed document ``{"tau": 0.62, "delta": 0.05, "rotationRetry": true}``.

    Raises ``AcceptError`` (a ``ValueError``) listing everything ``accept_errors`` finds, each
    message prefixed with ``source``.
    """
    errors = accept_errors(doc, where=source)
    if errors:
        raise AcceptError(errors)
    return AcceptRule(tau=float(doc["tau"]), delta=float(doc["delta"]), rotation_retry=doc["rotationRetry"])


def load_accept(path: str | Path) -> AcceptRule:
    """Read an accept-rule file (see ``parse_accept``)."""
    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path.name}: not valid JSON: {error}") from None
    return parse_accept(doc, source=path.name)


@dataclass(frozen=True)
class Choice:
    """The look the orientation rule settled on for one crop.

    Attributes:
        orientation: 0 for upright, ``k`` for ``k`` quarter turns counter-clockwise.
        look: the chosen look.
        retried: whether the turned looks were computed (the upright look was not accepted and
            the rule allows the retry).
        tau_passed: whether the chosen look's score reaches ``tau``.
        accepted: whether the chosen look is accepted, that is, whether a card is named.
    """

    orientation: int
    look: Look
    retried: bool
    tau_passed: bool
    accepted: bool

    @property
    def outcome(self) -> str:
        """``"named"`` or ``"nothing"``."""
        return NAMED if self.accepted else NOTHING


def _choice(orientation: int, look: Look, retried: bool, rule: AcceptRule) -> Choice:
    return Choice(
        orientation=orientation,
        look=look,
        retried=retried,
        tau_passed=rule.passes_tau(look),
        accepted=rule.accepts(look),
    )


def choose_look(upright: Look, rotations: Callable[[], Sequence[Look]], rule: AcceptRule = AcceptRule()) -> Choice:
    """Apply the orientation rule to one crop.

    ``rotations`` returns the looks at one, two and three quarter turns, in that order. It is
    called at most once, and only when the upright look is not accepted and the rule's
    ``rotation_retry`` is true, so a caller can defer the cost of embedding the turned crops
    until they are needed.
    """
    if rule.accepts(upright) or not rule.rotation_retry:
        return _choice(0, upright, False, rule)
    looks = [upright, *rotations()]
    if len(looks) != 4:
        raise ValueError(f"expected three turned looks, got {len(looks) - 1}")
    best = -1
    for k, look in enumerate(looks):
        if rule.accepts(look) and (best < 0 or look.score > looks[best].score):
            best = k
    if best < 0:
        best = 0
        for k in (1, 2, 3):
            if looks[k].score > looks[best].score:
                best = k
    return _choice(best, looks[best], True, rule)


def choose_among(looks: Sequence[Look], rule: AcceptRule = AcceptRule()) -> Choice:
    """The orientation rule for a crop whose four looks are already known (upright first).

    The turned looks are disregarded when the rule's ``rotation_retry`` is false.
    """
    if len(looks) != 4:
        raise ValueError(f"expected four looks, got {len(looks)}")
    return choose_look(looks[0], lambda: looks[1:], rule)
