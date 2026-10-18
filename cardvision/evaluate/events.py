# SPDX-License-Identifier: Apache-2.0
"""Hover events: the labeled positions of a manifest, what was shown there, and how it counts.

**Events.** An event is one position of one frame where a pointer could rest, with a label
saying what is there.

* Every labeled point of a row is an event (``kind`` of the event: ``point``).
* Every box of an exhaustively boxed row that is a face, a back or a ``not-card`` box, and is
  not an ignore region, gives an event at its center, the mean of its four corners (``center``).
  A center that falls outside the frame gives none. Ignore regions and ``unreadable`` boxes
  give none: an ignore region is not one card, and an ``unreadable`` box has no answer.
* A center that lies inside another labeled box of the same frame gives none either. The
  other box may be of any kind, an ignore region, a ``not-card`` box or an ``unreadable`` box
  included, and containment is the hover rule's own test of a point against a four-corner
  outline (``cardvision.evaluate.hover.point_in_convex_quad``; a point on an edge is inside).
  Where two labels overlap, the manifest does not say which of them is on top at that
  position, so a card lying under another would otherwise be scored as if it showed. These
  centers are counted, as ``center-inside-another-box``, with the other labels that give no
  event. Labeled points are never tested this way: a person chose each of them.

**Roles.** What the label says decides how the event is counted.

==================  ===========================================================================
``face``            a face named by its label and not labeled unreadable (``readable`` is
                    ``yes``, ``maybe`` or was not recorded). These are the *readable faces*.
``face-unreadable`` a named face labeled ``readable: no``: it cannot be named from this frame
                    alone. Outside every rate.
``face-unnamed``    a face nobody named (``unknown`` or ``unlabeled``). Outside every rate.
``back``            a card back. Nothing should be shown.
``off-card``        a ``not-card`` label. Nothing should be shown.
``unreadable``      a point labeled ``unreadable``. Outside every rate.
==================  ===========================================================================

**Outcomes.** With the name the hover rule shows at the position (``cardvision.evaluate.hover``):

* on a ``face``: **right** when the shown card is the labeled card's identity (under the
  default rule, printings of one name are one identity), **wrong** when a card of another
  identity is shown, **nothing** when none is;
* on a ``back``: **wrong** when any name is shown, else **nothing**;
* on an ``off-card`` position: **nothing** when no name is shown; **adjacent** when a name is
  shown and a face labeled with that identity in the same frame has its center inside the
  shown box (the box of a neighboring card reaches the position: a miss of geometry, not of
  identity, and counted apart); otherwise **wrong**;
* on the roles outside every rate: **shown** or **nothing**, counted and not scored.

**Identity.** Which cards are the same is the catalog's rule, and the scoring takes it as
``identities``: card id to identity key (``Catalog.identities()``). The labeled card's key
is compared with the shown card's (``shown_identity``). A caller that has names and no
catalog leaves ``identities`` out, and the names are then compared as text: cards that share
a name are one identity.

The labeled faces that can make an off-card tooltip adjacent are the frame's named faces that
are not ignore regions: the center of a box, the position of a point. Containment in the shown
box includes its edges.

**Rates** (``tally`` and ``rates``), each over a set of events:

* *named right* = right / (right + wrong + nothing), over readable faces;
* *wrong of shown* = (faces named wrong + names shown on backs + non-adjacent names shown on
  off-card positions) / (faces named right + those three).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cardvision.data.schema import Row
from cardvision.evaluate.hover import Box, HoverLayer, Shown, point_in_convex_quad
from cardvision.evaluate.stats import GroupBootstrap, rate
from cardvision.infer.pipeline import reported_box

FACE = "face"
FACE_UNREADABLE = "face-unreadable"
FACE_UNNAMED = "face-unnamed"
BACK = "back"
OFF_CARD = "off-card"
UNREADABLE = "unreadable"
SCORED_ROLES = (FACE, BACK, OFF_CARD)
ROLES = (FACE, FACE_UNREADABLE, FACE_UNNAMED, BACK, OFF_CARD, UNREADABLE)

RIGHT = "right"
WRONG = "wrong"
NOTHING = "nothing"
ADJACENT = "adjacent"
SHOWN = "shown"

NAMED_RIGHT_DEFINITION = (
    "named right = right / (right + wrong + nothing), over readable faces (faces labeled unreadable, and "
    "faces nobody named, are outside it)"
)
WRONG_OF_SHOWN_DEFINITION = (
    "wrong of shown = (faces named wrong + names shown on backs + non-adjacent names shown on off-card "
    "positions) / (faces named right + those three)"
)
ADJACENT_DEFINITION = (
    "adjacent = a name shown on an off-card position while a face labeled with that name in the same frame "
    "has its center inside the shown box; counted apart, in neither rate"
)
# The reason under which a box center covered by another label is counted.
CENTER_INSIDE_ANOTHER_BOX = "center-inside-another-box"
CENTER_INSIDE_ANOTHER_BOX_DEFINITION = (
    "center inside another box = a box's center that lies inside another labeled box of the same frame (of any "
    "kind, ignore regions and not-card boxes included; a center on an edge is inside) gives no event and is in no "
    f"rate, and is counted among the labels left out as {CENTER_INSIDE_ANOTHER_BOX}; labeled points are not "
    "tested this way"
)


@dataclass(frozen=True)
class Event:
    """One labeled position.

    Attributes:
        platform, video_id, t: the row's key.
        group, source_type: the row's recording group and source type.
        kind: ``point`` or ``center``.
        label: the label's ``id`` when it has one, else ``p<index>`` or ``b<index>``.
        x, y: the position as fractions of the frame.
        role: see the module's table.
        truth: the labeled card's name, for the two named face roles.
        truth_id: the labeled card id.
    """

    platform: str
    video_id: str
    t: float
    group: str
    source_type: str
    kind: str
    label: str
    x: float
    y: float
    role: str
    truth: str | None = None
    truth_id: str | None = None

    @property
    def key(self) -> str:
        """An id for the event, unique across a set of manifests."""
        return f"{self.platform}-{self.video_id}-t{round(self.t * 1000):010d}/{self.kind}/{self.label}"


def _role(kind: str, identity: str | None, card_id: str | None, readable: str | None) -> str | None:
    if kind == "face":
        if identity != "named" or card_id is None:
            return FACE_UNNAMED
        return FACE_UNREADABLE if readable == "no" else FACE
    return {"back": BACK, "not-card": OFF_CARD, "unreadable": UNREADABLE}.get(kind)


def _center(quad: Sequence[Sequence[float]]) -> tuple[float, float]:
    points = np.asarray(quad, dtype=np.float64).reshape(-1, 2)
    return float(points[:, 0].mean()), float(points[:, 1].mean())


def inside_another_box(row: Row, index: int, x: float, y: float) -> bool:
    """Whether the position ``(x, y)``, in fractions of the frame, lies inside a labeled box
    of the row other than box ``index``.

    Every other box counts, whatever its kind and whether or not it is an ignore region. The
    test is the hover rule's, on the box's four corners in frame pixels; a point on an edge is
    inside.
    """
    width, height = row.frame_width, row.frame_height
    px, py = x * width, y * height
    for other, box in enumerate(row.boxes or ()):
        if other != index and point_in_convex_quad(px, py, [(qx * width, qy * height) for qx, qy in box.quad]):
            return True
    return False


def row_events(row: Row, names: Mapping[str, str], skipped: Counter | None = None) -> list[Event]:
    """The events of one row, points first and then box centers, each in label order.

    ``names`` maps card id to name (``Catalog.names()``). ``skipped`` counts the labels that
    give no event, by reason: ``ignore-region``, ``unreadable-box``, ``center-outside-frame``,
    ``center-inside-another-box`` for a center that another labeled box of the frame covers,
    and ``card-not-in-catalog`` for a named face whose id the catalog does not hold. A box
    is counted under the first of these that applies, in that order, so the events and the
    counts together account for every point and box that has a role.
    """
    tally = skipped if skipped is not None else Counter()
    common = {"platform": row.platform, "video_id": row.video_id, "t": row.t, "group": row.group,
              "source_type": row.source_type}
    events: list[Event] = []

    def add(kind: str, label: str, x: float, y: float, role: str, card_id: str | None) -> None:
        truth = None
        if role in (FACE, FACE_UNREADABLE):
            if card_id not in names:
                tally["card-not-in-catalog"] += 1
                return
            truth = names[card_id]
        events.append(Event(kind=kind, label=label, x=x, y=y, role=role, truth=truth,
                            truth_id=card_id if truth is not None else None, **common))

    for index, point in enumerate(row.points or ()):
        role = _role(point.kind, point.identity, point.card_id, point.readable)
        if role is not None:
            add("point", point.id or f"p{index}", point.x, point.y, role, point.card_id)
    if row.completeness == "exhaustive-boxes":
        for index, box in enumerate(row.boxes or ()):
            if box.ignore:
                tally["ignore-region"] += 1
                continue
            if box.kind == "unreadable":
                tally["unreadable-box"] += 1
                continue
            role = _role(box.kind, box.identity, box.card_id, box.readable)
            if role is None:
                continue
            x, y = _center(box.quad)
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                tally["center-outside-frame"] += 1
                continue
            if inside_another_box(row, index, x, y):
                tally[CENTER_INSIDE_ANOTHER_BOX] += 1
                continue
            add("center", box.id or f"b{index}", x, y, role, box.card_id)
    return events


def named_face_centers(row: Row, names: Mapping[str, str]) -> list[tuple[float, float, str]]:
    """``(x, y, name)`` of every named face of a row that is not an ignore region, as fractions
    of the frame: the labels that can make an off-card tooltip *adjacent*. Given a map of card
    id to identity key in place of ``names``, the third value is the face's identity."""
    out = []
    for point in row.points or ():
        if point.kind == "face" and point.identity == "named" and point.card_id in names:
            out.append((point.x, point.y, names[point.card_id]))
    for box in row.boxes or ():
        if box.kind == "face" and not box.ignore and box.identity == "named" and box.card_id in names:
            x, y = _center(box.quad)
            out.append((x, y, names[box.card_id]))
    return out


# ------------------------------------------------------------------------------------ outcomes
def box_contains(box: Box, px: float, py: float) -> bool:
    """Whether a point lies in an ``(x, y, width, height)`` box, edges included."""
    x, y, width, height = box
    return x <= px <= x + width and y <= py <= y + height


def is_adjacent(name: str | None, box: Box | None, faces: Iterable[tuple[float, float, str]]) -> bool:
    """Whether a labeled face of the shown name has its center inside the shown box.

    ``faces`` are ``(x, y, name)`` in the same pixels as ``box``. The names may be identity
    keys on both sides: what is compared is whether they are equal.
    """
    if name is None or box is None:
        return False
    return any(face_name == name and box_contains(box, x, y) for x, y, face_name in faces)


def classify(role: str, shown: str | None, truth: str | None = None, adjacent: bool = False) -> str:
    """The outcome of one event (see the module's rules).

    ``shown`` is the name the hover rule shows, or ``None``; ``truth`` the labeled name of a
    face; ``adjacent`` whether the adjacency rule holds for the shown name and box. ``shown``
    and ``truth`` may be identity keys in place of names: a face is right when they are equal.
    """
    if shown is None:
        return NOTHING
    if role == FACE:
        return RIGHT if shown == truth else WRONG
    if role == BACK:
        return WRONG
    if role == OFF_CARD:
        return ADJACENT if adjacent else WRONG
    return SHOWN


@dataclass(frozen=True)
class Scored:
    """An event with what was shown at it.

    ``identity`` is the shown card's identity key when the scoring had one (``shown_identity``);
    it is not written out. ``shown_key`` is what tells one shown card from another.
    """

    event: Event
    outcome: str
    shown: str | None = None
    shown_id: str | None = None
    box: Box | None = None
    identity: str | None = None

    @property
    def shown_key(self) -> str | None:
        """The shown card's identity key, or the shown name when the scoring had no key."""
        return self.shown if self.identity is None else self.identity

    def to_json(self) -> dict[str, Any]:
        e = self.event
        return {"key": e.key, "platform": e.platform, "video_id": e.video_id, "t": e.t, "group": e.group,
                "source_type": e.source_type, "kind": e.kind, "label": e.label, "x": e.x, "y": e.y, "role": e.role,
                "truth": e.truth, "truth_id": e.truth_id, "outcome": self.outcome, "shown": self.shown,
                "shown_id": self.shown_id, "box": None if self.box is None else reported_box(self.box)}


def shown_identity(shown: Shown, names: Mapping[str, str], identities: Mapping[str, str] | None) -> str | None:
    """The identity key of what is shown, or ``None`` when nothing is.

    ``names`` and ``identities`` map card id to name and to identity key, for the catalog of
    the configuration that showed it. The key is the shown card id's, when that catalog
    holds the id under the shown name. Otherwise it is the shown name itself: without
    ``identities``, and for a card shown by a runtime that had another catalog.
    """
    if shown.name is None or identities is None or names.get(shown.card_id) != shown.name:
        return shown.name
    return identities[shown.card_id]


def truth_identity(event: Event, identities: Mapping[str, str] | None) -> str | None:
    """The identity key of an event's labeled card (``None`` when the label names no card):
    its key in ``identities``, or the labeled name without ``identities``."""
    return event.truth if identities is None or event.truth_id is None else identities[event.truth_id]


def score_event(event: Event, shown: Shown, faces_px: Iterable[tuple[float, float, str]],
                shown_key: str | None = None, truth_key: str | None = None) -> Scored:
    """Classify one event given what the hover rule resolved at its position.

    ``shown_key`` and ``truth_key`` are the identity keys of the shown card and of the labeled
    card (``shown_identity``, ``Catalog.identities()``), and ``faces_px`` then carries keys
    too. Left out, the shown name and the labeled name are compared.
    """
    if shown_key is None and truth_key is None:
        shown_key, truth_key = shown.name, event.truth
    adjacent = event.role == OFF_CARD and is_adjacent(shown_key, shown.box, faces_px)
    return Scored(event=event, outcome=classify(event.role, shown_key, truth_key, adjacent), shown=shown.name,
                  shown_id=shown.card_id, box=shown.box, identity=shown_key)


def score_row(row: Row, events: Sequence[Event], layer: HoverLayer, names: Mapping[str, str],
              identities: Mapping[str, str] | None = None,
              shown_by: tuple[Mapping[str, str], Mapping[str, str]] | None = None) -> list[Scored]:
    """Score a row's events against its frame's hover layer.

    ``names`` and ``identities`` map card id to name and to identity key for the catalog the
    labels are read with (see the module's note on identity). ``shown_by`` is the same pair for
    the configuration whose layer this is, when it has a catalog of its own (the candidate of
    a comparison); left out, it is the labels' catalog.
    """
    width, height = layer.frame_width, layer.frame_height
    keys = names if identities is None else identities
    shown_names, shown_identities = (names, identities) if shown_by is None else shown_by
    faces_px = [(x * width, y * height, key) for x, y, key in named_face_centers(row, keys)]
    scored = []
    for event in events:
        shown = layer.at(event.x * width, event.y * height)
        scored.append(score_event(event, shown, faces_px, shown_identity(shown, shown_names, shown_identities),
                                  truth_identity(event, identities)))
    return scored


# -------------------------------------------------------------------------------------- rates
def tally(scored: Iterable[Scored]) -> dict[str, int]:
    """Counts of outcomes by role, under fixed keys."""
    count: Counter[tuple[str, str]] = Counter((s.event.role, s.outcome) for s in scored)
    out = {
        "faceRight": count[(FACE, RIGHT)],
        "faceWrong": count[(FACE, WRONG)],
        "faceNothing": count[(FACE, NOTHING)],
        "backShown": count[(BACK, WRONG)],
        "backNothing": count[(BACK, NOTHING)],
        "offCardShownNonAdjacent": count[(OFF_CARD, WRONG)],
        "offCardShownAdjacent": count[(OFF_CARD, ADJACENT)],
        "offCardNothing": count[(OFF_CARD, NOTHING)],
    }
    for role, label in ((FACE_UNREADABLE, "faceUnreadable"), (FACE_UNNAMED, "faceUnnamed"), (UNREADABLE, "unreadable")):
        out[f"{label}Shown"] = count[(role, SHOWN)]
        out[f"{label}Nothing"] = count[(role, NOTHING)]
    return out


def _is_wrong(s: Scored) -> bool:
    return s.event.role in SCORED_ROLES and s.outcome == WRONG


def _is_right(s: Scored) -> bool:
    return s.event.role == FACE and s.outcome == RIGHT


def rates(scored: Sequence[Scored], seed: int, replicates: int, exact: bool = True) -> dict[str, Any]:
    """The counts and the two rates of a set of scored events, with their intervals.

    The bootstrap resamples the recording groups that have at least one event in the set.
    With ``exact`` false the exact bound of each rate is left out: it takes long to compute
    for thousands of events, and a caller that scores many rules needs it for few of them.
    The counts, the values and the bootstrap intervals are the same either way.
    """
    counts = tally(scored)
    boot = GroupBootstrap({s.event.group for s in scored}, seed, replicates)
    groups = [s.event.group for s in scored]
    right = boot.per_group(groups, [float(_is_right(s)) for s in scored])
    wrong = boot.per_group(groups, [float(_is_wrong(s)) for s in scored])
    faces = boot.per_group(groups, [float(s.event.role == FACE) for s in scored])
    n_right, n_faces = counts["faceRight"], counts["faceRight"] + counts["faceWrong"] + counts["faceNothing"]
    n_wrong = counts["faceWrong"] + counts["backShown"] + counts["offCardShownNonAdjacent"]
    return {
        "events": len(scored),
        "groups": len(boot.groups),
        "counts": counts,
        "namedRight": rate(n_right, n_faces, bound="lower" if exact else None, bootstrap=boot, per_group=(right, faces)),
        "wrongOfShown": rate(n_wrong, n_right + n_wrong, bound="upper" if exact else None, bootstrap=boot,
                             per_group=(wrong, right + wrong)),
    }
