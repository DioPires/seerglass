# SPDX-License-Identifier: Apache-2.0
"""Check that a new bank only adds to an old one, apart from rows whose image changed.

Results measured with a bank hold for a later bank only if nothing they rested on moved. A
refresh passes when:

* every card id of the old bank is in the new one (ids are only added);
* the shared ids are in the same relative order;
* every shared entry is unchanged, its stored vector byte for byte, with one exception below;
* the two banks were made the same way (same game, recognizer label, vector size,
  preprocessing, views and aggregation), and the new version number is higher;
* the identity groups only grow: two old cards that were one identity still are, two that were
  not still are not, and a group changes only by gaining new cards. The two groups files must
  record the same identity rule (``rule.level``): a refresh does not change what counts as the
  same card.

The exception: a shared row may differ when both banks record the image it was embedded from
(``artV``, see ``cardvision.core.bank``) and the two values differ. The card's image changed,
and the row was embedded again from the new one. Such a row is accepted and listed with both
values. It did move, so results measured with the old bank do not cover it. A shared row whose
vector changed while both banks record the same image, or while either records none, is an
error naming the row: nothing explains the change.

A shared row that keeps its vector may gain an ``artV`` it did not have. That is reported as a
note. It may not lose one, and no other field of a shared entry may change.

A group's representative may change when a new card joins it: one with a smaller key, or the
card the others are printings of. That is reported as a note, not as a failure: the
representative is only a key for the identity. The groups are read with the representatives
each file records; this check does not derive identity itself.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any

from cardvision.bank.build import BANK_KIND, GROUPS_KIND
from cardvision.core.bank import (
    ART_VERSION_KEY,
    IdentityGroups,
    recorded_representatives,
    stored_art_versions,
    stored_vectors,
)
from cardvision.core.catalog import IDENTITY_PRINTING

_SAME_KEYS = ("game", "locale", "model", "dim", "dtype", "preprocessing", "views", "aggregation")


@dataclass
class RefreshReport:
    """What ``check_refresh`` found.

    Attributes:
        errors: why the new bank is not an accepted refresh of the old one (empty when it is).
        notes: differences that are allowed and worth knowing about.
        shared: how many card ids the two banks share. It always equals ``identical`` plus the
            lengths of ``reembedded``, ``unverified`` and ``changed``.
        identical: how many shared rows have a byte-identical stored vector and no recorded
            change of image.
        reembedded: the shared rows accepted as embedded again because their image changed,
            each as ``(card id, old artV, new artV)``, in the old bank's order.
        changed: the shared rows whose stored vector changed with nothing to explain it. Each
            is also named in ``errors``.
        unverified: the shared rows whose vector changed, whose old row records no image, and
            which the caller said it embedded again on purpose (``reembedded_unrecorded``).
            Always empty for a plain comparison of two banks.
        added: the card ids only the new bank has, in its order.
        appended: whether every added id comes after every shared one in the new bank.
    """

    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    shared: int = 0
    identical: int = 0
    reembedded: list[tuple[str, str, str]] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    appended: bool = True

    @property
    def ok(self) -> bool:
        return not self.errors


def _some(items: list[Any], limit: int = 5) -> str:
    shown = ", ".join(str(item) for item in items[:limit])
    return shown + (f", and {len(items) - limit} more" if len(items) > limit else "")


def _read_bank(doc: Any, label: str, errors: list[str]):
    if not isinstance(doc, Mapping) or doc.get("kind") != BANK_KIND:
        errors.append(f'the {label} bank is not an "{BANK_KIND}" document')
        return None
    try:
        ids, stored = stored_vectors(doc)
        recorded = stored_art_versions(doc)
    except ValueError as error:
        errors.append(f"the {label} bank cannot be read: {error}")
        return None
    repeated = sorted(card_id for card_id, count in Counter(ids).items() if count > 1)
    if repeated:
        errors.append(f"the {label} bank has more than one row for: {_some(repeated)}")
        return None
    return ids, stored, recorded


def _without_vector_and_image(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in entry.items() if key not in ("vec", ART_VERSION_KEY)}


def _read_groups(doc: Any, ids: list[str], label: str, errors: list[str]) -> IdentityGroups | None:
    if not isinstance(doc, Mapping) or doc.get("kind") != GROUPS_KIND or not isinstance(doc.get("groups"), list):
        errors.append(f'the {label} identity groups are not an "{GROUPS_KIND}" document')
        return None
    try:
        return IdentityGroups(doc["groups"], ids, prefer=recorded_representatives(doc))
    except (ValueError, TypeError) as error:
        errors.append(f"the {label} identity groups cannot be read: {error}")
        return None


def recorded_rule(doc: Any) -> Any:
    """The identity rule a groups document records (``rule.level``), or ``None`` when it
    records none."""
    rule = doc.get("rule") if isinstance(doc, Mapping) else None
    return rule.get("level") if isinstance(rule, Mapping) else None


def _check_groups(old: IdentityGroups, new: IdentityGroups, old_ids: list[str], report: RefreshReport,
                  by_key: bool = True) -> None:
    """``by_key`` says that a group's representative is its member with the smallest key (the
    ``name`` rule), which words the note about a representative that changed."""
    old_set = set(old_ids)
    old_groups = {tuple(group) for group in old.groups}
    kept = {tuple(card_id for card_id in group if card_id in old_set) for group in new.groups}
    kept = {group for group in kept if len(group) >= 2}
    for group in sorted(old_groups):
        if not any(set(group) <= set(candidate) for candidate in kept):
            report.errors.append(
                f"identity group {list(group)} of the old bank is split in the new groups: "
                "cards that were one identity must stay one"
            )
    for group in sorted(kept - old_groups):
        if any(set(group) < set(whole) for whole in old_groups):
            continue  # part of a split group, reported above
        report.errors.append(
            f"old cards {list(group)} are one identity in the new groups and were not all one identity "
            "before: a refresh may only add new cards to a group"
        )
    moved = [(card_id, old.of(card_id), new.of(card_id)) for card_id in old_ids if old.of(card_id) != new.of(card_id)]
    if moved and not report.errors:
        card_id, before, after = moved[0]
        why = "a new card with a smaller key" if by_key else "a new card that stands for it"
        report.notes.append(
            f"{len(moved)} old card(s) have a new representative, because {why} "
            f"joined their identity (e.g. {card_id}: {before} -> {after})"
        )


def check_refresh(old_bank: Any, new_bank: Any, old_groups: Any = None, new_groups: Any = None, *,
                  reembedded_unrecorded: Collection[str] = ()) -> RefreshReport:
    """Compare two parsed bank documents, and their identity groups when both are given.

    ``reembedded_unrecorded`` is for the tool that made the new bank from the old one and
    embedded some rows again although the old bank records no image for them. It names those
    card ids. A changed vector among them is listed in ``unverified`` instead of being an
    error. A comparison of two banks from files passes nothing, and such a row is an error.
    """
    report = RefreshReport()
    old = _read_bank(old_bank, "old", report.errors)
    new = _read_bank(new_bank, "new", report.errors)
    if old is None or new is None:
        return report
    (old_ids, old_stored, old_art), (new_ids, new_stored, new_art) = old, new

    for key in _SAME_KEYS:
        if key in old_bank and key in new_bank and old_bank[key] != new_bank[key]:
            report.errors.append(
                f'the banks were not made the same way: "{key}" is {old_bank[key]!r} in the old bank '
                f"and {new_bank[key]!r} in the new one"
            )
    old_version, new_version = old_bank.get("version"), new_bank.get("version")
    if isinstance(old_version, int) and isinstance(new_version, int) and new_version <= old_version:
        report.errors.append(f"the new bank's version {new_version} is not greater than the old bank's {old_version}")

    new_position = {card_id: position for position, card_id in enumerate(new_ids)}
    old_set = set(old_ids)
    removed = [card_id for card_id in old_ids if card_id not in new_position]
    if removed:
        report.errors.append(
            f"{len(removed)} card id(s) of the old bank are missing from the new one: {_some(removed)}"
        )
    shared_old = [card_id for card_id in old_ids if card_id in new_position]
    shared_new = [card_id for card_id in new_ids if card_id in old_set]
    report.shared = len(shared_old)
    report.added = [card_id for card_id in new_ids if card_id not in old_set]
    report.appended = shared_new == new_ids[:len(shared_new)]
    if shared_old != shared_new:
        first = next(i for i, (a, b) in enumerate(zip(shared_old, shared_new)) if a != b)
        report.errors.append(
            f"the shared card ids are not in the same order: shared row {first} is {shared_old[first]} "
            f"in the old bank and {shared_new[first]} in the new one"
        )

    old_entries = old_bank["entries"]
    new_entries = new_bank["entries"]
    old_position = {card_id: position for position, card_id in enumerate(old_ids)}
    allowed = set(reembedded_unrecorded)
    # Rows whose vector changed with nothing to explain it, by what the banks record of the image.
    same_image: list[str] = []
    old_unrecorded: list[str] = []
    new_unrecorded: list[str] = []
    neither_recorded: list[str] = []
    changed_otherwise: list[str] = []
    reembedded_otherwise: list[str] = []
    newly_recorded: list[str] = []
    width_differs = old_stored.shape[1:] != new_stored.shape[1:]
    for card_id in shared_old:
        i, j = old_position[card_id], new_position[card_id]
        before, after = old_art[i], new_art[j]
        same_vector = not width_differs and old_stored[i].tobytes() == new_stored[j].tobytes()
        same_rest = _without_vector_and_image(old_entries[i]) == _without_vector_and_image(new_entries[j])
        if width_differs:
            neither_recorded.append(card_id)
        elif before is not None and after is not None and before != after:
            report.reembedded.append((card_id, before, after))
            if not same_rest:
                reembedded_otherwise.append(card_id)
        elif not same_vector:
            if before is None and after is not None and card_id in allowed:
                report.unverified.append(card_id)
                if not same_rest:
                    reembedded_otherwise.append(card_id)
            elif before is None and after is None:
                neither_recorded.append(card_id)
            elif before is None:
                old_unrecorded.append(card_id)
            elif after is None:
                new_unrecorded.append(card_id)
            else:
                same_image.append(card_id)
        else:
            report.identical += 1
            if before is None and after is not None and same_rest:
                newly_recorded.append(card_id)
            elif old_entries[i] != new_entries[j]:
                changed_otherwise.append(card_id)
    report.changed = [card_id for card_id in shared_old
                      if card_id in {*same_image, *old_unrecorded, *new_unrecorded, *neither_recorded}]
    for rows, reason in (
        (neither_recorded, ""),
        (same_image, f", and both banks record the same image ({ART_VERSION_KEY}) for them"),
        (old_unrecorded, f", and the old bank records no image ({ART_VERSION_KEY}) for them"),
        (new_unrecorded, f", and the new bank records no image ({ART_VERSION_KEY}) for them"),
    ):
        if rows:
            report.errors.append(f"{len(rows)} shared row(s) are not byte-identical{reason}: {_some(rows)}")
    if changed_otherwise:
        report.errors.append(
            f"{len(changed_otherwise)} shared entr(ies) keep their vector and differ in another field: "
            f"{_some(changed_otherwise)}"
        )
    if reembedded_otherwise:
        report.errors.append(
            f"{len(reembedded_otherwise)} shared entr(ies) embedded again also differ in a field other than "
            f"the vector and {ART_VERSION_KEY}: {_some(reembedded_otherwise)}"
        )
    if newly_recorded:
        report.notes.append(
            f"{len(newly_recorded)} shared row(s) keep their vector and now record their image "
            f"({ART_VERSION_KEY}): {_some(newly_recorded)}"
        )
    if not report.appended and shared_old == shared_new:
        report.notes.append("the added card ids are not all at the end of the new bank")

    if (old_groups is None) != (new_groups is None):
        report.errors.append("identity groups were given for one bank only: give both, or neither")
    elif old_groups is not None:
        before = _read_groups(old_groups, old_ids, "old", report.errors)
        after = _read_groups(new_groups, new_ids, "new", report.errors)
        if before is not None and after is not None:
            old_rule, new_rule = recorded_rule(old_groups), recorded_rule(new_groups)
            if old_rule is not None and new_rule is not None and old_rule != new_rule:
                report.errors.append(
                    f'the old identity groups were made by the "{old_rule}" rule and the new ones by the '
                    f'"{new_rule}" rule: a refresh keeps the identity rule of the catalog'
                )
            _check_groups(before, after, old_ids, report, by_key=new_rule != IDENTITY_PRINTING)
    return report
