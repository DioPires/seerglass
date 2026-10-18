# SPDX-License-Identifier: Apache-2.0
"""Seeded role assignment for recording groups.

Footage is grouped by where it was recorded: videos that share an origin, such as a channel or
an event, form one *recording group*. A group gets one role and every frame of it carries that
role, so nothing recorded at one table ends up on both sides of a train/evaluation split.

Which role a group gets is decided by a seeded shuffle and nothing else: no model output, no
look at the frames, and no choice by whoever labeled them. The functions here are that rule.
They are deterministic: the same groups, in the same order, with the same seed and settings
give the same roles on any machine (the shuffle is ``random.Random(seed).shuffle``).

Two forms of the rule are provided.

``assign_by_quota``
    Groups are sorted within each stratum and shuffled; the first ``n`` of a stratum take one
    role, the next ``m`` another, and the rest a default. Use it when the number of groups each
    role needs is fixed ahead.

``assign_by_share``
    Groups are shuffled once; each in turn takes whichever of two roles is further below its
    target share in the strata the group belongs to. Use it when a share is fixed ahead, and
    when new groups join groups that already have roles (``counted`` carries the old tallies).

``tie_groups`` builds the groups in the first place, from rows that share any tie key.
"""

from __future__ import annotations

import dataclasses
import hashlib
import random
from collections import Counter
from collections.abc import Collection, Hashable, Iterable, Mapping, Sequence

from cardvision.data.schema import Row


def tie_groups(keys_per_item: Sequence[Collection[Hashable]]) -> list[list[int]]:
    """Group items that share a tie key, directly or through other items.

    ``keys_per_item[i]`` holds the tie keys of item ``i`` (its channel and its event, say).
    Two items with a key in common are in one group, and so are items linked through a chain
    of such pairs. Returns the groups as lists of item indices, each list ascending, the groups
    ordered by their smallest index. An item without keys is a group of its own.
    """
    parent = list(range(len(keys_per_item)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    holder: dict[Hashable, int] = {}
    for item, keys in enumerate(keys_per_item):
        for key in keys:
            if key in holder:
                parent[find(item)] = find(holder[key])
            else:
                holder[key] = item
    groups: dict[int, list[int]] = {}
    for item in range(len(keys_per_item)):
        groups.setdefault(find(item), []).append(item)
    return sorted(groups.values(), key=lambda members: members[0])


def shuffled(items: Sequence, seed: int) -> list:
    """The items in the order ``random.Random(seed).shuffle`` leaves them."""
    out = list(items)
    random.Random(seed).shuffle(out)
    return out


def assign_by_quota(
    strata: Mapping[str, Hashable],
    seed: int,
    quota: Mapping[Hashable, Sequence[tuple[str, int]]],
    rest: str = "train",
) -> dict[str, str]:
    """Roles by a fixed number of groups per stratum.

    Args:
        strata: each group's stratum, by group key. Strata are any sortable values of one
            type, for example a source type or a ``(source type, family)`` pair.
        seed: the seed of the shuffle.
        quota: per stratum, the roles to fill and how many groups each takes, in order. A
            stratum without an entry gives every group the ``rest`` role.
        rest: the role of every group left once a stratum's quota is filled.

    The strata are visited in sorted order with one random stream running through all of
    them. Within a stratum the group keys are sorted, shuffled, and dealt out: the first
    groups to the first role of the quota, and so on.
    """
    by_stratum: dict[Hashable, list[str]] = {}
    for group, stratum in strata.items():
        by_stratum.setdefault(stratum, []).append(group)
    rng = random.Random(seed)
    roles: dict[str, str] = {}
    for stratum in sorted(by_stratum):  # type: ignore[type-var]
        groups = sorted(by_stratum[stratum])
        rng.shuffle(groups)
        ladder = [role for role, count in quota.get(stratum, ()) for _ in range(count)]
        for position, group in enumerate(groups):
            roles[group] = ladder[position] if position < len(ladder) else rest
    return roles


def assign_by_share(
    groups: Sequence[tuple[str, Collection[str]]],
    seed: int,
    share: Mapping[str, float],
    *,
    held: str = "evaluation",
    open: str = "train",  # noqa: A002 - the name mirrors `held`
    forced_open: Collection[str] = (),
    counted: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, str]:
    """Roles by a target share per stratum, one group at a time.

    Args:
        groups: ``(group key, the strata it has footage in)``, in a fixed order. The order
            matters: it is what the seed shuffles.
        seed: the seed of the shuffle.
        share: per stratum, the share of its groups that should get the ``held`` role.
        held, open: the two roles.
        forced_open: groups that take the ``open`` role whatever the tallies say (footage that
            was already looked at, for example). They still count towards the tallies.
        counted: groups that already have a role and are not being assigned again, as
            ``{stratum: {role: number of groups}}``. They count towards each stratum's total
            and its tallies, so the shares hold over old and new groups together.

    The groups are visited in shuffled order. For each, the shortfall of either role is summed
    over the group's strata, where a stratum's shortfall is its target count (share times its
    number of groups) less the groups it already has in that role; the group takes ``held``
    when that role's shortfall is at least as large as the other's.
    """
    counted = counted or {}
    totals: Counter[str] = Counter()
    held_count: Counter[str] = Counter()
    open_count: Counter[str] = Counter()
    for stratum, tallies in counted.items():
        totals[stratum] += sum(tallies.values())
        held_count[stratum] += tallies.get(held, 0)
        open_count[stratum] += tallies.get(open, 0)
    for _, group_strata in groups:
        for stratum in set(group_strata):
            totals[stratum] += 1
    missing = sorted({stratum for _, group_strata in groups for stratum in group_strata} - set(share))
    if missing:
        raise ValueError(f"no share given for the stratum {', '.join(missing)}")

    order = list(range(len(groups)))
    random.Random(seed).shuffle(order)
    forced = set(forced_open)
    roles: dict[str, str] = {}
    for index in order:
        key, group_strata = groups[index]
        own = sorted(set(group_strata))
        if key in forced:
            takes_held = False
        else:
            short_held = sum(share[s] * totals[s] - held_count[s] for s in own)
            short_open = sum((1 - share[s]) * totals[s] - open_count[s] for s in own)
            takes_held = short_held >= short_open
        for stratum in own:
            (held_count if takes_held else open_count)[stratum] += 1
        roles[key] = held if takes_held else open
    return roles


def split_in_order(ordered: Sequence[str], share: float, first: str, second: str) -> dict[str, str]:
    """Give the first ``round(len(ordered) * share)`` groups one role and the rest another.

    ``ordered`` is a list of groups already in shuffled order (the visiting order of
    ``assign_by_share``, say). Rounding is Python's: a half goes to the even number.
    """
    count = round(len(ordered) * share)
    return {group: (first if position < count else second) for position, group in enumerate(ordered)}


def visiting_order(group_keys: Sequence[str], seed: int) -> list[str]:
    """The order in which ``assign_by_share`` visits groups given in this order."""
    order = list(range(len(group_keys)))
    random.Random(seed).shuffle(order)
    return [group_keys[index] for index in order]


def pseudonymous_group_id(name: str, salt: str, length: int = 10) -> str:
    """A stable id for a recording group that does not show what the group is.

    ``name`` is whatever the group is known by (a channel, an event) and ``salt`` a value that
    is kept unpublished; without the salt the id cannot be matched to a name by trying names.
    The same name and salt always give the same id.
    """
    if not salt:
        raise ValueError("a non-empty salt is needed; without one the id could be guessed")
    digest = hashlib.sha256(f"{salt}\x00{name}".encode()).hexdigest()
    return "g" + digest[:length]


# ------------------------------------------------------------------------- manifest rows
@dataclasses.dataclass
class RoleAssignment:
    """The outcome of ``assign_manifest_roles``.

    Attributes:
        rows: the rows given, each with its group's role.
        groups: per group of those rows, ``(role, how)``. ``how`` is ``assigned`` for a group
            the rule placed, ``kept`` for a group that already had a role among the published
            rows, and ``trained on`` for a group held in ``train`` because one of its frames
            has trained a released model.
        tallies: per source type, the number of groups in each role, published groups and
            these together.
    """

    rows: list[Row]
    groups: dict[str, tuple[str, str]]
    tallies: dict[str, dict[str, int]]


def _group_strata(rows: Iterable[Row]) -> dict[str, list[str]]:
    strata: dict[str, set[str]] = {}
    for row in rows:
        strata.setdefault(row.group, set()).add(row.source_type)
    return {group: sorted(values) for group, values in strata.items()}


def assign_manifest_roles(
    rows: Sequence[Row],
    seed: int,
    evaluation_share: float,
    *,
    shares: Mapping[str, float] | None = None,
    published: Sequence[Row] = (),
) -> RoleAssignment:
    """Give the recording groups of ``rows`` a role, ``train`` or ``evaluation``.

    The role a row arrives with is not read: whoever offers labels cannot choose it.

    * A group that already appears in ``published`` keeps the role it has there.
    * A group with a frame marked ``trained_on`` is held in ``train``: a frame that has
      trained a released model cannot be used to evaluate it.
    * Every other group is placed by ``assign_by_share``. The groups are taken in sorted
      order of their ids, the strata are their source types, each stratum's share is
      ``shares[source_type]`` or else ``evaluation_share``, and the published groups count
      towards the tallies.

    Raises ``ValueError`` when the published rows give one group two roles, or when a frame
    marked ``trained_on`` belongs to a group whose published role is not ``train``.
    """
    shares = dict(shares or {})
    published_role: dict[str, str] = {}
    for row in published:
        if published_role.setdefault(row.group, row.role) != row.role:
            raise ValueError(f'the published rows give group "{row.group}" two roles')
    published_strata = _group_strata(published)
    counted: dict[str, Counter[str]] = {}
    for group, strata in published_strata.items():
        for stratum in strata:
            counted.setdefault(stratum, Counter())[published_role[group]] += 1

    strata_of = _group_strata(rows)
    trained = {row.group for row in rows if row.trained_on}
    new = sorted(group for group in strata_of if group not in published_role)
    share = {stratum: shares.get(stratum, evaluation_share)
             for stratum in {s for group in new for s in strata_of[group]}}
    counted_for_rule = {stratum: {role: count for role, count in tallies.items() if role in ("train", "evaluation")}
                        for stratum, tallies in counted.items() if stratum in share}
    assigned = assign_by_share([(group, strata_of[group]) for group in new], seed, share,
                               held="evaluation", open="train", forced_open=trained, counted=counted_for_rule)

    groups: dict[str, tuple[str, str]] = {}
    for group in strata_of:
        if group in published_role:
            if group in trained and published_role[group] != "train":
                raise ValueError(f'group "{group}" has a frame marked trained_on, but its published role is '
                                 f'"{published_role[group]}"')
            groups[group] = (published_role[group], "kept")
        else:
            groups[group] = (assigned[group], "trained on" if group in trained else "assigned")
    tallies = {stratum: Counter(values) for stratum, values in counted.items()}
    for group in new:
        for stratum in strata_of[group]:
            tallies.setdefault(stratum, Counter())[groups[group][0]] += 1
    return RoleAssignment(
        rows=[dataclasses.replace(row, role=groups[row.group][0]) for row in rows],
        groups=groups,
        tallies={stratum: dict(values) for stratum, values in tallies.items()},
    )
