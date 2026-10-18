# SPDX-License-Identifier: Apache-2.0
"""The seeded role assignment: deterministic, whole groups, the shares it is asked for."""

from __future__ import annotations

import dataclasses
import re
from collections import Counter

import pytest

from cardvision import cli
from cardvision.data import manifest, roles
from cardvision.data.schema import Row, row_problems
from cardvision.data.validate import validate_manifests

SEED = 20260101


def make_row(video: int, group: str, source_type: str = "physical", role: str = "extra", t: float = 1.0,
             trained_on: bool = False) -> Row:
    return Row(platform="youtube", video_id=f"EXAMPLE{video:04d}", t=t, frame_width=1920, frame_height=1080,
               source_type=source_type, group=group, role=role, trained_on=trained_on,
               completeness="exhaustive-boxes", credit="Example Labeler", boxes=())


# --------------------------------------------------------------------------- building groups
def test_items_that_share_a_key_are_one_group_directly_or_through_others():
    keys = [
        {"channel-a"},                 # 0
        {"channel-b", "event-x"},      # 1
        {"channel-c"},                 # 2
        {"channel-d", "event-x"},      # 3: tied to 1 by the event
        {"channel-a", "event-y"},      # 4: tied to 0 by the channel
        set(),                         # 5: nothing to tie it
        {"channel-d"},                 # 6: tied to 3, and so to 1
        {"event-y"},                   # 7: tied to 4, and so to 0
    ]
    assert roles.tie_groups(keys) == [[0, 4, 7], [1, 3, 6], [2], [5]]
    assert roles.tie_groups([]) == []
    assert roles.tie_groups([set(), set()]) == [[0], [1]]


# ------------------------------------------------------------------------------ by share
def share_groups(count: int) -> list[tuple[str, list[str]]]:
    return [(f"g{index:02d}", ["physical"] if index % 3 else ["digital", "physical"]) for index in range(count)]


def test_assignment_by_share_is_pinned_for_a_seed():
    assert roles.assign_by_share(share_groups(12), SEED, {"physical": 0.4, "digital": 0.5}) == {
        "g00": "evaluation", "g01": "evaluation", "g02": "train", "g03": "train", "g04": "evaluation",
        "g05": "train", "g06": "evaluation", "g07": "evaluation", "g08": "train", "g09": "train",
        "g10": "train", "g11": "train",
    }


def test_assignment_by_share_is_deterministic_and_depends_on_the_seed():
    groups = share_groups(40)
    share = {"physical": 0.4, "digital": 0.5}
    first = roles.assign_by_share(groups, SEED, share)
    assert roles.assign_by_share(list(groups), SEED, dict(share)) == first
    assert any(roles.assign_by_share(groups, seed, share) != first for seed in range(5))
    # The order in which groups are visited is the seeded shuffle, and it is exposed.
    assert list(first) == roles.visiting_order([key for key, _ in groups], SEED)
    assert roles.visiting_order([f"g{index}" for index in range(6)], 5) == ["g1", "g0", "g3", "g5", "g2", "g4"]
    assert roles.shuffled(list("abcdefgh"), 1234) == ["b", "d", "c", "e", "f", "g", "a", "h"]


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize(("count", "share"), [(100, 0.4), (30, 0.5), (57, 0.65), (10, 0.0), (10, 1.0), (1, 0.4), (3, 0.34)])
def test_one_stratum_ends_at_its_share(seed, count, share):
    groups = [(f"g{index:03d}", ["physical"]) for index in range(count)]
    assigned = roles.assign_by_share(groups, seed, {"physical": share})
    assert set(assigned) == {key for key, _ in groups}
    held = sum(1 for role in assigned.values() if role == "evaluation")
    assert abs(held - share * count) <= 1


@pytest.mark.parametrize("seed", range(6))
def test_several_strata_each_end_near_their_share(seed):
    shares = {"physical/tournament": 0.4, "physical/webcam": 0.4, "digital": 0.6}
    groups = ([(f"t{index:02d}", ["physical/tournament"]) for index in range(40)]
              + [(f"w{index:02d}", ["physical/webcam"]) for index in range(20)]
              + [(f"d{index:02d}", ["digital"]) for index in range(30)])
    assigned = roles.assign_by_share(groups, seed, shares)
    for prefix, stratum, count in (("t", "physical/tournament", 40), ("w", "physical/webcam", 20), ("d", "digital", 30)):
        held = sum(1 for key, role in assigned.items() if key.startswith(prefix) and role == "evaluation")
        assert abs(held - shares[stratum] * count) <= 1


def test_forced_groups_stay_open_and_still_count():
    groups = [(f"g{index}", ["physical"]) for index in range(10)]
    forced = {"g0", "g1", "g2", "g3", "g4", "g5", "g6"}
    assigned = roles.assign_by_share(groups, SEED, {"physical": 0.5}, forced_open=forced)
    assert all(assigned[key] == "train" for key in forced)
    # Seven of ten are held open, so the three left all go to the other side.
    assert Counter(assigned.values()) == {"train": 7, "evaluation": 3}


def test_groups_that_already_have_roles_count_towards_the_shares():
    new = [(f"n{index}", ["physical"]) for index in range(4)]
    crowded = roles.assign_by_share(new, SEED, {"physical": 0.5}, counted={"physical": {"evaluation": 6, "train": 2}})
    assert set(crowded.values()) == {"train"}
    starved = roles.assign_by_share(new, SEED, {"physical": 0.5}, counted={"physical": {"train": 6, "evaluation": 2}})
    assert set(starved.values()) == {"evaluation"}
    balanced = roles.assign_by_share(new, SEED, {"physical": 0.5}, counted={"physical": {"train": 3, "evaluation": 3}})
    assert Counter(balanced.values()) == {"train": 2, "evaluation": 2}


def test_other_role_names_and_a_missing_share():
    assigned = roles.assign_by_share(share_groups(6), SEED, {"physical": 0.5, "digital": 0.5}, held="held", open="open")
    assert set(assigned.values()) == {"held", "open"}
    with pytest.raises(ValueError, match="no share given for the stratum digital"):
        roles.assign_by_share(share_groups(6), SEED, {"physical": 0.5})


def test_a_second_split_in_shuffled_order():
    order = ["g3", "g1", "g4", "g0", "g2"]
    assert roles.split_in_order(order, 2 / 3, "train", "held") == {
        "g3": "train", "g1": "train", "g4": "train", "g0": "held", "g2": "held"}
    # Rounding goes to the even number: half of three is two, half of five is two.
    assert list(roles.split_in_order(order[:3], 0.5, "a", "b").values()) == ["a", "a", "b"]
    assert list(roles.split_in_order(order, 0.5, "a", "b").values()) == ["a", "a", "b", "b", "b"]
    assert roles.split_in_order([], 0.5, "a", "b") == {}


# ------------------------------------------------------------------------------- by quota
def test_assignment_by_quota_is_pinned_for_a_seed():
    strata = {f"g{index:02d}": ("physical" if index % 2 else "digital") for index in range(10)}
    quota = {"digital": [("evaluation", 2)], "physical": [("evaluation", 1), ("extra", 1)]}
    assert roles.assign_by_quota(strata, SEED, quota) == {
        "g00": "train", "g01": "train", "g02": "train", "g03": "train", "g04": "evaluation",
        "g05": "evaluation", "g06": "evaluation", "g07": "extra", "g08": "train", "g09": "train",
    }


@pytest.mark.parametrize("seed", range(6))
def test_quotas_are_filled_exactly_per_stratum(seed):
    strata = {f"g{index:02d}": (("physical", "a") if index < 9 else ("physical", "b") if index < 12 else ("digital", "a"))
              for index in range(20)}
    quota = {("physical", "a"): [("held", 3), ("evaluation", 2)], ("physical", "b"): [("held", 5)]}
    assigned = roles.assign_by_quota(strata, seed, quota, rest="train")
    assert roles.assign_by_quota(dict(reversed(list(strata.items()))), seed, quota, rest="train") == assigned
    count = Counter((strata[group], role) for group, role in assigned.items())
    assert count[(("physical", "a"), "held")] == 3 and count[(("physical", "a"), "evaluation")] == 2
    assert count[(("physical", "a"), "train")] == 4
    assert count[(("physical", "b"), "held")] == 3  # a quota larger than the stratum takes all of it
    assert count[(("digital", "a"), "train")] == 8  # no quota: everything is the rest


# ------------------------------------------------------------------------- pseudonymous ids
def test_group_ids_hide_what_they_name():
    first = roles.pseudonymous_group_id("some channel", "a salt")
    assert first == roles.pseudonymous_group_id("some channel", "a salt")
    assert first != roles.pseudonymous_group_id("some channel", "another salt")
    assert first != roles.pseudonymous_group_id("another channel", "a salt")
    assert re.fullmatch(r"g[0-9a-f]{10}", first) and "channel" not in first
    assert len(roles.pseudonymous_group_id("x", "s", length=16)) == 17
    assert row_problems(make_row(1, first).to_json()) == []
    with pytest.raises(ValueError, match="salt"):
        roles.pseudonymous_group_id("some channel", "")


# ---------------------------------------------------------------------------- manifest rows
def offered_rows() -> list[Row]:
    rows = []
    for index in range(12):
        source = "digital" if index % 4 == 0 else "physical/tournament"
        for frame in range(3):
            rows.append(make_row(index, f"g-new-{index:02d}", source, t=float(frame + 1)))
    # One group holds two videos.
    rows.append(make_row(50, "g-new-03", "physical/tournament"))
    return rows


def test_every_group_gets_one_role_and_every_row_its_groups():
    rows = offered_rows()
    result = roles.assign_manifest_roles(rows, SEED, 0.4)
    assert len(result.rows) == len(rows)
    assert [dataclasses.replace(row, role="extra") for row in result.rows] == rows  # nothing but the role changed
    by_group: dict[str, set[str]] = {}
    for row in result.rows:
        by_group.setdefault(row.group, set()).add(row.role)
    assert all(len(found) == 1 for found in by_group.values())
    assert {group: next(iter(found)) for group, found in by_group.items()} == {g: role for g, (role, _) in result.groups.items()}
    assert {how for _, how in result.groups.values()} == {"assigned"}
    assert set(role for role, _ in result.groups.values()) == {"train", "evaluation"}
    assert result.tallies["digital"] == {"evaluation": 1, "train": 2}
    assert result.tallies["physical/tournament"] in ({"evaluation": 4, "train": 5}, {"evaluation": 3, "train": 6})


def test_the_role_a_row_arrives_with_changes_nothing():
    rows = offered_rows()
    wishful = [dataclasses.replace(row, role="evaluation") for row in rows]
    modest = [dataclasses.replace(row, role="train") for row in rows]
    expected = roles.assign_manifest_roles(rows, SEED, 0.4)
    assert roles.assign_manifest_roles(wishful, SEED, 0.4).rows == expected.rows
    assert roles.assign_manifest_roles(modest, SEED, 0.4).rows == expected.rows
    # Nor does the order of the rows.
    shuffled = roles.shuffled(rows, 99)
    assert roles.assign_manifest_roles(shuffled, SEED, 0.4).groups == expected.groups


def test_published_groups_keep_their_roles_and_count():
    published = [make_row(100 + index, f"g-old-{index:02d}", "physical/tournament", role="evaluation") for index in range(6)]
    published += [make_row(200, "g-old-extra", "physical/tournament", role="extra")]
    rows = [make_row(index, f"g-new-{index:02d}", "physical/tournament") for index in range(4)]
    rows += [make_row(300, "g-old-02", "physical/tournament", role="train"), make_row(301, "g-old-extra", "physical/tournament")]
    result = roles.assign_manifest_roles(rows, SEED, 0.5, published=published)
    assert result.groups["g-old-02"] == ("evaluation", "kept") and result.groups["g-old-extra"] == ("extra", "kept")
    # Six published groups already evaluate, so the four new ones all train.
    assert [result.groups[f"g-new-{index:02d}"] for index in range(4)] == [("train", "assigned")] * 4
    assert result.tallies["physical/tournament"] == {"evaluation": 6, "extra": 1, "train": 4}
    assert [row.role for row in result.rows[-2:]] == ["evaluation", "extra"]
    with pytest.raises(ValueError, match="two roles"):
        roles.assign_manifest_roles(rows, SEED, 0.5, published=[*published, make_row(400, "g-old-00", role="train")])


def test_a_group_that_trained_a_released_model_stays_in_train():
    rows = [make_row(index, f"g-new-{index:02d}") for index in range(6)]
    rows[2] = dataclasses.replace(rows[2], role="train", trained_on=True)
    for seed in range(10):
        result = roles.assign_manifest_roles(rows, seed, 1.0)
        assert result.groups["g-new-02"] == ("train", "trained on")
        assert [role for group, (role, _) in result.groups.items() if group != "g-new-02"] == ["evaluation"] * 5
        assert all(row_problems(row.to_json()) == [] for row in result.rows)
    published = [make_row(100, "g-new-02", role="evaluation")]
    with pytest.raises(ValueError, match="trained_on"):
        roles.assign_manifest_roles(rows, SEED, 0.5, published=published)


def test_a_share_can_differ_by_source_type():
    rows = [make_row(index, f"g-p-{index:02d}", "physical") for index in range(10)]
    rows += [make_row(100 + index, f"g-d-{index:02d}", "digital") for index in range(10)]
    result = roles.assign_manifest_roles(rows, SEED, 0.0, shares={"digital": 1.0})
    assert result.tallies == {"physical": {"train": 10}, "digital": {"evaluation": 10}}


# ------------------------------------------------------------------------- the command line
def test_the_command_writes_a_valid_manifest_with_roles(tmp_path, capsys):
    offered = tmp_path / "offered.jsonl"
    manifest.write_manifest(offered, offered_rows())
    out = tmp_path / "out" / "assigned.jsonl"
    out.parent.mkdir()
    arguments = ["assign-roles", str(offered), "--seed", str(SEED), "--evaluation-share", "0.4", "--out", str(out)]
    assert cli.main(arguments) == 0
    printed = capsys.readouterr().out
    assert f"seed {SEED}; evaluation share 0.4" in printed and "g-new-00  " in printed and "wrote 37 row(s)" in printed
    assert manifest.read_manifest(out) == roles.assign_manifest_roles(offered_rows(), SEED, 0.4).rows
    assert validate_manifests([out]).ok

    assert cli.main(arguments) == 1
    assert "pass --overwrite" in capsys.readouterr().err
    assert cli.main([*arguments, "--overwrite", "--share", "digital=1", "--published", str(out)]) == 0
    capsys.readouterr()
    # Every group was already published with a role, so nothing moves.
    assert manifest.read_manifest(out) == roles.assign_manifest_roles(offered_rows(), SEED, 0.4).rows


def test_the_command_creates_the_folder_of_its_output(tmp_path, capsys):
    offered = tmp_path / "offered.jsonl"
    manifest.write_manifest(offered, offered_rows())
    out = tmp_path / "not" / "yet" / "there" / "assigned.jsonl"
    assert not out.parent.exists()
    assert cli.main(["assign-roles", str(offered), "--seed", str(SEED), "--evaluation-share", "0.4", "--out", str(out)]) == 0
    assert f"wrote 37 row(s) to {out}" in capsys.readouterr().out
    assert manifest.read_manifest(out) == roles.assign_manifest_roles(offered_rows(), SEED, 0.4).rows
    # A run that is refused creates nothing.
    refused = tmp_path / "never" / "assigned.jsonl"
    assert cli.main(["assign-roles", str(tmp_path / "absent.jsonl"), "--seed", "1", "--evaluation-share", "0.5",
                     "--out", str(refused)]) == 1
    assert cli.main(["assign-roles", str(offered), "--seed", "1", "--evaluation-share", "0.5", "--share", "digital",
                     "--out", str(refused)]) == 1
    capsys.readouterr()
    assert not refused.parent.exists()


def test_the_command_refuses_bad_arguments(tmp_path, capsys):
    offered = tmp_path / "offered.jsonl"
    manifest.write_manifest(offered, offered_rows())
    base = ["assign-roles", str(offered), "--seed", "1", "--out", str(tmp_path / "out.jsonl")]
    with pytest.raises(SystemExit):
        cli.main(base)  # no share
    with pytest.raises(SystemExit):
        cli.main([*base, "--evaluation-share", "1.5"])
    capsys.readouterr()
    assert cli.main([*base, "--evaluation-share", "0.5", "--share", "digital"]) == 1
    assert "expected SOURCE_TYPE=SHARE" in capsys.readouterr().err
    assert cli.main(["assign-roles", str(tmp_path / "absent.jsonl"), "--seed", "1", "--evaluation-share", "0.5",
                     "--out", str(tmp_path / "out.jsonl")]) == 1
    assert not (tmp_path / "out.jsonl").exists()
