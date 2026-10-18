# SPDX-License-Identifier: Apache-2.0
"""The check that a new bank only adds to an old one."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from cardvision import cli
from cardvision.bank.build import groups_path_for
from cardvision.bank.refresh import check_refresh
from cardvision.core import bank as B

OLD_IDS = ["AAA-001", "AAA-002", "AAA-002A", "AAA-003", "BBB-001"]
NAMES = {"AAA-001": "One", "AAA-002": "Two", "AAA-002A": "Two", "AAA-003": "Three", "BBB-001": "Four",
         "BBB-002": "Five", "BBB-002A": "Five", "BBB-003": "Three", "AAA-000": "Four"}


def row(card_id: str) -> np.ndarray:
    """A fixed unit row per card id."""
    seed = int.from_bytes(card_id.encode(), "big") % (2**32)
    return B.fp16_roundtrip(B.l2_normalize(np.random.default_rng(seed).normal(size=16)))


def bank(ids, version=1, **header):
    extra = {"schemaVersion": 1, "kind": "embedding-bank", "game": "toy", "locale": "en", "version": version,
             "model": "tiny", "dim": 16, "dtype": "fp16", "views": 16, "aggregation": "mean", **header}
    return B.encode_bank(ids, np.stack([row(card_id) for card_id in ids]), extra=extra)


def groups(ids):
    listed = B.groups_by_name(ids, NAMES)
    return {"schemaVersion": 1, "kind": "identity-groups", "version": 1, "rule": {"level": "name"},
            "groups": listed, "representative": B.IdentityGroups(listed, ids).representative_map()}


OLD = bank(OLD_IDS)
OLD_GROUPS = groups(OLD_IDS)


def test_appending_new_cards_passes():
    new_ids = [*OLD_IDS, "BBB-002", "BBB-002A"]
    report = check_refresh(OLD, bank(new_ids, version=2), OLD_GROUPS, groups(new_ids))
    assert report.ok and report.errors == [] and report.notes == []
    assert report.shared == 5 and report.added == ["BBB-002", "BBB-002A"] and report.appended


def test_a_bank_compared_with_a_later_copy_of_itself_passes():
    report = check_refresh(OLD, bank(OLD_IDS, version=2))
    assert report.ok and report.shared == 5 and report.added == []


def test_a_removed_card_fails():
    report = check_refresh(OLD, bank(["AAA-001", "AAA-002", "AAA-003", "CCC-001"], version=2))
    assert report.errors == ["2 card id(s) of the old bank are missing from the new one: AAA-002A, BBB-001"]
    assert report.shared == 3 and report.added == ["CCC-001"]


def test_reordered_shared_cards_fail():
    report = check_refresh(OLD, bank(["AAA-001", "AAA-003", "AAA-002", "AAA-002A", "BBB-001"], version=2))
    assert report.errors == ["the shared card ids are not in the same order: shared row 1 is AAA-002 in the old "
                             "bank and AAA-003 in the new one"]


def test_a_changed_row_fails_even_by_one_fp16_step():
    new = bank([*OLD_IDS, "BBB-002"], version=2)
    stored = B.decode_vector(new["entries"][3]["vec"])
    stored[0] = np.nextafter(stored[0], np.float16(np.inf))
    new["entries"][3]["vec"] = B.encode_vector(stored)
    report = check_refresh(OLD, new)
    assert report.errors == ["1 shared row(s) are not byte-identical: AAA-003"]


def test_a_row_of_another_size_fails():
    new = B.encode_bank(OLD_IDS, np.ones((5, 8), np.float32), extra={"kind": "embedding-bank", "version": 2})
    assert "5 shared row(s) are not byte-identical" in check_refresh(OLD, new).errors[0]


def test_another_field_of_a_shared_entry_changing_fails():
    old = copy.deepcopy(OLD)
    for entry in old["entries"]:
        entry["source"] = "art"
    new = bank([*OLD_IDS, "BBB-002"], version=2)
    for entry in new["entries"]:
        entry["source"] = "art"
    assert check_refresh(old, new).ok
    new["entries"][1]["source"] = "photo"
    assert check_refresh(old, new).errors == [
        "1 shared entr(ies) keep their vector and differ in another field: AAA-002"]


@pytest.mark.parametrize("key, value", [("model", "other"), ("views", 8), ("game", "another"), ("dim", 32),
                                         ("preprocessing", {"id": "other"})])
def test_banks_made_differently_fail(key, value):
    old = bank(OLD_IDS, preprocessing={"id": "full128"})
    new = bank(OLD_IDS, version=2, preprocessing={"id": "full128"})
    new[key] = value
    errors = check_refresh(old, new).errors
    assert len(errors) == 1 and errors[0].startswith(f'the banks were not made the same way: "{key}" is ')


def test_the_version_must_go_up():
    assert check_refresh(OLD, bank(OLD_IDS, version=1)).errors == [
        "the new bank's version 1 is not greater than the old bank's 1"]


def test_cards_added_in_the_middle_pass_with_a_note():
    report = check_refresh(OLD, bank(["AAA-001", "BBB-002", *OLD_IDS[1:]], version=2))
    assert report.ok and not report.appended
    assert report.notes == ["the added card ids are not all at the end of the new bank"]


def test_every_problem_is_reported_at_once():
    new = bank(["AAA-003", "AAA-001", "AAA-002"], version=1, model="other")
    assert len(check_refresh(OLD, new).errors) == 4  # made differently, version, removed, order


def test_a_document_that_is_not_a_bank_fails():
    assert check_refresh({"kind": "identity-groups"}, OLD).errors == [
        'the old bank is not an "embedding-bank" document']
    assert check_refresh(OLD, [1, 2]).errors == ['the new bank is not an "embedding-bank" document']
    broken = {**OLD, "entries": [{"cardId": "AAA-001", "vec": "???"}]}
    assert check_refresh(OLD, broken).errors[0].startswith("the new bank cannot be read: card 'AAA-001'")
    twice = {**OLD, "entries": [*OLD["entries"], OLD["entries"][0]]}
    assert check_refresh(twice, OLD).errors == ["the old bank has more than one row for: AAA-001"]


# ------------------------------------------------------------------------- identity groups
def test_groups_may_gain_new_cards_and_new_groups():
    # BBB-003 joins the single card AAA-003 (a new group), BBB-002 and BBB-002A form a group of their own.
    new_ids = [*OLD_IDS, "BBB-002", "BBB-002A", "BBB-003"]
    new_groups = groups(new_ids)
    assert new_groups["groups"] == [["AAA-002", "AAA-002A"], ["AAA-003", "BBB-003"], ["BBB-002", "BBB-002A"]]
    report = check_refresh(OLD, bank(new_ids, version=2), OLD_GROUPS, new_groups)
    assert report.ok and report.notes == []


def test_a_new_representative_is_a_note_not_a_failure():
    # AAA-000 shares BBB-001's name and has the smaller key, so it becomes the representative.
    new_ids = [*OLD_IDS, "AAA-000"]
    report = check_refresh(OLD, bank(new_ids, version=2), OLD_GROUPS, groups(new_ids))
    assert report.ok
    assert report.notes == ["1 old card(s) have a new representative, because a new card with a smaller key "
                            "joined their identity (e.g. BBB-001: BBB-001 -> AAA-000)"]


def test_a_split_group_fails():
    new_groups = {**OLD_GROUPS, "groups": [], "representative": {}}
    report = check_refresh(OLD, bank(OLD_IDS, version=2), OLD_GROUPS, new_groups)
    assert report.errors == ["identity group ['AAA-002', 'AAA-002A'] of the old bank is split in the new groups: "
                             "cards that were one identity must stay one"]


def test_old_cards_newly_made_one_identity_fail():
    merged = {**OLD_GROUPS, "groups": [["AAA-002", "AAA-002A"], ["AAA-001", "AAA-003"]]}
    report = check_refresh(OLD, bank(OLD_IDS, version=2), OLD_GROUPS, merged)
    assert report.errors == ["old cards ['AAA-001', 'AAA-003'] are one identity in the new groups and were not all "
                             "one identity before: a refresh may only add new cards to a group"]
    swallowed = {**OLD_GROUPS, "groups": [["AAA-001", "AAA-002", "AAA-002A"]]}
    report = check_refresh(OLD, bank(OLD_IDS, version=2), OLD_GROUPS, swallowed)
    assert len(report.errors) == 1 and "['AAA-001', 'AAA-002', 'AAA-002A'] are one identity" in report.errors[0]


def test_groups_for_one_bank_only_fail():
    report = check_refresh(OLD, bank(OLD_IDS, version=2), OLD_GROUPS, None)
    assert report.errors == ["identity groups were given for one bank only: give both, or neither"]


def test_malformed_groups_fail():
    new = bank(OLD_IDS, version=2)
    assert check_refresh(OLD, new, OLD_GROUPS, {"kind": "embedding-bank"}).errors == [
        'the new identity groups are not an "identity-groups" document']
    overlapping = {**OLD_GROUPS, "groups": [["AAA-002", "AAA-002A"], ["AAA-002", "AAA-003"]]}
    assert check_refresh(OLD, new, overlapping, OLD_GROUPS).errors[0].startswith(
        "the old identity groups cannot be read: card 'AAA-002' appears more than once")


# ----------------------------------------------------------------------------- command line
def write(path, document):
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_check_refresh_command(tmp_path, capsys):
    new_ids = [*OLD_IDS, "BBB-002", "AAA-000"]
    old_path, new_path = write(tmp_path / "old.json", OLD), write(tmp_path / "new.json", bank(new_ids, version=2))
    # Without groups files beside the banks the command says so, unless told to compare the banks only.
    assert cli.main(["check-refresh", str(old_path), str(new_path)]) == 1
    assert f"no identity groups file at {groups_path_for(old_path)}" in capsys.readouterr().err
    assert cli.main(["check-refresh", str(old_path), str(new_path), "--bank-only"]) == 0
    assert capsys.readouterr().out.strip() == (
        "ok: 5 shared rows, in the same order: 5 byte-identical, 0 re-embedded (image changed); "
        "2 card id(s) added")
    write(groups_path_for(old_path), OLD_GROUPS)
    write(groups_path_for(new_path), groups(new_ids))
    assert cli.main(["check-refresh", str(old_path), str(new_path)]) == 0
    printed = capsys.readouterr().out
    assert "2 card id(s) added; identity groups only grew" in printed and "note: 1 old card(s) have a new" in printed
    elsewhere = write(tmp_path / "g.json", groups(new_ids))
    assert cli.main(["check-refresh", str(old_path), str(new_path), "--new-groups", str(elsewhere)]) == 0


def test_check_refresh_command_fails_with_every_reason(tmp_path, capsys):
    old_path = write(tmp_path / "old.json", OLD)
    new_path = write(tmp_path / "new.json", bank(["AAA-003", "AAA-001"], version=2))
    assert cli.main(["check-refresh", str(old_path), str(new_path), "--bank-only"]) == 1
    printed = capsys.readouterr()
    assert printed.out == "" and "cardvision check-refresh: failed" in printed.err
    assert "3 card id(s) of the old bank are missing from the new one" in printed.err
    assert "the shared card ids are not in the same order" in printed.err
    assert cli.main(["check-refresh", str(old_path), str(tmp_path / "absent.json"), "--bank-only"]) == 1
    (tmp_path / "text.json").write_text("not json", encoding="utf-8")
    assert cli.main(["check-refresh", str(old_path), str(tmp_path / "text.json"), "--bank-only"]) == 1
    assert "text.json: not valid JSON" in capsys.readouterr().err


# ------------------------------------------------------------- rows that record their image
def art(card_id: str, take: int = 1) -> str:
    """The ``artV`` of a made-up image file of a card; another ``take`` is another image."""
    return B.art_version(f"{card_id} image {take}".encode())


def recording(document: dict, leave=()) -> dict:
    """A copy of a bank in which every row, bar those in ``leave``, records its image."""
    document = copy.deepcopy(document)
    for entry in document["entries"]:
        if entry["cardId"] not in leave:
            entry["artV"] = art(entry["cardId"])
    return document


def embed_again(document: dict, card_id: str, take: int | None = 2) -> dict:
    """Move one row by an fp16 step, and record another image for it unless ``take`` is None."""
    entry = next(entry for entry in document["entries"] if entry["cardId"] == card_id)
    stored = B.decode_vector(entry["vec"])
    stored[0] = np.nextafter(stored[0], np.float16(np.inf))
    entry["vec"] = B.encode_vector(stored)
    if take is not None:
        entry["artV"] = art(card_id, take)
    return document


RECORDED = recording(OLD)
LATER_IDS = [*OLD_IDS, "BBB-002"]


def test_a_row_embedded_again_passes_when_both_banks_record_different_images():
    new = embed_again(recording(bank(LATER_IDS, version=2)), "AAA-003")
    report = check_refresh(RECORDED, new)
    assert report.ok and report.errors == [] and report.notes == []
    assert report.reembedded == [("AAA-003", art("AAA-003"), art("AAA-003", 2))]
    assert (report.shared, report.identical, report.changed, report.added) == (5, 4, [], ["BBB-002"])


def test_a_recorded_change_of_image_is_accepted_even_when_the_vector_is_the_same():
    # The same picture saved again is another file: the row is embedded again and comes out the same.
    new = recording(bank(LATER_IDS, version=2))
    new["entries"][0]["artV"] = art("AAA-001", 2)
    report = check_refresh(RECORDED, new)
    assert report.ok and report.reembedded == [("AAA-001", art("AAA-001"), art("AAA-001", 2))]
    assert report.identical == 4


def test_a_changed_row_with_the_same_recorded_image_fails():
    new = embed_again(recording(bank(LATER_IDS, version=2)), "AAA-003", take=None)
    report = check_refresh(RECORDED, new)
    assert report.errors == ["1 shared row(s) are not byte-identical, and both banks record the same image (artV) "
                             "for them: AAA-003"]
    assert report.changed == ["AAA-003"] and report.reembedded == [] and report.identical == 4


def test_a_changed_row_fails_when_either_bank_records_no_image_for_it():
    new = embed_again(recording(bank(LATER_IDS, version=2)), "AAA-003")
    # The old bank records none.
    report = check_refresh(recording(OLD, leave=["AAA-003"]), new)
    assert report.errors == ["1 shared row(s) are not byte-identical, and the old bank records no image (artV) "
                             "for them: AAA-003"]
    assert report.changed == ["AAA-003"] and report.reembedded == []
    # The new bank records none.
    del new["entries"][3]["artV"]
    report = check_refresh(RECORDED, new)
    assert report.errors == ["1 shared row(s) are not byte-identical, and the new bank records no image (artV) "
                             "for them: AAA-003"]
    # Neither does: the error of a bank written before rows recorded their image.
    report = check_refresh(OLD, embed_again(bank(LATER_IDS, version=2), "AAA-003", take=None))
    assert report.errors == ["1 shared row(s) are not byte-identical: AAA-003"]


def test_the_counts_of_shared_rows_add_up():
    new = recording(bank(LATER_IDS, version=2))
    embed_again(new, "AAA-001")              # accepted: both record, and the images differ
    embed_again(new, "AAA-002", take=None)   # refused: the same image on both sides
    embed_again(new, "BBB-001")              # refused: the old bank records none for it
    report = check_refresh(recording(OLD, leave=["BBB-001"]), new)
    assert len(report.errors) == 2
    assert [row[0] for row in report.reembedded] == ["AAA-001"] and report.changed == ["AAA-002", "BBB-001"]
    assert report.identical == 2 and report.unverified == []
    assert report.shared == report.identical + len(report.reembedded) + len(report.changed) == 5
    assert report.shared + len(report.added) == len(new["entries"])


def test_a_row_may_start_to_record_its_image_and_may_not_stop():
    new = recording(bank(LATER_IDS, version=2))
    report = check_refresh(OLD, new)
    assert report.ok and report.identical == 5 and report.reembedded == []
    assert report.notes == ["5 shared row(s) keep their vector and now record their image (artV): "
                            "AAA-001, AAA-002, AAA-002A, AAA-003, BBB-001"]
    assert check_refresh(RECORDED, bank(LATER_IDS, version=2)).errors == [
        "5 shared entr(ies) keep their vector and differ in another field: "
        "AAA-001, AAA-002, AAA-002A, AAA-003, BBB-001"]


def test_a_row_embedded_again_may_not_change_in_any_other_field():
    old = recording(OLD)
    new = embed_again(recording(bank(LATER_IDS, version=2)), "AAA-003")
    for document in (old, new):
        for entry in document["entries"]:
            entry["source"] = "art"
    assert check_refresh(old, new).ok
    new["entries"][3]["source"] = "photo"
    assert check_refresh(old, new).errors == [
        "1 shared entr(ies) embedded again also differ in a field other than the vector and artV: AAA-003"]


def test_recorded_images_excuse_neither_a_new_order_nor_a_missing_card():
    reordered = recording(bank(["AAA-001", "AAA-003", "AAA-002", "AAA-002A", "BBB-001"], version=2))
    for card_id in OLD_IDS:
        embed_again(reordered, card_id)
    report = check_refresh(RECORDED, reordered)
    assert report.errors == ["the shared card ids are not in the same order: shared row 1 is AAA-002 in the old "
                             "bank and AAA-003 in the new one"]
    shorter = embed_again(recording(bank(OLD_IDS[:4], version=2)), "AAA-001")
    assert check_refresh(RECORDED, shorter).errors == [
        "1 card id(s) of the old bank are missing from the new one: BBB-001"]


def test_the_tool_that_embedded_unrecorded_rows_again_may_say_so():
    new = embed_again(recording(bank(LATER_IDS, version=2)), "AAA-003")
    old = recording(OLD, leave=["AAA-003"])
    assert not check_refresh(old, new).ok
    report = check_refresh(old, new, reembedded_unrecorded=["AAA-003", "AAA-001"])
    assert report.ok and report.unverified == ["AAA-003"] and report.changed == [] and report.reembedded == []
    assert report.shared == report.identical + len(report.unverified) == 5
    # It excuses nothing else: a row that records the same image on both sides must not move.
    assert not check_refresh(RECORDED, embed_again(recording(bank(LATER_IDS, version=2)), "AAA-003", take=None),
                             reembedded_unrecorded=["AAA-003"]).ok


def test_a_malformed_record_of_an_image_fails():
    new = recording(bank(LATER_IDS, version=2))
    new["entries"][1]["artV"] = "ABCDEF0123456789"  # upper case
    assert check_refresh(RECORDED, new).errors == [
        "the new bank cannot be read: card 'AAA-002' (entries[1]): artV 'ABCDEF0123456789' is not 16 lower-case "
        "hexadecimal digits"]


def test_check_refresh_command_lists_the_rows_embedded_again(tmp_path, capsys):
    new = embed_again(recording(bank([*LATER_IDS, "BBB-002A"], version=2)), "AAA-003")
    old_path, new_path = write(tmp_path / "old.json", RECORDED), write(tmp_path / "new.json", new)
    assert cli.main(["check-refresh", str(old_path), str(new_path), "--bank-only"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "ok: 5 shared rows, in the same order: 4 byte-identical, 1 re-embedded (image changed); 2 card id(s) added",
        "re-embedded (image changed) (1):",
        f"  AAA-003  artV {art('AAA-003')} -> {art('AAA-003', 2)}",
    ]
    # The same change with no image recorded in the old bank is refused, and the row is named.
    write(old_path, OLD)
    assert cli.main(["check-refresh", str(old_path), str(new_path), "--bank-only"]) == 1
    printed = capsys.readouterr()
    assert printed.out == ""
    assert "1 shared row(s) are not byte-identical, and the old bank records no image (artV) for them: AAA-003" \
        in printed.err
