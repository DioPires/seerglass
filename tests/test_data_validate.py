# SPDX-License-Identifier: Apache-2.0
"""The manifest validator: every way a contributed manifest can be refused."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import pytest

from cardvision import cli
from cardvision.data.validate import file_problem, folder_problems, manifest_files, validate_manifests
from support_data import (
    OTHER_VIDEO_ID,
    VIDEO_ID,
    box_doc,
    plain_box_doc,
    point_doc,
    points_row_doc,
    row_doc,
    write_lines,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "data" / "example.manifest.jsonl"
TEMPLATE_CATALOG = ROOT / "games" / "_template" / "catalog.json"

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + bytes(20)
JPEG_BYTES = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + bytes(20)


def check(tmp_path, docs, **options):
    return validate_manifests([write_lines(tmp_path / "labels" / "m.jsonl", docs)], **options)


def messages(report) -> str:
    return "\n".join(str(problem) for problem in report.problems)


# ------------------------------------------------------------------------------ valid input
def test_the_example_manifest_passes_with_and_without_a_catalog():
    report = validate_manifests([EXAMPLE], catalog_ids={"SET-001", "SET-002"})
    assert report.ok and report.rows == 4 and report.files == [str(EXAMPLE)]
    assert validate_manifests([EXAMPLE]).ok


def test_the_data_folder_of_the_repository_holds_text_only():
    assert folder_problems(ROOT / "data") == []


def test_validation_opens_no_network_connection(monkeypatch, tmp_path):
    def refuse(*args, **kwargs):
        raise AssertionError("the validator tried to open a network connection")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    assert cli.main(["validate-manifest", str(EXAMPLE), "--catalog", str(TEMPLATE_CATALOG)]) == 0
    assert check(tmp_path, [row_doc(role="judge")]).codes() == ["schema"]


# ---------------------------------------------------------------------------- broken rows
def test_a_line_that_is_not_json_is_refused(tmp_path):
    report = check(tmp_path, [row_doc(), "{broken", '{"t": NaN}', row_doc(t=2.0)])
    assert report.codes() == ["json", "json"] and report.rows == 2
    assert [problem.line for problem in report.problems] == [2, 3]


def test_a_manifest_without_rows_is_refused(tmp_path):
    report = check(tmp_path, [])
    assert report.codes() == ["schema"] and "holds no row" in messages(report)


@pytest.mark.parametrize(
    ("doc", "expected"),
    [
        (row_doc(drop=("credit",)), 'missing "credit"'),
        (row_doc(drop=("group",)), 'missing "group"'),
        (row_doc(drop=("trained_on",)), 'missing "trained_on"'),
        (row_doc(t=-3), "below the minimum"),
        (row_doc(frame_width=100000), "above the maximum"),
        (row_doc(note="free text"), 'unknown field "note"'),
        (row_doc(role="evaluation", trained_on=True), "only a train row can have trained"),
        (row_doc(drop=("boxes",)), 'missing "boxes"'),
        (points_row_doc(boxes=[box_doc()], drop=()), "boxes: must have at most 0"),
        (points_row_doc(points=[]), "points: must have at least 1"),
        (row_doc(boxes=[box_doc(quad=[[480, 270], [672, 270], [672, 540], [480, 540]])]), "above the maximum"),
        (row_doc(boxes=[plain_box_doc("back", card_id="SET-001")]), "is not allowed here"),
        (points_row_doc(points=[point_doc(x=1.5)]), "above the maximum"),
    ],
)
def test_a_row_outside_the_schema_is_refused(tmp_path, doc, expected):
    report = check(tmp_path, [doc])
    assert set(report.codes()) == {"schema"} and expected in messages(report) and report.rows == 0
    assert ":1: [schema]" in messages(report)


@pytest.mark.parametrize(
    ("doc", "expected"),
    [
        (row_doc(t=1.00001), "more than three decimals"),
        (row_doc(reachable=True, reachable_checked="2026-13-01"), "not a calendar date"),
        (row_doc(boxes=[box_doc(quad=[[0.25, 0.25], [0.25, 0.5], [0.35, 0.5], [0.35, 0.25]])]), "counter-clockwise"),
        (row_doc(boxes=[box_doc(quad=[[0.25, 0.25], [0.35, 0.5], [0.35, 0.25], [0.25, 0.5]])]), "convex"),
        (row_doc(boxes=[box_doc(quad=[[1.2, 0.2], [1.4, 0.2], [1.4, 0.6], [1.2, 0.6]])]), "wholly outside"),
        (row_doc(boxes=[box_doc(), box_doc()]), "repeats an earlier box"),
        (row_doc(boxes=[box_doc(id="a"), plain_box_doc(id="a")]), "used twice"),
    ],
)
def test_a_row_that_breaks_a_rule_beyond_the_schema_is_refused(tmp_path, doc, expected):
    report = check(tmp_path, [doc])
    assert report.codes() == ["row"] and expected in messages(report)


# --------------------------------------------------------------------------- across the rows
def test_a_row_given_twice_is_refused_within_a_file_and_across_files(tmp_path):
    report = check(tmp_path, [row_doc(), row_doc(t=13.0), row_doc()])
    assert report.codes() == ["duplicate-row"] and report.problems[0].line == 3
    assert "is already given at" in messages(report) and "m.jsonl:1" in messages(report)

    first = write_lines(tmp_path / "set" / "a.jsonl", [row_doc()])
    second = write_lines(tmp_path / "set" / "b.jsonl", [row_doc(t=12.517)])
    report = validate_manifests([first, second])
    assert report.codes() == ["duplicate-row"] and report.problems[0].file == str(second)
    # The same time in another video, or on the other platform, is another row.
    assert check(tmp_path, [row_doc(), row_doc(video_id=OTHER_VIDEO_ID, group="g-test-02")]).ok


def test_a_group_split_across_roles_is_refused(tmp_path):
    report = check(tmp_path, [row_doc(role="train"), row_doc(t=30.0, role="evaluation"),
                              row_doc(video_id=OTHER_VIDEO_ID, role="train")])
    assert report.codes() == ["group"] and report.problems[0].line == 2
    assert "never split across roles" in messages(report)
    across = validate_manifests([write_lines(tmp_path / "x" / "a.jsonl", [row_doc(role="train")]),
                                 write_lines(tmp_path / "x" / "b.jsonl", [row_doc(t=1.0, role="extra")])])
    assert across.codes() == ["group"]


def test_a_video_in_two_groups_is_refused(tmp_path):
    report = check(tmp_path, [row_doc(), row_doc(t=30.0, group="g-test-02")])
    assert report.codes() == ["group"] and "belongs to one recording group" in messages(report)
    # Two videos in one group is what a group is for.
    assert check(tmp_path, [row_doc(), row_doc(video_id=OTHER_VIDEO_ID)]).ok


def test_a_source_type_must_be_listed(tmp_path):
    report = check(tmp_path, [row_doc(source_type="digital/client-a")])
    assert report.codes() == ["source-type"] and "not a listed source type" in messages(report)
    assert check(tmp_path, [row_doc(source_type="digital/client-a")], source_types=("physical", "digital/client-a")).ok
    assert check(tmp_path, [row_doc(source_type="physical/webcam")]).ok


def test_a_card_id_must_be_in_the_catalog_when_one_is_given(tmp_path):
    docs = [row_doc(boxes=[box_doc(), box_doc(card_id="SET-404", quad=[[0.5, 0.5], [0.6, 0.5], [0.6, 0.7], [0.5, 0.7]])]),
            points_row_doc(video_id=OTHER_VIDEO_ID, group="g-test-02", points=[point_doc(card_id="SET-405")])]
    report = check(tmp_path, docs, catalog_ids={"SET-001"})
    assert report.codes() == ["unknown-card", "unknown-card"]
    assert 'boxes[1].card_id: "SET-404" is not in the catalog' in messages(report)
    assert 'points[0].card_id: "SET-405" is not in the catalog' in messages(report)
    assert check(tmp_path, docs).ok
    assert check(tmp_path, docs, catalog_ids={"SET-001", "SET-404", "SET-405"}).ok


def test_a_contribution_cannot_claim_a_role_or_a_check(tmp_path):
    offered = row_doc(role="extra")
    assert check(tmp_path, [offered], contribution=True).ok
    assert check(tmp_path, [row_doc(role="evaluation")], contribution=True).codes() == ["contribution"]
    report = check(tmp_path, [row_doc(role="train", trained_on=True, reachable=True, reachable_checked="2026-01-01")],
                   contribution=True)
    assert report.codes() == ["contribution"] * 3
    assert 'has the role "extra" until the seeded rule assigns one' in messages(report)
    # Outside contribution mode the same rows are fine.
    assert check(tmp_path, [row_doc(role="train", trained_on=True)]).ok


def test_a_recording_of_your_own_is_valid_and_cannot_be_offered(tmp_path, capsys):
    """A manifest elsewhere may use the platform ``local`` freely. A row offered to the project
    may not: nobody else can obtain the video."""
    own = row_doc(platform="local", video_id="kitchen-table", role="extra", group="g-own-01")
    mixed = [own, row_doc(role="extra")]
    assert check(tmp_path, mixed).ok
    report = check(tmp_path, mixed, contribution=True)
    assert report.codes() == ["contribution"] and report.problems[0].line == 1
    assert 'platform is "local"' in messages(report) and "nobody else can obtain the video" in messages(report)
    # A role and a time are compared across platforms as for any other row.
    assert check(tmp_path, [own, dict(own)]).codes() == ["duplicate-row"]
    assert check(tmp_path, [own, dict(own, t=13.0, role="train")]).codes() == ["group"]
    path = write_lines(tmp_path / "own" / "m.jsonl", [own])
    assert cli.main(["validate-manifest", str(path)]) == 0
    assert "valid: 1 row(s)" in capsys.readouterr().out
    assert cli.main(["validate-manifest", str(path), "--contribution"]) == 1
    assert "[contribution] platform is \"local\": a recording of your own cannot be offered" in capsys.readouterr().err


def test_the_published_manifests_name_no_recording_of_one_s_own():
    for path in sorted((ROOT / "data").glob("*.jsonl")):
        assert all(json.loads(line)["platform"] != "local" for line in path.read_text(encoding="utf-8").splitlines()), path.name


# ------------------------------------------------------------------- files beside a manifest
def bad_file(folder: Path, name: str, data: bytes) -> Path:
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.mark.parametrize(
    ("name", "data", "expected"),
    [
        ("frame.png", PNG_BYTES, "suffix .png"),
        ("frame.JPG", JPEG_BYTES, "suffix .jpg"),
        ("drawing.svg", b"<" + b"svg xmlns='http://www.w3.org/2000/svg'/>", "suffix .svg"),
        ("clip.mp4", b"\x00\x00\x00\x18ftypmp42", "suffix .mp4"),
        ("crops.npy", b"\x93NUMPY", "suffix .npy"),
        ("notes.txt", PNG_BYTES, "holds PNG image data"),
        ("labels.json", JPEG_BYTES, "holds JPEG image data"),
        ("more.jsonl.bak", b"GIF89a" + bytes(10), "holds GIF image data"),
        ("picture", b"RIFF\x10\x00\x00\x00WEBPVP8 ", "holds WebP image data"),
        ("movie", b"\x00\x00\x00\x18ftypisom", "holds image or video data"),
        ("archive.bin", b"PK\x03\x04\x00\x00data", "not a text file (it holds a zero byte)"),
        ("latin.txt", "café".encode("latin-1"), "not valid UTF-8"),
        ("page.md", b"![x](data:" + b"image/png;base64,iVBORw0KGgo=)", "embeds an image inline"),
        ("page.html", b"<img src='data:image/svg+xml,%3Csvg%3E'>", "embeds an image inline"),
        ("deep/er/frame.webp", b"RIFF", "suffix .webp"),
    ],
)
def test_anything_that_is_not_plain_text_beside_a_manifest_is_refused(tmp_path, name, data, expected):
    manifest_path = write_lines(tmp_path / "labels" / "m.jsonl", [row_doc()])
    intruder = bad_file(tmp_path / "labels", name, data)
    report = validate_manifests([manifest_path])
    assert report.codes() == ["file"] and report.problems[0].file == str(intruder)
    assert expected in messages(report) and report.rows == 1
    assert validate_manifests([manifest_path], check_folders=False).ok


def test_text_files_beside_a_manifest_are_accepted(tmp_path):
    manifest_path = write_lines(tmp_path / "labels" / "m.jsonl", [row_doc()])
    bad_file(tmp_path / "labels", "README.md", "# Notes\n\nCafé ✓\n".encode())
    bad_file(tmp_path / "labels", "schema.json", b"{}\n")
    bad_file(tmp_path / "labels", "empty.txt", b"")
    bad_file(tmp_path / "labels", "about.md", b"No picture may be embedded as `data:image/...` either.\n")
    bad_file(tmp_path / "labels", ".git/objects/pack", PNG_BYTES)
    assert validate_manifests([manifest_path]).ok
    assert file_problem(tmp_path / "labels" / "README.md") is None


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symbolic links")
def test_a_symbolic_link_beside_a_manifest_is_refused(tmp_path):
    manifest_path = write_lines(tmp_path / "labels" / "m.jsonl", [row_doc()])
    target = bad_file(tmp_path, "elsewhere.txt", b"text")
    try:
        (tmp_path / "labels" / "link.txt").symlink_to(target)
    except OSError:
        pytest.skip("symbolic links cannot be made here")
    report = validate_manifests([manifest_path])
    assert report.codes() == ["file"] and "symbolic link" in messages(report)


def test_a_manifest_that_is_itself_an_image_is_refused(tmp_path):
    path = bad_file(tmp_path / "labels", "m.jsonl", PNG_BYTES)
    report = validate_manifests([path])
    assert set(report.codes()) == {"file"} and report.rows == 0


# ------------------------------------------------------------------------------ which files
def test_a_folder_stands_for_every_manifest_in_it(tmp_path):
    first = write_lines(tmp_path / "set" / "a.jsonl", [row_doc()])
    second = write_lines(tmp_path / "set" / "more" / "b.jsonl", [row_doc(t=1.0)])
    assert manifest_files([tmp_path / "set"]) == [first, second]
    assert manifest_files([first, tmp_path / "set"]) == [first, second]
    assert validate_manifests([tmp_path / "set"]).rows == 2
    with pytest.raises(FileNotFoundError):
        manifest_files([tmp_path / "absent.jsonl"])
    (tmp_path / "nothing").mkdir()
    with pytest.raises(ValueError, match="holds no .jsonl file"):
        manifest_files([tmp_path / "nothing"])
    with pytest.raises(ValueError, match="a manifest is a .jsonl file"):
        manifest_files([bad_file(tmp_path, "labels.json", b"{}")])


# ------------------------------------------------------------------------- the command line
def test_the_command_reports_every_problem_and_fails(tmp_path, capsys):
    path = write_lines(tmp_path / "labels" / "m.jsonl", [
        row_doc(drop=("credit",)),
        row_doc(t=2.0, boxes=[box_doc(card_id="SET-404")]),
    ])
    bad_file(tmp_path / "labels", "frame.png", PNG_BYTES)
    assert cli.main(["validate-manifest", str(path), "--catalog", str(TEMPLATE_CATALOG)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "cardvision validate-manifest: failed" in captured.err
    assert "frame.png: [file]" in captured.err
    assert 'm.jsonl:1: [schema] row: missing "credit"' in captured.err
    assert "m.jsonl:2: [unknown-card]" in captured.err
    assert "3 problem(s) in 1 file(s)" in captured.err


def test_the_command_accepts_a_valid_manifest(capsys):
    assert cli.main(["validate-manifest", str(EXAMPLE), "--catalog", str(TEMPLATE_CATALOG)]) == 0
    assert "valid: 4 row(s) in 1 file(s); card ids checked against the catalog" in capsys.readouterr().out
    assert cli.main(["validate-manifest", str(EXAMPLE)]) == 0
    assert "card ids not checked" in capsys.readouterr().out


def test_the_command_reads_a_sources_file_and_checks_contributions(tmp_path, capsys):
    path = write_lines(tmp_path / "labels" / "m.jsonl", [row_doc(source_type="digital/client-a", role="extra")])
    assert cli.main(["validate-manifest", str(path)]) == 1
    assert "[source-type]" in capsys.readouterr().err
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps({"sourceTypes": ["digital/client-a"]}))
    assert cli.main(["validate-manifest", str(path), "--sources", str(sources), "--contribution"]) == 0
    capsys.readouterr()
    claimed = write_lines(tmp_path / "claimed" / "m.jsonl", [row_doc(role="evaluation")])
    assert cli.main(["validate-manifest", str(claimed), "--contribution"]) == 1
    assert "[contribution]" in capsys.readouterr().err


def test_the_command_fails_cleanly_on_a_missing_file_or_a_broken_catalog(tmp_path, capsys):
    assert cli.main(["validate-manifest", str(tmp_path / "absent.jsonl")]) == 1
    assert "no such file or folder" in capsys.readouterr().err
    broken = tmp_path / "catalog.json"
    broken.write_text("{}")
    assert cli.main(["validate-manifest", str(EXAMPLE), "--catalog", str(broken)]) == 1
    assert "validate-manifest: failed" in capsys.readouterr().err


def test_the_test_video_id_is_a_placeholder():
    assert VIDEO_ID.startswith("EXAMPLE") and OTHER_VIDEO_ID.startswith("EXAMPLE")
