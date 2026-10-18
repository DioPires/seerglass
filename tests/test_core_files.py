# SPDX-License-Identifier: Apache-2.0
"""A file written in one step is either whole or untouched: the manifest writer and the helper
under it, with failures injected during the write."""

from __future__ import annotations

import os
import stat
import sys

import pytest

from cardvision import cli
from cardvision.core import files
from cardvision.core.files import write_text_atomic
from cardvision.data import manifest
from cardvision.data.schema import parse_row
from support_data import row_doc

needs_modes = pytest.mark.skipif(sys.platform == "win32", reason="file permissions are POSIX ones")


def leftovers(folder) -> list[str]:
    return sorted(path.name for path in folder.iterdir() if path.name.startswith("."))


class Breaks:
    """Stands in for a function of ``os``: it raises, as a full disk or a stopped run would."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        raise OSError(28, "No space left on device")


def test_text_is_written_and_a_file_is_replaced(tmp_path):
    path = tmp_path / "notes.txt"
    write_text_atomic(path, "first\n")
    assert path.read_text(encoding="utf-8") == "first\n"
    write_text_atomic(path, "second, with an é\n")
    assert path.read_bytes() == "second, with an é\n".encode("utf-8")
    assert leftovers(tmp_path) == []
    with pytest.raises(FileNotFoundError):
        write_text_atomic(tmp_path / "no-such-folder" / "notes.txt", "x")


@pytest.mark.parametrize("step", ["fsync", "chmod", "replace"])
def test_a_write_that_fails_leaves_the_old_file_and_no_temporary_one(tmp_path, monkeypatch, step):
    path = tmp_path / "labels.jsonl"
    path.write_text("the only copy\n", encoding="utf-8")
    broken = Breaks()
    monkeypatch.setattr(files.os, step, broken)
    with pytest.raises(OSError, match="No space left"):
        write_text_atomic(path, "a newer copy that never arrives\n")
    assert broken.calls == 1
    assert path.read_text(encoding="utf-8") == "the only copy\n"
    assert leftovers(tmp_path) == [] and [entry.name for entry in tmp_path.iterdir()] == ["labels.jsonl"]


def test_a_write_cut_short_part_way_leaves_the_old_file(tmp_path, monkeypatch):
    """The failure comes after some of the text is already on disk: the half-written text is in
    the temporary file, never in the file itself."""
    path = tmp_path / "labels.jsonl"
    path.write_text("the only copy\n", encoding="utf-8")
    seen: list[bytes] = []
    real_fsync = os.fsync

    def cut_short(descriptor):
        real_fsync(descriptor)
        (partial,) = [entry for entry in tmp_path.iterdir() if entry.name.endswith(files.PARTIAL_SUFFIX)]
        seen.append(partial.read_bytes())
        assert path.read_text(encoding="utf-8") == "the only copy\n"       # the file itself is still whole
        raise KeyboardInterrupt

    monkeypatch.setattr(files.os, "fsync", cut_short)
    with pytest.raises(KeyboardInterrupt):
        write_text_atomic(path, "a newer copy\n")
    assert seen == [b"a newer copy\n"]
    assert path.read_text(encoding="utf-8") == "the only copy\n" and leftovers(tmp_path) == []


@needs_modes
def test_a_replaced_file_keeps_its_permissions_and_a_new_one_gets_the_usual_ones(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text("old\n", encoding="utf-8")
    for mode in (0o640, 0o600, 0o664):
        os.chmod(path, mode)
        write_text_atomic(path, f"written at {mode:o}\n")
        assert stat.S_IMODE(path.stat().st_mode) == mode
    plain = tmp_path / "plain.txt"
    plain.write_text("x", encoding="utf-8")
    fresh = tmp_path / "fresh.txt"
    write_text_atomic(fresh, "x")
    assert stat.S_IMODE(fresh.stat().st_mode) == stat.S_IMODE(plain.stat().st_mode)


@needs_modes
def test_a_file_that_may_not_be_written_to_is_refused(tmp_path):
    """A plain write to a read-only file fails. Moving a new file into its place would succeed,
    because that only needs the folder; the protection is kept all the same."""
    path = tmp_path / "labels.jsonl"
    path.write_text("kept as it is\n", encoding="utf-8")
    os.chmod(path, 0o444)
    if os.access(path, os.W_OK):
        pytest.skip("this user may write to any file")
    with pytest.raises(PermissionError):
        write_text_atomic(path, "replaced\n")
    assert path.read_text(encoding="utf-8") == "kept as it is\n" and leftovers(tmp_path) == []
    with pytest.raises(PermissionError):
        manifest.write_manifest(path, rows())


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symbolic links")
def test_a_symbolic_link_is_written_through(tmp_path):
    real = tmp_path / "real" / "labels.jsonl"
    real.parent.mkdir()
    real.write_text("old\n", encoding="utf-8")
    link = tmp_path / "link.jsonl"
    try:
        link.symlink_to(real)
    except OSError:
        pytest.skip("symbolic links cannot be made here")
    write_text_atomic(link, "new\n")
    assert link.is_symlink() and real.read_text(encoding="utf-8") == "new\n"


# ------------------------------------------------------------------------------ the manifest
def rows():
    return [parse_row(row_doc(t=1.0)), parse_row(row_doc(t=2.0))]


def test_a_manifest_whose_write_fails_is_left_as_it_was(tmp_path, monkeypatch):
    path = tmp_path / "m.jsonl"
    manifest.write_manifest(path, rows()[:1])
    before = path.read_bytes()
    monkeypatch.setattr(files.os, "replace", Breaks())
    with pytest.raises(OSError):
        manifest.write_manifest(path, rows())
    assert path.read_bytes() == before and len(manifest.read_manifest(path)) == 1
    assert [entry.name for entry in tmp_path.iterdir()] == ["m.jsonl"]


@needs_modes
def test_a_manifest_keeps_its_permissions_when_it_is_rewritten(tmp_path):
    path = tmp_path / "m.jsonl"
    manifest.write_manifest(path, rows()[:1])
    os.chmod(path, 0o640)
    assert manifest.write_manifest(path, rows()) == 2
    assert stat.S_IMODE(path.stat().st_mode) == 0o640 and len(manifest.read_manifest(path)) == 2


def test_record_reachable_that_cannot_write_leaves_the_manifest_alone(tmp_path, monkeypatch, capsys):
    """``record-reachable`` rewrites a manifest in place. When the write fails, the command says
    so and every byte of the manifest is as it was."""
    from cardvision.data.extract import IndexEntry, append_index

    path = tmp_path / "labels" / "m.jsonl"
    path.parent.mkdir()
    manifest.write_manifest(path, rows())
    before = path.read_bytes()
    for row in rows():
        append_index(tmp_path / "frames", IndexEntry(platform=row.platform, video_id=row.video_id, t=row.t, status="ok",
                                                     source="download", file="frames/a.jpg"))
    monkeypatch.setattr(files.os, "replace", Breaks())
    assert cli.main(["record-reachable", str(path), "--index", str(tmp_path / "frames"), "--date", "2026-01-31"]) == 1
    assert "No space left on device" in capsys.readouterr().err
    assert path.read_bytes() == before and [entry.name for entry in path.parent.iterdir()] == ["m.jsonl"]
