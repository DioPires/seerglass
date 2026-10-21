# SPDX-License-Identifier: Apache-2.0
"""The image fetcher (``cardvision.mtg.images``) with a fake server: what is fetched, what is
kept from an earlier run, retries, refusals and tombstones. Nothing reaches the network."""

from __future__ import annotations

import io
import sqlite3
import urllib.error

import pytest

from cardvision.mtg import images as I
from cardvision.mtg.__main__ import main

JPEG = b"\xff\xd8\xff\xe0" + b"x" * 100


class NoWait:
    def acquire(self) -> None:
        pass

    def pause(self, seconds: float) -> None:
        self.paused = seconds


def _error(url, code, headers=None):
    return urllib.error.HTTPError(url, code, "error", headers or {}, io.BytesIO(b""))


class Server:
    """Answers each address with a JPEG, or with the errors queued for it first."""

    def __init__(self, **errors):
        self.errors = {key: list(value) for key, value in errors.items()}
        self.asked: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.asked.append(url)
        name = url.rsplit("/", 1)[-1].split(".")[0]
        queued = self.errors.get(name)
        if queued:
            failure = queued.pop(0)
            if isinstance(failure, bytes):
                return failure
            raise _error(url, failure)
        return JPEG


def _url(name, version="1"):
    return f"https://cards.scryfall.io/normal/front/{name}.jpg?{version}"


def _run(tmp_path, files, urls, server, **options):
    return I.fetch_images(files, urls, tmp_path, fetch=server, limiter=NoWait(), sleep=lambda s: None, workers=1,
                          **options)


def test_fetches_what_is_missing_and_keeps_what_is_there(tmp_path):
    files = {"a:1": "a.jpg", "b:1": "b.jpg"}
    urls = {"a:1": _url("a"), "b:1": _url("b")}
    report = _run(tmp_path, files, urls, Server())
    assert sorted(report.fetched) == ["a:1", "b:1"] and (tmp_path / "a.jpg").read_bytes() == JPEG
    server = Server()
    report = _run(tmp_path, files, urls, server)
    assert report.present == 2 and not report.fetched and not server.asked
    # A new version of one image is fetched again.
    report = _run(tmp_path, files, {**urls, "b:1": _url("b", "2")}, Server())
    assert report.refetched == ["b:1"]
    row = sqlite3.connect(tmp_path / I.STATE_FILE).execute("SELECT version, format FROM images WHERE id='b:1'").fetchone()
    assert row == ("2", "normal/jpg")


def test_a_deleted_file_is_fetched_again(tmp_path):
    files, urls = {"a:1": "a.jpg"}, {"a:1": _url("a")}
    _run(tmp_path, files, urls, Server())
    (tmp_path / "a.jpg").unlink()
    assert _run(tmp_path, files, urls, Server()).refetched == ["a:1"]


def test_missing_images_retries_and_refusals(tmp_path):
    files = {"a:1": "a.jpg", "b:1": "b.jpg", "c:1": "c.jpg", "d:1": "d.jpg"}
    urls = {key: _url(key[0]) for key in files}
    server = Server(a=[404], b=[503, 500], c=[403], d=[b"<html>"])
    report = _run(tmp_path, files, urls, server)
    assert report.missing == ["a:1"]
    assert report.fetched == ["b:1"]  # after two failures
    assert set(report.failed) == {"c:1", "d:1"} and "not a JPEG" in report.failed["d:1"]
    assert not (tmp_path / "d.jpg").exists()


def test_a_run_stops_after_too_many_failures_in_a_row(tmp_path, monkeypatch):
    monkeypatch.setattr(I, "MAX_CONSECUTIVE_FAILURES", 3)
    files = {f"x{i}:1": f"x{i}.jpg" for i in range(6)}
    urls = {key: _url(key.split(":")[0]) for key in files}
    server = Server(**{f"x{i}": [403] for i in range(6)})
    with pytest.raises(I.FetchStopped, match="3 requests in a row failed"):
        _run(tmp_path, files, urls, server)
    assert len(server.asked) == 3


def test_an_id_that_leaves_the_catalog_is_tombstoned_not_deleted(tmp_path):
    urls = {"a:1": _url("a"), "b:1": _url("b")}
    _run(tmp_path, {"a:1": "a.jpg", "b:1": "b.jpg"}, urls, Server())
    report = _run(tmp_path, {"a:1": "a.jpg"}, urls, Server())
    assert report.tombstoned == ["b:1"] and (tmp_path / "b.jpg").is_file()
    state = sqlite3.connect(tmp_path / I.STATE_FILE)
    assert state.execute("SELECT removed_at FROM images WHERE id='b:1'").fetchone()[0] is not None
    _run(tmp_path, {"a:1": "a.jpg", "b:1": "b.jpg"}, urls, Server())
    assert state.execute("SELECT removed_at FROM images WHERE id='b:1'").fetchone()[0] is None


def test_a_state_file_without_the_format_column_is_brought_up_to_date(tmp_path):
    old = sqlite3.connect(tmp_path / I.STATE_FILE)
    old.execute("CREATE TABLE images (id TEXT PRIMARY KEY, file TEXT NOT NULL, url TEXT NOT NULL, version TEXT NOT NULL, "
                "sha256 TEXT NOT NULL, bytes INTEGER NOT NULL, fetched_at TEXT NOT NULL, removed_at TEXT)")
    old.execute("INSERT INTO images VALUES ('a:1', 'a.jpg', ?, '1', 'h', 3, 't', NULL)", (_url("a"),))
    old.commit()
    old.close()
    (tmp_path / "a.jpg").write_bytes(JPEG)
    todo, wanted, present = I.pending({"a:1": "a.jpg"}, {"a:1": _url("a")}, tmp_path)
    assert (todo, wanted, present) == ([], 1, 1)
    server = Server()
    assert _run(tmp_path, {"a:1": "a.jpg"}, {"a:1": _url("a")}, server).present == 1 and not server.asked


def test_the_command_fetches_nothing_without_download(tmp_path, capsys, monkeypatch):
    import json

    catalog = {"game": "mtg", "locale": "en", "identity": "printing",
               "cards": [{"id": "a:1", "name": "A", "kind": None, "printingOf": None, "art": "a.jpg", "extra": {}}]}
    (tmp_path / "catalog.json").write_text(json.dumps(catalog))
    (tmp_path / "images.json").write_text(json.dumps({"a:1": _url("a")}))

    def refuse(*args, **kwargs):
        raise AssertionError("fetched without --download")

    monkeypatch.setattr(I, "fetch_images", refuse)
    code = main(["fetch-images", "--catalog", str(tmp_path / "catalog.json"), "--images", str(tmp_path / "images.json"),
                 "--art", str(tmp_path / "art")])
    assert code == 0 and "pass --download" in capsys.readouterr().out
    assert not (tmp_path / "art").exists()
