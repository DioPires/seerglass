# SPDX-License-Identifier: Apache-2.0
"""The image fetcher: each catalog entry's ``normal`` JPEG, into a folder outside the repository.

The catalog builder (``cardvision.mtg.catalog``) gives each entry with art a file name (its
``art``) and, in a separate map, the address of Scryfall's ``normal`` image (488 x 680 pixels,
JPEG, a few tens of kilobytes). This module downloads the files that are not there yet.

Politeness: one ``RateLimiter`` paces every request, about ten a second by default, over a small
pool of threads, each request names this project in its ``User-Agent``, and a 429 or 503 from
the server pauses the whole pool for as long as it asks (``Retry-After``). After too many
failures in a row the run stops, and says so, rather than keep knocking.

State: a SQLite file in the art folder records, for each id, the file, the address it came from,
the image's version (the query string Scryfall puts on every image address, which changes when
the image is replaced) and format (the size and file type of the address, ``normal/jpg``: a
switch to another size or type is a change of every image), the SHA-256 and size of the bytes,
and when it was fetched. A file that is present and recorded with the current version and format
is not fetched again, so a run can be stopped and started at any time. An id that is in the state file and no longer in the catalog is never
deleted: its row gets ``removed_at`` (a tombstone) and its file stays, so a label that names it
can still be shown.

A file is written under a temporary name and renamed, so a stopped run never leaves half an
image under the real name. A body that does not start like a JPEG is refused.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
import urllib.error
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from cardvision.mtg.scryfall import RateLimiter, is_rate_limited, request, retry_after

STATE_FILE = ".fetch-state.sqlite"
JPEG_MAGIC = b"\xff\xd8\xff"
DEFAULT_RATE = 10.0
DEFAULT_WORKERS = 8
# Consecutive failed requests (after retries) that stop a run.
MAX_CONSECUTIVE_FAILURES = 20
RETRIES = 4

Fetch = Callable[[str], bytes]


class FetchStopped(RuntimeError):
    """The run stopped early: the server kept refusing or failing. The message says why."""


@dataclass
class FetchReport:
    """What a run did. Lists hold catalog ids."""

    wanted: int = 0
    present: int = 0
    fetched: list[str] = field(default_factory=list)
    refetched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)  # the server had no image (404)
    failed: dict[str, str] = field(default_factory=dict)
    tombstoned: list[str] = field(default_factory=list)
    bytes: int = 0


def image_version(url: str) -> str:
    """The version Scryfall writes on an image address: its query string (a timestamp)."""
    return urlsplit(url).query


def image_format(url: str) -> str:
    """The size and file type of an image address: ``normal/jpg`` for
    ``https://cards.scryfall.io/normal/front/...jpg``."""
    path = urlsplit(url).path
    size = path.strip("/").split("/")[0]
    return f"{size}/{path.rsplit('.', 1)[-1].lower() if '.' in path else ''}"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def open_state(art_dir: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(art_dir / STATE_FILE, check_same_thread=False)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS images (id TEXT PRIMARY KEY, file TEXT NOT NULL, url TEXT NOT NULL, "
        "version TEXT NOT NULL, sha256 TEXT NOT NULL, bytes INTEGER NOT NULL, fetched_at TEXT NOT NULL, "
        "removed_at TEXT, format TEXT)")
    columns = {row[1] for row in connection.execute("PRAGMA table_info(images)")}
    if "format" not in columns:  # a state file from before the format was recorded: its rows' addresses say it
        connection.execute("ALTER TABLE images ADD COLUMN format TEXT")
        rows = connection.execute("SELECT id, url FROM images").fetchall()
        connection.executemany("UPDATE images SET format = ? WHERE id = ?", [(image_format(url), i) for i, url in rows])
    connection.commit()
    return connection


def pending(files: Mapping[str, str], urls: Mapping[str, str], art_dir: str | Path,
            state: sqlite3.Connection | None = None) -> tuple[list[tuple[str, str, str, bool]], int, int]:
    """What a run would fetch: ``(id, file, address, fetched before)`` for each image that is not
    present at its current version and format; and how many are wanted and present."""
    art_dir = Path(art_dir)
    known: dict[str, tuple] = {}
    if state is not None:
        known = {row[0]: row[1:] for row in state.execute("SELECT id, file, version, format FROM images")}
    elif (art_dir / STATE_FILE).is_file():
        with sqlite3.connect(f"file:{art_dir / STATE_FILE}?mode=ro", uri=True) as connection:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(images)")}
            query = "SELECT id, file, version, format FROM images" if "format" in columns else \
                "SELECT id, file, version, url FROM images"
            known = {row[0]: row[1:] for row in connection.execute(query)}
            if "format" not in columns:
                known = {key: (name, version, image_format(url)) for key, (name, version, url) in known.items()}
    todo: list[tuple[str, str, str, bool]] = []
    wanted = present = 0
    for card_id, name in files.items():
        url = urls.get(card_id)
        if url is None:
            continue
        wanted += 1
        row = known.get(card_id)
        if row is not None and row == (name, image_version(url), image_format(url)) and (art_dir / name).is_file():
            present += 1
            continue
        todo.append((card_id, name, url, row is not None))
    return todo, wanted, present


def _http_fetch(url: str) -> bytes:
    return request(url, "image/jpeg, image/*;q=0.8")[0]


def fetch_images(
    files: Mapping[str, str],
    urls: Mapping[str, str],
    art_dir: str | Path,
    *,
    rate: float = DEFAULT_RATE,
    workers: int = DEFAULT_WORKERS,
    limit: int | None = None,
    fetch: Fetch = _http_fetch,
    limiter: RateLimiter | None = None,
    sleep: Callable[[float], None] = time.sleep,
    progress: Callable[[int, int], None] | None = None,
) -> FetchReport:
    """Download the images of ``files`` (id to file name) from ``urls`` (id to address).

    Ids in ``files`` without an address are skipped. ``limit`` caps the number of downloads in
    this run (for a trial). ``fetch``, ``limiter`` and ``sleep`` are seams for the tests.
    Raises ``FetchStopped`` after ``MAX_CONSECUTIVE_FAILURES`` failed requests in a row.
    """
    art_dir = Path(art_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    state = open_state(art_dir)
    lock = threading.Lock()
    limiter = limiter or RateLimiter(rate)
    report = FetchReport()
    known = {row[0] for row in state.execute("SELECT id FROM images")}
    todo, report.wanted, report.present = pending(files, urls, art_dir, state)
    if limit is not None:
        todo = todo[:limit]

    failures_in_a_row = 0
    stop: list[str] = []

    def one(item: tuple[str, str, str, bool]) -> None:
        nonlocal failures_in_a_row
        card_id, name, url, again = item
        if stop:
            return
        error: Exception | None = None
        for attempt in range(RETRIES + 1):
            limiter.acquire()
            try:
                body = fetch(url)
                error = None
                break
            except urllib.error.HTTPError as caught:
                error = caught
                if caught.code == 404:
                    break
                if is_rate_limited(caught) or caught.code == 503:
                    limiter.pause(retry_after(caught, 2.0 ** (attempt + 1)))
                elif caught.code < 500:
                    break
                else:
                    sleep(2.0 ** attempt)
            except OSError as caught:  # connection reset, timeout
                error = caught
                sleep(2.0 ** attempt)
        with lock:
            if error is not None:
                if isinstance(error, urllib.error.HTTPError) and error.code == 404:
                    report.missing.append(card_id)
                    failures_in_a_row = 0
                    return
                report.failed[card_id] = str(error)
                failures_in_a_row += 1
                if failures_in_a_row >= MAX_CONSECUTIVE_FAILURES and not stop:
                    stop.append(f"{failures_in_a_row} requests in a row failed; the last: {url}: {error}")
                return
            failures_in_a_row = 0
        if not body.startswith(JPEG_MAGIC):
            with lock:
                report.failed[card_id] = f"not a JPEG ({len(body)} bytes)"
            return
        target = art_dir / name
        partial = target.with_name(target.name + ".partial")
        partial.write_bytes(body)
        partial.replace(target)
        with lock:
            state.execute("INSERT OR REPLACE INTO images (id, file, url, version, sha256, bytes, fetched_at, removed_at, "
                          "format) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)",
                          (card_id, name, url, image_version(url), hashlib.sha256(body).hexdigest(), len(body),
                           _now(), image_format(url)))
            (report.refetched if again else report.fetched).append(card_id)
            report.bytes += len(body)
            done = len(report.fetched) + len(report.refetched)
            if done % 500 == 0:
                state.commit()
            if progress is not None:
                progress(done, len(todo))

    try:
        if workers <= 1:
            for item in todo:
                one(item)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(one, todo))
    finally:
        with lock:
            gone = [card_id for card_id in known if card_id not in files]
            stamp = _now()
            for card_id in gone:
                if state.execute("SELECT removed_at FROM images WHERE id = ?", (card_id,)).fetchone()[0] is None:
                    state.execute("UPDATE images SET removed_at = ? WHERE id = ?", (stamp, card_id))
                    report.tombstoned.append(card_id)
            back = [card_id for card_id in files if card_id in known]
            state.executemany("UPDATE images SET removed_at = NULL WHERE id = ? AND removed_at IS NOT NULL",
                              [(card_id,) for card_id in back])
            state.commit()
            state.close()
    if stop:
        raise FetchStopped(stop[0])
    return report
