# SPDX-License-Identifier: Apache-2.0
"""Reading Scryfall: its bulk-data files, and polite HTTP for its image host.

Scryfall publishes every card once a day as bulk files (https://scryfall.com/docs/api/bulk-data).
One download of ``default_cards`` (every printing, in English where an English printing exists)
replaces about a hundred thousand API calls, so the catalog is always built from it. The bulk
index lists each file with a ``download_uri`` (a JSON array) and, on newer servers, a
``jsonl_download_uri`` (one card per line, gzip compressed); ``read_cards`` reads either.

Every request names this project in its ``User-Agent`` and states what it accepts, as Scryfall
asks. Requests to the image host are paced by a ``RateLimiter`` shared by every thread. This
module does not use any other rate limiter: the bulk image download is a separate client from a
per-card API server and keeps out of its budget.
"""

from __future__ import annotations

import gzip
import io
import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from cardvision.core.tls import (
    CERTIFICATE_HELP,
    CertificatesMissing,
    is_certificate_failure,
    tls_context,
)

USER_AGENT = "Seerglass/0.1 (+https://github.com/DioPires/seerglass)"
BULK_INDEX = "https://api.scryfall.com/bulk-data"
# Bulk-data types this package reads.
DEFAULT_CARDS = "default_cards"
ORACLE_CARDS = "oracle_cards"


class RateLimiter:
    """At most ``rate`` acquisitions a second over every thread, spaced evenly.

    Each ``acquire`` waits for the next free slot; slots are handed out in the order threads
    ask. A pause (``pause``) pushes every later slot back, which is how a 429 from the server
    slows the whole pool and not only the thread that saw it.
    """

    def __init__(self, rate: float, clock=time.monotonic, sleep=time.sleep) -> None:
        if not rate > 0:
            raise ValueError(f"the rate must be above 0 requests a second, got {rate!r}")
        self.interval = 1.0 / rate
        self._clock = clock
        self._sleep = sleep
        self._next = clock()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = self._clock()
            slot = max(now, self._next)
            self._next = slot + self.interval
        if slot > now:
            self._sleep(slot - now)

    def pause(self, seconds: float) -> None:
        with self._lock:
            self._next = max(self._next, self._clock() + seconds)


def request(url: str, accept: str, timeout: float = 60.0) -> tuple[bytes, dict[str, str]]:
    """GET ``url`` with this project's User-Agent. Returns the body and the response headers.

    Raises ``urllib.error.HTTPError`` for an HTTP error status, and ``CertificatesMissing`` when
    an https server's certificate cannot be verified."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=tls_context()) as response:
            return response.read(), dict(response.headers.items())
    except OSError as error:
        if is_certificate_failure(error):
            raise CertificatesMissing(f"{url}: {CERTIFICATE_HELP}") from None
        raise


def bulk_entry(kind: str = DEFAULT_CARDS) -> dict[str, Any]:
    """The bulk index's entry for one bulk-data type (``default_cards``, ``oracle_cards`` ...)."""
    body, _ = request(BULK_INDEX, "application/json")
    for entry in json.loads(body)["data"]:
        if entry.get("type") == kind:
            return entry
    raise ValueError(f"Scryfall's bulk index has no {kind!r} file")


def download_bulk(out_dir: str | Path, kind: str = DEFAULT_CARDS) -> Path:
    """Download one bulk-data file into ``out_dir`` and return its path.

    The file keeps the name the server gives it, which carries its timestamp, so a day's file
    is never overwritten by the next day's. A file already there is not downloaded again. The
    index entry is written beside it as ``<file>.meta.json``."""
    entry = bulk_entry(kind)
    url = entry.get("jsonl_download_uri") or entry["download_uri"]
    out = Path(out_dir) / url.rsplit("/", 1)[-1]
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        body, _ = request(url, "application/json, application/gzip, */*", timeout=600.0)
        partial = out.with_name(out.name + ".partial")
        partial.write_bytes(body)
        partial.replace(out)
    out.with_name(out.name + ".meta.json").write_text(json.dumps(entry, indent=1) + "\n", encoding="utf-8")
    return out


def read_cards(path: str | Path) -> Iterator[dict[str, Any]]:
    """The card objects of a bulk file: JSON Lines or one JSON array, gzip compressed or not."""
    path = Path(path)
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    text = io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8")
    first = text.read(1)
    while first and first.isspace():
        first = text.read(1)
    if first == "[":
        text.seek(0)
        yield from json.load(text)
        return
    text.seek(0)
    for line in text:
        line = line.strip().rstrip(",")
        if line:
            yield json.loads(line)


def is_rate_limited(error: Exception) -> bool:
    """Whether an error is the server asking to slow down (HTTP 429)."""
    return isinstance(error, urllib.error.HTTPError) and error.code == 429


def retry_after(error: Exception, default: float) -> float:
    """The pause a 429 or 503 asks for (its ``Retry-After`` header, in seconds), or ``default``."""
    headers = getattr(error, "headers", None)
    value = headers.get("Retry-After") if headers is not None else None
    try:
        return max(0.0, float(value)) if value is not None else default
    except ValueError:
        return default
