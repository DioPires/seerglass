# SPDX-License-Identifier: Apache-2.0
"""Record, in a manifest, whether each row's video could be found on its platform.

A row's ``reachable`` and ``reachable_checked`` are set from the index that a ``cardvision
fetch --download`` run wrote (``cardvision.data.extract``): the run is the check, and the index
is its record. A row is judged by its last line in the index, and only on evidence about the
video itself.

* The row's frame was obtained from a download (``ok``): ``reachable`` becomes true.
* The video was downloaded and does not hold the row's frame (``unusable``: no frame near the
  row's time matches the row's fingerprint, or the video has another shape): ``reachable``
  becomes false.
* The download failed and the platform said why (``failed`` with the cause ``unavailable``:
  the video is private, removed or does not exist): ``reachable`` becomes false.
* Anything else leaves the row as it was. That includes a download that failed for any other
  reason: a restricted video (an age gate, a members' video, another country), a network
  error, the cutting program, a timeout, or a failed line that records no cause. Such a
  failure says nothing about whether the video is there. It also includes a row the index does
  not list, a row that had no video (``no-video``), and a row whose frame came from a file the
  user gave or from a kept file, which says nothing about the platform today.

The date recorded is the date, in UTC, on which the row's index line was written. An index
whose lines record no time was written by an older version of the fetch command, and the date
of that run then has to be given.

A recording of the user's own (platform ``local``) is on no platform, so its two fields are
never set.

Nothing here opens a network connection.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from cardvision.data.extract import INDEX_NAME, IndexEntry, index_date
from cardvision.data.schema import LOCAL_PLATFORM, Row, RowKey

# The cause of a failed download that says the video itself could not be found
# (``cardvision.data.fetch.CAUSE_UNAVAILABLE``).
NOT_FOUND_CAUSE = "unavailable"


def check_date(text: str) -> str:
    """``text`` when it is a calendar date written ``YYYY-MM-DD``. Raises ``ValueError`` otherwise."""
    try:
        if datetime.date.fromisoformat(text).isoformat() != text:
            raise ValueError
    except ValueError:
        raise ValueError(f"{text!r}: a date is written YYYY-MM-DD") from None
    return text


def read_index_file(path: str | Path) -> dict[RowKey, IndexEntry]:
    """The index of a frame folder, by row key, from the index file or from its folder. A row
    listed more than once keeps its last line. Raises ``FileNotFoundError`` when there is no
    index and ``ValueError`` when a line is not an index line."""
    path = Path(path)
    if path.is_dir():
        path = path / INDEX_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path}: no such index file")
    entries: dict[RowKey, IndexEntry] = {}
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                entry = IndexEntry.from_json(json.loads(line))
                if entry.written is not None:
                    index_date(entry.written)
            except (ValueError, KeyError, TypeError):
                raise ValueError(f"{path}: line {number} is not a line of a frame folder's index") from None
            entries[entry.key] = entry
    return entries


def reachable_verdict(row: Row, entry: IndexEntry | None) -> bool | None:
    """What the index says about a row: true, false, or ``None`` when it says nothing."""
    if row.platform == LOCAL_PLATFORM or entry is None or entry.source != "download":
        return None
    if entry.status == "ok":
        return True
    if entry.status == "unusable":
        return False
    if entry.status == "failed" and entry.cause == NOT_FOUND_CAUSE:
        return False
    return None


def _why(entry: IndexEntry) -> str:
    """An index line's reason, as a person reads it."""
    if entry.reason == "fingerprint":
        return f"no frame within a second of t looks like the labeled one (nearest differs by {entry.distance} bits)"
    return entry.reason or "no reason recorded"


@dataclass(frozen=True)
class Judged:
    """One row to report, with what its index line said.

    Attributes:
        platform, video_id, t: the row.
        status: the line's status, ``unusable`` or ``failed``.
        cause: the line's cause, for a failed download; ``None`` when it records none.
        reason: the line's reason, in words.
    """

    platform: str
    video_id: str
    t: float
    status: str
    cause: str | None
    reason: str


@dataclass(frozen=True)
class ReachableSummary:
    """What a recording did.

    Attributes:
        reachable, not_reachable: rows set to true and to false.
        unchanged: rows left as they were.
        unmatched: index lines that name no row of the manifest.
        lost: the rows set to false, in the manifest's order.
        failed_unchanged: the rows whose download failed for a reason that says nothing about
            the video, and that were therefore left as they were, in the manifest's order.
        dates: the dates recorded, each once, in ascending order.
    """

    reachable: int = 0
    not_reachable: int = 0
    unchanged: int = 0
    unmatched: int = 0
    lost: tuple[Judged, ...] = ()
    failed_unchanged: tuple[Judged, ...] = ()
    dates: tuple[str, ...] = ()


def record_reachable(rows: Sequence[Row], index: Mapping[RowKey, IndexEntry],
                     date: str | None = None) -> tuple[list[Row], ReachableSummary]:
    """The rows with ``reachable`` and ``reachable_checked`` set from ``index``, in the order
    given, and an account of what was done. Every other field of every row is left as it is.

    The date recorded for a row is the UTC date on which its index line was written. ``date``,
    when given, is recorded for every row instead. Raises ``ValueError`` when a line that sets
    a row records no time and no ``date`` is given.
    """
    if date is not None:
        check_date(date)
    out: list[Row] = []
    found = undated = 0
    lost: list[Judged] = []
    failed_unchanged: list[Judged] = []
    dates: set[str] = set()
    for row in rows:
        entry = index.get(row.key)
        verdict = reachable_verdict(row, entry)
        if verdict is None:
            out.append(row)
            if entry is not None and row.platform != LOCAL_PLATFORM and entry.source == "download" and entry.status == "failed":
                failed_unchanged.append(Judged(row.platform, row.video_id, row.t, entry.status, entry.cause, _why(entry)))
            continue
        if date is None and entry.written is None:
            undated += 1
            continue
        checked = date if date is not None else index_date(entry.written)
        dates.add(checked)
        found += verdict
        if verdict is False:
            lost.append(Judged(row.platform, row.video_id, row.t, entry.status, entry.cause, _why(entry)))
        out.append(dataclasses.replace(row, reachable=verdict, reachable_checked=checked))
    if undated:
        raise ValueError(f"{undated} line(s) of the index that set a row record no time: the index was written by an "
                         "older version of cardvision fetch. Pass --date YYYY-MM-DD with the date of that fetch run")
    keys = {row.key for row in rows}
    summary = ReachableSummary(reachable=found, not_reachable=len(lost), unchanged=len(rows) - found - len(lost),
                               unmatched=sum(1 for key in index if key not in keys), lost=tuple(lost),
                               failed_unchanged=tuple(failed_unchanged), dates=tuple(sorted(dates)))
    return out, summary
