# SPDX-License-Identifier: Apache-2.0
"""Read, write and filter label manifests.

A manifest is a UTF-8 text file with one JSON object per line (JSON Lines); blank lines are
skipped. ``read_manifest`` checks each row against the format and refuses the file if any row
fails; checks that span rows (duplicates, groups split across roles) belong to
``cardvision.data.validate``.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Iterable, Iterator
from pathlib import Path
from typing import Any

from cardvision.core.files import write_text_atomic
from cardvision.data.schema import COMPLETENESS, ROLES, ManifestError, Row, row_problems


def _refuse_constant(token: str) -> Any:
    raise ValueError(f"{token} is not a JSON number")


def parse_line(text: str) -> Any:
    """One manifest line as a JSON value. ``NaN`` and ``Infinity`` are refused, as JSON does."""
    return json.loads(text, parse_constant=_refuse_constant)


def numbered_lines(path: str | Path) -> Iterator[tuple[int, str]]:
    """``(line number, text)`` of every non-blank line of a manifest file, numbered from 1."""
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if line.strip():
                yield number, line


def read_manifest(path: str | Path) -> list[Row]:
    """Every row of a manifest file, in file order.

    Raises ``ManifestError`` listing each line that is not valid JSON or not a valid row.
    """
    rows: list[Row] = []
    problems: list[str] = []
    for number, line in numbered_lines(path):
        try:
            doc = parse_line(line)
        except ValueError as error:
            problems.append(f"line {number}: not valid JSON ({error})")
            continue
        found = row_problems(doc)
        if found:
            problems.extend(f"line {number}: {problem}" for problem in found)
        else:
            rows.append(Row.from_json(doc))
    if problems:
        raise ManifestError(f"{path}: " + "; ".join(problems))
    return rows


def row_line(row: Row) -> str:
    """A row as its line of the manifest file, without the line ending."""
    return json.dumps(row.to_json(), ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def write_manifest(path: str | Path, rows: Iterable[Row]) -> int:
    """Write rows to a manifest file, one per line, in the order given. Returns the count.

    Every row is checked first, so a file written here always reads back; a row that does not
    fit the format raises ``ManifestError`` and nothing is written.

    The file is written in one step (``cardvision.core.files.write_text_atomic``): a write that
    fails or is interrupted leaves a manifest that was already there as it was.
    """
    lines = []
    for index, row in enumerate(rows):
        problems = row_problems(row.to_json())
        if problems:
            raise ManifestError(f"row {index + 1} ({row.platform} {row.video_id} t={row.t}): " + "; ".join(problems))
        lines.append(row_line(row))
    write_text_atomic(path, "".join(line + "\n" for line in lines))
    return len(lines)


def _as_set(value: str | Collection[str] | None, allowed: Collection[str] | None, what: str) -> set[str] | None:
    if value is None:
        return None
    chosen = {value} if isinstance(value, str) else set(value)
    if allowed is not None and not chosen <= set(allowed):
        raise ValueError(f"unknown {what}: {', '.join(sorted(chosen - set(allowed)))}")
    return chosen


def iter_rows(
    rows: Iterable[Row],
    *,
    role: str | Collection[str] | None = None,
    completeness: str | Collection[str] | None = None,
    source_type: str | Collection[str] | None = None,
    source_family: str | Collection[str] | None = None,
    trained_on: bool | None = None,
    reachable: bool | None = None,
) -> Iterator[Row]:
    """The rows that match every filter given; a filter left at ``None`` matches everything.

    ``reachable=True`` keeps rows whose video was found at the last check; ``reachable=False``
    keeps rows whose video was not. Rows never checked match neither.
    """
    roles = _as_set(role, ROLES, "role")
    kinds = _as_set(completeness, COMPLETENESS, "completeness")
    types = _as_set(source_type, None, "source type")
    families = _as_set(source_family, None, "source family")
    for row in rows:
        if roles is not None and row.role not in roles:
            continue
        if kinds is not None and row.completeness not in kinds:
            continue
        if types is not None and row.source_type not in types:
            continue
        if families is not None and row.source_family not in families:
            continue
        if trained_on is not None and row.trained_on is not trained_on:
            continue
        if reachable is not None and row.reachable is not reachable:
            continue
        yield row


def rows_by_role(rows: Iterable[Row]) -> dict[str, list[Row]]:
    """Rows grouped by role, every role present as a key, each list in the order given."""
    out: dict[str, list[Row]] = {role: [] for role in ROLES}
    for row in rows:
        out[row.role].append(row)
    return out


def rows_by_group(rows: Iterable[Row]) -> dict[str, list[Row]]:
    """Rows grouped by recording group, groups in order of first appearance."""
    out: dict[str, list[Row]] = {}
    for row in rows:
        out.setdefault(row.group, []).append(row)
    return out
