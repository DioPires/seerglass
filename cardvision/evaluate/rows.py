# SPDX-License-Identifier: Apache-2.0
"""Which manifest rows an evaluation uses, and an account of every row it does not.

A figure is only as honest as its denominator. Every row read from the manifests ends in
exactly one place:

* **used**: its frame was found and it was scored;
* **skipped**, with a reason: the row is not in the role being evaluated (``role``), or it has
  no label the measure reads (``no-labels``);
* **unusable**, with a reason: the row should have been scored and could not be, because its
  frame is not in the folder (``frame-missing``) or the frame there has another shape than the
  one the labels were made on (``frame-shape``).

An unusable row is listed by its key in the result, with what it would have contributed, and is
never dropped silently.

The rows may also be generated scenes, read from the generator's own labels and not from a
manifest (``cardvision.evaluate.scenes``). The account then says so, and has no role.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cardvision.data.extract import FrameMissing, FrameStore, frame_check
from cardvision.data.manifest import read_manifest
from cardvision.data.schema import Row
from cardvision.evaluate.config import file_sha256

DEFAULT_ROLE = "evaluation"


@dataclass
class Account:
    """What became of the rows (see the module's description).

    Attributes:
        manifests: ``{"file", "sha256", "rows"}`` per manifest file read.
        role: the role evaluated; ``None`` for generated scenes, which have no role.
        read: rows read from the manifests.
        skipped, unusable: rows by reason.
        used: rows scored.
        frames_present: rows with labels for the measure whose frame file was found.
        not_scored: one entry per unusable row: its key, the reason and the ``lost`` labels.
        labels_skipped: labels of used rows that the measure leaves out, by reason.
        labels_lost: labels of unusable rows that would have been scored, by what they are.
        scenes: ``None`` for rows read from manifests. For generated scenes, what
            ``cardvision.evaluate.scenes.read_scenes`` says about them; the result then carries
            it and the report opens with it.
    """

    manifests: list[dict[str, Any]] = field(default_factory=list)
    role: str | None = DEFAULT_ROLE
    read: int = 0
    skipped: Counter = field(default_factory=Counter)
    unusable: Counter = field(default_factory=Counter)
    used: int = 0
    frames_present: int = 0
    not_scored: list[dict[str, Any]] = field(default_factory=list)
    labels_skipped: Counter = field(default_factory=Counter)
    labels_lost: Counter = field(default_factory=Counter)
    scenes: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        out = {
            "manifests": self.manifests,
            "role": self.role,
            "rowsRead": self.read,
            "rowsUsed": self.used,
            "rowsSkipped": dict(sorted(self.skipped.items())),
            "rowsUnusable": dict(sorted(self.unusable.items())),
            "framesPresent": self.frames_present,
            "labelsLeftOut": dict(sorted(self.labels_skipped.items())),
            "labelsNotScored": dict(sorted(self.labels_lost.items())),
            "rowsNotScored": self.not_scored,
        }
        if self.scenes is not None:
            out["scenes"] = self.scenes
        return out


def read_rows(manifests: Sequence[str | Path], role: str = DEFAULT_ROLE) -> tuple[list[Row], Account]:
    """Read the manifests and keep the rows of ``role``, in file order.

    Raises ``ValueError`` when two rows share a platform, video id and time.
    """
    account = Account(role=role)
    rows: list[Row] = []
    seen: set[tuple] = set()
    for path in manifests:
        found = read_manifest(path)
        account.manifests.append({"file": Path(path).name, "sha256": file_sha256(path), "rows": len(found)})
        for row in found:
            if row.key in seen:
                raise ValueError(f"{Path(path).name}: the row {row.platform} {row.video_id} t={row.t} is given twice")
            seen.add(row.key)
            account.read += 1
            if row.role == role:
                rows.append(row)
            else:
                account.skipped["role"] += 1
    return rows, account


def scorable_rows(
    rows: Sequence[Row],
    frames: FrameStore | str | Path,
    account: Account,
    labels: Callable[[Row], Counter],
) -> Iterator[tuple[Row, np.ndarray]]:
    """``(row, frame)`` for every row that can be scored, accounting for the others.

    ``labels(row)`` counts what the measure would score in the row, by kind; an empty count
    means the row has nothing for the measure and is skipped with reason ``no-labels``.
    """
    from cardvision.core.images import load_rgb

    store = frames if isinstance(frames, FrameStore) else FrameStore(frames)
    for row in rows:
        wanted = labels(row)
        if not sum(wanted.values()):
            account.skipped["no-labels"] += 1
            continue
        reason, rgb = None, None
        try:
            path = store.path(row)
        except FrameMissing:
            reason = "frame-missing"
        else:
            account.frames_present += 1
            rgb = load_rgb(path)
            if frame_check(row, rgb.shape[1], rgb.shape[0], store.tolerance) is not None:
                reason = "frame-shape"
        if reason is not None:
            account.unusable[reason] += 1
            account.labels_lost.update(wanted)
            account.not_scored.append({"platform": row.platform, "video_id": row.video_id, "t": row.t,
                                       "group": row.group, "source_type": row.source_type, "reason": reason,
                                       "lost": dict(sorted(wanted.items()))})
            continue
        account.used += 1
        yield row, rgb


def slices(items: Sequence[Any], source_type: Callable[[Any], str]) -> dict[str, dict[str, list[Any]]]:
    """Items split three ways: ``overall`` (everything), ``bySourceType`` (the exact source
    type) and ``bySourceFamily`` (the part before any slash)."""
    by_type: dict[str, list[Any]] = {}
    by_family: dict[str, list[Any]] = {}
    for item in items:
        kind = source_type(item)
        by_type.setdefault(kind, []).append(item)
        by_family.setdefault(kind.split("/", 1)[0], []).append(item)
    return {"overall": {"all": list(items)}, "bySourceType": dict(sorted(by_type.items())),
            "bySourceFamily": dict(sorted(by_family.items()))}


def sliced(items: Sequence[Any], source_type: Callable[[Any], str], measure: Callable[[list[Any]], Any]) -> dict[str, Any]:
    """``measure`` applied to every slice of ``slices``: ``{"overall": ..., "bySourceType":
    {type: ...}, "bySourceFamily": {family: ...}}``."""
    parts = slices(items, source_type)
    return {"overall": measure(parts["overall"]["all"]),
            "bySourceType": {name: measure(part) for name, part in parts["bySourceType"].items()},
            "bySourceFamily": {name: measure(part) for name, part in parts["bySourceFamily"].items()}}
