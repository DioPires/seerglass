# SPDX-License-Identifier: Apache-2.0
"""A game's footage sources: the source types it adds, and which of them crop the hand row.

A manifest row names its ``source_type``. Two families are fixed, ``physical`` and ``digital``,
and a few details are built in (``cardvision.data.schema.SOURCE_TYPES``). A game that needs
more, its digital clients for instance, lists them in a small JSON file::

    {
      "sourceTypes": ["digital/client-a", "digital/client-b"],
      "handRow": {"sourceTypes": ["digital/client-b"], "minCenterY": 0.85}
    }

``sourceTypes`` extends the built-in list. ``handRow`` (optional) names the sources whose
client draws the cards in hand cut off along the bottom of the screen, and how far down the
frame that row starts; the label conversions use it (``cardvision.data.targets``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cardvision.data.schema import SOURCE_TYPE_PATTERN, SOURCE_TYPES

DEFAULT_HAND_ROW_MIN_CENTER_Y = 0.85


class SourceConfigError(ValueError):
    """The sources file does not have the documented shape. The message says where."""


@dataclass(frozen=True)
class SourceConfig:
    """The source types a game accepts and its hand-row setting.

    Attributes:
        source_types: every accepted value of ``source_type``, built-in ones included.
        hand_row_source_types: sources whose client draws hand cards cut off at the bottom.
        hand_row_min_center_y: a box counts as in the hand row when its center is at least
            this fraction of the way down the frame.
    """

    source_types: tuple[str, ...] = SOURCE_TYPES
    hand_row_source_types: tuple[str, ...] = ()
    hand_row_min_center_y: float = DEFAULT_HAND_ROW_MIN_CENTER_Y


def _source_type_list(value: Any, where: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SourceConfigError(f"{where} must be a list of source types")
    for item in value:
        if re.fullmatch(SOURCE_TYPE_PATTERN, item) is None:
            raise SourceConfigError(
                f'{where}: "{item}" is not a source type (physical or digital, optionally followed by '
                "a slash and a lower-case detail)"
            )
    return value


def parse_source_config(doc: Any, source: str = "sources") -> SourceConfig:
    """Check a parsed sources file and return it as a ``SourceConfig``."""
    if not isinstance(doc, dict):
        raise SourceConfigError(f"{source}: must be a JSON object")
    unknown = set(doc) - {"sourceTypes", "handRow"}
    if unknown:
        raise SourceConfigError(f"{source}: unknown key {', '.join(sorted(unknown))}")
    added = _source_type_list(doc.get("sourceTypes", []), f'{source}: "sourceTypes"')
    types = tuple(dict.fromkeys([*SOURCE_TYPES, *added]))
    hand = doc.get("handRow", {})
    if not isinstance(hand, dict) or set(hand) - {"sourceTypes", "minCenterY"}:
        raise SourceConfigError(f'{source}: "handRow" must be an object with "sourceTypes" and "minCenterY"')
    hand_types = _source_type_list(hand.get("sourceTypes", []), f'{source}: "handRow.sourceTypes"')
    missing = [item for item in hand_types if item not in types]
    if missing:
        raise SourceConfigError(f'{source}: "handRow.sourceTypes" names {", ".join(missing)}, not a listed source type')
    center = hand.get("minCenterY", DEFAULT_HAND_ROW_MIN_CENTER_Y)
    if isinstance(center, bool) or not isinstance(center, (int, float)) or not 0 <= center <= 1:
        raise SourceConfigError(f'{source}: "handRow.minCenterY" must be a number between 0 and 1')
    return SourceConfig(source_types=types, hand_row_source_types=tuple(hand_types), hand_row_min_center_y=float(center))


def load_source_config(path: str | Path) -> SourceConfig:
    """Read and check a sources file."""
    with open(path, encoding="utf-8") as handle:
        try:
            doc = json.load(handle)
        except ValueError as error:
            raise SourceConfigError(f"{path}: not valid JSON ({error})") from None
    return parse_source_config(doc, str(path))
