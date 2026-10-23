# SPDX-License-Identifier: Apache-2.0
"""Deck lists as Magic sites write them, made ready for ``labeler/decklist_from_text.py``.

Deck-list sites name a card with two faces by its front face alone ("Callous Sell-Sword"),
while the catalog names it by both ("Callous Sell-Sword // Burn Together"). The labelling tool
matches whole names and guesses nothing, so such a line would stay unmatched. ``full_names``
rewrites a line's name to the catalog's full name when the name is not a catalog name itself
and exactly one catalog name starts with it followed by `` // ``. Every other line is kept as
it is, and the tool reports what it still cannot match.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

FACE_SEPARATOR = " // "
LINE = re.compile(r"^(?P<count>\s*\d+\s*[x×]?\s+)?(?P<name>\S.*?)\s*$")


def full_names(text: str, cards: Iterable[Mapping[str, Any]]) -> tuple[str, list[tuple[str, str]]]:
    """The deck list ``text`` with front-face names made full, and the ``(name, full name)``
    pairs rewritten, once each, in order. Names compare without regard to case."""
    cards = list(cards)
    names = {card["name"].casefold() for card in cards}
    fronts: dict[str, set[str]] = {}
    for card in cards:
        name = card["name"]
        if FACE_SEPARATOR in name:
            fronts.setdefault(name.split(FACE_SEPARATOR, 1)[0].casefold(), set()).add(name)
    lines, rewritten = [], []
    for line in text.splitlines():
        match = LINE.match(line)
        if match and not line.lstrip().startswith("#"):
            name = match["name"]
            full = fronts.get(name.casefold(), set())
            if name.casefold() not in names and len(full) == 1:
                (whole,) = full
                line = f"{match['count'] or ''}{whole}"
                if (name, whole) not in rewritten:
                    rewritten.append((name, whole))
        lines.append(line)
    return "\n".join(lines) + ("\n" if text.endswith("\n") else ""), rewritten
