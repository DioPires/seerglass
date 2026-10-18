# SPDX-License-Identifier: Apache-2.0
"""Card images that deserve a second look before a bank is trusted.

A bank is only as good as the images it is computed from, and a folder of card images
collected for a whole game usually has a few that are not what their file name says. Nothing
here decides anything: each finding is a list for a person to look at.

**Not a picture.** A file that is empty, a vector file (SVG, PDF, PostScript), a web page saved
under an image's name, or anything else Pillow cannot read (``file_problem``). A recognizer
compares pixels, and such a file has none to give.

**A placeholder.** A file Pillow reads that cannot be a card: smaller than
``PLACEHOLDER_MIN_SIDE`` pixels on a side, or one flat color (``picture_problem``).

**One image file under different cards.** Two or more cards of *different identities* whose
image files are byte for byte the same (``shared_images``). Printings of one identity may well
share a file; different cards should not. It usually means that one card's image was filed
under another card's id, or that one stand-in picture was saved for every card that has none.

**Nearly the same bank row under different cards.** Cards of different identities whose bank
rows have a cosine of ``NEAR_IDENTICAL_COSINE`` or more (``near_identical_rows``). It usually
means the same picture filed under two cards, perhaps saved twice in two formats, or one
card's image filed under another's id. The accept rule cannot tell such cards apart: each
takes the margin from the other, and neither is named.

Which cards are the same identity is the catalog's rule (``Catalog.identity_of``).

The limit of 0.99 was chosen on the toy game of the test suite. There, one image embedded
under two card ids gives two rows with a cosine above 0.998: a row is the mean of sixteen
views whose random damage is seeded by the card id, so the two rows are close and not equal.
Rows of different toy cards stay below 0.97, with a stand-in recognizer that has no skill. The
limit has not been tuned on any real game: it is where to start looking, not a verdict.
"""

from __future__ import annotations

import io
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from cardvision.core.bank import decode_bank, stored_art_versions
from cardvision.core.catalog import Catalog

# Rows of different identities at or above this cosine are reported (see the module docstring).
NEAR_IDENTICAL_COSINE = 0.99
# A picture with a side under this many pixels is reported as a placeholder.
PLACEHOLDER_MIN_SIDE = 32
# How many rows are compared with the whole bank at a time.
_BLOCK = 1024


def file_problem(data: bytes) -> str | None:
    """Why the bytes of an image file hold no picture, or ``None`` when Pillow reads one.

    The first bytes tell the common cases apart, so that the reason is one a person can act
    on; anything else that Pillow cannot read is reported as that.
    """
    if not data:
        return "an empty file"
    head = data[:2048].lstrip().lower()
    if head.startswith(b"<svg") or (head.startswith(b"<?xml") and b"<svg" in head):
        return "a vector file (SVG), which has no pixels to compare"
    if head.startswith((b"%pdf", b"%!ps")):
        return "a PDF or PostScript file, which has no pixels to compare"
    if head.startswith((b"<!doctype html", b"<html")):
        return "a web page saved under an image's name"
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
    except Exception:  # the image library raises many kinds
        return "not an image that Pillow can read"
    return None


def picture_problem(image: np.ndarray) -> str | None:
    """Why a decoded ``(H, W, 3)`` picture cannot be a card's image, or ``None``: it is tiny,
    or it is one flat color."""
    height, width = image.shape[:2]
    if min(width, height) < PLACEHOLDER_MIN_SIDE:
        return f"a picture of {width} x {height} pixels, too small to be a card: a placeholder"
    if int(image.max()) == int(image.min()):
        return "a picture of one flat color: a placeholder"
    return None


NOT_PICTURE = "not-picture"
PLACEHOLDER = "placeholder"


def image_problem(data: bytes) -> tuple[str, str] | None:
    """What is wrong with the bytes of an image file, as ``(kind, reason)``, or ``None``.

    The kind is ``NOT_PICTURE`` with the reason of ``file_problem``, or, for a file that is a
    picture, ``PLACEHOLDER`` with the reason of ``picture_problem``.
    """
    problem = file_problem(data)
    if problem is not None:
        return NOT_PICTURE, problem
    with Image.open(io.BytesIO(data)) as image:
        problem = picture_problem(np.asarray(image.convert("RGB")))
    return None if problem is None else (PLACEHOLDER, problem)


def _identity(catalog: Catalog, keys: dict[str, str], card_id: str) -> str:
    """A card's identity key; a card the catalog does not hold is an identity of its own."""
    if card_id not in keys:
        keys[card_id] = catalog.identity_of(card_id) if card_id in catalog else f"\0{card_id}"
    return keys[card_id]


def shared_images(images: Iterable[tuple[str, str | None]], catalog: Catalog) -> list[list[str]]:
    """The groups of cards of two or more identities that have one image file.

    ``images`` gives ``(card id, a hash of the image file's bytes)``; a card whose hash is
    ``None`` is left out. Each group is sorted, and the list is ordered by each group's first
    id. Cards of one identity that share a file are not reported.
    """
    by_image: dict[str, set[str]] = {}
    for card_id, version in images:
        if version is not None:
            by_image.setdefault(version, set()).add(card_id)
    keys: dict[str, str] = {}
    groups = [sorted(cards) for cards in by_image.values()
              if len({_identity(catalog, keys, card_id) for card_id in cards}) >= 2]
    return sorted(groups, key=lambda group: group[0])


def near_identical_rows(ids: Sequence[str], rows: np.ndarray, catalog: Catalog,
                        threshold: float = NEAR_IDENTICAL_COSINE) -> list[list[str]]:
    """The groups of cards of two or more identities whose bank rows are nearly the same.

    ``ids`` and ``rows`` are a bank's card ids and unit rows (``cardvision.core.bank``). Two
    rows of different identities are joined when their cosine is ``threshold`` or more, and
    rows joined through a third are one group, so a stand-in picture under a hundred cards is
    one group of a hundred and not thousands of pairs. Each group is sorted, and the list is
    ordered by each group's first id.
    """
    rows = np.asarray(rows, dtype=np.float32)
    if len(ids) < 2:
        return []
    keys: dict[str, str] = {}
    index: dict[str, int] = {}
    identity = np.array([index.setdefault(_identity(catalog, keys, card_id), len(index)) for card_id in ids])
    parent = list(range(len(ids)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for start in range(0, len(ids), _BLOCK):
        block = rows[start:start + _BLOCK] @ rows.T
        close = block >= threshold
        close &= identity[start:start + _BLOCK, None] != identity[None, :]
        for i, j in zip(*np.nonzero(close)):
            a, b = find(start + int(i)), find(int(j))
            if a != b:
                parent[max(a, b)] = min(a, b)
    groups: dict[int, set[str]] = {}
    for i, card_id in enumerate(ids):
        groups.setdefault(find(i), set()).add(card_id)
    return sorted((sorted(cards) for cards in groups.values() if len(cards) >= 2), key=lambda group: group[0])


@dataclass
class Suspects:
    """What was found, each list for a person to look at (see the module docstring).

    Attributes:
        not_pictures: ``(card id, reason)`` for image files that hold no picture.
        placeholders: ``(card id, reason)`` for pictures that cannot be a card.
        shared: groups of cards of different identities with one image file.
        near_identical: groups of cards of different identities with nearly the same bank row.
    """

    not_pictures: list[tuple[str, str]] = field(default_factory=list)
    placeholders: list[tuple[str, str]] = field(default_factory=list)
    shared: list[list[str]] = field(default_factory=list)
    near_identical: list[list[str]] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.not_pictures or self.placeholders or self.shared or self.near_identical)

    def add(self, card_id: str, problem: tuple[str, str] | None) -> None:
        """File a card under what ``image_problem`` found for its image; nothing for ``None``."""
        if problem is not None:
            kind, reason = problem
            (self.not_pictures if kind == NOT_PICTURE else self.placeholders).append((card_id, reason))


def bank_suspects(bank: Mapping[str, Any], catalog: Catalog) -> Suspects:
    """What a bank document shows by itself, without the images: the cards of different
    identities whose rows record one image file (``artV``), and those whose rows are nearly
    the same. A row that records no image takes no part in the first."""
    ids, rows = decode_bank(bank)
    return Suspects(shared=shared_images(zip(ids, stored_art_versions(bank)), catalog),
                    near_identical=near_identical_rows(ids, rows, catalog))


def _some(group: Sequence[str], limit: int = 8) -> str:
    return ", ".join(group[:limit]) + (f", and {len(group) - limit} more" if len(group) > limit else "")


def suspect_lines(found: Suspects, listed: int = 20) -> list[str]:
    """The findings as lines of a report: nothing when there are none. At most ``listed``
    entries of each kind are written out, and the rest are counted."""
    lines: list[str] = []

    def more(count: int) -> list[str]:
        return [f"  and {count - listed} more"] if count > listed else []

    for title, entries in (("image files that hold no picture", found.not_pictures),
                           ("pictures that look like placeholders", found.placeholders)):
        if entries:
            lines.append(f"{title} ({len(entries)}):")
            lines += [f"  {card_id}  {reason}" for card_id, reason in entries[:listed]]
            lines += more(len(entries))
    if found.shared:
        lines.append(f"one image file under different cards ({len(found.shared)} group(s)):")
        lines += [f"  {_some(group)}" for group in found.shared[:listed]]
        lines += more(len(found.shared))
        lines.append("  This usually means one card's image filed under another card's id, or one stand-in picture "
                     "saved for cards that have none.")
    if found.near_identical:
        lines.append(f"nearly the same bank row under different cards, cosine {NEAR_IDENTICAL_COSINE:g} or more "
                     f"({len(found.near_identical)} group(s)):")
        lines += [f"  {_some(group)}" for group in found.near_identical[:listed]]
        lines += more(len(found.near_identical))
        lines.append("  This usually means the same picture filed under two cards, or one card's image filed under "
                     "another's id. The accept rule cannot tell such cards apart, and names neither.")
    return lines
