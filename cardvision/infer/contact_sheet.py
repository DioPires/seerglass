# SPDX-License-Identifier: Apache-2.0
"""A contact sheet: each crop beside the three cards it most resembles.

This is the first thing to look at when trying the models on new footage or on another game.
One row per recognized crop shows the crop itself, the decision (a name, or nothing) with its
score and margin, and the three best cards of different identities with their similarities:
differently named cards, unless the catalog states another identity rule. Every card is
given with its id. When the card images can be found, each candidate is shown as a picture;
when they cannot, the sheet still lists the names and scores.

A sheet of many rows is written as pages: several images of at most ``DEFAULT_PAGE_ROWS``
rows each, named ``NAME-1.png``, ``NAME-2.png`` and so on (``sheet_paths``). One image of
every crop of a video is too tall to look at.

Card images are located through the catalog: a card's ``art`` entry is a URL, or a path
resolved against a folder or URL base supplied by the caller. Images are read only to draw the
sheet; nothing is stored.
"""

from __future__ import annotations

import io
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from cardvision.core.catalog import IDENTITY_NAME, Catalog, is_url
from cardvision.core.tls import CERTIFICATE_HELP, CertificatesMissing, is_certificate_failure, tls_context
from cardvision.infer.pipeline import FrameResult
from cardvision.recognize import Recognition

DEFAULT_MAX_ROWS = 60
# At most this many rows go on one image; a longer sheet is written as several.
DEFAULT_PAGE_ROWS = 20

_BACKGROUND = (24, 24, 28)
_TEXT = (232, 232, 236)
_MUTED = (150, 150, 160)
_RULE = (60, 60, 68)
_NAMED = (40, 200, 110)
_NOTHING = (240, 170, 40)

MARGIN = 8
HEADER_HEIGHT = 28
ROW_HEIGHT = 176
CROP_BOX = (150, 150)
INFO_WIDTH = 250
CANDIDATE_WIDTH = 230
THUMBNAIL_BOX = (104, 132)
CANDIDATES = 3
# Left edge of the first candidate column: everything to its right is candidates.
CANDIDATES_X = MARGIN + CROP_BOX[0] + MARGIN + INFO_WIDTH
_LINE_HEIGHT = 14
_NAME_CHARACTERS = 36


@dataclass(frozen=True)
class SheetRow:
    """One row of the sheet: a crop, where it came from, and what was concluded."""

    title: str
    crop: np.ndarray
    recognition: Recognition


def sheet_rows(result: FrameResult) -> list[SheetRow]:
    """The rows for one frame: one per recognized face, in the frame's order."""
    return [SheetRow(title=f"{result.name} #{card.index:02d}", crop=card.crop, recognition=card.recognition)
            for card in result.faces]


def _fetch(url: str, timeout: float) -> bytes:
    context = tls_context() if url.lower().startswith("https://") else None
    try:
        with urllib.request.urlopen(url, timeout=timeout, context=context) as response:  # noqa: S310 - http(s) only, see ArtLibrary
            return response.read()
    except OSError as error:
        if is_certificate_failure(error):
            raise CertificatesMissing(f"{url}: {CERTIFICATE_HELP}") from None
        raise


class ArtLibrary:
    """Card images, looked up through the catalog and kept in memory for the run.

    ``base`` is the folder or URL prefix that relative ``art`` entries are resolved against;
    with no base, only entries that are themselves URLs can be found. A card whose image cannot
    be located, read or decoded simply has none: ``image`` returns ``None`` and the id is added
    to ``missing``. After ``give_up_after`` remote failures in a row no further download is
    attempted, so an unreachable server costs a few timeouts and not one per card. A server
    whose certificate cannot be verified ends downloading at once, and ``certificate_problem``
    holds what to install.
    """

    def __init__(
        self,
        catalog: Catalog,
        base: str | Path | None = None,
        *,
        timeout: float = 10.0,
        give_up_after: int = 3,
        fetch: Callable[[str, float], bytes] = _fetch,
    ) -> None:
        self.catalog = catalog
        self.base = base
        self.timeout = timeout
        self.give_up_after = give_up_after
        self.missing: set[str] = set()
        self.certificate_problem: str | None = None
        self._fetch = fetch
        self._images: dict[str, Image.Image | None] = {}
        self._remote_failures = 0

    def _load(self, card_id: str) -> Image.Image | None:
        try:
            location = self.catalog.art_location(card_id, self.base)
        except (KeyError, ValueError):
            return None
        try:
            if is_url(location):
                if self._remote_failures >= self.give_up_after or self.certificate_problem is not None:
                    return None
                try:
                    data = self._fetch(location, self.timeout)
                except CertificatesMissing as error:    # every download would fail alike
                    self.certificate_problem = str(error)
                    return None
                except Exception:
                    self._remote_failures += 1
                    return None
                self._remote_failures = 0
                source: io.BytesIO | str = io.BytesIO(data)
            else:
                source = location
            with Image.open(source) as image:
                return image.convert("RGB")
        except (OSError, ValueError):
            return None

    def image(self, card_id: str) -> Image.Image | None:
        """The card's image as RGB, or ``None`` when it is unavailable."""
        if card_id not in self._images:
            self._images[card_id] = self._load(card_id)
            if self._images[card_id] is None:
                self.missing.add(card_id)
        return self._images[card_id]


def _shorten(text: str, limit: int = _NAME_CHARACTERS) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def sheet_size(row_count: int) -> tuple[int, int]:
    """``(width, height)`` in pixels of a sheet with ``row_count`` rows."""
    width = CANDIDATES_X + CANDIDATES * CANDIDATE_WIDTH + MARGIN
    return width, HEADER_HEIGHT + row_count * ROW_HEIGHT


def contact_sheet(rows: Sequence[SheetRow], art: ArtLibrary | None = None) -> Image.Image:
    """Draw the sheet for ``rows``. Without ``art``, or for a card whose image is unavailable,
    the candidate is shown by name and score alone."""
    width, height = sheet_size(len(rows))
    sheet = Image.new("RGB", (width, height), _BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    info_x = MARGIN + CROP_BOX[0] + MARGIN
    first_candidate_x = CANDIDATES_X
    draw.text((MARGIN, 8), "crop", fill=_MUTED, font=font)
    draw.text((info_x, 8), "decision", fill=_MUTED, font=font)
    # The candidates are of different identities. Under the name rule that means other names.
    by_name = art is None or art.catalog.identity == IDENTITY_NAME
    draw.text((first_candidate_x, 8), "best differently named cards" if by_name else "best different cards",
              fill=_MUTED, font=font)

    for position, row in enumerate(rows):
        top = HEADER_HEIGHT + position * ROW_HEIGHT
        draw.line((0, top, width, top), fill=_RULE)
        found = row.recognition

        crop = ImageOps.contain(Image.fromarray(row.crop).convert("RGB"), CROP_BOX)
        sheet.paste(crop, (MARGIN, top + MARGIN))

        decision = "NAMED" if found.accepted else "NOTHING"
        lines = [
            (_shorten(row.title), _MUTED),
            (decision, _NAMED if found.accepted else _NOTHING),
            (_shorten(found.name) if found.name is not None else "-", _TEXT),
            (found.card_id or "-", _TEXT),
            (f"score {found.similarity:.3f}   margin {found.margin:.3f}", _TEXT),
            (f"look {found.look}" + ("   turned looks tried" if found.retried else ""), _MUTED),
        ]
        for index, (text, color) in enumerate(lines):
            draw.text((info_x, top + MARGIN + index * _LINE_HEIGHT), text, fill=color, font=font)

        for slot, candidate in enumerate(found.names[:CANDIDATES]):
            x = first_candidate_x + slot * CANDIDATE_WIDTH
            text_top = top + MARGIN
            picture = art.image(candidate.card_id) if art is not None else None
            if picture is not None:
                thumbnail = ImageOps.contain(picture, THUMBNAIL_BOX)
                sheet.paste(thumbnail, (x, top + MARGIN))
                if slot == 0 and found.accepted:
                    draw.rectangle(
                        (x - 2, top + MARGIN - 2, x + thumbnail.width + 1, top + MARGIN + thumbnail.height + 1),
                        outline=_NAMED, width=2,
                    )
                text_top += THUMBNAIL_BOX[1] + 4
            draw.text((x, text_top), _shorten(candidate.name), fill=_TEXT, font=font)
            draw.text((x, text_top + _LINE_HEIGHT), f"{candidate.card_id}   {candidate.similarity:.3f}", fill=_MUTED, font=font)
    return sheet


def sheet_paths(path: str | Path, row_count: int, page_rows: int = DEFAULT_PAGE_ROWS) -> list[Path]:
    """The image files a sheet of ``row_count`` rows is written to, in order.

    ``path`` itself when the rows fit on one page of ``page_rows`` rows, and also when there
    are none. Otherwise one file per page, the page's number added to the name:
    ``sheet.png`` becomes ``sheet-1.png``, ``sheet-2.png`` and so on.
    """
    if page_rows < 1:
        raise ValueError(f"a page of the contact sheet has at least 1 row, got {page_rows}")
    path = Path(path)
    pages = max(1, -(-row_count // page_rows))
    if pages == 1:
        return [path]
    return [path.with_name(f"{path.stem}-{number}{path.suffix}") for number in range(1, pages + 1)]


def write_contact_sheet(
    path: str | Path,
    rows: Sequence[SheetRow],
    art: ArtLibrary | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
    page_rows: int = DEFAULT_PAGE_ROWS,
) -> int:
    """Save the sheet for the first ``max_rows`` rows as image files (PNG is recommended).

    At most ``page_rows`` rows go on one image. Rows that fit on one page are written to
    ``path``; more are written as pages, to the files ``sheet_paths`` names. Returns the number
    of rows drawn. With no rows, a sheet holding only its header is written to ``path``, so
    that the file always exists after a run that asked for it.
    """
    shown = list(rows[:max_rows])
    files = sheet_paths(path, len(shown), page_rows)
    files[0].parent.mkdir(parents=True, exist_ok=True)
    for number, file in enumerate(files):
        contact_sheet(shown[number * page_rows:(number + 1) * page_rows], art).save(file)
    return len(shown)
