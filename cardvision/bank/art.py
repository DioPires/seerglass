# SPDX-License-Identifier: Apache-2.0
"""Compare the image each bank row records with the image a card has now.

A bank row is embedded from one image file, and records which one as ``artV``
(``cardvision.core.bank``). The card's image can change afterwards: a preview picture is
replaced by the final one, a scan is redone. The row then lags behind the image until it is
embedded again. ``check_bank_art`` reads a bank, a catalog and the current images, and sorts
every row into one of these:

* **matching**: the row's ``artV`` is the hash of the image there now;
* **changed**: the row was embedded from a different image, and both hashes are given;
* **unrecorded**: the row has no ``artV``, so nothing can be said about it;
* **missing**: the row has an ``artV`` and there is no image for its card at the art location;
* **unreadable**: the row has an ``artV`` and its card's image could not be read;
* **unknown**: the row's card id is not in the catalog, so it has no image to look for;
* **no art**: the row has an ``artV`` and the catalog now lists its card without art
  (``"art": null``), so it has no image to look for either.

The check reads and reports. It changes nothing, and it does not say whether the bank may be
used: what a row embedded from an earlier image costs is for a measurement to say.
``cardvision build-bank --base`` embeds the changed rows again.

Beside that sorting, the report lists what deserves a second look
(``cardvision.bank.suspect``): image files that hold no picture, pictures that look like
placeholders, one image file under cards of different identities, and rows of different
identities that are nearly the same. ``check_art`` looks at the images alone, for every card
of a catalog, before any bank exists; it lists the cards without art apart, since they have no
image to look at.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cardvision.bank.build import art_is_missing, read_art
from cardvision.bank.suspect import Suspects, bank_suspects, image_problem, shared_images
from cardvision.core.bank import art_version, stored_art_versions, stored_vectors
from cardvision.core.catalog import Catalog
from cardvision.core.tls import CertificatesMissing


@dataclass
class BankArtReport:
    """What ``check_bank_art`` found. Every list is in the bank's row order.

    Attributes:
        rows: how many rows the bank has. It equals the lengths of the seven lists together.
        matching: the card ids of rows embedded from the image there now.
        changed: the rows embedded from a different image, each as
            ``(card id, recorded artV, artV of the image there now)``.
        unrecorded: the card ids of rows that record no image.
        missing: the card ids of rows that record an image and have none at the art location.
        unreadable: the rows whose image could not be read, each as ``(card id, reason)``.
        unknown: the card ids of rows whose card is not in the catalog.
        no_art: the card ids of rows that record an image and whose card the catalog now lists
            without art.
        suspects: what deserves a second look (``cardvision.bank.suspect``). It is beside the
            seven lists, not one of them: a row can match its image and that image be a
            placeholder.
    """

    rows: int = 0
    matching: list[str] = field(default_factory=list)
    changed: list[tuple[str, str, str]] = field(default_factory=list)
    unrecorded: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unreadable: list[tuple[str, str]] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    no_art: list[str] = field(default_factory=list)
    suspects: Suspects = field(default_factory=Suspects)

    @property
    def recorded(self) -> int:
        """How many rows record their image."""
        return self.rows - len(self.unrecorded)


def check_bank_art(bank: Mapping[str, Any], catalog: Catalog, art: str | Path) -> BankArtReport:
    """Sort a bank's rows by how the image each records compares with the image there now.

    ``art`` is the folder, or the ``http(s)`` URL prefix, that each card's ``art`` is resolved
    against; from a URL prefix every recorded row's image is fetched. Any document that
    ``cardvision.core.bank`` can read is accepted, with or without the header a build writes.
    Raises ``ValueError`` when the entries cannot be read, and ``CertificatesMissing`` when an
    https server's certificate cannot be verified.
    """
    ids, _ = stored_vectors(bank)
    recorded = stored_art_versions(bank)
    report = BankArtReport(rows=len(ids))
    # The images are not needed for these two: the rows record which file each was made from.
    try:
        report.suspects = bank_suspects(bank, catalog)
    except ValueError:
        pass  # rows that cannot be normalized are compared with nothing
    problems: dict[str, tuple[str, str] | None] = {}  # by location: what is wrong with the image there now
    current: dict[str, str | None | Exception] = {}  # by location, so that an image is read once
    for card_id, version in zip(ids, recorded):
        if version is None:
            report.unrecorded.append(card_id)
            continue
        if card_id not in catalog:
            report.unknown.append(card_id)
            continue
        if not catalog.card(card_id).has_art:
            report.no_art.append(card_id)
            continue
        location = catalog.art_location(card_id, art)
        if location not in current:
            try:
                data = read_art(location)
                current[location], problems[location] = art_version(data), image_problem(data)
            except CertificatesMissing:   # every image would fail alike: stop, and say what to install
                raise
            except Exception as error:  # the file system and the network raise many kinds
                current[location] = None if art_is_missing(error) else error
        now = current[location]
        report.suspects.add(card_id, problems.get(location))
        if now is None:
            report.missing.append(card_id)
        elif isinstance(now, Exception):
            report.unreadable.append((card_id, str(now)))
        elif now == version:
            report.matching.append(card_id)
        else:
            report.changed.append((card_id, version, now))
    return report


@dataclass
class ArtReport:
    """What ``check_art`` found among the images of a catalog's cards, in catalog order.

    Attributes:
        cards: how many cards the catalog has.
        usable: the card ids whose image is a picture that could be a card's.
        no_art: the cards without art (``"art": null``): no image is looked for.
        missing: the card ids with no image at the art location.
        unreadable: the cards whose image could not be read, each as ``(card id, reason)``.
        suspects: the image files that hold no picture, the pictures that look like
            placeholders, and the image files that cards of different identities share
            (``cardvision.bank.suspect``). A card with a shared image is also in ``usable``.
    """

    cards: int = 0
    usable: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unreadable: list[tuple[str, str]] = field(default_factory=list)
    suspects: Suspects = field(default_factory=Suspects)
    no_art: list[str] = field(default_factory=list)


def check_art(catalog: Catalog, art: str | Path) -> ArtReport:
    """Look at the image of every card of a catalog, before a bank is built from them.

    ``art`` is the folder, or the ``http(s)`` URL prefix, that each card's ``art`` is resolved
    against; from a URL prefix every image is fetched. Nothing is embedded, so no recognizer is
    needed, and nothing is changed. Raises ``CertificatesMissing`` when an https server's
    certificate cannot be verified.
    """
    report = ArtReport(cards=len(catalog))
    versions: list[tuple[str, str | None]] = []
    seen: dict[str, tuple[str, tuple[str, str] | None]] = {}  # by location, so that an image is read once
    for card in catalog:
        if not card.has_art:
            report.no_art.append(card.id)
            continue
        location = catalog.art_location(card.id, art)
        if location not in seen:
            try:
                data = read_art(location)
            except CertificatesMissing:   # every image would fail alike: stop, and say what to install
                raise
            except Exception as error:  # the file system and the network raise many kinds
                if art_is_missing(error):
                    report.missing.append(card.id)
                else:
                    report.unreadable.append((card.id, str(error)))
                continue
            seen[location] = (art_version(data), image_problem(data))
        version, problem = seen[location]
        report.suspects.add(card.id, problem)
        if problem is None:
            report.usable.append(card.id)
            versions.append((card.id, version))
    report.suspects.shared = shared_images(versions, catalog)
    return report
