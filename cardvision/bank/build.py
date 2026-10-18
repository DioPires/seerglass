# SPDX-License-Identifier: Apache-2.0
"""Build an embedding bank, and the identity groups that go with it, from card images.

A bank row
----------
For each card: take its views (``cardvision.bank.views``: the image and 15 synthesized copies),
preprocess each one exactly as a crop is preprocessed at run time, embed them with the
recognizer, scale each embedding to unit length, average them, and scale the average to unit
length again. The row is then put in the form a reader will see after loading it (narrowed to
fp16, widened, renormalized: ``cardvision.core.bank.fp16_roundtrip``) and stored as fp16.

A row depends on its own card only, so the cards can be embedded in any grouping.

The bank file
-------------
The format ``cardvision.core.bank`` reads, with this header::

    {"schemaVersion": 1, "kind": "embedding-bank", "game": "toy", "locale": "en",
     "version": 1, "model": "<label of the recognizer>", "dim": 128, "dtype": "fp16",
     "preprocessing": {"id": "full128", "size": 128, "centerMask": false, "poolVersion": 2,
                       "crop": "truncate"},
     "centerMask": false, "views": 16, "aggregation": "mean",
     "cardCount": 20, "entryCount": 20,
     "entries": [{"cardId": "TOY-001", "vec": "<fp16, base64>", "artV": "<16 hex>"}, ...]}

``version`` is a whole number that tells one bank of a game from the next. ``model`` is a free
label for the recognizer that produced the rows; a release requires it to equal the label of
the recognizer it pins. ``centerMask`` repeats the preprocessing contract's value at the top
level for readers that look for it there. ``tau`` is present only when the caller asked for
it: it records the accept threshold the bank is meant to be used with, for runtimes that read
the threshold from the bank. A release requires it to equal the release's own.

``artV`` records the image a row was embedded from: the first 16 lower-case hexadecimal digits
of the SHA-256 of the bytes of the image file, or of the bytes fetched when the art comes from
a URL (``cardvision.core.bank.art_version``). Every row this module embeds carries it. A bank
written before the field existed has none; it still loads, and its rows are of unknown image.

A card without art (``"art": null`` in the catalog, ``cardvision.core.catalog``) has no image
to embed, so a bank built from the catalog has no row for it; ``BankBuild.no_art`` lists such
cards. Its identity still counts in the catalog, and a bank, its identity groups and a release
all hold without it: they speak of the bank's ids only. A bank extended from one that had a row
for the card keeps that row (below), since a refresh never removes a card.

Rows are in catalog order. Nothing in the file depends on when or where it was built, but the
row values can differ in the last fp16 step between processor architectures and between
versions of the imaging and inference libraries.

Extending a bank
----------------
Because of that last point, and because a published bank has been measured as it is, a bank is
extended, not rebuilt, when cards are added: ``build_bank(..., base=old)`` keeps the old bank's
rows in their old order, then appends the catalog's other cards in catalog order. An old
bank holding a card the catalog no longer has is refused: a refresh never removes a card.

What happens to an old row depends on the image it records:

* its ``artV`` equals the hash of the image now at the art location: the entry is copied
  unchanged, byte for byte;
* its ``artV`` differs: the card's image has changed since the row was made, so the row is
  embedded again from the current image, in the same position, and reported;
* it has no ``artV``: the build cannot tell, so the entry is copied unchanged and reported as
  "image not recorded". With ``reembed_unrecorded`` such a row is embedded again instead, which
  also makes it record its image. On the same machine and library versions, an unchanged image
  gives back the same bytes;
* its image is not found at the art location: the entry is copied unchanged and reported;
* the catalog now lists the card without art: the entry is copied unchanged and reported apart
  (``BankBuild.no_art_kept``). Only a bank built without a base drops such a row, and that is
  not an additive refresh.

So after a refresh a row can still be behind its card's current image in two cases: the old
bank did not record the image, or the image was not there to compare.
``cardvision.bank.art.check_bank_art`` reports both, and the rows whose image differs, for any
bank.

Identity groups
---------------
Which card ids are one identity is the catalog's rule (``cardvision.core.catalog``,
"Identity"). By default two ids are one identity when their catalog names are exactly equal;
a catalog that states ``"identity": "printing"`` joins an entry with the entries whose
``printingOf`` names it, and nothing else. How similar two printings look enters neither rule.
The groups file lists every identity that has two or more ids in the bank::

    {"schemaVersion": 1, "kind": "identity-groups", "game": "toy", "locale": "en", "version": 1,
     "bank": {"version": 1, "sha256": "<of the bank file>"},
     "rule": {"level": "name"}, "identityCount": 18,
     "groups": [["TOY-003", "TOY-003A"], ...],
     "representative": {"TOY-003A": "TOY-003", ...}}

``rule.level`` records the catalog's rule the groups were made by, ``name`` or ``printing``.
``representative`` maps every other member of a group to the group's representative. Under
``name`` that is the member with the smallest ``cardvision.core.bank.representative_key``.
Under ``printing`` it is the entry the others are printings of, when the bank holds it, and
otherwise the member with the smallest key. The representative is a key for "the same
identity" and nothing more: which printing to show for a match is a separate setting of the
release. ``identityCount`` counts the identities over the bank's ids, single cards included.
"""

from __future__ import annotations

import io
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cardvision.bank.documents import canonical_bytes, sha256_hex, write_document
from cardvision.bank.suspect import file_problem, picture_problem
from cardvision.bank.views import VIEWS_PER_CARD, card_views
from cardvision.core.bank import (
    ART_VERSION_KEY,
    IdentityGroups,
    art_version,
    encode_bank,
    fp16_roundtrip,
    l2_normalize,
    stored_art_versions,
    stored_vectors,
)
from cardvision.core.catalog import Catalog, is_url
from cardvision.core.onnx import output_width, run_batched
from cardvision.core.preprocess import CONTRACTS, FULL128, POOL_V2, input_size, preprocessor
from cardvision.core.tls import CERTIFICATE_HELP, CertificatesMissing, is_certificate_failure, tls_context

BANK_SCHEMA_VERSION = 1
BANK_KIND = "embedding-bank"
GROUPS_SCHEMA_VERSION = 1
GROUPS_KIND = "identity-groups"
VECTOR_DTYPE_NAME = "fp16"
AGGREGATION = "mean"

# Preprocessing contracts as data, by id: what a bank and a release write down so that any
# runtime can tell whether it implements the contract the rows were made with.
#   size        side of the square model input, in pixels
#   centerMask  whether part of the crop is masked out before pooling
#   poolVersion the pooling rule (cardvision.core.preprocess)
#   crop        how a box that reaches past the frame is handled: "truncate" cuts it at the
#               frame edge, without padding
PREPROCESSING: dict[str, dict[str, Any]] = {
    FULL128: {"id": FULL128, "size": input_size(FULL128), "centerMask": False, "poolVersion": POOL_V2,
              "crop": "truncate"},
}

Progress = Callable[[int, int], None]


class BankBuildError(ValueError):
    """The bank cannot be built from these inputs. The message says why."""


def preprocessing_descriptor(contract: str) -> dict[str, Any]:
    """The description of a preprocessing contract that bank and release documents carry."""
    if contract not in PREPROCESSING or contract not in CONTRACTS:
        raise ValueError(f"unknown preprocessing contract {contract!r} (known: {sorted(PREPROCESSING)})")
    return dict(PREPROCESSING[contract])


def read_art(location: str, timeout: float = 60.0) -> bytes:
    """The bytes of a card image file, from a local path or an ``http(s)`` URL.

    Raises ``CertificatesMissing`` when an https server's certificate cannot be verified."""
    if not is_url(location):
        return Path(location).read_bytes()
    request = urllib.request.Request(location, headers={"User-Agent": "cardvision"})
    context = tls_context() if location.lower().startswith("https://") else None
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:  # noqa: S310 - http(s) only
            return response.read()
    except OSError as error:
        if is_certificate_failure(error):
            raise CertificatesMissing(f"{location}: {CERTIFICATE_HELP}") from None
        raise


def decode_art(data: bytes) -> np.ndarray:
    """The bytes of an image file as ``(H, W, 3)`` uint8 RGB, as ``cardvision.core.images`` reads a file."""
    with Image.open(io.BytesIO(data)) as image:
        return np.array(image.convert("RGB"))


def load_art(location: str, timeout: float = 60.0) -> np.ndarray:
    """Read a card image from a local path or an ``http(s)`` URL as ``(H, W, 3)`` uint8 RGB."""
    return decode_art(read_art(location, timeout))


def art_is_missing(error: Exception) -> bool:
    """Whether an error from ``read_art`` means that there is no image at the location."""
    return isinstance(error, FileNotFoundError) or (isinstance(error, urllib.error.HTTPError) and error.code == 404)


def embed_card(session: Any, art: np.ndarray, card_id: str, contract: str = FULL128,
               views: int = VIEWS_PER_CARD) -> np.ndarray:
    """The unit-length embeddings of one card's views: ``(views, D)`` float32."""
    preprocess = preprocessor(contract)
    embeddings = run_batched(session, [preprocess(view) for view in card_views(art, card_id, views)])
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


def rows_from_views(per_view: np.ndarray) -> np.ndarray:
    """Bank rows from ``(N, views, D)`` unit embeddings: ``(N, D)`` float32, as a reader sees them.

    The views are averaged, the mean is scaled to unit length, and the result is passed through
    the storage round trip, so that encoding these rows stores exactly what a reader decodes.
    """
    return fp16_roundtrip(l2_normalize(per_view.mean(1)))


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _check_base(base: Mapping[str, Any], catalog: Catalog, model: str | None, dim: int,
                descriptor: Mapping[str, Any], views: int) -> tuple[list[str], list[str | None]]:
    """Validate the bank being extended and return its card ids and their ``artV``, in order."""
    if not isinstance(base, Mapping) or base.get("kind") != BANK_KIND:
        raise BankBuildError(f'the base bank is not an "{BANK_KIND}" document')
    try:
        base_ids, stored = stored_vectors(base)
        recorded = stored_art_versions(base)
    except ValueError as error:
        raise BankBuildError(f"the base bank cannot be read: {error}") from None
    if not base_ids:
        raise BankBuildError("the base bank has no entries")
    repeated = sorted(card_id for card_id, count in Counter(base_ids).items() if count > 1)
    if repeated:
        raise BankBuildError(
            f"the base bank has more than one row for {len(repeated)} card id(s) (e.g. {repeated[:3]}); "
            "only a bank with one row per card can be extended"
        )
    if stored.shape[1] != dim:
        raise BankBuildError(f"the base bank's rows have {stored.shape[1]} values, the recognizer outputs {dim}")
    removed = [card_id for card_id in base_ids if card_id not in catalog]
    if removed:
        raise BankBuildError(
            f"{len(removed)} card id(s) of the base bank are not in the catalog (e.g. {removed[:5]}): "
            "extending a bank only adds cards, it never removes one"
        )
    expected = {"game": catalog.game, "locale": catalog.locale, "dim": dim, "dtype": VECTOR_DTYPE_NAME,
                "preprocessing": dict(descriptor), "views": views, "aggregation": AGGREGATION}
    if model is not None:
        expected["model"] = model
    for key, value in expected.items():
        if key in base and base[key] != value:
            raise BankBuildError(
                f'the base bank was built with "{key}" {base[key]!r}, this build uses {value!r}: '
                "rows made differently cannot share a bank"
            )
    return base_ids, recorded


@dataclass
class BankBuild:
    """The result of ``build_bank``.

    Attributes:
        bank: the bank document.
        embedded: the new card ids embedded and appended in this build, in row order.
        reused: the card ids whose entries were copied unchanged from the base bank, in row
            order.
        reembedded: the card ids of the base bank whose rows were embedded again, in place,
            in row order: their image differs from the one the base row records, or the base
            row records none and the build was told to embed such rows again.
        art_versions: for each re-embedded id, the ``artV`` of the base row (``None`` when it
            records none) and the ``artV`` of the image the new row was embedded from.
        unrecorded: the reused ids whose base row records no image, so that the build could not
            tell whether the image has changed.
        art_missing: the reused ids whose image was not found, so that it could not be compared
            with the one the base row records, or embedded again.
        skipped: catalog cards left out because their image was not found (only when the
            build was told to skip them).
        no_art: the catalog's cards without art (``"art": null``) that the bank has no row
            for, in catalog order. They are never embedded.
        no_art_kept: the reused ids whose card the catalog now lists without art: the base
            row is copied unchanged, in row order. They are in no other list but ``reused``.
        placeholders: the cards embedded in this build whose image cannot be a card's, each as
            ``(card id, reason)``: a tiny picture, or one flat color
            (``cardvision.bank.suspect``). They are embedded all the same, and reported.
    """

    bank: dict[str, Any]
    embedded: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)
    reembedded: list[str] = field(default_factory=list)
    art_versions: dict[str, tuple[str | None, str]] = field(default_factory=dict)
    unrecorded: list[str] = field(default_factory=list)
    art_missing: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    placeholders: list[tuple[str, str]] = field(default_factory=list)
    no_art: list[str] = field(default_factory=list)
    no_art_kept: list[str] = field(default_factory=list)


def _read_card_art(card_id: str, location: str) -> bytes | None:
    """The bytes of a card's image file, or ``None`` when there is no image at the location."""
    try:
        return read_art(location)
    except CertificatesMissing as error:   # every image would fail alike: stop, and say what to install
        raise BankBuildError(str(error)) from None
    except Exception as error:  # the file system and the network raise many kinds
        if art_is_missing(error):
            return None
        raise BankBuildError(f"the image of card {card_id!r} at {location} cannot be read ({error})") from None


def build_bank(
    catalog: Catalog,
    art: str | Path,
    session: Any,
    *,
    model: str | None = None,
    version: int | None = None,
    base: Mapping[str, Any] | None = None,
    tau: float | None = None,
    contract: str = FULL128,
    skip_missing_art: bool = False,
    reembed_unrecorded: bool = False,
    progress: Progress | None = None,
) -> BankBuild:
    """Embed the catalog's cards into a bank document.

    Args:
        catalog: the game's cards.
        art: the folder, or the ``http(s)`` URL prefix, that each card's ``art`` is resolved
            against.
        session: the recognizer, as an ONNX Runtime session (``cardvision.core.onnx``).
        model: the label written as the bank's ``model``. Required for a new bank; when
            extending, it defaults to the base bank's and must equal it.
        version: the bank's version number. Defaults to 1, or to one more than the base bank's.
        base: an existing bank document to extend. Its rows keep their order. A row is copied
            unchanged unless the image it records differs from the image now at ``art``, in
            which case it is embedded again in place; the catalog's other cards are embedded
            and appended (see the module docstring).
        tau: an accept threshold to record in the bank (see the module docstring). When
            extending, it defaults to the base bank's.
        contract: the id of the preprocessing contract the recognizer was trained with.
        skip_missing_art: leave out a new card whose image is not found instead of failing.
        reembed_unrecorded: when extending, also embed again every row of the base bank that
            records no image, instead of copying it.
        progress: called as ``progress(done, total)`` after each embedded card.

    Raises ``BankBuildError`` when the inputs do not fit together.
    """
    descriptor = preprocessing_descriptor(contract)
    views = VIEWS_PER_CARD
    dim = output_width(session)

    base_ids: list[str] = []
    base_recorded: list[str | None] = []
    if base is not None:
        base_ids, base_recorded = _check_base(base, catalog, model, dim, descriptor, views)
        model = model if model is not None else base.get("model")
        if version is None and _positive_int(base.get("version")):
            version = base["version"] + 1
        if tau is None and "tau" in base:
            tau = base["tau"]
    if not isinstance(model, str) or not model.strip():
        raise BankBuildError('a bank needs a "model" label naming the recognizer that made its rows')
    version = 1 if version is None else version
    if not _positive_int(version):
        raise BankBuildError(f"the bank version must be a whole number of at least 1, got {version!r}")
    if base is not None and _positive_int(base.get("version")) and version <= base["version"]:
        raise BankBuildError(
            f"the bank version must be greater than the base bank's {base['version']}, got {version}"
        )
    if tau is not None and (isinstance(tau, bool) or not isinstance(tau, (int, float)) or not 0 < tau <= 1):
        raise BankBuildError(f"tau must be a number above 0 and at most 1, got {tau!r}")

    in_base = set(base_ids)
    # A card without art has no image to embed: it gets no row (a base row for one is kept, below,
    # and listed as no_art_kept).
    no_art = [card_id for card_id in catalog.ids_without_art() if card_id not in in_base]
    wanted = [card_id for card_id in catalog.ids_with_art() if card_id not in in_base]
    locations = {card_id: catalog.art_location(card_id, art)
                 for card_id in [*base_ids, *wanted] if catalog.card(card_id).has_art}
    if not is_url(str(art)):
        # A local folder is checked up front, so that every missing image is reported at once.
        absent = [card_id for card_id in wanted
                  if not is_url(locations[card_id]) and not Path(locations[card_id]).is_file()]
        if absent and not skip_missing_art:
            raise BankBuildError(
                f"no image for {len(absent)} card(s), for example {absent[0]!r} at {locations[absent[0]]}"
            )

    # The base rows: which are copied, and which are embedded again from the image there now.
    reused: list[str] = []
    unrecorded: list[str] = []
    art_missing: list[str] = []
    no_art_kept: list[str] = []
    changed: dict[str, bytes] = {}
    art_versions: dict[str, tuple[str | None, str]] = {}
    for card_id, recorded in zip(base_ids, base_recorded):
        if card_id not in locations:  # the catalog now lists the card without art: keep its row
            reused.append(card_id)
            no_art_kept.append(card_id)
            continue
        if recorded is None and not reembed_unrecorded:
            reused.append(card_id)
            unrecorded.append(card_id)
            continue
        data = _read_card_art(card_id, locations[card_id])
        if data is None:
            reused.append(card_id)
            (unrecorded if recorded is None else art_missing).append(card_id)
            continue
        current = art_version(data)
        if recorded == current:
            reused.append(card_id)
        else:
            changed[card_id] = data
            art_versions[card_id] = (recorded, current)
    reembedded = list(changed)

    total = len(reembedded) + len(wanted)
    done = 0
    embedded: list[str] = []
    skipped: list[str] = []
    placeholders: list[tuple[str, str]] = []
    made: list[str] = []  # every card embedded in this build: the base rows first, then the new cards
    made_versions: list[str] = []
    per_view: list[np.ndarray] = []
    for card_id in [*reembedded, *wanted]:
        done += 1
        data = changed.pop(card_id, None)
        if data is None:
            data = _read_card_art(card_id, locations[card_id])
        if data is None:
            if skip_missing_art:
                skipped.append(card_id)
                continue
            raise BankBuildError(f"the image of card {card_id!r} at {locations[card_id]} not found")
        try:
            image = decode_art(data)
        except Exception as error:  # the image library raises many kinds
            raise BankBuildError(
                f"the image of card {card_id!r} at {locations[card_id]} cannot be read "
                f"({file_problem(data) or error}); cardvision check-bank-art --catalog ... --art ..., with no "
                "bank, lists every such image at once"
            ) from None
        problem = picture_problem(image)
        if problem is not None:
            placeholders.append((card_id, problem))
        per_view.append(embed_card(session, image, card_id, contract, views))
        made.append(card_id)
        made_versions.append(art_version(data))
        if card_id not in in_base:
            embedded.append(card_id)
        if progress is not None:
            progress(done, total)

    if not base_ids and not embedded:
        raise BankBuildError("no card could be embedded: the bank would be empty"
                             + (f" ({len(no_art)} card(s) have no art)" if no_art else ""))
    made_entries: dict[str, dict[str, Any]] = {}
    if made:
        rows = rows_from_views(np.stack(per_view).astype(np.float32))
        if not np.isfinite(rows).all():
            bad = [card_id for card_id, row in zip(made, rows) if not np.isfinite(row).all()]
            raise BankBuildError(f"the recognizer gave an embedding that cannot be normalized for {bad[:3]}")
        made_entries = dict(zip(made, encode_bank(made, rows, art_versions=made_versions)["entries"]))

    entries: list[dict[str, Any]] = []
    for entry in (base["entries"] if base is not None else []):
        again = made_entries.get(entry["cardId"])
        # A row embedded again keeps its position and any other key of its entry.
        entries.append(dict(entry) if again is None else
                       {**entry, "vec": again["vec"], ART_VERSION_KEY: again[ART_VERSION_KEY]})
    entries += [made_entries[card_id] for card_id in embedded]
    header: dict[str, Any] = {
        "schemaVersion": BANK_SCHEMA_VERSION,
        "kind": BANK_KIND,
        "game": catalog.game,
        "locale": catalog.locale,
        "version": version,
        "model": model,
        "dim": dim,
        "dtype": VECTOR_DTYPE_NAME,
        "preprocessing": descriptor,
        "centerMask": descriptor["centerMask"],
    }
    if tau is not None:
        header["tau"] = tau
    header.update({"views": views, "aggregation": AGGREGATION,
                   "cardCount": len({entry["cardId"] for entry in entries}), "entryCount": len(entries)})
    return BankBuild(bank={**header, "entries": entries}, embedded=embedded, reused=reused,
                     reembedded=reembedded, art_versions=art_versions, unrecorded=unrecorded,
                     art_missing=art_missing, skipped=skipped, placeholders=placeholders, no_art=no_art,
                     no_art_kept=no_art_kept)


def identity_groups_document(bank: Mapping[str, Any], bank_sha256: str, catalog: Catalog,
                             version: int | None = None) -> dict[str, Any]:
    """The identity groups of a bank under the catalog's identity rule (see the module
    docstring).

    ``bank_sha256`` is the hash of the bank file the groups are for, which ties the two
    together. ``version`` defaults to the bank's. Every bank id must be in the catalog.
    """
    ids, _ = stored_vectors(bank)
    unknown = sorted({card_id for card_id in ids if card_id not in catalog})
    if unknown:
        raise BankBuildError(f"{len(unknown)} bank id(s) are not in the catalog (e.g. {unknown[:5]})")
    version = bank.get("version") if version is None else version
    if not _positive_int(version):
        raise BankBuildError(f"the identity groups version must be a whole number of at least 1, got {version!r}")
    groups = catalog.identity_groups(ids)
    identity = IdentityGroups(groups, ids, prefer=catalog.identity_anchors())
    return {
        "schemaVersion": GROUPS_SCHEMA_VERSION,
        "kind": GROUPS_KIND,
        "game": catalog.game,
        "locale": catalog.locale,
        "version": version,
        "bank": {"version": bank.get("version"), "sha256": bank_sha256},
        "rule": {"level": catalog.identity},
        "identityCount": len(identity.identity_ids),
        "groups": groups,
        "representative": identity.representative_map(),
    }


def groups_path_for(bank_path: str | Path) -> Path:
    """Where the identity groups of a bank file are written by default: beside it."""
    bank_path = Path(bank_path)
    return bank_path.with_name(bank_path.stem + ".identity-groups.json")


def write_bank(bank: Mapping[str, Any], catalog: Catalog, out: str | Path,
               groups_out: str | Path | None = None, overwrite: bool = False) -> tuple[Path, Path]:
    """Write a bank document and its identity groups. Returns the two paths.

    Both files are checked before either is written, so a refusal to replace one of them
    leaves both as they were.
    """
    out = Path(out)
    groups_out = groups_path_for(out) if groups_out is None else Path(groups_out)
    bank_body = canonical_bytes(bank)
    groups_body = canonical_bytes(identity_groups_document(bank, sha256_hex(bank_body), catalog))
    if not overwrite:
        for path, body in ((out, bank_body), (groups_out, groups_body)):
            if path.exists() and path.read_bytes() != body:
                write_document(path, body)  # raises, naming the file
    write_document(out, bank_body, overwrite)
    write_document(groups_out, groups_body, overwrite)
    return out, groups_out
