# SPDX-License-Identifier: Apache-2.0
"""Card images that deserve a second look: files that hold no picture, placeholders, one image
file under different cards, and nearly the same bank row under different cards.

Every image is the toy game's, or is written here from a few bytes. Each finding is reported
and none is a failure: ``build-bank`` and ``check-bank-art`` exit with 0 whatever they list.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import toy_game
from cardvision import cli
from cardvision.bank import art as A
from cardvision.bank import build as BB
from cardvision.bank import suspect as S
from cardvision.core.bank import decode_bank
from cardvision.core.catalog import parse_catalog
from cardvision.core.onnx import create_session
from test_infer_toy_models import build_recognizer

# The first bytes of a vector file, put together here so that this file holds no drawing itself.
VECTOR = "svg"
SVG = f'<?xml version="1.0"?>\n<{VECTOR} xmlns="http://www.w3.org/2000/{VECTOR}" width="10" height="10"/>'
PAGE = "<!DOCTYPE html><html><body>Not found</body></html>"


def png(image: Image.Image) -> bytes:
    import io

    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


# ---------------------------------------------------------------------- one file, one picture
def test_a_file_that_holds_no_picture_is_told_by_its_first_bytes():
    assert S.file_problem(b"") == "an empty file"
    assert S.file_problem(SVG.encode()) == "a vector file (SVG), which has no pixels to compare"
    assert S.file_problem(f"  <{VECTOR} viewBox='0 0 1 1'/>".encode()) == "a vector file (SVG), which has no pixels to compare"
    assert S.file_problem(b"%PDF-1.7\n...") == "a PDF or PostScript file, which has no pixels to compare"
    assert S.file_problem(PAGE.encode()) == "a web page saved under an image's name"
    assert S.file_problem(b"\x00\x01 some bytes that are no image") == "not an image that Pillow can read"
    whole = png(Image.new("RGB", (64, 64), (200, 30, 30)))
    assert S.file_problem(whole) is None
    assert S.file_problem(whole[:40]) == "not an image that Pillow can read"      # a download that broke off


def test_a_picture_that_cannot_be_a_card_is_a_placeholder(toy):
    card = np.asarray(Image.open(toy[1] / "TOY-001.png").convert("RGB"))
    assert S.picture_problem(card) is None
    assert S.picture_problem(np.zeros((8, 8, 3), dtype=np.uint8)) == "a picture of 8 x 8 pixels, too small to be a card: a placeholder"
    assert S.picture_problem(card[:, :31]) == "a picture of 31 x 336 pixels, too small to be a card: a placeholder"
    assert S.picture_problem(card[:, :32]) is None and S.PLACEHOLDER_MIN_SIDE == 32
    assert S.picture_problem(np.full((336, 240, 3), 128, dtype=np.uint8)) == "a picture of one flat color: a placeholder"
    # The two together, on the bytes of a file.
    assert S.image_problem(png(Image.fromarray(card))) is None
    assert S.image_problem(SVG.encode()) == (S.NOT_PICTURE, "a vector file (SVG), which has no pixels to compare")
    assert S.image_problem(png(Image.new("RGB", (240, 336), (9, 9, 9)))) == (S.PLACEHOLDER, "a picture of one flat color: a placeholder")


# ------------------------------------------------------------ one image under different cards
def cards(identity: str | None = None) -> dict:
    """Two cards named Kite, one of them with a printing, and a card named Lamp."""
    document = {"game": "made-up", "locale": "en", "cards": [
        {"id": "AB-001", "name": "Kite", "printingOf": None, "art": "AB-001.png"},
        {"id": "AB-001A", "name": "Kite", "printingOf": "AB-001", "art": "AB-001A.png"},
        {"id": "AB-002", "name": "Kite", "printingOf": None, "art": "AB-002.png"},
        {"id": "AB-003", "name": "Lamp", "printingOf": None, "art": "AB-003.png"},
    ]}
    return document if identity is None else {**document, "identity": identity}


def test_one_image_file_under_cards_of_different_identities_is_reported():
    by_name, by_printing = parse_catalog(cards()), parse_catalog(cards("printing"))
    # A card and its printing with one file: one identity under either rule, and nothing to report.
    printing = [("AB-001", "aaaa"), ("AB-001A", "aaaa"), ("AB-002", "bbbb"), ("AB-003", "cccc")]
    assert S.shared_images(printing, by_name) == S.shared_images(printing, by_printing) == []
    # Kite and Lamp with one file: different cards under either rule.
    crossed = [("AB-001", "aaaa"), ("AB-003", "aaaa"), ("AB-002", "bbbb"), ("AB-001A", None)]
    assert S.shared_images(crossed, by_name) == S.shared_images(crossed, by_printing) == [["AB-001", "AB-003"]]
    # The two Kites with one file: one identity by name, two different cards by printing.
    namesakes = [("AB-002", "aaaa"), ("AB-001A", "aaaa"), ("AB-001", "aaaa"), ("AB-003", "cccc")]
    assert S.shared_images(namesakes, by_name) == []
    assert S.shared_images(namesakes, by_printing) == [["AB-001", "AB-001A", "AB-002"]]
    # A card the catalog does not hold is an identity of its own.
    assert S.shared_images([("AB-003", "dddd"), ("ZZ-9", "dddd")], by_name) == [["AB-003", "ZZ-9"]]


def test_rows_of_different_identities_that_are_nearly_the_same_are_grouped():
    by_name, by_printing = parse_catalog(cards()), parse_catalog(cards("printing"))
    ids = ["AB-001", "AB-001A", "AB-002", "AB-003"]
    base = np.eye(4, 16, dtype=np.float32)
    rows = base.copy()
    rows[2] = base[0] + 0.05 * base[2]          # AB-002 almost on AB-001: cosine 0.9988
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    assert float(rows[0] @ rows[2]) == pytest.approx(0.99875, abs=1e-4)
    assert S.near_identical_rows(ids, rows, by_name) == []                       # two Kites: one identity by name
    assert S.near_identical_rows(ids, rows, by_printing) == [["AB-001", "AB-002"]]
    rows[3] = rows[0]                                                           # and Lamp exactly on AB-001
    assert S.near_identical_rows(ids, rows, by_name) == [["AB-001", "AB-002", "AB-003"]]   # joined through AB-001
    assert S.near_identical_rows(ids, rows, by_printing) == [["AB-001", "AB-002", "AB-003"]]
    assert S.near_identical_rows(ids, rows, by_printing, threshold=0.9999) == [["AB-001", "AB-003"]]
    # Just under the limit is not reported; the limit itself is.
    rows = base.copy()
    rows[3] = 0.98 * base[0] + np.sqrt(1 - 0.98 ** 2) * base[3]
    assert S.near_identical_rows(ids, rows, by_name) == []
    rows[3] = 0.995 * base[0] + np.sqrt(1 - 0.995 ** 2) * base[3]
    assert S.near_identical_rows(ids, rows, by_name) == [["AB-001", "AB-003"]] and S.NEAR_IDENTICAL_COSINE == 0.99
    assert S.near_identical_rows(ids[:1], rows[:1], by_name) == []


# ----------------------------------------------------------------- the toy game, and the limit
def staged(folder: Path, toy_art: Path) -> tuple[Path, Path]:
    """The toy game with its images spoiled: ``(catalog path, art folder)``.

    TOY-016 has the image of TOY-015, another card. TOY-P01 has the image of TOY-010, the card
    it is a printing of. TOY-002 is a vector file, TOY-004 a web page, TOY-006 empty, TOY-008
    not an image; TOY-009 is 8 pixels wide and TOY-011 one flat color; TOY-012 is not there.
    """
    art = folder / "art"
    shutil.copytree(toy_art, art)
    document = toy_game.catalog_document()
    by_id = {card["id"]: card for card in document["cards"]}
    by_id["TOY-016"]["art"] = "TOY-015.png"
    by_id["TOY-P01"]["art"] = "TOY-010.png"
    (art / "TOY-002.png").write_text(SVG, encoding="utf-8")
    (art / "TOY-004.png").write_text(PAGE, encoding="utf-8")
    (art / "TOY-006.png").write_bytes(b"")
    (art / "TOY-008.png").write_bytes(b"not an image at all")
    Image.new("RGB", (8, 8), (10, 20, 30)).save(art / "TOY-009.png")
    Image.new("RGB", (240, 336), (128, 128, 128)).save(art / "TOY-011.png")
    (art / "TOY-012.png").unlink()
    catalog = folder / "catalog.json"
    catalog.write_text(json.dumps(document, indent=1), encoding="utf-8")
    return catalog, art


SHARED_LINES = ["one image file under different cards (1 group(s)):", "  TOY-015, TOY-016",
                "  This usually means one card's image filed under another card's id, or one stand-in picture saved for "
                "cards that have none."]
NEAR_LINES = ["nearly the same bank row under different cards, cosine 0.99 or more (1 group(s)):", "  TOY-015, TOY-016",
              "  This usually means the same picture filed under two cards, or one card's image filed under another's id. "
              "The accept rule cannot tell such cards apart, and names neither."]


def test_the_images_of_a_catalog_are_looked_at_before_any_bank_exists(toy, tmp_path, capsys):
    catalog_path, art = staged(tmp_path, toy[1])
    report = A.check_art(parse_catalog(json.loads(catalog_path.read_text())), art)
    assert report.cards == 20 and report.missing == ["TOY-012"] and report.unreadable == []
    assert [card_id for card_id, _ in report.suspects.not_pictures] == ["TOY-002", "TOY-004", "TOY-006", "TOY-008"]
    assert [card_id for card_id, _ in report.suspects.placeholders] == ["TOY-009", "TOY-011"]
    assert report.suspects.shared == [["TOY-015", "TOY-016"]] and report.suspects.near_identical == []
    assert len(report.usable) == 20 - 1 - 4 - 2 and {"TOY-015", "TOY-016", "TOY-P01"} <= set(report.usable)

    # The command: check-bank-art with no bank. It lists, and exits with 0.
    assert cli.main(["check-bank-art", "--catalog", str(catalog_path), "--art", str(art)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"card images: {art} (20 catalog cards; 13 with a usable image)",
        "  1 with no image under --art",
        "  4 with a file that holds no picture",
        "  2 with a picture that looks like a placeholder",
        "no image under --art, for example: TOY-012",
        "image files that hold no picture (4):",
        "  TOY-002  a vector file (SVG), which has no pixels to compare",
        "  TOY-004  a web page saved under an image's name",
        "  TOY-006  an empty file",
        "  TOY-008  not an image that Pillow can read",
        "pictures that look like placeholders (2):",
        "  TOY-009  a picture of 8 x 8 pixels, too small to be a card: a placeholder",
        "  TOY-011  a picture of one flat color: a placeholder",
        *SHARED_LINES,
    ]
    # The toy game as it is drawn has nothing to list.
    assert cli.main(["check-bank-art", "--catalog", str(toy[0]), "--art", str(toy[1])]) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"card images: {toy[1]} (20 catalog cards; 20 with a usable image)", "  0 with no image under --art",
        "  0 with a file that holds no picture", "  0 with a picture that looks like a placeholder"]


def without(catalog_path: Path, *card_ids: str) -> Path:
    document = json.loads(catalog_path.read_text(encoding="utf-8"))
    document["cards"] = [card for card in document["cards"] if card["id"] not in card_ids]
    out = catalog_path.with_name("catalog-readable.json")
    out.write_text(json.dumps(document, indent=1), encoding="utf-8")
    return out


def test_build_bank_names_the_file_it_cannot_read_and_lists_what_to_look_at(toy, tmp_path, capsys):
    catalog_path, art = staged(tmp_path, toy[1])
    recognizer = build_recognizer(tmp_path / "recognizer.onnx")
    build = ["build-bank", "--art", str(art), "--recognizer", str(recognizer), "--skip-missing-art"]
    # A file that holds no picture stops the build, as before, and the message says what the file is.
    assert cli.main([*build, "--catalog", str(catalog_path), "--out", str(tmp_path / "never.json")]) == 1
    said = capsys.readouterr().err
    assert f"the image of card 'TOY-002' at {art / 'TOY-002.png'} cannot be read (a vector file (SVG), which has no pixels to compare)" in said
    assert "cardvision check-bank-art --catalog ... --art ..., with no bank, lists every such image at once" in said
    assert not (tmp_path / "never.json").exists()

    # Without the four entries that have no picture, the bank is built and the rest is listed.
    readable = without(catalog_path, "TOY-002", "TOY-004", "TOY-006", "TOY-008")
    bank = tmp_path / "bank.json"
    assert cli.main([*build, "--catalog", str(readable), "--out", str(bank)]) == 0
    lines = capsys.readouterr().out.splitlines()
    start = lines.index(f"identity groups: {tmp_path / 'bank.identity-groups.json'}")
    assert lines[start - 2:start] == [
        f"bank: {bank} (15 rows of 128 values; 15 embedded, 0 kept from the base bank)", "left out, no image found (1): TOY-012"]
    assert lines[start + 1:] == [
        "look at these before you rely on the bank:",
        "  pictures that look like placeholders (2):",
        "    TOY-009  a picture of 8 x 8 pixels, too small to be a card: a placeholder",
        "    TOY-011  a picture of one flat color: a placeholder",
        *(f"  {line}" for line in SHARED_LINES),
        *(f"  {line}" for line in NEAR_LINES),
    ]

    # check-bank-art on that bank lists the same, after its usual lines, and exits with 0.
    assert cli.main(["check-bank-art", str(bank), "--catalog", str(readable), "--art", str(art)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[:5] == [f"bank: {bank} (15 rows; 15 record their source image)", "  15 embedded from the image now under --art",
                         "  0 embedded from a different image", "  0 with no recorded image", "  0 with no image under --art"]
    assert lines[5:] == [
        "pictures that look like placeholders (2):",
        "  TOY-009  a picture of 8 x 8 pixels, too small to be a card: a placeholder",
        "  TOY-011  a picture of one flat color: a placeholder",
        *SHARED_LINES, *NEAR_LINES]


def test_a_clean_bank_has_nothing_to_look_at(toy, tmp_path, capsys):
    recognizer = build_recognizer(tmp_path / "recognizer.onnx")
    bank = tmp_path / "bank.json"
    assert cli.main(["build-bank", "--catalog", str(toy[0]), "--art", str(toy[1]), "--recognizer", str(recognizer),
                     "--out", str(bank)]) == 0
    assert capsys.readouterr().out.splitlines()[-2:] == [
        f"bank: {bank} (20 rows of 128 values; 20 embedded, 0 kept from the base bank)",
        f"identity groups: {tmp_path / 'bank.identity-groups.json'}"]
    assert cli.main(["check-bank-art", str(bank), "--catalog", str(toy[0]), "--art", str(toy[1])]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 5


def test_the_limit_holds_on_the_toy_game(toy, tmp_path):
    """What the module says of its limit: one image under two card ids gives rows with a cosine
    above 0.998, and rows of different toy cards stay below 0.97."""
    document = toy_game.catalog_document()
    next(card for card in document["cards"] if card["id"] == "TOY-016")["art"] = "TOY-015.png"
    catalog = parse_catalog(document)
    session = create_session(build_recognizer(tmp_path / "recognizer.onnx"))
    bank = BB.build_bank(catalog, toy[1], session, model="toy").bank
    ids, rows = decode_bank(bank)
    cosine = rows.astype(np.float64) @ rows.astype(np.float64).T
    twin = cosine[ids.index("TOY-015"), ids.index("TOY-016")]
    assert 0.998 < twin < 1.0
    different = [cosine[a, b] for a in range(len(ids)) for b in range(a + 1, len(ids))
                 if {ids[a], ids[b]} != {"TOY-015", "TOY-016"} and not catalog.same_identity(ids[a], ids[b])]
    assert max(different) < 0.97 < S.NEAR_IDENTICAL_COSINE < twin
    found = S.bank_suspects(bank, catalog)
    assert found.shared == found.near_identical == [["TOY-015", "TOY-016"]] and not found.not_pictures
    # The same bank read under the rule that makes the two one card has nothing to report.
    next(card for card in document["cards"] if card["id"] == "TOY-016")["name"] = "Plum Triangle"
    assert not S.bank_suspects(bank, parse_catalog(document))
    assert S.bank_suspects(bank, parse_catalog({**document, "identity": "printing"})).near_identical == [["TOY-015", "TOY-016"]]


def test_long_lists_are_cut_and_counted():
    found = S.Suspects(not_pictures=[(f"C-{i:03d}", "an empty file") for i in range(25)],
                       shared=[[f"C-{i:03d}" for i in range(12)]])
    lines = S.suspect_lines(found)
    assert lines[0] == "image files that hold no picture (25):" and lines[20] == "  C-019  an empty file"
    assert lines[21] == "  and 5 more"
    assert lines[23] == "  C-000, C-001, C-002, C-003, C-004, C-005, C-006, C-007, and 4 more"
    assert S.suspect_lines(S.Suspects()) == [] and not S.Suspects() and found


def test_a_base_row_that_cannot_be_compared_does_not_fail_the_build(toy, tmp_path, capsys):
    """The list is for a person and never a failure. A base bank may hold a row of zeros, which
    has no direction to compare: the bank is written all the same, one line says why the list
    was not made, and the command exits with 0."""
    from cardvision.core.bank import VECTOR_DTYPE, encode_vector

    recognizer = build_recognizer(tmp_path / "recognizer.onnx")
    build = ["build-bank", "--catalog", str(toy[0]), "--art", str(toy[1]), "--recognizer", str(recognizer)]
    base = tmp_path / "base.json"
    assert cli.main([*build, "--out", str(base)]) == 0
    document = json.loads(base.read_text(encoding="utf-8"))
    row = next(entry for entry in document["entries"] if entry["cardId"] == "TOY-004")
    row["vec"] = encode_vector(np.zeros(document["dim"], dtype=VECTOR_DTYPE))
    base.write_text(json.dumps(document), encoding="utf-8")
    capsys.readouterr()

    bank = tmp_path / "bank.json"
    assert cli.main([*build, "--base", str(base), "--out", str(bank)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.splitlines()[-3:] == [
        f"bank: {bank} (20 rows of 128 values; 0 embedded, 20 kept from the base bank)",
        f"identity groups: {tmp_path / 'bank.identity-groups.json'}",
        "the cards that share an image file or a bank row could not be listed: card 'TOY-004' has a stored vector "
        "that cannot be normalized"]
    # The row is kept as the base bank has it, and what the build itself saw of the images is still listed.
    assert json.loads(bank.read_text(encoding="utf-8"))["entries"] == document["entries"]
    Image.new("RGB", (240, 336), (128, 128, 128)).save(tmp_path / "TOY-P02.png")
    art = Path(shutil.copytree(toy[1], tmp_path / "art"))
    shutil.copy(tmp_path / "TOY-P02.png", art / "TOY-P02.png")
    more = json.loads(toy[0].read_text(encoding="utf-8"))
    more["cards"].append({"id": "TOY-P02", "name": "Gray Card", "kind": "unit", "printingOf": None, "art": "TOY-P02.png"})
    (tmp_path / "more.json").write_text(json.dumps(more), encoding="utf-8")
    assert cli.main(["build-bank", "--catalog", str(tmp_path / "more.json"), "--art", str(art), "--recognizer",
                     str(recognizer), "--base", str(base), "--out", str(tmp_path / "more-bank.json")]) == 0
    assert capsys.readouterr().out.splitlines()[-4:] == [
        "the cards that share an image file or a bank row could not be listed: card 'TOY-004' has a stored vector "
        "that cannot be normalized",
        "look at these before you rely on the bank:",
        "  pictures that look like placeholders (1):",
        "    TOY-P02  a picture of one flat color: a placeholder"]
