# SPDX-License-Identifier: Apache-2.0
"""What a run can write: annotated frames, crops, the contact sheet and the prefill file."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cardvision.core.catalog import load_catalog, parse_catalog
from cardvision.core.images import load_rgb
from cardvision.infer import (
    ArtLibrary,
    Pipeline,
    annotate,
    contact_sheet,
    prefill_box,
    prefill_flag,
    prefill_record,
    sheet_paths,
    sheet_rows,
    write_annotated,
    write_contact_sheet,
    write_crops,
    write_frame,
    write_prefill,
)
from cardvision.infer.contact_sheet import CANDIDATES_X, HEADER_HEIGHT, ROW_HEIGHT, sheet_size
from cardvision.infer.render import BACK_COLOR, NAMED_COLOR, NOTHING_COLOR, card_color, card_label
from cardvision.recognize import load_accept
from test_infer_toy_models import EXPECTED_BOXES, SCENE_CARDS, ToySetup
import toy_game


@pytest.fixture(scope="module")
def setup(tmp_path_factory, toy) -> ToySetup:
    return ToySetup(tmp_path_factory.mktemp("outputs"), *toy)


@pytest.fixture(scope="module")
def result(setup):
    pipeline = Pipeline.open(setup.detector, setup.recognizer, setup.bank, load_catalog(setup.catalog), load_accept(setup.accept))
    return pipeline.analyze(setup.frame, name="table.png", source=str(setup.image), t=44.0)


def colorful(image: Image.Image) -> np.ndarray:
    """Mask of the pixels that are clearly not gray. Text and background on the sheet are gray."""
    pixels = np.asarray(image.convert("RGB")).astype(np.int16)
    return (pixels.max(axis=2) - pixels.min(axis=2)) > 40


# ----------------------------------------------------------------------------- annotated frames


def test_labels_and_colors(result):
    named, turned, unknown, doubtful, landscape, back, clipped = result.cards
    assert card_label(named).startswith("Amber Circle 1.00") and card_color(named) == NAMED_COLOR
    assert card_label(unknown).startswith("? (") and card_color(unknown) == NOTHING_COLOR
    assert card_label(back) == "back 0.70" and card_color(back) == BACK_COLOR
    assert card_color(turned) == NAMED_COLOR and card_color(doubtful) == NOTHING_COLOR


def test_annotate_draws_on_a_copy(setup, result):
    before = setup.frame.copy()
    image = annotate(setup.frame, result)
    np.testing.assert_array_equal(setup.frame, before)
    assert image.size == (1280, 768)
    drawn = np.asarray(image)
    assert (drawn != setup.frame).any()
    # The outline of the first box, on its top edge, is the "named" color; the back's is the back color.
    assert tuple(drawn[40, 160]) == NAMED_COLOR
    assert tuple(drawn[756, 540]) == BACK_COLOR
    # The middle of a card is untouched.
    np.testing.assert_array_equal(drawn[150:250, 100:200], setup.frame[150:250, 100:200])


def test_write_annotated(tmp_path, setup, result):
    path = write_annotated(tmp_path / "new" / "folder", setup.frame, result)
    assert path.name == "table.jpg"
    with Image.open(path) as image:
        assert image.size == (1280, 768) and image.format == "JPEG"


def test_write_crops(tmp_path, setup, result):
    paths = write_crops(tmp_path / "crops", result)
    assert [path.name for path in paths] == ["table-00.png", "table-01.png", "table-02.png", "table-03.png", "table-04.png", "table-06.png"]
    np.testing.assert_array_equal(load_rgb(paths[0]), load_rgb(setup.art_dir / "TOY-001.png"))
    np.testing.assert_array_equal(load_rgb(paths[1]), np.rot90(load_rgb(setup.art_dir / "TOY-003.png")))
    assert load_rgb(paths[5]).shape == (336, 120, 3)


def test_write_frame_keeps_the_pixels_when_the_format_is_lossless(tmp_path, setup):
    path = write_frame(tmp_path / "frames", setup.frame, "table.png")
    np.testing.assert_array_equal(load_rgb(path), setup.frame)
    jpeg = write_frame(tmp_path / "frames", setup.frame, "clip-f00001.jpg")
    assert load_rgb(jpeg).shape == setup.frame.shape


# ----------------------------------------------------------------------------- the contact sheet


def test_one_row_per_face(result):
    rows = sheet_rows(result)
    assert [row.title for row in rows] == ["table.png #00", "table.png #01", "table.png #02", "table.png #03", "table.png #04", "table.png #06"]
    assert rows[0].recognition.name == "Amber Circle"
    np.testing.assert_array_equal(rows[1].crop, result.cards[1].crop)


def test_sheet_with_card_images(setup, result):
    rows = sheet_rows(result)
    art = ArtLibrary(load_catalog(setup.catalog), setup.art_dir)
    sheet = contact_sheet(rows, art)
    assert sheet.size == sheet_size(6) == (CANDIDATES_X + 3 * 230 + 8, HEADER_HEIGHT + 6 * ROW_HEIGHT)
    assert art.missing == set()
    # Against the same sheet drawn without images: every candidate cell gained a picture, and
    # nothing left of the candidates changed.
    changed = (np.asarray(sheet) != np.asarray(contact_sheet(rows, None))).any(axis=2)
    assert not changed[:, :CANDIDATES_X - 4].any()
    for position in range(6):
        top = HEADER_HEIGHT + position * ROW_HEIGHT
        for slot in range(3):
            left = CANDIDATES_X + slot * 230
            assert changed[top:top + ROW_HEIGHT, left:left + 226].sum() > 5000
    assert colorful(sheet)[:, CANDIDATES_X:].sum() > 20000


def test_the_first_candidate_of_a_named_crop_is_its_own_card(setup, result):
    from PIL import ImageOps

    from cardvision.infer.contact_sheet import MARGIN, THUMBNAIL_BOX

    art = ArtLibrary(load_catalog(setup.catalog), setup.art_dir)
    sheet = np.asarray(contact_sheet(sheet_rows(result)[:1], art))
    with Image.open(setup.art_dir / "TOY-001.png") as image:
        thumbnail = np.asarray(ImageOps.contain(image.convert("RGB"), THUMBNAIL_BOX))
    top, left = HEADER_HEIGHT + MARGIN, CANDIDATES_X
    inner = sheet[top + 2:top + thumbnail.shape[0] - 2, left + 2:left + thumbnail.shape[1] - 2]
    np.testing.assert_array_equal(inner, thumbnail[2:-2, 2:-2])


def test_sheet_without_card_images_shows_names_and_scores_only(setup, result):
    rows = sheet_rows(result)
    for art in (None, ArtLibrary(load_catalog(setup.catalog), setup.folder / "no-such-folder"),
                ArtLibrary(load_catalog(setup.catalog))):
        sheet = contact_sheet(rows, art)
        assert sheet.size == sheet_size(6)
        mask = colorful(sheet)
        assert mask[:, CANDIDATES_X:].sum() == 0                 # no pictures among the candidates
        assert mask[:, :CANDIDATES_X].sum() > 6 * 2000           # the crops are still there
        # ... but the candidate columns are not empty: the names and scores are written there.
        candidates = np.asarray(sheet)[HEADER_HEIGHT:, CANDIDATES_X:]
        assert (candidates.max(axis=2) > 120).sum() > 500
        if art is not None:
            assert "TOY-001" in art.missing


def test_one_missing_image_does_not_spoil_the_sheet(tmp_path, setup, result):
    partial = tmp_path / "art"
    partial.mkdir()
    for card_id, *_ in toy_game.CARDS:
        if card_id != "TOY-001":
            (partial / f"{card_id}.png").write_bytes((setup.art_dir / f"{card_id}.png").read_bytes())
    (partial / "TOY-002.png").write_bytes(b"not an image")
    art = ArtLibrary(load_catalog(setup.catalog), partial)
    assert art.image("TOY-001") is None and art.image("TOY-002") is None
    assert art.image("TOY-003").size == (240, 336)
    assert art.image("NOT-A-CARD") is None
    contact_sheet(sheet_rows(result), art)
    assert {"TOY-001", "TOY-002", "NOT-A-CARD"} <= art.missing


def png_bytes(color) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (24, 36), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_images_can_come_from_a_url_base_without_touching_the_network():
    catalog = parse_catalog({"game": "toy", "locale": "en", "cards": [
        {"id": "A-1", "name": "Alpha", "art": "a 1.png"},
        {"id": "B-1", "name": "Beta", "art": "https://elsewhere.example/b.png"},
    ]})
    asked = []

    def fetch(url, timeout):
        asked.append((url, timeout))
        return png_bytes((200, 30, 30))

    art = ArtLibrary(catalog, "https://cards.example/art/", timeout=4.0, fetch=fetch)
    assert art.image("A-1").size == (24, 36) and art.image("B-1").mode == "RGB"
    assert art.image("A-1") is art.image("A-1")                           # fetched once, then remembered
    assert asked == [("https://cards.example/art/a%201.png", 4.0), ("https://elsewhere.example/b.png", 4.0)]


def test_an_unreachable_server_is_given_up_on():
    cards = [{"id": f"A-{i}", "name": f"Card {i}", "art": f"{i}.png"} for i in range(8)]
    catalog = parse_catalog({"game": "toy", "locale": "en", "cards": cards})
    asked = []

    def fetch(url, timeout):
        asked.append(url)
        raise OSError("connection refused")

    art = ArtLibrary(catalog, "https://cards.example/", give_up_after=3, fetch=fetch)
    assert [art.image(card["id"]) for card in cards] == [None] * 8
    assert len(asked) == 3 and len(art.missing) == 8


def test_write_contact_sheet_limits_the_rows(tmp_path, setup, result):
    rows = sheet_rows(result)
    path = tmp_path / "sheets" / "sheet.png"
    assert write_contact_sheet(path, rows, None, max_rows=4) == 4
    with Image.open(path) as image:
        assert image.size == sheet_size(4)
    assert write_contact_sheet(path, [], None) == 0
    with Image.open(path) as image:
        assert image.size == sheet_size(0)


def test_a_long_sheet_is_written_as_pages(tmp_path, result):
    """Every crop of a video on one image is too tall to look at. A sheet of more rows than a
    page holds is written as several images, numbered from 1 in the name."""
    from cardvision.infer.contact_sheet import DEFAULT_MAX_ROWS, DEFAULT_PAGE_ROWS

    rows = sheet_rows(result) * 4                                       # 24 rows
    folder = tmp_path / "pages"
    assert (DEFAULT_MAX_ROWS, DEFAULT_PAGE_ROWS) == (60, 20)
    assert write_contact_sheet(folder / "sheet.png", rows, None) == 24
    assert sorted(path.name for path in folder.iterdir()) == ["sheet-1.png", "sheet-2.png"]      # no sheet.png beside them
    sizes = []
    for path in sheet_paths(folder / "sheet.png", 24):
        with Image.open(path) as image:
            sizes.append(image.size)
    assert sizes == [sheet_size(20), sheet_size(4)] and sheet_size(20)[1] < 4000
    # The pages are the one long sheet cut up: the second page starts with the twenty-first row.
    second = np.asarray(contact_sheet(rows[20:], None))
    with Image.open(folder / "sheet-2.png") as image:
        np.testing.assert_array_equal(np.asarray(image.convert("RGB")), second)

    # The row limit counts over all pages; the page size is the caller's to set.
    small = tmp_path / "small"
    assert write_contact_sheet(small / "matches.jpg", rows, None, max_rows=15, page_rows=10) == 15
    assert sorted(path.name for path in small.iterdir()) == ["matches-1.jpg", "matches-2.jpg"]
    with Image.open(small / "matches-2.jpg") as image:
        assert image.size == sheet_size(5)
    # Rows that fit on one page are written under the name given, as before.
    one = tmp_path / "one"
    assert write_contact_sheet(one / "sheet.png", rows[:20], None) == 20
    assert [path.name for path in one.iterdir()] == ["sheet.png"]


def test_the_files_of_a_sheet():
    assert sheet_paths("out/sheet.png", 0) == sheet_paths("out/sheet.png", 1) == sheet_paths("out/sheet.png", 20) == [Path("out/sheet.png")]
    assert sheet_paths("out/sheet.png", 21) == [Path("out/sheet-1.png"), Path("out/sheet-2.png")]
    assert [path.name for path in sheet_paths("my.sheet.png", 57)] == ["my.sheet-1.png", "my.sheet-2.png", "my.sheet-3.png"]
    assert [path.name for path in sheet_paths("sheet", 5, page_rows=2)] == ["sheet-1", "sheet-2", "sheet-3"]
    with pytest.raises(ValueError, match="at least 1 row"):
        sheet_paths("sheet.png", 5, page_rows=0)


# ----------------------------------------------------------------------------- the prefill file


def test_prefill_record_shape(result):
    record = json.loads(json.dumps(prefill_record(result)))
    assert set(record) == {"t", "name", "frameW", "frameH", "ms", "boxes"}
    assert (record["t"], record["name"], record["frameW"], record["frameH"]) == (44.0, "table.png", 1280, 768)
    assert isinstance(record["ms"], float)
    assert len(record["boxes"]) == 6                                  # the faces; the back is not written
    assert [box["box"] for box in record["boxes"]] == [b for i, b in enumerate(EXPECTED_BOXES) if i != 5]
    for box in record["boxes"]:
        assert set(box) == {"box", "cardId", "sim", "margin", "top5", "rot", "flag"}
        assert len(box["top5"]) == 5 and all(set(entry) == {"id", "sim"} for entry in box["top5"])
        assert [entry["sim"] for entry in box["top5"]] == sorted((entry["sim"] for entry in box["top5"]), reverse=True)
        assert box["sim"] == box["top5"][0]["sim"]


def test_prefill_ids_flags_and_turns(result):
    boxes = prefill_record(result)["boxes"]
    assert [box["cardId"] for box in boxes] == ["TOY-001", "TOY-003", None, None, "TOY-017", None]
    assert [box["flag"] for box in boxes[:5]] == [None, None, "below-tau", "low-margin", None]
    assert boxes[5]["flag"] in ("below-tau", "low-margin")
    assert [box["rot"] for box in boxes[:2]] == [0, 3]
    assert boxes[0]["top5"][0] == {"id": "TOY-001", "sim": 1.0}
    assert boxes[3]["margin"] == 0.0 and boxes[3]["sim"] == 1.0


def test_prefill_flag_and_box_per_card(result):
    named, _, unknown, doubtful, _, back, _ = result.cards
    assert prefill_flag(named) is None and prefill_flag(back) is None
    assert prefill_flag(unknown) == "below-tau" and prefill_flag(doubtful) == "low-margin"
    assert prefill_box(named)["cardId"] == "TOY-001"
    with pytest.raises(ValueError, match="only a recognized face"):
        prefill_box(back)


def test_write_prefill_is_json_lines(tmp_path, result):
    path = tmp_path / "out" / "prefill.jsonl"
    assert write_prefill(path, [result, result]) == 2
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == json.loads(json.dumps(prefill_record(result)))
    assert write_prefill(path, []) == 0 and path.read_text() == ""


def test_scene_cards_are_what_the_prefill_names(result):
    named = {box["cardId"] for box in prefill_record(result)["boxes"] if box["cardId"]}
    assert named <= {card_id for card_id, *_ in SCENE_CARDS}
