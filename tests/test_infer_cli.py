# SPDX-License-Identifier: Apache-2.0
"""``cardvision infer`` end to end on generated images, with toy models built at test time."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cardvision import cli
from cardvision.bank import build as bank_build
from cardvision.bank import release as release_module
from cardvision.core.catalog import load_catalog
from cardvision.core.images import load_rgb
from cardvision.core.onnx import create_session
from cardvision.detect import INPUT_HEIGHT, INPUT_WIDTH
from cardvision.infer.contact_sheet import sheet_size
from test_infer_toy_models import (
    EXPECTED_BOXES,
    EXPECTED_CLASSES,
    EXPECTED_NAMES,
    EXPECTED_SCORES,
    SCENE_BACK,
    SCENE_TINY,
    SCENE_WEAK,
    ToySetup,
    build_detector,
    detector_rows,
    save_image,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def setup(tmp_path_factory, toy) -> ToySetup:
    return ToySetup(tmp_path_factory.mktemp("cli"), *toy)


def run(capsys, *arguments: str) -> tuple[int, list[dict], str]:
    """Run the command line in this process: exit code, parsed output lines, standard error."""
    code = cli.main(["infer", *arguments])
    captured = capsys.readouterr()
    return code, [json.loads(line) for line in captured.out.splitlines()], captured.err


def test_the_subcommand_is_registered(capsys):
    assert "infer" in cli.SUBCOMMANDS
    assert cli.main([]) == 0
    assert "infer" in capsys.readouterr().out


def test_boxes_and_names_as_json_lines(setup, capsys):
    code, lines, err = run(capsys, str(setup.image), *setup.arguments())
    assert code == 0
    (frame,) = lines
    assert (frame["frame"], frame["source"], frame["frameW"], frame["frameH"]) == ("table.png", str(setup.image), 1280, 768)
    assert [box["box"] for box in frame["boxes"]] == EXPECTED_BOXES
    assert [box["class"] for box in frame["boxes"]] == EXPECTED_CLASSES
    assert [box["name"] for box in frame["boxes"] if box["class"] == "face"] == EXPECTED_NAMES
    assert "1 frame(s), 6 face(s), 1 back(s), 3 named" in err


def with_detector(arguments: list[str], detector: Path) -> list[str]:
    arguments = list(arguments)
    arguments[arguments.index("--detector") + 1] = str(detector)
    return arguments


def test_cards_found_add_no_diagnostic_to_the_summary(setup, capsys):
    code, _, err = run(capsys, str(setup.image), *setup.arguments())
    assert code == 0
    assert err == "cardvision infer: 1 frame(s), 6 face(s), 1 back(s), 3 named\n"


def test_a_frame_with_no_box_says_how_close_the_detector_came(setup, capsys, tmp_path):
    detector = build_detector(tmp_path / "near.onnx", detector_rows([(10, 10, 100, 100), (400, 10, 100, 100)], [0.21, 0.1], [0, 1]))
    code, lines, err = run(capsys, str(setup.image), *with_detector(setup.arguments(), detector))
    assert code == 0
    assert [line["boxes"] for line in lines] == [[]]
    assert err.splitlines() == [
        "cardvision infer: no box in 1 of 1 frame(s); the detector keeps a box at a score of 0.30 or more",
        "cardvision infer: table.png: highest score 0.21, threshold 0.30",
        "cardvision infer: 1 frame(s), 0 face(s), 0 back(s), 0 named",
    ]


def test_a_score_just_under_the_threshold_is_not_rounded_up_to_it(setup, capsys, tmp_path):
    detector = build_detector(tmp_path / "close.onnx", detector_rows([(10, 10, 100, 100)], [0.299], [0]))
    _, _, err = run(capsys, str(setup.image), *with_detector(setup.arguments(), detector))
    assert "highest score 0.29, threshold 0.30" in err


def test_no_row_at_all_reads_as_a_zero_score(setup, capsys, tmp_path):
    detector = build_detector(tmp_path / "none.onnx", detector_rows([(10, 10, 100, 100)], [0.0], [0]))
    _, _, err = run(capsys, str(setup.image), *with_detector(setup.arguments(), detector))
    assert "highest score 0.00, threshold 0.30" in err


def test_boxes_that_are_all_too_small_are_said_to_be_too_small(setup, capsys, tmp_path):
    detector = build_detector(tmp_path / "tiny.onnx", detector_rows([(10, 10, 20, 20)], [0.9], [0]))
    code, lines, err = run(capsys, str(setup.image), *with_detector(setup.arguments(), detector))
    assert code == 0 and lines[0]["boxes"] == []
    assert "table.png: 1 box(es) found, but each is under 24 pixels wide or high, too small to read" in err
    assert "highest score" not in err


def test_frames_with_no_box_are_listed_up_to_a_limit_and_the_rest_counted(setup, capsys, tmp_path):
    detector = build_detector(tmp_path / "near.onnx", detector_rows([(10, 10, 100, 100)], [0.25], [0]))
    folder = tmp_path / "frames"
    folder.mkdir()
    for number in range(8):
        save_image(folder / f"f{number}.png", setup.frame)
    code, lines, err = run(capsys, str(folder), *with_detector(setup.arguments(), detector))
    assert code == 0 and len(lines) == 8
    rows = err.splitlines()
    assert rows[0] == "cardvision infer: no box in 8 of 8 frame(s); the detector keeps a box at a score of 0.30 or more"
    assert [row for row in rows if "highest score 0.25, threshold 0.30" in row] == rows[1:6]
    assert rows[6] == "cardvision infer: and 3 more frame(s) with no box; the highest score in them is 0.25"
    assert rows[7] == "cardvision infer: 8 frame(s), 0 face(s), 0 back(s), 0 named"
    assert rows[8:] == ["cardvision infer: boxes per frame, fewest / median / most: faces 0 / 0 / 0, backs 0 / 0 / 0"]


def test_frames_left_uncounted_say_which_kind_they_are(capsys):
    """Past the frames listed one by one, a frame whose boxes were all too small is counted
    apart from a frame in which nothing reached the threshold. The highest score given is that
    of the second kind only: a box that was dropped for its size can have any score."""
    from cardvision.infer import cli as infer_cli
    from cardvision.infer.pipeline import FrameResult

    class Threshold:
        threshold = 0.3

    def no_box(name: str, best_score: float, too_small: int = 0) -> FrameResult:
        return FrameResult(name=name, source=name, t=0.0, frame_width=1280, frame_height=768, analysis_width=1280,
                           analysis_height=768, cards=(), ms=1.0, best_score=best_score, too_small=too_small)

    listed = [no_box(f"f{number}.png", 0.05) for number in range(infer_cli.EMPTY_FRAMES_LISTED)]
    mixed = [*listed, no_box("empty.png", 0.12), no_box("tiny.png", 0.9, too_small=2)]
    infer_cli._explain_empty_frames(mixed, 9, Threshold(), 24)
    lines = capsys.readouterr().err.splitlines()
    assert lines[0] == "cardvision infer: no box in 7 of 9 frame(s); the detector keeps a box at a score of 0.30 or more"
    assert lines[-1] == ("cardvision infer: and 2 more frame(s) with no box: 1 with only boxes too small to read, "
                         "and the highest score in the other 1 is 0.12")
    assert "0.9" not in lines[-1] and len(lines) == 7
    # Only frames with boxes that were too small: no score is given at all.
    infer_cli._explain_empty_frames([*listed, no_box("tiny.png", 0.9, too_small=1), no_box("tinier.png", 0.8, too_small=3)],
                                    7, Threshold(), 24)
    assert capsys.readouterr().err.splitlines()[-1] == (
        "cardvision infer: and 2 more frame(s) with no box: 2 with only boxes too small to read")
    # Only frames in which nothing reached the threshold: the line reads as it always did.
    infer_cli._explain_empty_frames([*listed, no_box("a.png", 0.12), no_box("b.png", 0.21)], 7, Threshold(), 24)
    assert capsys.readouterr().err.splitlines()[-1] == (
        "cardvision infer: and 2 more frame(s) with no box; the highest score in them is 0.21")


def test_several_frames_add_the_count_of_boxes_per_frame(setup, capsys, tmp_path):
    """A card that is missed in every frame does not show in the totals. With more than one
    frame, a last line gives the fewest, the median and the most boxes in a frame, faces and
    backs apart, to set against what is on the table."""
    from cardvision.infer import cli as infer_cli

    folder = tmp_path / "frames"
    folder.mkdir()
    for number in range(3):
        save_image(folder / f"f{number}.png", setup.frame)
    code, lines, err = run(capsys, str(folder), *setup.arguments())
    assert code == 0 and len(lines) == 3
    assert err.splitlines() == [
        "cardvision infer: 3 frame(s), 18 face(s), 3 back(s), 9 named",
        "cardvision infer: boxes per frame, fewest / median / most: faces 6 / 6 / 6, backs 1 / 1 / 1",
    ]
    assert infer_cli._spread([6, 5, 6]) == "5 / 6 / 6" and infer_cli._spread([0, 7]) == "0 / 3.5 / 7" and infer_cli._spread([4]) == "4 / 4 / 4"


def test_every_optional_output(setup, capsys, tmp_path):
    out = tmp_path / "out"
    code, lines, err = run(
        capsys, str(setup.image), *setup.arguments(),
        "--art", str(setup.art_dir),
        "--annotate", str(out / "annotated"),
        "--crops", str(out / "crops"),
        "--contact-sheet", str(out / "sheet.png"),
        "--prefill-out", str(out / "prefill.jsonl"),
        "--frames", str(out / "frames"),
    )
    assert code == 0 and len(lines) == 1

    with Image.open(out / "annotated" / "table.jpg") as image:
        assert image.size == (1280, 768)
    crops = sorted(path.name for path in (out / "crops").iterdir())
    assert crops == ["table-00.png", "table-01.png", "table-02.png", "table-03.png", "table-04.png", "table-06.png"]
    np.testing.assert_array_equal(load_rgb(out / "crops" / "table-00.png"), load_rgb(setup.art_dir / "TOY-001.png"))
    with Image.open(out / "sheet.png") as image:
        assert image.size == sheet_size(6)
    np.testing.assert_array_equal(load_rgb(out / "frames" / "table.png"), setup.frame)

    (record,) = [json.loads(line) for line in (out / "prefill.jsonl").read_text().splitlines()]
    assert record["name"] == "table.png" and (record["frameW"], record["frameH"]) == (1280, 768)
    assert [box["cardId"] for box in record["boxes"]] == ["TOY-001", "TOY-003", None, None, "TOY-017", None]
    assert [box["flag"] for box in record["boxes"]][:5] == [None, None, "below-tau", "low-margin", None]
    assert "no card image could be loaded" not in err


def test_the_contact_sheet_without_art_says_so_and_is_still_written(setup, capsys, tmp_path):
    sheet = tmp_path / "sheet.png"
    code, _, err = run(capsys, str(setup.image), *setup.arguments(), "--contact-sheet", str(sheet))
    assert code == 0
    with Image.open(sheet) as image:
        assert image.size == sheet_size(6)
    assert "the contact sheet shows their names and scores only" in err
    assert "pass --art" in err


def test_the_contact_sheet_says_what_to_install_when_the_art_server_certificate_cannot_be_verified(setup, capsys, tmp_path, monkeypatch):
    import ssl
    import urllib.error
    import urllib.request

    from cardvision.core.tls import CERTIFICATE_HELP

    def unverified(request, timeout=None, context=None):
        raise urllib.error.URLError(ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"))

    monkeypatch.setattr(urllib.request, "urlopen", unverified)
    sheet = tmp_path / "sheet.png"
    code, _, err = run(capsys, str(setup.image), *setup.arguments(), "--art", "https://cards.example/art/", "--contact-sheet", str(sheet))
    assert code == 0 and sheet.is_file()
    assert "no card image was downloaded: https://cards.example/art/" in err and CERTIFICATE_HELP in err


def test_the_contact_sheet_row_limit(setup, capsys, tmp_path):
    sheet = tmp_path / "sheet.png"
    code, _, err = run(capsys, str(setup.image), *setup.arguments(), "--contact-sheet", str(sheet), "--contact-sheet-rows", "2")
    assert code == 0
    with Image.open(sheet) as image:
        assert image.size == sheet_size(2)
    assert "shows the first 2 crops; 4 more were left out" in err
    assert f"cardvision infer: the contact sheet is {sheet}\n" in err


def test_a_long_contact_sheet_is_written_as_pages_and_the_files_are_named(setup, capsys, tmp_path):
    out = tmp_path / "out"
    sheet = out / "sheet.png"
    code, _, err = run(capsys, str(setup.image), *setup.arguments(), "--contact-sheet", str(sheet), "--contact-sheet-page-rows", "4")
    assert code == 0
    assert sorted(path.name for path in out.iterdir()) == ["sheet-1.png", "sheet-2.png"]
    with Image.open(out / "sheet-1.png") as first, Image.open(out / "sheet-2.png") as second:
        assert (first.size, second.size) == (sheet_size(4), sheet_size(2))
    assert f"cardvision infer: the contact sheet has 2 pages of at most 4 crops: {out / 'sheet-1.png'} to {out / 'sheet-2.png'}\n" in err
    # The row limit is over all pages: five rows, three to a page.
    code, _, err = run(capsys, str(setup.image), *setup.arguments(), "--contact-sheet", str(tmp_path / "five" / "s.png"),
                       "--contact-sheet-rows", "5", "--contact-sheet-page-rows", "3")
    assert code == 0 and sorted(path.name for path in (tmp_path / "five").iterdir()) == ["s-1.png", "s-2.png"]
    assert "shows the first 5 crops; 1 more were left out" in err
    with Image.open(tmp_path / "five" / "s-2.png") as second:
        assert second.size == sheet_size(2)
    for flag in ("--contact-sheet-rows", "--contact-sheet-page-rows"):
        code, lines, err = run(capsys, str(setup.image), *setup.arguments(), "--contact-sheet", str(sheet), flag, "0")
        assert code == 2 and lines == [] and f"{flag} must be at least 1" in err


def test_the_crops_of_a_video_go_on_pages_of_twenty(setup, capsys, tmp_path):
    """Four frames of six faces are twenty-four crops: a page of twenty and a page of four."""
    folder = tmp_path / "frames"
    folder.mkdir()
    for number in range(4):
        save_image(folder / f"f{number}.png", setup.frame)
    out = tmp_path / "out"
    code, _, err = run(capsys, str(folder), *setup.arguments(), "--contact-sheet", str(out / "sheet.png"))
    assert code == 0 and sorted(path.name for path in out.iterdir()) == ["sheet-1.png", "sheet-2.png"]
    with Image.open(out / "sheet-1.png") as first, Image.open(out / "sheet-2.png") as second:
        assert (first.size, second.size) == (sheet_size(20), sheet_size(4))
    assert "the contact sheet has 2 pages of at most 20 crops" in err


def test_the_accept_rule_defaults_to_the_published_values(setup, capsys):
    arguments = setup.arguments()
    without_accept = arguments[:arguments.index("--accept")]
    code, lines, _ = run(capsys, str(setup.image), *without_accept)
    assert code == 0
    faces = [box for box in lines[0]["boxes"] if box["class"] == "face"]
    # Under the looser published thresholds the toy recognizer names the card it merely resembles.
    assert faces[2]["accepted"] is True and faces[2]["similarity"] < 0.99
    assert faces[2]["similarity"] >= 0.62 and faces[2]["margin"] >= 0.05


def test_several_images_and_a_folder(setup, capsys, tmp_path):
    folder = tmp_path / "frames"
    folder.mkdir()
    save_image(folder / "b.png", setup.frame)
    save_image(folder / "a.png", setup.frame)
    code, lines, err = run(capsys, str(setup.image), str(folder), *setup.arguments())
    assert code == 0
    assert [line["frame"] for line in lines] == ["table.png", "a.png", "b.png"]
    assert "3 frame(s), 18 face(s), 3 back(s), 9 named" in err


def test_video_input(setup, capsys, tmp_path):
    cv2 = pytest.importorskip("cv2")
    clip = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 4.0, (1280, 768))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write the test video")
    for _ in range(8):
        writer.write(np.ascontiguousarray(setup.frame[:, :, ::-1]))
    writer.release()

    out = tmp_path / "out"
    code, lines, _ = run(capsys, str(clip), *setup.arguments(), "--every", "1",
                         "--prefill-out", str(out / "prefill.jsonl"), "--frames", str(out / "frames"))
    assert code == 0
    # Four frames a second: the frames taken are numbers 0 and 4, and t is the middle of each one's time on screen.
    assert [(line["frame"], line["t"]) for line in lines] == [("clip-f00001.jpg", 0.125), ("clip-f00002.jpg", 1.125)]
    assert all([box["box"] for box in line["boxes"]] == EXPECTED_BOXES for line in lines)
    # The prefill file names the frames that --frames wrote.
    records = [json.loads(line) for line in (out / "prefill.jsonl").read_text().splitlines()]
    assert [record["name"] for record in records] == ["clip-f00001.jpg", "clip-f00002.jpg"]
    assert [record["t"] for record in records] == [0.125, 1.125]
    assert sorted(path.name for path in (out / "frames").iterdir()) == ["clip-f00001.jpg", "clip-f00002.jpg"]

    # Asked for PNG, the frames are named and stored as PNG, without loss, and the prefill file follows.
    png = tmp_path / "png"
    code, lines, _ = run(capsys, str(clip), *setup.arguments(), "--every", "1", "--frame-format", "png",
                         "--prefill-out", str(png / "prefill.jsonl"), "--frames", str(png / "frames"))
    assert code == 0 and [(line["frame"], line["t"]) for line in lines] == [("clip-f00001.png", 0.125), ("clip-f00002.png", 1.125)]
    assert [json.loads(line)["name"] for line in (png / "prefill.jsonl").read_text().splitlines()] == [
        "clip-f00001.png", "clip-f00002.png"]
    from cardvision.core.images import load_rgb
    from cardvision.infer.frames import read_video

    stored = load_rgb(png / "frames" / "clip-f00002.png")
    assert np.array_equal(stored, list(read_video(clip, 1.0, "png"))[1].rgb)
    with pytest.raises(ValueError, match="unknown image format"):
        next(read_video(clip, 1.0, "bmp"))


def test_video_without_opencv_is_a_clear_error(setup, capsys, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"")
    code, lines, err = run(capsys, str(clip), *setup.arguments())
    assert code == 2 and lines == []
    assert 'python -m pip install -e ".[video]"' in err


@pytest.mark.parametrize(
    ("flag", "message"),
    [
        ("--detector", "no such detector file"),
        ("--recognizer", "no such recognizer file"),
        ("--bank", "no such bank file"),
        ("--catalog", "no such catalog file"),
        ("--accept", "no such accept rule file"),
    ],
)
def test_a_missing_file_is_reported_by_role(setup, capsys, flag, message):
    arguments = setup.arguments()
    arguments[arguments.index(flag) + 1] = str(setup.folder / "missing.file")
    code, lines, err = run(capsys, str(setup.image), *arguments)
    assert code == 2 and lines == []
    assert message in err and err.startswith("cardvision infer: error:")


def test_a_missing_input_is_reported_before_anything_runs(setup, capsys):
    code, lines, err = run(capsys, str(setup.folder / "nowhere.png"), *setup.arguments())
    assert code == 2 and lines == [] and "no such file or folder" in err


def test_a_bank_card_missing_from_the_catalog_is_reported(setup, capsys, tmp_path):
    doc = json.loads(setup.catalog.read_text())
    doc["cards"] = [card for card in doc["cards"] if card["id"] not in ("TOY-004", "TOY-003A", "TOY-P01")]
    smaller = tmp_path / "catalog.json"
    smaller.write_text(json.dumps(doc))
    arguments = setup.arguments()
    arguments[arguments.index("--catalog") + 1] = str(smaller)
    code, _, err = run(capsys, str(setup.image), *arguments)
    assert code == 2 and "bank card 'TOY-004' is not in the catalog" in err


def test_an_invalid_accept_file_is_reported(setup, capsys, tmp_path):
    bad = tmp_path / "accept.json"
    bad.write_text(json.dumps({"tau": 0.6}))
    arguments = setup.arguments()
    arguments[arguments.index("--accept") + 1] = str(bad)
    code, _, err = run(capsys, str(setup.image), *arguments)
    assert code == 2 and "accept.json.delta must be a number" in err
    assert "accept.json.rotationRetry must be true or false" in err    # every problem, not just the first


def test_the_rotation_retry_can_be_turned_off_in_the_accept_file(setup, capsys, tmp_path):
    """With the retry on, the card lying on its side is named at a turned look; with it off, no
    turned look is tried, so that card is not named and nothing else changes."""
    rule = json.loads(setup.accept.read_text(encoding="utf-8"))
    assert rule["rotationRetry"] is True
    code, (with_retry,), _ = run(capsys, str(setup.image), *setup.arguments())
    assert code == 0
    off = tmp_path / "accept.json"
    off.write_text(json.dumps({**rule, "rotationRetry": False}), encoding="utf-8")
    arguments = setup.arguments()
    arguments[arguments.index("--accept") + 1] = str(off)
    code, (without,), _ = run(capsys, str(setup.image), *arguments)
    assert code == 0
    faces = [(a, b) for a, b in zip(with_retry["boxes"], without["boxes"]) if a["class"] == "face"]
    assert len(faces) == 6 and all(a["box"] == b["box"] for a, b in faces)
    turned = [(a, b) for a, b in faces if a["look"] != 0 and a["accepted"]]
    assert [(a["cardId"], a["retried"]) for a, _ in turned] == [("TOY-003", True)]
    assert all(b["look"] == 0 and b["retried"] is False for _, b in faces)
    assert turned[0][1]["cardId"] is None and turned[0][1]["outcome"] == "nothing"
    # A card named upright is decided exactly as before.
    assert all(a == b for a, b in faces if a["retried"] is False)


def test_the_models_are_required(setup, capsys):
    code, lines, err = run(capsys, str(setup.image))
    assert code == 2 and lines == []
    assert err.startswith("cardvision infer: error:") and "give --release, or all of" in err
    assert "missing --detector, --recognizer, --bank, --catalog" in err


def test_some_but_not_all_of_the_model_files_is_an_error(setup, capsys):
    code, lines, err = run(capsys, str(setup.image), "--detector", str(setup.detector), "--bank", str(setup.bank))
    assert code == 2 and lines == []
    assert "missing --recognizer, --catalog" in err


# ------------------------------------------------------------------------------------ release
@pytest.fixture(scope="module")
def release(setup, tmp_path_factory) -> Path:
    """A release folder over the toy parts: the toy scene's detector rows padded to a full set of
    anchors, a bank built by the bank tool, and the toy accept rule."""
    root = tmp_path_factory.mktemp("release")
    shutil.copy(setup.catalog, root / "catalog.json")
    shutil.copy(setup.recognizer, root / "recognizer.onnx")
    boxes = [*EXPECTED_BOXES[:5], SCENE_BACK, EXPECTED_BOXES[6], SCENE_TINY, SCENE_WEAK]
    rows = detector_rows([tuple(box) for box in boxes], [*EXPECTED_SCORES, 0.6, 0.2],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0])
    anchors = release_module.yolox_anchor_count(INPUT_WIDTH, INPUT_HEIGHT)
    full = np.zeros((anchors, 7), dtype=np.float32)
    full[:len(rows)] = rows
    build_detector(root / "detector.onnx", full)
    catalog = load_catalog(root / "catalog.json")
    bank = bank_build.build_bank(catalog, setup.art_dir, create_session(root / "recognizer.onnx"), model="toy").bank
    bank_build.write_bank(bank, catalog, root / "bank.json")
    shutil.copy(setup.accept, root / "accept.json")
    release_module.build_release(
        root / "release.json", detector=root / "detector.onnx", recognizer=root / "recognizer.onnx",
        bank=root / "bank.json", catalog=root / "catalog.json", accept=json.loads(setup.accept.read_text()),
    )
    return root


def release_files(root: Path, *, accept: bool = True) -> list[str]:
    """The same release as separate files, which is how it was built."""
    arguments = ["--detector", str(root / "detector.onnx"), "--recognizer", str(root / "recognizer.onnx"),
                 "--bank", str(root / "bank.json"), "--catalog", str(root / "catalog.json")]
    return [*arguments, "--accept", str(root / "accept.json")] if accept else arguments


def test_a_release_gives_the_same_output_as_the_separate_files(setup, release, capsys):
    code, from_release, err = run(capsys, str(setup.image), "--release", str(release / "release.json"))
    assert code == 0, err
    code, from_files, _ = run(capsys, str(setup.image), *release_files(release))
    assert code == 0
    assert from_release == from_files
    (frame,) = from_release
    assert [box["box"] for box in frame["boxes"]] == EXPECTED_BOXES
    assert "1 frame(s), 6 face(s)" in err


def test_a_release_brings_its_accept_rule(setup, release, capsys):
    """Without an accept file the published values apply and decide some cards otherwise, so a
    run that matches the one with the toy rule has read the release's rule."""
    _, (from_release,), _ = run(capsys, str(setup.image), "--release", str(release / "release.json"))
    _, (published,), _ = run(capsys, str(setup.image), *release_files(release, accept=False))
    _, (strict,), _ = run(capsys, str(setup.image), *release_files(release))
    assert from_release == strict and from_release != published


def test_a_release_serves_the_optional_outputs_too(setup, release, capsys, tmp_path):
    out = tmp_path / "out"
    code, lines, err = run(capsys, str(setup.image), "--release", str(release / "release.json"),
                           "--prefill-out", str(out / "prefill.jsonl"), "--crops", str(out / "crops"))
    assert code == 0 and len(lines) == 1, err
    (record,) = [json.loads(line) for line in (out / "prefill.jsonl").read_text().splitlines()]
    assert record["name"] == "table.png" and len(record["boxes"]) == 6
    assert len(list((out / "crops").iterdir())) == 6


def test_a_changed_file_makes_the_release_unusable(setup, release, capsys, tmp_path):
    broken = Path(shutil.copytree(release, tmp_path / "broken"))
    (broken / "bank.json").write_bytes((broken / "bank.json").read_bytes() + b"\n")
    code, lines, err = run(capsys, str(setup.image), "--release", str(broken / "release.json"))
    assert code == 2 and lines == []
    assert err.startswith("cardvision infer: error: the release cannot be used:") and "bank" in err


def test_a_missing_release_or_a_missing_file_in_it_is_reported(setup, release, capsys, tmp_path):
    code, lines, err = run(capsys, str(setup.image), "--release", str(tmp_path / "nowhere.json"))
    assert code == 2 and lines == [] and "the release cannot be used" in err and "no release document" in err
    gone = Path(shutil.copytree(release, tmp_path / "gone"))
    (gone / "detector.onnx").unlink()
    code, lines, err = run(capsys, str(setup.image), "--release", str(gone / "release.json"))
    assert code == 2 and lines == [] and "the release cannot be used" in err and "detector" in err


def test_a_release_the_reference_detector_cannot_follow_is_refused_in_words_that_fit_infer(setup, release, capsys, tmp_path):
    from cardvision.bank.documents import canonical_bytes, read_json

    unsupported = Path(shutil.copytree(release, tmp_path / "unsupported"))
    doc = read_json(unsupported / "release.json")
    doc["detector"]["decode"]["nms"]["classAgnostic"] = False
    (unsupported / "release.json").write_bytes(canonical_bytes(release_module.with_id(doc)))
    code, lines, err = run(capsys, str(setup.image), "--release", str(unsupported / "release.json"))
    assert code == 2 and lines == []
    assert "this release cannot be used here" in err and "classAgnostic is false" in err and "evaluated" not in err


@pytest.mark.parametrize("flag", ["--detector", "--recognizer", "--bank", "--catalog", "--accept"])
def test_a_release_and_a_component_file_do_not_mix(setup, release, capsys, flag):
    value = release_files(release)
    value = value[value.index(flag) + 1]
    code, lines, err = run(capsys, str(setup.image), "--release", str(release / "release.json"), flag, value)
    assert code == 2 and lines == []
    assert f"--release pins every file and setting; do not also give {flag}" in err


def test_the_release_option_is_in_the_help(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["infer", "--help"])
    assert exit_info.value.code == 0
    assert "--release R.json" in capsys.readouterr().out


def test_as_a_separate_process(setup, tmp_path):
    """The installed entry point: only JSON on standard output, files where asked."""
    sheet = tmp_path / "sheet.png"
    done = subprocess.run(
        [sys.executable, "-m", "cardvision", "infer", str(setup.image), *setup.arguments(),
         "--art", str(setup.art_dir), "--contact-sheet", str(sheet)],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert done.returncode == 0, done.stderr
    (frame,) = [json.loads(line) for line in done.stdout.splitlines()]
    assert [box["name"] for box in frame["boxes"] if box["class"] == "face"] == EXPECTED_NAMES
    assert sheet.is_file()
