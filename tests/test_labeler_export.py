# SPDX-License-Identifier: Apache-2.0
"""The labeling tool's manifest export, end to end: toy frames are labeled through the tool's
own save path, exported, and checked with ``cardvision validate-manifest --contribution``.

The labeling tool is a separate program that imports nothing from this package, so its export
is a second writer of the manifest format. These tests hold it to the first one: what it writes
must pass the validator, read back as the labels that went in, and carry the fingerprint the
reference code computes.
"""

from __future__ import annotations

import ast
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cardvision import cli
from cardvision.core.catalog import load_catalog
from cardvision.core.images import load_rgb
from cardvision.data import detector_targets, read_manifest
from cardvision.data.dhash import dhash, is_dhash

ROOT = Path(__file__).resolve().parents[1]
LABELER = ROOT / "labeler"
CREDIT = "Toy Labeler"
VIDEOS = {"table-a": ("youtube", "EXAMPLE0001"), "table-b": ("twitch", "123456789")}


@pytest.fixture(scope="module")
def tool():
    """The labeling tool's modules, imported from its folder as the tool itself does."""
    sys.path.insert(0, str(LABELER))
    try:
        import export
        import make_toy_data
        import serve
        yield export, make_toy_data, serve
    finally:
        sys.path.remove(str(LABELER))


@pytest.fixture(scope="module")
def labeled(tool, tmp_path_factory) -> Path:
    """A folder of toy data with every frame of both toy videos labeled and confirmed."""
    _export, make_toy_data, serve = tool
    folder = tmp_path_factory.mktemp("labeling") / "toy"
    make_toy_data.build(folder)
    # A manifest row names a video: say where each toy "video" is hosted. One asset states it
    # once; the other on each of its frames.
    sample = json.loads((folder / "sample.json").read_text(encoding="utf-8"))
    first, second = sample["assets"]
    first.update(platform=VIDEOS["table-a"][0], videoId=VIDEOS["table-a"][1])
    for record in second["frames"]:
        record.update(platform=VIDEOS["table-b"][0], videoId=VIDEOS["table-b"][1])
    (folder / "sample.json").write_text(json.dumps(sample), encoding="utf-8")

    serve.configure(serve.build_parser().parse_args([
        "--mode", "box", "--role", "training", "--sample", str(folder / "sample.json"),
        "--catalog", str(folder / "catalog.json"), "--art", str(folder / "art")]))
    truth = json.loads((folder / "truth.json").read_text(encoding="utf-8"))["frames"]
    for asset in sample["assets"]:
        for position, record in enumerate(asset["frames"]):
            targets = []
            for number, card in enumerate(truth[f"{asset['assetId']}/{record['name']}"], start=1):
                # truth.json's right answer, as the tool stores it: an ignore box around what shows of a
                # card too hidden to learn from, carrying no name
                named = card["outcome"] == "card" and not card["ignore"]
                target = {"id": f"B{number:03d}", "corners": card["corners"], "estimated": [False] * 4, "cornerSource": "box",
                          "outcome": card["outcome"], "ignore": card["ignore"], "identity": card["cardId"] if named else None,
                          "identitySource": "current-frame" if named else None,
                          "eligible": "yes" if named else "no" if card["outcome"] == "card-back" else None,
                          "conditions": {}, "labelSource": "manual"}
                if position == 0 and number == 1:   # one box began as a model's proposal and was corrected
                    target.update(cornerSource="prefill-edited", labelSource="prefill-corrected",
                                  prefill={"i": 0, "corners": card["corners"], "outcome": "card"})
                targets.append(target)
            row = {"name": record["name"], "t": record["t"], "frameW": make_toy_data.FRAME_W, "frameH": make_toy_data.FRAME_H,
                   "done": True, "targets": targets}
            status, saved = serve.save_frame(asset["assetId"], row)
            assert status == 200, saved
        status, saved = serve.save_meta(asset["assetId"], {"recording": {"source": "physical", "physicalGroup": "webcam"}})
        assert status == 200, saved
    return folder


@pytest.fixture(scope="module")
def offered(tool, labeled, tmp_path_factory) -> Path:
    """The exported manifest, in a folder of its own (a manifest's folder holds text only)."""
    export = tool[0]
    out = tmp_path_factory.mktemp("offer") / "toy.manifest.jsonl"
    assert export.main(["manifest", "--labels-dir", str(labeled / "labels"), "--sample", str(labeled / "sample.json"),
                        "--credit", CREDIT, "--out", str(out), "--catalog", str(labeled / "catalog.json")]) == 0
    return out


def test_the_export_passes_the_contribution_check(offered, labeled, capsys):
    arguments = ["validate-manifest", str(offered), "--catalog", str(labeled / "catalog.json")]
    assert cli.main([*arguments, "--contribution"]) == 0, capsys.readouterr().err
    assert cli.main(arguments) == 0
    # The check is a live one: the same rows with a role chosen are refused as a contribution.
    chosen = offered.parent / "chosen" / "toy.manifest.jsonl"
    chosen.parent.mkdir()
    chosen.write_text(offered.read_text(encoding="utf-8").replace('"role":"extra"', '"role":"train"'), encoding="utf-8")
    assert cli.main(["validate-manifest", str(chosen)]) == 0
    capsys.readouterr()
    assert cli.main(["validate-manifest", str(chosen), "--contribution"]) != 0
    assert "contribution" in capsys.readouterr().err


def test_the_rows_are_written_as_an_offer(offered):
    rows = read_manifest(offered)
    assert len(rows) == 10
    assert {(row.platform, row.video_id) for row in rows} == set(VIDEOS.values())
    assert all((row.role, row.trained_on, row.reachable, row.reachable_checked) == ("extra", False, None, None) for row in rows)
    assert all((row.credit, row.completeness, row.source_type) == (CREDIT, "exhaustive-boxes", "physical/webcam") for row in rows)
    assert all((row.frame_width, row.frame_height) == (960, 540) for row in rows)
    assert sorted(row.t for row in rows if row.video_id == "EXAMPLE0001") == [0.0, 1.0, 2.0, 3.0, 4.0]
    # Each video is a group of its own until somebody says which videos belong together.
    groups = {row.video_id: {r.group for r in rows if r.video_id == row.video_id} for row in rows}
    assert all(len(found) == 1 for found in groups.values()) and len({next(iter(found)) for found in groups.values()}) == 2
    # Only the frames with a box that began as a proposal (one per video) say that pre-fill was offered.
    assert sorted((row.video_id, row.t) for row in rows if row.prefill_offered) == [("123456789", 0.0), ("EXAMPLE0001", 0.0)]


def test_the_boxes_read_back_as_the_labels_that_went_in(offered, labeled):
    truth = json.loads((labeled / "truth.json").read_text(encoding="utf-8"))["frames"]
    catalog = load_catalog(labeled / "catalog.json")
    asset_of = {video: asset for asset, (_platform, video) in VIDEOS.items()}
    faces = backs = ignored = turned = past_the_edge = 0
    for row in read_manifest(offered):
        cards = truth[f"{asset_of[row.video_id]}/f{int(row.t) + 1:05d}.png"]
        assert len(row.boxes) == len(cards) >= 5
        for box, card in zip(row.boxes, cards):
            # Fractions of the frame, to a hundredth of a pixel of the corners that were drawn.
            assert np.allclose(box.pixel_quad(row.frame_width, row.frame_height), card["corners"], atol=0.01)
            assert all(-1 <= value <= 2 for point in box.quad for value in point)
            past_the_edge += any(not 0 <= value <= 1 for point in box.quad for value in point)
            assert (box.shape, box.ignore) == ("rotated-box", card["ignore"])
            if card["ignore"]:
                assert (box.kind, box.oriented, box.card_id, box.identity) == ("face", False, None, "unlabeled")
                ignored += 1
            elif card["outcome"] == "card-back":
                assert (box.kind, box.oriented, box.card_id, box.identity) == ("back", False, None, None)
                backs += 1
            else:
                assert (box.kind, box.oriented, box.card_id, box.identity, box.readable) == ("face", True, card["cardId"], "named", "yes")
                assert card["cardId"] in catalog
                faces += 1
        assert [box.origin for box in row.boxes].count("prefill") == (1 if row.prefill_offered else 0)
        assert {box.origin for box in row.boxes} <= {"manual", "prefill"}
        # The rows are ready for the label conversions: every box is a target.
        targets = detector_targets(row)
        assert len(targets) == len(cards) and {target.kind for target in targets} <= {"face", "back", "ignore"}
        turned += sum(1 for box in row.boxes if box.oriented and abs(box.quad[0][1] - box.quad[1][1]) > 0.01)
    # the toy tables have turned and tilted cards, hidden ones and a card past the frame's edge
    assert (faces, backs, ignored) == (53, 10, 5) and turned >= 10 and past_the_edge >= 5


def test_each_row_carries_the_reference_fingerprint_of_its_frame(offered, labeled):
    asset_of = {video: asset for asset, (_platform, video) in VIDEOS.items()}
    rows = read_manifest(offered)
    for row in rows:
        frame = labeled / "frames" / asset_of[row.video_id] / f"f{int(row.t) + 1:05d}.png"
        assert set(row.attributes) == {"frame_dhash"} and is_dhash(row.attributes["frame_dhash"])
        assert row.attributes["frame_dhash"] == dhash(load_rgb(frame))
    assert len({row.attributes["frame_dhash"] for row in rows}) > 1


def test_without_fingerprints_and_with_stated_groups(tool, labeled, tmp_path, capsys):
    export = tool[0]
    out = tmp_path / "offer" / "plain.manifest.jsonl"
    assert export.main(["manifest", "--labels-dir", str(labeled / "labels"), "--sample", str(labeled / "sample.json"),
                        "--credit", CREDIT, "--out", str(out), "--no-fingerprint", "--group", "g-toy-tables",
                        "--source-type", "table-b=physical/phone"]) == 0
    capsys.readouterr()
    assert cli.main(["validate-manifest", str(out), "--contribution"]) == 0
    rows = read_manifest(out)
    assert all(not row.attributes for row in rows) and {row.group for row in rows} == {"g-toy-tables"}
    assert {row.video_id: row.source_type for row in rows} == {"EXAMPLE0001": "physical/webcam", "123456789": "physical/phone"}


# ------------------------------------------------------------------------------ the card id
def test_the_labeling_tool_states_the_same_card_id_rule_as_the_validator(tool):
    from cardvision.data import schema
    labelcore = sys.modules["labelcore"]
    assert labelcore.CARD_ID_PATTERN == schema.CARD_ID_PATTERN
    assert labelcore.CARD_ID_MAX == 64


@pytest.mark.parametrize("card_id", ["TOY 001", "TOY-001\n", "TOY\t001", "TOY\u00a0001", "TOY\u0085001", "TOY\ufeff001",
                                     "TOY\x00001", "TOY-" + "0" * 61, "", None, 7])
def test_a_card_id_the_format_refuses_is_refused_by_the_labeling_tool(tool, card_id):
    labelcore = sys.modules["labelcore"]
    assert "must be 1 to 64 characters" in labelcore.card_id_error(card_id)


@pytest.mark.parametrize("card_id", ["TOY-001", "TOY-T01", "a", "x" * 64, "SFD/R01", "Caf\u00e9-1", "\u5361-1"])
def test_a_card_id_the_format_accepts_is_accepted_by_the_labeling_tool(tool, card_id):
    from cardvision.data import schema
    assert sys.modules["labelcore"].card_id_error(card_id) is None
    assert schema.fits_pattern(schema.CARD_ID_PATTERN, card_id)


@pytest.mark.parametrize("field", ["identity", "printing"])
def test_an_export_with_a_bad_card_id_names_the_frame_and_the_box(tool, labeled, field):
    export = tool[0]
    sets, _notes = export.read_sets(labeled / "labels", None, None)
    sample = export.read_sample(labeled / "sample.json")
    set_id, _meta, frames = sets[0]
    target = next(t for t in frames[0]["targets"] if t.get("identity"))
    target[field] = "TOY 001"
    with pytest.raises(SystemExit) as stopped:
        export.manifest_rows(sets, sample, CREDIT)
    message = str(stopped.value)
    assert f"{set_id}/{frames[0]['name']}: box {target['id']}: card id 'TOY 001' must be 1 to 64 characters" in message
    assert "cannot write the manifest" in message


# ------------------------------------------------------------------------- the fingerprint
def _filtered(image: np.ndarray, method: int) -> bytes:
    """A PNG's pixel data with every row stored by one filter method."""
    height, width, channels = image.shape
    rows = image.reshape(height, width * channels).astype(np.int16)
    above = np.vstack([np.zeros((1, width * channels), np.int16), rows[:-1]])
    left = np.hstack([np.zeros((height, channels), np.int16), rows[:, :-channels]])
    corner = np.hstack([np.zeros((height, channels), np.int16), above[:, :-channels]])
    if method == 0:
        stored = rows
    elif method == 1:
        stored = rows - left
    elif method == 2:
        stored = rows - above
    elif method == 3:
        stored = rows - (left + above) // 2
    else:
        guess = left + above - corner
        da, db, dc = np.abs(guess - left), np.abs(guess - above), np.abs(guess - corner)
        stored = rows - np.where((da <= db) & (da <= dc), left, np.where(db <= dc, above, corner))
    body = (stored % 256).astype(np.uint8)
    return b"".join(bytes([method]) + body[y].tobytes() for y in range(height))


def _png(image: np.ndarray, color: int, method: int, palette: bytes = b"") -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    height, width = image.shape[:2]
    raw = zlib.compress(_filtered(image, method))
    # The pixel data is split over two chunks, as an encoder is free to do.
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color, 0, 0, 0))
            + (chunk(b"PLTE", palette) if palette else b"") + chunk(b"IDAT", raw[:len(raw) // 2])
            + chunk(b"IDAT", raw[len(raw) // 2:]) + chunk(b"IEND", b""))


@pytest.mark.parametrize("method", [0, 1, 2, 3, 4])
@pytest.mark.parametrize(("color", "channels"), [(0, 1), (2, 3), (3, 1), (4, 2), (6, 4)])
def test_the_tool_s_fingerprint_is_the_reference_one(tool, tmp_path, method, color, channels):
    """Every PNG color type and every row filter, at sizes that do not divide by the grid: the
    tool's standard-library fingerprint equals the reference code's on the decoded picture."""
    export = tool[0]
    rng = np.random.default_rng(10 * color + method)
    for width, height in ((97, 61), (40, 23), (9, 8), (4, 3)):
        # Smooth blobs with a little noise, so that neighboring cells differ by clear margins and by small ones.
        yy, xx = np.mgrid[0:height, 0:width]
        base = 128 + 90 * np.sin(xx / max(width, 1) * rng.uniform(3, 9))[..., None] * np.cos(yy / max(height, 1) * rng.uniform(3, 9))[..., None]
        image = np.clip(base + rng.normal(0, 12, size=(height, width, channels)), 0, 255).astype(np.uint8)
        palette = bytes(rng.integers(0, 256, size=3 * 200, dtype=np.uint8)) if color == 3 else b""
        if color == 3:
            image = image % 200
        path = tmp_path / f"frame-{width}x{height}.png"
        path.write_bytes(_png(image, color, method, palette))
        with Image.open(path) as opened:
            assert opened.size == (width, height)
        assert export.png_dhash(path) == dhash(load_rgb(path)), (width, height)


def test_frames_the_tool_cannot_decode_get_no_fingerprint(tool, tmp_path):
    export = tool[0]
    picture = (np.arange(64 * 48 * 3) % 251).astype(np.uint8).reshape(48, 64, 3)
    Image.fromarray(picture).save(tmp_path / "frame.jpg", quality=90)
    Image.fromarray(picture[:, :, 0].astype(np.uint16) * 257).save(tmp_path / "deep.png")      # 16 bits a sample
    (tmp_path / "interlaced.png").write_bytes(_png(picture, 2, 0).replace(struct.pack(">IIBBBBB", 64, 48, 8, 2, 0, 0, 0),
                                                                        struct.pack(">IIBBBBB", 64, 48, 8, 2, 0, 0, 1)))
    (tmp_path / "cut-short.png").write_bytes(_png(picture, 2, 1)[:200])
    for name in ("frame.jpg", "deep.png", "interlaced.png", "cut-short.png", "absent.png"):
        assert export.png_dhash(tmp_path / name) is None, name
    Image.fromarray(picture).save(tmp_path / "frame.png")
    assert export.png_dhash(tmp_path / "frame.png") == dhash(picture)


# ------------------------------------------------------------------- the tool stands alone
def test_the_labeling_tool_imports_nothing_outside_its_folder():
    own = {path.stem for path in LABELER.glob("*.py")}
    for path in sorted(LABELER.glob("*.py")):
        imported = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
        outside = imported - own - set(sys.stdlib_module_names)
        assert outside == set(), f"{path.name} imports {sorted(outside)}"
