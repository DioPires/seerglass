# SPDX-License-Identifier: Apache-2.0
"""The detector trainer's data seam: the stream format, the built-in streams and a plug-in."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import support_train_detector as support
from cardvision.composites import read_labels
from cardvision.data import geometry
from cardvision.data.extract import frame_file_name
from cardvision.data.manifest import write_manifest
from cardvision.data.schema import Box, Point, Row
from cardvision.train_detector import sources as S
from cardvision.train_detector.config import ConfigError, parse_config
from helpers import pattern

WIDTH, HEIGHT = 1000, 500
HAND_SOURCE = "digital/client-b"
PLUG_IN = "support_train_detector:extra_stream"


def upright(x, y, width, height) -> tuple:
    return tuple((px / WIDTH, py / HEIGHT) for px, py in geometry.quad_from_box(x, y, width, height))


def face(quad, **fields) -> Box:
    values = {"oriented": True, "shape": "rotated-box", "card_id": "TOY-001", "identity": "named", "readable": "yes"}
    values.update(fields)
    return Box(kind="face", quad=quad, **values)


def labeled_row(t: float, *boxes: Box, **fields) -> Row:
    values = {"platform": "youtube", "video_id": "EXAMPLE0001", "t": t, "frame_width": WIDTH, "frame_height": HEIGHT,
              "source_type": "physical/tournament", "group": "g-toy-01", "role": "train", "trained_on": True,
              "completeness": "exhaustive-boxes", "credit": "Toy Labeler", "boxes": tuple(boxes)}
    values.update(fields)
    return Row(**values)


def target(cx, cy, w, h, k) -> dict:
    return {"cx": pytest.approx(cx), "cy": pytest.approx(cy), "w": pytest.approx(w), "h": pytest.approx(h), "k": k}


def config_for(streams: dict, mix: list[str], **top):
    share = 1.0 / len(mix)
    return parse_config({
        "batch_size": len(mix), "streams": streams,
        "mix": {"primary": [{"stream": name, "tiles": 1} for name in mix],
                "mosaic_extra": [{"stream": name, "share": share} for name in mix]},
        **top,
    })


@pytest.fixture(scope="module")
def data(toy, tmp_path_factory):
    return support.make_data(toy, tmp_path_factory.mktemp("detector-data"))


# ------------------------------------------------------------------------------- real stream
def write_real(folder: Path, rows: list[Row], skip: int | None = None, size=(WIDTH, HEIGHT)) -> dict:
    frames = folder / "frames"
    frames.mkdir(parents=True)
    for index, row in enumerate(rows):
        if index != skip:
            Image.fromarray(pattern(size[1], size[0])).save(frames / frame_file_name(row, ".png"))
    write_manifest(folder / "m.jsonl", rows)
    return {"manifest": str(folder / "m.jsonl"), "frames": str(frames)}


def test_the_real_stream_converts_labels_by_the_documented_rules(tmp_path):
    rows = [
        # A whole card (144 x 200: aspect 0.72), a back, an unreadable box, a not-card box, a
        # face marked ignore, and a face too square to be a whole card (190 x 200: 0.95).
        labeled_row(1.0,
                    face(upright(100, 100, 144, 200)),
                    Box(kind="back", quad=upright(400, 50, 100, 140)),
                    Box(kind="unreadable", quad=upright(600, 50, 100, 140)),
                    Box(kind="not-card", quad=upright(800, 50, 100, 140)),
                    face(upright(300, 250, 144, 200), ignore=True),
                    face(upright(500, 250, 190, 200))),
        # A card that runs 50 px past the right edge is clipped to the frame.
        labeled_row(2.0, Box(kind="back", quad=upright(950, 300, 100, 140))),
        labeled_row(3.0),
    ]
    stream = S.real_stream(config_for({"real": write_real(tmp_path, rows)}, ["real"]))
    assert stream.name == "real" and len(stream) == 3 and stream.degrade == 0.0
    assert stream.targets[0] == [
        target(0.172, 0.4, 0.144, 0.4, 0),   # the whole card: a face
        target(0.45, 0.24, 0.1, 0.28, 1),    # the back
        target(0.65, 0.24, 0.1, 0.28, 2),    # unreadable: an ignore region (the not-card box gives nothing)
        target(0.372, 0.7, 0.144, 0.4, 2),   # marked ignore
        target(0.595, 0.7, 0.19, 0.4, 2),    # the wrong shape for a whole card
    ]
    assert stream.targets[1] == [target(0.975, 0.74, 0.05, 0.28, 1)]
    assert stream.targets[2] == []
    # Frames come back in RGB at the store width, whatever size the file has.
    frame = stream.frames[0]
    assert frame.shape == (352, 704, 3) and frame.dtype == np.uint8
    expected = np.asarray(Image.fromarray(pattern(HEIGHT, WIDTH)).resize((704, 352), Image.BILINEAR))
    assert np.array_equal(frame, expected)
    assert stream.record["frames"] == 3 and stream.record["framesMissing"] == 0
    assert (stream.record["faces"], stream.record["backs"], stream.record["ignoreRegions"]) == (1, 2, 3)
    assert stream.record["manifests"] == [{"file": "m.jsonl", "sha256": S.file_sha256(tmp_path / "m.jsonl")}]
    S.check_stream(stream)


def test_the_real_stream_filters_rows_and_counts_missing_frames(tmp_path):
    card = face(upright(100, 100, 144, 200))
    rows = [labeled_row(1.0, card), labeled_row(2.0, card, trained_on=False),
            labeled_row(3.0, card, role="evaluation", trained_on=False),
            labeled_row(4.0, completeness="points", boxes=None, points=(Point(x=0.5, y=0.5, kind="not-card"),)),
            labeled_row(5.0, card)]
    tables = write_real(tmp_path, rows, skip=4)
    stream = S.real_stream(config_for({"real": tables}, ["real"]))
    assert len(stream) == 2 and stream.record["rows"] == 3 and stream.record["framesMissing"] == 1
    only = S.real_stream(config_for({"real": {**tables, "trained_on": True}}, ["real"]))
    assert len(only) == 1 and only.record["trainedOn"] is True
    both = S.real_stream(config_for({"real": {**tables, "roles": ["train", "evaluation"]}}, ["real"]))
    assert len(both) == 3
    # A frame of another shape cannot carry the labels and counts as missing.
    square = tmp_path / "square"
    with pytest.raises(S.SourceError, match="the real stream is empty: 1 matching row"):
        S.real_stream(config_for({"real": write_real(square, rows[:1], size=(500, 500))}, ["real"]))
    for bad, message in (({"frames": "f"}, "needs manifest"), ({**tables, "color": 1}, "unknown key color"),
                         ({**tables, "aspect_range": [0.5]}, "aspect_range"), ({**tables, "roles": "train"}, "roles")):
        with pytest.raises(ConfigError, match=message):
            S.real_stream(config_for({"real": bad}, ["real"]))


def test_the_real_stream_takes_the_hand_row_from_a_sources_file(tmp_path):
    # 190 x 200 (aspect 0.95) with its center 86 % of the way down, running past the bottom
    # edge: a cut-off hand card for a client that draws its hand that way, and an ignore region
    # anywhere else.
    low = face(upright(500, 330, 190, 200))
    rows = [labeled_row(1.0, low, source_type=HAND_SOURCE), labeled_row(2.0, low)]
    tables = write_real(tmp_path, rows)
    assert [t[0]["k"] for t in S.real_stream(config_for({"real": tables}, ["real"])).targets] == [2, 2]
    (tmp_path / "sources.json").write_text(json.dumps(
        {"sourceTypes": [HAND_SOURCE], "handRow": {"sourceTypes": [HAND_SOURCE], "minCenterY": 0.85}}), encoding="utf-8")
    stream = S.real_stream(config_for({"real": {**tables, "sources": str(tmp_path / "sources.json")}}, ["real"]))
    assert [t[0]["k"] for t in stream.targets] == [0, 2]
    assert stream.record["handRowSourceTypes"] == [HAND_SOURCE]
    assert stream.record["sources"]["file"] == "sources.json"
    off = S.real_stream(config_for({"real": {**tables, "aspect_range": False}}, ["real"]))
    assert [t[0]["k"] for t in off.targets] == [0, 0] and off.record["aspectRange"] is None


# ------------------------------------------------------------------------ composites stream
def test_the_target_codes_are_those_of_the_label_conversions():
    from cardvision.data.targets import DETECTOR_CODES

    assert (S.FACE, S.BACK, S.IGNORE) == (0, 1, 2) == tuple(DETECTOR_CODES[kind] for kind in ("face", "back", "ignore"))
    assert S.TARGET_CODES == (0, 1, 2)


def test_composite_targets_by_hand():
    def item(kind, **fields):
        return {"kind": kind, "cx": 0.5, "cy": 0.4, "w": 0.1, "h": 0.2, "visible": 1.0, "positive": False,
                "ignore": None, **fields}

    box = {"cx": 0.5, "cy": 0.4, "w": 0.1, "h": 0.2}
    scene = {"objects": [
        item("face", positive=True),
        item("face", ignore="lowvis", visible=0.3),
        item("face", ignore="small"),
        item("back"),
        item("back", visible=0.5),
        item("back", visible=0.49),
        item("back", visible=None),
        item("distractor"), item("die"), item("hand"), item("overlay"),
    ]}
    assert S.composite_targets(scene) == [{**box, "k": k} for k in (0, 2, 2, 1, 1, 2, 2)]
    assert S.composite_targets({"objects": []}) == []
    assert S.count_targets([S.composite_targets(scene), []]) == {"faces": 1, "backs": 2, "ignoreRegions": 4}


def test_the_composites_stream_reads_a_generated_folder(data):
    folder = Path(data["streams"]["composites"]["folder"])
    stream = S.composites_stream(config_for(data["streams"], ["composites"]))
    scenes = read_labels(folder)
    assert len(stream) == len(scenes) == 8
    assert [list(t) for t in stream.targets] == [S.composite_targets(scene) for scene in scenes]
    positives = sum(o["kind"] == "face" and o["positive"] for scene in scenes for o in scene["objects"])
    assert stream.record["faces"] == positives > 0
    assert stream.record["labelsSha256"] == hashlib.sha256((folder / "labels.jsonl").read_bytes()).hexdigest()
    assert stream.record["generator"] == {"format": "cardvision-composites", "formatVersion": 1, "split": "train",
                                          "seed": 22}
    for index in (0, 5):
        frame = stream.frames[index]
        assert frame.dtype == np.uint8 and frame.shape[1] == 704 and frame.shape[2] == 3
        assert frame.shape[0] == round(scenes[index]["height"] * 704 / scenes[index]["width"])
    S.check_stream(stream)
    with pytest.raises(S.SourceError, match="no labels.jsonl"):
        S.composites_stream(config_for({"composites": {"folder": str(folder / "images")}}, ["composites"]))


# ------------------------------------------------------------------------- negatives stream
def test_the_negatives_stream_is_a_folder_of_images(tmp_path):
    for name, size in (("b.png", (64, 48)), ("a.jpg", (1408, 800)), ("notes.txt", None)):
        if size is None:
            (tmp_path / name).write_text("not an image", encoding="utf-8")
        else:
            Image.fromarray(pattern(size[1], size[0])).save(tmp_path / name)
    (tmp_path / "deeper").mkdir()
    stream = S.negatives_stream(config_for({"negatives": {"folder": str(tmp_path)}}, ["negatives"]))
    assert len(stream) == 2 and list(stream.targets) == [[], []]
    assert [Path(p).name for p in stream.frames.paths] == ["a.jpg", "b.png"]
    assert stream.frames[0].shape == (400, 704, 3) and stream.frames[1].shape == (528, 704, 3)
    assert stream.record == {"frames": 2, "framesSha256": S.files_sha256([tmp_path / "a.jpg", tmp_path / "b.png"])}
    with pytest.raises(S.SourceError, match="holds no image files"):
        S.negatives_stream(config_for({"negatives": {"folder": str(tmp_path / "deeper")}}, ["negatives"]))
    with pytest.raises(S.SourceError, match="no such folder"):
        S.negatives_stream(config_for({"negatives": {"folder": str(tmp_path / "absent")}}, ["negatives"]))


# ------------------------------------------------------------------------------------ plug-in
def test_a_plug_in_stream_joins_the_built_in_ones(data):
    streams = {**data["streams"], "extra": {"frames": 5, "degrade": 0.6}}
    config = config_for(streams, ["real", "negatives", "composites", "extra"], extra_sources=[PLUG_IN])
    built = S.build_streams(config)
    assert list(built) == ["real", "negatives", "composites", "extra"]
    extra = built["extra"]
    # The function saw the config (5 frames asked for), and the table's degrade replaced its own.
    assert len(extra) == 5 and extra.degrade == 0.6 and built["real"].degrade == 0.0
    assert extra.frames[0].shape == (180, 320, 3) and [t["k"] for t in extra.targets[0]] == [0, 2]
    records = S.stream_records(built)
    assert records["extra"] == {"frames": 5, "degrade": 0.6, "origin": "made in the test"}
    assert records["real"]["frames"] == 6 and records["negatives"]["frames"] == 4
    assert json.loads(json.dumps(records)) == records
    # A stream the mix does not name is not built: no table is needed for it.
    alone = S.build_streams(config_for({"extra": {}}, ["extra"], extra_sources=[PLUG_IN]))
    assert list(alone) == ["extra"] and len(alone["extra"]) == support.EXTRA_FRAMES


def renamed_stream(config):
    return [replace(support.extra_stream(config), name="real")]


def two_streams(config):
    first = support.extra_stream(config)
    return [first, replace(first, name="second")]


def not_a_stream(config):
    return {"frames": [], "targets": []}


def test_what_the_seam_refuses(data):
    here = "test_train_detector_sources"
    with pytest.raises(ConfigError, match='names the stream "extra", which is not built in'):
        S.build_streams(config_for(data["streams"], ["negatives", "extra"]))
    with pytest.raises(ConfigError, match=r"\[streams.composites\] needs folder"):
        S.build_streams(config_for({}, ["composites"]))
    with pytest.raises(S.SourceError, match="name of a built-in stream"):
        S.build_streams(config_for(data["streams"], ["negatives"], extra_sources=[f"{here}:renamed_stream"]))
    with pytest.raises(ConfigError, match=r"returned second, which \[mix\] gives no tiles"):
        S.build_streams(config_for({}, ["extra"], extra_sources=[f"{here}:two_streams"]))
    with pytest.raises(S.SourceError, match="must return Stream objects"):
        S.build_streams(config_for({}, ["extra"], extra_sources=[f"{here}:not_a_stream"]))
    with pytest.raises(S.SourceError, match="cannot import"):
        S.build_streams(config_for({}, ["extra"], extra_sources=["no_such_module_anywhere:f"]))


def test_check_stream_holds_plug_ins_to_the_format():
    frame = np.zeros((20, 30, 3), np.uint8)
    target = {"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2, "k": 1}
    good = S.Stream("mine", [frame, frame], [[target], []])
    assert S.check_stream(good) is good and len(good) == 2
    for bad, message in (
        (S.Stream("mine", [frame], [[target], []]), "1 frame.s. and 2 target list"),
        (S.Stream("mine", [], []), "not empty"),
        (S.Stream("", [frame], [[]]), "needs a name"),
        (S.Stream("mine", [frame.astype(np.float32)], [[]]), "uint8 RGB array"),
        (S.Stream("mine", [frame[:, :, 0]], [[]]), "uint8 RGB array"),
        (S.Stream("mine", [frame], [[{**target, "k": 3}]]), "k is 0 .face., 1 .back. or 2"),
        (S.Stream("mine", [frame], [[{"cx": 0.5, "cy": 0.5}]]), "dictionary with the keys"),
        (S.Stream("mine", [frame], [[{**target, "w": 0.0}]]), "positive width and height"),
        (S.Stream("mine", [frame], [[{**target, "cx": float("nan")}]]), "finite fractions"),
        (S.Stream("mine", [frame], [[target]], degrade=1.5), "degrade is a probability"),
        (S.Stream("mine", 3, [[target]]), "must both have a length"),
    ):
        with pytest.raises(S.SourceError, match=message):
            S.check_stream(bad)


# ------------------------------------------------------------------------------ frame readers
def test_frame_readers(tmp_path):
    arrays = [pattern(40, 60), pattern(30, 90), pattern(50, 50)]
    paths = []
    for index, array in enumerate(arrays):
        paths.append(tmp_path / f"{index}.png")
        Image.fromarray(array).save(paths[-1])
    plain = S.ImageFrames(paths)
    assert len(plain) == 3 and all(np.array_equal(plain[i], arrays[i]) for i in range(3))
    kept = S.ImageFrames(paths, width=120, cache=2)
    assert [kept[i].shape for i in (0, 1, 2, 0)] == [(80, 120, 3), (40, 120, 3), (120, 120, 3), (80, 120, 3)]
    assert len(kept._kept) == 2 and len(pickle.loads(pickle.dumps(kept))._kept) == 0
    assert np.array_equal(kept[1], np.asarray(Image.fromarray(arrays[1]).resize((120, 40), Image.BILINEAR)))

    stack = np.stack([pattern(8, 12), pattern(8, 12)[::-1]])
    np.save(tmp_path / "stack.npy", stack)
    lazy = pickle.loads(pickle.dumps(S.NpyFrames(tmp_path / "stack.npy")))
    assert lazy._array is None and len(lazy) == 2 and np.array_equal(lazy[1], stack[1])

    joined = S.Concat([arrays[:2], S.Subset(arrays, [2, 0])])
    assert len(joined) == 4
    assert [joined[i].shape for i in range(4)] == [(40, 60, 3), (30, 90, 3), (50, 50, 3), (40, 60, 3)]
    assert S.files_sha256(paths) != S.files_sha256(paths[::-1])
