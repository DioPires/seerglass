# SPDX-License-Identifier: Apache-2.0
"""Toy models and scenes for the inference tests, built at run time, and checks of the builders.

Nothing here has any skill. The "recognizer" is an average pool and a fixed random projection,
which is enough to tell the toy game's cards and their four orientations apart. The "detector"
does not look at the image: it returns the rows it was built with, so a test decides where the
boxes are. A second detector reports the mean of each input channel, to observe exactly what
the input contract fed it. All of them are ordinary ONNX files, loaded like real ones.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from cardvision.core.bank import encode_bank, l2_normalize
from cardvision.core.catalog import load_catalog
from cardvision.core.images import load_rgb
from cardvision.core.onnx import create_session, embed
from cardvision.core.preprocess import FULL128, full128
from cardvision.detect import INPUT_HEIGHT, INPUT_WIDTH, letterbox_ratio

EMBEDDING_WIDTH = 128
TABLE_COLOR = (18, 52, 36)
# A frame of this size is scaled by exactly one half to fit the detector's canvas and needs no
# resize before analysis, so a box at whole pixels survives the round trip through the
# detector's coordinates without any rounding.
SCENE_SIZE = (1280, 768)


def _save(graph, path: Path) -> Path:
    import onnx
    from onnx import helper

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def build_recognizer(path: Path, declare_width: bool = True) -> Path:
    """An embedding model with the recognizer's interface: ``crops`` ``(n, 3, 128, 128)`` in,
    unit-length ``embeddings`` ``(n, 128)`` out. With ``declare_width`` false the file leaves the
    embedding width undeclared, as some exporters do."""
    from onnx import TensorProto, helper, numpy_helper

    weights = np.random.default_rng(42).normal(size=(3 * 8 * 8, EMBEDDING_WIDTH)).astype(np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("AveragePool", ["crops"], ["pooled"], kernel_shape=[16, 16], strides=[16, 16]),
            helper.make_node("Flatten", ["pooled"], ["flat"], axis=1),
            helper.make_node("MatMul", ["flat", "weights"], ["projected"]),
            helper.make_node("LpNormalization", ["projected"], ["embeddings"], axis=1, p=2),
        ],
        "toy-recognizer",
        [helper.make_tensor_value_info("crops", TensorProto.FLOAT, ["n", 3, 128, 128])],
        [helper.make_tensor_value_info("embeddings", TensorProto.FLOAT, ["n", EMBEDDING_WIDTH if declare_width else "d"])],
        [numpy_helper.from_array(weights, "weights")],
    )
    return _save(graph, Path(path))


def build_bank(
    path: Path,
    recognizer_path: Path,
    catalog_path: Path,
    art_dir: Path,
    *,
    leave_out: Sequence[str] = (),
    art_of: Mapping[str, str] | None = None,
) -> Path:
    """A bank with one row per catalog card, embedded from its image.

    ``leave_out`` lists cards given no row. ``art_of`` maps a card id to another card id whose
    image is embedded in its place, to stage two rows that look exactly alike.
    """
    catalog = load_catalog(catalog_path)
    art_of = dict(art_of or {})
    ids = [card.id for card in catalog if card.id not in leave_out]
    crops = [load_rgb(Path(art_dir) / catalog.card(art_of.get(card_id, card_id)).art) for card_id in ids]
    rows = l2_normalize(embed(create_session(recognizer_path), crops, full128))
    doc = encode_bank(ids, rows, {"dim": EMBEDDING_WIDTH, "preprocessing": FULL128})
    Path(path).write_text(json.dumps(doc), encoding="utf-8")
    return Path(path)


def detector_rows(
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    classes: Sequence[int],
    frame_size: tuple[int, int] = SCENE_SIZE,
) -> np.ndarray:
    """Raw detector rows that decode to the given ``(x, y, width, height)`` frame boxes.

    Objectness carries the score and the chosen class scores 1, the other 0.
    """
    ratio = letterbox_ratio(*frame_size)
    rows = np.zeros((len(boxes), 7), dtype=np.float32)
    for row, (x, y, width, height), score, cls in zip(rows, boxes, scores, classes):
        row[:4] = ((x + width / 2) * ratio, (y + height / 2) * ratio, width * ratio, height * ratio)
        row[4] = score
        row[5 + cls] = 1.0
    return rows


def build_detector(path: Path, rows: np.ndarray) -> Path:
    """A detector that returns ``rows`` for any image: ``images`` ``(1, 3, 384, 640)`` in,
    ``output`` ``(1, len(rows), 7)`` out."""
    from onnx import TensorProto, helper, numpy_helper

    rows = np.asarray(rows, dtype=np.float32)[None]
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["images"], ["mean"], keepdims=0),
            helper.make_node("Mul", ["mean", "zero"], ["nothing"]),
            helper.make_node("Add", ["rows", "nothing"], ["output"]),
        ],
        "toy-detector",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, INPUT_HEIGHT, INPUT_WIDTH])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, list(rows.shape))],
        [numpy_helper.from_array(rows, "rows"), numpy_helper.from_array(np.float32(0), "zero")],
    )
    return _save(graph, Path(path))


def build_probe_detector(path: Path) -> Path:
    """A detector whose single row starts with the mean of input channels 0, 1 and 2."""
    from onnx import TensorProto, helper, numpy_helper

    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["images"], ["means"], axes=[2, 3], keepdims=0),
            helper.make_node("Concat", ["means", "tail"], ["row"], axis=1),
            helper.make_node("Reshape", ["row", "shape"], ["output"]),
        ],
        "probe-detector",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, INPUT_HEIGHT, INPUT_WIDTH])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 7])],
        [
            numpy_helper.from_array(np.zeros((1, 4), dtype=np.float32), "tail"),
            numpy_helper.from_array(np.array([1, 1, 7], dtype=np.int64), "shape"),
        ],
    )
    return _save(graph, Path(path))


def make_scene(
    art_dir: Path,
    placements: Sequence[tuple[str, int, int, int]],
    size: tuple[int, int] = SCENE_SIZE,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Lay toy cards on a plain table.

    ``placements`` are ``(card id, x, y, quarter turns counter-clockwise)``. A card may hang
    over the edge of the table. Returns the frame and each card's full ``(x, y, width,
    height)`` box, which for an overhanging card reaches outside the frame.
    """
    width, height = size
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:] = TABLE_COLOR
    boxes = []
    for card_id, x, y, turns in placements:
        card = np.rot90(load_rgb(Path(art_dir) / f"{card_id}.png"), turns)
        card_height, card_width = card.shape[:2]
        x0, y0, x1, y1 = max(0, x), max(0, y), min(width, x + card_width), min(height, y + card_height)
        frame[y0:y1, x0:x1] = card[y0 - y:y1 - y, x0 - x:x1 - x]
        boxes.append((x, y, card_width, card_height))
    return frame, boxes


def save_image(path: Path, rgb: np.ndarray) -> Path:
    """Write an RGB array as a lossless image file."""
    Image.fromarray(rgb).save(path)
    return Path(path)


# A strict accept rule for the toy recognizer, whose different cards can resemble each other.
TOY_TAU, TOY_DELTA = 0.99, 0.02
LEFT_OUT = "TOY-012"

# One table of cards, with every kind of result on it. In order of detector score:
#   0  TOY-001 upright                       -> named
#   1  TOY-003 turned a quarter              -> named after the turned looks are tried
#   2  TOY-012, which the bank lacks         -> nothing: no bank row comes close
#   3  TOY-015, which TOY-016's row copies   -> nothing: two names equally close
#   4  TOY-017, a landscape card             -> named
#   5  a card back                           -> reported, never cropped
#   6  TOY-005 hanging over the right edge   -> box clipped to the frame; nothing
#   -  a 20 x 20 px box                      -> dropped: its crop is under 24 px
#   -  a box scoring 0.2                     -> under the detector's threshold
SCENE_CARDS = [("TOY-001", 40, 40, 0), ("TOY-003", 320, 40, 1), (LEFT_OUT, 700, 40, 0), ("TOY-015", 980, 40, 0),
               ("TOY-017", 40, 420, 0), ("TOY-005", 1160, 420, 0)]
SCENE_BACK = (420, 420, 240, 336)
SCENE_TINY = (700, 500, 20, 20)
SCENE_WEAK = (900, 500, 100, 100)
EXPECTED_BOXES = [
    [40.0, 40.0, 240.0, 336.0],
    [320.0, 40.0, 336.0, 240.0],
    [700.0, 40.0, 240.0, 336.0],
    [980.0, 40.0, 240.0, 336.0],
    [40.0, 420.0, 336.0, 240.0],
    [420.0, 420.0, 240.0, 336.0],
    [1160.0, 420.0, 120.0, 336.0],
]
EXPECTED_CLASSES = ["face", "face", "face", "face", "face", "back", "face"]
EXPECTED_NAMES = ["Amber Circle", "Coral Triangle", None, None, "Amber Stripes", None]   # the six faces
EXPECTED_SCORES = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65]


class ToySetup:
    """Paths of the toy models and files, and the scene, for one test module."""

    def __init__(self, folder: Path, catalog_path: Path, art_dir: Path) -> None:
        self.folder = Path(folder)
        self.catalog = Path(catalog_path)
        self.art_dir = Path(art_dir)
        self.recognizer = build_recognizer(self.folder / "recognizer.onnx")
        self.bank = build_bank(self.folder / "bank.json", self.recognizer, self.catalog, self.art_dir,
                               leave_out=[LEFT_OUT], art_of={"TOY-016": "TOY-015"})
        self.accept = self.folder / "accept.json"
        self.accept.write_text(json.dumps({"tau": TOY_TAU, "delta": TOY_DELTA, "rotationRetry": True}), encoding="utf-8")
        self.frame, card_boxes = make_scene(self.art_dir, SCENE_CARDS)
        boxes = [*card_boxes[:5], SCENE_BACK, card_boxes[5], SCENE_TINY, SCENE_WEAK]
        scores = [*EXPECTED_SCORES, 0.6, 0.2]
        classes = [0, 0, 0, 0, 0, 1, 0, 0, 0]
        # The model's rows are in a scrambled order; the decode sorts them by score.
        order = [4, 8, 1, 6, 0, 7, 3, 5, 2]
        rows = detector_rows(boxes, scores, classes)[order]
        self.detector = build_detector(self.folder / "detector.onnx", rows)
        self.image = save_image(self.folder / "table.png", self.frame)

    def arguments(self) -> list[str]:
        """The model arguments every ``cardvision infer`` call needs."""
        return ["--detector", str(self.detector), "--recognizer", str(self.recognizer), "--bank", str(self.bank),
                "--catalog", str(self.catalog), "--accept", str(self.accept)]


# ----------------------------------------------------------------------------- the builders work


def test_toy_recognizer_gives_unit_embeddings_that_tell_cards_and_turns_apart(tmp_path, toy):
    _, art_dir = toy
    session = create_session(build_recognizer(tmp_path / "recognizer.onnx"))
    card = load_rgb(art_dir / "TOY-001.png")
    other = load_rgb(art_dir / "TOY-002.png")
    vectors = embed(session, [card, card, other, np.ascontiguousarray(np.rot90(card))], full128)
    assert vectors.shape == (4, EMBEDDING_WIDTH)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert float(vectors[0] @ vectors[1]) > 0.9999
    assert float(vectors[0] @ vectors[2]) < 0.99
    assert float(vectors[0] @ vectors[3]) < 0.99


def test_toy_detector_returns_its_rows_whatever_the_image(tmp_path):
    rows = detector_rows([(40, 40, 240, 336), (700, 300, 336, 240)], [0.9, 0.8], [0, 1])
    session = create_session(build_detector(tmp_path / "detector.onnx", rows))
    for fill in (0.0, 255.0):
        image = np.full((1, 3, INPUT_HEIGHT, INPUT_WIDTH), fill, dtype=np.float32)
        np.testing.assert_array_equal(session.run(None, {"images": image})[0], rows[None])
    # Row 0, by hand: the frame is scaled by one half; center (160, 208) -> (80, 104), size halved.
    np.testing.assert_array_equal(rows[0], [80, 104, 120, 168, np.float32(0.9), 1, 0])
    np.testing.assert_array_equal(rows[1], [434, 210, 168, 120, np.float32(0.8), 0, 1])


def test_scene_places_cards_and_reports_their_boxes(toy):
    _, art_dir = toy
    frame, boxes = make_scene(art_dir, [("TOY-001", 40, 40, 0), ("TOY-002", 400, 60, 1), ("TOY-003", 1160, 500, 0)])
    assert frame.shape == (768, 1280, 3)
    assert boxes == [(40, 40, 240, 336), (400, 60, 336, 240), (1160, 500, 240, 336)]
    np.testing.assert_array_equal(frame[40:376, 40:280], load_rgb(art_dir / "TOY-001.png"))
    np.testing.assert_array_equal(frame[60:300, 400:736], np.rot90(load_rgb(art_dir / "TOY-002.png")))
    np.testing.assert_array_equal(frame[500:768, 1160:1280], load_rgb(art_dir / "TOY-003.png")[:268, :120])
    assert tuple(frame[0, 0]) == TABLE_COLOR
