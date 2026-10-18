# SPDX-License-Identifier: Apache-2.0
"""From streams to training images: tiles, the mosaic, the canvas and the arrays for the loss,
against cases worked out by hand."""

from __future__ import annotations

import hashlib
import pickle

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("torch")
pytest.importorskip("yolox.data.data_augment")

from PIL import Image  # noqa: E402

from cardvision.train_detector import augment as A  # noqa: E402
from cardvision.train_detector.config import parse_config  # noqa: E402
from cardvision.train_detector.sources import BACK, FACE, IGNORE, Stream  # noqa: E402

W, H = 640, 384


def flat(value: int, height: int = H, width: int = W) -> np.ndarray:
    return np.full((height, width, 3), value, np.uint8)


def config_for(primary, extra, **top):
    return parse_config({
        "batch_size": sum(tiles for _, tiles in primary),
        "mix": {"primary": [{"stream": name, "tiles": tiles} for name, tiles in primary],
                "mosaic_extra": [{"stream": name, "share": share} for name, share in extra]},
        **top,
    })


# ---------------------------------------------------------------------------------- one tile
def test_fit_tile_resizes_reorders_and_places_the_boxes():
    frame = np.zeros((200, 400, 3), np.uint8)
    frame[:] = (10, 20, 30)  # red, green, blue
    image, boxes = A.fit_tile(frame, [{"cx": 0.5, "cy": 0.5, "w": 0.25, "h": 0.5, "k": BACK},
                                      {"cx": 0.1, "cy": 0.2, "w": 0.2, "h": 0.4, "k": IGNORE}])
    # 400 x 200 fits the 640 x 384 canvas at 1.6: 640 x 320.
    assert image.shape == (320, 640, 3) and image.dtype == np.uint8
    assert image[0, 0].tolist() == [30, 20, 10]  # blue, green, red
    assert boxes.dtype == np.float32
    assert boxes.tolist() == [[240.0, 80.0, 400.0, 240.0, BACK], [0.0, 0.0, 128.0, 128.0, IGNORE]]
    tall, none = A.fit_tile(np.zeros((768, 320, 3), np.uint8), [])
    assert tall.shape == (384, 160, 3) and none.shape == (0, 5)


def test_pack_splits_targets_from_ignore_regions():
    boxes = np.array([
        [10, 10, 50, 70, FACE],
        [100, 100, 100.5, 200, FACE],   # half a pixel wide: dropped
        [200, 50, 260, 150, BACK],
        [300, 300, 400, 380, IGNORE],
        [0, 0, 1, 1, IGNORE],           # a side of exactly 1 px: dropped
        [5, 5, 9, 9, IGNORE],
    ], np.float32)
    targets, ignore, counts = A.pack(boxes, max_targets=4, max_ignore=3)
    assert targets.shape == (4, 5) and ignore.shape == (3, 4) and targets.dtype == ignore.dtype == np.float32
    assert targets.tolist() == [[0, 30, 40, 40, 60], [1, 230, 100, 60, 100], [0] * 5, [0] * 5]
    assert ignore.tolist() == [[350, 340, 100, 80], [7, 7, 4, 4], [0] * 4]
    assert counts.tolist() == [2, 2, 1]
    # The limits cut the arrays, not the counts.
    targets, ignore, counts = A.pack(boxes, max_targets=1, max_ignore=1)
    assert targets.tolist() == [[0, 30, 40, 40, 60]] and ignore.tolist() == [[350, 340, 100, 80]]
    assert counts.tolist() == [2, 2, 1]
    targets, ignore, counts = A.pack(np.zeros((0, 5), np.float32), 2, 2)
    assert not targets.any() and not ignore.any() and counts.tolist() == [0, 0, 0]


def test_finish_flips_fits_and_pads():
    image = flat(60, 100, 200)
    boxes = np.array([[10, 20, 50, 60, FACE]], np.float32)
    canvas, moved = A.finish(image.copy(), boxes.copy(), flip=1.0)
    # 200 x 100 fits at 3.2: 640 x 320, and the 64 rows below are padding.
    assert canvas.shape == (H, W, 3) and canvas.dtype == np.uint8
    assert (canvas[320:] == 114).all() and not (canvas[:320] == 114).all()
    # Mirrored in the 200 px wide image, then scaled: x 10..50 becomes 150..190, times 3.2.
    assert moved.tolist() == [[480.0, 64.0, 608.0, 192.0, FACE]]
    _, kept = A.finish(image.copy(), boxes.copy(), flip=0.0)
    assert kept.tolist() == [[32.0, 64.0, 160.0, 192.0, FACE]]
    full, same = A.finish(flat(60), np.array([[1, 2, 3, 4, BACK]], np.float32), flip=0.0)
    assert full.shape == (H, W, 3) and same.tolist() == [[1, 2, 3, 4, BACK]]


def test_mosaic_pastes_four_tiles_around_the_center():
    tiles = [(flat(value), np.array([[400, 250, 500, 350, kind]], np.float32))
             for value, kind in ((10, FACE), (60, BACK), (120, IGNORE), (200, FACE))]
    # No rotation, shift, zoom or shear: the result is the top-left canvas-sized part of the
    # double canvas, which shows one quarter of each tile around the center (320, 192).
    image, boxes = A.mosaic(tiles, 320, 192, degrees=0.0, translate=0.0, scale=(1.0, 1.0), shear=0.0)
    assert image.shape == (H, W, 3)
    assert (image[:192, :320] == 10).all() and (image[:192, 320:] == 60).all()
    assert (image[192:, :320] == 120).all() and (image[192:, 320:] == 200).all()
    # Tile 0 shows its bottom-right quarter, so its box moves by (-320, -192); tile 1 its
    # bottom-left (+320, -192); tile 2 its top-right (-320, +192); tile 3 its top-left
    # (+320, +192). Boxes that leave the canvas are clipped to it.
    assert boxes.tolist() == [[80, 58, 180, 158, FACE], [640, 58, 640, 158, BACK],
                              [80, 384, 180, 384, IGNORE], [640, 384, 640, 384, FACE]]


def test_degrade_changes_pixels_and_keeps_the_shape():
    rng = np.random.default_rng(1)
    frame = Image.fromarray(rng.integers(0, 256, (90, 160, 3), dtype=np.uint8))
    first = np.asarray(A.degrade(frame, np.random.RandomState(4)))
    again = np.asarray(A.degrade(frame, np.random.RandomState(4)))
    other = np.asarray(A.degrade(frame, np.random.RandomState(5)))
    assert first.shape == (90, 160, 3) and first.dtype == np.uint8
    assert np.array_equal(first, again) and not np.array_equal(first, other)
    assert not np.array_equal(first, np.asarray(frame))


# ------------------------------------------------------------------------------ whole images
def streams():
    card = [{"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.4, "k": FACE}]
    return {
        "bright": Stream("bright", [flat(200), flat(210), flat(220)], [card, card, card]),
        "dark": Stream("dark", [flat(20), flat(30)], [[], []]),
    }


def test_the_seed_of_an_image_is_pinned():
    # The seeding rule is part of what makes a run repeatable; changing it changes every run.
    # It is the first four bytes of SHA-256("salt/seed/index"), read little-endian.
    assert A.tile_seed("tiles", 1337, 0) == 3194814406
    assert A.tile_seed("tiles", 1337, 0) == int.from_bytes(hashlib.sha256(b"tiles/1337/0").digest()[:4], "little")
    assert A.tile_seed("tiles", 1337, 1) != A.tile_seed("tiles", 1338, 1) != A.tile_seed("other", 1338, 1)


def test_batch_position_decides_the_first_tile():
    config = config_for([("bright", 3), ("dark", 1)], [("bright", 0.5), ("dark", 0.5)], steps=3, seed=9,
                        augment={"mosaic": 0.0})
    tiles = A.Tiles(streams(), config)
    assert len(tiles) == 12
    for j in range(len(tiles)):
        image, targets, ignore, counts = tiles[j]
        assert image.shape == (3, H, W) and image.dtype == np.float32
        assert targets.shape == (200, 5) and ignore.shape == (480, 4) and counts.shape == (3,)
        # Without mosaic an image is one frame: flat, and bright or dark by its position.
        assert np.ptp(image, axis=(1, 2)).max() == 0
        if j % 4 < 3:
            assert image.mean() > 114 and counts.tolist() == [1, 0, 0]
            assert targets[0].tolist() == [0.0, 320.0, 192.0, 128.0, pytest.approx(153.6)]
        else:
            assert image.mean() < 114 and counts.tolist() == [0, 0, 0] and not targets.any()
    with pytest.raises(IndexError):
        tiles[12]
    with pytest.raises(ValueError, match="streams that were not built: dark"):
        A.Tiles({"bright": streams()["bright"]}, config)


def test_images_repeat_and_do_not_depend_on_the_order_they_are_made_in():
    config = config_for([("bright", 1), ("dark", 1)], [("bright", 0.5), ("dark", 0.5)], steps=4, seed=3)
    tiles = A.Tiles(streams(), config)
    forward = [tiles[j] for j in range(8)]
    copy = pickle.loads(pickle.dumps(tiles))
    for j in reversed(range(8)):
        for a, b in zip(forward[j], copy[j]):
            assert np.array_equal(a, b)
    other = A.Tiles(streams(), config.with_changes(seed=4))
    assert any(not np.array_equal(forward[j][0], other[j][0]) for j in range(8))


def test_the_closing_phase_has_no_mosaic():
    config = config_for([("bright", 1), ("dark", 1)], [("bright", 0.5), ("dark", 0.5)], steps=10, seed=1,
                        augment={"mosaic": 1.0, "no_augment_share": 0.5})
    assert config.no_augment_from == 5
    tiles = A.Tiles(streams(), config)
    spread = [float(np.ptp(tiles[j][0], axis=(1, 2)).max()) for j in range(20)]
    # A mosaic shows several frames or the gray beyond them; a plain image is one flat frame.
    assert all(value > 0 for value in spread[:10]) and all(value == 0 for value in spread[10:])
    targets = sum(int(tiles[j][3][0]) for j in range(10))
    assert targets > 5  # more than the five the bright first tiles alone would give


def test_a_stream_can_ask_for_degraded_frames():
    rng = np.random.default_rng(2)
    noise = [rng.integers(0, 256, (H, W, 3), dtype=np.uint8)]
    config = config_for([("noise", 1)], [("noise", 1.0)], steps=4, seed=6, augment={"mosaic": 0.0, "flip": 0.0})
    clean = A.Tiles({"noise": Stream("noise", noise, [[]])}, config)
    rough = A.Tiles({"noise": Stream("noise", noise, [[]], degrade=1.0)}, config)
    for j in range(4):
        assert not np.array_equal(clean[j][0], rough[j][0])
        assert np.array_equal(clean[j][0], clean[j][0])


def test_further_mosaic_tiles_follow_the_shares():
    config = config_for([("bright", 1)], [("bright", 0.25), ("dark", 0.75)], steps=1)
    tiles = A.Tiles(streams(), config)
    rng = np.random.RandomState(0)
    picks = [tiles.pick_extra(rng) for _ in range(4000)]
    share = sum(name == "bright" for name, _ in picks) / len(picks)
    assert share == pytest.approx(0.25, abs=0.03)
    assert {index for name, index in picks if name == "bright"} == {0, 1, 2}
    assert {index for name, index in picks if name == "dark"} == {0, 1}
