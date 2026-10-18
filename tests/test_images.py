# SPDX-License-Identifier: Apache-2.0
"""Reading image files as RGB arrays."""

from __future__ import annotations

import numpy as np
from PIL import Image

import toy_game
from cardvision.core.images import load_rgb


def test_toy_art_loads_as_rgb_uint8(toy):
    _, art_dir = toy
    portrait = load_rgb(art_dir / "TOY-001.png")
    landscape = load_rgb(str(art_dir / "TOY-017.png"))
    assert portrait.dtype == np.uint8 and portrait.shape == (toy_game.PORTRAIT[1], toy_game.PORTRAIT[0], 3)
    assert landscape.shape == (toy_game.LANDSCAPE[1], toy_game.LANDSCAPE[0], 3)
    assert np.array_equal(portrait, np.asarray(toy_game.draw_card("TOY-001", "Amber Circle", "unit")))


def test_the_array_is_a_writable_copy(toy):
    image = load_rgb(toy[1] / "TOY-002.png")
    image[0, 0] = 0  # must not raise
    assert image.flags.writeable and image.flags.c_contiguous


def test_other_modes_are_converted_to_rgb(tmp_path):
    rgba = np.zeros((6, 4, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 3] = 200, 128
    Image.fromarray(rgba, "RGBA").save(tmp_path / "rgba.png")
    out = load_rgb(tmp_path / "rgba.png")
    assert out.shape == (6, 4, 3) and (out[..., 0] == 200).all() and (out[..., 1:] == 0).all()

    gray = np.full((6, 4), 77, dtype=np.uint8)
    Image.fromarray(gray, "L").save(tmp_path / "gray.png")
    assert (load_rgb(tmp_path / "gray.png") == 77).all()

    Image.fromarray(gray, "L").convert("P").save(tmp_path / "palette.png")
    out = load_rgb(tmp_path / "palette.png")
    assert out.shape == (6, 4, 3) and (out == 77).all()
