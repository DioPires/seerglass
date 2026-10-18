# SPDX-License-Identifier: Apache-2.0
"""The physical chain: one result per seed, the image's shape kept, the card's box followed
through sleeve and tilt, and the game's kinds read from its settings, not from names."""

from __future__ import annotations

import random

import numpy as np
import pytest

from cardvision.augment import physical as P
from cardvision.composites.cards import KindSettings
from helpers import pattern

IMAGE = pattern(270, 194)
INNER = (43, 60, 108, 150)
KINDS = KindSettings.from_kinds({"foilProbability": {"leader": 1.0, "unit": 0.0}, "dice": ["leader"]})
QUIET = {**P.PHYSICAL, "p_sleeve": 0.0, "p_rot": 0.0, "p_glare": 0.0, "strength": (0.0, 0.0)}


@pytest.mark.parametrize("kind", ["leader", "unit", None])
def test_one_seed_gives_one_result_of_the_image_s_shape(kind):
    before = IMAGE.copy()
    outs = [P.physical_chain(IMAGE, INNER, kind, random.Random(seed), KINDS) for seed in (3, 3, 4)]
    for image, box in outs:
        assert image.shape == IMAGE.shape and image.dtype == np.uint8
        x, y, w, h = box
        assert x >= 0 and y >= 0 and x + w <= 194 and y + h <= 270 and w >= 108 - 1 and h >= 2
    assert np.array_equal(outs[0][0], outs[1][0]) and outs[0][1] == outs[1][1]
    assert not np.array_equal(outs[0][0], outs[2][0])
    assert np.array_equal(IMAGE, before)


def test_the_chain_takes_one_draw_from_the_row_s_stream():
    rng, twin = random.Random(8), random.Random(8)
    P.physical_chain(IMAGE, INNER, "unit", rng, KINDS)
    twin.getrandbits(64)
    assert rng.random() == twin.random()


def test_which_kinds_are_foil_or_carry_dice_comes_from_the_kind_settings():
    # With every other effect off, a kind whose foil probability is 1 is changed and a kind
    # whose probability is 0 is not: the difference is the foil finish.
    plain, _ = P.physical_chain(IMAGE, INNER, "unit", random.Random(5), KINDS, {**QUIET})
    foil, _ = P.physical_chain(IMAGE, INNER, "leader", random.Random(5), KINDS, {**QUIET})
    x, y, w, h = INNER
    assert not np.array_equal(plain[y:y + h, x:x + w], foil[y:y + h, x:x + w])
    assert np.array_equal(plain[:y], foil[:y])                                    # nothing outside the card moved
    # The same card kinds under other settings behave the other way round.
    swapped = KindSettings.from_kinds({"foilProbability": {"leader": 0.0, "unit": 1.0}})
    assert np.array_equal(P.physical_chain(IMAGE, INNER, "leader", random.Random(5), swapped, {**QUIET})[0][:y], plain[:y])
    assert not np.array_equal(P.physical_chain(IMAGE, INNER, "unit", random.Random(5), swapped, {**QUIET})[0], plain)
    # Without settings every kind is treated alike.
    a, _ = P.physical_chain(IMAGE, INNER, "leader", random.Random(5), None)
    b, _ = P.physical_chain(IMAGE, INNER, "unit", random.Random(5), None)
    assert np.array_equal(a, b)


def test_a_sleeve_is_a_band_around_the_card_and_grows_the_box():
    for seed in range(12):
        out, box = P.sleeve(IMAGE, INNER, np.random.default_rng(seed), 108)
        x, y, w, h = INNER
        bx, by, bw, bh = box
        border = x - bx
        assert 2 <= border <= 7 and (bw, bh) == (w + 2 * border, h + 2 * border) and by == y - border
        assert np.array_equal(out[y:y + h, x:x + w], IMAGE[y:y + h, x:x + w])     # the card itself is untouched
        assert not np.array_equal(out[by:y, bx:bx + bw], IMAGE[by:y, bx:bx + bw])
        assert np.array_equal(out[:by], IMAGE[:by])
    # At the image's edge the band is clipped.
    _, box = P.sleeve(IMAGE, (0, 0, 108, 150), np.random.default_rng(1), 108)
    assert box[0] == 0 and box[1] == 0


def test_a_tilt_returns_the_upright_box_around_the_tilted_card():
    out, box = P.rotate(IMAGE, INNER, 0.0)
    assert np.array_equal(out, IMAGE) and box == INNER
    out, box = P.rotate(IMAGE, INNER, 90.0)
    x, y, w, h = box
    assert (w, h) == (150, 108) and (x + w / 2, y + h / 2) == (43 + 54, 60 + 75)   # same center, sides swapped
    out, (x, y, w, h) = P.rotate(IMAGE, INNER, 20.0)
    assert out.shape == IMAGE.shape and w > 108 and h > 150


def test_lamp_glare_only_brightens():
    for seed in range(6):
        out = P.lamp_glare(IMAGE, INNER, np.random.default_rng(seed))
        assert out.shape == IMAGE.shape and np.all(out.astype(int) >= IMAGE.astype(int) - 1) and out.sum() > IMAGE.sum()


def test_motion_blur_keeps_the_crop_s_shape_and_touches_about_one_crop_in_seven():
    crop = pattern(150, 108)
    touched = 0
    for seed in range(300):
        out = P.motion_blur(crop, random.Random(seed))
        assert out.shape == crop.shape and out.dtype == np.uint8
        touched += out is not crop
    assert abs(touched / 300 - P.PHYSICAL["p_motion"]) < 0.07
    always = {**P.PHYSICAL, "p_motion": 1.0}
    a, b = P.motion_blur(crop, random.Random(2), always), P.motion_blur(crop, random.Random(2), always)
    assert np.array_equal(a, b) and not np.array_equal(a, crop)
