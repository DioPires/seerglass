# SPDX-License-Identifier: Apache-2.0
"""Occluders, glare, low light and rotation: own random streams, one result per seed, the
crop's shape kept, and the card's name and art never covered or burned out beyond the caps."""

from __future__ import annotations

import random

import numpy as np
import pytest

from cardvision.augment import components as C
from helpers import pattern

PORTRAIT = pattern(150, 108)
LANDSCAPE = pattern(108, 150)
OTHER = pattern(120, 86)[:, ::-1].copy()


def forced(name: str, **over) -> dict:
    """A component's settings with its probability set to 1."""
    return {**C.DEFAULTS[name], "p": 1.0, **over}


CALLS = {
    "glare": lambda crop, rng: C.glare(crop, rng, forced("glare"))[0],
    "occluders": lambda crop, rng: C.occluders(crop, rng, forced("occluders"), other_art=OTHER)[0],
    "low_light": lambda crop, rng: C.low_light(crop, rng, forced("low_light"))[0],
}


@pytest.mark.parametrize("name", list(CALLS))
@pytest.mark.parametrize("crop", [PORTRAIT, LANDSCAPE], ids=["portrait", "landscape"])
def test_one_seed_gives_one_result_of_the_crop_s_shape(name, crop):
    before = crop.copy()
    outs = [CALLS[name](crop, random.Random(seed)) for seed in (5, 5, 6, 7, 8)]
    assert all(out.shape == crop.shape and out.dtype == np.uint8 for out in outs)
    assert np.array_equal(outs[0], outs[1])
    assert any(not np.array_equal(outs[0], out) for out in outs[2:])
    assert np.array_equal(crop, before)                                           # the input is not written to


def test_each_component_has_its_own_stream_per_seed_epoch_and_row():
    first = C.stream_for(7, 0, 12, "glare").random()
    assert first == C.stream_for(7, 0, 12, "glare").random()
    others = [C.stream_for(8, 0, 12, "glare"), C.stream_for(7, 1, 12, "glare"), C.stream_for(7, 0, 13, "glare"),
              C.stream_for(7, 0, 12, "occluders"), C.stream_for(7, 0, 12, "low_light"), C.stream_for(7, 0, 12, "rotation"),
              random.Random(C.row_seed(7, 0, 12))]
    assert len({first, *[rng.random() for rng in others]}) == len(others) + 1
    with pytest.raises(KeyError):
        C.stream_for(7, 0, 12, "no-such-component")


def test_a_component_s_probability_leaves_most_crops_alone():
    for name, call in (("glare", C.glare), ("occluders", C.occluders), ("low_light", C.low_light)):
        touched = sum(call(PORTRAIT, random.Random(seed))[1] is not None for seed in range(400))
        assert abs(touched / 400 - C.DEFAULTS[name]["p"]) < 0.08, name
        crop, info = call(PORTRAIT, random.Random(0), {**C.DEFAULTS[name], "p": 0.0})
        assert crop is PORTRAIT and info is None


def test_the_identity_region_is_the_upper_part_of_a_portrait_card_and_most_of_a_landscape_one():
    assert C.identity_region(100, 70) == (8, 7, 61, 62)
    assert C.identity_region(70, 100) == (10, 5, 90, 56)
    assert C.identity_region(100, 100) == (12, 7, 88, 62)                         # square counts as portrait


def test_an_occluder_never_covers_more_than_the_caps():
    cfg = forced("occluders")
    kinds = set()
    for seed in range(300):
        crop = PORTRAIT if seed % 2 else LANDSCAPE
        out, info = C.occluders(crop, random.Random(seed), cfg, other_art=OTHER)
        kinds.add(info["kind"])
        if info.get("rejected"):
            assert out is crop
            continue
        assert 0.01 < info["area"] <= cfg["area_max"] and info["identityCover"] <= cfg["identity_cover_max"]
        assert not np.array_equal(out, crop)
    assert kinds == {"card", "fingers", "disc", "sleeve_edge"}


def test_a_negative_row_never_gets_a_card_laid_over_it():
    cfg = forced("occluders")
    kinds = {C.occluders(PORTRAIT, random.Random(seed), cfg, other_art=OTHER, negative=True)[1]["kind"] for seed in range(300)}
    assert kinds == {"fingers", "disc", "sleeve_edge"}


def test_glare_never_burns_out_more_of_the_identity_region_than_the_cap():
    cfg = forced("glare", streak_amp=(3.0, 3.0), bloom_amp=(3.0, 3.0))            # far too strong on purpose
    kinds = set()
    for seed in range(120):
        out, info = C.glare(PORTRAIT, random.Random(seed), cfg)
        kinds.add(info["kind"])
        x0, y0, x1, y1 = C.identity_region(*PORTRAIT.shape[:2])
        saturated = (out[y0:y1, x0:x1] >= 250).all(-1).mean()
        # weakened by 0.6 each try; after the last try the result is kept whatever it is
        assert saturated <= cfg["identity_saturated_max"] or info["scale"] <= 0.6 ** (cfg["tries"] - 1) + 1e-9
    assert kinds == {"streak", "bloom", "foil", "washout"}


def test_low_light_darkens_but_keeps_the_identity_region_above_the_floor():
    bright = np.full((150, 108, 3), 200, np.uint8)
    out, info = C.low_light(bright, random.Random(3), forced("low_light", gain=(0.5, 0.5), gamma=(1.0, 1.0), wb=0.0, tint=0.0))
    assert info["gain"] == 0.5 and abs(out.mean() - 100) < 3
    dark = np.full((150, 108, 3), 30, np.uint8)
    cfg = forced("low_light", gain=(0.45, 0.45), gamma=(1.5, 1.5), noise_base=(0.0, 0.0))
    out, info = C.low_light(dark, random.Random(3), cfg)
    x0, y0, x1, y1 = C.identity_region(150, 108)
    assert out[y0:y1, x0:x1].mean() / 255 >= cfg["identity_mean_min"] - 0.01
    assert info["gain"] > 0.45                                                    # the gain was raised to the floor


def test_rotation_tilts_about_the_card_and_returns_the_box_around_it():
    image = pattern(270, 194)
    inner = (43, 60, 108, 150)
    cfg = forced("rotation")
    out, box, degrees = C.rotation(image, inner, random.Random(9), cfg)
    again, box_again, degrees_again = C.rotation(image, inner, random.Random(9), cfg)
    assert out.shape == image.shape and np.array_equal(out, again) and box == box_again and degrees == degrees_again
    assert 15.0 <= abs(degrees) <= 45.0
    x, y, w, h = box
    assert w > 108 and x >= 0 and y >= 0 and x + w <= 194 and y + h <= 270       # wider than the upright card, inside the image
    signs = {C.rotation(image, inner, random.Random(seed), cfg)[2] > 0 for seed in range(40)}
    assert signs == {True, False}
    same, same_box, zero = C.rotation(image, inner, random.Random(9), {**cfg, "p": 0.0})
    assert same is image and same_box == inner and zero == 0.0
