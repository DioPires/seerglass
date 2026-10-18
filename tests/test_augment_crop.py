# SPDX-License-Identifier: Apache-2.0
"""The crop and card-image augmentations: one seed gives one result, shapes are what the
docstrings say, and each does what it is named for."""

from __future__ import annotations

import random

import numpy as np
import pytest

from cardvision.augment import crop as A
from cardvision.augment.crop import CropAugment
from helpers import pattern

CROP = pattern(150, 108)
IMAGE = pattern(270, 194)  # a crop with room around the card
INNER = (43, 60, 108, 150)
PATCH = pattern(200, 200)[::-1].copy()

SAME_SHAPE = {
    "background_swap": lambda rng, cfg: A.background_swap(CROP, PATCH, rng, cfg),
    "cutout": lambda rng, cfg: A.cutout(CROP, rng, cfg),
    "jpeg": lambda rng, cfg: A.jpeg(CROP, rng, cfg),
    "photometric": lambda rng, cfg: A.photometric(CROP, rng, cfg),
    "center_mask": lambda rng, cfg: A.center_mask(CROP, cfg),
}
ANY_SHAPE = {
    "frame_jitter": lambda rng, cfg: A.frame_jitter(IMAGE, INNER, rng, cfg),
    "resolution_jitter": lambda rng, cfg: A.resolution_jitter(CROP, rng, cfg),
    "augment_crop": lambda rng, cfg: A.augment_crop(IMAGE, INNER, PATCH, rng, cfg),
    "augment_art": lambda rng, cfg: A.augment_art(CROP, rng, cfg),
}
ALWAYS = CropAugment(bg_swap_p=1.0, cutout_p=1.0, jpeg_p=1.0, center_mask=True)


@pytest.mark.parametrize("name", [*SAME_SHAPE, *ANY_SHAPE])
def test_one_seed_gives_one_result_and_another_seed_another(name):
    function = {**SAME_SHAPE, **ANY_SHAPE}[name]
    results = [function(random.Random(seed), ALWAYS) for seed in (11, 11, 12)]
    assert all(r.dtype == np.uint8 and r.ndim == 3 and r.shape[2] == 3 for r in results)
    assert np.array_equal(results[0], results[1])
    if name != "center_mask":  # the mask draws nothing
        assert results[0].shape != results[2].shape or not np.array_equal(results[0], results[2])


@pytest.mark.parametrize("name", list(SAME_SHAPE))
def test_these_keep_the_crop_s_shape_and_leave_the_input_alone(name):
    before = CROP.copy()
    for seed in range(8):
        assert SAME_SHAPE[name](random.Random(seed), ALWAYS).shape == CROP.shape
    assert np.array_equal(CROP, before)


def test_every_switch_turns_its_augmentation_off():
    off = CropAugment(frame_jitter=False, res_jitter=False, bg_swap=False, cutout=False, jpeg=False, photometric=False)
    rng = random.Random(1)
    x, y, w, h = INNER
    assert np.array_equal(A.frame_jitter(IMAGE, INNER, rng, off), IMAGE[y:y + h, x:x + w])
    for name in ("background_swap", "cutout", "jpeg", "photometric", "center_mask"):
        assert SAME_SHAPE[name](rng, off) is CROP
    assert A.resolution_jitter(CROP, rng, off) is CROP
    # With everything off the plain chain is the card's box and the art chain is the image.
    assert np.array_equal(A.augment_crop(IMAGE, INNER, PATCH, rng, off), IMAGE[y:y + h, x:x + w])
    assert np.array_equal(A.augment_art(CROP, rng, off), CROP)


def test_frame_jitter_stays_inside_the_image_and_near_the_card():
    cfg = CropAugment()
    for seed in range(200):
        cut = A.frame_jitter(IMAGE, INNER, random.Random(seed), cfg)
        h, w = cut.shape[:2]
        assert 2 <= w <= IMAGE.shape[1] and 2 <= h <= IMAGE.shape[0]
        # scale 0.8 to 1.25 and aspect 0.9 to 1.1 bound the size (the image may clip it)
        assert w <= 108 * 1.25 * 1.1 + 1 and h <= 150 * 1.25 / 0.9 + 1
        assert w >= 108 * 0.8 * 0.9 - 2 and h >= 150 * 0.8 / 1.1 - 2
    # A box at the image's corner still gives a crop of at least 2 px.
    assert min(A.frame_jitter(IMAGE, (190, 266, 30, 30), random.Random(0), cfg).shape[:2]) >= 2


def test_resolution_jitter_only_reduces_and_never_below_twelve_pixels():
    for seed in range(50):
        out = A.resolution_jitter(CROP, random.Random(seed), CropAugment())
        assert 12 <= out.shape[0] <= 150 and 12 <= out.shape[1] <= 108
        assert out.shape[1] >= int(108 * 0.35) and out.shape[0] >= int(150 * 0.35)
    tiny = A.resolution_jitter(pattern(14, 14), random.Random(0), CropAugment(res_range=(0.1, 0.1)))
    assert tiny.shape[:2] == (12, 12)


def test_background_swap_keeps_the_center_and_replaces_the_border():
    cfg = CropAugment(bg_swap_p=1.0)
    out = A.background_swap(CROP, PATCH, random.Random(4), cfg)
    kx, ky = int(108 * 0.1), int(150 * 0.1)
    assert np.array_equal(out[ky:150 - ky, kx:108 - kx], CROP[ky:150 - ky, kx:108 - kx])
    assert not np.array_equal(out[:ky], CROP[:ky])
    # A patch smaller than the crop is resized to cover it.
    small = A.background_swap(CROP, PATCH[:20, :20], random.Random(4), cfg)
    assert small.shape == CROP.shape
    # No patch, or a probability of zero: the crop itself.
    assert A.background_swap(CROP, None, random.Random(4), cfg) is CROP
    assert A.background_swap(CROP, PATCH, random.Random(4), CropAugment(bg_swap_p=0.0)) is CROP


def test_cutout_paints_one_flat_rectangle_within_the_stated_area():
    for seed in range(40):
        out = A.cutout(CROP, random.Random(seed), CropAugment(cutout_p=1.0))
        changed = np.any(out != CROP, axis=2)
        if not changed.any():
            continue
        ys, xs = np.where(changed)
        box = out[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        assert len(np.unique(box)) == 1 and 10 <= int(box.flat[0]) <= 60          # dark, flat
    light = A.cutout(CROP, random.Random(3), CropAugment(cutout_p=1.0, cutout_light_p=1.0))
    assert 190 <= int(light[np.any(light != CROP, axis=2)].flat[0]) <= 240


def test_jpeg_changes_pixels_a_little_and_photometric_stays_in_range():
    out = A.jpeg(CROP, random.Random(2), CropAugment(jpeg_p=1.0))
    assert not np.array_equal(out, CROP)
    smooth = np.full((64, 64, 3), 120, np.uint8)
    assert np.abs(A.jpeg(smooth, random.Random(2), CropAugment(jpeg_p=1.0)).astype(int) - 120).max() <= 3
    for seed in range(20):
        lit = A.photometric(smooth, random.Random(seed), CropAugment())
        assert 120 * 0.8 - 20 <= lit.mean() <= 120 * 1.2 + 20


def test_center_mask_grays_the_border_and_draws_nothing():
    cfg = CropAugment(center_mask=True)
    out = A.center_mask(CROP, cfg)
    kx, ky = int(108 * 0.14), int(150 * 0.14)
    assert np.all(out[:ky] == 128) and np.all(out[:, :kx] == 128)
    assert np.array_equal(out[ky:150 - ky, kx:108 - kx], CROP[ky:150 - ky, kx:108 - kx])
    assert np.array_equal(A.center_mask(out, cfg), out)                           # doing it twice changes nothing


def test_the_art_chain_keeps_the_image_s_size_up_to_the_resolution_jitter():
    cfg = CropAugment(res_jitter=False)
    for seed in range(10):
        assert A.augment_art(CROP, random.Random(seed), cfg).shape == CROP.shape


def test_a_card_image_is_placed_on_a_background_with_room_around_it():
    image, inner = A.on_surface(CROP, PATCH, 0.4, random.Random(5))
    x, y, w, h = inner
    assert (w, h) == (108, 150) and (x, y) == (43, 60)
    assert image.shape == (150 + 2 * 60, 108 + 2 * 43, 3)
    assert np.array_equal(image[y:y + h, x:x + w], CROP)
    again, _ = A.on_surface(CROP, PATCH, 0.4, random.Random(5))
    assert np.array_equal(image, again)
    # Without a patch, or with no margin, the image is returned as it is.
    for patch, margin in ((None, 0.4), (PATCH, 0.0)):
        bare, box = A.on_surface(CROP, patch, margin, random.Random(5))
        assert bare is CROP and box == (0, 0, 108, 150)
