# SPDX-License-Identifier: Apache-2.0
"""The synthesized views a bank row is averaged over."""

from __future__ import annotations

import numpy as np
import pytest

from cardvision.bank import views as V
from cardvision.core.images import load_rgb
from helpers import pattern

ART = pattern(336, 240)  # a portrait card: 240 wide, 336 high


def test_a_card_has_sixteen_views_and_the_first_is_the_image_itself():
    views = V.card_views(ART, "TOY-001")
    assert V.VIEWS_PER_CARD == 16 and len(views) == 16
    assert views[0] is ART
    assert all(view.dtype == np.uint8 and view.ndim == 3 and view.shape[2] == 3 for view in views)
    assert len(V.card_views(ART, "TOY-001", count=4)) == 4
    assert np.array_equal(V.card_views(ART, "TOY-001", count=4)[3], views[3])


def test_views_are_deterministic():
    again = V.card_views(ART.copy(), "TOY-001")
    for first, second in zip(V.card_views(ART, "TOY-001"), again):
        assert first.shape == second.shape and np.array_equal(first, second)


def test_every_view_differs_from_the_others_and_from_the_image():
    views = V.card_views(ART, "TOY-001")
    assert len({(view.shape, view.tobytes()) for view in views}) == 16


def test_a_view_depends_on_the_card_id():
    one = V.synthesize_view_with_parameters(ART, "TOY-001", 1)
    other = V.synthesize_view_with_parameters(ART, "TOY-002", 1)
    assert one[1] != other[1]
    assert one[0].shape != other[0].shape or not np.array_equal(one[0], other[0])


def test_the_seed_is_the_card_id_and_the_view_number():
    # Pinned: these draws are what makes a published bank reproducible.
    assert V.VIEW_SEED == 1337
    assert V.view_rng("TOY-001", 1).integers(0, 2**32, size=3).tolist() == [2818793022, 721023431, 3592586693]
    assert V.view_rng("TOY-001", 1).integers(0, 2**32) != V.view_rng("TOY-001", 2).integers(0, 2**32)
    assert V.view_rng("TOY-001", 1).integers(0, 2**32) != V.view_rng("TOY-002", 1).integers(0, 2**32)


def test_the_amounts_drawn_for_a_view_are_pinned():
    _, small = V.synthesize_view_with_parameters(ART, "TOY-001", 1)
    assert small == {
        "jitter": [-0.0266, -0.0269, 0.0321, -0.0009], "gray": 60, "short": 108.3, "enlarged": False,
        "size": [108, 157], "gamma": 0.977, "gains": [1.034, 0.974, 0.956], "saturation": 1.118,
        "blur": 0.896, "jpegQuality": 72,
    }
    _, large = V.synthesize_view_with_parameters(ART, "TOY-001", 3)
    assert large == {
        "jitter": [0.0231, 0.0239, -0.0234, -0.0144], "gray": 29, "short": 296.0, "enlarged": True,
        "size": [296, 411], "gamma": 1.103, "gains": [1.004, 1.038, 0.949], "saturation": 0.956,
        "blur": 0.407, "jpegQuality": 51,
    }


@pytest.mark.parametrize("card_id", ["TOY-001", "TOY-017", "a card with spaces"])
def test_the_amounts_stay_inside_their_ranges(card_id):
    for view in range(1, 40):
        image, drawn = V.synthesize_view_with_parameters(ART, card_id, view)
        assert drawn["enlarged"] == (view % 4 == 3)
        low, high = (200, 340) if drawn["enlarged"] else (85, 125)
        assert low <= drawn["short"] <= high
        assert all(-0.04 <= amount <= 0.04 for amount in drawn["jitter"])
        assert 20 <= drawn["gray"] <= 90 and 35 <= drawn["jpegQuality"] <= 75
        assert 0.85 <= drawn["gamma"] <= 1.15 and 0.85 <= drawn["saturation"] <= 1.15
        assert all(0.94 <= gain <= 1.06 for gain in drawn["gains"]) and 0.3 <= drawn["blur"] <= 0.9
        assert list(image.shape[:2]) == drawn["size"][::-1]
        assert abs(min(image.shape[:2]) - drawn["short"]) <= 1


def test_the_image_is_not_modified():
    art = ART.copy()
    V.card_views(art, "TOY-001")
    assert np.array_equal(art, ART)


def test_a_landscape_image_stays_landscape(toy):
    _, art_dir = toy
    site = load_rgb(art_dir / "TOY-017.png")
    assert site.shape[1] > site.shape[0]
    assert all(view.shape[1] > view.shape[0] for view in V.card_views(site, "TOY-017"))


def test_a_tiny_image_is_never_shrunk_below_eight_pixels():
    sliver = pattern(400, 20)
    for view in range(1, 16):
        assert min(V.synthesize_view(sliver, "sliver", view).shape[:2]) >= 8


def test_bad_arguments_are_refused():
    with pytest.raises(ValueError, match="view 0 is the card image itself"):
        V.synthesize_view(ART, "TOY-001", 0)
    with pytest.raises(ValueError, match="uint8 RGB"):
        V.synthesize_view(ART.astype(np.float32), "TOY-001", 1)
    with pytest.raises(ValueError, match="uint8 RGB"):
        V.synthesize_view(ART[:, :, 0], "TOY-001", 1)
    with pytest.raises(ValueError, match="at least one view"):
        V.card_views(ART, "TOY-001", count=0)
