# SPDX-License-Identifier: Apache-2.0
"""Photometric effects: what printing, sleeves, lamps and a camera do to a card's colors.

Every effect is a distribution: a fixed fraction of cards gets it, at a strength drawn per
card. One scalar, ``strength``, scales all the strengths together; the fractions do not move
with it. Each function takes its own random stream from the caller and never the scene's, so
switching the effects on or off, or changing their strength, leaves a scene's layout, cards,
camera, degradation and labels exactly as they were.

The effects (per printed card, face or back, unless stated):

``cast``
    ``p_cast`` of cards: a gain of 1 + strength x U(-0.10, 0.10) per color channel, saturation
    multiplied by 1 - strength x U(0, 0.25), and a lift towards white of strength x U(0, 0.04).
    The card looks tinted, washed out and slightly hazy.
``contrast``
    ``p_contrast`` of cards: contrast about the card's mean color multiplied by
    1 - strength x U(0.08, 0.30), never below 0.3. The card looks flatter.
``dice``
    Cards of a kind that carries dice (the catalog's ``kinds.dice``), ``p_dice_art`` of them:
    one die, 30 to 40 % of the card's width across, painted over the upper part of the art. It
    hides part of the picture, as a counter resting on the card does.
``streaks``
    ``p_streak_sleeved`` of sleeved cards and ``p_streak_bare`` of the others: one to three
    straight glossy bands, each adding strength x U(0.10, 0.30) of white along a line across
    the card, as a lamp reflected in plastic does.
``foil``
    A face is foil with the probability its kind has in the catalog's
    ``kinds.foilProbability``, or ``foil_default`` when its kind has none. A foil face gets a
    faint rainbow sheen and one bright band (``finish.add_foil``); strength does not scale it.
``defocus``
    ``p_defocus`` of cards: an extra Gaussian blur of sigma strength x U(0.5, 1.4) px on the
    card as it lies in the image.
``exposure``
    Every scene, applied to the whole table image before any broadcast graphics and before
    the degradation: a gain of 2 ** (strength x U(-0.35, 0.35)) and a white balance of
    1 + strength x U(-0.06, 0.06) per channel.

``DEFAULT`` is the setting the scene generator uses. Its strength, 1.0, was fitted to real
crops: strengths were tried in a fixed rising order and the first one kept at which a
recognizer was right about as often on generated card crops as on real crops of the same size.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

# The fraction of cards that gets each effect, and the foil probability of a face whose kind
# has none in the catalog.
BASE = {
    "p_cast": 0.6,
    "p_contrast": 0.6,
    "p_dice_art": 0.5,
    "p_streak_sleeved": 0.6,
    "p_streak_bare": 0.15,
    "p_defocus": 0.25,
    "foil_default": 0.15,
}


def setting(strength, name=None, foil=None):
    """A photometric setting: the fixed fractions, one strength, and the per-kind foil
    probabilities (``foil``, kind name to probability; kinds not listed use ``foil_default``)."""
    s = dict(BASE)
    s["strength"] = float(strength)
    s["name"] = name or f"strength-{strength:g}"
    s["foil"] = dict(foil or {})
    return s


DEFAULT = setting(1.0, name="default")


def print_effects(prng, card, S, dice=False, die_fn=None):
    """Color cast, loss of contrast and, for a card that carries dice, a die over the art.

    ``card`` is an RGBA Pillow image with no sleeve yet. ``die_fn(prng, size)`` draws the die
    and returns ``(RGBA image, name)``; with ``dice`` false or no ``die_fn`` no die is drawn.
    Returns ``(card, log)`` where ``log`` records what was applied.
    """
    k = S["strength"]
    log = {}
    a = np.asarray(card, np.float32).copy()
    rgb = a[..., :3]
    if prng.random() < S["p_cast"]:
        g = 1 + k * prng.uniform(-0.10, 0.10, 3)
        sat = 1 - k * prng.uniform(0, 0.25)
        lift = k * prng.uniform(0, 0.04)
        rgb *= g
        lum = rgb.mean(2, keepdims=True)
        rgb[:] = lum + (rgb - lum) * max(0.0, sat)
        rgb[:] = rgb * (1 - lift) + 255 * lift
        log["cast"] = [round(float(x), 3) for x in g] + [round(float(sat), 3), round(float(lift), 3)]
    if prng.random() < S["p_contrast"]:
        c = max(0.3, 1 - k * prng.uniform(0.08, 0.30))
        m = a[..., 3] > 0
        mean = rgb[m].mean(0) if m.any() else rgb.reshape(-1, 3).mean(0)
        rgb[:] = mean + (rgb - mean) * c
        log["contrast"] = round(float(c), 3)
    a[..., :3] = np.clip(rgb, 0, 255)
    out = Image.fromarray(a.astype(np.uint8), "RGBA")
    if dice and die_fn is not None and prng.random() < S["p_dice_art"]:
        w, h = out.size
        s = prng.uniform(0.30, 0.40) * w
        die, sub = die_fn(prng, max(6, int(s)))
        cx, cy = w * (0.5 + prng.uniform(-0.1, 0.1)), h * prng.uniform(0.30, 0.45)
        out = out.copy()
        out.alpha_composite(die, (int(round(cx - die.width / 2)), int(round(cy - die.height / 2))))
        log["dice"] = sub
    return out, log


def streaks(prng, card, sleeved, S):
    """Glossy bands across a card. Returns ``(card, number of bands or None)``."""
    p = S["p_streak_sleeved"] if sleeved else S["p_streak_bare"]
    if prng.random() >= p:
        return card, None
    k = S["strength"]
    a = np.asarray(card, np.float32).copy()
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    add = np.zeros((h, w), np.float32)
    n = int(prng.integers(1, 4))
    for _ in range(n):
        th = math.radians(prng.uniform(0, 180))
        d = xx * math.cos(th) + yy * math.sin(th)
        c = prng.uniform(d.min(), d.max())
        wid = max(1.0, prng.uniform(0.04, 0.15) * w)
        add += k * prng.uniform(0.10, 0.30) * np.exp(-((d - c) / wid) ** 2)
    a[..., :3] = np.clip(a[..., :3] + 255 * add[..., None], 0, 255)
    return Image.fromarray(a.astype(np.uint8), "RGBA"), n


def foil_probability(kind, S):
    """The probability that a face of this kind is foil."""
    return S["foil"].get(kind, S["foil_default"])


def defocus_sigma(prng, S):
    """Extra blur for one card, in px (0.0 for the cards that get none)."""
    if prng.random() < S["p_defocus"]:
        return S["strength"] * prng.uniform(0.5, 1.4)
    return 0.0


def exposure(prng, img, S):
    """Camera gain and white balance over a float32 (H, W, 3) image in [0, 1], changed in
    place. Returns ``(image, log)``."""
    k = S["strength"]
    gain = 2 ** (k * prng.uniform(-0.35, 0.35))
    wb = 1 + k * prng.uniform(-0.06, 0.06, 3)
    img *= (gain * wb).astype(np.float32)
    return np.clip(img, 0, 1), {"gain": round(float(gain), 3), "wb": [round(float(x), 3) for x in wb]}
