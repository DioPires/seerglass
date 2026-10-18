# SPDX-License-Identifier: Apache-2.0
"""The physical chain: what a printed card on a table looks like through a camera.

``physical_chain`` works on an image with room around the card and the card's box in it
(``inner``), and returns both, because a sleeve and a tilt change the box:

1. print effects on the card itself: a color cast, a loss of contrast and, for a kind that
   carries dice, sometimes a die over the art (``cardvision.composites.photometric``);
2. a sleeve: a band around the card, clear and glossy or an opaque color;
3. glossy streaks across the sleeved card, and a foil finish with the probability its kind has;
4. a tilt of up to ``rot_deg`` degrees about the card's center: the new box is the upright
   box around the tilted card;
5. lamp glare: one or two soft bright blobs;
6. a slight defocus, then the camera's exposure and white balance over the whole image.

Which kinds carry dice and how often a kind is foil are the game's own facts; they come from
the catalog's ``kinds`` settings (``cardvision.composites.KindSettings``), never from a name
written here.

``motion_blur`` is applied later in a training chain, on the cut crop.

The chain draws from one NumPy generator seeded from the ``random.Random`` it is given, so a
seed fixes the result.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from cardvision.composites import photometric as P
from cardvision.composites import textures as T
from cardvision.composites.cards import KindSettings
from cardvision.composites.finish import add_foil

# The chain's settings. ``p_phys`` is the share of rows a training chain sends through it;
# ``strength`` is the range the photometric strength is drawn from (see
# ``cardvision.composites.photometric``); ``sleeve_px`` is the sleeve's border for a card
# 100 px wide; ``jpeg_p`` and ``jpeg_q`` replace the plain chain's JPEG settings for rows of a
# recipe that uses this chain.
PHYSICAL: dict[str, Any] = {
    "p_phys": 0.85,
    "strength": (0.0, 1.5),
    "p_sleeve": 0.35,
    "sleeve_px": (2.0, 6.0),
    "p_rot": 0.7,
    "rot_deg": 15.0,
    "p_glare": 0.3,
    "p_motion": 0.15,
    "motion_px": (3, 7),
    "jpeg_p": 0.7,
    "jpeg_q": (30, 90),
}


def sleeve(img: np.ndarray, inner, nrng: np.random.Generator, card_width: float, cfg: Mapping[str, Any] = PHYSICAL):
    """Draw a sleeve's border around the card: a band outside ``inner``, clipped to the image.
    One sleeve in four is clear and glossy (a light tint mixed in); the rest are an opaque
    color. Returns ``(image, inner)`` with ``inner`` grown to the sleeve's outline."""
    H, W = img.shape[:2]
    x, y, w, h = inner
    b = max(1, int(round(nrng.uniform(*cfg["sleeve_px"]) * card_width / 100.0)))
    x0, y0, x1, y1 = max(0, x - b), max(0, y - b), min(W, x + w + b), min(H, y + h + b)
    out = img.astype(np.float32).copy()
    band = np.zeros((H, W), bool)
    band[y0:y1, x0:x1] = True
    band[y:y + h, x:x + w] = False
    if nrng.random() < 0.25:
        a = nrng.uniform(50, 120) / 255.0
        out[band] = out[band] * (1 - a) + np.array([235, 240, 245], np.float32) * a
    else:
        col = T.rand_color(nrng, sat=(0, 1), val=(0.05, 0.9)) * 255.0
        out[band] = col
    return np.clip(out, 0, 255).astype(np.uint8), (x0, y0, x1 - x0, y1 - y0)


def rotate(img: np.ndarray, inner, deg: float):
    """Turn the image by ``deg`` degrees about the center of the card's box; what the turn
    uncovers is filled with the mean color of the image's border. Returns ``(image, inner)``
    where ``inner`` is the upright box around the turned card, clipped to the image."""
    H, W = img.shape[:2]
    x, y, w, h = inner
    cx, cy = x + w / 2.0, y + h / 2.0
    fill = tuple(int(v) for v in np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]).mean(0))
    im = Image.fromarray(img).rotate(deg, resample=Image.BILINEAR, center=(cx, cy), fillcolor=fill)
    th = math.radians(-deg)  # Pillow turns counter-clockwise for a positive angle; y grows downwards
    c, s = math.cos(th), math.sin(th)
    pts = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    xs = [cx + px * c - py * s for px, py in pts]
    ys = [cy + px * s + py * c for px, py in pts]
    x0, y0 = max(0, int(math.floor(min(xs)))), max(0, int(math.floor(min(ys))))
    x1, y1 = min(W, int(math.ceil(max(xs)))), min(H, int(math.ceil(max(ys))))
    return np.asarray(im), (x0, y0, max(2, x1 - x0), max(2, y1 - y0))


def lamp_glare(img: np.ndarray, inner, nrng: np.random.Generator) -> np.ndarray:
    """One or two soft, slightly warm bright blobs centered somewhere on the card."""
    H, W = img.shape[:2]
    x, y, w, h = inner
    out = img.astype(np.float32) / 255.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for _ in range(int(nrng.integers(1, 3))):
        gx, gy = x + nrng.uniform(0, w), y + nrng.uniform(0, h)
        sig = nrng.uniform(0.08, 0.30) * max(w, h)
        amp = nrng.uniform(0.15, 0.55)
        col = np.array([1.0, nrng.uniform(0.9, 1.0), nrng.uniform(0.75, 0.95)], np.float32)
        g = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * sig * sig))
        out = out + amp * g[..., None] * col
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def physical_chain(img: np.ndarray, inner, kind: str | None, rng: random.Random,
                   kinds: KindSettings | None = None, cfg: Mapping[str, Any] = PHYSICAL):
    """Run the chain on one image.

    Args:
        img: ``(H, W, 3)`` uint8 RGB with room around the card.
        inner: the card's box ``(x, y, width, height)`` in ``img``.
        kind: the card's kind in the catalog, or None.
        rng: the row's random stream; one draw seeds the chain's own generator.
        kinds: the game's kind settings; None for a game that states none.
        cfg: the settings (``PHYSICAL``).

    Returns ``(image, inner)``.
    """
    kinds = kinds if kinds is not None else KindSettings()
    nrng = np.random.default_rng(rng.getrandbits(64))
    S = P.setting(nrng.uniform(*cfg["strength"]), "random", foil=kinds.foil)
    x, y, w, h = [int(v) for v in inner]
    card = Image.fromarray(np.ascontiguousarray(img[y:y + h, x:x + w])).convert("RGBA")
    card, _ = P.print_effects(nrng, card, S, dice=kind in kinds.dice, die_fn=T.die_patch)
    img = img.copy()
    img[y:y + h, x:x + w] = np.asarray(card)[..., :3]
    sleeved = nrng.random() < cfg["p_sleeve"]
    if sleeved:
        img, (x, y, w, h) = sleeve(img, (x, y, w, h), nrng, w, cfg)
    reg = Image.fromarray(np.ascontiguousarray(img[y:y + h, x:x + w])).convert("RGBA")
    reg, _ = P.streaks(nrng, reg, sleeved, S)
    if nrng.random() < P.foil_probability(kind, S):
        reg = add_foil(nrng, reg)
    img[y:y + h, x:x + w] = np.asarray(reg)[..., :3]
    if nrng.random() < cfg["p_rot"]:
        img, (x, y, w, h) = rotate(img, (x, y, w, h), float(nrng.uniform(-cfg["rot_deg"], cfg["rot_deg"])))
    if nrng.random() < cfg["p_glare"]:
        img = lamp_glare(img, (x, y, w, h), nrng)
    sd = P.defocus_sigma(nrng, S)
    if sd > 0:
        img = np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(sd)))
    f = img.astype(np.float32) / 255.0
    f, _ = P.exposure(nrng, f, S)
    img = np.clip(np.round(f * 255.0), 0, 255).astype(np.uint8)
    return img, (x, y, w, h)


def motion_blur(crop: np.ndarray, rng: random.Random, cfg: Mapping[str, Any] = PHYSICAL) -> np.ndarray:
    """For ``p_motion`` of crops: a straight blur of 3 to 7 px in a random direction. The crop
    keeps its size."""
    if rng.random() >= cfg["p_motion"]:
        return crop
    k = rng.randint(*cfg["motion_px"])
    ang = rng.uniform(0, 180)
    im = Image.fromarray(crop)
    big = im.rotate(ang, resample=Image.BILINEAR, expand=True)
    a = np.asarray(big).astype(np.float32)
    ker = np.ones(k, np.float32) / k
    a = np.apply_along_axis(lambda r: np.convolve(r, ker, mode="same"), 1, a)
    big = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).rotate(-ang, resample=Image.BILINEAR, expand=True)
    bw, bh = big.size
    w, h = im.size
    return np.asarray(big.crop(((bw - w) // 2, (bh - h) // 2, (bw - w) // 2 + w, (bh - h) // 2 + h)))
