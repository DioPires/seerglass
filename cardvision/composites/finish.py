# SPDX-License-Identifier: Apache-2.0
"""What is done to a card image before it is placed, and the canvas it is placed on.

``rounded_rgba``, ``add_sleeve``, ``add_foil`` and ``color_jitter_rgba`` turn a flat card image
into the object that lies on the table. ``Canvas`` holds the scene as it is built: the image,
and beside it a map of which object owns each pixel, from which every label's visible share is
counted.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageFilter

from cardvision.composites import textures as T
from cardvision.composites.geometry import apply_h


def rounded_rgba(rgb, radius_frac):
    """The image with rounded corners cut into its alpha; the radius is a fraction of the width."""
    im = rgb.convert("RGBA")
    a = T.rounded_mask(im.width, im.height, im.width * radius_frac)
    if "A" in rgb.getbands():
        a = Image.fromarray(np.minimum(np.asarray(a), np.asarray(rgb.getchannel("A"))))
    im.putalpha(a)
    return im


def add_sleeve(rng, card, border_px):
    """Put the card in a sleeve: a border of ``border_px`` card pixels on every side, clear and
    glossy for a quarter of sleeves and an opaque random color for the rest.
    Returns ``(RGBA image, "clear" or "opaque")``."""
    b = max(1, int(round(border_px)))
    w, h = card.width + 2 * b, card.height + 2 * b
    kind = rng.random()
    if kind < 0.25:
        col = (235, 240, 245, int(rng.uniform(50, 120)))
    else:
        col = tuple(int(c * 255) for c in T.rand_color(rng, sat=(0, 1), val=(0.05, 0.9))) + (255,)
    sl = Image.new("RGBA", (w, h), col)
    sl.putalpha(Image.fromarray(np.minimum(np.asarray(T.rounded_mask(w, h, w * 0.05)), col[3]).astype(np.uint8)))
    # The card's alpha is below 255 only at its rounded corners, where the sleeve shows through.
    sl.alpha_composite(card, (b, b))
    return sl, ("clear" if kind < 0.25 else "opaque")


def add_foil(rng, card):
    """A foil finish: a faint rainbow that runs across the card in a random direction, mixed in
    at 4 to 14 %, and one bright band of 8 to 30 %."""
    w, h = card.size
    g = T.gradient(w, h, rng.uniform(0, 6.28))
    hue = (g * rng.uniform(1, 3) + rng.random()) % 1.0
    rain = np.stack([np.clip(np.abs(((hue + o) % 1.0) * 6 - 3) - 1, 0, 1) for o in (0, 2 / 3, 1 / 3)], -1)
    band = np.exp(-((g - rng.uniform(0.2, 0.8)) / rng.uniform(0.04, 0.15)) ** 2)[..., None]
    a = np.asarray(card, np.float32)
    rgb = a[..., :3] / 255
    k = rng.uniform(0.04, 0.14)
    rgb = np.clip(rgb * (1 - k) + rain * k + band * rng.uniform(0.08, 0.3), 0, 1)
    a[..., :3] = rgb * 255
    return Image.fromarray(a.astype(np.uint8), "RGBA")


def color_jitter_rgba(rng, im, amt=0.1):
    """A small random gain per color channel and overall."""
    a = np.asarray(im, np.float32)
    g = rng.uniform(1 - amt, 1 + amt, 3) * rng.uniform(0.85, 1.12)
    a[..., :3] = np.clip(a[..., :3] * g, 0, 255)
    return Image.fromarray(a.astype(np.uint8), "RGBA")


class Canvas:
    """The scene under construction.

    ``img`` is float32 (H, W, 3) in [0, 1]. ``ids`` is int16 (H, W): the id of the object whose
    pixel is on top, 0 for the background.
    """

    def __init__(self, bg):
        self.img = bg.astype(np.float32)
        self.H, self.W = bg.shape[:2]
        self.ids = np.zeros((self.H, self.W), np.int16)

    def paste_warped(self, rng, src, M, obj_id, shadow=True, feather=None, blur=0.0):
        """Draw ``src`` (RGBA) through ``M``, the 3 x 3 map from its pixel coordinates to image
        pixels, with a feathered edge and, if asked, a soft drop shadow."""
        w, h = src.size
        cs = apply_h(M, [(0, 0), (w, 0), (w, h), (0, h)])
        pad = 8
        x0, y0 = int(math.floor(cs[:, 0].min())) - pad, int(math.floor(cs[:, 1].min())) - pad
        x1, y1 = int(math.ceil(cs[:, 0].max())) + pad, int(math.ceil(cs[:, 1].max())) + pad
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(self.W, x1), min(self.H, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        Tm = np.array([[1, 0, cx0], [0, 1, cy0], [0, 0, 1.0]])
        Minv = np.linalg.inv(M @ np.eye(3)) @ Tm
        Minv /= Minv[2, 2]
        patch = src.transform((cx1 - cx0, cy1 - cy0), Image.PERSPECTIVE, tuple(Minv.ravel()[:8]), Image.BICUBIC)
        if blur > 0:
            patch = patch.filter(ImageFilter.GaussianBlur(blur))
        arr = np.asarray(patch, np.float32) / 255
        a_raw = arr[..., 3]
        fr = rng.uniform(0.3, 1.1) if feather is None else feather
        if fr > 0:
            a = np.asarray(patch.getchannel("A").filter(ImageFilter.GaussianBlur(fr)), np.float32) / 255
        else:
            a = a_raw
        region = self.img[cy0:cy1, cx0:cx1]
        if shadow:
            dx, dy = int(rng.integers(-3, 5)), int(rng.integers(0, 6))
            sh = patch.getchannel("A").filter(ImageFilter.GaussianBlur(rng.uniform(1.5, 4)))
            sh = np.asarray(sh, np.float32) / 255
            sh = np.roll(np.roll(sh, dy, 0), dx, 1) * rng.uniform(0.15, 0.45)
            region *= (1 - sh)[..., None]
        region[:] = region * (1 - a[..., None]) + arr[..., :3] * a[..., None]
        self.ids[cy0:cy1, cx0:cx1][a_raw > 0.5] = obj_id

    def paste_flat(self, src, x, y, obj_id, feather=0.8):
        """Draw ``src`` (RGBA) unwarped with its top-left corner at integer ``(x, y)``."""
        w, h = src.size
        cx0, cy0, cx1, cy1 = max(0, x), max(0, y), min(self.W, x + w), min(self.H, y + h)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        sub = src.crop((cx0 - x, cy0 - y, cx1 - x, cy1 - y))
        arr = np.asarray(sub, np.float32) / 255
        a = np.asarray(sub.getchannel("A").filter(ImageFilter.GaussianBlur(feather)), np.float32) / 255
        region = self.img[cy0:cy1, cx0:cx1]
        region[:] = region * (1 - a[..., None]) + arr[..., :3] * a[..., None]
        self.ids[cy0:cy1, cx0:cx1][arr[..., 3] > 0.5] = obj_id
