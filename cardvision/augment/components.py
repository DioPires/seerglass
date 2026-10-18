# SPDX-License-Identifier: Apache-2.0
"""Targeted augmentation components: occluders, glare, low light and wider rotation.

Each one is a named, separately switchable component of a training recipe. Three rules hold for
all of them:

* A component draws from its own random stream, made by ``stream_for`` from the training seed,
  the epoch, the row's index and the component's name. It never draws from the stream of the
  base chain, so switching occluders, glare or low light on or off changes nothing else about
  a row. Rotation changes the size of the crop that is cut, and a few later draws of the base
  chain depend on that size, so with rotation the rest of the row may differ too.
* A component knows where a card's identity is: the name and the art (``identity_region``).
  No occluder may cover, and no glare may burn out, more than a stated share of it, and low
  light may not darken it below a stated level. A crop that cannot be told apart is not a
  useful training example.
* Occluders are applied to positives and negatives alike, so that an occluder is never a clue
  to the class. The one exception is stated at ``occluders``.

``DEFAULTS`` holds each component's settings. The functions return ``(image, info)``, where
``info`` is None when the component's own probability left the image alone and otherwise
records what was drawn.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from cardvision.augment.physical import rotate

DEFAULTS: dict[str, dict[str, Any]] = {
    # Something lying over part of the card: another card, fingers, a counter or die, or the
    # edge of a sleeve. "area_max" is the largest share of the crop all occluders of one draw
    # may cover together; "identity_cover_max" the largest share of the identity region.
    "occluders": {"p": 0.35, "types": {"card": 0.40, "fingers": 0.30, "disc": 0.20, "sleeve_edge": 0.10},
                  "area_max": 0.35, "identity_cover_max": 0.40, "tries": 6},
    # Light on a glossy or foil surface: a streak, a bloom, a rainbow foil sheen or a washed-out
    # patch. "identity_saturated_max" is the largest share of identity-region pixels that may
    # end up at 250 or more in every channel; a stronger draw is weakened until it fits.
    "glare": {"p": 0.30, "types": {"streak": 0.35, "bloom": 0.25, "foil": 0.25, "washout": 0.15},
              "streak_amp": (0.25, 0.70), "bloom_amp": (0.30, 0.80), "foil_mix": (0.05, 0.15),
              "washout": (0.25, 0.55), "identity_saturated_max": 0.25, "tries": 4},
    # A dim scene: gain, gamma, white balance (warm or cool), a green or magenta tint, and the
    # sensor noise that comes with a dark picture. "identity_mean_min" is the floor on the
    # identity region's mean level (0 to 1) after gain and gamma.
    "low_light": {"p": 0.40, "gain": (0.45, 1.0), "gamma": (0.8, 1.5), "wb": 0.18, "tint": 0.05,
                  "noise_base": (1.0, 4.0), "chroma_share": 0.5, "identity_mean_min": 0.10},
    # A tilt beyond the physical chain's: applied to the image with room around the card,
    # before the crop is framed, so the crop is the upright box around the tilted card, as a
    # detector's box is. Quarter turns are not this component's business.
    "rotation": {"p": 0.30, "abs_deg": (15.0, 45.0)},
}

# One arbitrary constant per stream. They keep the streams apart; their values mean nothing,
# but they are fixed, so that a seed reproduces a run.
SALT = {"occluders": 0x0C1, "glare": 0x61A, "low_light": 0x10F, "rotation": 0x207, "surface": 0x5F3}


def row_seed(seed: int, epoch: int, index: int) -> int:
    """The seed of a row's base stream: one stream per training seed, epoch and row index."""
    return (seed * 1_000_003 + epoch * 10_007 + index) & 0x7FFFFFFF


def stream_for(seed: int, epoch: int, index: int, name: str) -> random.Random:
    """A component's own stream for one row, fixed by the training seed, the epoch, the row's
    index and the component's name."""
    return random.Random(((seed * 1_000_003 + epoch * 10_007 + index) * 1_315_423_911 + SALT[name]) & 0x7FFFFFFFFFFF)


def identity_region(h: int, w: int) -> tuple[int, int, int, int]:
    """The name and art of an upright card crop, as ``(x0, y0, x1, y1)``.

    On a portrait card they fill the upper part; on a card printed sideways (a crop clearly
    wider than tall) the art spans the card and the name sits low in the middle.
    """
    if w > h * 1.15:
        return int(0.10 * w), int(0.08 * h), int(0.90 * w), int(0.80 * h)
    return int(0.12 * w), int(0.07 * h), int(0.88 * w), int(0.62 * h)


def _pick(rng: random.Random, weights: Mapping[str, float]) -> str:
    r, acc = rng.random() * sum(weights.values()), 0.0
    for k, v in weights.items():
        acc += v
        if r <= acc:
            return k
    return k


# ------------------------------------------------------------------------------- rotation
def rotation(img: np.ndarray, inner, rng: random.Random, cfg: Mapping[str, Any] = DEFAULTS["rotation"]):
    """Tilt the image about the card's center by an angle whose size is drawn from
    ``abs_deg``, either way. Returns ``(image, inner, degrees)``; 0.0 degrees when the
    component's probability left the image alone."""
    if rng.random() >= cfg["p"]:
        return img, inner, 0.0
    deg = rng.uniform(*cfg["abs_deg"]) * (1 if rng.random() < 0.5 else -1)
    img, inner = rotate(img, inner, deg)
    return img, inner, deg


# ---------------------------------------------------------------------------------- glare
def _grid(h: int, w: int):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return yy, xx


def _glare_layer(h: int, w: int, kind: str, rng: random.Random, cfg: Mapping[str, Any]):
    """The layer of one glare type, float ``(h, w, 3)`` or ``(h, w, 1)`` in [0, 1], and how it
    is mixed in: ``"add"``, ``"wash"`` or ``("foil", share)``."""
    yy, xx = _grid(h, w)
    rng.getrandbits(32)  # a draw that nothing reads: removing it would change every later draw of a seed
    if kind == "streak":  # a long thin bright band at an angle
        th = rng.uniform(0, math.pi)
        c, s = math.cos(th), math.sin(th)
        d = (xx - rng.uniform(0.2, 0.8) * w) * s - (yy - rng.uniform(0.2, 0.8) * h) * c
        width = rng.uniform(0.02, 0.10) * max(h, w)
        g = np.exp(-(d / width) ** 2)
        along = (xx - w / 2) * c + (yy - h / 2) * s
        g *= np.exp(-(along / (rng.uniform(0.4, 1.2) * max(h, w))) ** 2)
        col = np.array([1.0, rng.uniform(0.94, 1.0), rng.uniform(0.85, 1.0)], np.float32)
        return g[..., None] * col * rng.uniform(*cfg["streak_amp"]), "add"
    if kind == "bloom":  # a lamp's reflection: a bright core with a soft halo
        gx, gy = rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h
        sig = rng.uniform(0.06, 0.22) * max(h, w)
        dist2 = (xx - gx) ** 2 + (yy - gy) ** 2
        g = np.exp(-dist2 / (2 * sig * sig)) + 0.35 * np.exp(-dist2 / (2 * (2.5 * sig) ** 2))
        return g[..., None] * np.ones(3, np.float32) * rng.uniform(*cfg["bloom_amp"]), "add"
    if kind == "foil":  # a rainbow that runs across the card
        th = rng.uniform(0, 2 * math.pi)
        g = ((xx * math.cos(th) + yy * math.sin(th)) / max(h, w)) % 1.0
        hue = (g * rng.uniform(1, 3) + rng.random()) % 1.0
        rain = np.stack([np.clip(np.abs(((hue + o) % 1.0) * 6 - 3) - 1, 0, 1) for o in (0, 2 / 3, 1 / 3)], -1)
        return rain.astype(np.float32), ("foil", rng.uniform(*cfg["foil_mix"]))
    # washout: contrast lost towards a bright haze, as on a glossy sleeve under a lamp
    gx, gy = rng.uniform(0.15, 0.85) * w, rng.uniform(0.15, 0.85) * h
    sig = rng.uniform(0.15, 0.35) * max(h, w)
    g = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * sig * sig)) * rng.uniform(*cfg["washout"])
    return g[..., None].astype(np.float32), "wash"


def glare(crop: np.ndarray, rng: random.Random, cfg: Mapping[str, Any] = DEFAULTS["glare"]):
    """One glare of a random type over the crop. If it burns out too much of the identity
    region, it is weakened (by 0.6 each time, up to ``tries`` times)."""
    if rng.random() >= cfg["p"]:
        return crop, None
    h, w = crop.shape[:2]
    kind = _pick(rng, cfg["types"])
    layer, mode = _glare_layer(h, w, kind, rng, cfg)
    x0, y0, x1, y1 = identity_region(h, w)
    base = crop.astype(np.float32) / 255.0
    scale = 1.0
    for _ in range(cfg["tries"]):
        if mode == "add":
            out = base + scale * layer
        elif mode == "wash":
            a = scale * layer
            out = base * (1 - a) + (0.5 * base.mean((0, 1), keepdims=True) + 0.5) * a
        else:
            k = scale * mode[1]
            out = base * (1 - k) + layer * k
        out = np.clip(out, 0, 1)
        sat = (out[y0:y1, x0:x1] >= 250 / 255).all(-1).mean() if y1 > y0 and x1 > x0 else 0.0
        if sat <= cfg["identity_saturated_max"]:
            break
        scale *= 0.6
    return np.round(out * 255).astype(np.uint8), {"kind": kind, "scale": round(scale, 3)}


# ------------------------------------------------------------------------------ occluders
def _skin(rng: random.Random) -> np.ndarray:
    tone = rng.uniform(0, 1)  # from light to dark skin, under warm or cool light
    base = np.array([235, 190, 160]) * (1 - tone) + np.array([95, 60, 40]) * tone
    return np.clip(base * rng.uniform(0.6, 1.1), 0, 255)


def _shape_mask(h: int, w: int, kind: str, rng: random.Random, other_art: np.ndarray | None = None):
    """One occluder: ``(mask, fill, shadow)``. ``mask`` is float ``(h, w)`` in [0, 1],
    ``fill`` the occluder's colors as float ``(h, w, 3)`` on the 0..255 scale."""
    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    fill = np.zeros((h, w, 3), np.float32)
    if kind == "card":  # another card sliding in from one edge
        cw, ch = int(w * rng.uniform(0.8, 1.1)), int(h * rng.uniform(0.8, 1.1))
        side = rng.randrange(4)
        depth = rng.uniform(0.12, 0.40)
        if side == 0:
            ox, oy = int(rng.uniform(-0.3, 0.3) * w), int(-ch + depth * h)
        elif side == 1:
            ox, oy = int(rng.uniform(-0.3, 0.3) * w), int(h - depth * h)
        elif side == 2:
            ox, oy = int(-cw + depth * w), int(rng.uniform(-0.3, 0.3) * h)
        else:
            ox, oy = int(w - depth * w), int(rng.uniform(-0.3, 0.3) * h)
        ang = rng.uniform(-20, 20)
        if other_art is not None:
            oa = Image.fromarray(other_art).resize((max(2, cw), max(2, ch)), Image.BILINEAR)
        else:
            oa = Image.new("RGB", (max(2, cw), max(2, ch)), tuple(int(v) for v in np.array([40, 40, 50]) * rng.uniform(0.5, 2)))
        om = Image.new("L", oa.size, 255)
        oa, om = oa.rotate(ang, resample=Image.BILINEAR, expand=True), om.rotate(ang, resample=Image.BILINEAR, expand=True)
        canvas = Image.new("RGB", (w, h))
        canvas.paste(oa, (ox, oy))
        m.paste(om, (ox, oy))
        fill = np.asarray(canvas).astype(np.float32)
        # a thin shadow under the covering card's edge
        sh = np.asarray(m.filter(ImageFilter.GaussianBlur(max(1, w * 0.02)))).astype(np.float32) / 255
        return np.clip(np.asarray(m).astype(np.float32) / 255, 0, 1), fill, sh
    if kind == "fingers":  # one to four fingers reaching in from one edge
        side = rng.randrange(4)
        n = rng.randint(1, 4)
        fw = rng.uniform(0.10, 0.17) * min(h, w) * (1.6 if w > h else 1)
        col = _skin(rng)
        start = rng.uniform(0.15, 0.85)
        for k in range(n):
            t = start + (k - (n - 1) / 2) * fw * 1.05 / (w if side in (0, 1) else h)
            L = rng.uniform(0.15, 0.40) * (h if side in (0, 1) else w)
            if side == 0:
                box = (t * w - fw / 2, -fw, t * w + fw / 2, L)
            elif side == 1:
                box = (t * w - fw / 2, h - L, t * w + fw / 2, h + fw)
            elif side == 2:
                box = (-fw, t * h - fw / 2, L, t * h + fw / 2)
            else:
                box = (w - L, t * h - fw / 2, w + fw, t * h + fw / 2)
            d.rounded_rectangle(box, radius=fw / 2, fill=255)
        mk = np.asarray(m.filter(ImageFilter.GaussianBlur(max(0.6, fw * 0.08)))).astype(np.float32) / 255
        soft = np.asarray(m.filter(ImageFilter.GaussianBlur(max(1.0, fw * 0.35)))).astype(np.float32) / 255
        shade = 0.70 + 0.35 * soft  # rounded: darker towards a finger's edges
        fill = (col[None, None, :] * shade[..., None]).astype(np.float32)
        return mk, fill, mk
    if kind == "disc":  # counters or dice: one to three discs, lit from above
        n = rng.randint(1, 3)
        col = np.array([rng.randint(0, 255) for _ in range(3)], np.float32)
        for _ in range(n):
            r = rng.uniform(0.07, 0.15) * w
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=255)
        mk = np.asarray(m).astype(np.float32) / 255
        yy, xx = _grid(h, w)
        fill = np.ones((h, w, 3), np.float32) * col
        fill = fill * (0.8 + 0.3 * (1 - yy / max(h, 1)))[..., None]
        return mk, np.clip(fill, 0, 255), mk
    # sleeve_edge: a colored or glossy band along one or two edges
    col = np.array([rng.randint(0, 255) for _ in range(3)], np.float32)
    for side in rng.sample(range(4), rng.randint(1, 2)):
        b = rng.uniform(0.04, 0.10) * min(h, w)
        box = [(0, 0, w, b), (0, h - b, w, h), (0, 0, b, h), (w - b, 0, w, h)][side]
        d.rectangle(box, fill=int(255 * rng.uniform(0.55, 1.0)))
    mk = np.asarray(m).astype(np.float32) / 255
    return mk, np.ones((h, w, 3), np.float32) * col, mk


def occluders(crop: np.ndarray, rng: random.Random, cfg: Mapping[str, Any] = DEFAULTS["occluders"],
              other_art: np.ndarray | None = None, negative: bool = False):
    """Lay one occluder of a random type over the crop.

    ``other_art`` is another card's image for the ``card`` type (a dark rectangle stands in
    without one). A draw that covers too much of the crop or of the identity region, or almost
    nothing, is drawn again, up to ``tries`` times; after that the crop is returned as it was
    and ``info`` says ``rejected``.

    On a negative row (``negative=True``) the ``card`` type is never used: a card face pasted
    onto a negative would teach the model to answer "no card" to card faces.
    """
    if rng.random() >= cfg["p"]:
        return crop, None
    h, w = crop.shape[:2]
    x0, y0, x1, y1 = identity_region(h, w)
    types = {k: v for k, v in cfg["types"].items() if not (negative and k == "card")}
    kind = _pick(rng, types)
    for t in range(cfg["tries"]):
        mk, fill, sh = _shape_mask(h, w, kind, rng, other_art if kind == "card" else None)
        area = float((mk > 0.5).mean())
        idc = float((mk[y0:y1, x0:x1] > 0.5).mean()) if y1 > y0 and x1 > x0 else 0.0
        if area <= cfg["area_max"] and idc <= cfg["identity_cover_max"] and area > 0.01:
            out = crop.astype(np.float32)
            if kind == "card":  # the drop shadow first, then the covering card
                out = out * (1 - 0.35 * np.clip(sh - mk, 0, 1))[..., None]
            out = out * (1 - mk[..., None]) + fill * mk[..., None]
            return np.clip(out, 0, 255).astype(np.uint8), {"kind": kind, "area": round(area, 3),
                                                           "identityCover": round(idc, 3), "try": t}
    return crop, {"kind": kind, "rejected": True}


# ------------------------------------------------------------------------------ low light
def low_light(crop: np.ndarray, rng: random.Random, cfg: Mapping[str, Any] = DEFAULTS["low_light"]):
    """Darken the crop as a dim room does: gamma, gain and white balance, then noise that is
    stronger the lower the gain. The identity region's mean level is not taken below
    ``identity_mean_min``."""
    if rng.random() >= cfg["p"]:
        return crop, None
    f = crop.astype(np.float32) / 255.0
    gain = rng.uniform(*cfg["gain"])
    gamma = rng.uniform(*cfg["gamma"])
    t = rng.uniform(-cfg["wb"], cfg["wb"])  # positive: warm light; negative: cool light
    tint = rng.uniform(-cfg["tint"], cfg["tint"])
    wb = np.array([1 + t, 1 + tint, 1 - t], np.float32)
    f = np.clip(f, 0, 1) ** gamma * gain * wb
    h, w = f.shape[:2]
    x0, y0, x1, y1 = identity_region(h, w)
    m = float(f[y0:y1, x0:x1].mean()) if y1 > y0 and x1 > x0 else float(f.mean())
    if m < cfg["identity_mean_min"] and m > 0:
        gain *= cfg["identity_mean_min"] / m
        f = f * (cfg["identity_mean_min"] / m)
    sigma = rng.uniform(*cfg["noise_base"]) / math.sqrt(max(gain, 0.2)) / 255.0
    nrng = np.random.default_rng(rng.getrandbits(32))
    lum = nrng.normal(0, sigma, f.shape[:2])[..., None]
    chroma = nrng.normal(0, sigma * cfg["chroma_share"], f.shape)
    f = f + lum + chroma
    out = np.clip(np.round(np.clip(f, 0, 1) * 255), 0, 255).astype(np.uint8)
    return out, {"gain": round(gain, 3), "gamma": round(gamma, 3), "wb": round(t, 3), "sigma255": round(sigma * 255, 2)}
