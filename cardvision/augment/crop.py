# SPDX-License-Identifier: Apache-2.0
"""Crop and card-image augmentations for recognizer training.

Every function works on ``(H, W, 3)`` uint8 RGB arrays at the crop's own resolution, before the
preprocessing contract pools it to the model's input, and draws from the ``random.Random`` it
is given, so a seed fixes the result. Each can be switched off in ``CropAugment``.

What each one stands for in real footage:

``frame_jitter``
    A detector's box is never exactly the card: the crop is cut a little off-center, a little
    too large or small, a little too wide or tall. Needs a crop with room around the card and
    the card's box inside it (``inner``).
``resolution_jitter``
    Cards are seen at many sizes; the crop is reduced by a random factor.
``background_swap``
    What surrounds and shows around a card varies; the crop's border is replaced by a patch of
    some other background.
``cutout``
    Another object covers part of the card: a dark, or sometimes light, rectangle.
``jpeg``
    Video compression.
``photometric``
    Lighting and color: brightness, contrast, saturation, sometimes blur and noise.
``center_mask``
    An input variant, not an augmentation: the border is painted gray. It is applied at
    training and at inference alike when a model is built that way. The ``full128`` contract
    does not use it.

A card is not flipped or turned by any of these: orientation is handled outside the model, by
trying the four quarter turns.

``augment_art`` is the chain for a clean card image, the other half of a training pair.
``on_surface`` places a card image on a background with room around it, which gives it the
form of a crop cut from a frame.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass
class CropAugment:
    """Which augmentations run, and their ranges.

    Attributes:
        frame_jitter, res_jitter, bg_swap, cutout, jpeg, photometric: switches.
        center_mask: paint the border gray (an input variant; see the module docstring).
        jitter_translate: the center moves by up to this fraction of the box's size.
        jitter_scale: the box is scaled by a factor in this range.
        jitter_aspect: and its width over height by a factor in this range.
        res_range: the crop is reduced to a fraction of its size in this range.
        bg_swap_p: probability of a background swap.
        bg_keep: the central fraction of the crop, per axis, that a swap keeps.
        cutout_p: probability of a cutout.
        cutout_max: the largest cutout, as a fraction of the crop's area.
        jpeg_p: probability of JPEG compression.
        jpeg_q: its quality range.
        center_keep: the central fraction, per axis, that the center mask keeps.
        brightness: range of the brightness factor.
        cutout_light_p: probability that a cutout is light instead of dark.
    """

    frame_jitter: bool = True
    res_jitter: bool = True
    bg_swap: bool = True
    cutout: bool = True
    jpeg: bool = True
    photometric: bool = True
    center_mask: bool = False
    jitter_translate: float = 0.15
    jitter_scale: tuple = (0.8, 1.25)
    jitter_aspect: tuple = (0.9, 1.1)
    res_range: tuple = (0.35, 1.0)
    bg_swap_p: float = 0.35
    bg_keep: float = 0.80
    cutout_p: float = 0.30
    cutout_max: float = 0.30
    jpeg_p: float = 0.5
    jpeg_q: tuple = (40, 90)
    center_keep: float = 0.72
    brightness: tuple = (0.8, 1.2)
    cutout_light_p: float = 0.0


def frame_jitter(img: np.ndarray, inner, rng: random.Random, cfg: CropAugment) -> np.ndarray:
    """Cut a jittered card box out of an image that has room around the card.

    ``inner`` is the card's box ``(x, y, width, height)`` in ``img``. With the jitter off, the
    box itself is cut. The cut is clipped to the image and is at least 2 px on each side.
    """
    H, W = img.shape[:2]
    x, y, w, h = inner
    if cfg.frame_jitter:
        s = rng.uniform(*cfg.jitter_scale)
        a = rng.uniform(*cfg.jitter_aspect)
        nw, nh = w * s * a, h * s / a
        cx = x + w / 2 + rng.uniform(-cfg.jitter_translate, cfg.jitter_translate) * w
        cy = y + h / 2 + rng.uniform(-cfg.jitter_translate, cfg.jitter_translate) * h
    else:
        nw, nh, cx, cy = w, h, x + w / 2, y + h / 2
    x0 = int(np.clip(cx - nw / 2, 0, W - 2))
    y0 = int(np.clip(cy - nh / 2, 0, H - 2))
    x1 = int(np.clip(cx + nw / 2, x0 + 2, W))
    y1 = int(np.clip(cy + nh / 2, y0 + 2, H))
    return img[y0:y1, x0:x1]


def resolution_jitter(crop: np.ndarray, rng: random.Random, cfg: CropAugment) -> np.ndarray:
    """Reduce the crop by a random factor (bilinear), never below 12 px on a side."""
    if not cfg.res_jitter:
        return crop
    s = rng.uniform(*cfg.res_range)
    h, w = crop.shape[:2]
    nw, nh = max(12, int(w * s)), max(12, int(h * s))
    return np.asarray(Image.fromarray(crop).resize((nw, nh), Image.BILINEAR))


def _window(patch: np.ndarray, w: int, h: int, rng: random.Random) -> np.ndarray:
    """A ``(h, w)`` piece of a background patch: a random window where the patch is larger,
    the whole patch resized where it is not."""
    ph, pw = patch.shape[:2]
    px = rng.randint(0, max(0, pw - w)) if pw > w else 0
    py = rng.randint(0, max(0, ph - h)) if ph > h else 0
    canvas = patch[py : py + h, px : px + w]
    if canvas.shape[:2] != (h, w):
        canvas = np.asarray(Image.fromarray(patch).resize((w, h), Image.BILINEAR))
    return canvas.copy()


def background_swap(crop: np.ndarray, patch: np.ndarray | None, rng: random.Random, cfg: CropAugment) -> np.ndarray:
    """Replace the crop's border with another background.

    The canvas is a piece of ``patch`` of the crop's size, and the central ``bg_keep`` of the
    crop is pasted onto it. Nothing happens without a patch.
    """
    if not cfg.bg_swap or patch is None or rng.random() > cfg.bg_swap_p:
        return crop
    h, w = crop.shape[:2]
    canvas = _window(patch, w, h, rng)
    kx, ky = int(w * (1 - cfg.bg_keep) / 2), int(h * (1 - cfg.bg_keep) / 2)
    canvas[ky : h - ky, kx : w - kx] = crop[ky : h - ky, kx : w - kx]
    return canvas


def cutout(crop: np.ndarray, rng: random.Random, cfg: CropAugment) -> np.ndarray:
    """Paint one rectangle of a flat dark (or, with ``cutout_light_p``, light) gray."""
    if not cfg.cutout or rng.random() > cfg.cutout_p:
        return crop
    h, w = crop.shape[:2]
    area = rng.uniform(0.05, cfg.cutout_max)
    cw = int(w * np.sqrt(area) * rng.uniform(0.7, 1.4))
    ch = int(h * area / max(cw / w, 1e-6) * w) if cw else 0
    cw, ch = min(cw, w - 1), min(max(ch, 4), h - 1)
    if cw < 2 or ch < 2:
        return crop
    x0 = rng.randint(0, w - cw)
    y0 = rng.randint(0, h - ch)
    out = crop.copy()
    fill = rng.randint(190, 240) if rng.random() < cfg.cutout_light_p else rng.randint(10, 60)
    out[y0 : y0 + ch, x0 : x0 + cw] = fill
    return out


def jpeg(crop: np.ndarray, rng: random.Random, cfg: CropAugment) -> np.ndarray:
    """Compress and decode the crop as a JPEG of random quality."""
    if not cfg.jpeg or rng.random() > cfg.jpeg_p:
        return crop
    buf = io.BytesIO()
    Image.fromarray(crop).save(buf, "JPEG", quality=rng.randint(*cfg.jpeg_q))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def photometric(crop: np.ndarray, rng: random.Random, cfg: CropAugment) -> np.ndarray:
    """Brightness, contrast and saturation factors, then a Gaussian blur for three crops in
    ten and Gaussian noise (sigma 5) for three in ten."""
    if not cfg.photometric:
        return crop
    im = Image.fromarray(crop)
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(*cfg.brightness))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.8, 1.2))
    im = ImageEnhance.Color(im).enhance(rng.uniform(0.8, 1.2))
    if rng.random() < 0.3:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    out = np.asarray(im).astype(np.int16)
    if rng.random() < 0.3:
        out = out + np.random.default_rng(rng.getrandbits(32)).normal(0, 5, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def center_mask(crop: np.ndarray, cfg: CropAugment) -> np.ndarray:
    """Keep the central ``center_keep`` of the crop and paint the rest gray (128). No random
    draw: with the mask on, the same is done to every input at inference."""
    if not cfg.center_mask:
        return crop
    h, w = crop.shape[:2]
    kx, ky = int(w * (1 - cfg.center_keep) / 2), int(h * (1 - cfg.center_keep) / 2)
    out = np.full_like(crop, 128)
    out[ky : h - ky, kx : w - kx] = crop[ky : h - ky, kx : w - kx]
    return out


def augment_crop(img: np.ndarray, inner, patch, rng: random.Random, cfg: CropAugment) -> np.ndarray:
    """The plain chain for a crop: frame jitter, background swap, cutout, resolution jitter,
    photometric, JPEG, center mask, in that order."""
    c = frame_jitter(img, inner, rng, cfg)
    c = background_swap(c, patch, rng, cfg)
    c = cutout(c, rng, cfg)
    c = resolution_jitter(c, rng, cfg)
    c = photometric(c, rng, cfg)
    c = jpeg(c, rng, cfg)
    return center_mask(c, cfg)


def augment_art(art: np.ndarray, rng: random.Random, cfg: CropAugment) -> np.ndarray:
    """The chain for a clean card image: a mild perspective warp (each corner moved inwards by
    up to 6 % of the side, when ``frame_jitter`` is on), then resolution jitter, photometric,
    JPEG and the center mask. The image keeps its size up to the resolution jitter."""
    h, w = art.shape[:2]
    im = Image.fromarray(art)
    if cfg.frame_jitter:
        j = 0.06
        q = [rng.uniform(0, j) * w, rng.uniform(0, j) * h,
             w - rng.uniform(0, j) * w, rng.uniform(0, j) * h,
             w - rng.uniform(0, j) * w, h - rng.uniform(0, j) * h,
             rng.uniform(0, j) * w, h - rng.uniform(0, j) * h]
        im = im.transform((w, h), Image.QUAD, q, Image.BILINEAR)
    c = np.asarray(im)
    c = resolution_jitter(c, rng, cfg)
    c = photometric(c, rng, cfg)
    c = jpeg(c, rng, cfg)
    return center_mask(c, cfg)


def on_surface(art: np.ndarray, patch: np.ndarray | None, margin: float, rng: random.Random):
    """Place a card image on a background, with ``margin`` times its width and height of
    background on each side, as a crop cut from a frame has.

    Returns ``(image, inner)``: ``inner`` is the card's box ``(x, y, width, height)`` in the
    image. Without a patch, or with a margin of 0, the card image itself is returned.
    """
    h, w = art.shape[:2]
    mx, my = int(round(margin * w)), int(round(margin * h))
    if patch is None or (mx == 0 and my == 0):
        return art, (0, 0, w, h)
    canvas = _window(patch, w + 2 * mx, h + 2 * my, rng)
    canvas[my : my + h, mx : mx + w] = art
    return canvas, (mx, my, w, h)
