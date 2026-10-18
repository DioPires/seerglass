# SPDX-License-Identifier: Apache-2.0
"""The views a bank row is averaged over.

A card image as published is cleaner than a card seen in video: it is larger, sharper, exactly
cropped and uncompressed. A bank row is therefore not the embedding of the image alone. It is
the mean over several views of it: the image itself, and copies put through the kind of damage
video does. View 0 is the image; every later view applies, in this order:

1. *edge jitter*: each of the four edges moves in or out by up to 4 % of that side, as an
   imprecise box around the card would; where the box reaches past the image the gap is filled
   with a flat gray;
2. *downscale*: a bilinear resize so that the short side lands between 85 and 125 pixels, or
   between 200 and 340 pixels on every fourth view (views 3, 7, 11, ...), which keeps some
   views near the size of a card shown enlarged;
3. *tone*: a gamma curve, a gain per color channel and a change of saturation;
4. *blur*: a Gaussian blur of radius 0.3 to 0.9 pixels;
5. *compression*: a JPEG round trip at quality 35 to 75 with 4:2:0 chroma subsampling.

Every amount is drawn from a random generator seeded by the card id and the view number, so a
view depends on nothing else: not on the other cards, not on the order of the build. The same
card image always gives the same parameters.

The pixels additionally depend on the imaging library: the resize, the blur and the JPEG codec
are Pillow's, and a different Pillow or JPEG library build may round some pixels differently.
That is one reason an existing bank is extended (see ``cardvision.bank.build``) rather than
rebuilt when cards are added.

The number of views, the seed and every range above are part of what a bank is. Changing any of
them changes every row.
"""

from __future__ import annotations

import io
import zlib
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

# Views per card, the card image itself included.
VIEWS_PER_CARD = 16

# First word of every view's seed. The other two are the CRC-32 of the card id and the view
# number.
VIEW_SEED = 1337

# Every view whose number has this remainder modulo 4 is downscaled less than the others.
_ENLARGED_REMAINDER = 3


def view_rng(card_id: str, view: int) -> np.random.Generator:
    """The random generator that decides one view of one card."""
    return np.random.default_rng([VIEW_SEED, zlib.crc32(card_id.encode()), view])


def synthesize_view_with_parameters(art: np.ndarray, card_id: str, view: int) -> tuple[np.ndarray, dict[str, Any]]:
    """One synthesized view of a card image, and the amounts that were drawn for it.

    ``art`` is the card image as an ``(H, W, 3)`` uint8 RGB array and ``view`` is 1 or more
    (view 0 is the image itself and is not synthesized). The order in which the amounts are
    drawn from the generator is fixed; it is part of what makes a view reproducible.
    """
    if view < 1:
        raise ValueError(f"view {view} is not synthesized: view 0 is the card image itself")
    if art.ndim != 3 or art.shape[2] != 3 or art.dtype != np.uint8:
        raise ValueError(f"expected an (H, W, 3) uint8 RGB image, got shape {art.shape} of {art.dtype}")
    rng = view_rng(card_id, view)
    art_h, art_w = art.shape[:2]

    # 1. Edge jitter. A positive amount moves that edge inwards.
    jitter = rng.uniform(-0.04, 0.04, size=4)  # left, top, right, bottom
    gray = int(rng.integers(20, 91))
    left, top = int(round(jitter[0] * art_w)), int(round(jitter[1] * art_h))
    right, bottom = int(round(jitter[2] * art_w)), int(round(jitter[3] * art_h))
    x0, y0, x1, y1 = left, top, art_w - right, art_h - bottom
    canvas = np.full((y1 - y0, x1 - x0, 3), gray, dtype=np.uint8)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(art_w, x1), min(art_h, y1)
    canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = art[sy0:sy1, sx0:sx1]

    # 2. Downscale.
    enlarged = view % 4 == _ENLARGED_REMAINDER
    short = rng.uniform(200, 340) if enlarged else rng.uniform(85, 125)
    h, w = canvas.shape[:2]
    k = short / min(h, w)
    target_w, target_h = max(8, int(round(w * k))), max(8, int(round(h * k)))
    image = Image.fromarray(canvas).resize((target_w, target_h), Image.BILINEAR)

    # 3. Tone. The power is taken in float32; the gains widen the values to float64.
    x = np.asarray(image, dtype=np.float32) / 255.0
    gamma = rng.uniform(0.85, 1.15)
    gains = rng.uniform(0.94, 1.06, size=3)
    saturation = rng.uniform(0.85, 1.15)
    x = np.power(x, gamma) * gains[None, None, :]
    luma = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])[..., None]
    x = luma + saturation * (x - luma)
    x = np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)

    # 4. Blur.
    radius = rng.uniform(0.3, 0.9)
    image = Image.fromarray(x).filter(ImageFilter.GaussianBlur(radius))

    # 5. Compression.
    quality = int(rng.integers(35, 76))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality, subsampling=2)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        out = np.array(decoded.convert("RGB"))

    parameters = {
        "jitter": [round(float(amount), 4) for amount in jitter],
        "gray": gray,
        "short": round(float(short), 1),
        "enlarged": enlarged,
        "size": [target_w, target_h],
        "gamma": round(float(gamma), 3),
        "gains": [round(float(gain), 3) for gain in gains],
        "saturation": round(float(saturation), 3),
        "blur": round(float(radius), 3),
        "jpegQuality": quality,
    }
    return out, parameters


def synthesize_view(art: np.ndarray, card_id: str, view: int) -> np.ndarray:
    """One synthesized view of a card image: an ``(h, w, 3)`` uint8 RGB array. ``view`` >= 1."""
    return synthesize_view_with_parameters(art, card_id, view)[0]


def card_views(art: np.ndarray, card_id: str, count: int = VIEWS_PER_CARD) -> list[np.ndarray]:
    """The ``count`` views of a card, in order: the image itself, then views 1 to ``count - 1``."""
    if count < 1:
        raise ValueError(f"a card needs at least one view, got {count}")
    return [art, *(synthesize_view(art, card_id, view) for view in range(1, count))]
