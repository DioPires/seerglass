# SPDX-License-Identifier: Apache-2.0
"""Training-time augmentations for card crops and card images.

Three groups, each in its own module:

``crop``
    The plain chain for a crop cut from a frame (frame jitter, background swap, cutout,
    resolution jitter, photometric, JPEG, the center-mask input variant), the chain for a clean
    card image (``augment_art``), and ``on_surface``, which gives a card image the form of a
    crop.
``physical``
    The physical chain (print effects, sleeve, streaks and foil, tilt, lamp glare, defocus,
    exposure) and ``motion_blur``. It is built on the photometric effects and textures of
    ``cardvision.composites``; which card kinds carry dice or are often foil comes from the
    catalog's ``kinds`` settings.
``components``
    Occluders, glare, low light and wider rotation: separately switchable, each with its own
    random stream and each bounded so that the card's name and art stay readable.

Everything takes and returns ``(H, W, 3)`` uint8 RGB arrays and an explicit random stream: the
same stream state always gives the same pixels. Only NumPy and Pillow are needed.

The order in which a training recipe applies them is in ``cardvision.train_recognizer.dataset``.
"""

from cardvision.augment.components import (
    DEFAULTS,
    glare,
    identity_region,
    low_light,
    occluders,
    rotation,
    row_seed,
    stream_for,
)
from cardvision.augment.crop import (
    CropAugment,
    augment_art,
    augment_crop,
    background_swap,
    center_mask,
    cutout,
    frame_jitter,
    jpeg,
    on_surface,
    photometric,
    resolution_jitter,
)
from cardvision.augment.physical import PHYSICAL, lamp_glare, motion_blur, physical_chain, rotate, sleeve

__all__ = [
    "DEFAULTS",
    "PHYSICAL",
    "CropAugment",
    "augment_art",
    "augment_crop",
    "background_swap",
    "center_mask",
    "cutout",
    "frame_jitter",
    "glare",
    "identity_region",
    "jpeg",
    "lamp_glare",
    "low_light",
    "motion_blur",
    "occluders",
    "on_surface",
    "photometric",
    "physical_chain",
    "resolution_jitter",
    "rotate",
    "rotation",
    "row_seed",
    "sleeve",
    "stream_for",
]
