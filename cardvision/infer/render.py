# SPDX-License-Identifier: Apache-2.0
"""Pictures of a result: the frame with its boxes drawn, and the crops as files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cardvision.infer.pipeline import Card, FrameResult

NAMED_COLOR = (40, 200, 110)
NOTHING_COLOR = (240, 170, 40)
BACK_COLOR = (90, 150, 240)


def card_label(card: Card, with_id: bool = False) -> str:
    """The text drawn beside a card's box. ``with_id`` puts the card id beside the name, for a
    game where different cards share a name and the name alone does not say which was meant."""
    found = card.recognition
    if found is None:
        return f"{card.class_name} {card.score:.2f}"
    if found.accepted:
        return f"{found.name} {found.card_id} {found.similarity:.2f}" if with_id else f"{found.name} {found.similarity:.2f}"
    if with_id:
        return f"? ({found.top_name} {found.top_id} {found.similarity:.2f})"
    return f"? ({found.top_name} {found.similarity:.2f})"


def card_color(card: Card) -> tuple[int, int, int]:
    """Green for a named card, amber for a face that was not named, blue for a back."""
    if card.recognition is None:
        return BACK_COLOR
    return NAMED_COLOR if card.recognition.accepted else NOTHING_COLOR


def annotate(rgb: np.ndarray, result: FrameResult, with_ids: bool = False) -> Image.Image:
    """A copy of the frame with every card's box and label drawn on it (``with_ids``: see
    ``card_label``)."""
    image = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    line = max(2, round(image.width / 640))
    for card in result.cards:
        x, y, width, height = card.box
        color = card_color(card)
        draw.rectangle((x, y, x + width, y + height), outline=color, width=line)
        label = card_label(card, with_ids)
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_width, text_height = right - left, bottom - top
        label_y = y - text_height - 6 if y - text_height - 6 >= 0 else y + line
        draw.rectangle((x, label_y, x + text_width + 6, label_y + text_height + 6), fill=color)
        draw.text((x + 3, label_y + 3 - top), label, fill=(0, 0, 0), font=font)
    return image


def write_annotated(directory: str | Path, rgb: np.ndarray, result: FrameResult, with_ids: bool = False) -> Path:
    """Save the annotated frame as ``<frame name without suffix>.jpg`` in ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{Path(result.name).stem}.jpg"
    annotate(rgb, result, with_ids).save(path, quality=90)
    return path


def crop_file_name(result: FrameResult, card: Card) -> str:
    """The file name of a card's crop: ``<frame name without suffix>-<index>.png``."""
    return f"{Path(result.name).stem}-{card.index:02d}.png"


def write_crops(directory: str | Path, result: FrameResult) -> list[Path]:
    """Save every face's crop, exactly as the recognizer saw it, as a PNG in ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for card in result.faces:
        path = directory / crop_file_name(result, card)
        Image.fromarray(card.crop).save(path)
        paths.append(path)
    return paths


def write_frame(directory: str | Path, rgb: np.ndarray, name: str) -> Path:
    """Save a frame, unmarked, under its own name in ``directory``.

    The suffix of ``name`` decides the format; a JPEG is written at quality 95.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    options = {"quality": 95} if path.suffix.lower() in (".jpg", ".jpeg") else {}
    Image.fromarray(rgb).save(path, **options)
    return path
