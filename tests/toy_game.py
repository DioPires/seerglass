# SPDX-License-Identifier: Apache-2.0
"""Draw a toy card game: 20 card images made of plain shapes, colors and text, plus a catalog.

The tests need card images, and no real game's art belongs in this repository, so the images
are generated. The drawing is deterministic: the same call always gives the same pixels.

Run it to look at the result::

    python tests/toy_game.py OUT_DIR

which writes ``OUT_DIR/catalog.json`` and ``OUT_DIR/art/<id>.png``.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PORTRAIT = (240, 336)  # width, height
LANDSCAPE = (336, 240)

_COLORS = {
    "Amber": (232, 163, 23),
    "Azure": (38, 132, 214),
    "Coral": (240, 108, 92),
    "Jade": (46, 160, 110),
    "Plum": (126, 62, 150),
    "Slate": (92, 104, 122),
}
_SHAPES = ("Circle", "Square", "Triangle", "Diamond", "Stripes", "Ring")

# (id, name, kind, printingOf). Eighteen base cards and two alternate printings: one with a
# letter suffix and one with promotional numbering, each sharing its base card's name.
_BASE = [
    (f"TOY-{number:03d}", f"{color} {shape}", kind)
    for number, (color, shape, kind) in enumerate(
        [
            ("Amber", "Circle", "unit"), ("Azure", "Square", "unit"), ("Coral", "Triangle", "unit"),
            ("Jade", "Diamond", "unit"), ("Plum", "Stripes", "unit"), ("Slate", "Ring", "unit"),
            ("Azure", "Circle", "unit"), ("Coral", "Square", "unit"), ("Jade", "Triangle", "unit"),
            ("Plum", "Diamond", "unit"), ("Slate", "Stripes", "unit"), ("Amber", "Ring", "unit"),
            ("Coral", "Circle", "spell"), ("Jade", "Square", "spell"), ("Plum", "Triangle", "spell"),
            ("Slate", "Diamond", "spell"), ("Amber", "Stripes", "site"), ("Azure", "Ring", "site"),
        ],
        start=1,
    )
]
CARDS: list[tuple[str, str, str, str | None]] = [
    *[(card_id, name, kind, None) for card_id, name, kind in _BASE],
    ("TOY-003A", "Coral Triangle", "unit", "TOY-003"),
    ("TOY-P01", "Plum Diamond", "unit", "TOY-010"),
]

KINDS = {
    "landscape": ["site"],
    "foilProbability": {"unit": 0.1, "spell": 0.0},
    "boost": {"shareKind": "site", "share": 0.1},
}


def _shade(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, round(channel * factor))) for channel in color)


def _draw_shape(draw: ImageDraw.ImageDraw, shape: str, box: tuple[int, int, int, int], fill, outline) -> None:
    left, top, right, bottom = box
    cx, cy = (left + right) // 2, (top + bottom) // 2
    if shape == "Circle":
        draw.ellipse(box, fill=fill, outline=outline, width=4)
    elif shape == "Square":
        draw.rectangle(box, fill=fill, outline=outline, width=4)
    elif shape == "Triangle":
        draw.polygon([(cx, top), (right, bottom), (left, bottom)], fill=fill, outline=outline)
    elif shape == "Diamond":
        draw.polygon([(cx, top), (right, cy), (cx, bottom), (left, cy)], fill=fill, outline=outline)
    elif shape == "Stripes":
        step = max(6, (bottom - top) // 8)
        for index, y in enumerate(range(top, bottom, step)):
            draw.rectangle((left, y, right, min(bottom, y + step - 1)), fill=fill if index % 2 == 0 else outline)
    elif shape == "Ring":
        draw.ellipse(box, outline=fill, width=18)
        inset = 36
        draw.ellipse((left + inset, top + inset, right - inset, bottom - inset), outline=outline, width=6)
    else:
        raise ValueError(f"unknown shape {shape!r}")


def draw_card(card_id: str, name: str, kind: str, alternate: bool = False) -> Image.Image:
    """One card image. An alternate printing keeps the name and changes the look."""
    color_name, shape = name.split()
    color = _COLORS[color_name]
    width, height = LANDSCAPE if kind == "site" else PORTRAIT
    background = _shade(color, 0.35 if not alternate else 1.25)
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # Frame, title bar and footer, like a card's layout.
    draw.rectangle((4, 4, width - 5, height - 5), outline=_shade(color, 1.4), width=6)
    draw.rectangle((14, 14, width - 15, 44), fill=_shade(color, 0.7))
    draw.text((20, 22), name, fill=(255, 255, 255), font=font)
    draw.rectangle((14, height - 40, width - 15, height - 15), fill=(24, 24, 28))
    draw.text((20, height - 34), f"{card_id}  {kind}", fill=(230, 230, 230), font=font)

    # The picture: one large shape, and a scatter of small blocks seeded by the card id so that
    # two cards with the same shape and color would still differ.
    margin = 40
    picture = (margin, 60, width - margin, height - 56)
    fill = color if not alternate else _shade(color, 0.5)
    _draw_shape(draw, shape, picture, fill, _shade(color, 1.5 if not alternate else 0.2))
    scatter = random.Random(card_id)
    for _ in range(14):
        x = scatter.randrange(18, width - 30)
        y = scatter.randrange(52, height - 56)
        size = scatter.randrange(4, 12)
        tone = scatter.choice([(255, 255, 255), (16, 16, 20), _shade(color, 1.6)])
        draw.rectangle((x, y, x + size, y + size), fill=tone)
    return image


def catalog_document() -> dict:
    """The toy game's catalog, with ``art`` given as file names."""
    return {
        "game": "toy",
        "locale": "en",
        "cards": [
            {"id": card_id, "name": name, "kind": kind, "printingOf": printing_of, "art": f"{card_id}.png"}
            for card_id, name, kind, printing_of in CARDS
        ],
        "kinds": KINDS,
    }


def write_toy_game(out_dir: str | Path) -> tuple[Path, Path]:
    """Write the catalog and the card images. Returns ``(catalog path, art folder)``."""
    out_dir = Path(out_dir)
    art_dir = out_dir / "art"
    art_dir.mkdir(parents=True, exist_ok=True)
    for card_id, name, kind, printing_of in CARDS:
        draw_card(card_id, name, kind, alternate=printing_of is not None).save(art_dir / f"{card_id}.png")
    catalog_path = out_dir / "catalog.json"
    catalog_path.write_text(json.dumps(catalog_document(), indent=2) + "\n", encoding="utf-8")
    return catalog_path, art_dir


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python tests/toy_game.py OUT_DIR")
    path, folder = write_toy_game(sys.argv[1])
    print(f"wrote {path} and {len(CARDS)} images in {folder}")
