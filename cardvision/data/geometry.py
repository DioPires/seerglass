# SPDX-License-Identifier: Apache-2.0
"""Geometry of labels: fractions of the frame, pixels, and the four corners of a card.

A manifest stores every coordinate as a fraction of the frame: ``x`` of the frame's width and
``y`` of its height, with the origin at the top-left corner and ``y`` growing downwards. A
fraction is turned into pixels by multiplying by the size of the frame at hand, so the same row
serves a frame decoded at any resolution of the same shape.

A card's outline is a quad: four corners, clockwise as seen on screen. When the quad is
*oriented*, the first corner is the card's own top-left, so the first edge is the card's top
whatever way the card is turned.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

Point2 = Sequence[float]
QuadLike = Sequence[Point2]
PixelBox = tuple[float, float, float, float]  # x, y, width, height


def to_pixels(quad: QuadLike, width: float, height: float) -> np.ndarray:
    """Fractional corners as a ``(4, 2)`` float array of pixel coordinates in a frame of the
    given size."""
    points = np.asarray(quad, dtype=np.float64).reshape(-1, 2)
    return points * np.array([float(width), float(height)])


def to_fractions(points: QuadLike, width: float, height: float, digits: int | None = 6) -> list[list[float]]:
    """Pixel coordinates in a frame of the given size as fractions of it.

    ``digits`` rounds the result (six decimals is a hundredth of a pixel at any frame size in
    use); pass ``None`` to keep full precision.
    """
    out = np.asarray(points, dtype=np.float64).reshape(-1, 2) / np.array([float(width), float(height)])
    if digits is not None:
        out = np.round(out, digits)
    return [[float(x), float(y)] for x, y in out]


def quad_from_box(x: float, y: float, width: float, height: float) -> list[list[float]]:
    """The corners of an upright box, clockwise from its top-left. Units are the caller's."""
    return [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]


def quad_from_rotated_box(
    center_x: float, center_y: float, width: float, height: float, angle_degrees: float
) -> list[list[float]]:
    """The corners of a rectangle turned about its center, in **pixels**.

    ``width`` is the length of the card's top edge and ``height`` the length of its side.
    ``angle_degrees`` is the direction of the top edge, measured clockwise on screen from the
    positive x axis: 0 for an upright card, 90 for a card turned a quarter clockwise. The
    corners come back clockwise from the card's own top-left. Work in pixels, not fractions: a
    rectangle is only a rectangle where the two axes share a unit.
    """
    angle = math.radians(angle_degrees)
    ux, uy = math.cos(angle), math.sin(angle)  # along the top edge
    vx, vy = -uy, ux  # down the card's side
    half_w, half_h = width / 2.0, height / 2.0
    corners = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
    return [[center_x + a * ux + b * vx, center_y + a * uy + b * vy] for a, b in corners]


def bounding_box(points: QuadLike, width: float | None = None, height: float | None = None) -> PixelBox:
    """The upright box ``(x, y, width, height)`` around a set of points.

    With a frame size, the box is clipped to the frame; a box left with no area has a width or
    height of zero.
    """
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    x0, y0 = float(array[:, 0].min()), float(array[:, 1].min())
    x1, y1 = float(array[:, 0].max()), float(array[:, 1].max())
    if width is not None and height is not None:
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(width), x1), min(float(height), y1)
    return x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)


def side_lengths(points: QuadLike) -> tuple[float, float]:
    """Lengths of the first edge (the top, for an oriented quad) and the second edge (the
    side), in the units of the points. Meaningful in pixels."""
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return float(np.linalg.norm(array[1] - array[0])), float(np.linalg.norm(array[2] - array[1]))


def side_aspect(points: QuadLike) -> float:
    """The shorter of a quad's first two edges divided by the longer: its aspect once the
    turn is taken out, between 0 and 1. A whole card of the usual shape is near 0.72."""
    first, second = side_lengths(points)
    longer = max(first, second)
    return 0.0 if longer == 0 else min(first, second) / longer


def short_side(points: QuadLike) -> float:
    """The shorter of a quad's first two edges."""
    return min(side_lengths(points))


def top_edge_angle(points: QuadLike) -> float:
    """Direction of a quad's first edge in degrees, measured clockwise on screen from the
    positive x axis, in ``(-180, 180]``: 0 for an upright card, 90 for one turned a quarter
    clockwise, 180 for one upside down."""
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    dx, dy = array[1] - array[0]
    angle = math.degrees(math.atan2(float(dy), float(dx)))
    return 180.0 if angle == -180.0 else angle


def up_angle(points: QuadLike) -> float:
    """How far an oriented quad's up direction is turned from screen-up, in degrees clockwise,
    in ``(-180, 180]``.

    The up direction runs from the quad's center (the mean of its four corners) to the middle
    of its top edge. For a rectangle that is the same angle as ``top_edge_angle``; for a quad
    that follows perspective the two can differ, since its top edge need not be square to it.
    """
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    up = (array[0] + array[1]) / 2 - array.mean(0)
    return math.degrees(math.atan2(up[0], -up[1]))


def quarter_turns(points: QuadLike) -> int:
    """How many counter-clockwise quarter turns bring an oriented quad nearest to upright.

    The value is the ``k`` of ``numpy.rot90``: applying ``numpy.rot90(crop, k)`` to the card's
    upright bounding-box crop gives the card with its top at the top, to the nearest quarter.
    It is ``up_angle`` rounded to a quarter turn: the direction from the quad's center to the
    middle of its top edge decides, not the slope of the top edge alone.
    """
    return int(round(up_angle(points) / 90.0)) % 4


def signed_area(points: QuadLike) -> float:
    """Signed area of a polygon by the shoelace rule; positive when the corners run clockwise
    on screen (``y`` grows downwards)."""
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    x, y = array[:, 0], array[:, 1]
    return float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) / 2.0)


def is_convex(points: QuadLike) -> bool:
    """Whether the corners form a convex polygon with no three of them in a line."""
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    count = len(array)
    signs = set()
    for index in range(count):
        a, b, c = array[index], array[(index + 1) % count], array[(index + 2) % count]
        cross = float((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
        if cross == 0.0:
            return False
        signs.add(cross > 0)
    return len(signs) == 1


def clockwise(quad: QuadLike) -> list[list[float]]:
    """The same corners in clockwise order, keeping the first corner first.

    A card seen in a video is never mirrored, so corners listed counter-clockwise from the
    card's top-left are top-left, bottom-left, bottom-right, top-right; reversing the last
    three gives the order this format uses and loses nothing.
    """
    points = [[float(x), float(y)] for x, y in quad]
    if signed_area(points) < 0:
        points = [points[0], *reversed(points[1:])]
    return points


def rotate_box_with_crop(box: PixelBox, crop_width: float, crop_height: float, turns: int) -> PixelBox:
    """Where a box inside a crop lands after the crop is turned by ``numpy.rot90(crop, turns)``.

    ``box`` is ``(x, y, width, height)`` in the crop before turning; ``crop_width`` and
    ``crop_height`` are the crop's size before turning.
    """
    x, y, width, height = box
    for _ in range(turns % 4):
        # One counter-clockwise quarter turn: the point (x, y) moves to (y, crop_width - x).
        x, y, width, height = y, crop_width - (x + width), height, width
        crop_width, crop_height = crop_height, crop_width
    return x, y, width, height
