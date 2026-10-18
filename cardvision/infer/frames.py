# SPDX-License-Identifier: Apache-2.0
"""Where frames come from: image files, folders of images, and video files.

Images are read with Pillow. Video needs OpenCV, which is an optional dependency: install it
with ``python -m pip install -e ".[video]"`` in the repository folder. Nothing here imports
it until a video is opened.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from cardvision.core.images import load_rgb

VIDEO_SUFFIXES = frozenset({".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".ts", ".flv"})
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"})
# The formats the frames of a video can be named for, and so written in.
VIDEO_FRAME_FORMATS = ("jpg", "png")

# The detector is run on a frame whose width lies in this range; a narrower frame is enlarged
# and a wider one reduced, keeping the aspect ratio. The detector and the accept rule were
# measured on frames prepared this way.
MIN_ANALYSIS_WIDTH = 720
MAX_ANALYSIS_WIDTH = 1920


class VideoSupportMissing(RuntimeError):
    """A video was given but OpenCV is not installed."""


@dataclass(frozen=True)
class Frame:
    """One frame to analyze.

    Attributes:
        name: a file name for the frame, unique within a run; outputs are named after it.
        source: the path of the image or video it came from.
        t: a time in seconds at which the frame is on screen in its video, between the
            frame's own timestamp and the next frame's (``read_video``); 0 for an image.
        rgb: the pixels, ``(H, W, 3)`` uint8.
    """

    name: str
    source: str
    t: float
    rgb: np.ndarray


def is_video(path: str | Path) -> bool:
    """Whether a path is treated as a video, judged by its suffix."""
    return Path(path).suffix.lower() in VIDEO_SUFFIXES


def round_half_up(value: float) -> int:
    """The nearest whole number; a value exactly between two whole numbers goes to the higher.

    Python's ``round`` sends such a value to the even neighbor. This is the rule of
    JavaScript's ``Math.round``, which the browser runtime uses for the analysis size.
    """
    below = math.floor(value)
    return below + 1 if value - below >= 0.5 else below


def analysis_size(
    width: int,
    height: int,
    min_width: int = MIN_ANALYSIS_WIDTH,
    max_width: int = MAX_ANALYSIS_WIDTH,
) -> tuple[int, int]:
    """``(width, height)`` of the analysis frame for a frame of the given size.

    The width is the nearest one within the range. The height keeps the frame's shape and is
    rounded to the nearest whole number, an exact half upward (``round_half_up``): 480 x 271
    is analyzed at 720 x 407, as in the browser runtime.
    """
    target = max(min_width, min(max_width, width))
    return target, round_half_up(target * height / width)


def analysis_frame(
    rgb: np.ndarray,
    min_width: int = MIN_ANALYSIS_WIDTH,
    max_width: int = MAX_ANALYSIS_WIDTH,
) -> np.ndarray:
    """The frame the detector analyzes: ``rgb`` itself when its width is within range,
    otherwise a copy resized to ``analysis_size`` with Pillow's bilinear filter."""
    height, width = rgb.shape[:2]
    size = analysis_size(width, height, min_width, max_width)
    if size[0] == width:
        return rgb
    return np.asarray(Image.fromarray(rgb).resize(size, Image.BILINEAR))


def expand_inputs(inputs: Sequence[str | Path]) -> list[Path]:
    """Turn the given paths into a list of files: a folder becomes the images in it, sorted by
    name. Raises ``FileNotFoundError`` for a path that does not exist and ``ValueError`` for a
    folder without images."""
    files: list[Path] = []
    for entry in inputs:
        path = Path(entry)
        if path.is_dir():
            found = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
            if not found:
                raise ValueError(f"{path}: the folder holds no image files")
            files.extend(found)
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"{path}: no such file or folder")
    return files


def _import_cv2():
    try:
        import cv2
    except ImportError:
        raise VideoSupportMissing(
            'reading video needs OpenCV, which is an optional dependency: python -m pip install -e ".[video]" in the repository folder'
        ) from None
    return cv2


def read_video(path: str | Path, every: float = 1.0, image_format: str = "jpg") -> Iterator[Frame]:
    """Sample a video: its first frame, then one frame every ``every`` seconds of video time.

    Frames are named ``<video name>-f00001.jpg``, ``-f00002.jpg`` and so on, in order; with
    ``image_format`` ``png`` the names end in ``.png``, and a frame written under its name is
    then stored without loss. Raises ``VideoSupportMissing`` when OpenCV is not installed and
    ``ValueError`` when the file cannot be opened or holds no frames.

    **Times are the frames' own.** The file is read from its start, frame by frame, and each
    frame's time is its presentation timestamp as the container gives it. So everything below
    holds for a file of constant frame rate and for one of variable frame rate, as a phone or
    a screen recorder writes.

    **Which frames.** The first frame is taken. After that, the frame taken for the moment
    ``k * every`` seconds (``k`` = 1, 2, ...) is the first frame at or after that moment, or
    less than one frame's time before it. Exactly: times are counted in frames' times at the
    rate the file reports for itself (one number, an average or a nominal rate), to the
    nearest whole one, and the frame taken is the first whose count is at least the moment's. In a file of constant frame rate that is frame number
    ``round(k * every * rate)``. A frame is taken once: where one frame is the first for
    several moments, because the frame before it stayed on screen past them, the next frame
    taken is the one for the first moment still to come. An interval of one frame's time or
    less takes every frame.

    **A frame's ``t`` names the frame.** It is a time at which the frame is on screen: halfway
    between the frame's own timestamp and the next frame's, to the millisecond
    (``cardvision.data.extract.on_screen_time``). A manifest row made from it is found again by
    the manifest's rule, "the last frame whose timestamp is at or before ``t``":
    ``cardvision.data.extract.VideoReader.frame_at(t)`` gives this very frame from the same
    file. The frame's timestamp itself would not do: rounded to the millisecond it can fall
    just before the frame. A frame is therefore handed on once the frame after it has been
    reached, and the last one when the file ends: its time is half a frame's time, at the
    rate the file reports, after its timestamp.
    """
    if every <= 0:
        raise ValueError(f"the sampling interval must be positive, got {every}")
    if image_format not in VIDEO_FRAME_FORMATS:
        raise ValueError(f"unknown image format {image_format!r}; choose one of {', '.join(VIDEO_FRAME_FORMATS)}")
    cv2 = _import_cv2()
    path = Path(path)
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"{path}: cannot be opened as a video")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps <= 0:
            raise ValueError(f"{path}: the video does not report a frame rate")
        from cardvision.data.extract import frame_stamp, on_screen_time

        index = 0  # position of the next frame in the stream
        taken = 0
        moment = 0  # the next frame is taken for the moment `moment * every` seconds
        every_frame = every * fps <= 1.0
        held: tuple[str, float, np.ndarray] | None = None  # a sample that waits for the next frame's timestamp
        while True:
            if not capture.grab():
                break
            time = frame_stamp(capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, index, fps)
            if held is not None:
                name, started, rgb = held
                held = None
                yield Frame(name=name, source=str(path), t=on_screen_time(started, time, fps), rgb=rgb)
            # Times are compared as numbers of frames' times at the reported rate, to the nearest one.
            reached = round(time * fps)
            if taken == 0 or every_frame or reached >= round(moment * every * fps):
                ok, bgr = capture.retrieve()
                if ok and bgr is not None:
                    taken += 1
                    # This frame also serves every later moment it has reached.
                    moment = max(moment + 1, int(reached / (every * fps)))
                    while round(moment * every * fps) <= reached:
                        moment += 1
                    held = (f"{path.stem}-f{taken:05d}.{image_format}", time, np.ascontiguousarray(bgr[:, :, ::-1]))
            index += 1
        if held is not None:
            name, started, rgb = held
            yield Frame(name=name, source=str(path), t=on_screen_time(started, None, fps), rgb=rgb)
        if taken == 0:
            raise ValueError(f"{path}: no frame could be read")
    finally:
        capture.release()


def read_frames(inputs: Sequence[str | Path], every: float = 1.0, image_format: str = "jpg") -> Iterator[Frame]:
    """Every frame of the given images, folders and videos, in the order given. ``image_format``
    is the format the frames of a video are named for (``read_video``); an image keeps its name.

    Two frames may not share a name, because outputs are named after frames; that is an error
    raised before the second one is read.
    """
    seen: dict[str, str] = {}

    def claim(frame_name: str, source: str) -> None:
        if frame_name in seen:
            raise ValueError(f"{source} and {seen[frame_name]} would both produce a frame named {frame_name}")
        seen[frame_name] = source

    for path in expand_inputs(inputs):
        if is_video(path):
            for frame in read_video(path, every, image_format):
                claim(frame.name, frame.source)
                yield frame
        else:
            claim(path.name, str(path))
            yield Frame(name=path.name, source=str(path), t=0.0, rgb=load_rgb(path))
