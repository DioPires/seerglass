# SPDX-License-Identifier: Apache-2.0
"""The extraction contract: how a manifest row's frame is obtained from its video.

Labels are fractions of a frame, so they land on the right pixels only if the frame is the one
that was labeled. This module fixes how that frame is found.

**Which stream.** The best video-only stream that is at most 1080 pixels high, MP4 preferred
(``FORMAT_SELECTOR``).

**Which frame.** A row's ``t`` is a time, in seconds from the start of the video, at which the
labeled frame is on screen. The frame for ``t`` is the last frame whose own timestamp is at or
before ``t``. In a video of constant frame rate that is frame number ``floor(t * rate)``,
counting from 0 (``frame_index``). Published rows put ``t`` in the middle of the frame's time
on screen, so rounding to the millisecond never moves it onto a neighbor.

**How a frame's time is read.** A frame's timestamp is its presentation timestamp, as the
video's container gives it. ``VideoReader`` reads it from OpenCV for each frame it decodes, and
tells frames apart by these timestamps alone, whether the frame rate is constant or variable
(a phone or a screen recorder often writes a variable one). The frame rate a file reports
for itself is one number, an average or a nominal rate, and it places no frame. It serves as
a first guess of where to seek, and that guess is checked: the first frame decoded after a
seek must have a timestamp at or before the time wanted, or the reader seeks again, further
back, and in the end reads the file from its start. From a place known to be early enough the
reader steps forward frame by frame.

**What size.** The frame is used at the size it decodes to; nothing is cropped or padded. Its
shape must be the shape the labels were made on: width over height must match the row's
``frame_width`` over ``frame_height`` within ``RATIO_TOLERANCE``, else the row is unusable with
this video (``frame_check``). The size itself may differ, because fractions scale; a frame
written at another size than the row records is marked in the index (``other_size``).

**Not the same bytes.** A frame fetched again is not byte-identical to the frame that was
labeled: platforms re-encode their streams, the stream chosen today may differ in resolution
or codec, and the labeled frames were themselves saved as JPEG. Figures computed on re-fetched
frames can therefore differ slightly from published ones.

**How the frames were first sampled.** Labeled frames were picked from frames sampled at a
fixed rate, one per second, from the whole video when it ran 45 minutes or less and otherwise
from three ten-minute sections starting at 25 %, 50 % and 75 % of its length. The sampler's
n-th frame (from 1) is the last frame before ``n - 0.5`` seconds (``sampled_frame_time``).
Some frames were instead taken singly at a chosen second, as the first frame at or after it
(``frame_time_at_or_after``).

**Which frame, when the row has a fingerprint.** A downloaded stretch of video can be cut a
frame or a second off. A row that carries ``frame_dhash`` (see ``cardvision.data.dhash``) is
therefore matched by content: every frame on screen within ``FINGERPRINT_WINDOW_SECONDS`` of
``t``, either side, is fingerprinted, and the one nearest in Hamming distance is used if it is
within ``SAME_FRAME_MAX_HAMMING`` bits (``nearest_frame``). If none is, the row is unusable
with this video (reason ``fingerprint``). A window cut short by the end of the file holds the
frames that exist. A window that lies wholly past the end holds none, and the row has failed:
the file's last frame is not compared with it (``VideoReader.frames_between``).

Reading a video needs OpenCV, an optional dependency: ``python -m pip install -e ".[video]"`` in the
repository folder.
"""

from __future__ import annotations

import datetime
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cardvision.data.dhash import SAME_FRAME_MAX_HAMMING, dhash, hamming
from cardvision.data.schema import Row, RowKey

# The stream to fetch, in the format-selection syntax of the optional downloader: the best
# video-only stream at most 1080 pixels high, as MP4 when there is one; then the same in any
# container; then a combined stream within the limit; then whatever exists.
FORMAT_SELECTOR = "bv[height<=1080][ext=mp4]/bv[height<=1080]/b[height<=1080]/bv/b"
MAX_HEIGHT = 1080

# Width over height of a fetched frame may differ from the row's by this fraction of it.
RATIO_TOLERANCE = 0.01

# How the candidate frames were first sampled.
SAMPLING_RATE = 1.0  # frames per second
WHOLE_VIDEO_LIMIT_SECONDS = 45 * 60
SECTION_SECONDS = 10 * 60
SECTION_STARTS = (0.25, 0.50, 0.75)  # of the video's length

# A row with a fingerprint is matched against the frames this far either side of its time.
FINGERPRINT_WINDOW_SECONDS = 1.0

INDEX_NAME = "index.jsonl"
FRAMES_FOLDER = "frames"
# How an index line writes the moment it was written: UTC, ISO 8601, to the second.
INDEX_TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

_EPSILON = 1e-6


class VideoSupportMissing(RuntimeError):
    """A video file has to be read but OpenCV is not installed."""


class FrameNotFound(ValueError):
    """The video has no frame at the requested time, or it cannot be read."""


# ------------------------------------------------------------------------------- frame times
def frame_index(t: float, rate: float) -> int:
    """Number (from 0) of the frame on screen at ``t`` seconds in a video of constant frame
    rate: the last frame whose timestamp is at or before ``t``."""
    if t < 0 or rate <= 0:
        raise ValueError(f"a time and a frame rate must be positive, got t={t} and rate={rate}")
    return int(math.floor(t * rate + _EPSILON))


# Frame numbers and times are tied by the frame rate, so the rate has to be the exact one:
# 30000 / 1001, not 29.97. A rounded rate is off by a whole frame after a few hours of video.
# Reading a frame (VideoReader) does not depend on this: it goes by each frame's own timestamp.


def mid_frame_time(index: int, rate: float) -> float:
    """The middle of a frame's time on screen, rounded to the millisecond: the value to store
    as a row's ``t``."""
    return round((index + 0.5) / rate, 3)


def frame_stamp(stamp: float, index: int, rate: float) -> float:
    """The time of a frame as the readers of this package take it: its own timestamp
    ``stamp``, or its number over the frame rate where the backend gives no timestamps (such a
    backend reports 0 for every frame)."""
    by_count = index / rate
    return by_count if (stamp <= 0 and index > 0) or stamp != stamp else stamp


def on_screen_time(time: float, next_time: float | None, rate: float) -> float:
    """A time at which a frame is on screen, rounded to the millisecond: the value to store as
    a row's ``t`` for a frame read from a file.

    ``time`` is the frame's timestamp and ``next_time`` the next frame's (``frame_stamp``), or
    ``None`` for the last frame of the file. The time returned lies halfway between the two,
    or half a frame's time after ``time`` for the last frame. It is at or after the frame's
    own timestamp and before the next frame's, so the frame for it, by the rule under "which
    frame", is this frame. On video of constant frame rate it is ``mid_frame_time``. A
    timestamp alone, rounded to the millisecond, can fall just before the frame it belongs to.
    """
    end = next_time if next_time is not None and next_time > time else time + 1.0 / rate
    return round((time + end) / 2.0, 3)


def sampled_frame_time(number: int, rate: float, start: float = 0.0, sampling_rate: float = SAMPLING_RATE) -> float:
    """The ``t`` of the ``number``-th frame (from 1) that a fixed-rate sampler took.

    A sampler that keeps ``sampling_rate`` frames per second writes, as its n-th frame, the
    last source frame before ``(n - 0.5) / sampling_rate`` seconds. ``rate`` is the video's own
    exact frame rate, and ``start`` the time in the video at which the sampled stretch began (0
    for a whole video, the section's start otherwise).
    """
    if number < 1:
        raise ValueError(f"frames are numbered from 1, got {number}")
    boundary = (number - 0.5) / sampling_rate
    index = max(0, int(math.ceil(boundary * rate - _EPSILON)) - 1)
    return round(start + (index + 0.5) / rate, 3)


def frame_time_at_or_after(second: float, rate: float) -> float:
    """The ``t`` of the first frame whose timestamp is at or after ``second``: the frame a
    single-frame grab at that second returns."""
    index = int(math.ceil(second * rate - _EPSILON))
    return mid_frame_time(index, rate)


def section_starts(duration: float) -> list[float]:
    """Where the sampled stretches of a video began: ``[0]`` for a video sampled whole, else
    the whole second at each of ``SECTION_STARTS`` of its length."""
    if duration <= WHOLE_VIDEO_LIMIT_SECONDS:
        return [0.0]
    return [float(int(duration * share)) for share in SECTION_STARTS]


# ------------------------------------------------------------------------------ frame checks
def frame_check(row: Row, width: int, height: int, tolerance: float = RATIO_TOLERANCE) -> str | None:
    """Why a frame of this size cannot stand for the row's frame, or ``None`` when it can.

    The test is on shape, not size: width over height must match the row's within
    ``tolerance`` (as a fraction of the row's ratio).
    """
    if width <= 0 or height <= 0:
        return "the frame is empty"
    expected = row.frame_width / row.frame_height
    got = width / height
    if abs(got / expected - 1.0) > tolerance:
        return (f"the frame is {width} x {height} (ratio {got:.4f}) but the labels were made on "
                f"{row.frame_width} x {row.frame_height} (ratio {expected:.4f}); the shapes differ "
                f"by more than {tolerance:.0%}, so the labels would not land on the same pixels")
    return None


def frame_file_name(row: Row, suffix: str = ".jpg") -> str:
    """The file name a row's frame is stored under: platform, video id and the time in
    milliseconds, for example ``youtube-EXAMPLE0001-t0000754517.jpg``."""
    return f"{row.platform}-{row.video_id}-t{round(row.t * 1000):010d}{suffix}"


# -------------------------------------------------------------------------- reading a video
def _import_cv2():
    try:
        import cv2
    except ImportError:
        raise VideoSupportMissing(
            'reading video needs OpenCV, which is an optional dependency: python -m pip install -e ".[video]" in the repository folder'
        ) from None
    return cv2


@dataclass(frozen=True)
class VideoFrame:
    """A frame read from a video file.

    Attributes:
        rgb: the pixels, ``(H, W, 3)`` uint8.
        index: the frame's number in the file, from 0, as OpenCV counts it. The count is exact
            in a file of constant frame rate, and wherever the frames were counted from the
            start of the file. After a seek in a file of variable frame rate it is a guess
            made from the rate the file reports.
        time: the frame's own timestamp in seconds.
    """

    rgb: np.ndarray
    index: int
    time: float


class VideoReader:
    """Read single frames of a video file by time. Use as a context manager.

    ``offset`` is the time in the original video at which this file starts: 0 for a whole
    video, the section's start for a file that holds one section of it.

    **Frame times.** A frame's time is its presentation timestamp, which OpenCV reports for
    each frame it decodes. Every question this class answers is settled by those timestamps:
    which frame is on screen at a time, which frames are on screen in a window, and whether
    the file ends before a time. ``rate`` is the frame rate the file reports for itself: one
    number, an average or a nominal rate. It is used for two things only: as a guess of where
    to seek, and as the time the last frame of the file stays on screen, which no timestamp
    tells. Where OpenCV gives no timestamps at all, a frame's number over the rate stands in
    (``frame_stamp``), and that is right for a constant rate only.

    **Seeking.** OpenCV seeks by frame number and turns the number into a time by that rate.
    In a file of variable frame rate it can land seconds early or late. So a seek is a
    guess that is checked: the first frame decoded after it must have a timestamp at or before
    the time wanted. If it has not, the reader seeks again further back, by at least twice as
    much each time, and in the end reads the file from its start. From a place known to be
    early enough it steps forward frame by frame. With ``exact`` it never seeks: it reads the
    file from its start, frame by frame, which is slow on a long file and does not depend on
    how well the file can be seeked.
    """

    # Seeking aims this many seconds early and steps forward, so that an inexact seek is
    # corrected by the timestamps of the frames passed. Frames earlier than this before the
    # time wanted are stepped over without being read out.
    _RUN_UP_SECONDS = 1.0
    # The last frame of a file stays on screen for one frame's time at the file's rate. A time
    # up to this many frames' times after its timestamp still counts as on it.
    _LAST_FRAME_TIMES = 1.5

    def __init__(self, path: str | Path, offset: float = 0.0, exact: bool = False) -> None:
        cv2 = _import_cv2()
        self._cv2 = cv2
        self.path = Path(path)
        self.offset = float(offset)
        self.exact = exact
        if not self.path.is_file():
            raise FrameNotFound(f"{self.path}: no such video file")
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise FrameNotFound(f"{self.path}: cannot be opened as a video")
        rate = self._capture.get(cv2.CAP_PROP_FPS)
        if not rate or rate != rate or rate <= 0:
            self._capture.release()
            raise FrameNotFound(f"{self.path}: the video does not report a frame rate")
        self.rate = float(rate)
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._next = 0  # number of the frame the next grab returns, when known
        self._counted = True  # whether the place in the file was reached from its start, without a seek
        self._passed: float | None = None  # timestamp of the last frame stepped past since the start or a seek
        self._pending: tuple[int, float] | None = None  # a frame that was grabbed and is still to be used

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._capture.release()

    def _rewind(self) -> None:
        self._capture.release()
        self._capture = self._cv2.VideoCapture(str(self.path))
        self._next = 0
        self._counted = True
        self._passed = None
        self._pending = None

    def _seek(self, position: int) -> None:
        """Ask OpenCV for frame number ``position``. Where it lands is for the caller to check."""
        cv2 = self._cv2
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, position)
        self._next = int(round(self._capture.get(cv2.CAP_PROP_POS_FRAMES)))
        self._counted = False
        self._passed = None
        self._pending = None

    def _grab(self) -> tuple[int, float] | None:
        """Advance one frame; return its number and timestamp, or ``None`` at the end."""
        if self._pending is not None:
            grabbed, self._pending = self._pending, None
            return grabbed
        cv2 = self._cv2
        if not self._capture.grab():
            return None
        position = int(round(self._capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
        index = position if position >= 0 else self._next
        # A backend without timestamps reports 0 throughout; frame_stamp falls back on the frame count.
        time = frame_stamp(self._capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, index, self.rate)
        self._next = index + 1
        return index, time

    def _step(self, keep_from: float, start: float, end: float) -> tuple[list[VideoFrame], float | None, float | None, bool]:
        """Step forward from the place in the file to the first frame after ``end``.

        All times are seconds of this file. Returns the frames read out: the last frame whose
        timestamp is at or before ``start`` and each later frame whose timestamp is not after
        ``end``. Frames before ``keep_from`` are stepped past and not read out. Also returns
        the timestamp of the first frame met, the timestamp of the last frame stepped past
        without being read out, and whether the file ended.
        """
        frames: list[VideoFrame] = []
        first: float | None = None
        before = self._passed
        while True:
            grabbed = self._grab()
            if grabbed is None:
                return frames, first, before, True
            index, time = grabbed
            if first is None:
                first = time
            if time > end + _EPSILON:
                self._pending = grabbed  # the next call may want it
                return frames, first, before, False
            self._passed = time
            if time < keep_from - _EPSILON:
                before = time
                continue
            ok, bgr = self._capture.retrieve()
            if not ok or bgr is None:
                continue
            frame = VideoFrame(rgb=np.ascontiguousarray(bgr[:, :, ::-1]), index=index, time=time + self.offset)
            if time <= start + _EPSILON:
                frames = [frame]  # still before the window: only the latest of these stays
            else:
                frames.append(frame)

    def _on_screen(self, start: float, end: float) -> tuple[list[VideoFrame], float | None]:
        """The frames on screen from ``start`` to ``end`` seconds of this file, in order: the
        last frame whose timestamp is at or before ``start``, and each later frame whose
        timestamp is not after ``end``. Where the file's first frame comes after ``start``,
        the list begins with that frame. The list can be empty.

        Also returns the timestamp of the file's last frame when the file ended before a frame
        after ``end`` was met, and ``None`` when it did not.
        """
        wanted = start  # the reader has to get to a frame at or before this time
        keep_from = start - self._RUN_UP_SECONDS
        run_up = self._RUN_UP_SECONDS
        from_start = self.exact
        went_back = False

        def place() -> int:
            """The frame number to ask for: ``run_up`` seconds before the time wanted, by the file's rate."""
            return frame_index(max(0.0, wanted), self.rate) - int(math.ceil(run_up * self.rate))

        while True:
            if from_start:
                if not self._counted or (self._passed is not None and self._passed >= keep_from - _EPSILON):
                    self._rewind()
                checked = True
            else:
                position = max(0, place())
                self._seek(position)
                checked = position == 0
            frames, first, before, ended = self._step(keep_from, start, end)
            if not checked and (first is None or first > wanted + _EPSILON):
                # The seek landed after the time wanted, or past the end: aim further back, by
                # as much again or by what the seek was late, whichever is more.
                run_up = max(2.0 * run_up, run_up + (first - wanted if first is not None else 0.0))
                while 0 < place() == position:
                    run_up *= 2.0  # the same place would be asked for again
                from_start = place() <= 0
                continue
            missed = before is not None and (not frames or frames[0].time - self.offset > start + _EPSILON)
            if missed and ended and not frames and start > before + self._LAST_FRAME_TIMES / self.rate:
                missed = False  # the file ended before the window: its last frame is not wanted
            if missed and not went_back:
                # The frame on screen at the start was stepped past: it stays on screen for
                # longer than the run-up. Its timestamp is known now, so go back for it.
                wanted = keep_from = before
                run_up = self._RUN_UP_SECONDS
                went_back = True
                continue
            return frames, self._passed if ended else None

    def frame_at(self, t: float) -> VideoFrame:
        """The frame on screen at ``t`` seconds of the original video: the last frame whose
        timestamp is at or before ``t``.

        Raises ``FrameNotFound`` when ``t`` lies before this file's start or after its end.
        """
        local = t - self.offset
        if local < -_EPSILON:
            raise FrameNotFound(f"{self.path}: t={t} lies before the start of this file ({self.offset} s)")
        local = max(0.0, local)
        frames, last = self._on_screen(local, local)
        if not frames:
            if last is not None:
                raise FrameNotFound(f"{self.path}: the video ends before t={t}")
            raise FrameNotFound(f"{self.path}: no frame at t={t}")
        # The last frame of a file stays on screen for one frame's time and no longer.
        if last is not None and local > last + self._LAST_FRAME_TIMES / self.rate:
            raise FrameNotFound(f"{self.path}: the video ends before t={t}")
        return frames[0]

    def frames_between(self, start: float, end: float) -> list[VideoFrame]:
        """Every frame that is on screen at some moment from ``start`` to ``end`` seconds of
        the original video, in order: the frame on screen at ``start`` and each later frame
        whose timestamp is not after ``end``.

        A window reaching before this file's start or past its end gives the frames that exist.
        Raises ``FrameNotFound`` when there are none: the window lies before the file's start,
        or it starts after the file's last frame has left the screen. The last frame is not
        handed out for a window that lies wholly past the end.
        """
        local_start = max(0.0, start - self.offset)
        local_end = end - self.offset
        if local_end < -_EPSILON:
            raise FrameNotFound(f"{self.path}: the window {start} to {end} lies before the start of this file ({self.offset} s)")
        frames, last = self._on_screen(local_start, local_end)
        if last is not None and local_start > last + self._LAST_FRAME_TIMES / self.rate:
            raise FrameNotFound(f"{self.path}: the video ends before t={round(start, 6)}, where the window that runs "
                                f"to t={round(end, 6)} starts")
        if not frames:
            raise FrameNotFound(f"{self.path}: no frame from t={start} to t={end}")
        return frames


def nearest_frame(frames: Sequence[VideoFrame], fingerprint: str, t: float) -> tuple[VideoFrame, int]:
    """The frame whose fingerprint is nearest ``fingerprint``, with its distance in bits.

    Ties go to the frame nearest ``t`` in time, then to the earlier one. Whether the distance
    is small enough to call it the same frame is for the caller: it is when it is at most
    ``SAME_FRAME_MAX_HAMMING``. Raises ``ValueError`` when ``frames`` is empty.
    """
    if not frames:
        raise ValueError("no frames to compare")
    keyed = [((hamming(dhash(frame.rgb), fingerprint), abs(frame.time - t), frame.time), frame) for frame in frames]
    (distance, _, _), frame = min(keyed, key=lambda pair: pair[0])
    return frame, distance


def read_frame_at(path: str | Path, t: float, offset: float = 0.0, exact: bool = False) -> VideoFrame:
    """The frame on screen at ``t`` seconds, from a video file that starts ``offset`` seconds
    into the original video. ``exact`` reads the file from its start, frame by frame, instead
    of seeking: slow on a long file, and independent of how well the file can be seeked."""
    with VideoReader(path, offset=offset, exact=exact) as reader:
        return reader.frame_at(t)


# ---------------------------------------------------------------------------- the frame index
def index_time(moment: datetime.datetime | None = None) -> str:
    """A moment as an index line records it: in UTC, ISO 8601, to the second, for example
    ``2026-01-31T09:30:00Z``. The present moment when none is given. A moment without a time
    zone is taken to be in UTC."""
    if moment is None:
        moment = datetime.datetime.now(datetime.timezone.utc)
    elif moment.tzinfo is not None:
        moment = moment.astimezone(datetime.timezone.utc)
    return moment.strftime(INDEX_TIME_FORMAT)


def index_date(written: str) -> str:
    """The date, in UTC, of a moment an index line recorded (``index_time``), as
    ``YYYY-MM-DD``. Raises ``ValueError`` when the text is not such a moment."""
    return datetime.datetime.strptime(written, INDEX_TIME_FORMAT).date().isoformat()


@dataclass(frozen=True)
class IndexEntry:
    """One line of a frame folder's index: what became of one row's frame.

    Attributes:
        platform, video_id, t: the row's key.
        status: ``ok``, ``unusable`` (a frame was read but fails the shape check),
            ``no-video`` (no file was given and nothing was downloaded) or ``failed``.
        file: the frame's path relative to the folder, when ``status`` is ``ok``.
        width, height: the size of the frame read, when one was.
        frame_time: the timestamp of the frame read, in seconds of the original video.
        scale: frame width over the row's ``frame_width``, when ``status`` is ``ok``.
        source: where the video file came from: ``local`` (a file the user gave), ``download``
            or ``kept`` (a file kept from an earlier download).
        reason: why, when ``status`` is not ``ok``. ``fingerprint`` means no frame near ``t``
            looked like the row's ``frame_dhash``.
        cause: for a failed download, why in one word: ``unavailable`` (the platform says the
            video is private, removed or does not exist), ``restricted`` (the video is there
            and this request may not have it) or ``error`` (anything else, which says nothing
            about the video). See ``cardvision.data.fetch.failure_cause``.
        distance: for a row with a ``frame_dhash``, the Hamming distance in bits between it
            and the frame chosen (``ok``), or the nearest frame found (``fingerprint``).
        offset: for a row with a ``frame_dhash`` and a chosen frame, ``frame_time`` minus
            ``t``, in seconds, to the millisecond.
        attempts: for a row served by a download, how many times its part of the video was
            asked for: 1, or more when a failed download was tried again.
        other_size: true when the frame written is not ``frame_width`` by ``frame_height``
            pixels, the size the labels were made on. The shape is the same, so the labels
            fit; ``scale`` says by how much the size differs.
        written: when the line was written (``index_time``). ``None`` on a line of an index
            written before lines recorded it.
    """

    platform: str
    video_id: str
    t: float
    status: str
    file: str | None = None
    width: int | None = None
    height: int | None = None
    frame_time: float | None = None
    scale: float | None = None
    source: str | None = None
    reason: str | None = None
    distance: int | None = None
    offset: float | None = None
    attempts: int | None = None
    other_size: bool | None = None
    cause: str | None = None
    written: str | None = None

    @property
    def key(self) -> RowKey:
        return self.platform, self.video_id, self.t

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"platform": self.platform, "video_id": self.video_id, "t": self.t, "status": self.status}
        for name in ("file", "width", "height", "frame_time", "scale", "source", "reason", "cause", "distance", "offset",
                     "attempts", "other_size", "written"):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        return out

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> IndexEntry:
        return cls(platform=doc["platform"], video_id=doc["video_id"], t=float(doc["t"]), status=doc["status"],
                   file=doc.get("file"), width=doc.get("width"), height=doc.get("height"),
                   frame_time=doc.get("frame_time"), scale=doc.get("scale"), source=doc.get("source"),
                   reason=doc.get("reason"), distance=doc.get("distance"), offset=doc.get("offset"),
                   attempts=doc.get("attempts"), other_size=doc.get("other_size"), cause=doc.get("cause"),
                   written=doc.get("written"))


def read_index(folder: str | Path) -> dict[RowKey, IndexEntry]:
    """A frame folder's index, by row key. A row listed more than once keeps its last entry;
    a folder without an index gives an empty mapping."""
    path = Path(folder) / INDEX_NAME
    entries: dict[RowKey, IndexEntry] = {}
    if not path.is_file():
        return entries
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                entry = IndexEntry.from_json(json.loads(line))
                entries[entry.key] = entry
    return entries


def append_index(folder: str | Path, entry: IndexEntry) -> None:
    """Add one entry to a frame folder's index, creating the file when needed."""
    path = Path(folder) / INDEX_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry.to_json(), ensure_ascii=False) + "\n")


class FrameMissing(LookupError):
    """A frame folder holds no usable frame for a row."""


class FrameStore:
    """The frames in a folder written by ``cardvision fetch``, looked up by manifest row.

    A folder without an index also works when its files carry the names ``frame_file_name``
    gives, directly in the folder or in its ``frames`` subfolder.
    """

    _SUFFIXES = (".jpg", ".png")

    def __init__(self, folder: str | Path, tolerance: float = RATIO_TOLERANCE) -> None:
        self.folder = Path(folder)
        self.tolerance = tolerance
        self._index = read_index(self.folder)

    def path(self, row: Row) -> Path:
        """Where the row's frame is. Raises ``FrameMissing`` when there is none."""
        entry = self._index.get(row.key)
        if entry is not None:
            if entry.status != "ok" or entry.file is None:
                raise FrameMissing(f"{row.platform} {row.video_id} t={row.t}: {entry.status} ({entry.reason})")
            path = self.folder / entry.file
            if not path.is_file():
                raise FrameMissing(f"{row.platform} {row.video_id} t={row.t}: the index names {entry.file}, which is absent")
            return path
        for suffix in self._SUFFIXES:
            for base in (self.folder / FRAMES_FOLDER, self.folder):
                path = base / frame_file_name(row, suffix)
                if path.is_file():
                    return path
        raise FrameMissing(f"{row.platform} {row.video_id} t={row.t}: no frame in {self.folder}")

    def has(self, row: Row) -> bool:
        try:
            self.path(row)
        except FrameMissing:
            return False
        return True

    def load(self, row: Row) -> np.ndarray:
        """The row's frame as an ``(H, W, 3)`` uint8 RGB array.

        Raises ``FrameMissing`` when there is none, or when its shape fails ``frame_check``.
        """
        from cardvision.core.images import load_rgb

        rgb = load_rgb(self.path(row))
        problem = frame_check(row, rgb.shape[1], rgb.shape[0], self.tolerance)
        if problem is not None:
            raise FrameMissing(f"{row.platform} {row.video_id} t={row.t}: {problem}")
        return rgb

    def frames(self, rows) -> Iterator[tuple[Row, np.ndarray]]:
        """``(row, frame)`` for every row that has a usable frame; the others are skipped."""
        for row in rows:
            try:
                yield row, self.load(row)
            except FrameMissing:
                continue
