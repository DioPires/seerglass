# SPDX-License-Identifier: Apache-2.0
"""Obtain the frames a manifest's rows describe, following the extraction contract.

A frame comes from a video file. The file is either one the user already has (``video_files``)
or, when the user asks for it, one fetched by the optional downloader. Videos are handled one
at a time, never in parallel, and a downloaded file is deleted once its frames are read, unless
the user asks to keep it.

**Recordings of your own.** A row whose platform is ``local`` names a recording that is on no
platform. It has no address and is never downloaded: its frames come from a file the user
gives, and from nowhere else.

**Kept video files.** With ``keep_video``, each downloaded file is moved to that folder in
place of being deleted. Its name says which video it is and where it starts in that video
(``kept_video_name``):

- ``<platform>-<video id>+<start>.<ext>`` for a section, for example
  ``youtube-EXAMPLE0001+77.mp4``. ``<start>`` is the time in the video, in seconds, at which
  the file starts: a whole number of seconds is written without a decimal point, any other time
  to the millisecond at most (``12.5``).
- ``<platform>-<video id>.<ext>`` for a whole video.

The folder also gets an index, ``video-files.json`` (``KEPT_INDEX_NAME``): one entry per kept
file, with the platform, the video id, the section's start and end in seconds of the video
(both ``null`` for a whole video) and the times ``t`` of the rows the file serves. ``start`` is
the offset this module itself reads the file with: a frame at time ``x`` of the file is the
frame at ``start + x`` of the video.

A kept file is also read again. With ``keep_video``, a row that a file in that folder covers
gets its frame from the file, and nothing is asked of the platform for it (``kept_file_for``).
A whole-video file covers every row of its video. A section the folder's index lists covers
the rows whose times the index lists for it. A section file the index does not list is known
by its name alone, and covers the rows at or after its start, up to the next such section.

Each row's outcome is appended to the output folder's index as soon as it is known, so a run
that is interrupted, or that lacked some videos, can be repeated: rows that already have a
frame are skipped.

**When a download fails.** The index records why as one word (``failure_cause``): the video is
``unavailable``, it is ``restricted``, or the failure says nothing about the video (``error``).
A part of a video that fails with the cause ``error`` is asked for once more after the usual
pause (``retries``), and the index records how many times it was asked for. A failure with the
cause ``unavailable`` or ``restricted`` is the platform's answer about that video, and the part
is not asked for again.

**When the platform refuses.** A rate limit, or a check that the client is not a bot, is not a
failed row: asking again makes it worse. The run stops at once (``REFUSAL_PATTERNS``,
``DownloadRefused``). The rows already handled keep their index lines, the rows of the refused
part get none, and the summary says how many rows were not tried.

**When downloads fail one after another.** A part of a video is cut by a helper program, and
when the platform's media server turns that program away, all that comes back is its exit
code: a rate limit cannot be read from it. So the run also stops when several parts in a row
have failed for no stated reason (``stop_after_failures``). Their rows keep their ``failed``
lines, and the rows after them were not tried.

The downloader is an optional dependency (``python -m pip install -e ".[download]"`` in the
repository folder) and is not imported until a download is asked for.
"""

from __future__ import annotations

import datetime
import importlib.util
import json
import math
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from cardvision.core.tls import CERTIFICATE_HELP, CertificatesMissing, is_certificate_failure
from cardvision.data.dhash import SAME_FRAME_MAX_HAMMING
from cardvision.data.extract import (
    FINGERPRINT_WINDOW_SECONDS,
    FORMAT_SELECTOR,
    FRAMES_FOLDER,
    RATIO_TOLERANCE,
    FrameNotFound,
    IndexEntry,
    VideoReader,
    VideoSupportMissing,
    append_index,
    frame_check,
    frame_file_name,
    index_time,
    nearest_frame,
    read_index,
)
from cardvision.data.schema import LOCAL_PLATFORM, Row

DOWNLOAD_NOTICE = (
    "Downloading is your own act. Nothing is fetched unless you pass --download, and then the\n"
    "platform is asked for one video at a time. The platform's terms of service apply to you and\n"
    "may restrict downloading and automated access; the videos remain the property of whoever\n"
    "holds their rights. If in doubt, use video files you already have (--video-file ID=PATH)."
)

VIDEO_URLS = {
    "youtube": "https://www.youtube.com/watch?v={video_id}",
    "twitch": "https://www.twitch.tv/videos/{video_id}",
}

# Why a row is left without a frame when no file serves its video.
NO_VIDEO_REASON = "no video file was given for this video and --download was not asked for"
NO_LOCAL_VIDEO_REASON = ('no video file was given for this recording of your own (platform "local"), which is '
                         "never downloaded: pass --video-file ID=PATH")

IMAGE_FORMATS = {"jpg": ".jpg", "png": ".png"}
JPEG_QUALITY = 92

# A downloaded section runs from this long before the earliest wanted time to this long after
# the latest, and wanted times closer together than JOIN_SECONDS share a section.
SECTION_BEFORE_SECONDS = 3.0
SECTION_AFTER_SECONDS = 3.0
JOIN_SECONDS = 60.0

DEFAULT_PAUSE_SECONDS = 5.0

# A part of a video whose download fails is asked for this many more times.
DEFAULT_RETRIES = 1

# The run also stops when this many parts in a row have failed for no stated reason. 0: never.
DEFAULT_STOP_AFTER_FAILURES = 3

# The downloader's signs that the platform refused the request, as a rate limit, as a check
# that the client is not a bot, or as a blocked session: an HTTP 429, the downloader's own
# words for a session that was rate-limited, for a captcha and for a blocked address, the bot
# check's wording, and a redirect to a "sorry" page. Each is a regular expression searched for
# in the error's text, whatever the case. They are narrow on purpose:
#
# - the downloader ends many ordinary errors with stock advice ("Confirm you are on the latest
#   version"), and that must not be read as a refusal;
# - "sign in to confirm" alone is not a sign. The bot check says "Sign in to confirm you're
#   not a bot", and the age gate of a single video says "Sign in to confirm your age".
REFUSAL_PATTERNS = (
    r"HTTP Error 429",
    r"HTTPError 429",
    r"Too Many Requests",
    r"not a bot",
    r"rate-limited",
    r"isn.t available, try again later",
    r"captcha",
    r"likely being blocked",
    r"/sorry/",
)

# Why a download failed, as one word for the index line (its ``cause``). ``record-reachable``
# reads it: only ``unavailable`` says that the video could not be found.
CAUSE_UNAVAILABLE = "unavailable"   # the platform says the video is private, removed or does not exist
CAUSE_RESTRICTED = "restricted"     # the video is there, and this request may not have it
CAUSE_REFUSED = "refused"           # the platform refused the request; no line is written for it
CAUSE_ERROR = "error"               # anything else: the network, the cutting program, a timeout
CAUSES = (CAUSE_UNAVAILABLE, CAUSE_RESTRICTED, CAUSE_REFUSED, CAUSE_ERROR)

# The video is there, and whoever asks may not have it: an age gate, a video for a channel's
# members or subscribers, a video kept out of a country, a protected stream. Regular
# expressions, searched for like the ones above.
RESTRICTED_PATTERNS = (
    r"confirm your age",
    r"age-restricted",
    r"inappropriate for some users",
    r"members-only",
    r"subscriber-only",
    r"in your country",
    r"from your location",
    r"geo restriction",
    r"DRM protected",
)

# The platform says the video cannot be had by anybody: it is private, it was removed, its
# account was closed, or there is no such video. The last pattern is the platform's bare
# statement with nothing after it; followed by another reason it proves nothing by itself.
UNAVAILABLE_PATTERNS = (
    r"private video",
    r"video is private",
    r"video( \S+)? does not exist",
    r"video has been removed",
    r"video is no longer available",
    r"account associated with this video has been terminated",
    r"(video unavailable|this video is unavailable|this video is not available)\.?\s*$",
)

# A downloader's error is kept as the start of its first line, where the cause is stated, and
# its end, each this many characters at most.
ERROR_HEAD_CHARACTERS = 300
ERROR_TAIL_CHARACTERS = 300

# The index of a folder of kept video files.
KEPT_INDEX_NAME = "video-files.json"
KEPT_INDEX_KIND = "cardvision-video-files"
KEPT_INDEX_VERSION = 1


class DownloadSupportMissing(RuntimeError):
    """A download was asked for but the optional downloader is not installed."""


class DownloadFailed(RuntimeError):
    """The downloader could not produce a video file.

    ``cause`` says why in one word, one of ``CAUSES``, when whoever raised the error knew the
    downloader's whole text; ``None`` leaves it to be read from the message (``failure_cause``).
    """

    def __init__(self, message: str, cause: str | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class DownloadRefused(DownloadFailed):
    """The platform refused the request: a rate limit, or a check that the client is not a bot.
    No further request should follow it."""

    def __init__(self, message: str, cause: str | None = CAUSE_REFUSED) -> None:
        super().__init__(message, cause)


class VideoTooLong(DownloadFailed):
    """The video is longer than the longest whole video the run was told to download. Nothing was
    downloaded. Its ``cause`` is ``error``: the length says nothing about whether the video is there."""

    def __init__(self, message: str) -> None:
        super().__init__(message, CAUSE_ERROR)


def _first_sign(patterns: Sequence[str], text: str) -> str | None:
    for pattern in patterns:
        found = re.search(pattern, text, re.IGNORECASE)
        if found is not None:
            return found.group(0)
    return None


def refusal_sign(text: str) -> str | None:
    """The words in a downloader's error that say the platform refused the request, or ``None``
    when the error is an ordinary one (``REFUSAL_PATTERNS``)."""
    return _first_sign(REFUSAL_PATTERNS, text)


def failure_cause(text: str) -> str:
    """Why a download failed, as one of ``CAUSES``, read from the downloader's error.

    A refusal comes first. Then a restriction, because the platform puts "Video unavailable"
    in front of some of them ("... not made this video available in your country"). Then the
    words that say the video cannot be had by anybody. Anything else is ``error``: a failure
    that says nothing about the video.
    """
    if refusal_sign(text) is not None:
        return CAUSE_REFUSED
    if _first_sign(RESTRICTED_PATTERNS, text.strip()) is not None:
        return CAUSE_RESTRICTED
    if _first_sign(UNAVAILABLE_PATTERNS, text.strip()) is not None:
        return CAUSE_UNAVAILABLE
    return CAUSE_ERROR


def error_summary(text: str) -> str:
    """A downloader's error, short enough for an index line: all of it when it is short, else
    the start of its first line, where the cause is stated, and its end."""
    text = text.strip()
    if len(text) <= ERROR_HEAD_CHARACTERS + ERROR_TAIL_CHARACTERS:
        return text
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return f"{first[:ERROR_HEAD_CHARACTERS]} [...] {text[-ERROR_TAIL_CHARACTERS:]}"


@dataclass(frozen=True)
class Download:
    """A video file the downloader produced.

    Attributes:
        path: the file.
        offset: the time in the original video at which the file starts, in seconds.
    """

    path: Path
    offset: float = 0.0


# (platform, video id, section start or None for the whole video, section end, folder) -> file
Downloader = Callable[[str, str, "float | None", "float | None", Path], Download]


def video_url(platform: str, video_id: str) -> str:
    """The address of a video on its platform. Raises ``ValueError`` for a recording of your
    own (platform ``local``), which has none."""
    if platform == LOCAL_PLATFORM:
        raise ValueError(f"{platform} {video_id}: a recording of your own is on no platform and has no address; "
                         f"give its file with --video-file {video_id}=PATH")
    return VIDEO_URLS[platform].format(video_id=video_id)


def downloader_installed() -> bool:
    """Whether the optional downloader can be imported."""
    return importlib.util.find_spec("yt_dlp") is not None


def sections_supported() -> bool:
    """Whether a part of a video can be downloaded on its own, which needs ``ffmpeg``."""
    return shutil.which("ffmpeg") is not None


def download_video(platform: str, video_id: str, start: float | None, end: float | None, folder: Path,
                   max_minutes: float | None = None) -> Download:
    """Fetch a video, or the stretch of it from ``start`` to ``end`` seconds, into ``folder``.

    The stream is chosen by ``FORMAT_SELECTOR``. A stretch is cut precisely, by re-encoding at
    the cut, and the returned ``offset`` is its start. Raises ``DownloadSupportMissing`` when
    the downloader is not installed, ``DownloadFailed`` when no file results, its subclass
    ``DownloadRefused`` when the error says the platform refused the request, and
    ``CertificatesMissing`` when the server's certificate could not be verified. The error
    carries its ``cause``, read from the downloader's whole text before it is shortened.

    With ``max_minutes``, a video that the downloader reports as longer is not downloaded and
    ``VideoTooLong`` is raised. The length comes from the information the downloader reads
    about the video before it downloads anything, so the limit costs no request of its own. A
    video whose length is not reported is downloaded.
    """
    try:
        import yt_dlp
    except ImportError:
        raise DownloadSupportMissing(
            'downloading needs an optional dependency: python -m pip install -e ".[download]" in the repository folder'
        ) from None
    folder.mkdir(parents=True, exist_ok=True)
    options: dict = {
        "format": FORMAT_SELECTOR,
        "outtmpl": str(folder / "video.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
    }
    too_long: list[float] = []
    if max_minutes is not None:
        def match_filter(info: Mapping, *, incomplete: bool = False) -> str | None:
            length = info.get("duration")
            if isinstance(length, (int, float)) and not isinstance(length, bool) and length > max_minutes * 60:
                too_long.append(float(length))
                return f"the video is longer than {max_minutes:g} minutes"
            return None

        options["match_filter"] = match_filter
    if start is not None and end is not None:
        options["download_ranges"] = yt_dlp.utils.download_range_func(None, [(start, end)])
        options["force_keyframes_at_cuts"] = True
    try:
        with yt_dlp.YoutubeDL(options) as session:
            session.download([video_url(platform, video_id)])
    except Exception as error:  # the downloader raises its own error types
        text = str(error)
        if is_certificate_failure(text):
            raise CertificatesMissing(f"{platform} {video_id}: {CERTIFICATE_HELP}") from None
        cause = failure_cause(text)
        kind = DownloadRefused if cause == CAUSE_REFUSED else DownloadFailed
        raise kind(f"{platform} {video_id}: {error_summary(text)}", cause) from None
    if too_long:
        raise VideoTooLong(f"{platform} {video_id}: the video is {math.ceil(too_long[0] / 60)} minutes long, "
                           f"longer than the limit of {max_minutes:g} minutes (--max-whole-minutes); it was not downloaded")
    files = sorted(path for path in folder.iterdir() if path.is_file() and path.suffix not in (".part", ".ytdl", ".json"))
    if not files:
        raise DownloadFailed(f"{platform} {video_id}: the downloader produced no file", CAUSE_ERROR)
    return Download(path=files[0], offset=float(start or 0.0))


def plan_sections(
    times: Iterable[float],
    before: float = SECTION_BEFORE_SECONDS,
    after: float = SECTION_AFTER_SECONDS,
    join: float = JOIN_SECONDS,
) -> list[tuple[float, float, list[float]]]:
    """Group wanted times into the stretches of video to download.

    Returns ``(start, end, the times it serves)`` for each stretch, in order. A stretch starts
    on a whole second at least ``before`` seconds ahead of its first time and ends on a whole
    second at least ``after`` seconds past its last; times closer together than ``join``
    seconds share one.
    """
    sections: list[tuple[float, float, list[float]]] = []
    for t in sorted(set(times)):
        if sections and t - sections[-1][2][-1] <= join:
            start, _, served = sections[-1]
            sections[-1] = (start, float(math.ceil(t + after)), [*served, t])
        else:
            sections.append((float(max(0, math.floor(t - before))), float(math.ceil(t + after)), [t]))
    return sections


# --------------------------------------------------------------------------- kept video files
def seconds_text(seconds: float) -> str:
    """A time as a kept file's name writes it: a whole number of seconds without a decimal
    point (``77``), any other time to the millisecond at most (``12.5``, ``0.033``)."""
    if seconds < 0 or seconds != seconds:
        raise ValueError(f"a time in a video is 0 or more, got {seconds}")
    text = f"{seconds:.3f}".rstrip("0").rstrip(".")
    return text or "0"


def kept_video_name(platform: str, video_id: str, start: float | None, suffix: str) -> str:
    """The name a downloaded file is kept under.

    ``start`` is the time in the video at which the file starts, or ``None`` for a whole video:
    ``youtube-EXAMPLE0001+77.mp4`` and ``youtube-EXAMPLE0001.mp4``. ``suffix`` is the file's
    extension with its dot.
    """
    section = "" if start is None else f"+{seconds_text(start)}"
    return f"{platform}-{video_id}{section}{suffix}"


@dataclass(frozen=True)
class KeptVideo:
    """One kept video file, as the index of its folder lists it.

    Attributes:
        file: the file's name in the folder.
        platform, video_id: the video.
        start: the time in the video at which the file starts, in seconds; ``None`` for a
            whole video.
        end: the time in the video up to which the section was asked for; ``None`` for a
            whole video.
        rows: the times ``t`` of the rows the file serves, in ascending order.
    """

    file: str
    platform: str
    video_id: str
    start: float | None
    end: float | None
    rows: tuple[float, ...] = ()

    def to_json(self) -> dict:
        return {"file": self.file, "platform": self.platform, "video_id": self.video_id,
                "start": self.start, "end": self.end, "rows": list(self.rows)}

    @classmethod
    def from_json(cls, doc: Mapping) -> KeptVideo:
        start, end = doc["start"], doc["end"]
        return cls(file=doc["file"], platform=doc["platform"], video_id=doc["video_id"],
                   start=None if start is None else float(start), end=None if end is None else float(end),
                   rows=tuple(float(t) for t in doc["rows"]))


def read_kept_videos(folder: str | Path) -> dict[str, KeptVideo]:
    """The index of a folder of kept video files, by file name. A folder without an index
    gives an empty mapping. Raises ``ValueError`` when the index is not one this package wrote."""
    path = Path(folder) / KEPT_INDEX_NAME
    if not path.is_file():
        return {}
    refusal = f"{path}: not an index of kept video files (kind {KEPT_INDEX_KIND}, version {KEPT_INDEX_VERSION})"
    try:
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
        if not isinstance(doc, dict) or doc.get("kind") != KEPT_INDEX_KIND or doc.get("version") != KEPT_INDEX_VERSION:
            raise ValueError(refusal)
        return {entry["file"]: KeptVideo.from_json(entry) for entry in doc["files"]}
    except (KeyError, TypeError, ValueError):
        raise ValueError(refusal) from None


def write_kept_videos(folder: str | Path, kept: Mapping[str, KeptVideo]) -> Path:
    """Write the index of a folder of kept video files, in the order of video and start."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    entries = sorted(kept.values(), key=lambda entry: (entry.platform, entry.video_id, entry.start is not None, entry.start or 0.0))
    doc = {"kind": KEPT_INDEX_KIND, "version": KEPT_INDEX_VERSION, "files": [entry.to_json() for entry in entries]}
    path = folder / KEPT_INDEX_NAME
    handle, temporary = tempfile.mkstemp(dir=folder, prefix=".partial-", suffix=".json")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump(doc, out, indent=1)
            out.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return path


def keep_video_file(folder: str | Path, result: Download, platform: str, video_id: str, start: float | None,
                    end: float | None, rows: Iterable[float]) -> KeptVideo:
    """Move a downloaded file into ``folder`` under its kept name and list it in the index.

    ``start`` and ``end`` are what the downloader was asked for; ``None`` twice means the whole
    video. The start that is recorded, and written in the name, is ``result.offset``: the offset
    the file is read with. A file kept earlier under the same name is replaced. Its rows stay
    listed when their times lie inside the new file.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    whole = start is None or end is None
    offset = None if whole else float(result.offset)
    name = kept_video_name(platform, video_id, offset, result.path.suffix)
    shutil.move(str(result.path), str(folder / name))
    index = read_kept_videos(folder)
    served = {float(t) for t in rows}
    earlier = index.get(name)
    if earlier is not None:
        served.update(t for t in earlier.rows if whole or offset <= t <= end)
    entry = KeptVideo(file=name, platform=platform, video_id=video_id, start=offset, end=None if whole else float(end),
                      rows=tuple(sorted(served)))
    index[name] = entry
    write_kept_videos(folder, index)
    return entry


# Files in a folder of kept video files that are never a video: the index, and what an
# interrupted download leaves behind.
_NOT_A_KEPT_VIDEO = (".json", ".part", ".ytdl")
_SECTION_START = re.compile(r"^[0-9]+(\.[0-9]+)?$")


def kept_videos_of(folder: str | Path, platform: str, video_id: str) -> list[KeptVideo]:
    """The files of one video in a folder of kept video files, each as an index entry.

    A file the folder's index lists is described by its entry. Any other file is known by its
    name (``kept_video_name``): ``<platform>-<video id>.<ext>`` is the whole video, and
    ``<platform>-<video id>+<start>.<ext>`` a section that starts there; its ``end`` is not
    known and is given as ``None``, and it lists no rows. Only files that exist are returned.
    A folder that does not exist gives an empty list.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return []
    index = read_kept_videos(folder)
    found = [entry for entry in index.values()
             if (entry.platform, entry.video_id) == (platform, video_id) and (folder / entry.file).is_file()]
    stem = f"{platform}-{video_id}"
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.name in index or path.name.startswith(".") or path.suffix.lower() in _NOT_A_KEPT_VIDEO:
            continue
        if path.stem == stem:
            found.append(KeptVideo(file=path.name, platform=platform, video_id=video_id, start=None, end=None))
        elif path.stem.startswith(stem + "+") and _SECTION_START.fullmatch(path.stem[len(stem) + 1:]):
            found.append(KeptVideo(file=path.name, platform=platform, video_id=video_id,
                                   start=float(path.stem[len(stem) + 1:]), end=None))
    return found


def kept_file_for(kept: Sequence[KeptVideo], t: float, whole_only: bool = False) -> KeptVideo | None:
    """The kept file of a video that covers the row at time ``t``, or ``None``.

    In this order: a whole-video file; a section the index lists the row's time for; a section
    known by its name alone, the one that starts last at or before ``t`` (whether it reaches
    ``t`` is found out when it is read). With ``whole_only``, sections are not used.
    """
    for entry in kept:
        if entry.start is None:
            return entry
    if whole_only:
        return None
    for entry in kept:
        if entry.end is not None and t in entry.rows:
            return entry
    named = [entry for entry in kept if entry.end is None and entry.start is not None and entry.start <= t]
    return max(named, key=lambda entry: entry.start, default=None)


def has_kept_videos(folder: str | Path) -> bool:
    """Whether a folder holds a kept video file: one its index lists, or one with the name of
    a kept file."""
    folder = Path(folder)
    if not folder.is_dir():
        return False
    if any((folder / name).is_file() for name in read_kept_videos(folder)):
        return True
    return any(path.is_file() and not path.name.startswith(".") and path.suffix.lower() not in _NOT_A_KEPT_VIDEO
               and re.match(r"^[a-z]+-.+", path.stem) for path in folder.iterdir())


def parse_video_files(entries: Sequence[str]) -> dict[str, Path]:
    """``ID=PATH`` arguments as a mapping. ``ID`` is a video id, or ``platform:id`` when the
    same id exists on two platforms. Raises ``ValueError`` for a malformed entry or a missing
    file."""
    out: dict[str, Path] = {}
    for entry in entries:
        name, separator, location = entry.partition("=")
        if not separator or not name or not location:
            raise ValueError(f"{entry!r}: expected ID=PATH")
        path = Path(location).expanduser()
        if not path.is_file():
            raise ValueError(f"{entry!r}: no such file {path}")
        out[name] = path
    return out


def parse_row_selection(text: str, count: int) -> list[int]:
    """Row numbers (from 1) named by a selection such as ``1-20,35`` as indices from 0, in
    ascending order. Raises ``ValueError`` for anything malformed or out of range."""
    chosen: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"{text!r}: an empty entry in the row selection")
        first, separator, last = part.partition("-")
        try:
            low = int(first)
            high = int(last) if separator else low
        except ValueError:
            raise ValueError(f"{part!r}: expected a row number or a range such as 3-8") from None
        if low < 1 or high < low or high > count:
            raise ValueError(f"{part!r}: rows are numbered 1 to {count}")
        chosen.update(range(low - 1, high))
    return sorted(chosen)


@dataclass
class FetchSummary:
    """What a fetch did.

    Attributes:
        entries: the outcome of every row handled in this run, in order.
        skipped: rows that already had a frame and were left alone.
        kept: the downloaded files that were kept in this run, in order.
        kept_used: the names of the files already in the folder of kept files that were read
            in this run, in order.
        refused: the downloader's words when the platform refused a download and the run
            stopped; ``None`` otherwise.
        stopped: how many parts in a row had failed for no stated reason when the run stopped
            for that; ``None`` otherwise.
        remaining: rows that were not tried because the run stopped.
    """

    entries: list[IndexEntry] = field(default_factory=list)
    skipped: int = 0
    kept: list[KeptVideo] = field(default_factory=list)
    kept_used: list[str] = field(default_factory=list)
    refused: str | None = None
    stopped: int | None = None
    remaining: int = 0

    def count(self, status: str) -> int:
        return sum(1 for entry in self.entries if entry.status == status)

    @property
    def stopped_early(self) -> bool:
        """Whether the run stopped before its end, so that some rows were not tried."""
        return self.refused is not None or self.stopped is not None

    @property
    def complete(self) -> bool:
        """Whether the run went to its end and every row handled now has a frame."""
        return not self.stopped_early and all(entry.status == "ok" for entry in self.entries)

    def other_sizes(self) -> dict[tuple[int, int, int, int], int]:
        """How many frames were written at another size than their row records, by
        ``(frame width, frame height, the row's width, the row's height)``."""
        counts: dict[tuple[int, int, int, int], int] = {}
        for entry in self.entries:
            if entry.status == "ok" and entry.other_size and entry.scale:
                size = (entry.width, entry.height, round(entry.width / entry.scale), round(entry.height / entry.scale))
                counts[size] = counts.get(size, 0) + 1
        return counts


def _save(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".partial-", suffix=path.suffix)
    os.close(handle)
    try:
        image = Image.fromarray(rgb)
        if path.suffix == ".jpg":
            image.save(temporary, format="JPEG", quality=JPEG_QUALITY)
        else:
            image.save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _local_file(row: Row, video_files: Mapping[str, Path]) -> Path | None:
    return video_files.get(f"{row.platform}:{row.video_id}") or video_files.get(row.video_id)


def fetch_frames(
    rows: Sequence[Row],
    out: str | Path,
    *,
    video_files: Mapping[str, Path] | None = None,
    download: bool = False,
    downloader: Downloader | None = None,
    whole_video: bool = False,
    keep_video: str | Path | None = None,
    image_format: str = "jpg",
    exact_seek: bool = False,
    tolerance: float = RATIO_TOLERANCE,
    pause: float = DEFAULT_PAUSE_SECONDS,
    retries: int = DEFAULT_RETRIES,
    stop_after_failures: int = DEFAULT_STOP_AFTER_FAILURES,
    max_whole_minutes: float | None = None,
    say: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime.datetime] | None = None,
) -> FetchSummary:
    """Obtain the frame of every row and write it under ``out``.

    Args:
        rows: the rows wanted.
        out: the output folder. Frames go to ``out/frames`` and the index to ``out/index.jsonl``.
        video_files: local video files by video id (or ``platform:id``); each must hold the
            whole video, starting at its beginning.
        download: fetch the videos that have no local file. Off unless the user asks. It never
            applies to a recording of your own (platform ``local``): such a row is served from
            ``video_files`` or from a kept file, or left without a frame.
        downloader: the function that fetches (``download_video`` when omitted).
        whole_video: download each video whole instead of the stretches around the wanted
            times. Whole videos give exactly the contract's frame; stretches are lighter and
            can be one frame off, because their cut is only as precise as a frame.
        keep_video: a folder of kept video files. A row that a file there covers is read from
            it and not downloaded (``kept_file_for``; with ``whole_video``, whole-video files
            only). Every file downloaded in this run is moved there in place of being deleted,
            under the name ``kept_video_name`` gives, and listed in the folder's index
            (``KEPT_INDEX_NAME``). Files given in ``video_files`` are not copied.
        image_format: ``jpg`` or ``png``.
        exact_seek: read each file from its start, frame by frame, instead of seeking. Slow
            on a long file; the frames are the same either way.
        tolerance: the shape tolerance of ``frame_check``.
        pause: seconds to wait between two downloads.
        retries: how many more times a part of a video is asked for when its download fails
            for a reason that says nothing about the video (cause ``error``). A refusal by the
            platform is never tried again, and neither is a part of a video that the platform
            says is ``unavailable`` or ``restricted``: the later parts of that video are not
            asked for either, and their rows are ``failed`` with the same cause, with ``attempts`` 0.
        stop_after_failures: stop the run when this many parts in a row have failed, each
            after its retries, for no stated reason (cause ``error``). 0 never stops. A part
            that downloads sets the count back to nothing. A failure with a stated reason (the
            video is ``unavailable`` or ``restricted``) is the platform's answer about one
            video: it neither counts nor sets the count back.
        max_whole_minutes: do not download a whole video that the downloader reports as longer
            than this many minutes. Its rows are ``failed`` with the cause ``error`` and a reason
            that says so; it is not tried again and does not count towards ``stop_after_failures``.
            Applies to a whole video (``whole_video``, or when sections cannot be cut); the
            downloader is then called with ``max_minutes=`` as well. ``None``: no limit.
        say: where progress lines go.
        sleep: the function that waits (replaced in tests).
        now: the clock that dates each index line, in UTC (replaced in tests).

    A row with a ``frame_dhash`` is matched by content: the frame used is the one, within a
    second either side of ``t``, whose fingerprint is nearest, if it is within
    ``SAME_FRAME_MAX_HAMMING`` bits; otherwise the row is ``unusable`` with reason
    ``fingerprint`` and no frame is written. The index records the distance and the time offset.
    Where that second lies wholly past the end of the file there is no frame to match, and the
    row is ``failed``: the file's last frame is not compared with it.

    A frame of the row's shape and another size is written and marked ``other_size`` in the
    index. When the platform refuses a download, or several parts in a row fail, the run stops:
    ``refused`` or ``stopped``, and ``remaining``, of the summary say so.

    Every index line records when it was written, and a failed download records its ``cause``.

    Raises ``VideoSupportMissing`` when a video has to be read and OpenCV is not installed, and
    ``CertificatesMissing`` when a download fails because the server's certificate could not
    be verified.
    """
    if image_format not in IMAGE_FORMATS:
        raise ValueError(f"unknown image format {image_format!r}; choose one of {', '.join(IMAGE_FORMATS)}")
    if retries < 0:
        raise ValueError(f"the number of retries is 0 or more, got {retries}")
    if max_whole_minutes is not None and not max_whole_minutes > 0:
        raise ValueError(f"the longest whole video is more than 0 minutes, got {max_whole_minutes}")
    if stop_after_failures < 0:
        raise ValueError(f"the number of failures in a row that stops the run is 0 or more, got {stop_after_failures}")
    out = Path(out)
    suffix = IMAGE_FORMATS[image_format]
    video_files = dict(video_files or {})
    summary = FetchSummary()
    index = read_index(out)

    pending: dict[tuple[str, str], list[Row]] = {}
    for row in rows:
        entry = index.get(row.key)
        if entry is not None and entry.status == "ok" and entry.file and (out / entry.file).is_file():
            summary.skipped += 1
            continue
        pending.setdefault((row.platform, row.video_id), []).append(row)

    def record(row: Row, status: str, **details) -> None:
        if status == "failed" and details.get("source") == "download":
            details.setdefault("cause", CAUSE_ERROR)     # the file came, and could not be read or ends early
        entry = IndexEntry(platform=row.platform, video_id=row.video_id, t=row.t, status=status,
                           written=index_time(now() if now is not None else None), **details)
        append_index(out, entry)
        summary.entries.append(entry)

    def outcome(reader: VideoReader, row: Row) -> tuple[str, dict]:
        """What the file gives for one row: a status and the details of its index line. The
        frame is written when the status is ``ok``."""
        fingerprint = row.attributes.get("frame_dhash")
        distance = None
        try:
            if fingerprint is None:
                frame = reader.frame_at(row.t)
            else:
                window = reader.frames_between(row.t - FINGERPRINT_WINDOW_SECONDS, row.t + FINGERPRINT_WINDOW_SECONDS)
                height, width = window[0].rgb.shape[:2]
                problem = frame_check(row, width, height, tolerance)
                if problem is not None:
                    return "unusable", {"width": width, "height": height, "frame_time": round(window[0].time, 3),
                                        "reason": problem}
                frame, distance = nearest_frame(window, fingerprint, row.t)
                if distance > SAME_FRAME_MAX_HAMMING:
                    return "unusable", {"width": width, "height": height, "reason": "fingerprint", "distance": distance}
        except FrameNotFound as error:
            return "failed", {"reason": str(error)}
        height, width = frame.rgb.shape[:2]
        problem = frame_check(row, width, height, tolerance)
        if problem is not None:
            return "unusable", {"width": width, "height": height, "frame_time": round(frame.time, 3), "reason": problem}
        relative = f"{FRAMES_FOLDER}/{frame_file_name(row, suffix)}"
        _save(frame.rgb, out / relative)
        found = {} if distance is None else {"distance": distance, "offset": round(frame.time - row.t, 3)}
        if (width, height) != (row.frame_width, row.frame_height):
            found["other_size"] = True
        return "ok", {"file": relative, "width": width, "height": height, "frame_time": round(frame.time, 3),
                      "scale": round(width / row.frame_width, 4), **found}

    def read_rows(reader: VideoReader, wanted: Sequence[Row], source: str, **details) -> None:
        for row in wanted:
            status, found = outcome(reader, row)
            record(row, status, source=source, **found, **details)

    def read_kept(wanted: Sequence[Row], may_download: bool) -> list[Row]:
        """Read the rows that a kept file covers; return the rows still without a frame.

        A row whose kept file does not give its frame goes on to the download when there is
        one to go on to. Otherwise its outcome is recorded here, with one exception: a section
        known by its name alone was only a guess at covering the row, so a row it does not
        reach is handed back as a row that no kept file covers."""
        platform, video_id = wanted[0].platform, wanted[0].video_id
        kept = kept_videos_of(keep_video, platform, video_id)
        by_file: dict[str, tuple[KeptVideo, list[Row]]] = {}
        left: list[Row] = []
        for row in wanted:
            entry = kept_file_for(kept, row.t, whole_only=whole_video)
            if entry is None:
                left.append(row)
            else:
                by_file.setdefault(entry.file, (entry, []))[1].append(row)
        for entry, served in by_file.values():
            say(f"{platform} {video_id}: {len(served)} frame(s) from the kept file {entry.file}")
            summary.kept_used.append(entry.file)
            guessed = entry.start is not None and entry.end is None      # a section known by its name alone
            try:
                with VideoReader(Path(keep_video) / entry.file, offset=entry.start or 0.0, exact=exact_seek) as reader:
                    results = [(row, *outcome(reader, row)) for row in served]
            except FrameNotFound as error:
                results = [(row, "failed", {"reason": str(error)}) for row in served]
            again = 0
            for row, status, found in results:
                if status == "ok" or not (may_download or (guessed and status == "failed")):
                    record(row, status, source="kept", **found)
                else:
                    left.append(row)
                    again += 1
            if again and may_download:
                say(f"{platform} {video_id}: the kept file {entry.file} did not give the frame of {again} row(s); "
                    "they are downloaded")
            elif again:
                say(f"{platform} {video_id}: the kept file {entry.file} does not reach the frame of {again} row(s)")
        return [row for row in wanted if row in left]

    downloads_made = 0
    failures_in_a_row = 0
    for (platform, video_id), wanted in pending.items():
        local = _local_file(wanted[0], video_files)
        if local is not None:
            say(f"{platform} {video_id}: {len(wanted)} frame(s) from {local}")
            try:
                with VideoReader(local, exact=exact_seek) as reader:
                    read_rows(reader, wanted, "local")
            except FrameNotFound as error:
                for row in wanted:
                    record(row, "failed", source="local", reason=str(error))
            continue
        may_download = download and platform != LOCAL_PLATFORM
        if keep_video is not None:
            wanted = read_kept(wanted, may_download)
            if not wanted:
                continue
        if platform == LOCAL_PLATFORM:
            for row in wanted:
                record(row, "no-video", reason=NO_LOCAL_VIDEO_REASON)
            say(f"{platform} {video_id}: no video file given; pass --video-file {video_id}=PATH; "
                f"{len(wanted)} row(s) left without a frame")
            continue
        if not download:
            for row in wanted:
                record(row, "no-video", reason=NO_VIDEO_REASON)
            say(f"{platform} {video_id}: no video file given; {len(wanted)} row(s) left without a frame")
            continue

        fetch = downloader or download_video
        by_time: dict[float, list[Row]] = {}
        for row in wanted:
            by_time.setdefault(row.t, []).append(row)
        if whole_video or not sections_supported():
            stretches: list[tuple[float | None, float | None, list[float]]] = [(None, None, sorted(by_time))]
        else:
            stretches = list(plan_sections(by_time))
        say(f"{platform} {video_id}: downloading {len(stretches)} part(s) for {len(wanted)} frame(s)")
        answered: str | None = None     # the cause, once the platform has said the video is gone or closed
        for position, (start, end, times) in enumerate(stretches):
            served = [row for t in times for row in by_time[t]]
            if answered is not None:
                for row in served:
                    record(row, "failed", source="download", cause=answered, attempts=0,
                           reason=f"{platform} {video_id}: not asked for; an earlier part of this video was answered {answered}")
                continue
            attempts = 0
            while True:
                if downloads_made and pause > 0:
                    sleep(pause)
                downloads_made += 1
                attempts += 1
                with tempfile.TemporaryDirectory(prefix="cardvision-fetch-") as scratch:
                    try:
                        limit = {"max_minutes": max_whole_minutes} if start is None and max_whole_minutes is not None else {}
                        result = fetch(platform, video_id, start, end, Path(scratch), **limit)
                    except DownloadFailed as error:
                        cause = error.cause or failure_cause(str(error))
                        if isinstance(error, DownloadRefused) or cause == CAUSE_REFUSED:
                            summary.refused = str(error)
                            break
                        if isinstance(error, VideoTooLong):     # the video is there: no second request, no stop
                            for row in served:
                                record(row, "failed", source="download", reason=str(error), cause=cause, attempts=attempts)
                            break
                        # Only a failure that says nothing about the video is worth a second request.
                        if attempts <= retries and cause == CAUSE_ERROR:
                            say(f"{platform} {video_id}: a download failed ({error}); it is tried once more")
                            continue
                        for row in served:
                            record(row, "failed", source="download", reason=str(error), cause=cause, attempts=attempts)
                        if cause in (CAUSE_UNAVAILABLE, CAUSE_RESTRICTED):
                            answered = cause
                            later = sum(len(by_time[t]) for _, _, more in stretches[position + 1:] for t in more)
                            if later:
                                say(f"{platform} {video_id}: the platform says {cause}; the {later} frame(s) of its later "
                                    "part(s) are not asked for")
                        if cause == CAUSE_ERROR:
                            failures_in_a_row += 1
                            if stop_after_failures and failures_in_a_row >= stop_after_failures:
                                summary.stopped = failures_in_a_row
                        break
                    failures_in_a_row = 0
                    try:
                        with VideoReader(result.path, offset=result.offset, exact=exact_seek) as reader:
                            read_rows(reader, served, "download", attempts=attempts)
                    except FrameNotFound as error:
                        for row in served:
                            record(row, "failed", source="download", reason=str(error), attempts=attempts)
                    if keep_video is not None and result.path.is_file():
                        # The file is read with result.offset above, so that is the start recorded.
                        summary.kept.append(keep_video_file(keep_video, result, platform, video_id, start, end, times))
                break
            if summary.stopped_early:
                break
        if summary.stopped_early:
            break
    if summary.stopped_early:
        summary.remaining = sum(len(wanted) for wanted in pending.values()) - len(summary.entries)
        if summary.refused is None and summary.remaining == 0:
            summary.stopped = None      # the last part of the run: nothing was left to try
    return summary


__all__ = [
    "CAUSES",
    "CAUSE_ERROR",
    "CAUSE_REFUSED",
    "CAUSE_RESTRICTED",
    "CAUSE_UNAVAILABLE",
    "DEFAULT_PAUSE_SECONDS",
    "DEFAULT_RETRIES",
    "DEFAULT_STOP_AFTER_FAILURES",
    "DOWNLOAD_NOTICE",
    "KEPT_INDEX_KIND",
    "KEPT_INDEX_NAME",
    "KEPT_INDEX_VERSION",
    "REFUSAL_PATTERNS",
    "RESTRICTED_PATTERNS",
    "UNAVAILABLE_PATTERNS",
    "CertificatesMissing",
    "Download",
    "DownloadFailed",
    "DownloadRefused",
    "DownloadSupportMissing",
    "Downloader",
    "FetchSummary",
    "KeptVideo",
    "VideoSupportMissing",
    "VideoTooLong",
    "download_video",
    "downloader_installed",
    "error_summary",
    "failure_cause",
    "fetch_frames",
    "has_kept_videos",
    "keep_video_file",
    "kept_file_for",
    "kept_video_name",
    "kept_videos_of",
    "parse_row_selection",
    "parse_video_files",
    "plan_sections",
    "read_kept_videos",
    "refusal_sign",
    "seconds_text",
    "sections_supported",
    "video_url",
    "write_kept_videos",
]
