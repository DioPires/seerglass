# SPDX-License-Identifier: Apache-2.0
"""Check label manifests before they are accepted.

``validate_manifests`` reads one or more manifest files and reports every problem it finds:

* a line that is not JSON, or a row that does not fit the format (``schema``, ``row``);
* a row given twice (``duplicate-row``);
* a recording group split across roles, or a video placed in two groups (``group``);
* a source type that is not listed (``source-type``);
* a card id the catalog does not hold, when a catalog is given (``unknown-card``);
* in contribution mode, a row that claims a role or a reachability check, or that names a
  recording of the contributor's own, which nobody else can obtain (``contribution``);
* a file beside the manifest that is not plain text, or is an image or a video (``file``).

Nothing here opens a network connection, decodes an image or follows a link: the check is safe
to run on files from a stranger.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cardvision.data.manifest import numbered_lines, parse_line
from cardvision.data.schema import LOCAL_PLATFORM, ROW_SCHEMA, SOURCE_TYPES, detail_problems, schema_errors

MANIFEST_SUFFIX = ".jsonl"

# Files with these suffixes are refused whatever they contain.
PICTURE_SUFFIXES = frozenset({
    ".png", ".jpg", ".jpeg", ".jpe", ".jfif", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".ico", ".svg", ".svgz",
    ".avif", ".heic", ".heif", ".jxl", ".psd", ".raw", ".ppm", ".pgm", ".pbm", ".pnm", ".tga", ".npy", ".npz",
    ".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".flv", ".wmv", ".mpg", ".mpeg", ".3gp",
})

# The first bytes of the common image and video containers.
_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "PNG image"),
    (b"\xff\xd8\xff", "JPEG image"),
    (b"GIF87a", "GIF image"),
    (b"GIF89a", "GIF image"),
    (b"II*\x00", "TIFF image"),
    (b"MM\x00*", "TIFF image"),
    (b"\x1a\x45\xdf\xa3", "video"),
    (b"\x00\x00\x01\x00", "icon"),
)

# A text file can still smuggle a picture in as an inline data address: the media type, any
# parameters, a comma, then the picture itself.
_INLINE_IMAGE = re.compile(rb"data:image/[A-Za-z0-9.+-]+(;[A-Za-z0-9=._+-]+)*,")


@dataclass(frozen=True)
class Problem:
    """One thing wrong with a manifest or its folder.

    Attributes:
        file: the file concerned.
        line: its line in a manifest (from 1), or ``None`` for a problem with a whole file.
        code: the kind of problem; the module docstring lists them.
        message: what is wrong, in words.
    """

    file: str
    line: int | None
    code: str
    message: str

    def __str__(self) -> str:
        place = self.file if self.line is None else f"{self.file}:{self.line}"
        return f"{place}: [{self.code}] {self.message}"


@dataclass
class Report:
    """The outcome of a validation.

    Attributes:
        files: the manifest files read.
        rows: how many rows fitted the format.
        problems: everything found, in file and line order.
    """

    files: list[str] = field(default_factory=list)
    rows: int = 0
    problems: list[Problem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def codes(self) -> list[str]:
        """The code of every problem, in order."""
        return [problem.code for problem in self.problems]


def _sniff(head: bytes) -> str | None:
    for signature, name in _SIGNATURES:
        if head.startswith(signature):
            return name
    if head[:4] == b"RIFF" and head[8:12] in (b"WEBP", b"AVI "):
        return "WebP image" if head[8:12] == b"WEBP" else "video"
    if head[4:8] == b"ftyp":
        return "image or video"
    return None


def file_problem(path: Path) -> str | None:
    """Why a file may not sit beside a manifest, or ``None`` when it may.

    A file passes when it is a regular file of UTF-8 text, does not carry the name or the first
    bytes of an image or video format, and does not embed an image inline.
    """
    if path.is_symlink():
        return "is a symbolic link; only regular text files are accepted"
    if path.suffix.lower() in PICTURE_SUFFIXES:
        return f"has the suffix {path.suffix.lower()} of an image, array or video file; a manifest folder holds text only"
    data = path.read_bytes()
    kind = _sniff(data[:16])
    if kind is not None:
        return f"holds {kind} data; a manifest folder holds text only"
    if b"\x00" in data:
        return "is not a text file (it holds a zero byte)"
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return "is not a text file (it is not valid UTF-8)"
    if _INLINE_IMAGE.search(data) is not None:
        return "embeds an image inline (data:image/...); a manifest folder holds no image in any form"
    return None


def folder_problems(folder: str | Path) -> list[Problem]:
    """A problem for every file under ``folder`` that ``file_problem`` refuses.

    Version-control folders are skipped; everything else, at any depth, is checked.
    """
    problems = []
    for path in sorted(Path(folder).rglob("*")):
        if ".git" in path.relative_to(folder).parts:
            continue
        if path.is_dir() and not path.is_symlink():
            continue
        reason = file_problem(path)
        if reason is not None:
            problems.append(Problem(str(path), None, "file", reason))
    return problems


def manifest_files(paths: Iterable[str | Path]) -> list[Path]:
    """The manifest files named by ``paths``: a file stands for itself, a folder for every
    ``.jsonl`` file under it. Raises ``FileNotFoundError`` for a path that does not exist and
    ``ValueError`` for a folder without manifests or a file with another suffix."""
    files: list[Path] = []
    for entry in paths:
        path = Path(entry)
        if path.is_dir():
            found = sorted(path.rglob(f"*{MANIFEST_SUFFIX}"))
            if not found:
                raise ValueError(f"{path}: the folder holds no {MANIFEST_SUFFIX} file")
            files.extend(found)
        elif path.is_file():
            if path.suffix != MANIFEST_SUFFIX:
                raise ValueError(f"{path}: a manifest is a {MANIFEST_SUFFIX} file")
            files.append(path)
        else:
            raise FileNotFoundError(f"{path}: no such file or folder")
    return list(dict.fromkeys(files))


def _card_ids(doc: dict[str, Any]) -> Iterable[tuple[str, str]]:
    for field_name in ("boxes", "points"):
        for index, label in enumerate(doc.get(field_name) or []):
            card_id = label.get("card_id")
            if card_id is not None:
                yield f"{field_name}[{index}]", card_id


def validate_manifests(
    paths: Sequence[str | Path],
    *,
    catalog_ids: Collection[str] | None = None,
    source_types: Collection[str] = SOURCE_TYPES,
    contribution: bool = False,
    check_folders: bool = True,
) -> Report:
    """Check manifest files, together: duplicates and group rules span all of them.

    Args:
        paths: manifest files, or folders whose ``.jsonl`` files are all read.
        catalog_ids: the card ids of the game's catalog; when given, every ``card_id``
            must be one of them.
        source_types: the accepted values of ``source_type``.
        contribution: check the rows as an offered contribution: each must have the role
            ``extra``, ``trained_on`` false and no reachability check, because those three are
            set when the labels are accepted, not by whoever offers them. Each must also name a
            video on a platform: a row of a recording of the contributor's own (platform
            ``local``) is refused, because nobody else can obtain that video.
        check_folders: also refuse any file under each manifest's folder that is not plain
            text. Turn this off only to check a single file that sits among unrelated files.
    """
    files = manifest_files(paths)
    report = Report(files=[str(path) for path in files])
    accepted = set(source_types)
    known = None if catalog_ids is None else set(catalog_ids)

    if check_folders:
        for folder in dict.fromkeys(path.parent for path in files):
            report.problems.extend(folder_problems(folder))

    first_seen: dict[tuple[str, str, float], tuple[str, int]] = {}
    group_role: dict[str, tuple[str, str, int]] = {}
    video_group: dict[tuple[str, str], tuple[str, str, int]] = {}

    for path in files:
        name = str(path)

        def add(line: int, code: str, message: str, name: str = name) -> None:
            report.problems.append(Problem(name, line, code, message))

        try:
            lines = list(numbered_lines(path))
        except UnicodeDecodeError:
            report.problems.append(Problem(name, None, "file", "is not a text file (it is not valid UTF-8)"))
            continue
        if not lines:
            report.problems.append(Problem(name, None, "schema", "the manifest holds no row"))
        for number, text in lines:
            try:
                doc = parse_line(text)
            except ValueError as error:
                add(number, "json", f"not valid JSON ({error})")
                continue
            found = schema_errors(doc, ROW_SCHEMA)
            if found:
                for message in found:
                    add(number, "schema", message)
                continue
            for message in detail_problems(doc):
                add(number, "row", message)
            report.rows += 1

            key = (doc["platform"], doc["video_id"], float(doc["t"]))
            if key in first_seen:
                earlier_file, earlier_line = first_seen[key]
                add(number, "duplicate-row",
                    f"{key[0]} {key[1]} at t={key[2]} is already given at {earlier_file}:{earlier_line}")
            else:
                first_seen[key] = (name, number)

            group, role = doc["group"], doc["role"]
            if group in group_role and group_role[group][0] != role:
                earlier_role, earlier_file, earlier_line = group_role[group]
                add(number, "group",
                    f'group "{group}" has the role "{role}" here and "{earlier_role}" at '
                    f"{earlier_file}:{earlier_line}; a recording group is never split across roles")
            group_role.setdefault(group, (role, name, number))
            video = (doc["platform"], doc["video_id"])
            if video in video_group and video_group[video][0] != group:
                earlier_group, earlier_file, earlier_line = video_group[video]
                add(number, "group",
                    f'{video[0]} {video[1]} is in group "{group}" here and "{earlier_group}" at '
                    f"{earlier_file}:{earlier_line}; a video belongs to one recording group")
            video_group.setdefault(video, (group, name, number))

            if doc["source_type"] not in accepted:
                add(number, "source-type",
                    f'source_type "{doc["source_type"]}" is not a listed source type '
                    f"({', '.join(sorted(accepted))})")
            if known is not None:
                for where, card_id in _card_ids(doc):
                    if card_id not in known:
                        add(number, "unknown-card", f'{where}.card_id: "{card_id}" is not in the catalog')
            if contribution:
                if role != "extra":
                    add(number, "contribution",
                        f'role is "{role}": a contributed row has the role "extra" until the seeded rule assigns one')
                if doc["trained_on"]:
                    add(number, "contribution", "trained_on is true: a contributed row has trained no released model")
                if doc["reachable"] is not None:
                    add(number, "contribution", "reachable is set: the reachability check is made when labels are accepted")
                if doc["platform"] == LOCAL_PLATFORM:
                    add(number, "contribution",
                        f'platform is "{LOCAL_PLATFORM}": a recording of your own cannot be offered, because nobody '
                        "else can obtain the video; offered rows name a video on a platform")
    return report
