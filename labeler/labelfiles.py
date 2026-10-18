# SPDX-License-Identifier: Apache-2.0
"""How the labeling tool reads its files from disk: one reader and one rule, shared by the
project screen (project.py), the labeling page (serve.py) and the export (export.py), so the
three cannot disagree about what a sample lists or what a label set holds.

    read_text(path, max_bytes)     a file's text, only when it is a regular file (after links are
                                   resolved) and, with `max_bytes`, no larger than that; else why not
    read_sample(path, max_bytes)   the videos and frames a sample file lists (labelcore.sample_assets)
    read_label_set(folder)         one label set: its meta.json, the rows of its frames.jsonl, and
                                   every line that could not be read, with the file and the line
    label_sets(labels_dir, assets) the label sets of a sample in a labels folder: one for each
                                   video, then the repeat sets made from them
    read_removed(labels_dir)       the frames removed from a project, and the nearby frames used
                                   instead (removed-frames.json in the labels folder)
    apply_frame_changes(assets, record)   a sample's videos as the project labels them: removed
                                   frames left out, nearby frames used instead put in. Every reader
                                   of a sample applies it where it knows the labels folder.

A path that a request names is never read whole without the first check: a named pipe would
hold a thread for good, a device would be read without end, and a very large file would be read
into memory. A file named on the command line is read with no size limit, as before.

A label file is never guessed at. A line of frames.jsonl that is not a JSON object is damage:
the set's readable rows are still returned, with the damage beside them (`rows.damage`), and
every reader then says so: the project screen and the export name the file and the line, the
labeling page opens the set read-only so that nothing is written over it.

Standard library only.
"""
from __future__ import annotations

import json
import math
import os
import re
import stat
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

import labelcore as lc  # noqa: E402

# The largest sample, catalog, deck list, suggestions file or project file that is read when a
# request names it (20 MB: a catalog of every card of a large game is a few MB).
SAMPLE_BYTES = 20 * 1024 * 1024

# Why a file was not read, in words (the codes are what the pages are told).
PROBLEM_WORDS = {
    "missing": "nothing is there",
    "not-file": "it is not a file (a folder, a device or a pipe)",
    "too-big": f"it is larger than {SAMPLE_BYTES // (1024 * 1024)} MB",
    "unreadable": "it cannot be read",
}


def file_problem(path, max_bytes=None):
    """None when `path` is a regular file (after links are resolved) of at most `max_bytes`,
    else missing | not-file | too-big | unreadable. Nothing is opened."""
    try:
        st = os.stat(path)
    except (FileNotFoundError, NotADirectoryError):  # nothing at that name, or a file where a folder of its path would be
        return "missing"
    except (OSError, ValueError):
        return "unreadable"
    if not stat.S_ISREG(st.st_mode):
        return "not-file"
    if max_bytes is not None and st.st_size > max_bytes:
        return "too-big"
    return None


def read_bytes(path, max_bytes=None) -> tuple:
    """(bytes, None), or (None, problem): missing, not-file, too-big, unreadable. The file is
    checked before it is opened and again once it is open (it may have been replaced meanwhile),
    and never more than `max_bytes` are read."""
    problem = file_problem(path, max_bytes)
    if problem:
        return None, problem
    try:
        # never wait for a writer: what was a file a moment ago may be a pipe now
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0))
    except (FileNotFoundError, NotADirectoryError):
        return None, "missing"
    except (OSError, ValueError):
        return None, "unreadable"
    try:
        with os.fdopen(fd, "rb") as f:
            st = os.fstat(f.fileno())
            if not stat.S_ISREG(st.st_mode):
                return None, "not-file"
            if max_bytes is None:
                return f.read(), None
            data = f.read(max_bytes + 1)
            return (None, "too-big") if len(data) > max_bytes else (data, None)
    except OSError:
        return None, "unreadable"


def read_text(path, max_bytes=None) -> tuple:
    """(text, None), or (None, problem): as `read_bytes`; a file that is not UTF-8 is unreadable.
    Line ends are read as Python reads a text file (\r\n and \r become \n), so the text is
    what `Path.read_text` gave before this reader existed."""
    data, problem = read_bytes(path, max_bytes)
    if problem:
        return None, problem
    try:
        return data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n"), None
    except UnicodeDecodeError:
        return None, "unreadable"


def read_json(path, max_bytes=None) -> tuple:
    """(value, None), or (None, problem): as `read_text`, and not-json."""
    text, problem = read_text(path, max_bytes)
    if problem:
        return None, problem
    try:
        return json.loads(text), None
    except (json.JSONDecodeError, RecursionError):
        return None, "not-json"


def read_sample(path, max_bytes=None) -> tuple:
    """(assets, None), or (None, problem): missing, not-file, too-big, unreadable, not-json,
    not-sample. `assets`: what the sample lists (`labelcore.sample_assets`)."""
    doc, problem = read_json(path, max_bytes)
    if problem:
        return None, problem
    assets = lc.sample_assets(doc)
    return (None, "not-sample") if assets is None else (assets, None)


class Rows(list):
    """The frame rows of a label set, with what could not be read beside them: `damage` is
    [{"file", "line", "problem"}], empty for a set that is whole."""

    def __init__(self, rows=(), damage=()):
        super().__init__(rows)
        self.damage = list(damage)


def damage_words(damage: list) -> str:
    """Damage as a sentence part: "FILE, line 3: not JSON" (the first, and how many more)."""
    if not damage:
        return ""
    d = damage[0]
    first = f"{d['file']}{', line ' + str(d['line']) if d.get('line') else ''}: {d['problem']}"
    return first + (f" (and {len(damage) - 1} more)" if len(damage) > 1 else "")


def own_damage(meta, rows) -> list:
    """The damage of a label set that the tool acts on: every line that could not be read in a
    set this tool wrote, and a meta.json that cannot be read, whatever wrote it. A folder of
    files the tool did not write (no meta.json, or one that names no mode of the tool) is
    foreign: it is left as it is, and its lines are not this tool's to judge."""
    damage = list(getattr(rows, "damage", ()))
    if not damage or any(Path(d["file"]).name == "meta.json" for d in damage):
        return damage
    return damage if isinstance(meta, dict) and meta.get("mode") in lc.MODES else []


def read_label_set(folder) -> tuple:
    """(meta or None, rows) of the label set in `folder`. `rows` is a `Rows`: the lines of
    frames.jsonl that are JSON objects, in file order, and under `rows.damage` every line that is
    not, with its file and line number; a meta.json or a frames.jsonl that is there and cannot be
    read at all is damage too. Files that are absent are not damage: the set is new."""
    folder = Path(folder)
    damage = []
    meta, problem = read_json(folder / "meta.json")
    if problem and problem != "missing":
        damage.append({"file": str(folder / "meta.json"), "line": None,
                       "problem": "not JSON" if problem == "not-json" else PROBLEM_WORDS[problem]})
    if problem is None and not isinstance(meta, dict):
        damage.append({"file": str(folder / "meta.json"), "line": None, "problem": "not a JSON object"})
    meta = meta if isinstance(meta, dict) else None
    rows = []
    name = folder / "frames.jsonl"
    text, problem = read_text(name)
    if problem and problem != "missing":
        damage.append({"file": str(name), "line": None, "problem": PROBLEM_WORDS[problem]})
    for number, line in enumerate((text or "").split("\n"), start=1):  # lines end at a line break and nowhere else
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, RecursionError):
            damage.append({"file": str(name), "line": number, "problem": "not JSON"})
            continue
        if not isinstance(row, dict):
            damage.append({"file": str(name), "line": number, "problem": "not a JSON object"})
            continue
        rows.append(row)
    return meta, Rows(rows, damage)


def label_sets(labels_dir, assets: list) -> list:
    """The label sets of a sample in a labels folder, as every screen lists them:
    [{"setId", "frameAsset", "asset", "meta", "rows"}]. One set for each video of the sample
    (the set of the same id), in the sample's order; then the folders whose meta.json names one
    of those videos as its `frameAsset` (repeat sets), in name order, each with the video's
    frames that it has rows for."""
    by_id = {a["assetId"]: a for a in assets}
    out = []
    for aid, asset in by_id.items():
        meta, rows = read_label_set(Path(labels_dir) / aid)
        out.append({"setId": aid, "frameAsset": aid, "asset": asset, "meta": meta, "rows": rows})
    root = Path(labels_dir)
    try:
        folders = sorted(d for d in root.iterdir() if d.name not in by_id and d.is_dir() and lc.ID_SAFE.fullmatch(d.name))
    except OSError:
        folders = []
    for d in folders:
        meta, rows = read_label_set(d)
        video = (meta or {}).get("frameAsset")
        if isinstance(video, str) and video in by_id:
            names = {row.get("name") for row in rows}
            asset = dict(by_id[video])
            asset["frames"] = [f for f in asset["frames"] if f.get("name") in names]
            if "removedFrames" in asset:
                asset["removedFrames"] = [n for n in asset["removedFrames"] if n in names]
            out.append({"setId": d.name, "frameAsset": video, "asset": asset, "meta": meta, "rows": rows})
    return out


# ---- frames removed from a project ---------------------------------------------------------------

# One file in the labels folder lists the frames a person removed from the project, and the
# nearby frames of the same video put in instead. The sample file is never rewritten, and no
# label is deleted: a removed frame's labels stay in its set's frames.jsonl, so restoring it
# brings them back as they were.
REMOVED_FILE = "removed-frames.json"
REMOVE_REASONS = ("blurry-or-small", "no-cards", "near-duplicate", "other")
NEARBY_MAX = 6  # the nearby frames offered in place of a removed one


def empty_removed() -> dict:
    return {"schemaVersion": 1, "tool": "labeler", "removed": [], "replacements": []}


def is_time(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _removed_entry(e) -> dict | None:
    if not isinstance(e, dict) or not isinstance(e.get("assetId"), str) or not lc.ID_SAFE.fullmatch(e["assetId"]):
        return None
    if not isinstance(e.get("name"), str) or not lc.FRAME_NAME.fullmatch(e["name"]):
        return None
    if e.get("reason") is not None and e.get("reason") not in REMOVE_REASONS:
        return None
    return {"assetId": e["assetId"], "name": e["name"], "reason": e.get("reason"), "at": e.get("at") if isinstance(e.get("at"), str) else None}


def _replacement_entry(e) -> dict | None:
    if not isinstance(e, dict) or not isinstance(e.get("assetId"), str) or not lc.ID_SAFE.fullmatch(e["assetId"]):
        return None
    added = e.get("added")
    if not isinstance(e.get("removed"), str) or not lc.FRAME_NAME.fullmatch(e["removed"]) or not isinstance(added, dict):
        return None
    if not isinstance(added.get("name"), str) or not lc.FRAME_NAME.fullmatch(added["name"]) or ("t" in added and not is_time(added["t"])):
        return None
    return {"assetId": e["assetId"], "removed": e["removed"], "added": {k: added[k] for k in ("t", "name") if k in added},
            "at": e.get("at") if isinstance(e.get("at"), str) else None}


def read_removed(labels_dir) -> tuple:
    """(record, None), or (an empty record, problem) when the file is there and cannot be used:
    not-file, too-big, unreadable, not-json, not-record (an entry this tool would not write).
    No file is no frame removed. The record is never guessed at: one entry that does not fit
    makes the whole file unusable, and the tool then removes or restores nothing until it is
    corrected."""
    doc, problem = read_json(Path(labels_dir) / REMOVED_FILE, SAMPLE_BYTES)
    if problem == "missing":
        return empty_removed(), None
    if problem:
        return empty_removed(), problem
    if not isinstance(doc, dict) or doc.get("tool") != "labeler" or not isinstance(doc.get("removed", []), list) \
            or not isinstance(doc.get("replacements", []), list):
        return empty_removed(), "not-record"
    removed = [_removed_entry(e) for e in doc.get("removed", [])]
    replacements = [_replacement_entry(e) for e in doc.get("replacements", [])]
    if any(e is None for e in removed + replacements):
        return empty_removed(), "not-record"
    return {**empty_removed(), "removed": removed, "replacements": replacements}, None


def write_removed(labels_dir, record: dict) -> None:
    """Write the record whole, to a temporary file that then replaces the old one."""
    folder = Path(labels_dir)
    folder.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, indent=1, sort_keys=True) + "\n"
    fd, tmp = tempfile.mkstemp(dir=str(folder), prefix=f".{REMOVED_FILE}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, folder / REMOVED_FILE)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def removed_keys(record: dict) -> set:
    """{(video id, frame name)} of the frames removed."""
    return {(e["assetId"], e["name"]) for e in record.get("removed", [])}


def apply_frame_changes(assets: list, record: dict) -> list:
    """The videos of a sample (`labelcore.sample_assets`) as the project labels them: each
    nearby frame put in instead of a removed one is added to its video (after the frames the
    sample lists), and every removed frame is left out. Each video also gets `removedFrames`:
    the names of its frames that were removed, in its order. The assets given are not changed."""
    gone = removed_keys(record)
    out = []
    for a in assets:
        aid = a["assetId"]
        frames = list(a["frames"])
        listed = {f["name"] for f in frames}
        for rep in record.get("replacements", []):
            if rep["assetId"] == aid and rep["added"]["name"] not in listed:
                frames.append(dict(rep["added"]))
                listed.add(rep["added"]["name"])
        out.append({**a, "frames": [f for f in frames if (aid, f["name"]) not in gone],
                    "removedFrames": [f["name"] for f in frames if (aid, f["name"]) in gone]})
    return out


def _natural(name: str) -> tuple:
    """Sort key: runs of digits compare as numbers (f2 before f10), as make_sample.py orders a folder."""
    return tuple((0, int(p), "") if p.isdigit() else (1, 0, p.lower()) for p in re.split(r"(\d+)", name))


def folder_frames(folder) -> list:
    """The frame files of a folder (names the tool accepts), in make_sample.py's order."""
    try:
        names = [e.name for e in os.scandir(folder) if lc.FRAME_NAME.fullmatch(e.name) and e.is_file()]
    except OSError:
        return []
    return sorted(names, key=_natural)


# A frame's number in its name, `fNNNNN.<ext>` (also after a prefix, `<video>-fNNNNN.<ext>`):
# the rule of labelcore.frame_index and make_sample.py, which number frames from 1.
FRAME_NUMBER = re.compile(r"^(?:.*[-_.])?f(\d+)\.[A-Za-z]+$")


def frame_number(name: str):
    m = FRAME_NUMBER.match(str(name or ""))
    return int(m.group(1)) if m else None


def estimate_time(order: list, known: dict, name: str):
    """A frame's time in the video, worked out from the frames whose time is known (`known`:
    {name: t}), or None: a time is never guessed.

    By the frames' numbers (`fNNNNN`) when the frame has one: frames cut at a steady pace have
    times in a line with their numbers, so with two numbered known frames the time is read off
    that line, between the two nearest that bracket it (else the two nearest on one side). With
    one, make_sample.py's own rule gives the pace: frame N of frames cut `fps` a second is at
    (N − 0.5) / fps, so fps = (N − 0.5) / t. A frame without a number is placed by its position
    in the folder (`order`), only between two known frames that bracket it. Seconds, to the
    millisecond; None for a time before the video's start."""
    t = None
    n = frame_number(name)
    if n is not None:
        points = sorted({frame_number(k): float(v) for k, v in known.items() if frame_number(k) is not None and is_time(v)}.items())
        same = [v for k, v in points if k == n]
        below = [p for p in points if p[0] < n]
        above = [p for p in points if p[0] > n]
        if same:
            return same[0]
        if len(points) >= 2:
            (k0, t0), (k1, t1) = (below[-1], above[0]) if below and above else (below[-2], below[-1]) if below else (above[0], above[1])
            t = t0 + (t1 - t0) * (n - k0) / (k1 - k0)
        elif len(points) == 1:
            k0, t0 = points[0]
            if k0 >= 1 and t0 > 0:
                t = (n - 0.5) * t0 / (k0 - 0.5)
    else:
        where = {k: i for i, k in enumerate(order)}
        if name not in where:
            return None
        points = sorted((where[k], float(v)) for k, v in known.items() if k in where and is_time(v))
        i = where[name]
        below = [p for p in points if p[0] < i]
        above = [p for p in points if p[0] > i]
        if below and above:
            (i0, t0), (i1, t1) = below[-1], above[0]
            t = t0 + (t1 - t0) * (i - i0) / (i1 - i0)
    if t is None or not math.isfinite(t) or t < 0:
        return None
    return round(t, 3)


def nearby_frames(folder, listed: list, record: dict, asset_id: str, name: str, limit: int = NEARBY_MAX) -> list:
    """The frames of a video's folder that could be used instead of its frame `name`: files the
    sample does not list (`listed`: the sample's own frames of that video, [{t, name}]) and that
    no earlier choice put in, the nearest to `name` in the folder's order first, at most `limit`,
    returned in the folder's order as [{name, t}] (`t` estimated, None when it cannot be)."""
    order = folder_frames(folder)
    if name not in order:
        return []
    taken = {f["name"] for f in listed} | {r["added"]["name"] for r in record.get("replacements", []) if r["assetId"] == asset_id}
    at = order.index(name)
    free = [n for n in order if n not in taken and n != name]
    pos = {n: i for i, n in enumerate(order)}
    near = sorted(sorted(free, key=lambda n: (abs(pos[n] - at), pos[n]))[:limit], key=lambda n: pos[n])
    known = {f["name"]: f.get("t") for f in listed}
    return [{"name": n, "t": estimate_time(order, known, n)} for n in near]
