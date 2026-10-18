# SPDX-License-Identifier: Apache-2.0
"""Write a sample file for the labeler: which frames of which videos to label.

    python3 make_sample.py --out sample.json --frames-root DIR [--n 4] [--seed 1]
    python3 make_sample.py --out sample.json --asset ID=DIR [--asset ID2=DIR2 ...] --all
    python3 make_sample.py --out sample.json --frames-root DIR --all --times prefill.jsonl
    python3 make_sample.py --out sample.json --asset ID=DIR --all --times prefill.jsonl --local

A sample is `{"assets": [{"assetId", "framesDir", "frames": [{"t", "name"}]}]}`. Each asset is
one video (one folder of frames) and becomes one label set of the same id.

- `--frames-root DIR`: every sub-folder of DIR that holds frames is an asset, named after the
  folder. `--asset ID=DIR` names one folder explicitly (repeatable; both may be combined).
- Per asset: N frames (`--n`, default 4) spread evenly over the folder, each at a seeded position
  inside its slice, or every frame with `--all`. No image is opened: the choice depends on the
  file names, the seed and the asset id only.
- Each asset records its `framesDir` relative to the sample file, so the server serves the
  frames from where they are (nothing is copied) and the folder can be moved as a whole.
- `t` (seconds, used to order frames, and written as a manifest row's time) comes from
  `--times FILE` when given: JSON lines with `name` and `t`, such as the inference command's
  prefill file. Otherwise it is worked out for frames that were cut `--fps` a second from the
  start of the video: a name ending in `fNNNNN` gets (NNNNN − 0.5) / `--fps`, and any other
  name (its position in the folder, from 0, + 0.5) / `--fps`. That is the middle of the frame's
  time on screen, a time at which the frame is on screen, and not the moment it starts: a
  start time, rounded to the millisecond, can fall just before the frame.
- `--local`: the frames come from recordings of your own, which are on no video platform. Each
  asset then also records `"platform": "local"` and its own id as `"videoId"`, which is what the
  manifest export needs to write rows for such a recording (`export.py manifest`). The asset id
  must then start with a letter or a digit and have 64 characters at most.
- `--title ID=TEXT`, `--upload-date ID=YYYY-MM-DD`, `--language ID=TAG` (each repeatable, all
  optional): what is known of a video, copied into its asset as `title`, `uploadDate` and
  `language`. The labeling page shows them under "About this recording", to judge a frame by
  when and where it was recorded. Nothing is looked up: give them only when you know them.

Frame files are `.jpg`, `.jpeg` or `.png` with names made of letters, digits, `_`, `-` and `.`;
asset ids of letters, digits, `_` and `-`. Other files are skipped and counted.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import labelcore as lc  # noqa: E402
from export import write_text_atomic  # noqa: E402

INDEX_RE = re.compile(r"(?:^|[-_.])f(\d+)$", re.IGNORECASE)
LOCAL_PLATFORM = "local"  # a recording of one's own, on no video platform (the manifest format's name for it)
LOCAL_VIDEO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def _natural(name: str) -> tuple:
    """Sort key: runs of digits compare as numbers (f2 before f10)."""
    return tuple(int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", name))


def frames_of(d: Path) -> tuple:
    """(frame file names in order, number of other files skipped)."""
    names, skipped = [], 0
    for p in d.iterdir():
        if not p.is_file() or p.name.startswith("."):
            continue
        if lc.FRAME_NAME.fullmatch(p.name):
            names.append(p.name)
        else:
            skipped += 1
    return sorted(names, key=_natural), skipped


def pick(names: list, n: int, seed: int, asset: str) -> list:
    """N names spread evenly: one at a seeded position inside each of N equal slices."""
    if n >= len(names):
        return list(names)
    rng = random.Random(f"{seed}:{asset}")
    out = []
    for k in range(n):
        lo, hi = (k * len(names)) // n, ((k + 1) * len(names)) // n
        out.append(names[rng.randrange(lo, max(lo + 1, hi))])
    return out


def frame_time(name: str, position: int, fps: float, times: dict) -> float:
    """A time at which the frame is on screen. From `times` when the frame is listed there;
    else the middle of the frame's time on screen, to the millisecond, taking the frames to be
    cut `fps` a second from the start of the video and numbered from 1 by their names (or, for
    a name without a number, by `position` in the folder, from 0)."""
    if name in times:
        return times[name]
    m = INDEX_RE.search(name.rsplit(".", 1)[0])
    index = int(m.group(1)) - 1 if m else position
    return round((max(0, index) + 0.5) / fps, 3)


def read_times(path: Path) -> dict:
    """{frame name: t} from JSON lines carrying `name` and `t` (a prefill file)."""
    out = {}
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--times: line {n + 1} is not JSON ({exc})")
        if isinstance(row, dict) and isinstance(row.get("name"), str) and isinstance(row.get("t"), (int, float)):
            out[row["name"]] = float(row["t"])
    return out


def find_assets(frames_root, explicit: list) -> list:
    """[(asset id, folder)] from --frames-root (its sub-folders with frames) and --asset ID=DIR."""
    found = []
    if frames_root:
        root = Path(frames_root).expanduser().resolve()
        if not root.is_dir():
            raise SystemExit(f"--frames-root: not a folder: {root}")
        for d in sorted(root.iterdir(), key=lambda p: _natural(p.name)):
            if d.is_dir() and not d.name.startswith(".") and frames_of(d)[0]:
                found.append((d.name, d))
    for spec in explicit or []:
        asset, sep, folder = spec.partition("=")
        if not sep or not folder:
            raise SystemExit(f"--asset takes ID=DIR, not {spec!r}")
        d = Path(folder).expanduser().resolve()
        if not d.is_dir():
            raise SystemExit(f"--asset {asset}: not a folder: {d}")
        found.append((asset, d))
    seen = set()
    for asset, _d in found:
        if not lc.ID_SAFE.fullmatch(asset):
            raise SystemExit(f"asset id {asset!r}: use letters, digits, _ and - (rename the folder, or name it with --asset ID=DIR)")
        if asset in seen:
            raise SystemExit(f"asset id {asset!r} is given twice")
        seen.add(asset)
    return found


SOURCE_FLAGS = {"title": "--title", "uploadDate": "--upload-date", "language": "--language"}


def read_about(given: dict, asset_ids) -> dict:
    """{asset id: {field: value}} from the ID=VALUE flags (`given`: {field: [specs]}), each value
    checked as the labeler checks a sample's (labelcore.source_field_ok)."""
    out = {}
    for field, specs in given.items():
        flag = SOURCE_FLAGS[field]
        for spec in specs or []:
            asset, sep, value = spec.partition("=")
            if not sep:
                raise SystemExit(f"{flag} takes ID=VALUE, not {spec!r}")
            if asset not in asset_ids:
                raise SystemExit(f"{flag} {asset}: no asset of that id")
            if not lc.source_field_ok(field, value):
                raise SystemExit(f"{flag} {asset}: {value!r} is not " + {"title": f"a title of one line, {lc.TITLE_MAX} characters at most",
                                                                         "uploadDate": f"a date as YYYY-MM-DD, from {lc.FIRST_UPLOAD_YEAR} to today",
                                                                         "language": "a language tag such as en or pt-BR"}[field])
            out.setdefault(asset, {})[field] = value
    return out


def build(assets: list, out_dir: Path, n: int, seed: int, all_frames: bool, fps: float, times: dict,
          local: bool = False, about: dict | None = None) -> tuple:
    """(the sample document, a list of notes for the person running it). `local`: the assets are
    recordings of one's own; each records the platform "local" and its id as its video id.
    `about`: {asset id: {title, uploadDate, language}}, copied into those assets (read_about)."""
    doc, notes = [], []
    for asset, d in assets:
        if local and not LOCAL_VIDEO_ID.fullmatch(asset):
            raise SystemExit(f"asset id {asset!r}: with --local the id is the recording's video id, which starts "
                             "with a letter or a digit and has 64 characters at most")
        names, skipped = frames_of(d)
        if not names:
            raise SystemExit(f"{asset}: no frame files (.jpg, .jpeg, .png) in {d}")
        if skipped:
            notes.append(f"{asset}: {skipped} file(s) skipped (not a frame file name the server accepts)")
        position = {name: i for i, name in enumerate(names)}
        chosen = names if all_frames else pick(names, n, seed, asset)
        doc.append({"assetId": asset, **({"platform": LOCAL_PLATFORM, "videoId": asset} if local else {}),
                    **(about or {}).get(asset, {}),
                    "framesDir": os.path.relpath(d, out_dir).replace(os.sep, "/"),
                    "frames": [{"t": frame_time(nm, position[nm], fps, times), "name": nm} for nm in chosen]})
    return {"schemaVersion": 1, "seed": seed, "assets": doc}, notes


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="the sample file to write (refused if it exists, unless --force)")
    p.add_argument("--frames-root", default=None, help="a folder whose sub-folders each hold one video's frames")
    p.add_argument("--asset", action="append", metavar="ID=DIR", help="one video's frames folder, with its id (repeatable)")
    p.add_argument("--n", type=int, default=4, help="frames per asset (default 4)")
    p.add_argument("--all", action="store_true", help="every frame of each asset")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--fps", type=float, default=1.0, help="frames per second of the extracted frames (default 1), for `t`")
    p.add_argument("--times", default=None, help="JSON lines with `name` and `t` per frame (a prefill file), for `t`")
    p.add_argument("--local", action="store_true", dest="own_recordings",
                   help="the frames come from recordings of your own: each asset records \"platform\": \"local\" and "
                        "its id as \"videoId\", for the manifest export")
    p.add_argument("--title", action="append", metavar="ID=TEXT", help="the video's title (repeatable, optional)")
    p.add_argument("--upload-date", action="append", metavar="ID=YYYY-MM-DD", help="the day the video was published (repeatable, optional)")
    p.add_argument("--language", action="append", metavar="ID=TAG", help="the video's language, such as en (repeatable, optional)")
    p.add_argument("--force", action="store_true", help="overwrite an existing sample file")
    a = p.parse_args(argv)
    if not a.frames_root and not a.asset:
        raise SystemExit("give --frames-root DIR and/or --asset ID=DIR")
    if a.n < 1 or a.fps <= 0:
        raise SystemExit("--n must be at least 1 and --fps above 0")
    out = Path(a.out).expanduser().resolve()
    if out.exists() and not a.force:
        raise SystemExit(f"{out} exists (use --force)")
    assets = find_assets(a.frames_root, a.asset)
    if not assets:
        raise SystemExit("no folder with frame files found")
    times = read_times(Path(a.times).expanduser()) if a.times else {}
    about = read_about({"title": a.title, "uploadDate": a.upload_date, "language": a.language}, {x for x, _d in assets})
    data, notes = build(assets, out.parent, a.n, a.seed, a.all, a.fps, times, a.own_recordings, about)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(out, json.dumps(data, indent=1) + "\n")
    for note in notes:
        print(note)
    print(f"wrote {out}: {sum(len(x['frames']) for x in data['assets'])} frames of {len(data['assets'])} assets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
