# SPDX-License-Identifier: Apache-2.0
"""Turn a pasted deck list into a deck-list file of card ids for the labeler.

    python3 decklist_from_text.py --catalog catalog.json --out-dir decklists --video <VIDEO ID>
                                  [--label "Player A"] [--in pasted.txt] [--force] [--strict] [--dry-run]

Reads the pasted text from --in (or stdin): card names one per line, possibly with counts
("3x Name", "3 Name", "Name x3", "Name ×3", "Name (3)"). Each line is matched against the
catalog as the picker reads it (serve.load_catalog): card ids (case-insensitive), then exact
names (case, typographic quotes and spacing aside). Nothing is guessed:

  - a line that matches nothing is REPORTED and written as a `# UNMATCHED:` comment (fix it by
    hand, e.g. with the card id); section headers ("Main deck", "Sideboard:") land here too;
  - a name that several printings share is written as all of them (the picker shows each; the
    labeler picks the printing on screen) and reported as ambiguous.

Writes `<out-dir>/<video>.txt`, or `<video>.<label-slug>.txt` with --label (one file per player;
the server loads every `<video>*.txt` for the video and shows each list under its label, taken
from the file's `# label:` line). --strict writes nothing when a line is unmatched.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import labelcore as lc  # noqa: E402



def slug(label: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", label.strip()).strip("-").lower()
    return s or "list"


def convert(text: str, catalog: list) -> dict:
    """{rows: [{line, count, ids, names}], unmatched: [line], ambiguous: [{line, ids}]}."""
    by_id = {str(c.get("id") or "").upper(): c.get("id") for c in catalog if c.get("id")}
    by_name = lc.catalog_name_index(catalog)
    names = {c.get("id"): c.get("name") for c in catalog}
    rows, unmatched, ambiguous = [], [], []
    for raw in (text or "").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        n, entry = lc.split_count(line)
        ids = lc.resolve_entry(entry, by_id, by_name)
        if not ids and n is not None:
            ids = lc.resolve_entry(line, by_id, by_name)
            n = None if ids else n
        if not ids:
            unmatched.append(raw.strip())
            rows.append({"line": raw.strip(), "count": n, "ids": [], "names": []})
            continue
        if len(ids) > 1:
            ambiguous.append({"line": raw.strip(), "ids": ids})
        rows.append({"line": raw.strip(), "count": n, "ids": ids, "names": [names.get(i) for i in ids]})
    return {"rows": rows, "unmatched": unmatched, "ambiguous": ambiguous}


def render(res: dict, label: str | None, video: str) -> str:
    now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    out = []
    if label:
        out.append(f"# label: {label}")
    out.append(f"# video: {video}; converted {now} by decklist_from_text.py from pasted text")
    out.append("# one card id per line (optional count first); '#' starts a comment")
    for r in res["rows"]:
        if not r["ids"]:
            out.append(f"# UNMATCHED: {r['line']}")
            continue
        if len(r["ids"]) > 1:
            out.append(f"# ambiguous: {r['line']!r} is {len(r['ids'])} printings; all listed")
        for cid, name in zip(r["ids"], r["names"]):
            out.append(f"{'' if r['count'] is None else str(r['count']) + ' '}{cid}  # {name}")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="the video id the list belongs to (the label set's frame asset)")
    p.add_argument("--label", default=None, help="player/deck label; gives <video>.<slug>.txt")
    p.add_argument("--in", dest="inp", default=None, help="pasted text file (default: stdin)")
    p.add_argument("--catalog", required=True, help="the catalog file the labeler is started with")
    p.add_argument("--out-dir", required=True, help="the deck-lists folder (the server's --decklists-dir)")
    p.add_argument("--force", action="store_true", help="overwrite an existing file")
    p.add_argument("--strict", action="store_true", help="write nothing if any line is unmatched")
    p.add_argument("--dry-run", action="store_true", help="report only")
    a = p.parse_args(argv)
    if not lc.ID_SAFE.fullmatch(a.video):
        raise SystemExit("--video must be a video id ([A-Za-z0-9_-])")
    text = Path(a.inp).read_text(encoding="utf-8") if a.inp else sys.stdin.read()
    import serve  # the picker's own reading of the catalog (cards, then the tokens it adds)
    catalog = Path(a.catalog).expanduser().resolve()
    if not catalog.is_file():
        raise SystemExit(f"--catalog: not a file: {catalog}")
    serve.CFG.catalog = catalog
    res = convert(text, serve.load_catalog())
    matched = sum(1 for r in res["rows"] if r["ids"])
    print(f"{matched} lines matched, {len(res['ambiguous'])} ambiguous (all printings listed), {len(res['unmatched'])} unmatched")
    for amb in res["ambiguous"]:
        print(f"  ambiguous: {amb['line']!r} → {', '.join(amb['ids'])}")
    for line in res["unmatched"]:
        print(f"  UNMATCHED: {line!r}")
    if a.dry_run:
        return 0
    if a.strict and res["unmatched"]:
        print("--strict: nothing written")
        return 1
    out = Path(a.out_dir) / (f"{a.video}.{slug(a.label)}.txt" if a.label else f"{a.video}.txt")
    if out.exists() and not a.force:
        raise SystemExit(f"{out} exists (use --force)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(res, a.label, a.video), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
