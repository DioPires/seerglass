# SPDX-License-Identifier: Apache-2.0
"""Draw a seeded random sample of already-labeled points/targets from a label set into a NEW
label set with every previous answer removed, for a delayed consistency repeat: the same person
labels the same items again some days later, and compare_repeat.py scores the two passes.

    python3 make_repeat_set.py --labels-dir DIR --src <SET> [--n 40] [--seed 1] [--out <SET>-rep1]

The new set keeps the source's mode, role, point rule, zoom, recording tags and deck list, and
records `repeatOf` {set, seed, n, drawn, createdAt}; `frameAsset` points at the source's frames.
Point mode keeps only the sampled points' positions; corner/training mode keeps the sampled
targets' corners (so the same card is asked about) and drops outcome, identity, provenance,
eligibility, conditions, label source, prefill and derived fields. Start serve.py with the same
--mode/--role to label it; the set appears in the set menu. Nothing in the source set is changed.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import random
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import labelcore as lc  # noqa: E402



def read_set(root: Path, set_id: str):
    d = root / set_id
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    with (d / "frames.jsonl").open(encoding="utf-8") as f:
        frames = [json.loads(line) for line in f if line.strip()]
    return meta, frames


def labeled_items(meta: dict, frames: list) -> list:
    """(frame name, item) for every complete point/target, in file order."""
    field = "points" if meta.get("mode") == "point" else "targets"
    return [(fr["name"], it) for fr in frames for it in fr.get(field) or [] if lc.target_complete(it)]


def make_repeat(root: Path, src: str, out: str, n: int, seed: int) -> dict:
    meta, frames = read_set(root, src)
    if lc.is_foreign_set(meta, frames):
        raise SystemExit("the source is not a label set of this tool")
    if (root / out).exists():
        raise SystemExit(f"{root / out} exists; pick another --out")
    mode = meta["mode"]
    pool = labeled_items(meta, frames)
    if not pool:
        raise SystemExit("the source has no labeled points/targets yet")
    rng = random.Random(seed)
    drawn = rng.sample(range(len(pool)), min(n, len(pool)))
    keep: dict = {}
    for k in sorted(drawn):
        name, it = pool[k]
        keep.setdefault(name, []).append(lc.strip_answers(it, mode))
    field = "points" if mode == "point" else "targets"
    rows = []
    for fr in frames:
        if fr["name"] in keep:
            rows.append({"name": fr["name"], "t": fr.get("t"), "frameW": fr.get("frameW"), "frameH": fr.get("frameH"),
                         "done": False, field: keep[fr["name"]]})
    new_meta = {k: meta[k] for k in ("schemaVersion", "tool", "mode", "role", "zoom", "pointRule", "recording",
                                      "decklist", "annotator") if k in meta}
    new_meta.update({
        "setId": out, "frameAsset": meta.get("frameAsset") or src, "blind": True,
        "createdAt": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "repeatOf": {"set": src, "seed": seed, "n": n, "drawn": len(drawn), "pool": len(pool),
                     "createdAt": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")},
    })
    d = root / out
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps(new_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (d / "frames.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    return {"out": str(d), "drawn": len(drawn), "pool": len(pool), "frames": len(rows)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels-dir", required=True, help="the labels root (the server's --labels-dir)")
    p.add_argument("--src", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    root = Path(a.labels_dir).expanduser().resolve()
    res = make_repeat(root, a.src, a.out or f"{a.src}-rep{a.seed}", a.n, a.seed)
    print(f"repeat set {res['out']}: {res['drawn']} of {res['pool']} labeled items on {res['frames']} frames")


if __name__ == "__main__":
    main()
