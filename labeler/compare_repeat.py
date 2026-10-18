# SPDX-License-Identifier: Apache-2.0
"""Agreement between a label set and its consistency repeat (make_repeat_set.py).

    python3 compare_repeat.py --labels-dir DIR --repeat <SET>-rep1 [--json]

Items are matched by (frame name, point index) or (frame name, target id). Only items answered
in BOTH passes are scored; the unanswered count is reported. Reported:
- outcome agreement (card / card back / not a card / unreadable);
- identity agreement on items both passes call a card (exact id, "unknown" included), and the
  same restricted to items both passes mark eligible = yes;
- eligibility agreement (yes / no / uncertain) on items both passes call a card, with Cohen's
  kappa and the 3 × 3 table (first pass in rows);
- the disagreeing items, listed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import labelcore as lc  # noqa: E402
from make_repeat_set import read_set  # noqa: E402


def index(meta, frames) -> dict:
    field = "points" if meta.get("mode") == "point" else "targets"
    key = "i" if field == "points" else "id"
    return {(fr["name"], it[key]): it for fr in frames for it in fr.get(field) or []}


def kappa(pairs: list, cats) -> float | None:
    n = len(pairs)
    if not n:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    pa = {c: sum(1 for a, _ in pairs if a == c) / n for c in cats}
    pb = {c: sum(1 for _, b in pairs if b == c) / n for c in cats}
    pe = sum(pa[c] * pb[c] for c in cats)
    return None if pe >= 1 else (po - pe) / (1 - pe)


def compare(root: Path, repeat: str) -> dict:
    rmeta, rframes = read_set(root, repeat)
    src = (rmeta.get("repeatOf") or {}).get("set")
    if not src:
        raise SystemExit(f"{repeat} is not a repeat set (no repeatOf)")
    smeta, sframes = read_set(root, src)
    first, second = index(smeta, sframes), index(rmeta, rframes)
    both, unanswered, missing = [], 0, 0
    for k, b in second.items():
        a = first.get(k)
        if a is None:
            missing += 1
        elif not lc.target_complete(b):
            unanswered += 1
        else:
            both.append((k, a, b))
    out_agree = sum(1 for _, a, b in both if a["outcome"] == b["outcome"])
    cards = [(k, a, b) for k, a, b in both if a["outcome"] == "card" and b["outcome"] == "card"]
    id_agree = sum(1 for _, a, b in cards if a.get("identity") == b.get("identity"))
    elig_yes = [(k, a, b) for k, a, b in cards if a.get("eligible") == "yes" and b.get("eligible") == "yes"]
    id_agree_yes = sum(1 for _, a, b in elig_yes if a.get("identity") == b.get("identity"))
    pairs = [(a.get("eligible"), b.get("eligible")) for _, a, b in cards]
    table = {x: {y: sum(1 for p in pairs if p == (x, y)) for y in lc.ELIGIBLE} for x in lc.ELIGIBLE}
    dis = []
    for k, a, b in both:
        fields = [f for f in ("outcome", "identity", "eligible") if a.get(f) != b.get(f)]
        if fields:
            dis.append({"frame": k[0], "item": k[1], **{f: [a.get(f), b.get(f)] for f in fields}})
    rate = lambda x, n: (x / n) if n else None  # noqa: E731
    return {
        "source": src, "repeat": repeat, "mode": rmeta.get("mode"), "items": len(second),
        "answeredBoth": len(both), "unansweredInRepeat": unanswered, "missingInSource": missing,
        "outcomeAgreement": rate(out_agree, len(both)), "outcomeAgree": out_agree,
        "bothCard": len(cards), "identityAgreement": rate(id_agree, len(cards)), "identityAgree": id_agree,
        "bothEligibleYes": len(elig_yes), "identityAgreementEligibleYes": rate(id_agree_yes, len(elig_yes)),
        "eligibilityAgreement": rate(sum(1 for a, b in pairs if a == b), len(pairs)),
        "eligibilityKappa": kappa(pairs, lc.ELIGIBLE), "eligibilityTable": table, "disagreements": dis,
    }


def fmt(v) -> str:
    return "—" if v is None else f"{100 * v:.1f} %"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels-dir", required=True, help="the labels root (the server's --labels-dir)")
    p.add_argument("--repeat", required=True)
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    r = compare(Path(a.labels_dir).expanduser().resolve(), a.repeat)
    if a.json:
        print(json.dumps(r, indent=1))
        return
    print(f"{r['repeat']} vs {r['source']} ({r['mode']} mode): {r['items']} items, {r['answeredBoth']} answered in both"
          f" ({r['unansweredInRepeat']} unanswered in the repeat, {r['missingInSource']} missing in the source)")
    print(f"outcome agreement      {fmt(r['outcomeAgreement'])}  ({r['outcomeAgree']}/{r['answeredBoth']})")
    print(f"identity agreement     {fmt(r['identityAgreement'])}  ({r['identityAgree']}/{r['bothCard']} both card)")
    print(f"  among eligible=yes   {fmt(r['identityAgreementEligibleYes'])}  (n={r['bothEligibleYes']})")
    k = r["eligibilityKappa"]
    print(f"eligibility agreement  {fmt(r['eligibilityAgreement'])}  kappa {'—' if k is None else f'{k:.3f}'}")
    print("  first \\ repeat  " + "  ".join(f"{c:>9}" for c in lc.ELIGIBLE))
    for x in lc.ELIGIBLE:
        print(f"  {x:>14}  " + "  ".join(f"{r['eligibilityTable'][x][y]:>9}" for y in lc.ELIGIBLE))
    for d in r["disagreements"]:
        print("  differs:", json.dumps(d))


if __name__ == "__main__":
    main()
