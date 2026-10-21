# SPDX-License-Identifier: Apache-2.0
"""``python -m cardvision.mtg COMMAND``: the Magic tools, in the order they are run.

    bulk           download Scryfall's default_cards bulk file
    build-catalog  the catalog, its image map and its exclusions, from a bulk file
    fetch-images   the catalog's normal-size images, into a folder outside the repository
                   (only with --download; without it, it says how many it would fetch)
    look-groups    how many printings share a look, per format
    embed          embed every printing (bank rows and held-out query views) with a recognizer
    crowding       the crowding measurements, from the embeddings
    scorer         bench data for browser/bench: a bank, its scoring graphs and a frame of looks

``games/mtg/README.md`` has the commands with their arguments, and where each writes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path


def _write_json(path: Path, value, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=indent, separators=None if indent else (",", ":"))
    partial = path.with_name(path.name + ".partial")
    partial.write_text(text + "\n", encoding="utf-8")
    partial.replace(path)


def _bulk(args: argparse.Namespace) -> int:
    from cardvision.mtg.scryfall import download_bulk

    path = download_bulk(args.out, args.type)
    print(f"bulk file: {path}")
    return 0


def _build_catalog(args: argparse.Namespace) -> int:
    from cardvision.mtg.catalog import build_catalog
    from cardvision.mtg.scryfall import read_cards

    out = Path(args.out)
    result = build_catalog(read_cards(args.bulk))
    _write_json(out / "catalog.json", result.catalog)
    _write_json(out / "images.json", result.images)
    _write_json(out / "exclusions.json", result.exclusions, indent=0)
    _write_json(out / "aliases.json", result.aliases, indent=1)
    summary = {"bulk": Path(args.bulk).name, **result.counts}
    _write_json(out / "catalog-summary.json", summary, indent=1)
    print(json.dumps(summary, indent=1))
    print(f"wrote {out / 'catalog.json'}: {result.counts['entries']} entries, {result.counts['identities']} identities")
    return 0


def _fetch_images(args: argparse.Namespace) -> int:
    from cardvision.core.catalog import load_catalog
    from cardvision.mtg.images import FetchStopped, fetch_images, pending

    catalog = load_catalog(args.catalog)
    urls = json.loads(Path(args.images).read_text(encoding="utf-8"))
    # Smallest scopes first, so that a partial run already covers whole formats.
    def order(card) -> tuple:
        legal, games = card.extra.get("legal", ()), card.extra.get("games", ())
        return ("standard" not in legal, "arena" not in games, "pioneer" not in legal, "modern" not in legal, card.id)

    files = {card.id: card.art for card in sorted(catalog, key=order) if card.art is not None}

    def progress(done: int, total: int) -> None:
        if done % 1000 == 0 or done == total:
            print(f"  {done} / {total}", flush=True)

    if not args.download:
        todo, wanted, present = pending(files, urls, args.art)
        print(f"fetch-images: {wanted} wanted, {present} already present, {len(todo)} to fetch from Scryfall "
              f"(about {len(todo) * 95e3 / 1e9:.1f} GB, {len(todo) / args.rate / 3600:.1f} h at {args.rate:g} a second). "
              "Nothing was fetched: pass --download to fetch them.")
        return 0
    try:
        report = fetch_images(files, urls, args.art, rate=args.rate, workers=args.workers, limit=args.limit,
                              progress=progress)
    except FetchStopped as error:
        print(f"fetch-images: stopped: {error}", file=sys.stderr)
        return 3
    print(f"fetch-images: {report.wanted} wanted, {report.present} already present, {len(report.fetched)} fetched, "
          f"{len(report.refetched)} fetched again (image changed), {len(report.missing)} not on the server, "
          f"{len(report.failed)} failed, {len(report.tombstoned)} newly tombstoned; "
          f"{report.bytes / 1e6:.1f} MB written")
    for card_id, reason in list(report.failed.items())[:20]:
        print(f"  failed {card_id}: {reason}", file=sys.stderr)
    return 1 if report.failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m cardvision.mtg", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    bulk = sub.add_parser("bulk", help="download a Scryfall bulk file")
    bulk.add_argument("--out", required=True, help="folder for the bulk file (outside the repository)")
    bulk.add_argument("--type", default="default_cards", help="bulk-data type (default: default_cards)")
    bulk.set_defaults(_run=_bulk)

    catalog = sub.add_parser("build-catalog", help="build catalog.json, images.json and exclusions.json")
    catalog.add_argument("--bulk", required=True, help="a default_cards bulk file (.jsonl.gz or .json)")
    catalog.add_argument("--out", required=True, help="folder for the outputs")
    catalog.set_defaults(_run=_build_catalog)

    fetch = sub.add_parser("fetch-images", help="download the catalog's normal-size images")
    fetch.add_argument("--catalog", required=True)
    fetch.add_argument("--images", required=True, help="the images.json build-catalog wrote")
    fetch.add_argument("--art", required=True, help="the image folder (outside the repository)")
    fetch.add_argument("--rate", type=float, default=10.0, help="requests a second (default 10)")
    fetch.add_argument("--workers", type=int, default=8)
    fetch.add_argument("--limit", type=int, default=None, help="fetch at most this many in this run")
    fetch.add_argument("--download", action="store_true",
                       help="fetch the images; without it, only say how many would be fetched")
    fetch.set_defaults(_run=_fetch_images)

    from cardvision.mtg import commands

    commands.register(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args._run(args)


if __name__ == "__main__":
    sys.exit(main())
