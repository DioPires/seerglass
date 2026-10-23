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
    bank-from-embeddings
                   a base bank for cardvision build-bank --base, from embed's shards
    arena-catalog  the Arena catalog: the Arena pool, a tile entry per permanent, and tokens
    arena-tiles    the tile entries' pictures, cut from the card images
    arena-log      a table of the cards in play, from an MTG Arena Player.log (run it where the log is)
    paper-catalog  the paper catalog: every printing on paper, and tokens
    release-catalog
                   the catalog a release ships: each entry cut to what the runtime reads
    label-catalog  the catalog for the labelling tool, its pictures loaded from Scryfall
    cut-frames     frames to label, cut from a video the way cardvision fetch cuts them again
    decklist-names a deck list with front-face names made full, for labeler/decklist_from_text.py

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


def _pool_catalog(args: argparse.Namespace, environment: str, tiles: bool) -> int:
    from cardvision.mtg.arena import pool_catalog, token_entries
    from cardvision.mtg.scryfall import read_cards

    full = json.loads(Path(args.catalog).read_text(encoding="utf-8"))
    sets = {entry["extra"]["set"] for entry in full["cards"] if environment in entry["extra"].get("games", ())}
    tokens, token_urls = token_entries(read_cards(args.bulk), sets) if args.bulk else ([], {})
    result = pool_catalog(full, tokens, environment=environment, tiles=tiles)
    out = Path(args.out)
    _write_json(out / "catalog.json", result.catalog)
    face_urls = json.loads(Path(args.images).read_text(encoding="utf-8")) if args.images else {}
    kept = {entry["id"] for entry in result.catalog["cards"]}
    urls = {key: url for key, url in {**face_urls, **token_urls}.items() if key in kept}
    _write_json(out / "images.json", urls)
    _write_json(out / f"{environment}-summary.json", result.counts, indent=1)
    print(json.dumps(result.counts, indent=1))
    print(f"wrote {out / 'catalog.json'}: {result.counts['entries']} entries, {result.counts['identities']} identities; "
          f"{out / 'images.json'}: {len(urls)} image addresses")
    return 0


def _arena_catalog(args: argparse.Namespace) -> int:
    return _pool_catalog(args, "arena", tiles=not args.no_tiles)


def _paper_catalog(args: argparse.Namespace) -> int:
    return _pool_catalog(args, "paper", tiles=False)


def _arena_tiles(args: argparse.Namespace) -> int:
    from cardvision.mtg.arena import render_tiles

    catalog = json.loads(Path(args.catalog).read_text(encoding="utf-8"))
    report = render_tiles(catalog, args.art, overwrite=args.overwrite)
    print(f"arena-tiles: {report.written} written, {report.present} already there, "
          f"{len(report.missing_source)} without their card's image")
    return 0


def _arena_log(args: argparse.Namespace) -> int:
    from cardvision.mtg.arena_log import main as log_main

    return log_main([args.log, "--out", args.out])


def _release_catalog(args: argparse.Namespace) -> int:
    from cardvision.mtg.catalog import release_catalog

    full = Path(args.catalog)
    result = release_catalog(json.loads(full.read_text(encoding="utf-8")))
    out = Path(args.out)
    _write_json(out, result)
    print(f"wrote {out}: {len(result['cards'])} entries, {out.stat().st_size / 1e6:.1f} MB "
          f"(from {full.stat().st_size / 1e6:.1f} MB)")
    return 0


def _label_catalog(args: argparse.Namespace) -> int:
    from cardvision.mtg.catalog import label_catalog

    document = json.loads(Path(args.catalog).read_text(encoding="utf-8"))
    images = json.loads(Path(args.images).read_text(encoding="utf-8"))
    result = label_catalog(document, images)
    out = Path(args.out)
    _write_json(out, result)
    without = sum(1 for entry in result["cards"] if entry["art"] is None)
    print(f"wrote {out}: {len(result['cards'])} entries, {without} without a picture")
    return 0


def _cut_frames(args: argparse.Namespace) -> int:
    from cardvision.mtg.frames import cut_frames

    try:
        cut_frames(args.platform, args.video, args.t, args.out)
    except (ValueError, RuntimeError) as error:
        print(f"cut-frames: {error}", file=sys.stderr)
        return 2
    return 0


def _decklist_names(args: argparse.Namespace) -> int:
    from cardvision.mtg.decklists import full_names

    document = json.loads(Path(args.catalog).read_text(encoding="utf-8"))
    text, rewritten = full_names(Path(args.input).read_text(encoding="utf-8"), document["cards"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    for name, whole in rewritten:
        print(f"{name} -> {whole}")
    print(f"wrote {out}: {len(rewritten)} name(s) made full")
    return 0


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

    arena = sub.add_parser("arena-catalog", help="the Arena catalog: Arena pool, tile entries and tokens")
    arena.add_argument("--catalog", required=True, help="the full catalog build-catalog wrote")
    arena.add_argument("--images", default=None, help="the full catalog's images.json, to keep the pool's addresses")
    arena.add_argument("--bulk", default=None, help="the default_cards bulk file, for the tokens (none without it)")
    arena.add_argument("--no-tiles", action="store_true", help="leave out the tile entries")
    arena.add_argument("--out", required=True, help="folder for catalog.json, images.json and arena-summary.json")
    arena.set_defaults(_run=_arena_catalog)

    tiles = sub.add_parser("arena-tiles", help="cut the tile entries' pictures from the card images")
    tiles.add_argument("--catalog", required=True, help="an Arena catalog")
    tiles.add_argument("--art", required=True, help="the image folder; tiles are written under its tiles/ folder")
    tiles.add_argument("--overwrite", action="store_true", help="cut the tiles that are already there again")
    tiles.set_defaults(_run=_arena_tiles)

    log = sub.add_parser("arena-log", help="the cards in play, from an MTG Arena Player.log")
    log.add_argument("log", help="a Player.log copied after the session")
    log.add_argument("--out", required=True, help="the table, JSON")
    log.set_defaults(_run=_arena_log)

    paper = sub.add_parser("paper-catalog", help="the paper catalog: every paper printing, and tokens")
    paper.add_argument("--catalog", required=True, help="the full catalog build-catalog wrote")
    paper.add_argument("--images", default=None, help="the full catalog's images.json, to keep the faces' addresses")
    paper.add_argument("--bulk", default=None, help="the default_cards bulk file, for the tokens (none without it)")
    paper.add_argument("--out", required=True, help="folder for catalog.json, images.json and paper-summary.json")
    paper.set_defaults(_run=_paper_catalog)

    release = sub.add_parser("release-catalog", help="the catalog a release ships, cut to what the runtime reads")
    release.add_argument("--catalog", required=True, help="a catalog: the full one, or the Arena one")
    release.add_argument("--out", required=True, help="the cut catalog, for the release folder")
    release.set_defaults(_run=_release_catalog)

    label = sub.add_parser("label-catalog", help="the catalog for the labelling tool, pictures from Scryfall")
    label.add_argument("--catalog", required=True, help="a catalog: the paper one, or the Arena one made with --no-tiles")
    label.add_argument("--images", required=True, help="that catalog's images.json")
    label.add_argument("--out", required=True, help="the labelling catalog")
    label.set_defaults(_run=_label_catalog)

    cut = sub.add_parser("cut-frames", help="frames to label, cut the way cardvision fetch cuts them again")
    cut.add_argument("--platform", required=True, help="where the video is: twitch or youtube")
    cut.add_argument("--video", required=True, help="the video's id on its platform")
    cut.add_argument("--t", required=True, type=float, action="append", help="a time in seconds (repeatable)")
    cut.add_argument("--out", required=True, help="the frames folder (outside the repository)")
    cut.set_defaults(_run=_cut_frames)

    names = sub.add_parser("decklist-names", help="a deck list with front-face names made full")
    names.add_argument("--catalog", required=True, help="the catalog the labelling tool uses")
    names.add_argument("--in", dest="input", required=True, help="the deck list as the site wrote it")
    names.add_argument("--out", required=True, help="the deck list to give decklist_from_text.py")
    names.set_defaults(_run=_decklist_names)

    from cardvision.mtg import commands

    commands.register(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args._run(args)


if __name__ == "__main__":
    sys.exit(main())
