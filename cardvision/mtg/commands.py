# SPDX-License-Identifier: Apache-2.0
"""The measuring commands of ``python -m cardvision.mtg``: look groups, embedding, the crowding
curve, the scorer, and a base bank from the embeddings. Registered by ``cardvision.mtg.__main__``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# The random ladder: banks of 5,000 and 30,000 random printings and then every printing, scored
# on the first one's printings.
RANDOM_SIZES = (5000, 30000)


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    partial.replace(path)


def _look_groups(args: argparse.Namespace) -> int:
    from cardvision.core.catalog import load_catalog
    from cardvision.mtg.looks import look_report, markdown_table

    report = look_report(load_catalog(args.catalog))
    _write(Path(args.out), report)
    print(markdown_table(report))
    print(f"wrote {args.out}")
    return 0


def _embed(args: argparse.Namespace) -> int:
    from cardvision.core.catalog import load_catalog
    from cardvision.mtg.crowding import embed
    from cardvision.mtg.looks import scope_ids

    catalog = load_catalog(args.catalog)
    ids = scope_ids(catalog, args.scope) if args.scope else None

    def progress(done: int, total: int) -> None:
        print(f"  {done} / {total}", flush=True)

    result = embed(catalog, args.art, args.recognizer, args.out, ids=ids, workers=args.workers, progress=progress)
    print(json.dumps({key: value for key, value in result.items() if key != "failed"}))
    for card_id, error in list(result["failed"].items())[:20]:
        print(f"  failed {card_id}: {error}", file=sys.stderr)
    return 1 if result["failed"] else 0


def _crowding(args: argparse.Namespace) -> int:
    from cardvision.core.catalog import load_catalog
    from cardvision.mtg.crowding import load_embeddings, run_crowding

    catalog = load_catalog(args.catalog)
    ids, rows, queries = load_embeddings(args.embeddings)
    families = args.families.split(",")
    report = run_crowding(catalog, ids, rows, queries, tau=args.tau, delta=args.delta, seed=args.seed,
                          random_ladder=() if args.no_random else RANDOM_SIZES, families=families,
                          progress=lambda message: print(message, flush=True))
    report["embeddings"] = str(args.embeddings)
    _write(Path(args.out), report)
    for name, ladder in report["ladders"].items():
        for family, values in ladder["families"].items():
            print(f"{name} / {family}: {json.dumps(values['decision'])}")
    print(f"wrote {args.out}")
    return 0


def _scorer(args: argparse.Namespace) -> int:
    """Bench data: a bank in the runtime's JSON format, its identity groups, the scoring graphs
    for it and a frame of queries, so that the browser bench scores exactly the same numbers."""
    from cardvision.core.bank import encode_bank, fp16_roundtrip
    from cardvision.mtg.scorer import reference_scores, write_scorer

    out = Path(args.out)
    rng = np.random.default_rng(args.seed)
    rows = fp16_roundtrip(rng.standard_normal((args.rows, args.dim)).astype(np.float32))
    ids = [f"c{i:06d}" for i in range(args.rows)]
    # Identities of 1 to 6 printings, three on average, as in Magic's catalog.
    sizes = rng.integers(1, 6, size=args.rows)
    ident = np.repeat(np.arange(len(sizes)), sizes)[: args.rows].astype(np.int32)
    groups: dict[int, list[str]] = {}
    for card_id, key in zip(ids, ident):
        groups.setdefault(int(key), []).append(card_id)
    multi = [members for members in groups.values() if len(members) >= 2]
    bank = {"schemaVersion": 1, "kind": "embedding-bank", "game": "bench", "locale": "en", "version": 1,
            "model": "bench", "dim": args.dim, "dtype": "fp16", "preprocessing": "full128", "centerMask": False,
            "views": 16, "aggregation": "mean", **encode_bank(ids, rows)}
    bank["cardCount"] = bank["entryCount"] = args.rows
    out.mkdir(parents=True, exist_ok=True)
    (out / "bank.json").write_text(json.dumps(bank, separators=(",", ":")), encoding="utf-8")
    (out / "bank.f16").write_bytes(rows.astype("<f2").tobytes())
    _write(out / "identity-groups.json", {"schemaVersion": 1, "kind": "identity-groups", "version": 1,
                                          "groups": multi, "representative": {m: g[0] for g in multi for m in g[1:]}})
    picks = rng.integers(0, args.rows, size=args.looks)
    queries = rows[picks] + 0.04 * rng.standard_normal((args.looks, args.dim)).astype(np.float32)
    queries = (queries / np.linalg.norm(queries, axis=1, keepdims=True)).astype(np.float32)
    (out / "queries.f32").write_bytes(queries.astype("<f4").tobytes())
    best_row, best, other = reference_scores(rows, ident, queries)
    _write(out / "reference.json", {"bestRow": best_row.tolist(), "best": best.tolist(), "other": other.tolist()})
    for name, options in {"scorer-topk.onnx": {"top_k": args.top_k}, "scorer-argmax.onnx": {"with_topk": False},
                          "scorer-bank-input.onnx": {"top_k": args.top_k, "bank_as_input": True}}.items():
        write_scorer(rows, ident, out / name, **options)
    _write(out / "meta.json", {"rows": args.rows, "dim": args.dim, "looks": args.looks, "topK": args.top_k,
                               "identities": len(groups), "seed": args.seed})
    print(f"wrote bench data for {args.rows} rows and {args.looks} looks in {out}")
    return 0


def _bank_from_embeddings(args: argparse.Namespace) -> int:
    from cardvision.bank import build
    from cardvision.core.bank import encode_bank
    from cardvision.core.catalog import load_catalog
    from cardvision.mtg.crowding import load_embeddings

    catalog = load_catalog(args.catalog)
    found: dict[str, np.ndarray] = {}
    for folder in args.embeddings:
        ids, rows, _ = load_embeddings(folder)
        found.update(zip(ids, rows))
    art = Path(args.art)
    ids, rows, versions = [], [], []
    for card in catalog:
        if card.art is None or card.id not in found or not (art / card.art).is_file():
            continue
        ids.append(card.id)
        rows.append(found[card.id])
        versions.append(build.art_version((art / card.art).read_bytes()))
    # The header build_bank writes (cardvision.bank.build), for the rows embed made exactly as it does.
    descriptor = build.preprocessing_descriptor(build.FULL128)
    header = {"schemaVersion": build.BANK_SCHEMA_VERSION, "kind": build.BANK_KIND, "game": catalog.game,
              "locale": catalog.locale, "version": 1, "model": args.model, "dim": int(rows[0].shape[0]),
              "dtype": build.VECTOR_DTYPE_NAME, "preprocessing": descriptor, "centerMask": descriptor["centerMask"],
              "views": build.VIEWS_PER_CARD, "aggregation": build.AGGREGATION,
              "cardCount": len(set(ids)), "entryCount": len(ids)}
    _write(Path(args.out), encode_bank(ids, np.stack(rows), extra=header, art_versions=versions))
    print(f"wrote {args.out}: {len(ids)} rows of the catalog's {sum(1 for card in catalog if card.art)} cards with art; "
          "pass it to cardvision build-bank --base, which checks every row and embeds the rest")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    """Add the measuring commands to the parser's subcommands."""
    looks = sub.add_parser("look-groups", help="look-group statistics per format")
    looks.add_argument("--catalog", required=True)
    looks.add_argument("--out", required=True, help="the report, JSON")
    looks.set_defaults(_run=_look_groups)

    embed = sub.add_parser("embed", help="bank rows and held-out query views for every printing")
    embed.add_argument("--catalog", required=True)
    embed.add_argument("--art", required=True)
    embed.add_argument("--recognizer", required=True)
    embed.add_argument("--out", required=True, help="folder for the embedding shards (outside the repository)")
    embed.add_argument("--scope", default=None, help="only this scope's printings (see cardvision.mtg.looks)")
    embed.add_argument("--workers", type=int, default=4)
    embed.set_defaults(_run=_embed)

    crowding = sub.add_parser("crowding", help="the crowding measurements from the embeddings")
    crowding.add_argument("--catalog", required=True)
    crowding.add_argument("--embeddings", required=True)
    crowding.add_argument("--out", required=True, help="the report, JSON")
    crowding.add_argument("--families", default="stream,bank", help="query families (see cardvision.mtg.crowding)")
    crowding.add_argument("--no-random", action="store_true", help="leave out the random ladder")
    crowding.add_argument("--tau", type=float, default=0.62)
    crowding.add_argument("--delta", type=float, default=0.05)
    crowding.add_argument("--seed", type=int, default=0)
    crowding.set_defaults(_run=_crowding)

    base = sub.add_parser("bank-from-embeddings",
                          help="a base bank for build-bank --base, from the embed command's shards")
    base.add_argument("--catalog", required=True)
    base.add_argument("--art", required=True, help="the image folder, for each row's artV")
    base.add_argument("--embeddings", required=True, nargs="+", help="one or more folders of embedding shards")
    base.add_argument("--model", required=True, help="the recognizer's label, as the release will pin it")
    base.add_argument("--out", required=True)
    base.set_defaults(_run=_bank_from_embeddings)

    scorer = sub.add_parser("scorer", help="bench data: a bank, its scoring graphs and a frame of looks")
    scorer.add_argument("--rows", type=int, required=True)
    scorer.add_argument("--dim", type=int, default=128)
    scorer.add_argument("--looks", type=int, default=56)
    scorer.add_argument("--top-k", type=int, default=8)
    scorer.add_argument("--seed", type=int, default=0)
    scorer.add_argument("--out", required=True)
    scorer.set_defaults(_run=_scorer)
