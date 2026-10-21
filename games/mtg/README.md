<!-- SPDX-License-Identifier: Apache-2.0 -->

# Magic: The Gathering

Seerglass's Magic tools. The code is in [`cardvision/mtg/`](../../cardvision/mtg/), apart from
the upstream code, so that upstream merges stay clean. This folder holds no catalog yet: the
catalog is built from Scryfall's bulk data, and is about 80 MB.

This is Phase 0: measuring how hard Magic is for the released recognizer before anything is
trained. Nothing here is trained on Magic, and every figure from a model is from the released
**Riftbound** recognizer, used as a stand-in.

## The commands, in order

Every command writes outside the checkout. `$WORK` below is such a folder.

```sh
python -m cardvision.mtg bulk --out "$WORK/bulk"
python -m cardvision.mtg build-catalog --bulk "$WORK"/bulk/default-cards-*.jsonl.gz --out "$WORK/mtg"
python -m cardvision.mtg look-groups --catalog "$WORK/mtg/catalog.json" --out "$WORK/mtg/look-groups.json"
python -m cardvision.mtg fetch-images --catalog "$WORK/mtg/catalog.json" --images "$WORK/mtg/images.json" \
    --art "$WORK/art" --download
python -m cardvision.mtg embed --catalog "$WORK/mtg/catalog.json" --art "$WORK/art" \
    --recognizer "$WORK/weights/recognizer.onnx" --out "$WORK/emb"
python -m cardvision.mtg crowding --catalog "$WORK/mtg/catalog.json" --embeddings "$WORK/emb" \
    --out "$WORK/crowding.json"
python -m cardvision.mtg scorer --rows 100000 --out "$WORK/bench/100000"
```

| Command | What it does | Writes |
| --- | --- | --- |
| `bulk` | Downloads Scryfall's `default_cards` bulk file (every printing; about 80 MB compressed), under the name the server gives it, which carries its date (`default-cards-20261021090544.jsonl.gz`). A file already there is not downloaded again. | the `.jsonl.gz` file and a `.meta.json` beside it |
| `build-catalog` | The catalog in this repository's format, from the bulk file. See [the catalog](#the-catalog). | `catalog.json`, `images.json` (id to image address), `exclusions.json`, `aliases.json`, `catalog-summary.json` |
| `look-groups` | How many printings share a look, per format. No model. See [look groups](#look-groups). | a JSON report, and a Markdown table on the terminal |
| `fetch-images` | Each entry's `normal` image (488 × 680 JPEG) from Scryfall's image host, at ten requests a second. **Only with `--download`**; without it, it says how many it would fetch and how long that would take. Resumable: a SQLite state file in the art folder records what was fetched, at which image version. | one JPEG per entry, and `.fetch-state.sqlite` |
| `embed` | Each entry's bank row (exactly as `cardvision build-bank` makes one: 16 views, averaged) and held-out query views, with a recognizer. Embeds only what is new, so it can follow a running fetch. About 15 cards a second on four cores. | `shard-NNNNN.json` with `.rows.f32` and `.queries.f32` |
| `crowding` | The crowding measurements, from the embeddings. See [crowding](#crowding). | a JSON report |
| `scorer` | Bench data for [`browser/bench/`](../../browser/bench/): a made-up bank in the runtime's format, its scoring graphs and a frame of looks. | a folder |

### Card art and Scryfall

AGENTS.md says not to fetch card art (rule 4). `fetch-images` is the one exception, asked for
by the project's owner, and it fetches only when given `--download`. The images are © Wizards of
the Coast and stay out of git: the art folder and every output above go outside the checkout.
Scryfall asks for a `User-Agent` and for at most ten requests a second; the fetcher sends
`Seerglass/0.1 (+https://github.com/DioPires/seerglass)` and keeps to that rate, and a 429 or
503 pauses it for as long as the server asks. Card data from Scryfall is used under
[its terms](https://scryfall.com/docs/api).

## The catalog

One entry per **face** of a printing: a printing's id is `set:collector_number`
(`m19:314`), and the back face of a double-faced card adds `:b` (`znr:134:b`). Split, flip,
adventure and aftermath cards are one picture and one entry. The catalog's `identity` is
`printing`: each Oracle card is one identity, its earliest printing (with art) is the
representative, and every other printing points at it with `printingOf`.

Kept: English printings that exist on paper or on Arena (`games`). Left out, each with its
reason in `exclusions.json`: other languages; printings only on Magic Online; tokens, emblems,
art cards, Dungeons, substitute cards, oversized cards, planes, schemes, vanguards, front
cards and sticker sheets.

Aliases: a printing whose illustration another Oracle card also uses (an Un-set variant, a
misprint) joins that card's identity only when the two have a printing in one environment
(paper or Arena); otherwise it stays apart and carries `extra.sharesArtWith`. Alchemy's
rebalanced `A-` cards join the card they rebalance. `aliases.json` lists every alias.

Each entry's `extra` carries what the later stages need: `set`, `collector_number`,
`scryfall_id`, `face`, `oracle_id`, `illustration_id`, `frame`, `frame_effects`, `border_color`,
`full_art`, `textless`, `promo_types`, `finishes`, `foilOnly` (no non-foil finish: etched-only
counts), `lang`, `arena_id`, `mtgo_id`, `layout`, `games`, `set_type`, `digital`,
`released_at`, `image_status` and `legal` (the formats it is legal in).

## Look groups

Two printings *look alike* when they share the illustration, the frame, the border, full-art,
textless and the frame effects: what still tells them apart is the set symbol, the collector
line or a promo stamp, a few pixels on a stream. A printing alone in its look group can be
named exactly by a model that sees its picture well; one with a *visual twin* cannot, from the
picture alone. The definitions, and the two other keys measured for comparison, are in
[`cardvision/mtg/looks.py`](../../cardvision/mtg/looks.py).

## Crowding

`embed` makes, for every entry, its bank row and six query views no bank row has seen: four
from the bank's own view synthesizer (the `bank` family) and two sized and damaged as a card on
a stream is (the `stream` family: a short side of 40 to 70 pixels, the box off by up to 8 %,
blur and strong compression). Each query is embedded in its four orientations, so the
runtime's rotation retry is applied as it is.

`crowding` scores one fixed set of queries against nested banks that all hold it, so that
what changes between two banks is the bank alone:

- **arena**: the Arena printings of Standard cards, against Standard-on-Arena, then the Arena
  pool, then every printing;
- **paper**: the printings of Standard cards, against Standard, Pioneer, Modern, then every
  printing;
- **random**: 5,000 random printings, against 5,000, 30,000, then every printing.

Each ladder also has a control bank of random printings the size of its second rung. Each
query is scored three ways: against the bank as it is (**own row in**: an upper bound, since
a query is a copy of its own row's scan); with its own row left out (**own row out**: another
scan of what else the bank holds of the card, the realistic case when the card has another
printing of the same look); and, for cards the bank does not hold, how many it names anyway
(**open set**). Each is reported at the accept rule (tau 0.62, delta 0.05, rotation retry on),
and over a grid of tau 0.50 to 0.90 and delta 0 to 0.15, with the best point that keeps the
wrong share of named cards to 1, 2 and 5 %. Banks are compared on the same queries with
McNemar's test and a bootstrap clustered by illustration.

What it is not: accuracy on video. The queries are synthetic, with no detector error,
perspective, glare, sleeve, overlay or occlusion.

The results of the first full run are in the Phase 0 report, linked from the pull request that
added this folder.
