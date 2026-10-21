<!-- SPDX-License-Identifier: Apache-2.0 -->

# Magic: The Gathering

Seerglass's Magic tools. The code is in [`cardvision/mtg/`](../../cardvision/mtg/), apart from
the upstream code, so that upstream merges stay clean. This folder holds the accept rules for
Arena and for paper and the footage sources, and no catalog: the catalog is built from
Scryfall's bulk data, and is about 80 MB.

Phase 0 measured how hard Magic is for the released recognizer before anything is trained.
Phase 1 started on MTG Arena footage (see [Arena](#arena)) and now aims at streams of paper
cards (see [Paper](#paper)). Nothing here is trained on Magic yet,
and every figure from a model is from the released **Riftbound** recognizer, used as a
stand-in.

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
wrong share of named cards to 1, 2 and 5 %. A bank's *operating point* is the one that names
the most while keeping both the wrong share of named cards and the share of open-set cards
named to 2 %. Banks are compared on the same queries, at their own operating points and at the
first bank's, with McNemar's test and a bootstrap clustered by illustration.

What it is not: accuracy on video. The queries are synthetic, with no detector error,
perspective, glare, sleeve, overlay or occlusion.

The results of the first full run are in the Phase 0 report, linked from the pull request that
added this folder.

## Arena

Phase 1 aims at MTG Arena footage first, names only. Its tools build a catalog and a bank for
Arena from the full catalog, and read Arena's own log of a game.

```sh
python -m cardvision.mtg arena-catalog --catalog "$WORK/mtg/catalog.json" --images "$WORK/mtg/images.json" \
    --bulk "$WORK"/bulk/default-cards-*.jsonl.gz --out "$WORK/arena"
python -m cardvision.mtg arena-tiles --catalog "$WORK/arena/catalog.json" --art "$WORK/art"
cardvision build-bank --catalog "$REL/catalog.json" --art "$WORK/art" --recognizer "$REL/recognizer.onnx" \
    --out "$REL/bank.json" --groups-out "$REL/identity-groups.json" --skip-missing-art
cardvision build-release --detector "$REL/detector.onnx" --recognizer "$REL/recognizer.onnx" \
    --bank "$REL/bank.json" --catalog "$REL/catalog.json" --accept "$REL/accept.json" --out "$REL/release.json"
python -m cardvision.mtg arena-log Player.log --out game-1.json
```

`$REL` is a release folder holding copies of `$WORK/arena/catalog.json`,
[`accept.json`](accept.json) and the two models.

`build-bank` embeds about four cards a second on four cores, so a bank of the Arena pool and
its tiles takes about three hours. `embed` makes the same rows, value for value, about four
times faster (a test checks that the two agree), and `bank-from-embeddings` turns its shards
into a base bank. `build-bank --base` then checks every row against its image, embeds what is
missing and writes the identity groups:

```sh
python -m cardvision.mtg embed --catalog "$REL/catalog.json" --art "$WORK/art" \
    --recognizer "$REL/recognizer.onnx" --out "$WORK/emb-arena"
python -m cardvision.mtg bank-from-embeddings --catalog "$REL/catalog.json" --art "$WORK/art" \
    --embeddings "$WORK/emb-arena" --model recognizer --out "$WORK/base.json"
cardvision build-bank --catalog "$REL/catalog.json" --art "$WORK/art" --recognizer "$REL/recognizer.onnx" \
    --base "$WORK/base.json" --out "$REL/bank.json" --groups-out "$REL/identity-groups.json" --skip-missing-art
```

| Command | What it does | Writes |
| --- | --- | --- |
| `arena-catalog` | The Arena catalog: every face of the Arena pool, a tile entry for each permanent, and the tokens of the pool's sets. See [the Arena catalog](#the-arena-catalog). | `catalog.json`, `images.json` (the pool's and the tokens' image addresses), `arena-summary.json` |
| `arena-tiles` | Each tile entry's picture, cut from its card's image in the art folder. | `tiles/` in the art folder |
| `arena-log` | A table of the cards in play, from a `Player.log`. See [Arena's log](#arenas-log). | a JSON table |
| `bank-from-embeddings` | A base bank from `embed`'s shards: their bank rows, with each image's `artV`. | a bank file |

### The Arena catalog

Arena draws a card whole in the hand (where only its top part shows), on the stack and in the
inspect view, but draws a permanent on the battlefield as a tile, wider than tall: the name
bar over the art, and the power and toughness, no rules text. A bank row made from the whole
card does not match a tile, so each permanent face also has a *tile entry*, `<face id>:art`,
whose picture runs from the top of the name bar to the bottom of the art box. It is the same
identity as its card (`printingOf` the identity's representative, `extra.artOf` the face), so
the two rows never narrow each other's margin. The shape (`extra.tile` is `name-art-v1`) was
measured on public Arena footage; release v0 used a square from the middle of the art
(`square-art-v0`). Instants and sorceries have no tile entry.

Tokens come from the token sets of the pool's sets, since Scryfall marks almost none as being
on Arena. A token named like a card of the pool joins that card's identity; any other token is
named `<name> token` (`Soldier token`), one identity per name. Emblems and helper cards (The
Monarch, Day and Night) are left out. The tokens' images are not fetched by the Phase 0 run:
`fetch-images` with the Arena catalog and its `images.json` fetches them, under the same
`--download` rule as above, and `build-bank --skip-missing-art` leaves out the tokens without
one until then.

Each identity's representative is chosen again among the Arena entries, and `extra` keeps only
the fields the Arena tools read. To label frames, give the labelling tool a catalog made with
`--no-tiles`: labels name the face, and a tile entry would show each permanent twice in the
picker.

[`accept.json`](accept.json) is the Arena accept rule: tau 0.72 and delta 0.07, Phase 0's
operating point for the Arena pool, with the rotation retry off. A tapped permanent on Arena
tilts by a few degrees and never turns a quarter turn, so the retry finds nothing, and turned
off it removes a third of the names given to cards outside the bank at the same rate of
correct names (on Phase 0's synthetic views). It is to be calibrated again on labelled Arena
frames. [`sources.json`](sources.json) adds the source type `digital/mtg-arena` and marks its
hand row; the row's start (85 % of the way down the frame) is also to be measured.

### Arena's log

With *Detailed Logs (Plugin Support)* on (Arena's Options, Account), Arena writes what its
game server sends into `Player.log`: the cards in each zone, by Arena id (Scryfall's
`arena_id`), but not where they are on the screen. `arena-log` turns it into a table of each
moment at which what is in view changed: per zone, each card's Arena id, whose it is, whether
it is tapped, and the time. It is the candidate list of a recording, and with the recording's
start time, what was in play at each frame.

The log holds personal data: player names, account and match ids, chat. The reader keeps none
of it, only card ids, zones, seats and times, and it runs where the log is:
[`cardvision/mtg/arena_log.py`](../../cardvision/mtg/arena_log.py) needs nothing but Python
3.10, so `python arena_log.py Player.log --out game-1.json` works on the player's own computer
without installing this repository. Share the table, never the log. Arena replaces
`Player.log` each time it starts, keeping one `Player-prev.log`, so copy it after each session.
Card names come from Scryfall's `arena_id`; Arena's own card database and files are not read.

## Paper

Phase 1 now aims at streams of paper cards, tournament coverage first. A paper release's bank
holds every printing on paper, and the tokens:

```sh
python -m cardvision.mtg paper-catalog --catalog "$WORK/mtg/catalog.json" --images "$WORK/mtg/images.json" \
    --bulk "$WORK"/bulk/default-cards-*.jsonl.gz --out "$WORK/paper"
python -m cardvision.mtg release-catalog --catalog "$WORK/paper/catalog.json" --out "$REL/catalog.json"
cardvision build-bank --catalog "$REL/catalog.json" --art "$WORK/art" --recognizer "$REL/recognizer.onnx" \
    --base "$WORK/base.json" --out "$REL/bank.json" --groups-out "$REL/identity-groups.json" --skip-missing-art
cardvision build-release --detector "$REL/detector.onnx" --recognizer "$REL/recognizer.onnx" \
    --bank "$REL/bank.json" --catalog "$REL/catalog.json" --accept "$REL/accept.json" --out "$REL/release.json"
```

`$REL` is a release folder holding a copy of [`accept-paper.json`](accept-paper.json), named
`accept.json`, and the two models; `$WORK/base.json` is made by `embed` and
`bank-from-embeddings` as [for Arena](#arena).

| Command | What it does | Writes |
| --- | --- | --- |
| `paper-catalog` | The paper catalog: every printing on paper, and the tokens of every paper set's token sets. See [the paper catalog](#the-paper-catalog). | `catalog.json`, `images.json`, `paper-summary.json` |
| `label-catalog` | The catalog for the [labelling tool](../../labeler/README.md): the same cut, with each entry's `art` its image's address on Scryfall's image host, so that the tool needs no image folder. Give it the paper catalog, or an Arena catalog made with `--no-tiles`. | a catalog file |
| `release-catalog` | The catalog a release ships: each entry cut to what the browser runtime reads (`id`, `name`, `kind`, `printingOf`, `art`, and `token` and `facets` where an entry has them). | a catalog file |

### The paper catalog

Every entry of the full catalog whose printing exists on paper: the 3,594 faces only on Arena
(Alchemy's digital cards, on the bulk file of 2026-10-21) never lie on a table, and a bank row
for one could only ever win wrongly. Tokens are taken and grouped as for
[Arena](#the-arena-catalog), from the token sets of every set with a paper printing: a token
named like a card joins that card's identity, and any other is named `<name> token`. A paper
card is a whole card wherever it lies, so there are no tile entries. On the bulk file of
2026-10-21: 101,401 faces and 2,836 token faces, 34,244 identities (387 of them tokens).
`fetch-images` with the paper catalog and its `images.json` fetches only the images not yet
in the art folder (after a Phase 0 run, the tokens'), under the `--download` rule above.

### The release catalog

The runtime reads nothing in `extra`, and holds every entry in memory. On the full catalog the
cut takes the file from 80.5 MB to 13.7 MB (4.1 MB gzipped), and, measured in Node, its parsed
form from about 311 MB of memory to 33 MB. Ids, names, `printingOf` and `art` are unchanged,
so the identities are the same: a bank built with either catalog is valid with the other. An
Arena release can be cut the same way.

[`accept-paper.json`](accept-paper.json) is the paper accept rule: tau 0.72 and delta 0.07,
Arena's values until labelled paper frames calibrate it, with the rotation retry on, since a
tapped paper card turns a quarter turn.

Tournament coverage is the built-in source type `physical/tournament`, and a game filmed by a
player's own camera is `physical/webcam`. [`sources.json`](sources.json) adds `physical/show`:
a produced show filmed from several cameras, such as a Commander show.
