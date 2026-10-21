<!-- SPDX-License-Identifier: Apache-2.0 -->

# Seerglass

Card recognition for Magic: The Gathering in streams and videos. Rest the pointer on a card in
a YouTube or Twitch video and Seerglass names it, and its exact printing when the picture shows
it. A browser extension comes first; a phone app later.

Seerglass is based on RiftSight's open vision model: it is a fork of
[riftsight-vision](https://github.com/ryangswe/riftsight-vision), which recognizes Riftbound
cards in gameplay video. It is not RiftSight, and RiftSight does not stand behind it.

**Not affiliated with or endorsed by Wizards of the Coast.** Magic: The Gathering, the card
names and the card art belong to Wizards of the Coast. Card data comes from
[Scryfall](https://scryfall.com). This repository holds no card art.

## Status

**Early.** Everything below this section, and the documentation it links to, is still
riftsight-vision's: the tools, the Riftbound models and their measurements. The Magic parts are
being built:

- a Magic catalog generated from Scryfall's bulk data, one entry per printing;
- a bank of every printing, built with the recognizer;
- detector and recognizer fine-tuned on Magic footage (MTG Arena, MTGO, paper, event coverage);
- a two-level answer: the card's name, then its printing when the frame can tell printings
  apart, and "printing uncertain" when it cannot;
- a browser extension for YouTube and Twitch.

The Python package keeps its upstream name, `cardvision`.

## What it is built on

Two small models find the cards in a video frame and name them. A detector boxes card faces
and backs; a recognizer turns each face into a vector that is matched against a bank built from
card images; an accept rule shows nothing when the match is not trusted. Adding cards means
adding rows to the bank, with no retraining. [docs/how-it-works.md](docs/how-it-works.md) is
the full description.

- **Try upstream's demo with nothing to install:** <https://ryangswe.github.io/riftsight-vision/>.
- **New here?** [docs/start-here.md](docs/start-here.md) picks the page for you.
- **Have a spreadsheet of your cards?** [docs/your-game.md](docs/your-game.md) takes it, your
  card images and a video to card names on that video.

## How well it works

These are upstream's figures, for Riftbound; Magic has none yet. On a small public evaluation
set used during development, which flatters them somewhat, the released models named 92.5 % of readable card faces right, and 1.6 % of the names shown were
wrong. The [model card](docs/model-card.md#evaluation) has the figures by source and the limits.

## Quickstart

This runs the whole pipeline on a toy game drawn from colored shapes: no card art, no video,
no download beyond the Python packages. You need Python 3.10 or later (macOS, Linux, or Ubuntu
under WSL 2 on Windows), and Node 22 or later for the browser part. Paste this into the terminal:

```
git clone https://github.com/ryangswe/riftsight-vision.git
cd riftsight-vision
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
export WORK="$HOME/Downloads/cardvision-toy"
mkdir -p "$WORK"
python examples/toy_quickstart.py "$WORK/toy"
```

The last line it prints should be `1 frame, 6 faces, 3 named`. The toy's models are stand-ins
with no skill: they show what the tools do, not how well the released models work.
**[docs/quickstart.md](docs/quickstart.md) is the full walkthrough:** Windows, what each line
means, building a bank, naming the cards, and the demo page in a browser.

## Documentation

| I want to… | Read |
| --- | --- |
| Start here: pick the page for me | [docs/start-here.md](docs/start-here.md) |
| Run the toy quickstart, step by step | [docs/quickstart.md](docs/quickstart.md) |
| Get names on my own game's video, from my card list and images | [docs/your-game.md](docs/your-game.md) |
| Make a browser extension for my game | [docs/extension.md](docs/extension.md) |
| Train the models for my game | [docs/train.md](docs/train.md) |
| Measure, with numbers I can state | [docs/measure.md](docs/measure.md) |

Everything else: [docs/reference.md](docs/reference.md). A coding agent starts with [AGENTS.md](AGENTS.md).

## Built with this

Other games built on riftsight-vision, listed upstream. Catalogs, banks and weights for other
games live in their authors' own repositories.

- **[Kenbunshoku](https://github.com/ryangswe/kenbunshoku)**, a browser extension for One Piece
  Card Game videos, built on the browser runtime. Its catalog, bank, weights and labels are on
  Hugging Face ([models](https://huggingface.co/JuicyKaraageNo1/kenbunshoku-onepiece),
  [labels](https://huggingface.co/datasets/JuicyKaraageNo1/kenbunshoku-onepiece-labels)); its measurements are in the
  [adapt guide](docs/adapt-your-game.md#measured-on-another-game-the-one-piece-card-game).

## How this was built

From riftsight-vision's author: I'm a solo software engineer with some machine learning
experience; computer vision isn't my core competency. I built this with heavy use of AI coding
tools. It was checked with tests, with comparisons against the code that trained the released
models, with a private held-out test scored once, and with outside reviews by other AI models
and fresh AI sessions, not by people.
[docs/how-this-was-built.md](docs/how-this-was-built.md) says what each check does and does not show.

## Support

Seerglass is maintained by [DioPires](https://github.com/DioPires). Bugs and wrong
recognitions go to this repository's issues: a video id and a time, never a frame or a
screenshot. See [SUPPORT.md](SUPPORT.md).

Upstream riftsight-vision is maintained by its own author; please do not send them questions
about Seerglass.

## License

- **Code and weights:** Apache-2.0 ([LICENSE](LICENSE)), for this project's own work.
- **Bank, identity groups and catalog:** Apache-2.0 for this project's own work in the bank and
  identity groups, and for the catalog's selection and arrangement; the card names, ids and
  other card data in the catalog are not licensed here ([NOTICE](NOTICE)).
- **Label manifests:** CC BY 4.0, for the annotations only ([LICENSE-DATA](LICENSE-DATA)).
- **Starting points and attributions:** [NOTICE](NOTICE). The detector starts from YOLOX-Nano
  weights by Megvii; the recognizer from torchvision's MobileNetV3-Small weights.
- **What the grant covers:** this project's own work. The bank and the weights derive from
  card images that this repository does not contain and does not license ([NOTICE](NOTICE),
  [docs/faq.md](docs/faq.md), the [license table](PROVENANCE.md#licenses-by-artifact)).

## Trademark

The name Seerglass is this project's own. The Apache license does not cover the RiftSight name
or logo, and Seerglass does not use them beyond saying, truthfully, that it is based on
RiftSight's open vision model ([TRADEMARK.md](TRADEMARK.md)).

## Not endorsed

Seerglass is a fan project. It is not endorsed by Wizards of the Coast, Riot Games or the
RiftSight project. Magic: The Gathering and its card names and art belong to Wizards of the
Coast; Riftbound and its card names and art belong to Riot Games.
