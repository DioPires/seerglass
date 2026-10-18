<!-- SPDX-License-Identifier: Apache-2.0 -->

# riftsight-vision

Open weights, labels and tools for recognizing trading cards in gameplay video: built for
Riftbound, laid out so you can point it at your own game.

This is the card recognition that the RiftSight browser extension ships. Two small models
find the cards in a video frame and name them, so that a viewer can rest the pointer on a
card and read it.

**A fan project. Not endorsed by Riot Games.** Riftbound, the card names and the card art
belong to Riot Games.

**It reads cards. It cannot draw them.** This is recognition, not image generation.

**New here? Start with [the process at a glance](docs/overview.md):** two diagrams, one of
what you do and one of how the pipeline works.

## What is in this repository

| | |
| --- | --- |
| `cardvision/` | The Python package and command line: inference, the bank and release tools, both trainers, the generator of training scenes, export, and evaluation. |
| `labeler/` | A labeling tool that runs on your own machine: a small Python server and one web page, no install. A model pre-fills boxes and names, and a person corrects them. |
| `browser/` | The same pipeline for a web page, in TypeScript on ONNX Runtime Web: it loads a release and says which card would be shown under the pointer on a video or image. With a demo page for a local file. See [In a browser](#in-a-browser). |
| `data/` | Label manifests: for single frames of public videos, where the cards are and which cards they are. Text only. The format is in `data/SCHEMA.md`. |
| `games/riftbound/` | Riftbound's catalog (card ids and names), accept rule, and the two training recipes. |
| `games/_template/` | The starting point for another game. |
| `docs/` | How it works, the model card, the data card, the evaluation method, the adapt guide, and what was tried and dropped. |
| `tests/` | The tests. They run on a toy game drawn by code. |
| `examples/` | `toy_quickstart.py`: the quickstart below as one script. |

**As release assets, not in git:** the detector and recognizer weights (ONNX), the bank and
its identity groups, the catalog, and the release document that pins them together. See [Getting the released weights](#getting-the-released-weights).

## What is not in this repository

- **No card art, no video frames, no images from any game.** The tools take a folder of card
  images that you are entitled to use. The bank is computed from card images and the
  recognizer was trained on them; the docs say so plainly. The images themselves are not here.
- **No videos.** A label manifest names a public video by its id and a time in it. Getting
  the frame is a step you run yourself.
- **The private part of the training data.** The released weights were also trained on images
  from our own private games on a digital client. Those are not published.
- **The private test set.** The headline figures in the model card were measured on a private
  held-out test set. They are historical, from release `5cd3a4bf37154976`, whose bank is older
  than the published one. Keeping the set private reduces direct contamination; once its
  results have guided changes, it serves as a regression benchmark.
- **The browser extension.** `browser/` holds the recognition runtime that runs in a web page
  and a demo page for a file on your own disk, also hosted on GitHub Pages. It is not an
  extension, and it does not attach to any website. An example extension built on it is
  described in the [adapt guide](docs/adapt-your-game.md#9-put-it-in-a-browser-extension-optional).

**So you cannot rebuild the released weights exactly.** Private images and Riot's card art
were part of the training mix, and neither is published. What the published recipe reaches on
public inputs and card images alone is stated in the
[model card](docs/model-card.md#3-public-only-recipe). On the public evaluation set, a small
one (53 frames, 20 recording groups, 1,053 readable faces), models trained that way named
92.2 % of readable faces right, against 92.5 % for the released weights. The paired difference
is not distinguishable from zero: −0.28 points, with a 95 % interval by recording group of
−4.03 to +3.17. The public-only detector came out weaker: swapped in behind the released
recognizer, it named 1.42 points fewer faces right (paired interval −3.60 to +0.79, which
includes zero). On physical tables more of the shown names were wrong (2.6 % against 1.0 %;
that paired interval includes zero too). This is a small public benchmark used during
development, not an independent test of generalization, and some of its digital identity labels
were assigned with model suggestions ([data card](docs/data-card.md#how-the-labels-were-made)).

## Quickstart on a toy game

This runs the whole pipeline on a toy card game that the repository draws from colored
shapes. It needs no card art, no video and no download beyond the Python packages. Every
command here was run as written, on macOS with Python 3.10; the output shown is real. A
workflow, `.github/workflows/quickstart.yml`, runs the same steps on Linux and on Windows in
WSL.

You need Python 3.10 or later, and a copy of the repository: step 0 below gets it. Run
everything from the repository's root, the folder that holds this README.

**How to read the steps.** A block after "Paste this into the terminal" holds commands: copy
the whole block, paste it into the terminal and press Return. A block after "You should see
something like" is what the commands print: you do not type it. Your timings and folder names
will differ.

**Which computer?** The commands are the same on macOS and Linux. On Windows they run in
Ubuntu under WSL 2 (the Windows Subsystem for Linux), where every step applies word for word.
A Windows terminal such as PowerShell is not supported. What was checked on which platform,
GPU training included, is in the [adapt guide](docs/adapt-your-game.md#before-you-start).

- **macOS:** as written.
- **Linux:** install Python 3.10 or later, with its `venv` module, and Node 22 or later (for
  the browser part) with your package manager. Where a step says `open`, use `xdg-open`.
- **Windows:** open PowerShell with "Run as administrator", type `wsl --install` and restart
  the computer. Then open **Ubuntu** from the Start menu. The first time, it asks you to choose
  a user name and a password. Everything from here on happens in Ubuntu's terminal.

On Windows, in Ubuntu, paste this into the terminal (`sudo` asks for that password):

```
sudo apt update
sudo apt install --yes python3-venv git curl
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.8/install.sh | bash
```

Then close Ubuntu, open it again, and paste this into the terminal to install Node:

```
nvm install --lts
```

Put the repository in your Linux home folder (`cd ~`, then step 0 below), not under `/mnt/c`:
Ubuntu is much slower with files on the Windows drive. To see the current folder in File
Explorer, paste `explorer.exe .` into the terminal. A page that a step serves in Ubuntu, such as
the demo at `http://127.0.0.1:8765/`, opens in your Windows browser with `localhost` in place of
`127.0.0.1`: `http://localhost:8765/`.

**0. Get the code.** The toy walkthrough uses files in the source tree (`examples/` and
`tests/`), so it needs the whole repository, not just the Python package. With `git`, paste
this into the terminal, in the folder that is to hold the repository (on Windows, in Ubuntu,
after `cd ~`):

```
git clone https://github.com/ryangswe/riftsight-vision.git
cd riftsight-vision
```

That makes the folder `riftsight-vision` and moves into it. It is the repository's root, where
every command below runs. Without `git`: on the repository's page on GitHub, press **Code**,
then **Download ZIP**. Unpack it, and `cd` into the unpacked folder (GitHub names it after the
branch, such as `riftsight-vision-main`). On Windows, unpack it in Ubuntu's home folder, not on
the Windows drive.

**1. Install.** Paste this into the terminal, from the repository's root:

```
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
cardvision --version
```

pip prints many lines while it installs (a notice that a newer pip exists is harmless). You
should see something like this as the last line:

```
cardvision 0.1.0
```

**2. Run the toy quickstart.** One script draws the toy game (20 card images and a
catalog) into a work folder, builds two stand-in models, and runs the pipeline on a frame it
generates. The work folder is `cardvision-toy` in your Downloads folder. It is outside the
repository on purpose, because the repository must hold no images and no model files, and you
can delete it when you are done.

Paste this into the terminal, from the repository's root:

```
export WORK="$HOME/Downloads/cardvision-toy"
mkdir -p "$WORK"
python examples/toy_quickstart.py "$WORK/toy"
```

You should see something like:

```
wrote $WORK/toy/catalog.json and 20 images in $WORK/toy/art
wrote stand-in models, a staged bank and a test scene in $WORK/toy
table.png: 7 boxes
  0  face  named Amber Circle    best match TOY-001 Amber Circle, similarity 1.000, margin 0.139, look 0
  1  face  named Coral Triangle  best match TOY-003 Coral Triangle, similarity 1.000, margin 0.172, look 3
  2  face  named nothing         best match TOY-001 Amber Circle, similarity 0.842, margin 0.108, look 0
  3  face  named nothing         best match TOY-015 Plum Triangle, similarity 1.000, margin 0.000, look 0
  4  face  named Amber Stripes   best match TOY-017 Amber Stripes, similarity 1.000, margin 0.139, look 0
  5  back  never named
  6  face  named nothing         best match TOY-005 Plum Stripes, similarity 0.785, margin 0.040, look 3
1 frame, 6 faces, 3 named
```

Here `$WORK` stands for your folder: the terminal prints the full path,
`/Users/YOU/Downloads/cardvision-toy/toy/...` on a Mac. To see the files, paste `open "$WORK"`
into the terminal on macOS, `xdg-open "$WORK"` on Linux, and `(cd "$WORK" && explorer.exe .)`
in Ubuntu on Windows.

**If you open a new terminal,** it forgets `$WORK` and the Python environment. Paste this
into the terminal again first, from the repository's root:

```
. .venv/bin/activate
export WORK="$HOME/Downloads/cardvision-toy"
```

Running a step again replaces the files it wrote, so you can repeat any step. To start over
from nothing, delete the whole folder with `rm -rf "$HOME/Downloads/cardvision-toy"`. That
deletes everything in it at once and does not ask, so keep nothing of your own in it.

The released weights are release assets, so the quickstart uses two tiny stand-ins. They and
the toy game belong to the test suite: the script imports them from `tests/`.

The stand-ins have no skill. The recognizer is an average pool and a random projection,
enough to tell the toy cards apart. **The detector does not look at the image**: it returns
the boxes of the cards in the test scene, `table.png`. They show what the tools do, not how
well the released models work.

What the output shows:

- **A card lying sideways is still named.** Box 1 has `look 3`: it was not accepted upright,
  so the three other orientations were tried.
- **A card back is reported and never named.** That is box 5.
- **Saying nothing is a normal result.** The script's bank, `bank.json`, is staged to show
  it: one card is left out of the bank and two cards share a row. Box 2 is the card the bank
  lacks: its best match scores too low. Box 3 matches two rows equally, so its margin is
  zero. Box 6 is cut in half by the frame's edge, and its best match scores too low.

**3. Build a bank** from all the card images. This is the real command; for a real game you
pass the released recognizer. Paste this into the terminal, from the repository's root:

```
cardvision build-bank --catalog "$WORK/toy/catalog.json" --art "$WORK/toy/art" \
    --recognizer "$WORK/toy/recognizer.onnx" --out "$WORK/toy/bank-all.json"
```

You should see something like:

```
  20 / 20 cards
bank: $WORK/toy/bank-all.json (20 rows of 128 values; 20 embedded, 0 kept from the base bank)
identity groups: $WORK/toy/bank-all.identity-groups.json
```

**4. Detect and name the cards** in the test scene, with the real command and the full bank.
Paste this into the terminal, from the repository's root:

```
cardvision infer "$WORK/toy/table.png" \
    --detector "$WORK/toy/detector.onnx" --recognizer "$WORK/toy/recognizer.onnx" \
    --bank "$WORK/toy/bank-all.json" --catalog "$WORK/toy/catalog.json" \
    --accept "$WORK/toy/accept.json" --annotate "$WORK/out"
```

It prints one JSON object per frame, as one long line. You should see something like this at
its start (wrapped here for reading):

```
{"source": "$WORK/toy/table.png", "frame": "table.png", "t": 0.0,
 "frameW": 1280, "frameH": 768, "analysisW": 1280, "analysisH": 768,
 "boxes": [
  {"i": 0, "box": [40.0, 40.0, 240.0, 336.0], "score": 0.95, "class": "face",
   "outcome": "named", "name": "Amber Circle", "cardId": "TOY-001",
   "topId": "TOY-001", "topName": "Amber Circle", "similarity": 0.995918, "margin": 0.12566,
   "look": 0, "retried": false, "tauPassed": true, "accepted": true},
  {"i": 1, "box": [320.0, 40.0, 336.0, 240.0], "score": 0.9, "class": "face",
   "outcome": "named", "name": "Coral Triangle", "cardId": "TOY-003",
   "topId": "TOY-003", "topName": "Coral Triangle", "similarity": 0.997997, "margin": 0.157303,
   "look": 3, "retried": true, "tauPassed": true, "accepted": true},
  ...
```

Each `box` is `[x, y, width, height]`: the card's top-left corner and size, in pixels of the
original frame (`frameW` by `frameH`). [What `cardvision infer` prints](docs/how-it-works.md#what-cardvision-infer-prints)
describes every field.

After it, on standard error, you should see something like:

```
cardvision infer: 1 frame(s), 6 face(s), 1 back(s), 5 named
```

When the detector keeps no box in a frame, it says so before this line and gives the
highest score the detector saw, so that you can tell a near miss from nothing at all
([docs/adapt-your-game.md](docs/adapt-your-game.md), tier 0).

What that shows:

- **Six faces, five named.** With every card in the bank, boxes 2 and 3 are named. Box 6,
  the card cut in half, is still named nothing.
- The sideways card has `"look": 3` and `"retried": true`, and the card back has
  `"class": "back"` and no name.
- `$WORK/out/table.jpg` is the frame with the boxes and names drawn on it. On macOS,
  `open "$WORK/out/table.jpg"` shows it.

**Next:** [docs/how-it-works.md](docs/how-it-works.md) explains each step, and
[docs/adapt-your-game.md](docs/adapt-your-game.md) takes you from here to your own game.

## Getting the released weights

The weights, the bank and the release document are release assets of
[v0.1.0](https://github.com/ryangswe/riftsight-vision/releases/tag/v0.1.0), release
`201478a00d6ef6ba`. Each file's address is
`https://github.com/ryangswe/riftsight-vision/releases/download/v0.1.0/` followed by its name:

| File | What it is | SHA-256 |
| --- | --- | --- |
| `release.json` | The release document | `5054fc41bc3d85a66d934e494ef9f114a7b27ac08eca0a6328c95750428e0a73` |
| `detector.onnx` | The detector | `dd9dc32b74756d83316bf8347784f09fad69d6f54c55384c1d093cb312727185` |
| `recognizer.onnx` | The recognizer | `a65ee03103893c39f8a0bbde9b3d18b8540130205df7883c4467b2673c47fd3b` |
| `bank.json` | The bank | `8dd173a237a19bcd86c52b3cdb88b54f726e690c1d9e19f1bec6d75e63ce969e` |
| `identity-groups.json` | The identity groups | `1c4cf595abc76aa71a338e4ca16344f7a652b8023416d7cd57a1555148cefc61` |
| `catalog.json` | The catalog | `75e563c8aa39cda85e6f24b683dbb1311cc7c4c18d809df2808a5f36fbfc855d` |
| `accept.json` | The accept rule, for commands that take `--accept` | `b3a3ec47fcfdaafc0923707218eeb056684e5b1407611a6e716c47fdb8b26dc3` |

The release document pins the first six by SHA-256 and size, and holds the accept rule too.

Put the files in one folder and check them. `verify-release` compares every file with the
hash the release document pins, and refuses the release whole if anything differs. (These two
commands were not run here: they need the release assets.) Paste this into the terminal, from
the repository's root:

```
cardvision verify-release RELEASE-FOLDER/release.json
cardvision infer YOUR-VIDEO.mp4 --release RELEASE-FOLDER/release.json --annotate "$WORK/annotated"
```

`--release` checks the release the same way `verify-release` does, and takes the detector,
recognizer, bank, catalog, accept rule and every other setting from it. The separate files
also work (`--detector`, `--recognizer`, `--bank`, `--catalog` and `--accept`), but not
together with `--release`.

Reading video needs OpenCV, which `requirements-lock.txt` installs. The weights of v0.1.0 are
published as ONNX only: no PyTorch checkpoint and no pickle.

Nothing needs card art to run the released models: the bank and the card names are enough.

## The commands

`cardvision --help` lists them; each has its own `--help`.

| Command | What it does |
| --- | --- |
| `cardvision infer` | Detect and name the cards in images or video, with the models from a release document (`--release`) or from separate files. Can also write annotated frames, crops, a contact sheet, and a pre-fill file for the labeling tool. |
| `cardvision build-bank` | Build the bank and its identity groups from card images, or extend an existing bank (`--base`): rows whose image changed are embedded again in place, new cards are added. |
| `cardvision check-refresh` | Check that a new bank only adds to an old one, and changes a row only where its image changed. |
| `cardvision check-bank-art` | Report which rows of a bank were embedded from the images now in your folder, and which were not. With no bank, look at the image of every card of a catalog. Either way it lists the images that deserve a second look: not a picture, a placeholder, or one file under different cards. |
| `cardvision build-release` | Pin models, bank, identity groups, catalog and settings in one release document. |
| `cardvision verify-release` | Check a release document and every file it pins. |
| `cardvision evaluate` | Score a configuration on labeled frames: `hover`, `crops`, `detector`, or `compare` two configurations. `hover --browser-events FILE` scores outcomes that the browser runtime's scoring page recorded on video files. `--scenes DIR` scores scenes that `make-composites` generated, by their exact labels: a first look, not a measurement on real video. |
| `cardvision calibrate` | Choose the accept rule's `tau` and `delta` on labeled frames: score a grid of pairs from one analysis of each frame, and recommend one under a limit on wrong names that you set. |
| `cardvision validate-manifest` | Check label manifests: format, ranges, duplicates, groups, card ids, and that no image is present. |
| `cardvision fetch` | Obtain a manifest's frames from video files you have, or by downloading if you ask. `--keep-video DIR` keeps the downloaded video files, and reads the ones already there. |
| `cardvision record-reachable` | Set a manifest's `reachable` fields from the index of a fetch run that downloaded. |
| `cardvision assign-roles` | Give new recording groups a role by a seeded shuffle. |
| `cardvision make-composites` | Generate labeled training scenes from card images. |
| `cardvision fetch-textures` | Download and verify the CC0 background scans the generator can use. |
| `cardvision train-recognizer` | Train the recognizer from a config file. Needs the training environment. |
| `cardvision train-detector` | Fine-tune the detector from a config file. Needs the training environment and YOLOX. |
| `cardvision export` | Export a checkpoint to ONNX and safetensors, with a parity check. |
| `python3 labeler/serve.py` | The labeling tool. Started with no arguments, its page asks what to label. [labeler/README.md](labeler/README.md) |
| `python3 labeler/export.py` | Export labels as COCO, or as a label manifest. |

Nothing reaches the network unless you ask for it by flag: `fetch --download`,
`fetch-textures`, `train-detector --download-pretrained`, and `train-recognizer` when it
starts from torchvision's pretrained weights.

**Tests:** paste this into the terminal, from the repository's root:

```
python -m pytest
```

In the environment above, when this page was written, this ran about 2,500 tests in about six
minutes, with 9 skipped because they need PyTorch. `requirements-lock-train.txt` says
how to build the training environment that runs those too. One more is skipped, the comparison
with the browser runtime, until `npm ci` has been run in `browser/`. On a machine without the
`ffmpeg` program, the tests that need it are skipped as well: one test of
`tests/test_data_extract.py`, in its four cases, and the tests of
`tests/test_data_frame_times.py` that read a recording of variable frame rate, in 38 cases.

## In a browser

`browser/` is the pipeline as a TypeScript package for a web page. It loads the release
document that `cardvision build-release` writes, runs both models on ONNX Runtime Web (WebGPU,
with a WebAssembly fallback), and answers one question: with the pointer here, over this video
or image, which card would be shown?

**Try it with nothing to install:** <https://ryangswe.github.io/riftsight-vision/>. The page
offers a choice of games. The **toy game (practice)** comes with the page: press **Load the
example table** and move the pointer over a card to see how hovering works (its models are
stand-ins with no skill, and its cards are colored shapes). The **One Piece Card Game** loads
the Kenbunshoku models from Hugging Face: drop a screenshot or a video of your own game and
hover the cards to see their names. It shows names only, no card image. Either way the models
run in your browser. Files you select are processed locally by this demo; its code does not
upload their contents. Selecting One Piece downloads model files from Hugging Face. It is the
demo page below, built as a static site (`browser/README.md`, "The hosted demo"), and the link
above works once the maintainer has published it.

Its own tests, and the build, need Node 22 or later. Paste this into the terminal, from the
repository's root (the first line moves into `browser/`):

```
cd browser
npm ci
npm test
npm run build
```

Among other lines, you should see something like:

```
added 58 packages, and audited 59 packages in 4s
 Test Files  28 passed (28)
      Tests  495 passed | 6 skipped (501)
dist/cardvision.js 902 KB, dist/worker.js 920 KB, dist/demo/demo.js 44 KB, dist/demo/score.js 30 KB, 8 ONNX Runtime files copied
OpenCV.js NOT copied: vendor/opencv/ lacks opencv.mjs, opencv_js.wasm, opencv.mjs.LEGAL.txt (build it with node scripts/build-opencv.mjs --work FOLDER). This build runs without the geometric stage: asked for, it reports itself off.
```

The six skipped tests and the `OpenCV.js NOT copied` line are expected: they need OpenCV.js,
an optional build ([browser/README.md](browser/README.md), "The geometric stage"). npm also
prints lines of its own, such as `npm warn install-scripts` notes (esbuild, fsevents,
protobufjs) and an `npm notice` that a new major version of npm is available: both are harmless.
Times and sizes may differ a little on your machine.

`npm run demo -- --release RELEASE-FOLDER` serves a demo page on `127.0.0.1`. The page is one
drop zone. You drop a video or image file from your disk on it, or choose one, and it starts:
it shows the file, outlines the card under the pointer and shows its name. A line under the
picture says which backend is running and what a frame costs. Add `--images FOLDER` and the
tooltip shows the card's image too. With `--file PATH` the server also prints an address that
opens that file with no file chooser, which is how a program that drives a browser opens the
demo.
[browser/README.md](browser/README.md) has the steps, a toy release to try it on, and the API.

**What the demo is, plainly.** It shows what the pipeline would display under the pointer on a
local file. It is not a browser extension, it does not attach to any website, and it makes no
request other than to its own local server: for the release folder and, if you name them, a
folder of card images and the one file to open. The hosted copy's loader requests the site's
own files and the files the chosen release pins, and checks each file's bytes against the hash
the release pins. Its Content-Security-Policy limits requests to the site itself, a folder
prefix on Hugging Face, and Hugging Face's storage hosts: it limits where a request can go, not
what a request carries ([browser/README.md](browser/README.md), "The Content-Security-Policy").

The same server offers a scoring page, `score.html`. It finds the labeled frames of a manifest
in video files from your disk, records what the browser runtime would show at each labeled
position, and writes a file that `cardvision evaluate hover --browser-events` scores. Before a
run it says how many rows it would read and how many have a matching video. The video
files may be whole videos, or the sections that `cardvision fetch --download --keep-video` keeps.
[docs/method.md](docs/method.md) says what that measures and what it does not.

Python and the browser runtime are held to each other by a test that runs both on the same
generated frames and compares boxes, decisions and what is shown under a grid of pointer
positions. From the repository's root (if you are in `browser/`, run `cd ..` first), paste this
into the terminal:

```
python -m pytest tests/test_browser_fixtures.py tests/test_browser_agreement.py -q
```

You should see something like:

```
49 passed in 28.63s
```

## Documentation

| | |
| --- | --- |
| [docs/overview.md](docs/overview.md) | The whole process at a glance: what you do, and how the pipeline works, in two diagrams. |
| [docs/how-it-works.md](docs/how-it-works.md) | The pipeline: detector, recognizer, bank, accept rule, rotation retry, hover rule, and the catalog file a game supplies. |
| [docs/adapt-your-game.md](docs/adapt-your-game.md) | Point it at another card game, in three tiers: try it, make it yours, ship it. |
| [AGENTS.md](AGENTS.md) | The same three tiers as steps a coding agent can run and check, and the rules an agent must follow here. |
| [docs/model-card.md](docs/model-card.md) | What the models are, what they were trained on, what they were measured to do, and their limits. |
| [docs/data-card.md](docs/data-card.md) | What the label manifests contain, how they were made, what a video id reveals, and how to have rows removed. |
| [docs/method.md](docs/method.md) | How to evaluate honestly with these tools, and how the released models were tested and chosen. |
| [docs/what-we-tried.md](docs/what-we-tried.md) | What was tried and dropped, and what we would do differently. |
| [docs/how-this-was-built.md](docs/how-this-was-built.md) | Who built this, with what help, and how it was checked. |
| [docs/faq.md](docs/faq.md) | AI use, the pretrained starting points, commercial use, the private test set, card art, rebuilding the weights, other games. |
| [data/README.md](data/README.md), [data/SCHEMA.md](data/SCHEMA.md) | The label manifests and their format. |
| [labeler/README.md](labeler/README.md), [labeler/FORMAT.md](labeler/FORMAT.md) | The labeling tool, its label format, and how to label. |
| [browser/README.md](browser/README.md) | The browser runtime: how to run it, its demo page and its scoring page, its API, and where it and the Python package differ. |
| [games/riftbound/README.md](games/riftbound/README.md), [games/_template/README.md](games/_template/README.md) | The files that describe a game. |
| [PROVENANCE.md](PROVENANCE.md) | Where everything comes from, and what each model started from. |
| [CONTRIBUTING.md](CONTRIBUTING.md), [SUPPORT.md](SUPPORT.md), [SECURITY.md](SECURITY.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) | How to take part, and what to expect. |
| [LICENSE](LICENSE), [LICENSE-DATA](LICENSE-DATA), [NOTICE](NOTICE), [TRADEMARK.md](TRADEMARK.md), [CITATION.cff](CITATION.cff) | Licenses, notices, names, and how to cite. |

## Does it work on my game?

Measured on one other game, the One Piece Card Game, on a small held-out set (25 frames from
13 videos in 9 recording groups, so the intervals are wide):

- the released weights as they stand, with a bank built from that game's card images, named
  93/296 = 31.4 % (23.7–42.5) of readable card faces right, with 11/104 = 10.6 % (4.2–28.0)
  of the names shown wrong;
- both models trained with the adapt guide on 23 labeled frames and generated scenes named
  221/297 = 74.4 % (50.3–83.8) right, with 9/230 = 3.9 % (1.6–11.8) of the names shown wrong;
- the optional geometric stage on top of those raised it to 249/297 = 83.8 % (62.5–91.8),
  with 7/256 = 2.7 % (0.6–10.3) wrong.

Physical play was well below digital: on it, the 23-frame trained models named 68/124 =
54.8 % (41.0–67.1) right, with 7/75 = 9.3 % (3.2–18.8) of the names shown wrong. One digital
recording group holds 135 of the 297 readable faces, so the overall figures lean on it. Wrong
of shown is the share of counted tooltips that are wrong under that evaluation's
target-sampling protocol. These figures are for that game and that protocol. The figures, the
caveats and the comparisons written down before scoring are in
[the adapt guide](docs/adapt-your-game.md#measured-on-another-game-the-one-piece-card-game);
the example's models and labels are in their own repositories: the [models](https://huggingface.co/JuicyKaraageNo1/kenbunshoku-onepiece) and the [labels](https://huggingface.co/datasets/JuicyKaraageNo1/kenbunshoku-onepiece-labels) on Hugging Face.

The detector's two classes are "card face" and "card back", not Riftbound's cards. The
recognizer needs no retraining to learn a new list of cards: you build a bank from your
card images. Tier 0 of the [adapt guide](docs/adapt-your-game.md) is how to find out how far
the released weights get on yours.

## Reporting a wrong card

Open a "Wrong recognition" issue. Give:

- the video's id and the time in the video;
- where on the frame the card is;
- what was shown, and what the card is.

**Do not attach a frame or a screenshot.** The id and the time are enough to find it.

## Built with this

Catalogs, banks and weights for other games live in their authors' own repositories, not
here. If you have built something with this, open an issue and it is listed.

- The One Piece Card Game: a catalog, a bank, fine-tuned weights and labels, with the
  measurements in the [adapt guide](docs/adapt-your-game.md#measured-on-another-game-the-one-piece-card-game):
  the [models](https://huggingface.co/JuicyKaraageNo1/kenbunshoku-onepiece) and the [labels](https://huggingface.co/datasets/JuicyKaraageNo1/kenbunshoku-onepiece-labels) on Hugging Face.

## How this was built

I'm a solo software engineer with some machine learning experience; computer vision isn't my
core competency. I built this with heavy use of AI coding tools.

It was checked with tests, with comparisons of the code here against the code that trained the
released models, with a private held-out test that was scored once (historical, on release
`5cd3a4bf37154976`), and with outside reviews, which were reviews by other AI models and by
fresh AI sessions, not by people. [docs/how-this-was-built.md](docs/how-this-was-built.md) says
what each check does and does not show. The method is in the repository so that you can judge
it, and there is a public evaluation set you can run yourself.

## Support

One person maintains this. Issues are read once a week, and removal and security requests are
handled within 7 days. This repository is a maintained snapshot of what shipped, not a
development trunk. [SUPPORT.md](SUPPORT.md) has the details.

## License

- **Code and weights:** Apache-2.0 ([LICENSE](LICENSE)), for this project's own work.
- **Bank, identity groups and catalog:** Apache-2.0 for this project's own work in the bank and
  identity groups, and for the catalog's selection and arrangement; the card names, ids and
  other card data in the catalog are not licensed here ([NOTICE](NOTICE)).
- **Label manifests:** CC BY 4.0, for the annotations only ([LICENSE-DATA](LICENSE-DATA)).
- **Starting points and attributions:** [NOTICE](NOTICE). The detector starts from YOLOX-Nano
  weights by Megvii; the recognizer from torchvision's MobileNetV3-Small weights.
- **What the grant covers:** this project's own work. The bank and the weights derive from
  card images that this repository does not contain and does not license. [NOTICE](NOTICE)
  says so, and [docs/faq.md](docs/faq.md) says what that means for your use.
- **Artifact by artifact:** the [license table](PROVENANCE.md#licenses-by-artifact) gives the
  grant, provenance, notices and unsettled rights of the code, weights, bank, identity groups,
  catalog and labels.

## Trademark

The Apache license does not cover the RiftSight name or logo. A fork or a product built on
this must use its own name and logo, and may say "based on RiftSight's open vision model".
[TRADEMARK.md](TRADEMARK.md)

## Not endorsed

RiftSight is a fan project and is not endorsed by Riot Games. Riftbound, the card names and
the card art belong to Riot Games.
