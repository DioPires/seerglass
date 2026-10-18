# SPDX-License-Identifier: Apache-2.0
"""The pipeline on a toy card game, in one script: draw the game, build stand-in models, run them.

Run it from a checkout of the repository::

    python examples/toy_quickstart.py OUT_DIR

``OUT_DIR`` must lie outside the repository, which holds no images and no model files. The
script writes into it:

- ``catalog.json`` and ``art/``: a toy game of 20 cards drawn from colored shapes;
- ``recognizer.onnx`` and ``detector.onnx``: two stand-in models;
- ``bank.json``: a bank of the toy cards, staged so that the accept rule has something to
  refuse (one card is left out of it and two cards share a row);
- ``accept.json``: an accept rule for the stand-in recognizer;
- ``table.png``: one generated frame, a table with seven cards on it.

Then it runs the pipeline on ``table.png`` with those files and prints what it found.

**The toy game and the stand-in models belong to the test suite.** This script imports them
from the repository's ``tests/`` folder (``tests/toy_game.py`` and
``tests/test_infer_toy_models.py``). So it needs a checkout and the packages that
``requirements-lock.txt`` installs; an installed package alone does not have them.

**The stand-ins have no skill.** The recognizer is an average pool and a random projection,
enough to tell the toy cards apart. The detector does not look at the image: it returns the
boxes of the cards in ``table.png``. The run shows what the tools do, not how well the
released models work.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USAGE = "usage: python examples/toy_quickstart.py OUT_DIR"


def build(out_dir: Path):
    """Write the toy game, the stand-in models, the staged bank and the scene into ``out_dir``.

    Returns the test suite's ``ToySetup``, which holds the path of each file.
    """
    # The toy game and the stand-in builders are test code, imported from tests/.
    sys.path.insert(0, str(ROOT / "tests"))
    import toy_game
    from test_infer_toy_models import ToySetup

    catalog, art = toy_game.write_toy_game(out_dir)
    print(f"wrote {catalog} and {len(toy_game.CARDS)} images in {art}")
    setup = ToySetup(out_dir, catalog, art)
    print(f"wrote stand-in models, a staged bank and a test scene in {out_dir}")
    return setup


def run(setup) -> int:
    """Run the pipeline on the scene and print one line per box. Returns how many cards were named."""
    from cardvision.core.catalog import load_catalog
    from cardvision.core.images import load_rgb
    from cardvision.infer.pipeline import Pipeline
    from cardvision.recognize import load_accept

    pipeline = Pipeline.open(setup.detector, setup.recognizer, setup.bank, load_catalog(setup.catalog),
                             load_accept(setup.accept))
    result = pipeline.analyze(load_rgb(setup.image), name=setup.image.name, source=str(setup.image))
    print(f"{result.name}: {len(result.cards)} boxes")
    for card in result.cards:
        found = card.recognition
        if found is None:
            print(f"  {card.index}  {card.class_name}  never named")
            continue
        said = f"named {found.name}" if found.accepted else "named nothing"
        print(f"  {card.index}  {card.class_name}  {said:20s}  best match {found.top_id} {found.top_name}, "
              f"similarity {found.similarity:.3f}, margin {found.margin:.3f}, look {found.look}")
    named = sum(1 for card in result.faces if card.recognition.accepted)
    print(f"1 frame, {len(result.faces)} faces, {named} named")
    return named


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(USAGE, file=sys.stderr)
        return 2
    out_dir = Path(argv[1])
    resolved = out_dir.resolve()
    if resolved == ROOT or ROOT in resolved.parents:
        print(f"{out_dir} lies inside the repository, which holds no images and no model files; "
              "choose a folder outside it", file=sys.stderr)
        return 2
    try:
        import onnx  # noqa: F401 - the stand-in models are built with it
    except ImportError:
        print("building the stand-in models needs the onnx package, which requirements-lock.txt installs: "
              "python -m pip install -r requirements-lock.txt", file=sys.stderr)
        return 2
    run(build(out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
