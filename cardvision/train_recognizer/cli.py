# SPDX-License-Identifier: Apache-2.0
"""The ``cardvision train-recognizer`` subcommand: a config file in, a run folder out.

``cardvision train-recognizer --config FILE --out DIR`` trains the recognizer's encoder as the
config says (``cardvision.train_recognizer.config``) and writes the run into ``DIR``: the
record, the log, the curve and the checkpoints ``best.pt`` and ``last.pt``. It needs the
``train`` extra (``python -m pip install -e ".[train]"`` in the repository folder).

``--init`` replaces the config's ``[model] init``: ``imagenet``, ``random``, or the path of a
recognizer checkpoint to start from (``cardvision.train_recognizer.model``).

Then::

    cardvision export recognizer DIR/best.pt --out exported/
    cardvision build-bank --catalog C.json --art ART --recognizer exported/recognizer.onnx --out bank.json
"""

from __future__ import annotations

import argparse
import sys

SUMMARY = "Train the recognizer's encoder from a config file."


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Declare the subcommand's arguments."""
    parser.add_argument("--config", required=True, metavar="FILE", help="the training config, a TOML file")
    parser.add_argument("--out", required=True, metavar="DIR", help="the folder to write the run into")
    parser.add_argument("--init", metavar="INIT",
                        help="starting weights of the encoder, in place of the config's: imagenet (torchvision's "
                             "ImageNet weights, downloaded on first use), random, or the path of a recognizer "
                             "checkpoint to start from (a .pt file of this trainer, or a .safetensors file)")
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"),
                        help="where to train (default: CARDVISION_DEVICE if set, else the best available)")
    parser.add_argument("--epochs", type=int, metavar="N", help="train for N epochs instead of the config's")
    parser.add_argument("--seed", type=int, metavar="S", help="the training seed, instead of the config's")
    parser.add_argument("--workers", type=int, metavar="N", help="loader processes, instead of the config's")
    parser.add_argument("--overwrite", action="store_true", help="replace a run already in the output folder")


def _note(message: str) -> None:
    print(f"cardvision train-recognizer: {message}", file=sys.stderr)


def _require_train_extra() -> None:
    missing = []
    for module in ("torch", "torchvision"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise RuntimeError(
            f"training needs {', '.join(missing)}, part of an optional dependency group: python -m pip install -e \".[train]\" in the repository folder"
        )


def _run(args: argparse.Namespace) -> int:
    _require_train_extra()
    from pathlib import Path

    from cardvision.train_recognizer.config import ConfigError, check_init, is_checkpoint, load_config

    cfg = load_config(args.config)
    if args.init:
        check_init(args.init, "--init")
        # A path given on the command line is relative to the folder the command is run from.
        cfg["model"]["init"] = str(Path(args.init).resolve()) if is_checkpoint(args.init) else args.init
    for key in ("epochs", "seed", "workers"):
        value = getattr(args, key)
        if value is not None:
            if value < (1 if key == "epochs" else 0):
                raise ConfigError(f"--{key} must be at least {1 if key == 'epochs' else 0}")
            cfg[key] = value
    from cardvision.train_recognizer.loop import train

    train(cfg, args.out, device=args.device, overwrite=args.overwrite)
    return 0


def run(args: argparse.Namespace) -> int:
    """Run the subcommand. Returns the process exit code: 0, or 2 after a reported error."""
    try:
        return _run(args)
    except (OSError, ValueError, RuntimeError) as error:
        _note(f"error: {error}")
    return 2
