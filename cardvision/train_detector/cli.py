# SPDX-License-Identifier: Apache-2.0
"""The ``cardvision train-detector`` subcommand: a config file in, a trained checkpoint out.

::

    cardvision train-detector --config cardvision/train_detector/default.toml --out runs/first \
        --download-pretrained
    cardvision export detector runs/first/last.pth --out runs/first

It needs the ``train`` extra and the YOLOX package (see ``cardvision.export.detector``). The
config format is described in ``cardvision.train_detector.config`` and what the run writes in
``cardvision.train_detector.trainer``.

With ``--download-pretrained`` the public pretrained weights are downloaded and verified
first, before PyTorch is looked for. So the same command can be run once to fetch that file,
on a machine where the training environment is not installed yet; it then says what training
needs, and the next run finds the file on disk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SUMMARY = "Fine-tune the card detector from a config file."


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Declare the subcommand's arguments."""
    parser.add_argument("--config", required=True, type=Path, metavar="FILE",
                        help="the recipe: a .toml or .json file (see cardvision/train_detector/default.toml)")
    parser.add_argument("--out", required=True, type=Path, metavar="DIR",
                        help="the folder to write the run into: run.json, train-log.jsonl, last.pth, metrics.json")
    parser.add_argument("--init", metavar="CKPT",
                        help='the checkpoint to start from, or "random" for untrained weights, in place of the '
                             "config's init (default: the config's, and without one the public COCO-pretrained "
                             "YOLOX-Nano weights)")
    parser.add_argument("--download-pretrained", action="store_true",
                        help="download the public pretrained weights if they are not on disk yet; nothing is "
                             "downloaded without this flag. The download and its check come first, before "
                             "PyTorch is needed")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                        help="where to train (default: auto, which is CUDA when there is one and the CPU otherwise)")
    parser.add_argument("--steps", type=int, metavar="N", help="optimizer steps, in place of the config's")
    parser.add_argument("--seed", type=int, metavar="N", help="the seed, in place of the config's")
    parser.add_argument("--workers", type=int, metavar="N",
                        help="loader processes, in place of the config's; the result does not depend on it")
    parser.add_argument("--trust-checkpoint", action="store_true",
                        help="unpickle --init without restrictions; only for a file you made yourself")
    parser.add_argument("--overwrite", action="store_true", help="replace a run already in the output folder")


def _note(message: str) -> None:
    print(f"cardvision train-detector: {message}", file=sys.stderr)


def _require_train_extra() -> None:
    missing = []
    for module in ("torch", "cv2"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise RuntimeError(
            f"training needs {', '.join(missing)}, part of an optional dependency group: "
            'python -m pip install -e ".[train]" in the repository folder'
        )
    from cardvision.train_detector.model import nano_experiment

    nano_experiment()  # raises with the install command when YOLOX is missing


def _run(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from cardvision.train_detector.config import load_config
    from cardvision.train_detector.sources import file_sha256

    config = load_config(args.config)
    changes = {}
    if args.steps is not None:
        changes["steps"] = args.steps
    if args.seed is not None:
        changes["seed"] = args.seed
    if args.workers is not None:
        changes["loader"] = replace(config.loader, workers=args.workers)
    if changes:
        config = config.with_changes(**changes)
    if args.download_pretrained and args.init is None and config.init is None:
        # Fetch and verify the one file before anything needs PyTorch, so that this works as a
        # first step of its own. The trainer finds the file on disk and checks its hash again.
        from cardvision.train_detector import model

        path = model.fetch_pretrained(download=True)
        print(f"pretrained weights: {path} (SHA-256 verified)", flush=True)
    _require_train_extra()
    from cardvision.train_detector.trainer import train

    train(config, args.out, init=args.init, download_pretrained=args.download_pretrained,
          trust_checkpoint=args.trust_checkpoint, device=args.device, overwrite=args.overwrite,
          config_sha256=file_sha256(args.config), log=lambda line: print(line, flush=True))
    print(f"next: cardvision export detector {args.out / 'last.pth'} --out {args.out}")
    return 0


def run(args: argparse.Namespace) -> int:
    """Run the subcommand. Returns the process exit code: 0, or 2 after a reported error."""
    try:
        return _run(args)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, LookupError) as error:
        _note(f"error: {error}")
    return 2
