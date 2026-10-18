# SPDX-License-Identifier: Apache-2.0
"""The detector's training loop.

``train`` runs a config from start to finish and writes into the output folder:

``run.json``
    The run record, written before the first step: the recipe with every default filled in,
    the seed, the starting weights and their SHA-256, what each stream held (counts and file
    hashes), the package versions and the device. It holds no clock time and no path other
    than those written in the config, so the same command on the same machine writes the
    same bytes. A run that starts from a checkpoint named by the config or by ``--init`` also
    says so in a line of its log (``cardvision.train_detector.model.describe_start``).
``train-log.jsonl``
    One line per step: the learning rate, the loss and its parts, and how many anchors fell
    in ignore regions.
``last.pth``
    **The result.** The weights after the last step, nothing else is kept and no step is
    picked by a score: ``model`` holds the averaged weights (an exponential moving average
    of the trained ones, the library's ``ModelEMA``), ``raw`` the trained weights themselves.
    ``cardvision export detector last.pth`` exports the averaged ones.
``metrics.json``
    The summary: steps, time, the last loss, how many targets were seen.

One step: set the learning rate (``cardvision.train_detector.model.learning_rate``), hand the
batch's ignore regions to the head, run the network on the batch, step the optimizer, update
the averaged weights. In the closing phase (``no_augment_share``) the head's L1 box loss is
switched on and the images come without mosaic. A loss that is not finite stops the run and
leaves ``DIVERGED.json``.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from cardvision.detect.letterbox import INPUT_HEIGHT, INPUT_WIDTH
from cardvision.train_detector.config import Config

RUN_RECORD = "run.json"
LOG_NAME = "train-log.jsonl"
CHECKPOINT_NAME = "last.pth"
METRICS_NAME = "metrics.json"
DIVERGED_NAME = "DIVERGED.json"
RECORD_FORMAT = "cardvision-train-detector"
RECORD_VERSION = 1
PREFETCH = 4


class Diverged(RuntimeError):
    """The loss stopped being a finite number."""


def choose_device(device: str = "auto") -> str:
    """``"auto"`` picks CUDA when there is one and the CPU otherwise (training on MPS has not
    been validated, so it is never picked automatically); a named device must exist."""
    import torch

    from cardvision.core.device import DEVICES, pick_device

    if device == "auto":
        return pick_device(allow_mps=False)
    if device not in DEVICES:
        raise ValueError(f"unknown device {device!r}: expected auto, {', '.join(DEVICES)}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda, but CUDA is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps, but MPS is not available")
    return device


def run_record(config: Config, streams: dict, init: dict[str, Any], device: str,
               config_sha256: str | None = None) -> dict[str, Any]:
    """The run record (see the module docstring)."""
    from cardvision.train_detector.model import NUM_CLASSES, versions
    from cardvision.train_detector.sources import stream_records

    return {
        "format": RECORD_FORMAT,
        "formatVersion": RECORD_VERSION,
        "seed": config.seed,
        "steps": config.steps,
        "noAugmentFrom": config.no_augment_from,
        "input": [INPUT_WIDTH, INPUT_HEIGHT],
        "classes": NUM_CLASSES,
        "config": config.to_json(),
        "configSha256": config_sha256,
        "init": init,
        "streams": stream_records(streams),
        "device": device,
        "versions": versions(),
    }


def _peak_memory() -> float | None:
    try:
        from cardvision.core.device import rss_max_mb

        return rss_max_mb()
    except ImportError:  # no "resource" module on this platform
        return None


def _existing(out_dir: Path, overwrite: bool) -> list[str]:
    """The files of an earlier run in the folder; an error if there are any and they may not go."""
    names = (RUN_RECORD, LOG_NAME, CHECKPOINT_NAME, METRICS_NAME, DIVERGED_NAME)
    present = [name for name in names if (out_dir / name).exists()]
    if present and not overwrite:
        raise FileExistsError(f"{out_dir} already holds a run ({', '.join(present)}); pass --overwrite to replace it")
    return present


def train(config: Config, out_dir: str | Path, *, init: str | Path | None = None, download_pretrained: bool = False,
          trust_checkpoint: bool = False, device: str = "auto", overwrite: bool = False,
          config_sha256: str | None = None, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Train a detector by ``config`` and write the run into ``out_dir``. Returns the summary.

    Args:
        init: a checkpoint to start from or ``"random"`` for untrained weights, in place of
            the config's ``init``. With ``None`` here and no ``init`` in the config, the start
            is the public COCO-pretrained YOLOX-Nano weights.
        download_pretrained: allow downloading those public weights if they are not on disk.
        trust_checkpoint: unpickle ``init`` without restrictions; only for a file you made.
        device: ``"auto"``, ``"cpu"``, ``"cuda"`` or ``"mps"``.
        config_sha256: the hash of the config file, for the run record.
        log: called with a line of text now and then; nothing is printed when ``None``.
    """
    import torch
    from yolox.utils import ModelEMA

    from cardvision.core.device import synchronize
    from cardvision.train_detector.augment import Tiles
    from cardvision.train_detector.loss import set_ignore_regions
    from cardvision.train_detector.model import (NUM_CLASSES, RANDOM_INIT, build_model, build_optimizer,
                                                 describe_start, initialize, learning_rate, nano_experiment)
    from cardvision.train_detector.sources import build_streams

    say = log if log is not None else (lambda line: None)
    out_dir = Path(out_dir)
    earlier = _existing(out_dir, overwrite)
    chosen = choose_device(device)
    dev = torch.device(chosen)
    streams = build_streams(config)

    torch.set_num_threads(config.loader.threads)
    try:
        import cv2

        cv2.setNumThreads(1)
    except ImportError:
        pass
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    experiment = nano_experiment(NUM_CLASSES)
    model = build_model(experiment)
    named = init if init is not None else config.init
    init_record = initialize(model, named, download=download_pretrained, trust=trust_checkpoint)
    model.head.ignore_overlap = config.ignore_overlap
    model.to(dev)
    model.train()
    o = config.optimizer
    optimizer = build_optimizer(experiment, config.batch_size, o.lr, o.momentum, o.weight_decay)
    ema = ModelEMA(model, o.ema_decay)

    # Nothing is written, and nothing of an earlier run removed, until the run can start.
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in earlier:
        (out_dir / name).unlink()
    record = run_record(config, streams, init_record, chosen, config_sha256)
    (out_dir / RUN_RECORD).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    say(f"training {config.steps} steps of {config.batch_size} on {chosen}; streams: "
        + ", ".join(f"{name} ({len(stream)} frames)" for name, stream in streams.items()))
    if named is not None and str(named) != RANDOM_INIT:
        say(describe_start(init_record))

    tiles = Tiles(streams, config)
    workers = config.loader.workers
    loader = torch.utils.data.DataLoader(
        tiles, batch_size=config.batch_size, shuffle=False, num_workers=workers, drop_last=True,
        persistent_workers=False, prefetch_factor=PREFETCH if workers else None, pin_memory=dev.type == "cuda")

    no_augment_from = config.no_augment_from
    a = config.augment
    stats = {"images": 0, "targets": 0, "backs": 0, "ignoreRegions": 0, "ignoredAnchors": 0,
             "imagesOverMaxTargets": 0, "imagesOverMaxIgnore": 0}
    losses: list[float] = []
    times: list[float] = []
    started = time.time()
    with open(out_dir / LOG_NAME, "w", encoding="utf-8") as log_file:
        for step, (images, targets, ignore, counts) in enumerate(loader):
            tick = time.time()
            images = images.to(dev, non_blocking=True)
            targets = targets.to(dev, non_blocking=True)
            ignore = ignore.to(dev, non_blocking=True)
            lr = learning_rate(step, o.lr, o.warmup_steps, no_augment_from, o.final_lr_ratio)
            for group in optimizer.param_groups:
                group["lr"] = lr
            model.head.use_l1 = step >= no_augment_from
            set_ignore_regions(model, ignore)
            outputs = model(images, targets)
            loss = outputs["total_loss"]
            if not torch.isfinite(loss):
                (out_dir / DIVERGED_NAME).write_text(json.dumps({"step": step + 1}) + "\n", encoding="utf-8")
                raise Diverged(f"the loss is not finite at step {step + 1}; the run stops and no checkpoint is written")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(model)
            synchronize(dev)
            times.append(time.time() - tick)

            total = counts.sum(0)
            stats["images"] += len(counts)
            stats["targets"] += int(total[0])
            stats["ignoreRegions"] += int(total[1])
            stats["backs"] += int(total[2])
            stats["ignoredAnchors"] += int(model.head.last_ignored_anchors)
            stats["imagesOverMaxTargets"] += int((counts[:, 0] > a.max_targets).sum())
            stats["imagesOverMaxIgnore"] += int((counts[:, 1] > a.max_ignore).sum())
            line = {"step": step + 1, "lr": lr, "ignoredAnchors": int(model.head.last_ignored_anchors),
                    **{name: float(value.detach()) if torch.is_tensor(value) else float(value)
                       for name, value in outputs.items()}}
            losses.append(line["total_loss"])
            log_file.write(json.dumps(line) + "\n")
            log_file.flush()
            if step < 2 or (step + 1) % 50 == 0 or step + 1 == config.steps:
                say(f"step {step + 1}/{config.steps}  lr {lr:.6f}  loss {line['total_loss']:.4f}")

    if len(losses) != config.steps:
        raise RuntimeError(f"the loader gave {len(losses)} batches for {config.steps} steps")
    summary = {
        "steps": len(losses),
        "firstLoss": losses[0],
        "lastLoss": losses[-1],
        "wallMinutes": round((time.time() - started) / 60, 2),
        "secondsPerStepMedian": round(float(np.median(times)), 3),
        "stats": stats,
        "device": chosen,
        "rssMaxMb": _peak_memory(),
    }

    def on_cpu(module: Any) -> dict[str, Any]:
        return {name: value.detach().cpu() for name, value in module.state_dict().items()}

    # Tensors and plain containers only, so that the file reads back in PyTorch's restricted mode.
    torch.save({"model": on_cpu(ema.ema), "raw": on_cpu(model), "summary": summary, "config": config.to_json(),
                "init": init_record}, out_dir / CHECKPOINT_NAME)
    (out_dir / METRICS_NAME).write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    say(f"wrote {out_dir / CHECKPOINT_NAME}: loss {summary['firstLoss']:.4f} at step 1, "
        f"{summary['lastLoss']:.4f} at step {summary['steps']}")
    return summary
