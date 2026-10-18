# SPDX-License-Identifier: Apache-2.0
"""The recognizer's training run: set-up, the step, the loop and what is written.

One step
========

For a batch of crops, clean card images and labels:

1. every ``bank_refresh_steps`` steps, the memory bank is recomputed: the clean images of every
   card that may be trained on, embedded in evaluation mode without gradients;
2. the crops are embedded; the positives (label 0 or more) give the ArcFace loss and, with
   their clean images embedded too, the contrastive loss;
3. the negatives give the hinge loss against the memory bank and the batch's clean images;
4. one AdamW step, and the learning rate moves along its schedule.

The learning rate is the config's ``lr`` times ``lr_factor``: a straight rise over the warm-up
steps, then half a cosine down to zero at the last step.

What a run writes into its folder
=================================

``run.json``
    The record: the resolved config, the seeds, the device, library versions, the SHA-256 of
    every input, the classes, the epoch's make-up and, at the end, the result.
``log.txt`` and ``curve.json``
    One line and one record per epoch.
``best.pt`` and ``last.pt``
    Checkpoints: ``{"model": the encoder's state_dict, "cards": the class list, "epoch": ...,
    "scores": ..., "config": ...}``. ``best.pt`` is the grid epoch chosen by
    ``cardvision.train_recognizer.validate``; ``cardvision export recognizer`` reads either.

A run whose ``[model] init`` is a checkpoint (``cardvision.train_recognizer.model.read_start``)
says so in three places, so that a model's lineage can be stated: a line of the log, the key
``init`` of ``run.json`` and the key ``init`` of both checkpoints, each with the file's name
and SHA-256. Such a run is the run that ``init = "random"`` gives with the encoder's weights
replaced by the file's before the first step. A run that starts from a word has no such key.
``cache/``
    The crops cut from the manifests and the composites.
"""

from __future__ import annotations

import json
import platform
import random
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cardvision import __version__
from cardvision.core.preprocess import preprocessor
from cardvision.train_recognizer.config import ConfigError, is_checkpoint, plain
from cardvision.train_recognizer.dataset import PairBatchSampler, TrainingSet, class_list, compose, make_loader
from cardvision.train_recognizer.losses import NEGATIVE_BACK, ArcFace, contrastive, negative_hinge, negative_scores
from cardvision.train_recognizer.model import Encoder, read_start
from cardvision.train_recognizer.sources import build_sources
from cardvision.train_recognizer.validate import Selection, criterion, on_grid, top1

RUN_NAME = "run.json"
RUN_FORMAT = "cardvision-recognizer-run"


def lr_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    """The learning rate at a step, as a fraction of the config's: ``step / warmup_steps``
    during the warm-up, then ``0.5 * (1 + cos(pi * progress))`` over the remaining steps."""
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1 + np.cos(np.pi * progress))


def name_index_of(classes: list[str], name_of: Mapping[str, str]) -> torch.Tensor:
    """For each class, the index of its card name among the classes' sorted names. Classes
    with the same index are printings of one name.

    Training passes the catalog's identity keys as ``name_of`` (``Catalog.identities()``),
    so classes with the same index are one identity under the catalog's rule: the names by
    default, and a card with its printings under the ``printing`` rule."""
    names = sorted({name_of[card_id] for card_id in classes})
    position = {name: index for index, name in enumerate(names)}
    return torch.tensor([position[name_of[card_id]] for card_id in classes], dtype=torch.long)


class Step:
    """The training step, holding the model, the head, the optimizer and the memory bank."""

    def __init__(self, model: torch.nn.Module, head: ArcFace, optimizer: torch.optim.Optimizer, scheduler: Any,
                 loss_cfg: Mapping[str, Any], bank_inputs: torch.Tensor, name_index: torch.Tensor | None) -> None:
        self.model, self.head, self.optimizer, self.scheduler = model, head, optimizer, scheduler
        self.cfg = loss_cfg
        self.bank_inputs = bank_inputs
        self.name_index = name_index
        self.bank: torch.Tensor | None = None
        self.steps = 0
        self.back_margin = loss_cfg["back_margin"] if loss_cfg["back_margin"] >= 0 else None
        self.stats = self.fresh_stats()

    @staticmethod
    def fresh_stats() -> dict[str, float]:
        return {"steps": 0, "negativeLoss": 0.0, "negatives": 0, "scoreSum": 0.0, "active": 0,
                "backs": 0, "backScoreSum": 0.0, "backActive": 0}

    def refresh(self) -> None:
        """Recompute the memory bank."""
        self.model.eval()
        with torch.no_grad():
            self.bank = torch.cat([self.model(self.bank_inputs[i:i + 512]) for i in range(0, len(self.bank_inputs), 512)])
        self.model.train()

    def __call__(self, crops: torch.Tensor, arts: torch.Tensor, labels: torch.Tensor) -> float:
        cfg = self.cfg
        device = self.bank_inputs.device
        crops, arts, labels = crops.to(device), arts.to(device), labels.to(device)
        if self.bank is None or self.steps % cfg["bank_refresh_steps"] == 0:
            self.refresh()
        self.steps += 1
        positive = labels >= 0
        emb = self.model(crops)
        loss = emb.sum() * 0.0
        art_emb = None
        # Two positives are the fewest that batch normalization can embed in training mode;
        # a batch of the recipe's size always has far more.
        if int(positive.sum()) >= 2:
            crop_emb, classes = emb[positive], labels[positive]
            loss = cfg["arcface_weight"] * self.head(crop_emb, classes, self.name_index)
            art_emb = self.model(arts[positive])
            loss = loss + cfg["contrastive_weight"] * contrastive(
                crop_emb, art_emb, classes, cfg["contrastive_temperature"], self.name_index)
        if bool((~positive).any()):
            is_back = labels[~positive] == NEGATIVE_BACK
            scores = negative_scores(emb[~positive], self.bank, art_emb)
            hinge, excess = negative_hinge(scores, is_back, cfg["negative_margin"], self.back_margin, cfg["back_weight"])
            loss = loss + cfg["negative_weight"] * hinge
            stats, seen, over = self.stats, scores.detach(), excess.detach() > 0
            stats["negativeLoss"] += float(hinge.detach())
            stats["negatives"] += int((~positive).sum())
            stats["scoreSum"] += float(seen.sum())
            stats["active"] += int(over.sum())
            stats["backs"] += int(is_back.sum())
            stats["backScoreSum"] += float(seen[is_back].sum())
            stats["backActive"] += int(over[is_back].sum())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.stats["steps"] += 1
        return float(loss.detach())


def versions() -> dict[str, str]:
    """The versions a run record carries."""
    import PIL
    import torchvision

    return {"cardvision": __version__, "python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "numpy": np.__version__, "pillow": PIL.__version__,
            "platform": platform.platform(), "machine": platform.machine()}


def train(cfg: dict[str, Any], out_dir: str | Path, *, device: str | None = None, overwrite: bool = False,
          log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Train a recognizer as a config says and write the run into ``out_dir``.

    Returns the result: the best grid epoch and its scores, the scores before training (when
    the config asks for them), the curve and the folder.
    """
    from cardvision.core.device import pick_device

    out_dir = Path(out_dir)
    if (out_dir / RUN_NAME).exists() and not overwrite:
        raise FileExistsError(f"{out_dir} already holds a run (pass overwrite to replace it)")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / "log.txt", "w", encoding="utf-8")

    def say(line: str) -> None:
        if log is not None:
            log(line)
        else:
            print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    try:
        return _train(cfg, out_dir, device or pick_device(), say)
    finally:
        log_file.close()


def _train(cfg: dict[str, Any], out_dir: Path, device: str, say: Callable[[str], None]) -> dict[str, Any]:
    from cardvision.composites.cards import KindSettings, load_kinds

    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    started = time.time()

    # A checkpoint to start from is read and checked first: a file that does not fit should
    # stop the run before the data is prepared.
    start = read_start(cfg["model"]["init"], cfg["model"]["dim"]) if is_checkpoint(cfg["model"]["init"]) else None

    sources = build_sources(cfg, out_dir, say)
    catalog = sources.catalog
    kinds = KindSettings.from_kinds(load_kinds(cfg["data"]["kinds"]) if cfg["data"]["kinds"] else catalog.kinds)
    composition = compose(cfg, sources.streams, catalog)
    classes = class_list(composition)
    if not classes:
        raise ConfigError("no positive rows to train on: the config's segments name no stream that has any")
    for name in composition.report.get("emptyStreams", []):
        say(f"note: stream {name} has no rows; its segment gives none")
    dataset = TrainingSet(composition, sources.streams, sources.art, sources.patches, classes, catalog, cfg, kinds)
    sampler = PairBatchSampler(composition.n_single, composition.n_units, composition.per_batch, cfg["batch"], seed,
                               offset=cfg["seeds"]["sampler_offset"])
    steps_per_epoch = len(sampler)
    if steps_per_epoch == 0:
        raise ConfigError(f"an epoch of {len(dataset)} rows does not fill one batch of {cfg['batch']}")
    loader = make_loader(dataset, sampler, cfg["workers"])

    epochs = cfg["epochs"]
    model = Encoder(init="random" if start else cfg["model"]["init"], dim=cfg["model"]["dim"])
    if start:
        model.load_state_dict(start[0])
        say(f"started from {start[1]['init']} (SHA-256 {start[1]['sha256']}): the encoder's "
            f"{start[1]['copiedTensors']} tensors, and nothing else from the file")
    model = model.to(device)
    head = ArcFace(len(classes), cfg["model"]["dim"], cfg["loss"]["arcface_scale"], cfg["loss"]["arcface_margin"]).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=cfg["optimizer"]["lr"],
                                  weight_decay=cfg["optimizer"]["weight_decay"])
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(cfg["optimizer"]["warmup_epochs"] * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, warmup_steps, total_steps))

    prepare = preprocessor(cfg["contract"])
    unseen_ids = set(sources.unseen_ids)
    bank_ids = sorted(card_id for card_id in sources.art if card_id not in unseen_ids)
    bank_inputs = torch.from_numpy(np.stack([prepare(sources.art[card_id]) for card_id in bank_ids])).to(device)
    name_index = name_index_of(classes, catalog.identities()).to(device) if cfg["components"]["printing_mask"] else None
    step = Step(model, head, optimizer, scheduler, cfg["loss"], bank_inputs, name_index)

    known = set(classes)
    seen_rows = [row for row in sources.seen.rows if row["card_id"] in known]
    seen_bank = sources.seen.bank if sources.seen.bank is not None else classes
    unseen_rows = sources.unseen.rows
    unseen_bank = sources.unseen.bank if sources.unseen.bank is not None else sorted(sources.art)
    if not seen_rows or not unseen_rows:
        empty = " and ".join(name for name, rows in (("seen", seen_rows), ("unseen", unseen_rows)) if not rows)
        say(f"note: the {empty} validation set is empty and scores 0; checkpoints are compared on what is left")

    def score_seen() -> float:
        return top1(model, seen_rows, seen_bank, sources.art, device, prepare, sources.seen.root)

    def score_unseen() -> float:
        return top1(model, unseen_rows, unseen_bank, sources.art, device, prepare, sources.unseen.root)

    record: dict[str, Any] = {
        "format": RUN_FORMAT, "formatVersion": 1, "config": plain(cfg), "seed": seed, "dataSeed": cfg["data_seed"],
        "device": device, "versions": versions(), "argv": list(sys.argv[1:]), "inputs": sources.inputs,
        "classes": len(classes), "sameNameClasses": _shared_names(classes, catalog.identities()),
        "composition": composition.report, "stepsPerEpoch": steps_per_epoch, "totalSteps": total_steps,
        "memoryBankCards": len(bank_ids),
        "validation": {"seenRows": len(seen_rows), "unseenRows": len(unseen_rows), "seenBank": len(seen_bank),
                       "unseenBank": len(unseen_bank), "heldOutGroups": sources.held_out_groups,
                       "unseenCards": sources.unseen_ids},
        "notes": sources.notes,
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
    }
    if start:
        record["init"] = dict(start[1])
    (out_dir / RUN_NAME).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    say(f"device={device} classes={len(classes)} rows/epoch={len(dataset)} steps/epoch={steps_per_epoch} "
        f"epochs={epochs} seed={seed} seen={len(seen_rows)} unseen={len(unseen_rows)} "
        f"segments={[(s['stream'], s['rows']) for s in composition.report['segments']]} "
        f"boost={composition.n_boost} pairs={composition.n_units}")

    val = cfg["validation"]
    initial = None
    if val["at_start"]:
        seen, unseen = score_seen(), score_unseen()
        initial = {"epoch": 0, "top1Seen": seen, "top1Unseen": unseen, "criterion": criterion(seen, unseen)}
        say(f"before training: top1-seen {seen:.3f} top1-unseen {unseen:.3f} criterion {initial['criterion']:.3f}")

    def checkpoint(epoch: int, scores: Mapping[str, Any] | None) -> dict[str, Any]:
        return {"model": model.state_dict(), "cards": list(classes), "epoch": epoch,
                "scores": dict(scores) if scores else {}, "config": plain(cfg),
                **({"init": dict(start[1])} if start else {})}

    selection = Selection()
    curve: list[dict[str, Any]] = [] if initial is None else [initial]
    model.train()
    for epoch in range(epochs):
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        t0 = time.time()
        total = 0.0
        step.stats = step.fresh_stats()
        for crops, arts, labels in loader:
            total += step(crops, arts, labels)
        stats = step.stats
        done = max(int(stats["steps"]), 1)
        negatives, backs = max(stats["negatives"], 1), max(stats["backs"], 1)
        line = (f"epoch {epoch + 1}/{epochs} loss {total / done:.4f} negLoss {stats['negativeLoss'] / done:.4f} "
                f"neg/step {stats['negatives'] / done:.1f} meanScore {stats['scoreSum'] / negatives:.3f} "
                f"active {stats['active'] / negatives:.3f} back/step {stats['backs'] / done:.1f} "
                f"backMeanScore {stats['backScoreSum'] / backs:.3f} {time.time() - t0:.0f}s")
        entry: dict[str, Any] = {"epoch": epoch + 1, "loss": total / done, "seconds": round(time.time() - t0, 1),
                                 "negatives": dict(stats)}
        if on_grid(epoch + 1, epochs, val["every"]):
            seen, unseen = score_seen(), score_unseen()
            value = criterion(seen, unseen)
            line += f" top1-seen {seen:.3f} top1-unseen {unseen:.3f} criterion {value:.3f}"
            entry.update({"top1Seen": seen, "top1Unseen": unseen, "criterion": value})
            if selection.offer(epoch + 1, seen, unseen):
                torch.save(checkpoint(epoch + 1, selection.scores), out_dir / "best.pt")
        elif val["log_unseen_every_epoch"]:
            unseen = score_unseen()
            line += f" (log only) top1-unseen {unseen:.3f}"
            entry["top1UnseenLogOnly"] = unseen
        line += f" wall {(time.time() - started) / 60:.1f}min"
        curve.append(entry)
        say(line)
        (out_dir / "curve.json").write_text(json.dumps(curve) + "\n", encoding="utf-8")
    torch.save(checkpoint(epochs, curve[-1]), out_dir / "last.pt")
    result = {"bestEpoch": selection.best_epoch, "bestCriterion": selection.best, "best": selection.scores,
              "initial": initial, "wallMinutes": round((time.time() - started) / 60, 1),
              "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    record["result"] = result
    (out_dir / RUN_NAME).write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    say(f"done: best criterion {selection.best:.3f} at epoch {selection.best_epoch}; {out_dir / 'best.pt'}")
    return {**result, "curve": curve, "out": str(out_dir), "classes": len(classes), "stepsPerEpoch": steps_per_epoch}


def _shared_names(classes: list[str], name_of: Mapping[str, str]) -> int:
    """How many card names have more than one class. Given identity keys in place of names
    (``Catalog.identities()``), how many identities do."""
    counts: dict[str, int] = {}
    for card_id in classes:
        counts[name_of[card_id]] = counts.get(name_of[card_id], 0) + 1
    return sum(count > 1 for count in counts.values())
