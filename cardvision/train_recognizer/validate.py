# SPDX-License-Identifier: Apache-2.0
"""Scoring during training, and which epoch's weights are kept.

``top1`` is retrieval accuracy against clean card images: every card of a bank is embedded
once from its image, each validation crop (the card's box, cut exactly, with no augmentation)
is embedded, and the crop counts as right when the nearest bank card is its own card id.

Two sets are scored (``cardvision.train_recognizer.sources``):

* ``seen``: real crops of training cards, from recording groups that were held out whole,
  against the training cards' images;
* ``unseen``: composite crops of cards that were kept out of training, against the images of
  every card. This is the set that notices an encoder which has learned its training cards and
  nothing that carries over to new ones.

They are scored on a grid of epochs: every ``every`` epochs and the last one. The checkpoint
kept as ``best.pt`` is the one with the highest ``0.5 * top1_seen + 0.5 * top1_unseen``; on a
tie the later epoch wins.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch

from cardvision.train_recognizer.sources import Row, load_image, whole_inner

BANK_CHUNK = 512
CROP_CHUNK = 256


@torch.no_grad()
def embed_images(model: torch.nn.Module, images: Sequence[np.ndarray], device: str,
                 prepare: Callable[[np.ndarray], np.ndarray], chunk: int = BANK_CHUNK) -> torch.Tensor:
    """The embeddings of a list of images, in evaluation mode; the model's mode is restored."""
    was_training = model.training
    model.eval()
    out = []
    for start in range(0, len(images), chunk):
        batch = np.stack([prepare(image) for image in images[start:start + chunk]])
        out.append(model(torch.from_numpy(batch).to(device)))
    if was_training:
        model.train()
    width = out[0].shape[1] if out else 0
    return torch.cat(out) if out else torch.zeros(0, width, device=device)


def card_box(row: Row, root: str | None = None) -> np.ndarray:
    """A validation row's crop: the card's box cut out of the row's image."""
    image = load_image(row, root)
    x, y, w, h = whole_inner(row["inner"], image.shape[1], image.shape[0])
    return image[y:y + h, x:x + w]


@torch.no_grad()
def top1(model: torch.nn.Module, rows: Sequence[Row], bank_ids: Sequence[str], art: Mapping[str, np.ndarray],
         device: str, prepare: Callable[[np.ndarray], np.ndarray], root: str | None = None) -> float:
    """The share of rows whose nearest clean card image, among ``bank_ids``, is their own card.
    0.0 for no rows."""
    if not rows:
        return 0.0
    cards = sorted(bank_ids)
    bank = embed_images(model, [art[card_id] for card_id in cards], device, prepare)
    correct = 0
    for start in range(0, len(rows), CROP_CHUNK):
        batch = rows[start:start + CROP_CHUNK]
        emb = embed_images(model, [card_box(row, root) for row in batch], device, prepare, chunk=CROP_CHUNK)
        nearest = (emb @ bank.T).argmax(1).cpu().numpy()
        correct += sum(cards[int(p)] == row["card_id"] for row, p in zip(batch, nearest))
    return correct / len(rows)


def criterion(seen: float, unseen: float) -> float:
    """The number checkpoints are compared by."""
    return 0.5 * seen + 0.5 * unseen


def on_grid(epoch: int, epochs: int, every: int) -> bool:
    """Whether a finished epoch (counted from 1) is one the two sets are scored on."""
    return epoch % every == 0 or epoch == epochs


class Selection:
    """Keeps track of the best epoch offered so far."""

    def __init__(self) -> None:
        self.best = -1.0
        self.best_epoch: int | None = None
        self.scores: dict[str, Any] = {}

    def offer(self, epoch: int, seen: float, unseen: float) -> bool:
        """Offer an epoch's scores; True when it is now the best (a tie goes to the later)."""
        value = criterion(seen, unseen)
        if value >= self.best:
            self.best, self.best_epoch = value, epoch
            self.scores = {"epoch": epoch, "top1Seen": seen, "top1Unseen": unseen, "criterion": value}
            return True
        return False
