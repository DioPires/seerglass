# SPDX-License-Identifier: Apache-2.0
"""The recognizer's training losses.

Three terms, on unit-length embeddings:

``ArcFace``
    Classification over the training cards with an additive angular margin: one weight row per
    card id, logits ``scale * cos(angle)``, and the margin added to the angle of the true
    class. It separates the training cards from one another.
``contrastive``
    Cross-domain: each crop is pulled towards the embedding of *its own* card's clean image
    and pushed from the clean images of the other cards in the batch (InfoNCE with a
    temperature). This is what makes a bank of clean card images work for cards that were
    never trained on.
``negative_hinge``
    For crops that show no card face (a card back, something that is not a card): the highest
    similarity to any clean card image must stay below a margin. The clean images are a memory
    bank of embeddings, refreshed every so many steps, together with the batch's own clean
    images. Negatives get no class and no bank entry.

**Same-name masking.** Printings that share a name are one identity, though each keeps its
own class and its own clean image. With a ``name_index`` (class index to name index), the
other printings of a sample's name are neither wrong classes in ArcFace nor negatives in the
contrastive term: their logits are removed from the softmax, so they receive no gradient.
Without it, every printing is a negative for every other. Which classes are one identity is
the catalog's rule (``Catalog.identities()``, from which training builds the index): under
the ``printing`` rule the index joins a card with its printings, and two different cards of
one name stay negatives for each other.

**Card-back weighting.** A negative marked as a card back can be given a larger weight and a
lower margin than the other negatives.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# Labels a training batch carries for rows that show no card face.
NEGATIVE = -1
NEGATIVE_BACK = -2


class ArcFace(nn.Module):
    """Additive-angular-margin classification head.

    Args:
        n_classes: the number of training cards.
        dim: the embedding width.
        scale: the factor on the cosines (``s``).
        margin: the angle added to the true class, in radians (``m``).
    """

    def __init__(self, n_classes: int, dim: int, scale: float, margin: float) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, dim) * 0.01)
        self.s, self.m = scale, margin

    def forward(self, emb: torch.Tensor, labels: torch.Tensor, name_index: torch.Tensor | None = None) -> torch.Tensor:
        """The loss for unit embeddings ``(n, dim)`` and class labels ``(n,)``. With
        ``name_index`` (``(n_classes,)`` long), the other classes of a sample's name are masked
        out."""
        cos = emb @ F.normalize(self.W, dim=1).T
        theta = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
        target = torch.cos(theta + self.m)
        onehot = F.one_hot(labels, self.W.shape[0]).bool()
        logits = self.s * torch.where(onehot, target, cos)
        if name_index is not None:
            same = (name_index[labels][:, None] == name_index[None, :]) & ~onehot
            logits = logits.masked_fill(same, float("-inf"))
        return F.cross_entropy(logits, labels)


def contrastive(crop_emb: torch.Tensor, art_emb: torch.Tensor, labels: torch.Tensor, temperature: float,
                name_index: torch.Tensor | None = None) -> torch.Tensor:
    """InfoNCE from each crop to its own clean image (row ``i`` of ``art_emb`` belongs to row
    ``i`` of ``crop_emb``).

    The other clean images of the *same card* in the batch are not negatives and are left out
    of the denominator; with ``name_index``, so are those of the same *name*.
    """
    sim = crop_emb @ art_emb.T / temperature
    same = labels[:, None] == labels[None, :]
    if name_index is not None:
        names = name_index[labels]
        same = same | (names[:, None] == names[None, :])
    eye = torch.eye(len(labels), dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(same & ~eye, float("-inf"))
    return F.cross_entropy(sim, torch.arange(len(labels), device=sim.device))


def negative_scores(neg_emb: torch.Tensor, bank: torch.Tensor, art_emb: torch.Tensor | None = None) -> torch.Tensor:
    """For each negative, its highest similarity to a clean card image: the larger of the best
    match in the memory bank (no gradient flows into the bank) and the best match among the
    batch's clean-image embeddings (gradient flows into both sides)."""
    s = (neg_emb @ bank.T).max(1).values
    if art_emb is not None and len(art_emb):
        s = torch.maximum(s, (neg_emb @ art_emb.T).max(1).values)
    return s


def negative_hinge(scores: torch.Tensor, is_back: torch.Tensor, margin: float, back_margin: float | None = None,
                   back_weight: float = 1.0, weight: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """The mean over negatives of ``w * relu(score - m)``.

    A negative marked in ``is_back`` uses ``back_weight`` and ``back_margin``; the others use
    ``weight`` and ``margin``. With ``back_margin`` None, every negative uses ``margin`` with
    weight 1. Returns ``(loss, excess)`` where ``excess`` is ``score - m`` per negative.
    """
    if back_margin is None:
        excess = scores - margin
        return F.relu(excess).mean(), excess
    m = torch.where(is_back, torch.full_like(scores, back_margin), torch.full_like(scores, margin))
    w = torch.where(is_back, torch.full_like(scores, back_weight), torch.full_like(scores, weight))
    excess = scores - m
    return (w * F.relu(excess)).mean(), excess
