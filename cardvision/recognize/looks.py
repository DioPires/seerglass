# SPDX-License-Identifier: Apache-2.0
"""Score one look at a crop against the bank: the best match and its margin.

A *look* is one embedding of a crop, in one of its four orientations. Its similarity to every
bank row is a dot product (both sides are unit vectors, so this is the cosine). Two numbers
summarize the look:

* the **score**: the highest similarity, and the bank row that has it (the first such row on a
  tie);
* the **margin**: the score minus the best similarity among rows of a *different identity*.

Identity is what the margin is measured against. Printings of one card share an identity, so a
crop that sits between two printings of the same card is not treated as doubtful; only a rival
of another identity narrows the margin. Which cards share an identity is the catalog's rule
(``Catalog.identity_of``): by default, the cards with the same name.

Similarities are accumulated in double precision from the single-precision embeddings and bank
rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

# Stands in for "the best other identity" when the bank holds only one identity. It is below
# any cosine, so the margin is then large and never the reason a look is refused.
NO_OTHER_IDENTITY = -2.0


@dataclass(frozen=True)
class Look:
    """One orientation's result: the best bank row, its similarity, and the margin."""

    top: int
    score: float
    margin: float


def identity_columns(ids: Sequence[str], identities: Mapping[str, str]) -> np.ndarray:
    """Identity index of each bank row: rows whose cards have the same identity share an index.

    ``ids`` are the bank's row ids and ``identities`` maps card id to identity key
    (``Catalog.identities()``). Keys are compared as exact strings. Raises ``KeyError`` naming
    the first bank id that has no key.
    """
    missing = [card_id for card_id in ids if card_id not in identities]
    if missing:
        raise KeyError(f"bank card {missing[0]!r} is not in the catalog ({len(missing)} bank id(s) are missing)")
    index = {key: position for position, key in enumerate(sorted({identities[card_id] for card_id in ids}))}
    return np.array([index[identities[card_id]] for card_id in ids], dtype=np.int64)


def name_columns(ids: Sequence[str], names: Mapping[str, str]) -> np.ndarray:
    """``identity_columns`` for a caller that has names and no catalog: rows whose cards
    share a name share an index. ``names`` maps card id to name."""
    return identity_columns(ids, names)


def bank_transposed(rows: np.ndarray) -> np.ndarray:
    """The bank's ``(N, D)`` float32 rows as a contiguous ``(D, N)`` float64 matrix."""
    return np.asarray(rows).astype(np.float64).T.copy()


def similarities(embeddings: np.ndarray, bank_t: np.ndarray) -> np.ndarray:
    """Similarity of every embedding to every bank row: ``(..., D)`` against ``(D, N)`` gives
    ``(..., N)`` float64. ``bank_t`` comes from ``bank_transposed``."""
    embeddings = np.asarray(embeddings)
    if embeddings.shape[-1] != bank_t.shape[0]:
        raise ValueError(f"the embeddings have {embeddings.shape[-1]} values but the bank's rows have {bank_t.shape[0]}")
    flat = embeddings.reshape(-1, embeddings.shape[-1]).astype(np.float64)
    if len(flat) == 0:
        return np.zeros((*embeddings.shape[:-1], bank_t.shape[1]))
    return (flat @ bank_t).reshape(*embeddings.shape[:-1], bank_t.shape[1])


def top_and_other(row: np.ndarray, column_identity: np.ndarray) -> tuple[int, float, float]:
    """For one row of similarities: the best column, its value, and the best value among
    columns of a different identity (``NO_OTHER_IDENTITY`` when there is none)."""
    top = int(np.argmax(row))
    best = float(row[top])
    different = column_identity != column_identity[top]
    other = float(row[different].max()) if different.any() else NO_OTHER_IDENTITY
    return top, best, other


def score_look(row: np.ndarray, column_identity: np.ndarray) -> Look:
    """Summarize one row of similarities as a ``Look``."""
    top, best, other = top_and_other(row, column_identity)
    return Look(top=top, score=best, margin=best - other)


def best_rows(row: np.ndarray, count: int) -> list[int]:
    """The ``count`` highest columns of a row of similarities, best first (ties in column
    order)."""
    return [int(column) for column in np.argsort(-np.asarray(row), kind="stable")[:count]]


def best_identities(row: np.ndarray, column_identity: np.ndarray, count: int) -> list[int]:
    """The best column of each of the ``count`` highest-scoring identities, best first."""
    seen: set[int] = set()
    chosen: list[int] = []
    for column in np.argsort(-np.asarray(row), kind="stable"):
        identity = int(column_identity[column])
        if identity in seen:
            continue
        seen.add(identity)
        chosen.append(int(column))
        if len(chosen) == count:
            break
    return chosen
