# SPDX-License-Identifier: Apache-2.0
"""The paper catalog: what a bank for streams of paper cards holds.

* **Faces.** Every entry of the full catalog (``cardvision.mtg.catalog``) whose printing exists
  on paper (``extra.games`` names ``paper``). Arena-only printings (Alchemy's digital cards,
  3,594 faces of the 104,995 on 2026-10-21) never lie on a table, and a bank row for one could
  only ever win wrongly.
* **Tokens** (``cardvision.mtg.arena.token_entries``), from the token sets of every set that
  has a paper printing, each marked ``"token": true`` and grouped as for Arena: a token named
  like a card joins that card's identity, and any other token is named ``"<name> token"``, one
  identity per name.

A paper card is a whole card wherever it lies, so there are no tile entries. Each identity's
representative is chosen again among the entries kept, and ``extra`` is cut to the fields the
Arena tools keep (``cardvision.mtg.arena.SLIM_EXTRA``): the same builder makes both catalogs
(``cardvision.mtg.arena.pool_catalog``). The accept rule for paper is
``games/mtg/accept-paper.json``, with the rotation retry on: a tapped paper card turns a
quarter turn.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from cardvision.mtg.arena import PoolCatalog, pool_catalog

ENVIRONMENT = "paper"


def paper_catalog(full: Mapping[str, Any], tokens: Iterable[Mapping[str, Any]] = ()) -> PoolCatalog:
    """The paper catalog, from the full catalog document and the token entries of
    ``token_entries``.

    The result is checked with the package's own loader before it is returned.
    """
    return pool_catalog(full, tokens, environment=ENVIRONMENT, tiles=False)
