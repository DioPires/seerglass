# SPDX-License-Identifier: Apache-2.0
"""The generated toy game, and the core modules working together on it."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import toy_game
from cardvision.core import bank as B
from cardvision.core.catalog import load_catalog
from cardvision.core.images import load_rgb
from cardvision.core.onnx import create_session, embed, embed_four_looks
from cardvision.core.preprocess import full128


def test_twenty_distinct_card_images(toy):
    catalog_path, art_dir = toy
    files = sorted(art_dir.iterdir())
    assert len(files) == 20 and all(path.suffix == ".png" for path in files)
    images = {path.stem: load_rgb(path) for path in files}
    assert set(images) == {card_id for card_id, *_ in toy_game.CARDS}
    assert len({image.tobytes() for image in images.values()}) == 20
    # An alternate printing shares the name and not the picture.
    assert images["TOY-003"].shape == images["TOY-003A"].shape
    assert not np.array_equal(images["TOY-003"], images["TOY-003A"])


def test_drawing_is_deterministic(toy, tmp_path):
    catalog_path, art_dir = toy
    again_catalog, again_art = toy_game.write_toy_game(tmp_path)
    assert again_catalog.read_bytes() == catalog_path.read_bytes()
    for path in art_dir.iterdir():
        assert np.array_equal(load_rgb(path), load_rgb(again_art / path.name))


def test_the_script_writes_the_game(tmp_path):
    script = Path(toy_game.__file__)
    done = subprocess.run([sys.executable, str(script), str(tmp_path / "out")], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "20 images" in done.stdout
    assert len(load_catalog(tmp_path / "out" / "catalog.json")) == 20
    assert subprocess.run([sys.executable, str(script)], capture_output=True).returncode != 0


def test_catalog_to_bank_to_match(toy, tiny_model, tmp_path):
    """Art -> embeddings -> stored bank -> loaded bank -> each card finds itself."""
    catalog_path, art_dir = toy
    catalog = load_catalog(catalog_path)
    session = create_session(tiny_model)
    art = [load_rgb(catalog.art_location(card_id, art_dir)) for card_id in catalog.ids()]

    rows = B.l2_normalize(embed(session, art, full128))
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(B.encode_bank(catalog.ids(), rows, extra={"game": catalog.game})), encoding="utf-8")

    ids, matrix, doc = B.load_bank(path)
    assert ids == catalog.ids() and doc["game"] == "toy" and matrix.shape == (20, 128)
    identity = B.IdentityGroups(B.groups_by_name(ids, catalog.names()), ids, known=catalog.ids())

    # Upright queries: the best row is the card itself.
    queries = B.l2_normalize(embed(session, art, full128))
    best = (queries @ matrix.T).argmax(axis=1)
    assert [ids[column] for column in best] == ids

    # A card lying on its side: take the best of the four looks. One of them is upright again.
    turned = [np.ascontiguousarray(np.rot90(image, 1)) for image in art]
    looks = B.l2_normalize(embed_four_looks(session, turned, full128))        # (20, 4, 128)
    scores = looks @ matrix.T                                               # (20, 4, 20)
    best = scores.max(axis=1).argmax(axis=1)
    assert [identity.of(ids[column]) for column in best] == [identity.of(card_id) for card_id in ids]
