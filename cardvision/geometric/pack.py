# SPDX-License-Identifier: Apache-2.0
"""The packed descriptor index: the same keypoints as a ``build-features`` index, laid out for a web page.

``cardvision pack-features`` repacks a pinned index (``cardvision.geometric.index``) into a
folder of a manifest and ``SHARDS`` binary shards. It never takes keypoints again: every
descriptor and position is copied from the index. A binary (ORB) index only.

A shard holds the rows of some whole cards: the descriptors first, ``rows * 32`` bytes, then the
positions, ``rows * 2`` little-endian ``float32`` values (x, y per row). Cards are dealt to the
shards in index order, each to the shard its first row falls in when the rows are cut into
``SHARDS`` equal parts, so the shards in order hold the index's rows in order. The manifest
(``manifest.json``, canonical JSON)::

    {"schemaVersion": 1, "kind": "geometric-index-packed",
     "source": {"sha256": "<SHA-256 of the index file>"}, "bank": {"sha256": "<the index's bank>"},
     "opencv": "5.0.0", "features": {...the index's settings block...},
     "cards": [{"cardId", "artV", "width", "height", "shard", "offset", "n"}, ...],
     "shards": [{"path": "shard-0.bin", "sha256": "<64 hex>", "rows": 1234}, ...]}

``offset`` is the card's first row inside its shard, in rows. The positions are the index's own
``float32`` values (the index stores ``float32``, and the stage reads them as ``float32``), so
the packed positions are exactly the stage's. The same index gives the same bytes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from cardvision.bank.documents import canonical_bytes, read_json, sha256_hex
from cardvision.geometric.index import DescriptorIndex, DescriptorIndexError, parse_index

PACKED_KIND = "geometric-index-packed"
PACKED_SCHEMA_VERSION = 1
SHARDS = 8
DESCRIPTOR_BYTES = 32
MANIFEST_NAME = "manifest.json"
_XY = np.dtype("<f4")


class PackError(ValueError):
    """An index cannot be packed, or a packed index does not match its source."""


def shard_name(position: int) -> str:
    return f"shard-{position}.bin"


def pack_index(index: DescriptorIndex, source_sha256: str, shards: int = SHARDS) -> tuple[dict[str, Any], list[bytes]]:
    """The manifest and the shard bytes of a loaded index whose file has SHA-256 ``source_sha256``."""
    descriptor = index.doc.get("descriptor") or {}
    if descriptor.get("dtype") != "uint8" or descriptor.get("width") != DESCRIPTOR_BYTES:
        raise PackError(f"only binary descriptors of {DESCRIPTOR_BYTES} bytes (ORB) can be packed")
    bank = index.doc.get("bank")
    if not isinstance(bank, Mapping) or not isinstance(bank.get("sha256"), str):
        raise PackError("the index records no bank (build it with --bank): a packed index must name its bank")
    total = index.feature_count()
    descs: list[list[np.ndarray]] = [[] for _ in range(shards)]
    xys: list[list[np.ndarray]] = [[] for _ in range(shards)]
    rows = [0] * shards
    cards = []
    start = 0
    for card_id in index.ids:
        found = index.cards[card_id]
        n = len(found)
        shard = min(shards - 1, start * shards // total) if total else 0
        descs[shard].append(np.ascontiguousarray(found.descriptors, dtype=np.uint8))
        xys[shard].append(np.ascontiguousarray(found.xy, dtype=_XY))
        cards.append({"cardId": card_id, "artV": index.art_versions[card_id], "width": found.size[0],
                      "height": found.size[1], "shard": shard, "offset": rows[shard], "n": n})
        rows[shard] += n
        start += n
    blobs = []
    for shard in range(shards):
        desc = b"".join(d.tobytes() for d in descs[shard])
        xy = b"".join(x.tobytes() for x in xys[shard])
        blobs.append(desc + xy)
    manifest = {
        "schemaVersion": PACKED_SCHEMA_VERSION,
        "kind": PACKED_KIND,
        "source": {"sha256": source_sha256},
        "bank": {"sha256": bank["sha256"]},
        "opencv": index.doc.get("opencv"),
        "features": dict(index.features),
        "cards": cards,
        "shards": [{"path": shard_name(i), "sha256": sha256_hex(blobs[i]), "rows": rows[i]} for i in range(shards)],
    }
    return manifest, blobs


def write_packed(folder: str | Path, manifest: dict[str, Any], blobs: list[bytes], overwrite: bool = False) -> bytes:
    """Write the shards and the manifest into ``folder`` and return the manifest's bytes. Refuses
    to replace a file that holds other bytes unless ``overwrite``."""
    folder = Path(folder)
    body = canonical_bytes(manifest)
    files = [(folder / MANIFEST_NAME, body)] + [(folder / entry["path"], blob)
                                                for entry, blob in zip(manifest["shards"], blobs)]
    if not overwrite:
        clash = [path.name for path, data in files if path.exists() and path.read_bytes() != data]
        if clash:
            raise PackError(f"{folder}: {', '.join(clash)} already exist with different content (use --overwrite)")
    folder.mkdir(parents=True, exist_ok=True)
    for path, data in files:
        path.write_bytes(data)
    return body


def unpack(folder: str | Path) -> tuple[dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Read a packed folder back: the manifest and, per card id, ``(descriptors, xy)``. Every shard
    is checked against its recorded SHA-256 and row count."""
    folder = Path(folder)
    manifest = read_json(folder / MANIFEST_NAME)
    blobs = []
    for entry in manifest["shards"]:
        blob = (folder / entry["path"]).read_bytes()
        if hashlib.sha256(blob).hexdigest() != entry["sha256"]:
            raise PackError(f"{entry['path']}: SHA-256 does not match the manifest")
        if len(blob) != entry["rows"] * (DESCRIPTOR_BYTES + 8):
            raise PackError(f"{entry['path']}: {len(blob)} bytes is not {entry['rows']} rows")
        blobs.append(blob)
    out = {}
    for card in manifest["cards"]:
        blob, rows = blobs[card["shard"]], manifest["shards"][card["shard"]]["rows"]
        a, n = card["offset"], card["n"]
        desc = np.frombuffer(blob, np.uint8, rows * DESCRIPTOR_BYTES).reshape(rows, DESCRIPTOR_BYTES)[a:a + n]
        xy = np.frombuffer(blob, _XY, rows * 2, rows * DESCRIPTOR_BYTES).reshape(rows, 2)[a:a + n]
        out[card["cardId"]] = (desc.copy(), xy.astype(np.float32))
    return manifest, out


def verify_round_trip(index: DescriptorIndex, folder: str | Path) -> None:
    """Raise ``PackError`` unless the packed folder holds exactly the index's keypoints: same
    cards in the same order, same descriptors, and the same positions bit for bit as ``float32``."""
    manifest, cards = unpack(folder)
    if [card["cardId"] for card in manifest["cards"]] != index.ids:
        raise PackError("the packed cards are not the index's, in its order")
    for card_id in index.ids:
        found = index.cards[card_id]
        desc, xy = cards[card_id]
        if not np.array_equal(desc, found.descriptors):
            raise PackError(f"{card_id}: descriptors differ after unpacking")
        if xy.shape != found.xy.shape or xy.view(np.uint32).tobytes() != found.xy.astype(np.float32).view(np.uint32).tobytes():
            raise PackError(f"{card_id}: positions differ after unpacking")


def pack_file(index_path: str | Path, out: str | Path, *, config: str | Path | None = None,
              overwrite: bool = False) -> tuple[dict[str, Any], bytes]:
    """Pack the index file ``index_path`` into ``out``. With ``config``, the file's SHA-256 must be
    the one that settings file pins. Verifies the result by unpacking it. Returns the manifest and
    its bytes."""
    index_path = Path(index_path)
    digest = sha256_hex(index_path.read_bytes())
    if config is not None:
        from cardvision.geometric.config import load_geometric

        pinned = load_geometric(config).index_sha256
        if pinned != digest:
            raise PackError(f"{index_path.name}: its SHA-256 is {digest}, not the {pinned} that {Path(config).name} pins")
    try:
        index = parse_index(read_json(index_path), source=index_path.name)
    except DescriptorIndexError as error:
        raise PackError(str(error)) from None
    manifest, blobs = pack_index(index, digest)
    body = write_packed(out, manifest, blobs, overwrite)
    verify_round_trip(index, out)
    return manifest, body


__all__ = ["PACKED_KIND", "PACKED_SCHEMA_VERSION", "SHARDS", "PackError", "pack_file", "pack_index", "unpack",
           "verify_round_trip", "write_packed"]
