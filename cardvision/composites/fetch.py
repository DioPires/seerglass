# SPDX-License-Identifier: Apache-2.0
"""Download the CC0 background scans named in an asset list, and verify every byte.

The asset list is a JSON file: either a list of records or an object with the list under
``"assets"``. ``cc0_textures.json`` beside this module is such a list, of 207 scans. A scene
picks its scan by position, so the order of the list is part of what makes a run repeatable.
A record::

    {"source": "polyhaven", "assetId": "oak_veneer_01", "category": "wood", "split": "train",
     "page": "https://polyhaven.com/a/oak_veneer_01",
     "listing": "https://api.polyhaven.com/files/oak_veneer_01",
     "license": "CC0 1.0", "licenseUrl": "https://polyhaven.com/license",
     "bytes": 579182, "sha256": "..."}

* ``source`` and ``assetId`` name the scan; the file is saved as ``<source>__<assetId>.jpg``.
* The download link is ``url`` when the record has one. Otherwise ``listing`` is fetched: the
  source's own JSON description of the asset's files, from which the link of the 1K JPEG color
  map is read (``color_map_url``).
* ``sha256`` is required. A download whose hash differs is not written, and the run fails.
* ``license`` must be ``"CC0 1.0"``; a record under any other license is refused.
* ``split`` is ``"train"`` or ``"val"``; when left out it is derived from the asset id
  (``split_of``), about one scan in eight going to validation.
* ``category`` decides whether the scan may lie under a game board (``surfaces.is_surface``).

The output folder gets ``<split>/<source>__<assetId>.jpg`` for each scan and ``index.json``,
the records of the verified files with their ``file`` path and resolved ``url``. A file that
is already there with the right hash is not downloaded again, so an interrupted run can be
repeated.

Only the color map of each material is taken: a flat picture of a surface, about 0.03 to
1.6 MB each. Material scans are used because they show a surface and nothing else.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cardvision
from cardvision.composites.surfaces import INDEX_NAME
from cardvision.core.tls import CERTIFICATE_HELP, is_certificate_failure, tls_context

ACCEPTED_LICENSE = "CC0 1.0"
MAX_BYTES = 64 * 1024 * 1024  # a download with no declared size is cut off here
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COLOR_KEYS = ("Diffuse", "diffuse", "diff", "Color", "col")


class FetchError(ValueError):
    """The asset list is not usable, or a download could not be resolved. The message says why."""


def split_of(asset_id: str) -> str:
    """``"val"`` for about one asset id in eight, ``"train"`` for the rest; fixed per id, so a
    scan never serves both training and validation scenes."""
    return "val" if int(hashlib.sha256(asset_id.encode()).hexdigest(), 16) % 8 == 0 else "train"


def color_map_url(listing: Any) -> str:
    """The link of the 1K JPEG color map in a source's file listing (2K when it has no 1K).

    The listing maps a map name to resolutions to formats to ``{"url": ...}``; the color map
    is the first of ``Diffuse``, ``diffuse``, ``diff``, ``Color``, ``col`` that is present.
    """
    if isinstance(listing, dict):
        for key in _COLOR_KEYS:
            maps = listing.get(key)
            if isinstance(maps, dict):
                sized = maps.get("1k") or maps.get("2k")
                if isinstance(sized, dict) and isinstance(sized.get("jpg"), dict) and sized["jpg"].get("url"):
                    return str(sized["jpg"]["url"])
    raise FetchError("the file listing has no 1K or 2K JPEG color map")


def load_asset_list(path: str | Path) -> list[dict[str, Any]]:
    """Read and check an asset list. Raises ``FetchError`` naming the first bad record."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    assets = doc.get("assets") if isinstance(doc, dict) else doc
    if not isinstance(assets, list) or not assets:
        raise FetchError(f'{Path(path).name}: expected a non-empty list of assets (or an object with "assets")')
    seen = set()
    for position, asset in enumerate(assets):
        where = f"{Path(path).name}: asset {position}"
        if not isinstance(asset, dict):
            raise FetchError(f"{where}: must be an object")
        for key in ("source", "assetId"):
            if not isinstance(asset.get(key), str) or not _SAFE_NAME.fullmatch(asset[key]):
                raise FetchError(f'{where}: "{key}" must be made of letters, digits, "_", "." and "-"')
        where = f"{where} ({asset['source']} {asset['assetId']})"
        if (asset["source"], asset["assetId"]) in seen:
            raise FetchError(f"{where}: listed twice")
        seen.add((asset["source"], asset["assetId"]))
        if not isinstance(asset.get("sha256"), str) or not _SHA256.fullmatch(asset["sha256"]):
            raise FetchError(f'{where}: "sha256" must be 64 lower-case hex digits')
        if asset.get("license") != ACCEPTED_LICENSE:
            raise FetchError(f'{where}: license {asset.get("license")!r} is not {ACCEPTED_LICENSE!r}')
        links = [asset.get(key) for key in ("url", "listing") if asset.get(key) is not None]
        if not links or not all(isinstance(link, str) and _is_http(link) for link in links):
            raise FetchError(f'{where}: needs a "url" or a "listing" that is an http(s) link')
        if asset.get("split", "train") not in ("train", "val"):
            raise FetchError(f'{where}: "split" must be "train" or "val"')
        if "bytes" in asset and (not isinstance(asset["bytes"], int) or asset["bytes"] < 1):
            raise FetchError(f'{where}: "bytes" must be a positive whole number')
    return assets


def _is_http(link: str) -> bool:
    return link.lower().startswith(("http://", "https://"))


def _get(url: str, limit: int, timeout: float) -> bytes:
    if not _is_http(url):
        raise FetchError(f"refused link {url!r}: only http and https are fetched")
    request = urllib.request.Request(url, headers={"User-Agent": f"cardvision/{cardvision.__version__} fetch-textures"})
    context = tls_context() if url.lower().startswith("https://") else None
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise FetchError(f"{url}: larger than the {limit} bytes expected")
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_textures(index: str | Path, out_dir: str | Path, delay: float = 0.3, timeout: float = 60.0,
                   log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Download every scan of an asset list into ``out_dir`` and write its ``index.json``.

    ``delay`` is the pause after each request, in seconds. Returns a summary:
    ``{"assets", "downloaded", "present", "failed": [{"source", "assetId", "reason"}]}``.
    A scan that fails (unreachable, no color map, wrong hash) is left out of ``index.json``
    and listed under ``failed``; the others are still fetched. A server whose certificate
    cannot be verified is another matter: every request would fail the same way, so the run
    stops with a ``FetchError`` that says what to install.
    """
    say = log if log is not None else (lambda message: None)
    assets = load_asset_list(index)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    downloaded = present = 0
    for asset in assets:
        split = asset.get("split") or split_of(asset["assetId"])
        rel = f"{split}/{asset['source']}__{asset['assetId']}.jpg"
        target = out_dir / rel
        url = asset.get("url")
        if target.is_file() and _sha256(target) == asset["sha256"]:
            present += 1
        else:
            try:
                if url is None:
                    url = color_map_url(json.loads(_get(asset["listing"], MAX_BYTES, timeout)))
                    time.sleep(delay)
                data = _get(url, asset.get("bytes", MAX_BYTES), timeout)
                time.sleep(delay)
                got = hashlib.sha256(data).hexdigest()
                if got != asset["sha256"]:
                    raise FetchError(f"sha256 is {got}, the list says {asset['sha256']}")
            except (FetchError, OSError, urllib.error.URLError, json.JSONDecodeError) as error:
                if is_certificate_failure(error):
                    raise FetchError(f"{asset['source']} {asset['assetId']}: {CERTIFICATE_HELP}") from None
                failed.append({"source": asset["source"], "assetId": asset["assetId"], "reason": str(error)})
                say(f"failed  {asset['source']} {asset['assetId']}: {error}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            downloaded += 1
            say(f"fetched {rel} ({len(data)} bytes)")
        record = {key: asset[key] for key in ("source", "assetId", "category", "page", "license", "licenseUrl",
                                              "sha256") if key in asset}
        record.update({"file": rel, "split": split, "bytes": target.stat().st_size})
        if url is not None:
            record["url"] = url
        records.append(record)
    (out_dir / INDEX_NAME).write_text(json.dumps(records, indent=1) + "\n", encoding="utf-8")
    return {"assets": len(assets), "downloaded": downloaded, "present": present, "failed": failed}
