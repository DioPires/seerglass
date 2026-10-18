# SPDX-License-Identifier: Apache-2.0
"""Fetching background scans, against a server on this machine. No test here uses the network:
the shipped asset list is read and checked, never fetched."""

from __future__ import annotations

import hashlib
import http.server
import io
import json
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cardvision.composites import fetch as F
from cardvision.composites import generate, read_labels
from cardvision.composites.surfaces import is_surface, load_texture_index
from cardvision.core.catalog import load_catalog

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = ROOT / "cardvision" / "composites" / "cc0_textures.json"


def tiny_jpeg(seed: int) -> bytes:
    """A small generated picture, as JPEG bytes."""
    pixels = (np.random.default_rng(seed).random((24, 32, 3)) * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


class Site:
    """A local HTTP server with a table of pages, counting the requests for each."""

    def __init__(self) -> None:
        self.pages: dict[str, bytes] = {}
        self.requests: list[str] = []
        self.agents: set[str] = set()
        site = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                site.requests.append(self.path)
                site.agents.add(self.headers.get("User-Agent", ""))
                body = site.pages.get(self.path)
                self.send_response(200 if body is not None else 404)
                self.send_header("Content-Length", str(len(body or b"")))
                self.end_headers()
                self.wfile.write(body or b"")

            def log_message(self, *args) -> None:
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@pytest.fixture
def site():
    server = Site()
    yield server
    server.close()


def asset(asset_id: str, data: bytes, **extra) -> dict:
    return {"source": "toy", "assetId": asset_id, "category": "wood", "split": "train",
            "page": f"https://example.invalid/a/{asset_id}", "license": "CC0 1.0",
            "licenseUrl": "https://example.invalid/license", "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), **extra}


@pytest.fixture
def listed(site, tmp_path):
    """Four assets: one by direct link, one through a file listing, one whose bytes differ from
    its hash, one that is not there. Returns ``(asset list path, the two good files' bytes)``."""
    direct, through_listing = tiny_jpeg(1), tiny_jpeg(2)
    changed = tiny_jpeg(4)[:-3] + b"\x00\xff\xd9"  # as long as the listed file, one byte different
    site.pages["/dl/oak_veneer.jpg"] = direct
    site.pages["/files/plain_cloth"] = json.dumps({
        "Displacement": {"1k": {"jpg": {"url": f"{site.base}/dl/wrong_map.jpg"}}},
        "Diffuse": {"1k": {"jpg": {"url": f"{site.base}/dl/plain_cloth_diff_1k.jpg", "size": len(through_listing)},
                           "png": {"url": f"{site.base}/dl/plain_cloth_diff_1k.png"}},
                    "2k": {"jpg": {"url": f"{site.base}/dl/plain_cloth_diff_2k.jpg"}}},
    }).encode()
    site.pages["/dl/plain_cloth_diff_1k.jpg"] = through_listing
    site.pages["/dl/pine_planks.jpg"] = changed
    assets = [
        asset("oak_veneer", direct, url=f"{site.base}/dl/oak_veneer.jpg"),
        asset("plain_cloth", through_listing, category="fabric", split="val", listing=f"{site.base}/files/plain_cloth"),
        asset("pine_planks", tiny_jpeg(4), url=f"{site.base}/dl/pine_planks.jpg"),
        asset("lost_marble", tiny_jpeg(5), listing=f"{site.base}/files/lost_marble"),
    ]
    path = tmp_path / "assets.json"
    path.write_text(json.dumps({"description": "test scans", "assets": assets}), encoding="utf-8")
    return path, {"oak_veneer": direct, "plain_cloth": through_listing}


# ------------------------------------------------------------------------------- downloading
def test_scans_are_downloaded_verified_and_indexed(site, listed, tmp_path):
    index, good = listed
    out = tmp_path / "scans"
    lines = []
    summary = F.fetch_textures(index, out, delay=0, log=lines.append)
    assert (summary["assets"], summary["downloaded"], summary["present"]) == (4, 2, 0)
    assert [(item["assetId"], item["source"]) for item in summary["failed"]] == [("pine_planks", "toy"), ("lost_marble", "toy")]
    assert "sha256 is" in summary["failed"][0]["reason"] and "404" in summary["failed"][1]["reason"]
    # Exactly the verified files are on disk, split by use, and nothing else.
    files = sorted(str(path.relative_to(out)) for path in out.rglob("*") if path.is_file())
    assert files == ["index.json", "train/toy__oak_veneer.jpg", "val/toy__plain_cloth.jpg"]
    assert (out / "train/toy__oak_veneer.jpg").read_bytes() == good["oak_veneer"]
    assert (out / "val/toy__plain_cloth.jpg").read_bytes() == good["plain_cloth"]
    records = load_texture_index(out)
    assert [(r["file"], r["split"], r["category"]) for r in records] == [
        ("train/toy__oak_veneer.jpg", "train", "wood"), ("val/toy__plain_cloth.jpg", "val", "fabric")]
    for record in records:
        data = (out / record["file"]).read_bytes()
        assert record["sha256"] == hashlib.sha256(data).hexdigest() and record["bytes"] == len(data)
        assert record["license"] == "CC0 1.0" and record["source"] == "toy" and record["page"].startswith("https://")
    assert records[0]["url"] == f"{site.base}/dl/oak_veneer.jpg"
    assert records[1]["url"] == f"{site.base}/dl/plain_cloth_diff_1k.jpg"  # read from the listing: 1K JPEG color map
    assert "/dl/wrong_map.jpg" not in site.requests and "/dl/plain_cloth_diff_2k.jpg" not in site.requests
    assert len(lines) == 4 and sum(line.startswith("fetched") for line in lines) == 2
    assert all(agent.startswith("cardvision/") for agent in site.agents)


def test_a_second_run_downloads_only_what_is_missing(site, listed, tmp_path):
    index, good = listed
    out = tmp_path / "scans"
    F.fetch_textures(index, out, delay=0)
    first = list(site.requests)
    site.pages["/dl/pine_planks.jpg"] = None  # gone now; it was never accepted anyway
    summary = F.fetch_textures(index, out, delay=0)
    assert (summary["downloaded"], summary["present"], len(summary["failed"])) == (0, 2, 2)
    again = site.requests[len(first):]
    assert "/dl/oak_veneer.jpg" not in again and "/dl/plain_cloth_diff_1k.jpg" not in again
    assert "/files/plain_cloth" not in again
    assert [r["assetId"] for r in load_texture_index(out)] == ["oak_veneer", "plain_cloth"]
    # A file damaged on disk is fetched again.
    (out / "train/toy__oak_veneer.jpg").write_bytes(b"not the picture")
    summary = F.fetch_textures(index, out, delay=0)
    assert summary["downloaded"] == 1 and (out / "train/toy__oak_veneer.jpg").read_bytes() == good["oak_veneer"]


def test_a_download_larger_than_listed_is_not_kept(site, tmp_path):
    data = tiny_jpeg(7)
    site.pages["/dl/big.jpg"] = data + b"\0" * 100
    index = tmp_path / "assets.json"
    index.write_text(json.dumps([asset("big", data, url=f"{site.base}/dl/big.jpg")]))
    summary = F.fetch_textures(index, tmp_path / "scans", delay=0)
    assert summary["downloaded"] == 0 and "larger than" in summary["failed"][0]["reason"]
    assert not (tmp_path / "scans" / "train").exists()


def test_the_split_is_derived_from_the_id_when_the_list_has_none(site, tmp_path):
    data = tiny_jpeg(8)
    site.pages["/dl/x.jpg"] = data
    record = asset("some_scan", data, url=f"{site.base}/dl/x.jpg")
    del record["split"], record["bytes"]
    index = tmp_path / "assets.json"
    index.write_text(json.dumps([record]))
    F.fetch_textures(index, tmp_path / "scans", delay=0)
    [written] = load_texture_index(tmp_path / "scans")
    assert written["split"] == F.split_of("some_scan") and written["file"].startswith(written["split"] + "/")
    assert written["bytes"] == len(data)


def test_generated_scenes_can_use_fetched_scans(site, listed, toy, tmp_path):
    index, _ = listed
    F.fetch_textures(index, tmp_path / "scans", delay=0)
    manifest = generate(load_catalog(toy[0]), toy[1], tmp_path / "out", n=6, seed=4, textures_dir=tmp_path / "scans")
    assert manifest["textures"] == {"scans": 1, "surfaceLike": 1} and manifest["settings"]["background"] == "mix"
    scanned = [r["background"] for r in read_labels(tmp_path / "out") if "assetId" in r["background"]]
    assert scanned and all(b["assetId"] == "oak_veneer" and b["license"] == "CC0 1.0" for b in scanned)


# --------------------------------------------------------------------------------- refusals
@pytest.mark.parametrize(("change", "message"), [
    ({"license": "CC BY 4.0"}, "is not 'CC0 1.0'"),
    ({"license": None}, "is not 'CC0 1.0'"),
    ({"assetId": "../escape"}, '"assetId" must be made of'),
    ({"source": "a/b"}, '"source" must be made of'),
    ({"sha256": "abc"}, "64 lower-case hex digits"),
    ({"sha256": None}, "64 lower-case hex digits"),
    ({"sha256": "0" * 64 + "\n"}, "64 lower-case hex digits"),
    ({"assetId": "oak\n"}, '"assetId" must be made of'),
    ({"source": "web\n"}, '"source" must be made of'),
    ({"url": "file:///etc/hosts"}, "http(s) link"),
    ({"url": None}, 'needs a "url" or a "listing"'),
    ({"split": "test"}, '"split" must be'),
    ({"bytes": 0}, '"bytes" must be'),
])
def test_an_unusable_asset_is_refused_before_any_request(site, tmp_path, change, message):
    record = asset("oak_veneer", tiny_jpeg(1), url=f"{site.base}/dl/oak_veneer.jpg")
    record.update(change)
    record = {key: value for key, value in record.items() if value is not None}
    index = tmp_path / "assets.json"
    index.write_text(json.dumps([record]))
    with pytest.raises(F.FetchError) as error:
        F.fetch_textures(index, tmp_path / "scans", delay=0)
    assert message in str(error.value) and site.requests == [] and not (tmp_path / "scans").exists()


def test_an_empty_or_repeating_list_is_refused(tmp_path):
    index = tmp_path / "assets.json"
    for document in ([], {"assets": []}, {"scans": [1]}, ["oak"]):
        index.write_text(json.dumps(document))
        with pytest.raises(F.FetchError):
            F.load_asset_list(index)
    twice = asset("oak_veneer", b"x", url="https://example.invalid/x.jpg")
    index.write_text(json.dumps([twice, twice]))
    with pytest.raises(F.FetchError, match="listed twice"):
        F.load_asset_list(index)


def test_the_color_map_is_found_in_a_listing():
    link = {"jpg": {"url": "https://example.invalid/a_1k.jpg"}}
    assert F.color_map_url({"Diffuse": {"1k": link}}) == "https://example.invalid/a_1k.jpg"
    assert F.color_map_url({"nor_gl": {"1k": link}, "col": {"2k": link}}) == "https://example.invalid/a_1k.jpg"
    assert F.color_map_url({"Diffuse": {"4k": link}, "Color": {"1k": link}}) == "https://example.invalid/a_1k.jpg"
    for listing in ({}, [], {"Diffuse": {"4k": link}}, {"Diffuse": {"1k": {"png": {"url": "x"}}}}, {"Rough": {"1k": link}}):
        with pytest.raises(F.FetchError, match="color map"):
            F.color_map_url(listing)


def test_the_split_of_an_id_is_fixed_and_mostly_training():
    assert F.split_of("oak_veneer_01") == F.split_of("oak_veneer_01")
    splits = [F.split_of(f"scan_{number}") for number in range(400)]
    assert set(splits) == {"train", "val"} and 25 <= splits.count("val") <= 80


# -------------------------------------------------------------------------- the shipped list
def test_the_shipped_asset_list_is_metadata_only_and_well_formed():
    document = json.loads(SHIPPED.read_text(encoding="utf-8"))
    assert set(document) == {"description", "assets"} and "CC0 1.0" in document["description"]
    assets = F.load_asset_list(SHIPPED)
    assert len(assets) == 207 and SHIPPED.stat().st_size < 200_000
    keys = {"source", "assetId", "category", "split", "page", "listing", "license", "licenseUrl", "bytes", "sha256"}
    for item in assets:
        assert set(item) == keys and item["source"] == "polyhaven" and item["license"] == "CC0 1.0"
        assert item["page"] == f"https://polyhaven.com/a/{item['assetId']}"
        assert item["listing"] == f"https://api.polyhaven.com/files/{item['assetId']}"
        assert item["licenseUrl"] == "https://polyhaven.com/license"
        assert item["split"] == F.split_of(item["assetId"]) and 10_000 < item["bytes"] < 5_000_000
    assert len({item["sha256"] for item in assets}) == 207
    assert sum(item["split"] == "val" for item in assets) == 24
    surface_like = [item for item in assets if is_surface(item["category"], item["assetId"])]
    assert len(surface_like) == 89 and {item["category"] for item in surface_like} == {"fabric", "wood", "concrete"}
    assert any(item["split"] == "val" for item in surface_like)


def test_no_image_is_kept_beside_the_generator():
    kept = sorted(path.name for path in SHIPPED.parent.iterdir() if path.is_file())
    assert all(name.endswith((".py", ".json")) for name in kept) and "cc0_textures.json" in kept
