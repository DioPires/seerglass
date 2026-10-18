# SPDX-License-Identifier: Apache-2.0
"""Labels made for measuring a model, from the labeling tool to the manifest format: a census
of card centers clicked and then named in Check names (with no model's proposal), and a point
set, are saved through the tool's own save path, exported as `points` rows, and checked with
``cardvision validate-manifest``. The rows read back as the answers that went in, and the
hover events made from them are the ones those answers call for.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from cardvision import cli
from cardvision.data import read_manifest
from cardvision.evaluate.events import row_events

ROOT = Path(__file__).resolve().parents[1]
LABELER = ROOT / "labeler"
VIDEOS = {"table-a": ("youtube", "EXAMPLE0001"), "table-b": ("twitch", "123456789")}


@pytest.fixture(scope="module")
def tool():
    sys.path.insert(0, str(LABELER))
    try:
        import export
        import labelcore
        import make_toy_data
        import serve
        yield export, labelcore, make_toy_data, serve
    finally:
        sys.path.remove(str(LABELER))


def _configure(serve, folder: Path, *argv: str) -> None:
    serve.CFG.__dict__.clear()
    serve.configure(serve.build_parser().parse_args([*argv, "--sample", str(folder / "sample.json"),
                                                     "--catalog", str(folder / "catalog.json")]))


def _save(serve, set_id: str, row: dict) -> dict:
    status, saved = serve.save_frame(set_id, row)
    assert status == 200, saved
    return saved["frame"]


@pytest.fixture(scope="module")
def measured(tool, tmp_path_factory) -> Path:
    """The toy project with a census of table-a (every card's center, then named with no
    proposal, one face left open) in labels/, and a point set of table-a in points/."""
    _export, lc, make_toy_data, serve = tool
    folder = tmp_path_factory.mktemp("measuring") / "toy"
    make_toy_data.build(folder)
    sample = json.loads((folder / "sample.json").read_text(encoding="utf-8"))
    for asset in sample["assets"]:
        platform, video = VIDEOS[asset["assetId"]]
        asset.update(platform=platform, videoId=video)
    (folder / "sample.json").write_text(json.dumps(sample), encoding="utf-8")
    truth = json.loads((folder / "truth.json").read_text(encoding="utf-8"))
    asset = sample["assets"][0]
    size = {"frameW": make_toy_data.FRAME_W, "frameH": make_toy_data.FRAME_H}

    # the census: one click on the center of every card, faces and backs
    _configure(serve, folder, "--mode", "center", "--role", "development")
    rows = []
    for record in asset["frames"]:
        centers = []
        for card in truth["frames"][f"table-a/{record['name']}"]:
            xs, ys = [p[0] for p in card["corners"]], [p[1] for p in card["corners"]]
            centers.append({"id": f"C{len(centers) + 1:03d}", "x": round(sum(xs) / 4 / size["frameW"], 6),
                            "y": round(sum(ys) / 4 / size["frameH"], 6), "kind": "back" if card["outcome"] == "card-back" else "face",
                            "cardId": card["cardId"]})
        rows.append((record, centers))
        stored = [{k: v for k, v in c.items() if k != "cardId"} for c in centers]
        _save(serve, "table-a", {"name": record["name"], "t": record["t"], **size, "done": True, "centers": stored, "centerSeq": len(stored)})

    # Check names, the form a start from the page runs: every center named with no proposal
    _configure(serve, folder, "--mode", "verify", "--role", "development", "--all-blind")
    sha = serve.CFG.verify["defaultSha"]
    left_open = None
    for record, centers in rows:
        for c in centers:
            if c["kind"] != "face":
                continue
            if left_open is None:
                left_open = (record["name"], c["id"])
                continue
            stored = json.loads(json.dumps(_stored(folder, record["name"])))
            mine = next(x for x in stored["centers"] if x["id"] == c["id"])
            mine.update(eligible="yes", blind=True)
            stored = _save(serve, "table-a", stored)
            mine = next(x for x in stored["centers"] if x["id"] == c["id"])
            mine.update(verifiedBy="picked", identity=c["cardId"], blind=True, verifiedAt={"x": mine["x"], "y": mine["y"], "armsManifestSha256": sha})
            _save(serve, "table-a", stored)
    assert lc.answer_complete(next(c for c in _stored(folder, rows[-1][0]["name"])["centers"] if c["kind"] == "face"))

    # a point set of the same video, in a labels folder of its own
    _configure(serve, folder, "--mode", "point", "--role", "development", "--labels-dir", str(folder / "points"), "--k", "4")
    for record in asset["frames"][:2]:
        index = lc.frame_index(record["name"], record["t"])
        points = lc.points_for_frame(index, 1, 4)
        for p in points:
            p.update(outcome="not-card", eligible="no")
        _save(serve, "table-a", {"name": record["name"], "t": record["t"], **size, "done": True, "points": points})
    serve.CFG.__dict__.clear()
    (folder / "left-open.json").write_text(json.dumps(left_open), encoding="utf-8")
    return folder


def _stored(folder: Path, name: str) -> dict:
    rows = [json.loads(line) for line in (folder / "labels" / "table-a" / "frames.jsonl").read_text().splitlines() if line.strip()]
    return next(r for r in rows if r["name"] == name)


@pytest.fixture(scope="module")
def manifest(tool, measured, tmp_path_factory) -> Path:
    export = tool[0]
    out = tmp_path_factory.mktemp("measure") / "toy.manifest.jsonl"
    assert export.main(["manifest", "--labels-dir", str(measured / "labels"), "--labels-dir", str(measured / "points"),
                        "--sample", str(measured / "sample.json"), "--credit", "Toy Labeler", "--source-type", "physical/webcam",
                        "--out", str(out), "--catalog", str(measured / "catalog.json")]) == 0
    return out


def test_the_points_rows_pass_the_validator(manifest, measured):
    assert cli.main(["validate-manifest", str(manifest), "--catalog", str(measured / "catalog.json"), "--contribution"]) == 0


def test_the_rows_read_back_as_the_answers(manifest, measured):
    rows = read_manifest(manifest)
    truth = json.loads((measured / "truth.json").read_text(encoding="utf-8"))
    frames = json.loads((measured / "sample.json").read_text(encoding="utf-8"))["assets"][0]["frames"]
    assert len(rows) == len(frames)
    open_frame, open_id = json.loads((measured / "left-open.json").read_text(encoding="utf-8"))
    for row, record in zip(rows, frames):
        assert row.completeness == "points"
        assert not row.boxes
        cards = truth["frames"][f"table-a/{record['name']}"]
        centers = [p for p in row.points if p.origin == "center"]
        assert [p.kind for p in centers] == ["back" if c["outcome"] == "card-back" else "face" for c in cards]
        for point, card in zip(centers, cards):
            if card["outcome"] == "card-back":
                assert (point.identity, point.card_id) == (None, None)
            elif (record["name"], point.id) == (open_frame, open_id):
                assert (point.identity, point.card_id, point.readable, dict(point.attributes)) == ("unlabeled", None, None, {})
            else:
                assert (point.identity, point.card_id, point.readable) == ("named", card["cardId"], "yes")
                assert dict(point.attributes) == {"readable_from": "crop", "blind": True}
        seeded = [p for p in row.points if p.origin == "seeded"]
        assert len(seeded) == (4 if record in frames[:2] else 0), "the point set's frames share a row with the census"
        assert all(p.kind == "not-card" for p in seeded)


def test_the_points_give_the_hover_events_their_answers_call_for(manifest, measured):
    catalog = json.loads((measured / "catalog.json").read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in catalog["cards"]}
    truth = json.loads((measured / "truth.json").read_text(encoding="utf-8"))
    frames = json.loads((measured / "sample.json").read_text(encoding="utf-8"))["assets"][0]["frames"]
    events = [e for row in read_manifest(manifest) for e in row_events(row, names)]
    cards = [c for record in frames for c in truth["frames"][f"table-a/{record['name']}"]]
    faces = sum(1 for c in cards if not c["outcome"] == "card-back")
    assert sum(1 for e in events if e.kind == "point" and e.role == "face") == faces - 1, "every named face, readable"
    assert sum(1 for e in events if e.role == "face-unnamed") == 1, "the face left open"
    assert sum(1 for e in events if e.role == "back") == len(cards) - faces
    assert sum(1 for e in events if e.role == "off-card") == 8, "the point set's 4 points on each of 2 frames"
