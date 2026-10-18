# SPDX-License-Identifier: Apache-2.0
"""``cardvision evaluate hover --browser-events``: hover outcomes recorded in a web browser,
read back and scored by the scorer that scores this package's own frames.

No browser runs here. The events files are written by this module in the form the scoring page
of ``browser/`` writes (``browser/score/run.ts``). In the main test the recorded outcomes are
this package's own, computed on the toy frames, so the two forms of the command must give the
same figures and the same listing.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cardvision import cli
from cardvision.core.images import load_rgb
from cardvision.data.extract import FrameStore
from cardvision.data.manifest import write_manifest
from cardvision.data.schema import Box, Point
from cardvision.evaluate import (
    BrowserEventsError,
    HoverLayer,
    open_configuration,
    read_browser_events,
    read_rows,
    run_browser_hover,
    spec_from_paths,
)
from cardvision.evaluate.browser import EVENTS_KIND, STATUSES, check_correspondence, manifest_warnings
from cardvision.evaluate.config import file_sha256
from cardvision.evaluate.events import ADJACENT, NOTHING
from cardvision.evaluate.rows import Account
from test_evaluate_cli import BOXED, BOXED_POINTS, ROWS, Evaluation, H, W, quad, row, write_release

USER_AGENT = "TestBrowser/1.0 (a made-up user agent)"


def header(release_id: str = "0123456789abcdef", role: str = "evaluation", identity: str | None = "name", **over) -> dict:
    """The first line of an events file, as the scoring page writes it. ``identity`` is the
    identity rule of the release it records; ``None`` leaves the field out, as a file written
    before the page recorded the rule has it."""
    doc = {
        "kind": EVENTS_KIND, "version": 1,
        "release": {"id": release_id, "game": "toy", "locale": "en",
                    "detector": {"version": "toy-detector", "threshold": 0.3, "nmsIou": 0.45, "maxProposals": 80, "minCropPx": 24},
                    "recognizer": {"version": "toy-recognizer"},
                    **({} if identity is None else {"identity": {"level": identity}}),
                    "accept": {"tau": 0.99, "delta": 0.02, "rotationRetry": True},
                    "hover": {"selection": "accepted-nms", "nmsIou": 0.3},
                    "analysisWidth": {"min": 720, "max": 1920}, "ortWeb": "1.29.0"},
        "runtime": {"name": "cardvision-browser", "version": "0.1.0"},
        "userAgent": USER_AGENT,
        "backend": {"asked": "wasm", "detector": "wasm", "recognizer": "wasm"},
        "role": role,
        "manifests": [{"file": "toy.manifest.jsonl", "sha256": "0" * 64, "rows": 4}],
        "frameSelection": {"windowSeconds": 1, "maxHamming": 6},
    }
    doc.update(over)
    return doc


def center(box: Box) -> tuple[float, float]:
    xs, ys = [corner[0] for corner in box.quad], [corner[1] for corner in box.quad]
    return (xs[0] + xs[1] + xs[2] + xs[3]) / 4, (ys[0] + ys[1] + ys[2] + ys[3]) / 4


def line(labeled, status: str = "kept", layer: HoverLayer | None = None, *, video=(W, H), analyzed=(W, H), **over) -> dict:
    """One row line. For a kept row, ``layer`` answers every position as the page would: the
    card shown, and the box the pointer is on, in pixels of the analyzed frame."""
    doc = {"platform": labeled.platform, "video_id": labeled.video_id, "t": labeled.t, "status": status,
           "labels": {"points": len(labeled.points or ()), "boxes": len(labeled.boxes or ())}}
    if status != "no-video":
        doc["video"] = {"file": f"{labeled.video_id}.mp4", "width": video[0], "height": video[1]}
    if status in ("kept", "unreachable"):
        doc["frame"] = {"mediaTime": 12.5, "offset": -0.017, "distance": 2 if status == "kept" else 23, "looked": 61,
                        "exactTime": True}
    if status == "kept":
        def outcome(index: int, x: float, y: float) -> dict:
            shown = layer.at(x * layer.frame_width, y * layer.frame_height)
            proposal = None if shown.index is None else layer.proposals[shown.index]
            return {"index": index, "x": x, "y": y, "cardId": shown.card_id, "name": shown.name,
                    "box": None if shown.box is None else list(shown.box),
                    "similarity": None if proposal is None else proposal.similarity, "margin": None,
                    "accepted": None if proposal is None else proposal.accepted}

        doc["analyzed"] = {"width": analyzed[0], "height": analyzed[1]}
        doc["backend"] = {"detector": "wasm", "recognizer": "wasm"}
        doc["points"] = [outcome(index, point.x, point.y) for index, point in enumerate(labeled.points or ())]
        doc["centers"] = [outcome(index, *center(box)) for index, box in enumerate(labeled.boxes or ())]
    doc.update(over)
    return doc


def write_events(path: Path, *docs: dict) -> Path:
    path.write_text("".join(json.dumps(doc) + "\n" for doc in docs), encoding="utf-8")
    return path


class BrowserSetup(Evaluation):
    """The toy evaluation, and an events file that holds this package's own outcomes."""

    def __init__(self, folder: Path, catalog_path: Path, art_dir: Path) -> None:
        super().__init__(folder, catalog_path, art_dir)
        self.config = open_configuration(spec_from_paths(self.toy.detector, self.toy.recognizer, self.toy.bank,
                                                         self.toy.catalog, self.toy.accept))
        store = FrameStore(self.frames)
        self.layers = {labeled.video_id: HoverLayer.of(self.config.pipeline.analyze(load_rgb(store.path(labeled))))
                       for labeled in (ROWS[0], ROWS[1], ROWS[3])}
        # The third row's frame is missing for the Python form; the browser did not reach it either.
        self.lines = [line(ROWS[0], layer=self.layers[ROWS[0].video_id]), line(ROWS[1], layer=self.layers[ROWS[1].video_id]),
                      line(ROWS[2], "unreachable")]
        self.events = write_events(folder / "browser-events.jsonl", header(), *self.lines)

    def browser_arguments(self, *extra: str, events: Path | None = None) -> list[str]:
        return ["--browser-events", str(events or self.events), "--manifest", str(self.manifest),
                "--catalog", str(self.toy.catalog), *extra]


@pytest.fixture(scope="module")
def setup(tmp_path_factory, toy) -> BrowserSetup:
    return BrowserSetup(tmp_path_factory.mktemp("evaluate-browser"), *toy)


def run(capsys, *arguments: str) -> tuple[int, str, str]:
    code = cli.main(["evaluate", *arguments])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ------------------------------------------------------------------------------ the same figures
def test_outcomes_equal_to_python_s_give_the_figures_and_the_listing_python_gives(setup, capsys, tmp_path):
    seed = ["--bootstrap", "200", "--seed", "4"]
    code, _, err = run(capsys, "hover", *setup.arguments("--out", str(tmp_path / "python"), *seed))
    assert code == 0, err
    code, report, err = run(capsys, "hover", *setup.browser_arguments("--out", str(tmp_path / "browser"), *seed))
    assert code == 0, err
    python = json.loads((tmp_path / "python" / "evaluate-hover.json").read_text())
    browser = json.loads((tmp_path / "browser" / "evaluate-hover.json").read_text())
    # Every count, rate, bound and bootstrap interval, overall and per source.
    assert browser["results"] == python["results"]
    assert browser["results"]["overall"]["counts"]["faceRight"] == 3 and browser["results"]["overall"]["events"] == 15
    assert browser["bootstrap"] == python["bootstrap"] and browser["definitions"] == python["definitions"]
    # The per-event listing, line for line: outcome, name shown, card id and box.
    listing = (tmp_path / "browser" / "evaluate-hover-events.jsonl").read_text()
    assert listing == (tmp_path / "python" / "evaluate-hover-events.jsonl").read_text()
    assert listing.count("\n") == 15
    assert (tmp_path / "browser" / "evaluate-hover.md").read_text() == report
    assert "2 row(s) scored from the browser's outcomes, 1 not scored (listed in the report)" in err


def test_the_library_scores_the_file_as_the_command_does(setup):
    rows, account = read_rows([setup.manifest])
    browser = read_browser_events(setup.events)
    assert browser.role == "evaluation" and len(browser.rows) == 3
    assert [found.status for found in browser.rows.values()] == ["kept", "kept", "unreachable"]
    scored = run_browser_hover(rows, browser, setup.config.catalog.names(), account)
    assert len(scored) == 15 and account.used == 2 and account.frames_present == 2
    assert account.unusable == {"unreachable": 1} and account.skipped == {"role": 1}
    assert sum(item.outcome == "right" for item in scored) == 3
    by_label = {(item.event.video_id, item.event.label): item for item in scored}
    assert by_label[("VIDEOAAAAA1", "edge-of-amber")].outcome == ADJACENT
    # A box that takes the pointer and shows nothing keeps its box and no name.
    copied = by_label[("VIDEOBBBBB2", "copied")]
    assert copied.outcome == NOTHING and copied.shown is None and copied.box is not None


# ------------------------------------------------------------------------------- the account
def test_rows_the_browser_did_not_keep_are_reported_as_rows_that_could_not_be_used(setup, capsys, tmp_path):
    lines = [line(ROWS[0], "unreachable"), line(ROWS[1], "failed", reason="the browser presented no frame at 12.5 s"),
             line(ROWS[2], "no-video")]
    events = write_events(tmp_path / "events.jsonl", header(), *lines)
    code, report, err = run(capsys, "hover", *setup.browser_arguments("--out", str(tmp_path / "out"), "--bootstrap", "20",
                                                                      events=events))
    assert code == 0, err
    result = json.loads((tmp_path / "out" / "evaluate-hover.json").read_text())
    inputs = result["inputs"]
    assert inputs["rowsRead"] == 4 and inputs["rowsUsed"] == 0 and inputs["framesPresent"] == 0
    assert inputs["rowsSkipped"] == {"role": 1}
    assert inputs["rowsUnusable"] == {"failed": 1, "no-video": 1, "unreachable": 1}
    assert inputs["rowsNotScored"][0] == {
        "platform": "youtube", "video_id": "VIDEOAAAAA1", "t": 12.517, "group": "g-one", "source_type": "digital/toy-client",
        "reason": "unreachable", "lost": {"back": 2, "face": 5, "face-unreadable": 1, "off-card": 3},
        "detail": "the nearest frame was 23 bits from the row's fingerprint",
    }
    assert inputs["labelsNotScored"] == {"back": 2, "face": 9, "face-unnamed": 1, "face-unreadable": 1, "off-card": 3,
                                         "unreadable": 1}
    assert "- rows unusable (not scored): failed 1, no-video 1, unreachable 1" in report
    assert ("youtube VIDEOAAAAA1 t=12.517 (digital/toy-client, group g-one): unreachable (the nearest frame was 23 bits "
            "from the row's fingerprint); would have given back 2, face 5, face-unreadable 1, off-card 3") in report
    assert "failed (the browser presented no frame at 12.5 s); would have given" in report
    assert "youtube VIDEOCCCCC3 t=12.517 (physical/webcam, group g-three): no-video; would have given face 2" in report
    # Nothing was scored: no row is in a numerator or a denominator.
    assert result["results"]["overall"]["events"] == 0 and result["results"]["overall"]["namedRight"]["denominator"] == 0
    assert "0 row(s) scored from the browser's outcomes, 3 not scored" in err


def test_a_video_of_another_shape_is_reported_with_its_size(setup, tmp_path):
    events = write_events(tmp_path / "events.jsonl", header(), line(ROWS[0], "frame-shape", video=(900, 900)),
                          setup.lines[1], setup.lines[2])
    rows, account = read_rows([setup.manifest])
    scored = run_browser_hover(rows, read_browser_events(events), setup.config.catalog.names(), account)
    assert account.unusable == {"frame-shape": 1, "unreachable": 1} and account.used == 1 and len(scored) == 4
    assert account.not_scored[0]["detail"] == "the video is 900 x 900"


def test_a_row_with_no_label_for_the_measure_is_skipped_whatever_the_browser_did(setup, tmp_path):
    unlabeled = row("VIDEOEEEEE5", "g-five", "digital/toy-client", boxes=[Box(kind="unreadable", quad=quad(10, 10, 50, 50))])
    manifest = tmp_path / "labels" / "one.manifest.jsonl"
    manifest.parent.mkdir()
    write_manifest(manifest, [unlabeled])
    events = write_events(tmp_path / "events.jsonl", header(), line(unlabeled, "unreachable"))
    rows, account = read_rows([manifest])
    assert run_browser_hover(rows, read_browser_events(events), setup.config.catalog.names(), account) == []
    assert account.skipped == {"no-labels": 1} and account.unusable == {} and account.used == 0


# -------------------------------------------------------------------------------- the report
def test_the_report_says_the_outcomes_came_from_a_browser_and_names_it(setup, capsys, tmp_path):
    code, report, err = run(capsys, "hover", *setup.browser_arguments("--out", str(tmp_path), "--bootstrap", "20"))
    assert code == 0, err
    assert "**the outcomes come from a browser run**: cardvision-browser 0.1.0 on ONNX Runtime Web 1.29.0" in report
    assert "This package ran no model and read no frame" in report
    assert "- backend: detector on wasm, recognizer on wasm (asked for: wasm only)" in report
    assert f"- user agent: `{USER_AGENT}`" in report
    assert "Rows in the events file: kept 2, unreachable 1, no-video 0, frame-shape 0, failed 0" in report
    assert "Fingerprint distance of the kept rows that have one: 2 bits on 2 rows" in report
    assert "release id: as the browser recorded it, not checked against a release document" in report
    assert "- release document id: `0123456789abcdef`" in report
    assert "accept rule: tau 0.99, delta 0.02, the turned looks retried" in report
    assert "hover selection: accepted-nms (suppression overlap 0.3)" in report
    described = json.loads((tmp_path / "evaluate-hover.json").read_text())["configuration"]
    assert set(described["files"]) == {"browserEvents", "catalog"}
    assert described["files"]["browserEvents"]["file"] == "browser-events.jsonl"
    assert described["browser"]["userAgent"] == USER_AGENT
    assert described["browser"]["backend"] == {"asked": "wasm", "detector": "wasm", "recognizer": "wasm"}
    assert described["browser"]["rows"] == {status: {"kept": 2, "unreachable": 1}.get(status, 0) for status in STATUSES}
    assert described["browser"]["releaseChecked"] is False
    # No folder of this machine is printed: files are given by name.
    assert str(setup.manifest.parent) not in report and str(setup.events.parent) not in report


def test_the_report_of_the_python_form_is_unchanged_by_all_this(setup, capsys):
    code, report, err = run(capsys, "hover", *setup.arguments("--bootstrap", "20"))
    assert code == 0, err
    assert "browser" not in report and "user agent" not in report


def test_a_release_document_gives_the_catalog_and_must_be_the_release_the_browser_ran(setup, capsys, tmp_path):
    release = write_release(tmp_path, setup, [(40, 40, 240, 336)], [0.9], [0], {"tau": 0.9, "delta": 0.01, "rotationRetry": True})
    doc = json.loads((release / "release.json").read_text())
    events = write_events(tmp_path / "events.jsonl", header(doc["id"]), *setup.lines)
    # The models are not opened in this form: the folder may hold the document and the catalog alone.
    (release / "detector.onnx").unlink()
    (release / "recognizer.onnx").unlink()
    base = ["--browser-events", str(events), "--manifest", str(setup.manifest), "--bootstrap", "20"]
    code, report, err = run(capsys, "hover", *base, "--release", str(release / "release.json"), "--out", str(tmp_path / "out"))
    assert code == 0, err
    assert "release id: as the browser recorded it, and equal to the release document's" in report
    described = json.loads((tmp_path / "out" / "evaluate-hover.json").read_text())["configuration"]
    assert set(described["files"]) == {"browserEvents", "release", "catalog"}
    assert described["files"]["catalog"]["sha256"] == doc["catalog"]["sha256"] and described["release"] == doc["id"]

    other = write_events(tmp_path / "other.jsonl", header("f" * 16), *setup.lines)
    code, _, err = run(capsys, "hover", "--browser-events", str(other), "--manifest", str(setup.manifest),
                       "--release", str(release / "release.json"))
    assert code == 2 and f"other.jsonl was recorded with release {'f' * 16}, and release.json is release {doc['id']}" in err

    (release / "catalog.json").write_bytes((release / "catalog.json").read_bytes() + b"\n")
    code, _, err = run(capsys, "hover", *base, "--release", str(release / "release.json"))
    assert code == 2 and "catalog.json: its SHA-256 is not the one release.json pins" in err


# ------------------------------------------------------------------------- the identity rule
def test_a_file_is_scored_only_under_the_identity_rule_it_was_recorded_under(setup, capsys, tmp_path):
    """The header records the identity rule of the release the browser ran. A catalog that
    states another rule is refused, in one sentence that says what to do. A file written before
    the rule was recorded is read as ``name``."""
    seed = ["--bootstrap", "20", "--seed", "4"]

    def score(events: Path, catalog: Path, out: Path | None = None) -> tuple[int, str]:
        code, _, err = run(capsys, "hover", "--browser-events", str(events), "--manifest", str(setup.manifest), "--catalog",
                           str(catalog), *seed, *(() if out is None else ("--out", str(out))))
        return code, err

    def result(out: Path) -> dict:
        found = json.loads((out / "evaluate-hover.json").read_text())
        del found["configuration"]["files"]["browserEvents"]   # the two files differ in their header, and so in their hash
        return found

    recorded = {level: write_events(tmp_path / f"recorded-{level}.jsonl", header(identity=level), *setup.lines)
                for level in ("name", "printing", None)}
    assert read_browser_events(recorded["name"]).identity == read_browser_events(recorded[None]).identity == "name"
    assert read_browser_events(recorded["printing"]).identity == "printing"
    assert [read_browser_events(recorded[level]).identity_recorded for level in ("name", "printing", None)] == [True, True, False]

    # The toy catalog states no rule, which is name: the page's file and the older file score alike.
    by_name = setup.toy.catalog
    assert score(recorded["name"], by_name, tmp_path / "now")[0] == 0 and score(recorded[None], by_name, tmp_path / "before")[0] == 0
    assert result(tmp_path / "now") == result(tmp_path / "before")
    assert (tmp_path / "now" / "evaluate-hover-events.jsonl").read_text() == (tmp_path / "before" / "evaluate-hover-events.jsonl").read_text()
    code, err = score(recorded["printing"], by_name)
    assert code == 2 and err == (
        'cardvision evaluate: error: recorded-printing.jsonl was recorded under the identity rule "printing", and '
        'catalog.json states "name": score it with the catalog of the release it was recorded with, or record the '
        "events again with a release of this catalog\n")

    # The same cards under the printing rule.
    by_printing = tmp_path / "printing" / "catalog.json"
    by_printing.parent.mkdir()
    by_printing.write_text(json.dumps({**json.loads(by_name.read_text()), "identity": "printing"}), encoding="utf-8")
    assert score(recorded["printing"], by_printing)[0] == 0
    code, err = score(recorded["name"], by_printing)
    assert code == 2 and 'recorded-name.jsonl was recorded under the identity rule "name", and catalog.json states "printing": ' in err
    code, err = score(recorded[None], by_printing)
    assert code == 2 and err == (
        'cardvision evaluate: error: recorded-None.jsonl records no identity rule, so it is read as recorded under "name", '
        'and catalog.json states "printing": record the events again, with this version of the scoring page and a release '
        "of this catalog\n")

    # A rule this package does not know is a malformed header.
    for level, said in (("art", "header: release.identity.level must be one of ['name', 'printing'], not 'art'"),
                        (7, "header: release.identity.level is not of the expected form")):
        doc = header()
        doc["release"]["identity"] = {"level": level}
        code, err = score(write_events(tmp_path / "odd.jsonl", doc, *setup.lines), by_name)
        assert code == 2 and said in err
    doc = header()
    doc["release"]["identity"] = "printing"
    code, err = score(write_events(tmp_path / "odd.jsonl", doc, *setup.lines), by_name)
    assert code == 2 and "header: release.identity.level is missing" in err


def test_a_release_document_and_the_file_recorded_with_it_state_one_rule(setup, capsys, tmp_path):
    """With ``--release`` the id ties the file to the release, and so to its rule. A file of that
    id that records another rule than the release's catalog states is refused all the same."""
    release = write_release(tmp_path, setup, [(40, 40, 240, 336)], [0.9], [0], {"tau": 0.9, "delta": 0.01, "rotationRetry": True})
    doc = json.loads((release / "release.json").read_text())
    assert doc["identity"]["level"] == "name"
    base = ["--manifest", str(setup.manifest), "--release", str(release / "release.json"), "--bootstrap", "20"]
    for level in ("name", None):
        events = write_events(tmp_path / "events.jsonl", header(doc["id"], identity=level), *setup.lines)
        code, _, err = run(capsys, "hover", "--browser-events", str(events), *base)
        assert code == 0, err
    events = write_events(tmp_path / "events.jsonl", header(doc["id"], identity="printing"), *setup.lines)
    code, _, err = run(capsys, "hover", "--browser-events", str(events), *base)
    assert code == 2 and 'events.jsonl was recorded under the identity rule "printing", and catalog.json states "name"' in err


# ------------------------------------------------------------------- the correspondence check
def refused(setup, capsys, tmp_path, *docs: dict, role: str = "evaluation") -> str:
    events = write_events(tmp_path / "events.jsonl", *docs)
    code, out, err = run(capsys, "hover", *setup.browser_arguments("--role", role, events=events))
    assert code == 2 and out == ""
    return err


def test_a_file_that_does_not_describe_the_manifest_s_rows_is_refused(setup, capsys, tmp_path):
    kept, points, missing = setup.lines
    # A row of the manifest has no line.
    err = refused(setup, capsys, tmp_path, header(), kept, points)
    assert "events.jsonl does not correspond to the manifests: it has no line for the row youtube VIDEOCCCCC3 t=12.517" in err
    # A line belongs to no row of the role.
    stray = line(row("VIDEOZZZZZ9", "g-nine", "physical/webcam", points=BOXED_POINTS), "no-video")
    err = refused(setup, capsys, tmp_path, header(), kept, points, missing, stray)
    assert "its row youtube VIDEOZZZZZ9 t=12.517 is not a row of role 'evaluation' in the manifests" in err
    # Another number of points, on a row that was kept and on one that was not.
    fewer = copy.deepcopy(points)
    fewer["points"] = fewer["points"][:-1]
    fewer["labels"]["points"] -= 1
    err = refused(setup, capsys, tmp_path, header(), kept, fewer, missing)
    assert ("the row youtube VIDEOBBBBB2 t=12.517 has 4 point(s) and 0 box(es) in the manifest, and 3 and 0 in the "
            "events file") in err
    err = refused(setup, capsys, tmp_path, header(), kept, points, {**missing, "labels": {"points": 0, "boxes": 5}})
    assert "the row youtube VIDEOCCCCC3 t=12.517 has 0 point(s) and 2 box(es) in the manifest, and 0 and 5" in err
    # A position that is not the manifest's.
    moved = copy.deepcopy(kept)
    moved["centers"][1]["x"] += 0.001
    err = refused(setup, capsys, tmp_path, header(), moved, points, missing)
    assert "the row youtube VIDEOAAAAA1 t=12.517: centers[1] was recorded at (" in err and "and the manifest puts it at (" in err
    # The file was recorded for another role.
    err = refused(setup, capsys, tmp_path, header(role="train"), kept, points, missing)
    assert "it holds the rows of role 'train', and the role being scored is 'evaluation'" in err
    err = refused(setup, capsys, tmp_path, header(), kept, points, missing, role="train")
    assert "it holds the rows of role 'evaluation', and the role being scored is 'train'" in err


def test_many_mismatches_are_summed_up(setup, tmp_path):
    rows, _ = read_rows([setup.manifest])
    events = write_events(tmp_path / "events.jsonl", header(),
                          *[line(row(f"VIDEO{index:06d}", "g", "physical/webcam", points=BOXED_POINTS), "no-video")
                            for index in range(12)])
    with pytest.raises(BrowserEventsError, match=r"; and 7 more$"):
        check_correspondence(rows, read_browser_events(events), "evaluation")


def test_a_malformed_file_is_refused_with_the_line_at_fault(setup, capsys, tmp_path):
    kept, points, missing = setup.lines

    def broken(change) -> dict:
        doc = copy.deepcopy(kept)
        change(doc)
        return doc

    cases = [
        ([], "events.jsonl: the file is empty"),
        ([{"kind": "something-else"}], "the first line is not the header of a cardvision-browser-events file"),
        ([header(version=2)], "version 2 is not one this package reads (1)"),
        ([header(userAgent=None)], "header: userAgent is not of the expected form"),
        ([{key: value for key, value in header().items() if key != "backend"}], "header: backend.asked is missing"),
        ([header(), {**missing, "status": "skipped"}], "line 2: unknown status 'skipped'"),
        ([header(), kept, kept], "line 3: the row youtube VIDEOAAAAA1 t=12.517 is given twice"),
        ([header(), broken(lambda doc: doc["points"].pop())], "line 2: points must list 3 outcome(s), one per label"),
        ([header(), broken(lambda doc: doc["centers"][0].update(index=4))], "line 2: centers[0] must be an object with index 0"),
        ([header(), broken(lambda doc: doc["points"][0].update(name=None))], "line 2: points[0] must give a cardId and a name, or null for both"),
        ([header(), broken(lambda doc: doc["points"][0].update(box=None))], "line 2: points[0] shows a card without a box"),
        ([header(), broken(lambda doc: doc.pop("analyzed"))], "line 2: analyzed.width is missing"),
        ([header(), {key: value for key, value in missing.items() if key != "labels"}], "line 2: labels.points is missing"),
    ]
    for docs, message in cases:
        assert message in refused(setup, capsys, tmp_path, *docs), message
    (tmp_path / "events.jsonl").write_text(json.dumps(header()) + "\n{not json\n", encoding="utf-8")
    code, _, err = run(capsys, "hover", *setup.browser_arguments(events=tmp_path / "events.jsonl"))
    assert code == 2 and "events.jsonl: line 2 is not valid JSON" in err


# ------------------------------------------------------------------ the frame-selection check
def test_a_kept_row_on_a_frame_the_stated_selection_would_not_keep_is_refused(setup, capsys, tmp_path):
    kept, points, missing = setup.lines

    def framed(**frame) -> dict:
        doc = copy.deepcopy(kept)
        doc["frame"].update(frame)
        return doc

    def selection(window, bits) -> dict:
        return header(frameSelection={"windowSeconds": window, "maxHamming": bits})

    row = "line 2: the row youtube VIDEOAAAAA1 t=12.517 was kept on a frame"
    cases = [
        # Further from the fingerprint than the file's own header allows.
        ([selection(1, 4), framed(distance=5)], f"{row} 5 bits from its fingerprint, and the file's own header gives 4 bits as the most"),
        # Within the header's limit, and further than this package's.
        ([selection(1, 8), framed(distance=7)], f"{row} 7 bits from its fingerprint, and this package keeps a frame at most 6 bits away"),
        # After the window the header states.
        ([header(), framed(offset=1.2)], f"{row} 1.2 s from its time, outside the window of 1 s either side that the file's own header gives"),
        # Before it, where the recorded time is a time the page asked for.
        ([header(), framed(offset=-1.2, exactTime=False)], f"{row} -1.2 s from its time, outside the window of 1 s either side"),
        # A header whose frame selection is not this package's, though every row keeps to it.
        ([selection(1, 8), kept], "events.jsonl: header: the frames were chosen within 1 s of each row's time and 8 bits of its "
                                  "fingerprint, and this package chooses within 1 s and 6 bits"),
        ([selection(2, 6), kept], "the frames were chosen within 2 s of each row's time and 6 bits of its fingerprint, and this "
                                  "package chooses within 1 s and 6 bits"),
        # What the check reads must be there.
        ([header(), framed(distance=1.5)], "line 2: frame.distance must be a whole number of bits, or null"),
        ([header(), {**kept, "frame": {"mediaTime": 12.5, "offset": 0.0, "exactTime": True}}], "line 2: frame.distance is missing"),
        ([header(), {**kept, "frame": {"mediaTime": 12.5, "distance": 0, "exactTime": True}}], "line 2: frame.offset is missing"),
        ([header(), {key: value for key, value in kept.items() if key != "frame"}], "line 2: frame.offset is missing"),
    ]
    for (first, changed), message in cases:
        assert message in refused(setup, capsys, tmp_path, first, changed, points, missing), message


def test_a_kept_row_at_the_edge_of_the_stated_selection_is_read(setup, tmp_path):
    kept, points, missing = setup.lines
    edges = [{"distance": 6}, {"distance": None}, {"offset": 1.0}, {"offset": -1.0, "exactTime": False},
             # Where the browser reported frame times, the frame on screen at the start of the window may have started
             # before the window, by however long it had been on screen.
             {"offset": -3.4, "exactTime": True}]
    for frame in edges:
        doc = copy.deepcopy(kept)
        doc["frame"].update(frame)
        run = read_browser_events(write_events(tmp_path / "events.jsonl", header(), doc, points, missing))
        assert run.rows[(kept["platform"], kept["video_id"], kept["t"])].line["frame"] == doc["frame"]
    # A row that was not kept is not held to the limit: it records the nearest frame found.
    assert read_browser_events(setup.events).rows[("youtube", "VIDEOCCCCC3", 12.517)].line["frame"]["distance"] == 23


# ---------------------------------------------------------------------- the manifests' hashes
def test_a_manifest_that_is_not_the_one_the_browser_read_is_a_warning_and_not_a_refusal(setup, capsys, tmp_path):
    actual = file_sha256(setup.manifest)
    name = setup.manifest.name

    def scored(*manifests: dict) -> str:
        events = write_events(tmp_path / "events.jsonl", header(manifests=list(manifests)), *setup.lines)
        code, _, err = run(capsys, "hover", *setup.browser_arguments("--bootstrap", "20", events=events))
        assert code == 0, err
        assert "2 row(s) scored from the browser's outcomes" in err
        return err

    # The manifest the browser read: nothing is said.
    assert "warning" not in scored({"file": name, "sha256": actual, "rows": 4})
    # The same name and other content.
    err = scored({"file": name, "sha256": "0" * 64, "rows": 4})
    assert (f"cardvision evaluate: warning: {name}: its SHA-256 is {actual}, and events.jsonl recorded {'0' * 64} for the "
            "manifest of that name; the file has changed since the browser read it") in err
    # Neither the name nor the content of a manifest the browser read.
    err = scored({"file": "other.manifest.jsonl", "sha256": "0" * 64, "rows": 4})
    assert (f"warning: {name}: its SHA-256 is {actual}, and events.jsonl recorded no manifest of that name or content "
            "(it recorded: other.manifest.jsonl)") in err
    assert "it recorded: none" in scored()


def test_a_manifest_is_known_by_its_content_whatever_its_name(setup, tmp_path):
    actual = file_sha256(setup.manifest)
    events = write_events(tmp_path / "events.jsonl", header(manifests=[{"file": "renamed.jsonl", "sha256": actual, "rows": 4}]),
                          *setup.lines)
    _, account = read_rows([setup.manifest])
    assert account.manifests == [{"file": setup.manifest.name, "sha256": actual, "rows": 4}]
    assert manifest_warnings(read_browser_events(events), account.manifests) == []
    assert len(manifest_warnings(read_browser_events(setup.events), account.manifests)) == 1


# -------------------------------------------------------------------------------- the command
def test_the_browser_form_takes_no_frames_and_no_model_and_belongs_to_hover(setup, capsys, tmp_path):
    code, _, err = run(capsys, "crops", *setup.browser_arguments())
    assert code == 2 and "--browser-events holds hover outcomes and has no meaning for crops" in err
    code, _, err = run(capsys, "hover", *setup.browser_arguments("--frames", str(setup.frames), "--bank", str(setup.toy.bank)))
    assert code == 2 and "do not also give --frames, --bank" in err
    base = ["--browser-events", str(setup.events), "--manifest", str(setup.manifest)]
    code, _, err = run(capsys, "hover", *base)
    assert code == 2 and "give --catalog, or --release" in err
    code, _, err = run(capsys, "hover", *base, "--catalog", str(setup.toy.catalog), "--release", str(tmp_path / "r.json"))
    assert code == 2 and "give --catalog or --release, not both" in err
    code, _, err = run(capsys, "hover", "--browser-events", str(tmp_path / "none.jsonl"), "--manifest", str(setup.manifest),
                       "--catalog", str(setup.toy.catalog))
    assert code == 2 and "no such events file" in err
    # Without an events file, the frames are still required, as before.
    code, _, err = run(capsys, "hover", "--manifest", str(setup.manifest), *setup.toy.arguments())
    assert code == 2 and "give --frames" in err


# ----------------------------------------------------------------------------------- sections
def test_rows_served_by_a_section_of_a_video_are_scored_like_any_other(setup):
    """A file that holds a section of a video adds ``video.start`` to the row's line, and the
    frame's time is then the time in the file plus that start. Nothing in the scoring reads
    either: the outcomes are the same outcomes."""
    rows, account = read_rows([setup.manifest])
    whole = run_browser_hover(rows, read_browser_events(setup.events), setup.config.catalog.names(), account)
    lines = copy.deepcopy(setup.lines)
    for doc in lines:
        doc["video"] = {**doc["video"], "file": f"youtube-{doc['video_id']}+9.mp4", "start": 9}
    events = write_events(setup.events.parent / "section-events.jsonl", header(), *lines)
    run = read_browser_events(events)
    assert [found.line["video"]["start"] for found in run.rows.values()] == [9, 9, 9]
    sections = run_browser_hover(rows, run, setup.config.catalog.names(), Account(role=account.role))
    assert [(item.event.label, item.outcome, item.shown, item.box) for item in sections] == [
        (item.event.label, item.outcome, item.shown, item.box) for item in whole]


# -------------------------------------------------------------------------------------- boxes
def test_a_box_is_brought_from_the_analyzed_frame_to_the_video_s_pixels(setup, tmp_path):
    """A video twice the analysis size: the page's boxes are in analyzed pixels, and the listing
    and the adjacency rule work in the video's, as they do for a frame this package analyzes."""
    labeled = row("VIDEOJJJJJ0", "g-ten", "digital/toy-client", boxes=BOXED[:1], points=BOXED_POINTS[:1])
    manifest = tmp_path / "labels" / "big.manifest.jsonl"
    manifest.parent.mkdir()
    write_manifest(manifest, [labeled])
    kept = line(labeled, layer=setup.layers[ROWS[0].video_id], video=(2 * W, 2 * H))
    events = write_events(tmp_path / "events.jsonl", header(), kept)
    rows, account = read_rows([manifest])
    scored = run_browser_hover(rows, read_browser_events(events), setup.config.catalog.names(), account)
    assert [item.event.label for item in scored] == ["edge-of-amber", "amber-circle"]
    assert scored[0].outcome == ADJACENT and scored[1].outcome == "right"
    assert scored[1].box == (80.0, 80.0, 480.0, 672.0)


def test_a_label_named_like_another_label_s_place_is_matched_by_position(setup, tmp_path):
    """An unnamed point takes the name ``p<index>``. A point that carries such a name itself is
    told apart from it by where it is."""
    points = [Point(x=160 / W, y=208 / H, kind="face", card_id="TOY-001", identity="named", readable="yes", id="p1"),
              Point(x=0.5, y=0.99, kind="not-card")]
    labeled = row("VIDEOKKKKK1", "g-eleven", "digital/toy-client", points=points)
    manifest = tmp_path / "labels" / "named.manifest.jsonl"
    manifest.parent.mkdir()
    write_manifest(manifest, [labeled])
    events = write_events(tmp_path / "events.jsonl", header(), line(labeled, layer=setup.layers[ROWS[0].video_id]))
    rows, account = read_rows([manifest])
    scored = run_browser_hover(rows, read_browser_events(events), setup.config.catalog.names(), Account())
    assert [(item.event.label, item.outcome, item.shown) for item in scored] == [("p1", "right", "Amber Circle"),
                                                                                 ("p1", "nothing", None)]
