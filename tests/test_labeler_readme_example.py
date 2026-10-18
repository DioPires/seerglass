# SPDX-License-Identifier: Apache-2.0
"""The label files that ``labeler/README.md`` says a script can write ("Labels written by a
script"): the three examples of that section are written to disk as they stand in the page, in
plain Python, and the manifest export is run on them with the command the page gives. The page
cannot drift from what the export accepts without this test failing.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

from cardvision import cli

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "labeler" / "README.md"


def section() -> str:
    text = README.read_text(encoding="utf-8")
    start, end = text.index("## Labels written by a script"), text.index("## Checking the labeling")
    assert start < end
    return text[start:end]


def blocks(kind: str) -> list[str]:
    """The fenced blocks of the section that open with ```kind (``""``: no kind), in order."""
    found: list[tuple[str, list[str]]] = []
    inside = False
    for line in section().splitlines():
        if line.startswith("```"):
            inside = not inside
            if inside:
                found.append((line[3:], []))
        elif inside:
            found[-1][1].append(line)
    assert not inside
    return ["".join(text + "\n" for text in lines) for opened, lines in found if opened == kind]


def test_the_section_has_the_three_files_the_layout_and_the_command():
    sample, meta, frame = (json.loads(block) for block in blocks("json"))
    assert set(sample) == {"assets"} and meta["mode"] == "box" and frame["done"] is True
    layout, command, output = blocks("")
    assert "sample.json" in layout and "meta.json" in layout and "frames.jsonl" in layout
    assert command.startswith("python3 labeler/export.py manifest ") and output.startswith("wrote ")
    # Every field the two tables name is in the examples or marked optional, and the other way round.
    first_cells = re.findall(r"^\| (`[^|]*) \|", section(), flags=re.MULTILINE)
    documented = {name for cell in first_cells for name in re.findall(r"`([A-Za-z]+)`", cell)}
    used = set(frame) | {key for target in frame["targets"] for key in target}
    assert used <= documented
    assert documented - used - set(meta) == {"recording", "printing", "ignore", "conditions", "labelSource", "cornerSource"}


def test_the_export_accepts_the_examples_as_they_stand(tmp_path, toy):
    sample, meta, frame = blocks("json")
    _, command, output = blocks("")
    asset = json.loads(sample)["assets"][0]["assetId"]
    work = tmp_path / "work"
    (work / "labels" / asset).mkdir(parents=True)
    (work / "sample.json").write_text(sample, encoding="utf-8")
    (work / "labels" / asset / "meta.json").write_text(meta, encoding="utf-8")
    # One frame's record is one line of the file.
    (work / "labels" / asset / "frames.jsonl").write_text(json.dumps(json.loads(frame)) + "\n", encoding="utf-8")

    words = shlex.split(command.replace("\\\n", " "))
    assert words[:2] == ["python3", "labeler/export.py"]
    done = subprocess.run([sys.executable, str(ROOT / "labeler" / "export.py"), *words[2:]], cwd=tmp_path, capture_output=True,
                          text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout == output.replace("...", str(tmp_path.resolve()))

    manifest = work / "manifest" / "scripted.manifest.jsonl"
    (row,) = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert (row["platform"], row["video_id"], row["t"], row["source_type"]) == ("local", asset, 12.5, "physical/phone")
    assert [(box["id"], box["kind"], box.get("card_id"), box.get("identity"), box.get("readable"), box["shape"]) for box in row["boxes"]] == [
        ("B001", "face", "TOY-001", "named", "yes", "rotated-box"), ("B002", "face", None, "unknown", "no", "rotated-box"),
        ("B003", "back", None, None, None, "rotated-box")]
    # The first corner of the second box is its top-left as the card lies: the card is on its side.
    assert row["boxes"][1]["quad"][0] == [round(765 / 1280, 6), round(300 / 720, 6)]
    catalog, _ = toy
    assert cli.main(["validate-manifest", str(manifest), "--catalog", str(catalog)]) == 0


def test_the_export_says_what_is_missing_without_a_source_type(tmp_path):
    """The message that the guides' ``--source-type`` answers."""
    sample, meta, frame = blocks("json")
    asset = json.loads(sample)["assets"][0]["assetId"]
    (tmp_path / "labels" / asset).mkdir(parents=True)
    (tmp_path / "sample.json").write_text(sample, encoding="utf-8")
    (tmp_path / "labels" / asset / "meta.json").write_text(meta, encoding="utf-8")
    (tmp_path / "labels" / asset / "frames.jsonl").write_text(json.dumps(json.loads(frame)) + "\n", encoding="utf-8")
    done = subprocess.run([sys.executable, str(ROOT / "labeler" / "export.py"), "manifest", "--labels-dir", str(tmp_path / "labels"),
                           "--sample", str(tmp_path / "sample.json"), "--credit", "Your Name", "--out", str(tmp_path / "out.jsonl")],
                          capture_output=True, text=True, check=False)
    assert done.returncode != 0 and "no source type: tag the recording's source in the tool, or pass --source-type" in done.stderr
