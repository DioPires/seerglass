# SPDX-License-Identifier: Apache-2.0
"""The example scripts under ``examples/`` run as the guides say they do."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from test_infer_toy_models import EXPECTED_CLASSES, EXPECTED_NAMES

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def quickstart(monkeypatch):
    """The quickstart script as a module. It puts ``tests/`` on the import path; that is undone."""
    pytest.importorskip("onnx")
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location("toy_quickstart", ROOT / "examples" / "toy_quickstart.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_toy_quickstart_writes_its_files_and_prints_what_the_pipeline_found(quickstart, tmp_path, capsys):
    out = tmp_path / "work" / "toy"
    assert quickstart.main(["toy_quickstart.py", str(out)]) == 0
    printed, errors = capsys.readouterr()
    assert errors == ""
    assert sorted(path.name for path in out.iterdir()) == [
        "accept.json", "art", "bank.json", "catalog.json", "detector.onnx", "recognizer.onnx", "table.png"]
    assert len(list((out / "art").glob("*.png"))) == 20
    lines = printed.splitlines()
    assert lines[0] == f"wrote {out / 'catalog.json'} and 20 images in {out / 'art'}"
    assert lines[1] == f"wrote stand-in models, a staged bank and a test scene in {out}"
    assert lines[2] == "table.png: 7 boxes"
    # One line per box, in the detector's score order, saying what the scene was staged to show.
    names = iter(EXPECTED_NAMES)
    for index, (line, cls) in enumerate(zip(lines[3:10], EXPECTED_CLASSES)):
        assert line.startswith(f"  {index}  {cls}  ")
        if cls == "back":
            assert line.endswith("never named")
        else:
            name = next(names)
            assert f"named {name or 'nothing'}  " in line and "best match TOY-" in line
    assert lines[10] == "1 frame, 6 faces, 3 named" and len(lines) == 11
    # The card turned a quarter is named from a turned look; the card the bank lacks is not named.
    assert lines[4].endswith("look 3") and "similarity 1.000" in lines[4]
    assert "best match TOY-015 Plum Triangle, similarity 1.000, margin 0.000" in lines[6]


def test_the_toy_quickstart_refuses_a_folder_inside_the_repository_and_a_wrong_call(quickstart, capsys):
    inside = ROOT / "examples" / "never-written"
    assert quickstart.main(["toy_quickstart.py", str(inside)]) == 2
    assert "lies inside the repository" in capsys.readouterr().err
    assert not inside.exists()
    assert quickstart.main(["toy_quickstart.py"]) == 2
    assert capsys.readouterr().err.strip() == quickstart.USAGE
