# SPDX-License-Identifier: Apache-2.0
"""Training the detector on toy data: a short run on the CPU from untrained weights, its
checkpoint through the exporter, the run record, and the released recipe run in form."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
pytest.importorskip("yolox.data.data_augment")

import support_train_detector as support  # noqa: E402
from cardvision import cli  # noqa: E402
from cardvision.export.checkpoint import load_checkpoint  # noqa: E402
from cardvision.train_detector import model as M  # noqa: E402
from cardvision.train_detector.augment import Tiles  # noqa: E402
from cardvision.train_detector.config import parse_config  # noqa: E402
from cardvision.train_detector.sources import build_streams  # noqa: E402
from cardvision.train_detector.trainer import train  # noqa: E402
from test_train_detector_config import RELEASED, completed  # noqa: E402

PLUG_IN = "support_train_detector:extra_stream"
SMOKE_STEPS = 20


@pytest.fixture(scope="module")
def data(toy, tmp_path_factory):
    return support.make_data(toy, tmp_path_factory.mktemp("detector-training"))


def small_recipe(data: dict, **top) -> dict:
    """A recipe sized for a test: four images a step, a short warm-up and a larger step size,
    so that twenty steps from untrained weights show the loss falling."""
    doc = {
        "seed": 3, "steps": SMOKE_STEPS, "batch_size": 4, "streams": data["streams"],
        "mix": {"primary": [{"stream": "real", "tiles": 2}, {"stream": "negatives", "tiles": 1},
                            {"stream": "composites", "tiles": 1}],
                "mosaic_extra": [{"stream": "real", "share": 0.4}, {"stream": "composites", "share": 0.4},
                                 {"stream": "negatives", "share": 0.2}]},
        "optimizer": {"lr": 0.003, "warmup_steps": 3},
        "loader": {"workers": 0, "threads": 4},
    }
    doc.update(top)
    return doc


@pytest.fixture(scope="module")
def smoke(data, tmp_path_factory):
    """A 20-step run from untrained weights, through the command line."""
    folder = tmp_path_factory.mktemp("smoke")
    config = folder / "recipe.json"
    config.write_text(json.dumps(small_recipe(data)), encoding="utf-8")
    code = cli.main(["train-detector", "--config", str(config), "--out", str(folder / "run"), "--init", "random",
                     "--device", "cpu"])
    return code, folder, config


def test_a_short_run_trains_and_writes_its_files(smoke):
    code, folder, config = smoke
    assert code == 0
    run = folder / "run"
    assert sorted(path.name for path in run.iterdir()) == ["last.pth", "metrics.json", "run.json", "train-log.jsonl"]
    log = [json.loads(line) for line in (run / "train-log.jsonl").read_text().splitlines()]
    assert [line["step"] for line in log] == list(range(1, SMOKE_STEPS + 1))
    losses = [line["total_loss"] for line in log]
    assert all(np.isfinite(losses))
    # The loss goes down: the last five steps against the first five.
    assert np.mean(losses[-5:]) < np.mean(losses[:5])
    # The learning rate follows the schedule, and the L1 term joins in the closing phase (step 19 on).
    assert [line["lr"] for line in log] == [M.learning_rate(step, 0.003, 3, 18, 0.05) for step in range(SMOKE_STEPS)]
    assert all(line["l1_loss"] == 0 for line in log[:18]) and all(line["l1_loss"] > 0 for line in log[18:])
    assert sum(line["ignoredAnchors"] for line in log) > 0

    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["steps"] == SMOKE_STEPS and metrics["stats"]["images"] == 4 * SMOKE_STEPS
    assert metrics["firstLoss"] == losses[0] and metrics["lastLoss"] == losses[-1]
    assert metrics["stats"]["targets"] > 0 and metrics["stats"]["backs"] > 0 and metrics["stats"]["ignoreRegions"] > 0

    record = json.loads((run / "run.json").read_text())
    assert record["init"] == {"init": "random", "sha256": None, "publicPretrained": False}
    assert record["configSha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert (record["seed"], record["steps"], record["noAugmentFrom"], record["input"]) == (3, 20, 18, [640, 384])
    assert record["config"]["optimizer"]["weight_decay"] == 0.0005  # defaults are filled in
    assert set(record["streams"]) == {"real", "negatives", "composites"}
    assert len(record["streams"]["real"]["framesSha256"]) == 64 and record["streams"]["real"]["frames"] == 6


def test_the_last_checkpoint_is_the_result_and_reads_back_safely(smoke):
    _, folder, _ = smoke
    checkpoint = load_checkpoint(folder / "run" / "last.pth")  # restricted mode: tensors and plain values only
    assert set(checkpoint) == {"model", "raw", "summary", "config", "init"}
    averaged, raw = checkpoint["model"], checkpoint["raw"]
    assert list(averaged) == list(raw) and len(raw) == len(M.build_model(M.nano_experiment(2)).state_dict())
    # The averaged weights trail the trained ones: close to them, not equal.
    name = "backbone.backbone.stem.conv.conv.weight"
    assert not torch.equal(averaged[name], raw[name])
    assert averaged["head.cls_preds.0.weight"].shape[0] == 2
    assert checkpoint["summary"]["steps"] == SMOKE_STEPS and checkpoint["config"]["seed"] == 3


def test_the_checkpoint_goes_through_the_exporter(smoke, capsys):
    pytest.importorskip("onnx")
    pytest.importorskip("safetensors")
    _, folder, _ = smoke
    out = folder / "exported"
    assert cli.main(["export", "detector", str(folder / "run" / "last.pth"), "--out", str(out)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert sorted(path.name for path in out.iterdir()) == [
        "detector-fp16w.onnx", "detector.export.json", "detector.onnx", "detector.safetensors"]
    assert report["kind"] == "detector" and report["output"]["shape"] == [1, 5040, 7]
    assert report["output"]["columns"][5:] == ["cls_face", "cls_back"]


def test_a_run_is_not_replaced_unless_asked(smoke, capsys):
    _, folder, config = smoke
    again = ["train-detector", "--config", str(config), "--out", str(folder / "run"), "--init", "random",
             "--device", "cpu", "--steps", "1"]
    assert cli.main(again) == 2
    assert "already holds a run" in capsys.readouterr().err
    assert json.loads((folder / "run" / "metrics.json").read_text())["steps"] == SMOKE_STEPS


def test_the_run_record_repeats_for_a_seed(data, tmp_path):
    config = parse_config(small_recipe(data, steps=2))
    first = train(config, tmp_path / "a", init="random", device="cpu")
    second = train(config, tmp_path / "b", init="random", device="cpu")
    assert (tmp_path / "a" / "run.json").read_bytes() == (tmp_path / "b" / "run.json").read_bytes()
    assert first["firstLoss"] == pytest.approx(second["firstLoss"], rel=1e-4)
    assert first["lastLoss"] == pytest.approx(second["lastLoss"], rel=1e-3)
    assert first["stats"] == second["stats"]
    other = train(config.with_changes(seed=4), tmp_path / "c", init="random", device="cpu")
    a = json.loads((tmp_path / "a" / "run.json").read_text())
    c = json.loads((tmp_path / "c" / "run.json").read_text())
    assert a != c and {key for key in a if a[key] != c[key]} == {"seed", "config"}
    assert other["firstLoss"] != first["firstLoss"]
    # Nothing in the record tells one moment from another.
    assert not {"time", "date", "started", "wallMinutes"} & set(a)


# ------------------------------------------------------------- starting from a checkpoint
def test_a_run_starts_from_another_run_s_checkpoint_and_says_so(smoke, data, tmp_path, monkeypatch):
    _, folder, _ = smoke
    start = folder / "run" / "last.pth"
    digest = hashlib.sha256(start.read_bytes()).hexdigest()
    file = load_checkpoint(start)
    total = len(file["model"])
    origin = {"init": "last.pth", "sha256": digest, "publicPretrained": False, "copiedTensors": total,
              "widenedFromOneClass": [], "freshClassTensors": []}

    # Every tensor is the file's averaged weights, and no other part of the file is used.
    model = M.build_model(M.nano_experiment(2))
    assert M.initialize(model, start) == origin
    assert list(model.state_dict()) == list(file["model"])
    assert all(torch.equal(value, file["model"][name]) for name, value in model.state_dict().items())
    assert not torch.equal(file["model"]["backbone.backbone.stem.conv.conv.weight"],
                           file["raw"]["backbone.backbone.stem.conv.conv.weight"])

    # The config's key starts a run from it; the log, the record and the checkpoint say so.
    config = parse_config(small_recipe(data, steps=2, init=str(start)))
    lines: list[str] = []
    second = train(config, tmp_path / "second", device="cpu", log=lines.append)
    assert f"started from last.pth (SHA-256 {digest}): all {total} tensors copied, and nothing else from the file" in lines
    record = json.loads((tmp_path / "second" / "run.json").read_text())
    assert record["init"] == origin and record["config"]["init"] == str(start)
    written = load_checkpoint(tmp_path / "second" / "last.pth")
    assert written["init"] == origin and set(written) == {"model", "raw", "summary", "config", "init"}
    # Only the weights came from the file: the steps and the schedule start over.
    log = [json.loads(line) for line in (tmp_path / "second" / "train-log.jsonl").read_text().splitlines()]
    assert [line["step"] for line in log] == [1, 2] and log[0]["lr"] == M.learning_rate(0, 0.003, 3, 2, 0.05)

    # The first step's loss is not that of a start from untrained weights ...
    plain = parse_config(small_recipe(data, steps=2))
    untrained = train(plain, tmp_path / "untrained", init="random", device="cpu")
    assert abs(second["firstLoss"] - untrained["firstLoss"]) > 0.05 * untrained["firstLoss"]

    # ... it is the loss of that run with the network's weights replaced by the file's.
    def replaced(model, init, download=False, trust=False):
        model.load_state_dict(torch.load(start, weights_only=True)["model"])
        return {"init": "random", "sha256": None, "publicPretrained": False}

    monkeypatch.setattr(M, "initialize", replaced)
    by_hand = train(plain, tmp_path / "by-hand", init="random", device="cpu")
    monkeypatch.undo()
    assert by_hand["firstLoss"] == pytest.approx(second["firstLoss"], rel=1e-4)
    assert by_hand["lastLoss"] == pytest.approx(second["lastLoss"], rel=1e-3)

    # A start from untrained weights says nothing more than it did.
    quiet: list[str] = []
    train(plain, tmp_path / "quiet", init="random", device="cpu", log=quiet.append)
    assert not any("started from" in line for line in quiet) and quiet[0].startswith("training 2 steps of 4 on cpu")
    assert "init" not in json.loads((tmp_path / "quiet" / "run.json").read_text())["config"]


def test_the_command_line_takes_the_config_s_place_and_the_exported_weights_start_a_run(smoke, data, tmp_path, capsys):
    pytest.importorskip("onnx")
    pytest.importorskip("safetensors")
    _, folder, _ = smoke
    start = folder / "run" / "last.pth"
    exported = tmp_path / "exported"
    assert cli.main(["export", "detector", str(start), "--out", str(exported)]) == 0
    capsys.readouterr()
    weights = exported / "detector.safetensors"
    file = load_checkpoint(start)["model"]
    model = M.build_model(M.nano_experiment(2))
    record = M.initialize(model, weights)
    assert (record["init"], record["copiedTensors"], record["freshClassTensors"]) == ("detector.safetensors", len(file), [])
    assert all(torch.equal(value, file[name]) for name, value in model.state_dict().items())

    # The config names one file and the command line another: the command line's is used.
    config = tmp_path / "recipe.json"
    config.write_text(json.dumps(small_recipe(data, steps=1, init=str(start))), encoding="utf-8")
    base = ["train-detector", "--config", str(config), "--device", "cpu"]
    assert cli.main([*base, "--out", str(tmp_path / "a")]) == 0
    assert f"started from last.pth (SHA-256 {hashlib.sha256(start.read_bytes()).hexdigest()})" in capsys.readouterr().out
    assert cli.main([*base, "--out", str(tmp_path / "b"), "--init", str(weights)]) == 0
    assert "started from detector.safetensors (SHA-256 " in capsys.readouterr().out
    assert json.loads((tmp_path / "b" / "run.json").read_text())["init"]["sha256"] == record["sha256"]
    assert cli.main([*base, "--out", str(tmp_path / "c"), "--init", "random"]) == 0
    assert "started from" not in capsys.readouterr().out
    assert json.loads((tmp_path / "c" / "run.json").read_text())["init"]["init"] == "random"
    # With a start named in the config, the public checkpoint is not asked for.
    assert cli.main([*base, "--out", str(tmp_path / "d"), "--download-pretrained"]) == 0
    assert "pretrained weights:" not in capsys.readouterr().out


def test_a_checkpoint_that_does_not_fit_is_refused_and_nothing_is_written(smoke, data, tmp_path, capsys):
    _, folder, _ = smoke
    file = load_checkpoint(folder / "run" / "last.pth")["model"]
    name = "head.reg_preds.0.bias"
    cases = {
        "shape.pth": ({**file, name: torch.zeros(8)}, f"the checkpoint's {name} has shape (8,); the network's has (4,)"),
        "short.pth": ({key: value for key, value in file.items() if key != name},
                      f"0 tensor(s) the network lacks and 1 it needs ({name}, ...)"),
        "long.pth": ({**file, "stray": torch.zeros(1)}, "1 tensor(s) the network lacks (stray, ...)"),
    }
    for index, (file_name, (state, message)) in enumerate(cases.items()):
        torch.save({"model": state}, tmp_path / file_name)
        config = tmp_path / f"recipe-{index}.json"
        config.write_text(json.dumps(small_recipe(data, steps=1, init=str(tmp_path / file_name))), encoding="utf-8")
        out = tmp_path / f"run-{index}"
        assert cli.main(["train-detector", "--config", str(config), "--out", str(out), "--device", "cpu"]) == 2
        assert message in capsys.readouterr().err and not out.exists()
    config = tmp_path / "absent.json"
    config.write_text(json.dumps(small_recipe(data, steps=1, init=str(tmp_path / "absent.pth"))), encoding="utf-8")
    assert cli.main(["train-detector", "--config", str(config), "--out", str(tmp_path / "run"), "--device", "cpu"]) == 2
    assert "absent.pth: no such checkpoint file" in capsys.readouterr().err and not (tmp_path / "run").exists()


def test_the_released_recipe_runs_in_form_with_a_plug_in_stream(data, tmp_path):
    try:
        import tomllib
    except ImportError:
        tomllib = pytest.importorskip("tomli")
    doc = tomllib.loads(completed(RELEASED.read_text(encoding="utf-8")))
    # The recipe as released, with toy inputs in place of its streams and a plug-in stream
    # made in the tests in place of the one that is not public.
    doc["extra_sources"] = [PLUG_IN]
    doc["streams"] = {**data["streams"], "extra": doc["streams"]["extra"]}
    doc["loader"] = {"workers": 0, "threads": 4}
    config = parse_config(doc).with_changes(steps=2)
    assert config.batch_size == 16 and config.primary_order().count("extra") == 5

    # It started from a one-class checkpoint, widened to two classes.
    torch.manual_seed(8)
    start = tmp_path / "one-class.pth"
    torch.save({"model": M.nano_experiment(1).get_model().state_dict()}, start)
    summary = train(config, tmp_path / "run", init=start, device="cpu")
    assert summary["steps"] == 2 and summary["stats"]["images"] == 32 and np.isfinite(summary["lastLoss"])
    record = json.loads((tmp_path / "run" / "run.json").read_text())
    assert record["init"]["init"] == "one-class.pth" and len(record["init"]["widenedFromOneClass"]) == 6
    assert record["streams"]["extra"] == {"frames": support.EXTRA_FRAMES, "degrade": 0.6, "origin": "made in the test"}
    assert record["config"]["mix"]["primary"][-1] == {"stream": "extra", "tiles": 5}
    assert record["config"]["extra_sources"] == [PLUG_IN]


def test_loader_processes_give_the_same_images(data):
    config = parse_config(small_recipe(data, steps=2, extra_sources=[PLUG_IN],
                                       mix={"primary": [{"stream": "real", "tiles": 2}, {"stream": "extra", "tiles": 2}],
                                            "mosaic_extra": [{"stream": "composites", "share": 0.5},
                                                             {"stream": "negatives", "share": 0.5}]}))
    tiles = Tiles(build_streams(config), config)
    loader = torch.utils.data.DataLoader(tiles, batch_size=4, shuffle=False, num_workers=2, drop_last=True)
    batches = list(loader)
    assert len(batches) == 2
    for step, (images, targets, ignore, counts) in enumerate(batches):
        assert images.shape == (4, 3, 384, 640) and images.dtype == torch.float32
        for position in range(4):
            image, target, region, count = tiles[step * 4 + position]
            assert np.array_equal(images[position].numpy(), image)
            assert np.array_equal(targets[position].numpy(), target)
            assert np.array_equal(ignore[position].numpy(), region)
            assert counts[position].tolist() == count.tolist()


def test_command_line_errors(data, tmp_path, capsys, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the network was used")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    monkeypatch.setenv(M.CACHE_ENV, str(tmp_path / "cache"))
    out = str(tmp_path / "run")
    assert cli.main(["train-detector", "--config", str(tmp_path / "absent.toml"), "--out", out]) == 2
    assert "no such config file" in capsys.readouterr().err
    # The released recipe's file is incomplete as published, and says so.
    assert cli.main(["train-detector", "--config", str(RELEASED), "--out", out]) == 2
    assert "adds up to 11 tiles" in capsys.readouterr().err
    config = tmp_path / "recipe.json"
    config.write_text(json.dumps(small_recipe(data, steps=1)), encoding="utf-8")
    # Without --init the start is the public checkpoint, which is never fetched unasked.
    assert cli.main(["train-detector", "--config", str(config), "--out", out, "--device", "cpu"]) == 2
    assert "--download-pretrained" in capsys.readouterr().err
    assert cli.main(["train-detector", "--config", str(config), "--out", out, "--init", "random", "--steps", "0"]) == 2
    assert "steps must be a whole number" in capsys.readouterr().err
    assert not Path(out).exists()  # a run that cannot start leaves nothing behind


def test_the_subcommand_is_registered():
    assert "train-detector" in cli.SUBCOMMANDS
    line = next(row for row in cli.available_subcommands().splitlines() if row.split()[:1] == ["train-detector"])
    assert line.split(None, 1)[1] == "Fine-tune the card detector from a config file."
