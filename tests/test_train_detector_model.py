# SPDX-License-Identifier: Apache-2.0
"""The detector's network, optimizer, schedule and starting weights."""

from __future__ import annotations

import hashlib
import io
import math
import urllib.request

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("yolox")

from cardvision.export.detector import build_yolox_nano  # noqa: E402
from cardvision.train_detector import model as M  # noqa: E402
from cardvision.train_detector.loss import IgnoreRegionHead  # noqa: E402

CLASS_TENSORS = [f"head.cls_preds.{level}.{part}" for level in range(3) for part in ("weight", "bias")]


def fresh(classes: int = 2, seed: int = 0):
    torch.manual_seed(seed)
    return M.build_model(M.nano_experiment(classes))


def shuffled(model, seed: int = 1):
    """The model's weights with noise added, so that they differ from any fresh network's."""
    generator = torch.Generator().manual_seed(seed)
    return {name: value + 0.1 * torch.randn(value.shape, generator=generator) if value.dtype.is_floating_point
            else value.clone() for name, value in model.state_dict().items()}


# -------------------------------------------------------------------------------- the network
def test_the_network_is_the_library_network_with_another_loss():
    model = fresh()
    assert isinstance(model.head, IgnoreRegionHead) and model.head.num_classes == M.NUM_CLASSES == 2
    assert model.head.ignore_boxes is None
    torch.manual_seed(0)
    plain = build_yolox_nano(2)
    built, library = model.state_dict(), plain.state_dict()
    # Same tensors, same values from the same seed: the subclass adds nothing and draws nothing.
    assert list(built) == list(library) and all(torch.equal(built[name], library[name]) for name in built)
    plain.load_state_dict(shuffled(model))  # what "cardvision export detector" does with a trained checkpoint
    # Fresh class and objectness biases stand for a prior probability of 0.01.
    for level in range(3):
        assert torch.allclose(built[f"head.cls_preds.{level}.bias"], torch.full((2,), -math.log(99.0)))
        assert torch.allclose(built[f"head.obj_preds.{level}.bias"], torch.full((1,), -math.log(99.0)))


def test_the_optimizer_has_three_groups_and_decays_one():
    experiment = M.nano_experiment(2)
    model = M.build_model(experiment)
    optimizer = M.build_optimizer(experiment, 16, 0.001, 0.9, 0.0005)
    assert isinstance(optimizer, torch.optim.SGD)
    norms, weights, biases = optimizer.param_groups
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0, 0.0005, 0]
    assert all(group["lr"] == 0.001 and group["momentum"] == 0.9 and group["nesterov"]
               for group in optimizer.param_groups)
    assert all(p.ndim == 1 for p in norms["params"]) and all(p.ndim == 4 for p in weights["params"])
    assert all(p.ndim == 1 for p in biases["params"])
    grouped = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert sorted(grouped) == sorted(id(p) for p in model.parameters())  # each parameter once
    other = M.build_optimizer(experiment, 4, 0.01, 0.8, 0.0)
    assert other is not optimizer and other.param_groups[1]["weight_decay"] == 0.0
    assert other.param_groups[0]["lr"] == 0.01 and other.param_groups[0]["momentum"] == 0.8


def test_the_learning_rate_by_hand():
    def rate(step):
        return M.learning_rate(step, peak=0.001, warmup_steps=30, no_augment_from=450, final_ratio=0.05)

    assert rate(0) == pytest.approx(0.001 / 30) and rate(14) == pytest.approx(0.0005)
    assert rate(29) == pytest.approx(0.001) and rate(30) == pytest.approx(0.001)
    # Halfway down the cosine, step 240, the rate is halfway between the peak and the floor.
    assert rate(240) == pytest.approx(0.001 * (0.05 + 0.95 * 0.5))
    assert rate(135) == pytest.approx(0.001 * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi / 4))))
    assert rate(450) == rate(499) == pytest.approx(0.00005)
    values = [rate(step) for step in range(500)]
    assert all(a < b for a, b in zip(values[:29], values[1:30]))
    assert all(a >= b for a, b in zip(values[30:], values[31:]))
    assert values[449] == pytest.approx(0.00005, rel=1e-3)
    # A run shorter than its warm-up never leaves it.
    assert M.learning_rate(19, 0.001, 30, 18, 0.05) == pytest.approx(0.001 * 20 / 30)


# --------------------------------------------------------------------------- starting weights
def test_a_one_class_checkpoint_is_widened_to_two():
    state = shuffled(fresh(1, seed=5))
    model = fresh(2, seed=6)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    record = M.load_init(model, state)
    after = model.state_dict()
    assert record == {"copiedTensors": len(after) - 6, "widenedFromOneClass": CLASS_TENSORS, "freshClassTensors": []}
    for name in after:
        if name in CLASS_TENSORS:
            assert after[name].shape[0] == 2 and state[name].shape[0] == 1
            assert torch.equal(after[name][0], state[name][0])     # face: the checkpoint's one class
            assert torch.equal(after[name][1], before[name][1])    # back: left as freshly built
        else:
            assert torch.equal(after[name], state[name])


def test_other_class_counts_leave_the_class_layers_fresh():
    eighty = shuffled(fresh(80, seed=5))  # the shape of the public COCO checkpoint
    model = fresh(2, seed=6)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    record = M.load_init(model, eighty)
    after = model.state_dict()
    assert record == {"copiedTensors": len(after) - 6, "widenedFromOneClass": [], "freshClassTensors": CLASS_TENSORS}
    assert all(torch.equal(after[name], before[name] if name in CLASS_TENSORS else eighty[name]) for name in after)
    # Two classes into two: everything is copied.
    two = shuffled(fresh(2, seed=7))
    assert M.load_init(model, two) == {"copiedTensors": len(after), "widenedFromOneClass": [], "freshClassTensors": []}
    assert all(torch.equal(value, two[name]) for name, value in model.state_dict().items())


def test_a_checkpoint_that_does_not_fit_is_refused():
    model = fresh(2)
    state = shuffled(model)
    lacking = {name: value for name, value in state.items() if name != "head.obj_preds.0.bias"}
    with pytest.raises(M.InitError, match="0 tensor.s. the network lacks and 1 it needs .head.obj_preds.0.bias"):
        M.load_init(model, lacking)
    with pytest.raises(M.InitError, match="1 tensor.s. the network lacks .stray"):
        M.load_init(model, {**state, "stray": torch.zeros(1)})
    name = "head.reg_preds.0.bias"
    with pytest.raises(M.InitError, match=f"the checkpoint's {name} has shape .8,.; the network's has .4,."):
        M.load_init(model, {**state, name: torch.zeros(8)})


def test_initialize_from_a_file_and_from_nothing(tmp_path):
    model = fresh(2)
    assert M.initialize(model, "random") == {"init": "random", "sha256": None, "publicPretrained": False}
    state = shuffled(fresh(1, seed=3))
    path = tmp_path / "start.pth"
    torch.save({"model": state, "note": "made in a test"}, path)
    record = M.initialize(model, path)
    assert record["init"] == "start.pth" and record["publicPretrained"] is False
    assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record["widenedFromOneClass"] == CLASS_TENSORS and record["copiedTensors"] == len(state) - 6
    with pytest.raises(M.InitError, match="no such checkpoint file"):
        M.initialize(model, tmp_path / "absent.pth")


def test_the_line_about_the_start_says_which_file_and_what_was_done_with_it(tmp_path):
    def start(classes: int) -> dict:
        path = tmp_path / f"{classes}-classes.pth"
        torch.save({"model": shuffled(fresh(classes, seed=3))}, path)
        return M.initialize(fresh(2), path)

    total = len(fresh(2).state_dict())
    same, one, eighty = start(2), start(1), start(80)
    assert M.describe_start(same) == (f"started from 2-classes.pth (SHA-256 {same['sha256']}): all {total} tensors "
                                      "copied, and nothing else from the file")
    assert M.describe_start(one) == (
        f"started from 1-classes.pth (SHA-256 {one['sha256']}): {total - 6} of {total} tensors copied; the 6 "
        "class-layer tensors widened from the checkpoint's one class: its class became the face class, and the back "
        "class is as freshly built")
    assert M.describe_start(eighty) == (
        f"started from 80-classes.pth (SHA-256 {eighty['sha256']}): {total - 6} of {total} tensors copied; the 6 "
        "class-layer tensors left as freshly built, because the checkpoint has another number of classes")
    assert len(same["sha256"]) == 64 and len({same["sha256"], one["sha256"], eighty["sha256"]}) == 3


# ------------------------------------------------------------------- the public checkpoint
def test_the_public_checkpoint_is_pinned():
    assert M.PRETRAINED_SHA256 == "cd28f55fbbc1829f99d9ac9b38a16d259a22889739c8728ea877610201feff7b"
    assert M.PRETRAINED_URL.startswith("https://github.com/Megvii-BaseDetection/YOLOX/releases/download/")
    assert M.PRETRAINED_URL.endswith("/yolox_nano.pth")


def test_nothing_is_downloaded_unless_asked(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the network was used")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.setenv(M.CACHE_ENV, str(tmp_path / "cache"))
    assert M.pretrained_path() == tmp_path / "cache" / "yolox_nano.pth"
    with pytest.raises(M.InitError, match="--download-pretrained"):
        M.fetch_pretrained()
    with pytest.raises(M.InitError, match="--download-pretrained"):
        M.initialize(fresh(2), None)
    assert not (tmp_path / "cache").exists()


def test_a_download_is_checked_against_the_pinned_hash(tmp_path, monkeypatch):
    content = b"not really weights"
    asked = []

    def fake(url, timeout=None, context=None):
        asked.append(url)
        return io.BytesIO(content)

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    target = tmp_path / "cache" / "yolox_nano.pth"
    # The real pinned hash does not match these bytes: the file is deleted.
    with pytest.raises(M.InitError, match="not the expected cd28f55f"):
        M.fetch_pretrained(download=True, path=target)
    assert asked == [M.PRETRAINED_URL] and list((tmp_path / "cache").iterdir()) == []
    # With the hash of these bytes it is kept, and not fetched a second time.
    digest = hashlib.sha256(content).hexdigest()
    assert M.fetch_pretrained(download=True, path=target, sha256=digest) == target
    assert target.read_bytes() == content
    assert M.fetch_pretrained(path=target, sha256=digest) == target and len(asked) == 2
    # A file on disk that fails the check is refused, whether or not downloading is allowed.
    with pytest.raises(M.InitError, match="Remove it and download again"):
        M.fetch_pretrained(download=True, path=target)
    assert len(asked) == 2


def test_versions_name_what_decides_a_run():
    versions = M.versions()
    assert {"python", "cardvision", "torch", "numpy", "yolox", "yoloxRevisionVerified"} <= set(versions)
    assert len(versions["yoloxRevisionVerified"]) == 40
