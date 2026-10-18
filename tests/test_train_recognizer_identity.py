# SPDX-License-Identifier: Apache-2.0
"""The recognizer trainer and the identity rule a catalog states. Skipped without PyTorch.

What the trainer treats as one class does not depend on the rule: every card id is a class of
its own, with its own clean image. The rule enters in one place, the mask of the losses that
keeps the classes of one identity from being negatives for each other.

Under the ``name`` rule, stated or left out, a run on the toy game must be exactly what it was
before a catalog could state a rule. The values below were recorded then: the classes, the
mask, the first batch the sampler draws and the loss of the first step. A second run with the
rule written out must repeat the first to the last bit: every batch, every loss, every weight.
"""

from __future__ import annotations

import hashlib
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import support_train_recognizer as support  # noqa: E402
from cardvision.core.catalog import load_catalog  # noqa: E402
from cardvision.train_recognizer import loop  # noqa: E402
from cardvision.train_recognizer.config import parse_config  # noqa: E402

# Recorded on the toy game before the identity rule existed (17 cards: the 20 less the 3 kept unseen).
CLASSES = ["TOY-001", "TOY-002", "TOY-003", "TOY-003A", "TOY-004", "TOY-006", "TOY-007", "TOY-008", "TOY-009", "TOY-010",
           "TOY-011", "TOY-013", "TOY-014", "TOY-016", "TOY-017", "TOY-018", "TOY-P01"]
MASK = [0, 4, 7, 7, 8, 13, 2, 6, 10, 11, 14, 5, 9, 12, 1, 3, 11]       # TOY-003 with TOY-003A, TOY-010 with TOY-P01
FIRST_BATCH = [2, 49, 108, 12, 6, 62, 148, 116, 124, 72, 130, 140, 80, 57, 52, 63]
FIRST_LOSS = 16.801116943359375


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    """Small batches on a CPU run faster on one thread than on many."""
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    return support.shared_toy_data(tmp_path_factory)


def sha(tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def variant(toy, folder, identity=None, rename=None):
    """The toy catalog written again, with a rule stated and cards renamed, beside the toy data."""
    document = json.loads(toy.catalog.read_text(encoding="utf-8"))
    for card in document["cards"]:
        card["name"] = (rename or {}).get(card["id"], card["name"])
    if identity is not None:
        document = {"game": document["game"], "locale": document["locale"], "identity": identity,
                    **{key: value for key, value in document.items() if key not in ("game", "locale")}}
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "catalog.json"
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def run(toy, out, monkeypatch, catalog=None) -> dict:
    """One short run through ``loop.train``, with what it did written down: the classes, the
    mask, every batch of the sampler, and each step's inputs and loss."""
    record: dict = {"sampler": [], "steps": [], "mask": None}
    class_list, name_index_of, call, iterate = loop.class_list, loop.name_index_of, loop.Step.__call__, loop.PairBatchSampler.__iter__

    def classes_of(composition):
        record["classes"] = list(class_list(composition))
        return record["classes"]

    def index_of(classes, name_of):
        index = name_index_of(classes, name_of)
        record["mask"] = index.tolist()
        return index

    def step(self, crops, arts, labels):
        loss = call(self, crops, arts, labels)
        record["steps"].append((labels.tolist(), sha(crops), sha(arts), loss))
        return loss

    def batches(self):
        for batch in iterate(self):
            record["sampler"].append(list(batch))
            yield batch

    with monkeypatch.context() as patch:
        patch.setattr(loop, "class_list", classes_of)
        patch.setattr(loop, "name_index_of", index_of)
        patch.setattr(loop.Step, "__call__", step)
        patch.setattr(loop.PairBatchSampler, "__iter__", batches)
        over = {} if catalog is None else {"data": {"catalog": str(catalog)}}
        cfg = parse_config(support.toy_document(toy, epochs=1, validation={"every": 1, "at_start": False}, **over))
        loop.train(cfg, out, device="cpu", log=lambda line: None)
    last = torch.load(out / "last.pt", weights_only=True)
    record["cards"] = last["cards"]
    record["weights"] = {key: sha(value) for key, value in last["model"].items()}
    record["sameNameClasses"] = json.loads((out / "run.json").read_text(encoding="utf-8"))["sameNameClasses"]
    return record


def test_under_the_name_rule_a_run_is_what_it_was(toy, tmp_path, monkeypatch):
    unstated = run(toy, tmp_path / "unstated", monkeypatch)
    # What was recorded before the rule existed.
    assert unstated["classes"] == unstated["cards"] == CLASSES
    assert unstated["mask"] == MASK and unstated["sameNameClasses"] == 2
    assert unstated["sampler"][0] == FIRST_BATCH
    assert unstated["steps"][0][3] == pytest.approx(FIRST_LOSS, rel=1e-5)
    # The mask is the one the names give, which is how it was built before.
    catalog = load_catalog(toy.catalog)
    assert catalog.identity == "name" and catalog.identities() == catalog.names()
    assert unstated["mask"] == loop.name_index_of(CLASSES, catalog.names()).tolist()

    # The rule written out changes nothing, to the last bit of every batch, loss and weight.
    stated = run(toy, tmp_path / "stated", monkeypatch, variant(toy, tmp_path / "stated-game", identity="name"))
    assert stated == unstated
    assert len(stated["steps"]) == 9 and len(stated["weights"]) > 100


def test_the_classes_stay_card_ids_and_the_mask_follows_the_rule(toy, tmp_path, monkeypatch):
    """TOY-007 takes the name of TOY-001: two different cards, one name. By name the mask joins
    them. By printing it does not, and a card stays joined with its alternate printing."""
    rename = {"TOY-007": "Amber Circle"}
    by_name = run(toy, tmp_path / "name", monkeypatch, variant(toy, tmp_path / "name-game", rename=rename))
    by_printing = run(toy, tmp_path / "printing", monkeypatch,
                      variant(toy, tmp_path / "printing-game", identity="printing", rename=rename))
    assert by_name["classes"] == by_printing["classes"] == CLASSES         # one class per card id, under either rule
    assert by_name["sampler"] == by_printing["sampler"]                    # and the same rows in the same batches

    def joined(mask: list[int]) -> list[list[str]]:
        groups: dict[int, list[str]] = {}
        for card_id, index in zip(CLASSES, mask):
            groups.setdefault(index, []).append(card_id)
        return sorted(group for group in groups.values() if len(group) > 1)

    assert joined(by_name["mask"]) == [["TOY-001", "TOY-007"], ["TOY-003", "TOY-003A"], ["TOY-010", "TOY-P01"]]
    assert joined(by_printing["mask"]) == [["TOY-003", "TOY-003A"], ["TOY-010", "TOY-P01"]]
    assert (by_name["sameNameClasses"], by_printing["sameNameClasses"]) == (3, 2)
    # By printing the mask is the one the unchanged toy game has: its names and its printings agree.
    assert joined(by_printing["mask"]) == joined(MASK)
