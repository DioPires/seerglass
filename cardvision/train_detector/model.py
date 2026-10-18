# SPDX-License-Identifier: Apache-2.0
"""The detector's network, its optimizer and its starting weights.

The network is YOLOX-Nano with two classes, face and back, and it is built by the YOLOX
package (Megvii, Apache-2.0) through that package's own ``yolox-nano`` experiment:
``Exp.get_model`` makes the network and sets up its normalization layers and prediction
biases, and ``Exp.get_optimizer`` makes the optimizer and decides which weights are decayed.
Both are called, not copied. The one thing changed afterwards is the head's class, so that its
loss knows about ignore regions (``cardvision.train_detector.loss``); that class adds no
weights, and a checkpoint trained here loads into the plain YOLOX-Nano network that
``cardvision export detector`` builds.

**Starting weights.** Training starts from a checkpoint (``load_init``):

* a tensor is copied when the checkpoint has one of the same name and shape;
* the class-prediction layers are the exception, because their shape depends on the number of
  classes. From a one-class checkpoint, the single class becomes the face class and the back
  class keeps its fresh initialization (*widening*). From a checkpoint with any other number
  of classes, such as the 80 of the public COCO weights, both classes keep their fresh
  initialization. Fresh means the library's: a bias set for a prior probability of 0.01;
* anything else that does not fit is an error.

The public COCO-pretrained YOLOX-Nano weights are the default start. They are not downloaded
unless asked for (``fetch_pretrained``), and are checked against a pinned SHA-256 before use.

A checkpoint named by the config's ``init`` or by ``--init`` is read by the same rules. A
``last.pth`` that this trainer wrote, or the ``.safetensors`` file of ``cardvision export
detector``, has this network's two classes, so every tensor is copied and none is left as
built. Only weights are read: ``last.pth`` also holds the trained weights before averaging, a
summary and a config, and none of them is used. The record that ``initialize`` returns goes
into ``run.json`` and into the run's own ``last.pth``, and ``describe_start`` words it for the
log: the file's name, its SHA-256, and what was done with the class layers.
"""

from __future__ import annotations

import os
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cardvision.core.tls import CERTIFICATE_HELP, is_certificate_failure, tls_context
from cardvision.detect.decode import CLASS_NAMES
from cardvision.export.detector import YOLOX_INSTALL, YOLOX_REVISION, YoloxMissing
from cardvision.train_detector.sources import file_sha256

NUM_CLASSES = len(CLASS_NAMES)
CLASS_LAYER_PREFIX = "head.cls_preds."

# The YOLOX-Nano weights that the YOLOX authors trained on COCO and published with their
# 0.1.1rc0 release, under the project's Apache-2.0 license.
PRETRAINED_NAME = "yolox_nano.pth"
PRETRAINED_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth"
PRETRAINED_SHA256 = "cd28f55fbbc1829f99d9ac9b38a16d259a22889739c8728ea877610201feff7b"
CACHE_ENV = "CARDVISION_CACHE"
RANDOM_INIT = "random"


class InitError(ValueError):
    """The starting checkpoint is missing, fails its hash or does not fit the network."""


def nano_experiment(num_classes: int = NUM_CLASSES) -> Any:
    """The YOLOX package's ``yolox-nano`` experiment, with the class count set."""
    try:
        from yolox.exp import get_exp
    except ImportError as error:
        raise YoloxMissing(
            f"the detector is built and trained with the YOLOX package, which could not be imported ({error}). "
            f"Install the train extra first, then: {YOLOX_INSTALL}"
        ) from None
    experiment = get_exp(None, "yolox-nano")
    experiment.num_classes = num_classes
    return experiment


def build_model(experiment: Any) -> Any:
    """The experiment's network, untrained, with the ignore-region loss in its head.

    The library builds the network; the head object is then given the subclass that replaces
    the loss. The subclass has no state of its own, so nothing else about the head changes.
    """
    from cardvision.train_detector.loss import IgnoreRegionHead

    model = experiment.get_model()
    model.head.__class__ = IgnoreRegionHead
    return model


def build_optimizer(experiment: Any, batch_size: int, lr: float, momentum: float, weight_decay: float) -> Any:
    """The library's optimizer for the experiment's network: SGD with Nesterov momentum in three
    groups, normalization weights, other weights (the only group with weight decay) and biases.

    ``lr`` is where the learning rate starts; the trainer sets it again before every step.
    """
    experiment.momentum = momentum
    experiment.weight_decay = weight_decay
    experiment.warmup_epochs = 1  # with warm-up epochs the library starts from warmup_lr
    experiment.warmup_lr = lr
    experiment.__dict__.pop("optimizer", None)
    return experiment.get_optimizer(batch_size)


def learning_rate(step: int, peak: float, warmup_steps: int, no_augment_from: int, final_ratio: float) -> float:
    """The learning rate of a step (counted from 0).

    A linear rise over ``warmup_steps`` steps to ``peak``; a half cosine down to
    ``final_ratio * peak``, reached at step ``no_augment_from``; then held there. YOLOX's own
    warm-up-cosine schedule has this shape too, with a quadratic warm-up and counted in epochs.
    """
    import math

    if step < warmup_steps:
        return peak * (step + 1) / warmup_steps
    if step >= no_augment_from:
        return peak * final_ratio
    progress = (step - warmup_steps) / max(1, no_augment_from - warmup_steps)
    return peak * (final_ratio + (1 - final_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


# ------------------------------------------------------------------------- starting weights
def pretrained_path() -> Path:
    """Where the public pretrained weights are kept: ``$CARDVISION_CACHE`` if set, otherwise
    ``~/.cache/cardvision``."""
    folder = os.environ.get(CACHE_ENV, "").strip()
    return (Path(folder) if folder else Path.home() / ".cache" / "cardvision") / PRETRAINED_NAME


def fetch_pretrained(download: bool = False, path: Path | None = None, url: str = PRETRAINED_URL,
                     sha256: str = PRETRAINED_SHA256) -> Path:
    """The public COCO-pretrained YOLOX-Nano checkpoint, verified.

    If the file is not at ``path`` it is downloaded from the YOLOX project's release page, but
    only when ``download`` is true. Either way its SHA-256 must equal the pinned one; a
    download that fails the check is deleted.
    """
    path = path if path is not None else pretrained_path()
    if not path.is_file():
        if not download:
            raise InitError(
                f"the default starting weights are the public COCO-pretrained YOLOX-Nano checkpoint, which is not "
                f"at {path}. Pass --download-pretrained to fetch it from {url}, or download it yourself and pass "
                f"it with --init, or pass --init {RANDOM_INIT} to start from untrained weights."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        try:
            with (urllib.request.urlopen(url, timeout=60, context=tls_context()) as response,  # noqa: S310
                  open(partial, "wb") as handle):
                for block in iter(lambda: response.read(1 << 20), b""):
                    handle.write(block)
        except OSError as error:
            partial.unlink(missing_ok=True)
            if is_certificate_failure(error):
                raise InitError(f"could not download {url}: {CERTIFICATE_HELP}") from None
            raise InitError(f"could not download {url}: {error}") from None
        got = file_sha256(partial)
        if got != sha256:
            partial.unlink()
            raise InitError(f"the file downloaded from {url} has SHA-256 {got}, not the expected {sha256}; "
                            "it was deleted")
        partial.replace(path)
    got = file_sha256(path)
    if got != sha256:
        raise InitError(f"{path} has SHA-256 {got}, not the {sha256} of the public YOLOX-Nano checkpoint. "
                        "Remove it and download again, or pass the checkpoint you mean with --init.")
    return path


def load_init(model: Any, state: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a checkpoint's weights into the model by the rules in the module docstring.

    ``state`` maps tensor names to tensors and must hold exactly the model's names. Returns
    what was done: how many tensors were copied whole, which class-layer tensors were widened
    from one class, and which were left at their fresh initialization.
    """
    own = model.state_dict()
    if set(state) != set(own):
        unexpected, absent = sorted(set(state) - set(own)), sorted(set(own) - set(state))
        raise InitError(
            f"the checkpoint is not a YOLOX-Nano detector: {len(unexpected)} tensor(s) the network lacks"
            f"{' (' + unexpected[0] + ', ...)' if unexpected else ''} and {len(absent)} it needs"
            f"{' (' + absent[0] + ', ...)' if absent else ''}"
        )
    copied, widened, fresh = 0, [], []
    for name, value in state.items():
        if own[name].shape == value.shape:
            own[name] = value.clone()
            copied += 1
        elif name.startswith(CLASS_LAYER_PREFIX) and value.shape[1:] == own[name].shape[1:]:
            if value.shape[0] == 1:
                own[name][0] = value[0]
                widened.append(name)
            else:
                fresh.append(name)
        else:
            raise InitError(f"the checkpoint's {name} has shape {tuple(value.shape)}; the network's has "
                            f"{tuple(own[name].shape)}")
    model.load_state_dict(own)
    return {"copiedTensors": copied, "widenedFromOneClass": widened, "freshClassTensors": fresh}


def initialize(model: Any, init: str | Path | None, download: bool = False, trust: bool = False) -> dict[str, Any]:
    """Give the model its starting weights and return a record of where they came from.

    ``init`` is a checkpoint path, ``"random"`` for the untrained network as built, or ``None``
    for the public pretrained checkpoint (``fetch_pretrained``). A checkpoint keeps its
    weights under the key ``model``, or is itself a dictionary of tensors.
    """
    from cardvision.export.checkpoint import load_checkpoint, state_dict_of

    if init is not None and str(init) == RANDOM_INIT:
        return {"init": RANDOM_INIT, "sha256": None, "publicPretrained": False}
    path = fetch_pretrained(download) if init is None else Path(init)
    if not path.is_file():
        raise InitError(f"{path}: no such checkpoint file")
    digest = file_sha256(path)
    record = load_init(model, state_dict_of(load_checkpoint(path, trust=trust)))
    return {"init": path.name, "sha256": digest, "publicPretrained": digest == PRETRAINED_SHA256, **record}


def describe_start(record: Mapping[str, Any]) -> str:
    """One line for the log about a run that started from a checkpoint file: which file, its
    SHA-256, how many tensors were copied, and what became of the class layers when they were
    not copied. ``record`` is what ``initialize`` returned for a file."""
    widened, fresh = record["widenedFromOneClass"], record["freshClassTensors"]
    total = record["copiedTensors"] + len(widened) + len(fresh)
    line = f"started from {record['init']} (SHA-256 {record['sha256']}): "
    if not widened and not fresh:
        return line + f"all {total} tensors copied, and nothing else from the file"
    line += f"{record['copiedTensors']} of {total} tensors copied"
    if widened:
        line += (f"; the {len(widened)} class-layer tensors widened from the checkpoint's one class: its class "
                 "became the face class, and the back class is as freshly built")
    if fresh:
        line += (f"; the {len(fresh)} class-layer tensors left as freshly built, because the checkpoint has another "
                 "number of classes")
    return line


def versions() -> dict[str, Any]:
    """The versions of the packages that decide a run's numbers."""
    import platform
    from importlib import metadata

    import cardvision

    out: dict[str, Any] = {"python": platform.python_version(), "cardvision": cardvision.__version__}
    for package in ("torch", "torchvision", "numpy", "opencv-python-headless", "opencv-python", "Pillow", "yolox"):
        try:
            out[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    out["yoloxRevisionVerified"] = YOLOX_REVISION
    return out
