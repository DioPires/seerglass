# SPDX-License-Identifier: Apache-2.0
"""The recognizer's network: an image encoder that returns unit-length embeddings.

``Encoder`` is torchvision's MobileNetV3-Small without its classifier: the convolutional
features, a global average pool, a linear projection from 576 channels to the embedding width,
batch normalization, and a scaling to unit length. Input ``(n, 3, S, S)`` float32 prepared by
the preprocessing contract (``cardvision.core.preprocess``), output ``(n, dim)``.

``init`` chooses the starting weights of the features: ``"imagenet"`` loads torchvision's
ImageNet-1k weights for MobileNetV3-Small (downloaded on first use; see their license in
torchvision's documentation), ``"random"`` leaves torchvision's random initialization. The
projection and the batch normalization always start from PyTorch's defaults.

The parameter names are ``features.*``, ``proj.*`` and ``bn.*``; a checkpoint's weights are the
encoder's ``state_dict()`` under the key ``"model"``.

**Starting from a checkpoint.** A run can also start from an encoder that was trained before
(``read_start``): a ``best.pt`` or ``last.pt`` of ``cardvision train-recognizer``, or the
``.safetensors`` file of ``cardvision export recognizer``. The whole encoder is replaced:
features, projection and batch normalization, with its running statistics. Nothing else is
read from the file: no class list, no scores, no config, and a checkpoint holds no optimizer
state. The training head, which has one weight vector per card, is not part of a checkpoint
and always starts fresh.

The load is strict. Every tensor the encoder has must be in the file with the same shape, and
the file must hold no other tensor. The first tensor that does not fit is named, and nothing
is loaded in part.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

EMBEDDING_DIM = 128
FEATURE_CHANNELS = 576  # what MobileNetV3-Small's features end with
INITS = ("imagenet", "random")


class Encoder(nn.Module):
    """MobileNetV3-Small features, pooled, projected to ``dim`` values and scaled to unit length."""

    def __init__(self, init: str = "random", dim: int = EMBEDDING_DIM) -> None:
        super().__init__()
        if init not in INITS:
            raise ValueError(f"init must be one of {', '.join(INITS)}, not {init!r}")
        try:
            from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        except ImportError:
            raise RuntimeError(
                'the recognizer\'s network needs torchvision, part of an optional dependency group: '
                'python -m pip install -e ".[train]" in the repository folder'
            ) from None
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if init == "imagenet" else None
        self.features = mobilenet_v3_small(weights=weights).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(FEATURE_CHANNELS, dim)
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pool(self.features(x)).flatten(1)
        return F.normalize(self.bn(self.proj(z)), dim=1)


def build_encoder(init: str = "random", dim: int = EMBEDDING_DIM) -> Encoder:
    """A new encoder. See the module docstring for ``init``."""
    return Encoder(init=init, dim=dim)


class StartError(ValueError):
    """The checkpoint to start from is missing, holds no weights or does not fit the encoder."""


def check_fit(state: Mapping[str, Any], reference: Mapping[str, Any], name: str) -> None:
    """Raise ``StartError`` at the first tensor of ``state`` that does not fit ``reference``.

    Both map tensor names to tensors; ``reference`` is the encoder's own ``state_dict()``, and
    its order is the order of the search. A tensor the encoder has and the file lacks comes
    first, then one of another shape, then one the file has and the encoder lacks. ``name`` is
    the file's name, for the message.
    """
    for key, own in reference.items():
        if key not in state:
            missing = sum(other not in state for other in reference)
            raise StartError(f"{name} does not fit the encoder: it has no tensor {key} ({missing} of the encoder's "
                             f"{len(reference)} tensors are missing). Is it a recognizer checkpoint?")
        if tuple(state[key].shape) != tuple(own.shape):
            raise StartError(f"{name} does not fit the encoder: {key} has shape {tuple(state[key].shape)} in the file "
                             f"and {tuple(own.shape)} in the encoder. The embedding width, [model] dim, and the "
                             "network must be the ones the checkpoint was trained with")
    extra = [key for key in state if key not in reference]
    if extra:
        raise StartError(f"{name} does not fit the encoder: it has {len(extra)} tensor(s) the encoder lacks, the "
                         f"first being {extra[0]}")


def read_start(path: str | Path, dim: int = EMBEDDING_DIM) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read the checkpoint a run starts from and check that it fits an encoder of width ``dim``.

    Returns the weights and the record of where they came from: ``{"init": the file's name,
    "sha256": the hash of its bytes, "copiedTensors": how many tensors it gives}``. The file
    is read in PyTorch's restricted mode, or as safetensors by its suffix
    (``cardvision.export.checkpoint``). The check builds an encoder to compare shapes with and
    leaves PyTorch's random state as it was. Raises ``StartError``.
    """
    from cardvision.export.checkpoint import load_checkpoint, state_dict_of

    path = Path(path)
    if not path.is_file():
        raise StartError(f"{path}: no such checkpoint file ([model] init)")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        state = state_dict_of(load_checkpoint(path))
    except Exception as error:  # an unreadable file, or one that is not a checkpoint
        raise StartError(f"{path.name} cannot be read as a recognizer checkpoint: {str(error).splitlines()[0]}") from None
    with torch.random.fork_rng():
        reference = Encoder(init="random", dim=dim).state_dict()
    check_fit(state, reference, path.name)
    return state, {"init": path.name, "sha256": digest, "copiedTensors": len(state)}


class _ExportEncoder(Encoder):
    """The encoder as ``cardvision export recognizer`` builds it when no network is named.
    Weights that do not fit it are refused with a message that says how to name another."""

    def load_state_dict(self, state_dict, *args, **kwargs):
        try:
            return super().load_state_dict(state_dict, *args, **kwargs)
        except RuntimeError as error:
            first = str(error).splitlines()[0]
            raise RuntimeError(
                "the checkpoint's weights do not fit the default recognizer network (MobileNetV3-Small, "
                f"{EMBEDDING_DIM} values): {first} If the checkpoint was trained with another network, "
                "pass --model MODULE:FUNCTION, a function that takes no arguments and returns that network "
                "(its state_dict must match the checkpoint's)."
            ) from None


def build_export_encoder() -> Encoder:
    """The network a recognizer checkpoint of ``cardvision train-recognizer`` holds, with
    random weights for the checkpoint's to replace. Nothing is downloaded."""
    return _ExportEncoder(init="random")
