# SPDX-License-Identifier: Apache-2.0
"""Read a training checkpoint and write its weights in the safetensors format.

A PyTorch checkpoint is a pickle, and unpickling a file can run code. Checkpoints are therefore
read in PyTorch's restricted mode, which admits tensors and plain containers only. A checkpoint
that also holds other objects has to be read with ``trust=True``; do that only for a file you
made yourself. The safetensors copy written by ``save_safetensors`` holds nothing but named
tensors and is safe to load from anyone.

PyTorch and safetensors are imported on first use.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

SAFETENSORS_SUFFIX = ".safetensors"
DEFAULT_KEY = "model"


def load_checkpoint(path: str | Path, trust: bool = False) -> Any:
    """Read a checkpoint file onto the CPU.

    A ``.safetensors`` file gives its dictionary of tensors. Anything else is read with
    ``torch.load`` in restricted mode, or in full pickle mode when ``trust`` is true.
    """
    path = Path(path)
    if path.suffix == SAFETENSORS_SUFFIX:
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    import pickle

    import torch

    if trust:
        return torch.load(path, map_location="cpu", weights_only=False)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        raise ValueError(
            f"{path.name} holds objects other than tensors and plain containers, so it cannot be read in "
            f"restricted mode ({str(error).splitlines()[0]}). If you made this file yourself, read it with "
            "trust enabled (--trust-checkpoint on the command line)."
        ) from None


def state_dict_of(checkpoint: Any, key: str = DEFAULT_KEY) -> dict[str, Any]:
    """The weights inside a checkpoint: ``checkpoint[key]`` when present, otherwise the
    checkpoint itself when it is already a dictionary of tensors."""
    import torch

    if isinstance(checkpoint, Mapping) and isinstance(checkpoint.get(key), Mapping):
        checkpoint = checkpoint[key]
    if isinstance(checkpoint, Mapping) and checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return dict(checkpoint)
    found = sorted(map(str, checkpoint))[:8] if isinstance(checkpoint, Mapping) else type(checkpoint).__name__
    raise ValueError(f"no weights found under the key {key!r}; the checkpoint holds {found}")


def save_safetensors(state_dict: Mapping[str, Any], path: str | Path) -> int:
    """Write the tensors of a state dictionary, and nothing else, to a safetensors file.

    Returns the number of tensors written.
    """
    import torch
    from safetensors.torch import save_file

    tensors = {
        name: value.detach().to("cpu").contiguous().clone()
        for name, value in state_dict.items() if isinstance(value, torch.Tensor)
    }
    if not tensors:
        raise ValueError("the state dictionary holds no tensors")
    save_file(tensors, str(path), metadata={"format": "pt"})
    return len(tensors)


def load_factory(spec: str) -> Callable[..., Any]:
    """Resolve ``"package.module:function"`` to the function it names."""
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"expected MODULE:FUNCTION, for example mypackage.model:build, got {spec!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ValueError(f"cannot import {module_name!r}: {error}") from None
    try:
        return getattr(module, attribute)
    except AttributeError:
        raise ValueError(f"{module_name!r} has no attribute {attribute!r}") from None
