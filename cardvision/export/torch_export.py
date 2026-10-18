# SPDX-License-Identifier: Apache-2.0
"""Shared steps of the two exports: writing the ONNX file, running it, describing the result.

The ONNX file is written with PyTorch's TorchScript-based exporter. The published models were
exported with it, and the graph it produces is what the parity tolerances in this package were
set against. Where the installed PyTorch offers a choice of exporter, that one is requested
explicitly.

A note on reproducibility across machines: the same checkpoint exported on two CPU
architectures can give ONNX files whose outputs differ in the last bits, and the same ONNX
file can give slightly different outputs under different execution providers. The differences
are far below the tolerances here, but a score that sits exactly on a threshold can fall on
either side. The export report therefore records the machine it was made on.
"""

from __future__ import annotations

import hashlib
import inspect
import platform
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from cardvision.core.onnx import create_session


class ParityError(RuntimeError):
    """The exported model's outputs differ from PyTorch's by more than the stated tolerance."""


def sha256_of(path: str | Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_record(path: str | Path) -> dict[str, Any]:
    """Name, size and SHA-256 of a written file, for the export report."""
    path = Path(path)
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_of(path)}


def claim_outputs(paths: list[Path], overwrite: bool) -> None:
    """Create the output folder and refuse to replace existing files unless ``overwrite``."""
    existing = [path.name for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite {', '.join(existing)} (pass overwrite to replace them)")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def export_onnx(
    model: Any,
    example: Any,
    path: str | Path,
    *,
    input_name: str,
    output_name: str,
    opset: int,
    dynamic_batch: bool,
) -> None:
    """Write ``model`` as an ONNX file with one named input and one named output.

    ``example`` is a tensor of the input's shape. With ``dynamic_batch`` the first axis of both
    the input and the output is left free; otherwise the example's shape is fixed in the graph.
    The file is validated with the ONNX checker before returning.
    """
    import onnx
    import torch

    options: dict[str, Any] = {
        "input_names": [input_name],
        "output_names": [output_name],
        "opset_version": opset,
        "do_constant_folding": True,
    }
    if dynamic_batch:
        options["dynamic_axes"] = {input_name: {0: "batch"}, output_name: {0: "batch"}}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        options["dynamo"] = False
    with warnings.catch_warnings():
        # The TorchScript-based exporter announces its own deprecation on every call.
        warnings.simplefilter("ignore", DeprecationWarning)
        warnings.simplefilter("ignore", FutureWarning)
        torch.onnx.export(model, example, str(path), **options)
    onnx.checker.check_model(onnx.load(str(path)))


def run_onnx(path: str | Path, batches: list[np.ndarray]) -> list[np.ndarray]:
    """Run each input array through an ONNX file on the CPU and return the first outputs."""
    session = create_session(path)
    name = session.get_inputs()[0].name
    return [session.run(None, {name: batch})[0] for batch in batches]


def environment() -> dict[str, str]:
    """The versions and the machine an export was made with, for the export report."""
    import onnx
    import onnxruntime
    import torch

    return {
        "machine": platform.machine(),
        "system": platform.system(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": onnxruntime.__version__,
    }
