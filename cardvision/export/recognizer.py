# SPDX-License-Identifier: Apache-2.0
"""Export a recognizer: PyTorch module to ONNX, a half-precision-weights copy, and safetensors.

The exported graph takes ``crops``, a ``(batch, 3, S, S)`` float32 tensor prepared by the
model's preprocessing contract, and returns ``embeddings``, ``(batch, D)`` float32. The batch
axis is free. Opset 17.

Parity is checked on random inputs: each embedding the ONNX file produces must point the same
way as PyTorch's for the same input, to a cosine of at least ``MIN_COSINE``. The copy with
half-precision weights is held to ``MIN_COSINE_FP16``, which is looser because its weights
were rounded. An export that misses either bound raises ``ParityError`` and should not be used.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from cardvision.core.preprocess import FULL128, input_size
from cardvision.export.checkpoint import save_safetensors
from cardvision.export.onnx_tools import fp16_weights, output_shape
from cardvision.export.torch_export import ParityError, claim_outputs, environment, export_onnx, file_record, run_onnx

INPUT_NAME = "crops"
OUTPUT_NAME = "embeddings"
OPSET = 17
MIN_COSINE = 0.9999
MIN_COSINE_FP16 = 0.999
PARITY_SAMPLES = 16
PARITY_SEED = 0


def row_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between corresponding rows of two ``(N, D)`` arrays, in float64."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def export_recognizer(
    model: Any,
    out_dir: str | Path,
    name: str = "recognizer",
    *,
    size: int | None = None,
    opset: int = OPSET,
    overwrite: bool = False,
    min_cosine: float = MIN_COSINE,
    min_cosine_fp16: float = MIN_COSINE_FP16,
) -> dict[str, Any]:
    """Export ``model`` (a ``torch.nn.Module`` with its weights loaded) into ``out_dir``.

    Writes ``<name>.onnx`` (float32), ``<name>-fp16w.onnx`` (the same graph with weights stored
    in half precision), ``<name>.safetensors`` (the weights alone) and ``<name>.export.json``
    (the report, which is also returned). ``size`` is the side of the square input; it defaults
    to that of the ``full128`` contract.

    Existing files are not replaced unless ``overwrite`` is true. Raises ``ParityError`` when
    the ONNX outputs miss the cosine bounds; the files are left in place for inspection, and
    the report is not written.
    """
    import torch

    size = input_size(FULL128) if size is None else size
    out_dir = Path(out_dir)
    fp32 = out_dir / f"{name}.onnx"
    fp16 = out_dir / f"{name}-fp16w.onnx"
    weights = out_dir / f"{name}.safetensors"
    report_path = out_dir / f"{name}.export.json"
    claim_outputs([fp32, fp16, weights, report_path], overwrite)

    model = model.to("cpu").eval()
    samples = np.random.default_rng(PARITY_SEED).standard_normal((PARITY_SAMPLES, 3, size, size)).astype(np.float32)
    with torch.no_grad():
        reference = model(torch.from_numpy(samples)).numpy()
    export_onnx(model, torch.zeros(1, 3, size, size), fp32,
                input_name=INPUT_NAME, output_name=OUTPUT_NAME, opset=opset, dynamic_batch=True)
    cosine = float(row_cosines(reference, run_onnx(fp32, [samples])[0]).min())
    if not cosine >= min_cosine:
        raise ParityError(f"{fp32.name}: lowest cosine against PyTorch is {cosine:.6f}, below {min_cosine}")
    converted = fp16_weights(fp32, fp16)
    cosine_fp16 = float(row_cosines(reference, run_onnx(fp16, [samples])[0]).min())
    if not cosine_fp16 >= min_cosine_fp16:
        raise ParityError(f"{fp16.name}: lowest cosine against PyTorch is {cosine_fp16:.6f}, below {min_cosine_fp16}")
    tensors = save_safetensors(model.state_dict(), weights)

    report = {
        "kind": "recognizer",
        "input": {"name": INPUT_NAME, "shape": ["batch", 3, size, size], "dtype": "float32"},
        "output": {"name": OUTPUT_NAME, "shape": output_shape(fp32), "dtype": "float32"},
        "opset": opset,
        "fp32": file_record(fp32),
        "fp16Weights": {**file_record(fp16), **converted},
        "safetensors": {**file_record(weights), "tensors": tensors},
        "parity": {
            "inputs": f"{PARITY_SAMPLES} standard-normal tensors, seed {PARITY_SEED}",
            "measure": "lowest cosine between the PyTorch and ONNX Runtime embedding of one input",
            "fp32": {"minCosine": cosine, "required": min_cosine},
            "fp16Weights": {"minCosine": cosine_fp16, "required": min_cosine_fp16},
        },
        "environment": environment(),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
