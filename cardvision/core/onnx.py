# SPDX-License-Identifier: Apache-2.0
"""ONNX Runtime helpers: open a model, run tensors through it in batches, embed crops.

Sessions run on the CPU unless the caller names other execution providers. Providers can differ
from one another in the last bits of a result, so the choice is always explicit and never taken
from what happens to be installed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from cardvision.core.preprocess import rot4

if TYPE_CHECKING:  # onnxruntime is imported only when a session is created
    import onnxruntime as ort

CPU_PROVIDER = "CPUExecutionProvider"
DEFAULT_PROVIDERS: tuple[str, ...] = (CPU_PROVIDER,)

Preprocess = Callable[[np.ndarray], np.ndarray]


def create_session(
    model_path: str | Path,
    *,
    providers: Sequence[str] = DEFAULT_PROVIDERS,
    intra_op_threads: int = 2,
    inter_op_threads: int = 1,
) -> ort.InferenceSession:
    """Open an ONNX model.

    ``providers`` is the ordered list of execution providers to use, CPU only by default. A
    provider that this installation of ONNX Runtime does not offer is an error, instead of the
    quiet fallback to another provider that ONNX Runtime would otherwise make.
    """
    import onnxruntime as ort

    requested = list(providers)
    if not requested:
        raise ValueError("at least one execution provider is required")
    missing = [name for name in requested if name not in ort.get_available_providers()]
    if missing:
        raise ValueError(
            f"execution provider(s) {missing} are not available in this installation of "
            f"ONNX Runtime (available: {ort.get_available_providers()})"
        )
    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    options.inter_op_num_threads = inter_op_threads
    return ort.InferenceSession(str(model_path), sess_options=options, providers=requested)


def output_width(session: ort.InferenceSession) -> int:
    """The size of the last axis of the model's first output, for example the embedding length.

    Raises ``ValueError`` when the model declares that axis as dynamic.
    """
    width = session.get_outputs()[0].shape[-1]
    if not isinstance(width, int):
        raise ValueError(f"the model's first output has a dynamic last axis ({width!r})")
    return width


def run_batched(session: ort.InferenceSession, tensors: Sequence[np.ndarray], batch_size: int = 256) -> np.ndarray:
    """Run equally shaped input tensors through the model and return its first output, float32.

    The tensors are stacked into batches of at most ``batch_size`` and fed to the model's first
    input; the per-batch outputs are concatenated in order, so row ``i`` belongs to
    ``tensors[i]``. No tensors gives an empty ``(0, output_width)`` array.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    if len(tensors) == 0:
        return np.zeros((0, output_width(session)), np.float32)
    name = session.get_inputs()[0].name
    outputs = [
        session.run(None, {name: np.ascontiguousarray(np.stack(tensors[i:i + batch_size]))})[0]
        for i in range(0, len(tensors), batch_size)
    ]
    return np.concatenate(outputs).astype(np.float32)


def embed(session: ort.InferenceSession, crops: Sequence[np.ndarray], preprocess: Preprocess) -> np.ndarray:
    """Embed each crop once, upright: ``(N, D)`` float32 for ``N`` RGB crops.

    ``preprocess`` turns one ``(H, W, 3)`` crop into the model's input tensor, for example
    ``cardvision.core.preprocess.full128``.
    """
    return run_batched(session, [preprocess(crop) for crop in crops])


def embed_four_looks(session: ort.InferenceSession, crops: Sequence[np.ndarray], preprocess: Preprocess) -> np.ndarray:
    """Embed each crop in its four orientations: ``(N, 4, D)`` float32 for ``N`` RGB crops.

    Look ``k`` of a crop is its preprocessed tensor turned ``k`` quarter turns (see
    ``cardvision.core.preprocess.rot4``). The crop is preprocessed once and the tensor is turned;
    preprocessing a turned crop would not, in general, give the same values.
    """
    tensors = [look for crop in crops for look in rot4(preprocess(crop))]
    flat = run_batched(session, tensors)
    return flat.reshape(len(crops), 4, flat.shape[-1])
