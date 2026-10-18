# SPDX-License-Identifier: Apache-2.0
"""Operations on exported ONNX files. These need the ``onnx`` package, not PyTorch."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# An initializer with fewer elements than this stays in single precision: the saving would be
# a few bytes, and small tensors are often shape or axis constants that must stay exact.
FP16_MIN_ELEMENTS = 16
_FP16_MAX = float(np.finfo(np.float16).max)


def fp16_weights(source: str | Path, destination: str | Path) -> dict[str, int]:
    """Write a copy of a model whose weights are stored in half precision.

    Every float32 initializer of at least ``FP16_MIN_ELEMENTS`` elements is stored as float16
    and followed by a ``Cast`` back to float32 under the initializer's original name. Inputs,
    outputs and all arithmetic stay float32, so any runtime that runs the original runs the
    copy, and the file is about half the size. The outputs differ slightly from the original's,
    because the weights were rounded; measure the difference before relying on the copy.

    An initializer holding a value too large for float16 is left in float32.

    Returns ``{"converted": n, "kept": m}``, the number of initializers in each group.
    """
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(str(source))
    graph = model.graph
    initializers, casts = [], []
    converted = kept = 0
    for initializer in list(graph.initializer):
        values = numpy_helper.to_array(initializer)
        fits = values.dtype == np.float32 and values.size >= FP16_MIN_ELEMENTS
        if fits:
            finite = values[np.isfinite(values)]
            fits = finite.size == 0 or float(np.abs(finite).max()) <= _FP16_MAX
        if fits:
            half_name = initializer.name + "__fp16"
            initializers.append(numpy_helper.from_array(values.astype(np.float16), half_name))
            casts.append(helper.make_node(
                "Cast", [half_name], [initializer.name], to=onnx.TensorProto.FLOAT, name=initializer.name + "__cast",
            ))
            converted += 1
        else:
            initializers.append(initializer)
            kept += 1
    del graph.initializer[:]
    graph.initializer.extend(initializers)
    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(casts + nodes)
    onnx.checker.check_model(model)
    onnx.save(model, str(destination))
    return {"converted": converted, "kept": kept}


def output_shape(path: str | Path) -> list[int | str]:
    """The declared shape of a model's first output; a dynamic axis is given by its name."""
    import onnx

    dims = onnx.load(str(path)).graph.output[0].type.tensor_type.shape.dim
    return [dim.dim_param if dim.dim_param else dim.dim_value for dim in dims]
