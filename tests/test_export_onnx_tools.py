# SPDX-License-Identifier: Apache-2.0
"""The half-precision-weights copy of an ONNX file. Needs ``onnx`` only, not PyTorch."""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from cardvision.core.onnx import create_session
from cardvision.export.onnx_tools import FP16_MIN_ELEMENTS, fp16_weights, output_shape
from helpers import EMBEDDING_WIDTH


def test_weights_are_stored_as_half_floats_behind_a_cast(tiny_model, tmp_path):
    copy = tmp_path / "tiny-fp16w.onnx"
    assert fp16_weights(tiny_model, copy) == {"converted": 1, "kept": 0}

    graph = onnx.load(str(copy)).graph
    (initializer,) = graph.initializer
    assert initializer.name == "weights__fp16" and initializer.data_type == TensorProto.FLOAT16
    assert tuple(initializer.dims) == (192, EMBEDDING_WIDTH)
    first = graph.node[0]
    assert (first.op_type, list(first.input), list(first.output), first.name) == ("Cast", ["weights__fp16"], ["weights"], "weights__cast")
    assert first.attribute[0].i == TensorProto.FLOAT
    assert [node.op_type for node in graph.node] == ["Cast", "AveragePool", "Flatten", "MatMul"]
    assert copy.stat().st_size < 0.6 * tiny_model.stat().st_size


def test_inputs_and_outputs_stay_float32_and_values_stay_close(tiny_model, tmp_path):
    copy = tmp_path / "tiny-fp16w.onnx"
    fp16_weights(tiny_model, copy)
    original, half = create_session(tiny_model), create_session(copy)
    assert half.get_inputs()[0].type == "tensor(float)" and half.get_outputs()[0].type == "tensor(float)"
    x = np.random.default_rng(1).uniform(-1, 1, (4, 3, 128, 128)).astype(np.float32)
    a, b = original.run(None, {"x": x})[0], half.run(None, {"x": x})[0]
    assert a.dtype == b.dtype == np.float32
    assert not np.array_equal(a, b)                                    # the weights were rounded
    cosine = (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    assert cosine.min() > 0.99999


def test_the_stored_values_are_the_originals_rounded_to_half_precision(tiny_model, tmp_path):
    copy = tmp_path / "tiny-fp16w.onnx"
    fp16_weights(tiny_model, copy)
    original = numpy_helper.to_array(onnx.load(str(tiny_model)).graph.initializer[0])
    stored = numpy_helper.to_array(onnx.load(str(copy)).graph.initializer[0])
    assert stored.dtype == np.float16
    np.testing.assert_array_equal(stored, original.astype(np.float16))


def small_model(path, initializers):
    """y = x + a + b + ..., for float32 ``initializers`` that broadcast against x of shape (1, 16)."""
    nodes, last = [], "x"
    for index, name in enumerate(initializers):
        nodes.append(helper.make_node("Add", [last, name], [f"s{index}"]))
        last = f"s{index}"
    nodes.append(helper.make_node("Identity", [last], ["y"]))
    graph = helper.make_graph(
        nodes, "small",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 16])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 16])],
        [numpy_helper.from_array(values, name) for name, values in initializers.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def test_small_and_oversized_initializers_are_left_alone(tmp_path):
    assert FP16_MIN_ELEMENTS == 16
    source = small_model(tmp_path / "small.onnx", {
        "large": np.linspace(-1, 1, 16, dtype=np.float32).reshape(1, 16),    # 16 elements: converted
        "short": np.full((1, 1), 0.3, dtype=np.float32),                      # 1 element: kept
        "huge": np.full((1, 16), 1.0e5, dtype=np.float32),                    # does not fit float16: kept
    })
    copy = tmp_path / "small-fp16w.onnx"
    assert fp16_weights(source, copy) == {"converted": 1, "kept": 2}
    kinds = {i.name: i.data_type for i in onnx.load(str(copy)).graph.initializer}
    assert kinds == {"large__fp16": TensorProto.FLOAT16, "short": TensorProto.FLOAT, "huge": TensorProto.FLOAT}
    x = np.zeros((1, 16), dtype=np.float32)
    a = create_session(source).run(None, {"x": x})[0]
    b = create_session(copy).run(None, {"x": x})[0]
    np.testing.assert_allclose(a, b, rtol=1e-6)
    assert np.isfinite(b).all()                                               # 1e5 did not become infinity


def test_infinite_values_survive_the_conversion(tmp_path):
    values = np.array([[np.inf, -np.inf] + [0.25] * 14], dtype=np.float32)
    source = small_model(tmp_path / "inf.onnx", {"limits": values})
    copy = tmp_path / "inf-fp16w.onnx"
    assert fp16_weights(source, copy) == {"converted": 1, "kept": 0}
    stored = numpy_helper.to_array(onnx.load(str(copy)).graph.initializer[0])
    np.testing.assert_array_equal(stored, values.astype(np.float16))


def test_output_shape_names_dynamic_axes(tiny_model):
    assert output_shape(tiny_model) == ["n", EMBEDDING_WIDTH]
