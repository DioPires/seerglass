# SPDX-License-Identifier: Apache-2.0
"""Session creation and the batched run and embedding helpers, on a throwaway model."""

from __future__ import annotations

import numpy as np
import pytest

from cardvision.core import onnx as O
from cardvision.core.preprocess import full128, rot4
from helpers import EMBEDDING_WIDTH


@pytest.fixture(scope="module")
def session(tiny_model):
    return O.create_session(tiny_model)


def crops(seed: int, count: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    sizes = rng.integers(8, 300, size=(count, 2))
    return [rng.integers(0, 256, size=(int(h), int(w), 3), dtype=np.uint8) for h, w in sizes]


def test_sessions_use_the_cpu_unless_told_otherwise(tiny_model):
    session = O.create_session(tiny_model)
    assert session.get_providers() == ["CPUExecutionProvider"]
    assert O.DEFAULT_PROVIDERS == ("CPUExecutionProvider",) == (O.CPU_PROVIDER,)
    assert O.create_session(str(tiny_model), providers=[O.CPU_PROVIDER]).get_providers() == ["CPUExecutionProvider"]


def test_thread_counts_are_passed_to_the_session(tiny_model):
    options = O.create_session(tiny_model, intra_op_threads=1, inter_op_threads=1).get_session_options()
    assert options.intra_op_num_threads == 1 and options.inter_op_num_threads == 1
    assert O.create_session(tiny_model).get_session_options().intra_op_num_threads == 2


def test_an_unavailable_provider_is_an_error_not_a_fallback(tiny_model):
    with pytest.raises(ValueError, match="NoSuchExecutionProvider"):
        O.create_session(tiny_model, providers=["NoSuchExecutionProvider", O.CPU_PROVIDER])
    with pytest.raises(ValueError, match="at least one"):
        O.create_session(tiny_model, providers=[])


def test_output_width(session):
    assert O.output_width(session) == EMBEDDING_WIDTH


def test_output_width_refuses_a_dynamic_axis():
    class Output:
        shape = ["n", "d"]

    class Session:
        def get_outputs(self):
            return [Output()]

    with pytest.raises(ValueError, match="dynamic"):
        O.output_width(Session())


def test_run_batched_returns_one_row_per_tensor_in_order(session):
    tensors = [full128(crop) for crop in crops(0, 9)]
    out = O.run_batched(session, tensors)
    assert out.shape == (9, EMBEDDING_WIDTH) and out.dtype == np.float32
    name = session.get_inputs()[0].name
    direct = session.run(None, {name: np.stack(tensors)})[0]
    assert np.array_equal(out, direct)
    for index, tensor in enumerate(tensors):
        alone = O.run_batched(session, [tensor])
        assert np.allclose(alone[0], out[index], rtol=1e-5, atol=1e-5)


def test_run_batched_splits_into_batches(session):
    tensors = [full128(crop) for crop in crops(1, 10)]
    whole = O.run_batched(session, tensors)
    for batch_size in (1, 3, 10, 64):
        split = O.run_batched(session, tensors, batch_size=batch_size)
        assert split.shape == whole.shape
        assert np.allclose(split, whole, rtol=1e-5, atol=1e-5)
    assert np.array_equal(O.run_batched(session, tensors, batch_size=3), O.run_batched(session, tensors, batch_size=3))
    with pytest.raises(ValueError, match="batch_size"):
        O.run_batched(session, tensors, batch_size=0)


def test_run_batched_with_no_tensors(session):
    out = O.run_batched(session, [])
    assert out.shape == (0, EMBEDDING_WIDTH) and out.dtype == np.float32


def test_embed_is_one_upright_look_per_crop(session):
    batch = crops(2, 6)
    out = O.embed(session, batch, full128)
    assert out.shape == (6, EMBEDDING_WIDTH) and out.dtype == np.float32
    assert np.array_equal(out, O.run_batched(session, [full128(crop) for crop in batch]))
    assert O.embed(session, [], full128).shape == (0, EMBEDDING_WIDTH)


def test_embed_four_looks_shape_and_order(session):
    batch = crops(3, 5)
    looks = O.embed_four_looks(session, batch, full128)
    assert looks.shape == (5, 4, EMBEDDING_WIDTH) and looks.dtype == np.float32
    expected = O.run_batched(session, [look for crop in batch for look in rot4(full128(crop))])
    assert np.array_equal(looks, expected.reshape(5, 4, EMBEDDING_WIDTH))
    upright = O.embed(session, batch, full128)
    assert np.allclose(looks[:, 0], upright, rtol=1e-5, atol=1e-5)
    # The four looks of one crop are different inputs and give different embeddings.
    for k in range(1, 4):
        assert not np.allclose(looks[:, 0], looks[:, k], atol=1e-3)


def test_embed_four_looks_is_deterministic(session):
    batch = crops(4, 4)
    assert np.array_equal(O.embed_four_looks(session, batch, full128), O.embed_four_looks(session, batch, full128))


def test_embed_four_looks_with_no_crops(session):
    assert O.embed_four_looks(session, [], full128).shape == (0, 4, EMBEDDING_WIDTH)


def test_a_turned_card_is_found_by_one_of_the_four_looks(session):
    """Turning the model input by a quarter turn shifts which look matches the upright one."""
    tensor = full128(crops(5, 1)[0])
    upright = O.run_batched(session, list(rot4(tensor)))
    turned = O.run_batched(session, list(rot4(np.ascontiguousarray(np.rot90(tensor, 1, axes=(1, 2))))))
    for k in range(4):
        assert np.allclose(turned[k], upright[(k + 1) % 4], rtol=1e-5, atol=1e-5)
