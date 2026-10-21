# SPDX-License-Identifier: Apache-2.0
"""The bank-scoring graph (``cardvision.mtg.scorer``) run by ONNX Runtime against the plain scan
it replaces, on a small made-up bank."""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
import pytest

from cardvision.mtg import scorer as S

pytest.importorskip("onnx")


def _world(n=300, dim=32, looks=12, seed=5):
    rng = np.random.default_rng(seed)
    bank = rng.standard_normal((n, dim)).astype(np.float32)
    bank = (bank / np.linalg.norm(bank, axis=1, keepdims=True)).astype(np.float16).astype(np.float32)
    bank /= np.linalg.norm(bank, axis=1, keepdims=True)
    ident = S.identity_indices([f"k{i // 3}" for i in range(n)])
    queries = bank[rng.integers(0, n, looks)] + 0.05 * rng.standard_normal((looks, dim)).astype(np.float32)
    return bank, ident, (queries / np.linalg.norm(queries, axis=1, keepdims=True)).astype(np.float32)


def _run(model, **feeds):
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    names = [output.name for output in session.get_outputs()]
    return dict(zip(names, session.run(None, feeds)))


@pytest.mark.parametrize("options", [{"top_k": 5}, {"with_topk": False}])
def test_the_graph_gives_the_scan_s_best_row_score_and_other(options):
    bank, ident, queries = _world()
    out = _run(S.scorer_model(bank, ident, **options), queries=queries)
    best_row, best, other = S.reference_scores(bank, ident, queries)
    np.testing.assert_array_equal(out["topRows"][:, 0], best_row)
    np.testing.assert_allclose(out["topScores"][:, 0], best, atol=1e-5)
    np.testing.assert_allclose(out["other"], other, atol=1e-5)
    if options.get("top_k"):
        assert out["topRows"].shape == (len(queries), 5)
        assert np.all(np.diff(out["topScores"], axis=1) <= 0)


def test_the_bank_can_be_an_input():
    bank, ident, queries = _world()
    out = _run(S.scorer_model(bank, ident, top_k=3, bank_as_input=True), queries=queries,
               bank=np.ascontiguousarray(bank.T))
    best_row, _, other = S.reference_scores(bank, ident, queries)
    np.testing.assert_array_equal(out["topRows"][:, 0], best_row)
    np.testing.assert_allclose(out["other"], other, atol=1e-5)


def test_a_bank_of_one_identity_has_no_other():
    bank, _, queries = _world(n=6)
    out = _run(S.scorer_model(bank, np.zeros(6, np.int32), top_k=2), queries=queries)
    assert np.all(out["other"] == S.NO_OTHER_IDENTITY)


def test_identity_indices_and_their_limit():
    assert S.identity_indices(["b", "a", "b", "c"]).tolist() == [0, 1, 0, 2]
    with pytest.raises(ValueError):
        S.scorer_model(np.eye(2, dtype=np.float32), np.array([0, 2**24]))
