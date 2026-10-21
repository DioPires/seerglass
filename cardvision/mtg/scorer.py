# SPDX-License-Identifier: Apache-2.0
"""Bank scoring as one ONNX graph, for a GPU.

The browser runtime scores a look against the bank in a JavaScript loop
(``browser/src/recognizer.ts``, ``topAndOther``): every row's dot product, the best row, then
the best row of a different identity. At Magic's scale that loop is the bottleneck: about
13 million multiply-adds per look at 100k rows, and a busy frame has dozens of looks. The same
work is a matrix product and two reductions, which a GPU does in one pass over the bank for all
the looks of a frame at once. This module writes that graph.

The graph
---------
Inputs: ``queries`` ``[L, D]`` float32, the unit embeddings of the L looks of a frame.

Constants (initializers): ``bank16`` ``[D, N]``, the bank's rows transposed, stored as fp16 (the
bank's own storage precision, so the file is half the size of float32), widened to float32 and
scaled to unit length inside the graph, as the bank readers do; ``rowIdentF`` ``[1, N]`` float32, a whole number per row that is equal exactly for
rows of one identity (float, so that every node runs on the GPU: WebGPU runtimes keep integer
index tensors on the CPU).

Outputs, per look:

* ``topScores`` ``[L, K]`` and ``topRows`` ``[L, K]`` (int64): the K best rows, best first.
  ``topRows[:, 0]`` is ``bestRow`` and ``topScores[:, 0]`` the score. The other K - 1 rows are
  where the printing stage looks for the best printings of the named card;
* ``other`` ``[L]``: the best score of a row whose identity differs from the best row's, or -2
  when every row is of that identity (``NO_OTHER_IDENTITY`` in the browser). This is exact: it
  is taken over every row (a mask built from ``rowIdent``), not over the top K, so a card with
  hundreds of printings (a basic land) cannot hide its nearest rival.

The margin is ``topScores[:, 0] - other``, as in ``topAndOther``. Two differences from the
JavaScript loop: the products are summed in float32, not float64 (differences of about 1e-7 on
unit vectors), and on an exact tie of two rows the first row is not guaranteed to win.

``with_topk=False`` writes the variant without TopK (ArgMax and ReduceMax in its place, K = 1),
for an execution provider without a TopK kernel.

``bank-as-initializer`` is also a file format: the graph holds the bank as fp16, so a release
can ship a binary bank that a GPU session loads without any JSON decoding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

OPSET = 17
NO_OTHER_IDENTITY = -2.0


def identity_indices(identity_keys: list[str]) -> np.ndarray:
    """One int32 per row, equal exactly for rows with equal identity keys (first seen, first)."""
    index: dict[str, int] = {}
    return np.array([index.setdefault(key, len(index)) for key in identity_keys], dtype=np.int32)


def scorer_model(bank: np.ndarray, row_ident: np.ndarray, top_k: int = 8, with_topk: bool = True,
                 bank_as_input: bool = False):
    """The scoring graph for ``bank`` ``(N, D)`` unit rows and ``row_ident`` ``(N,)`` ints.

    With ``bank_as_input`` the bank is not stored in the graph: it is an input ``bank`` ``[D, N]``
    float32, for a runtime that keeps it in GPU memory and passes it with every call.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    bank = np.asarray(bank, np.float32)
    n, dim = bank.shape
    if len(row_ident) != n or np.asarray(row_ident).max(initial=0) >= 2**24:
        raise ValueError("row_ident needs one whole number below 2**24 per bank row")
    top_k = min(top_k, n)
    nodes = []
    inputs = [helper.make_tensor_value_info("queries", TensorProto.FLOAT, ["L", dim])]
    initializers = [
        numpy_helper.from_array(np.asarray(row_ident, np.float32).reshape(1, n), "rowIdentF"),
        numpy_helper.from_array(np.array([NO_OTHER_IDENTITY], np.float32), "noOther"),
        numpy_helper.from_array(np.array([-1.0], np.float32), "minusOne"),
    ]
    if bank_as_input:
        inputs.append(helper.make_tensor_value_info("bank", TensorProto.FLOAT, [dim, n]))
        bank_name = "bank"
    else:
        # The rows as stored (fp16), widened and scaled to unit length again, as every bank reader
        # does after decoding. Runtimes fold these three nodes into one constant when the session
        # is created.
        initializers.append(numpy_helper.from_array(np.ascontiguousarray(bank.T.astype(np.float16)), "bank16"))
        nodes += [
            helper.make_node("Cast", ["bank16"], ["bankWide"], to=TensorProto.FLOAT),
            helper.make_node("ReduceL2", ["bankWide"], ["bankNorm"], axes=[0], keepdims=1),
            helper.make_node("Div", ["bankWide", "bankNorm"], ["bankF"]),
        ]
        bank_name = "bankF"
    nodes.append(helper.make_node("MatMul", ["queries", bank_name], ["scores"]))
    # The best row's identity, found without an integer index: WebGPU runtimes keep int64 tensors
    # off the GPU, and a Slice or Gather on one would send the scores back to the CPU. The rows
    # that reach the best score name their identity as a float (exact below 2**24 identities);
    # on an exact tie between identities the larger index wins.
    nodes += [
        helper.make_node("ReduceMax", ["scores"], ["best"], axes=[1], keepdims=1),  # [L, 1]
        helper.make_node("Equal", ["scores", "best"], ["atBest"]),  # [L, N]
        helper.make_node("Where", ["atBest", "rowIdentF", "minusOne"], ["identAtBest"]),
        helper.make_node("ReduceMax", ["identAtBest"], ["bestIdent"], axes=[1], keepdims=1),  # [L, 1]
        helper.make_node("Equal", ["rowIdentF", "bestIdent"], ["sameIdent"]),  # [L, N]
        helper.make_node("Where", ["sameIdent", "noOther", "scores"], ["rivals"]),
        helper.make_node("ReduceMax", ["rivals"], ["other"], axes=[1], keepdims=0),
    ]
    if with_topk:
        initializers.append(numpy_helper.from_array(np.array([top_k], np.int64), "k"))
        nodes.append(helper.make_node("TopK", ["scores", "k"], ["topScores", "topRows"], axis=1, largest=1, sorted=1))
        outputs_top = [helper.make_tensor_value_info("topScores", TensorProto.FLOAT, ["L", top_k]),
                       helper.make_tensor_value_info("topRows", TensorProto.INT64, ["L", top_k])]
    else:
        nodes += [
            helper.make_node("ArgMax", ["scores"], ["topRows"], axis=1, keepdims=1, select_last_index=0),
            helper.make_node("Identity", ["best"], ["topScores"]),
        ]
        outputs_top = [helper.make_tensor_value_info("topScores", TensorProto.FLOAT, ["L", 1]),
                       helper.make_tensor_value_info("topRows", TensorProto.INT64, ["L", 1])]
    graph = helper.make_graph(nodes, "bank-scorer", inputs,
                              [*outputs_top, helper.make_tensor_value_info("other", TensorProto.FLOAT, ["L"])],
                              initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)],
                              producer_name="cardvision.mtg.scorer")
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def write_scorer(bank: np.ndarray, row_ident: np.ndarray, out: str | Path, **options) -> Path:
    """Write the scoring graph to ``out`` (an ``.onnx`` file outside the repository)."""
    import onnx

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(scorer_model(bank, row_ident, **options), str(out))
    return out


def reference_scores(bank: np.ndarray, row_ident: np.ndarray, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """What ``topAndOther`` computes, in float64: best row (first on a tie), best, other."""
    s = np.asarray(queries, np.float64) @ np.asarray(bank, np.float64).T
    best_row = np.argmax(s, axis=1)
    best = s[np.arange(len(s)), best_row]
    same = np.asarray(row_ident)[None, :] == np.asarray(row_ident)[best_row][:, None]
    other = np.where(same, NO_OTHER_IDENTITY, s).max(axis=1)
    return best_row, best, other
