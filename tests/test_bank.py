# SPDX-License-Identifier: Apache-2.0
"""The bank's storage format and the identity groups."""

from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from cardvision.core import bank as B
from cardvision.core.catalog import load_catalog


def finite_fp16(seed: int, count: int) -> np.ndarray:
    """Random fp16 values drawn from every finite bit pattern: both zeros, subnormals, extremes."""
    bits = np.random.default_rng(seed).integers(0, 1 << 16, size=count * 2, dtype=np.uint16)
    values = bits.view(np.float16)
    return values[np.isfinite(values)][:count]


def unit_rows(seed: int, count: int, width: int = 128) -> np.ndarray:
    return B.l2_normalize(np.random.default_rng(seed).normal(size=(count, width)))


# --------------------------------------------------------------------------- vectors

def test_vector_bytes_are_little_endian_half_floats():
    assert B.encode_vector([1.0, -2.0, 0.5]) == "ADwAwAA4"  # 00 3c | 00 c0 | 00 38
    assert B.decode_vector("ADwAwAA4").tolist() == [1.0, -2.0, 0.5]
    assert B.VECTOR_DTYPE == np.dtype("<f2")


@pytest.mark.parametrize("seed", range(8))
def test_encode_then_decode_is_the_identity_on_fp16_values(seed):
    values = finite_fp16(seed, 128)
    decoded = B.decode_vector(B.encode_vector(values))
    assert decoded.dtype == np.float16
    # compared as bit patterns, so that -0.0 and subnormals count
    assert np.array_equal(decoded.view(np.uint16), values.view(np.uint16))
    assert B.encode_vector(decoded) == B.encode_vector(values)


@pytest.mark.parametrize("seed", range(4))
def test_encode_narrows_wider_floats_through_float32(seed):
    row = unit_rows(seed, 1)[0].astype(np.float64)
    decoded = B.decode_vector(B.encode_vector(row))
    assert np.array_equal(decoded, row.astype(np.float32).astype(np.float16))


def test_encode_vector_rejects_what_cannot_be_stored():
    with pytest.raises(ValueError, match="1-D"):
        B.encode_vector(np.zeros((2, 4)))
    with pytest.raises(ValueError, match="1-D"):
        B.encode_vector([])
    with pytest.raises(ValueError, match="NaN or infinite"):
        B.encode_vector([0.0, float("nan")])
    with pytest.raises(ValueError, match="NaN or infinite"):
        B.encode_vector([0.0, float("inf")])


def test_decode_vector_rejects_malformed_text():
    with pytest.raises(ValueError, match="base64"):
        B.decode_vector("not base64!")
    with pytest.raises(ValueError, match="base64"):
        B.decode_vector(None)
    with pytest.raises(ValueError, match="whole"):
        B.decode_vector(base64.b64encode(b"\x00\x3c\x00").decode())
    with pytest.raises(ValueError, match="whole"):
        B.decode_vector("")


def test_decoded_vector_is_writable():
    decoded = B.decode_vector("ADwAwAA4")
    decoded[0] = 0  # must not raise


# --------------------------------------------------------------------------- normalization

def test_l2_normalize_works_on_the_last_axis_in_float32():
    v = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float64)
    out = B.l2_normalize(v)
    assert out.dtype == np.float32
    assert np.allclose(out, [[0.6, 0.8], [0.0, 1.0]])
    assert np.allclose(B.l2_normalize([3.0, 4.0]), [0.6, 0.8])


@pytest.mark.parametrize("seed", range(4))
def test_fp16_roundtrip_matches_its_definition_and_stays_unit(seed):
    rows = unit_rows(seed, 20)
    out = B.fp16_roundtrip(rows)
    half = rows.astype(np.float16).astype(np.float32)
    assert out.dtype == np.float32 and out.shape == rows.shape
    assert np.array_equal(out, half / np.linalg.norm(half, axis=-1, keepdims=True))
    assert np.allclose(np.linalg.norm(out, axis=-1), 1.0, atol=1e-6)
    assert np.abs(out - rows).max() < 1e-3  # fp16 keeps about three decimal digits
    assert np.array_equal(B.fp16_roundtrip(rows[0]), out[0])


# --------------------------------------------------------------------------- documents

def test_encode_bank_writes_one_entry_per_row_and_carries_metadata():
    ids = ["AB-001", "AB-002", "AB-001"]
    rows = unit_rows(1, 3, width=16)
    doc = B.encode_bank(ids, rows, extra={"model": "toy", "dim": 16})
    assert doc["model"] == "toy" and doc["dim"] == 16
    assert [entry["cardId"] for entry in doc["entries"]] == ids
    assert all(set(entry) == {"cardId", "vec"} for entry in doc["entries"])
    assert doc["entries"][0]["vec"] == B.encode_vector(rows[0])
    json.dumps(doc)  # the document is plain JSON


@pytest.mark.parametrize("seed", range(6))
def test_bank_encode_then_decode_is_the_identity_on_fp16_values(seed):
    rng = np.random.default_rng(seed)
    count, width = int(rng.integers(1, 40)), int(rng.choice([8, 64, 128]))
    stored = np.stack([finite_fp16(seed * 100 + i, width) for i in range(count)])
    ids = [f"AB-{i:03d}" for i in range(count)]
    doc = json.loads(json.dumps(B.encode_bank(ids, stored)))  # through JSON text, as on disk
    back_ids, back = B.stored_vectors(doc)
    assert back_ids == ids
    assert back.dtype == np.float16 and back.shape == (count, width)
    assert np.array_equal(back.view(np.uint16), stored.view(np.uint16))
    assert B.encode_bank(back_ids, back) == doc


@pytest.mark.parametrize("seed", range(4))
def test_decode_bank_renormalizes_in_float64_and_returns_float32(seed):
    ids = [f"AB-{i:03d}" for i in range(12)]
    rows = unit_rows(seed, 12)
    decoded_ids, decoded = B.decode_bank(B.encode_bank(ids, rows))
    assert decoded_ids == ids and decoded.dtype == np.float32 and decoded.shape == rows.shape
    for row, out in zip(rows, decoded):
        v = row.astype(np.float16).astype(np.float64)
        assert np.array_equal(out, (v / np.sqrt(float(np.sum(v * v)))).astype(np.float32))
    assert np.allclose(np.linalg.norm(decoded.astype(np.float64), axis=1), 1.0, atol=1e-7)
    # The in-memory round trip renormalizes in float32, so it may differ in the last bit only.
    assert np.abs(decoded - B.fp16_roundtrip(rows)).max() <= 1e-7


def test_decode_bank_is_unchanged_by_a_second_trip():
    ids = [f"AB-{i:03d}" for i in range(5)]
    doc = B.encode_bank(ids, unit_rows(3, 5))
    _, stored = B.stored_vectors(doc)
    assert B.encode_bank(ids, stored) == doc


def test_encode_bank_rejects_bad_input():
    rows = unit_rows(0, 2, width=8)
    with pytest.raises(ValueError, match="one row per id"):
        B.encode_bank(["AB-001"], rows)
    with pytest.raises(ValueError, match="one row per id"):
        B.encode_bank(["AB-001", "AB-002"], rows[0])
    with pytest.raises(ValueError, match="entries"):
        B.encode_bank(["AB-001", "AB-002"], rows, extra={"entries": []})
    bad = rows.copy()
    bad[1, 3] = np.nan
    with pytest.raises(ValueError, match="AB-002"):
        B.encode_bank(["AB-001", "AB-002"], bad)


def test_reading_a_bank_names_the_card_at_fault():
    good = B.encode_vector(unit_rows(0, 1, width=8)[0])
    with pytest.raises(ValueError, match='"entries" list'):
        B.stored_vectors({})
    with pytest.raises(ValueError, match=r"entries\[0\]"):
        B.stored_vectors({"entries": [{"vec": good}]})
    with pytest.raises(ValueError, match="AB-002.*base64"):
        B.stored_vectors({"entries": [{"cardId": "AB-001", "vec": good}, {"cardId": "AB-002", "vec": "?"}]})
    with pytest.raises(ValueError, match="AB-002.*4 values.*8"):
        B.stored_vectors({"entries": [{"cardId": "AB-001", "vec": good},
                                      {"cardId": "AB-002", "vec": B.encode_vector(np.ones(4))}]})
    with pytest.raises(ValueError, match="AB-003.*cannot be normalized"):
        B.decode_bank({"entries": [{"cardId": "AB-003", "vec": B.encode_vector(np.zeros(8))}]})


def test_an_empty_bank_decodes_to_no_rows():
    ids, rows = B.decode_bank({"entries": []})
    assert ids == [] and rows.shape[0] == 0 and rows.dtype == np.float32


def test_load_bank_reads_a_file(tmp_path):
    ids = ["AB-001", "AB-002", "AB-002A"]
    rows = unit_rows(4, 3)
    doc = B.encode_bank(ids, rows, extra={"model": "toy"})
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    loaded_ids, loaded_rows, loaded_doc = B.load_bank(path)
    assert loaded_ids == ids and loaded_doc == doc
    assert np.array_equal(loaded_rows, B.decode_bank(doc)[1])
    assert np.array_equal(B.load_bank(str(path))[1], loaded_rows)


# --------------------------------------------------------------------------- identity groups

def test_representative_key_puts_plain_ids_first_in_numeric_order():
    assert B.representative_key("AB-012") == (0, "AB", 12, "AB-012")
    assert B.representative_key("AB-012A") == (1, "AB-012A")
    assert B.representative_key("ab-012") == (1, "ab-012")
    ids = ["XY-P01", "AB-012A", "AB-100", "AB-012", "AB-9", "ZZ-001"]
    assert sorted(ids, key=B.representative_key) == ["AB-9", "AB-012", "AB-100", "ZZ-001", "AB-012A", "XY-P01"]


def test_identity_groups_pick_a_representative_and_index_the_rows():
    groups = [["AB-002A", "AB-002"], ["XY-P01", "AB-005", "AB-005B"]]
    ids = ["AB-001", "AB-002A", "AB-002", "AB-005B", "AB-001", "XY-P01"]
    identity = B.IdentityGroups(groups, ids)
    assert identity.groups == [["AB-002", "AB-002A"], ["AB-005", "AB-005B", "XY-P01"]]
    assert identity.of("AB-002A") == "AB-002" and identity.of("AB-002") == "AB-002"
    assert identity.of("XY-P01") == "AB-005" and identity.of("AB-005B") == "AB-005"
    assert identity.of("AB-001") == "AB-001" and identity.of("NOT-A-CARD") == "NOT-A-CARD"
    assert identity.of(None) is None
    assert identity.ids == ids
    assert identity.column_identity.dtype == np.int64
    assert identity.column_identity.tolist() == [0, 1, 1, 2, 0, 2]
    assert identity.identity_ids == ["AB-001", "AB-002", "AB-005"]
    assert identity.dropped == []
    assert identity.representative_map() == {"AB-002A": "AB-002", "AB-005B": "AB-005", "XY-P01": "AB-005"}
    assert list(identity.representative_map()) == sorted(identity.representative_map())


def test_identity_groups_same():
    identity = B.IdentityGroups([["AB-002", "AB-002A"]], ["AB-001", "AB-002", "AB-002A"])
    assert identity.same("AB-002", "AB-002A") and identity.same("AB-001", "AB-001")
    assert not identity.same("AB-001", "AB-002")
    assert not identity.same(None, None) and not identity.same("AB-001", None)


def test_a_group_whose_representative_is_unknown_is_set_aside():
    groups = [["AB-002", "AB-002A"], ["AB-005", "AB-005B"]]
    ids = ["AB-002", "AB-002A", "AB-005", "AB-005B"]
    identity = B.IdentityGroups(groups, ids, known={"AB-002A", "AB-005", "AB-005B"})
    assert identity.dropped == [["AB-002", "AB-002A"]]
    assert identity.of("AB-002A") == "AB-002A"  # stays its own identity
    assert identity.of("AB-005B") == "AB-005"
    assert identity.column_identity.tolist() == [0, 1, 2, 2]
    assert identity.groups == groups  # the input is kept whole


def test_identity_groups_reject_overlap_and_empty_groups():
    with pytest.raises(ValueError, match="AB-002"):
        B.IdentityGroups([["AB-001", "AB-002"], ["AB-002", "AB-003"]], [])
    with pytest.raises(ValueError, match="AB-001"):
        B.IdentityGroups([["AB-001", "AB-001"]], [])
    with pytest.raises(ValueError, match="empty"):
        B.IdentityGroups([[]], [])


def test_groups_by_name_on_the_toy_game(toy):
    catalog = load_catalog(toy[0])
    groups = B.groups_by_name(catalog.ids(), catalog.names())
    assert groups == [["TOY-003", "TOY-003A"], ["TOY-010", "TOY-P01"]]
    identity = B.IdentityGroups(groups, catalog.ids(), known=catalog.ids())
    assert identity.of("TOY-003A") == "TOY-003" and identity.of("TOY-P01") == "TOY-010"
    assert len(identity.identity_ids) == 18


def test_groups_by_name_ignores_repeats_order_and_cards_without_a_name():
    names = {"AB-001": "Kite", "AB-001A": "Kite", "AB-002": "Lamp", "AB-003": "Kite", "AB-009": "Lamp"}
    ids = ["AB-003", "AB-001A", "AB-002", "AB-001", "AB-003", "ZZ-404"]  # AB-009 is not in the bank
    assert B.groups_by_name(ids, names) == [["AB-001", "AB-001A", "AB-003"]]
    assert B.groups_by_name([], names) == []
    assert B.groups_by_name(ids, {}) == []


def test_groups_by_name_compares_names_exactly():
    assert B.groups_by_name(["AB-001", "AB-002"], {"AB-001": "Kite", "AB-002": "kite"}) == []


# --------------------------------------------------------------------------- the historical rule

def _near(base: np.ndarray, cosine: float, seed: int) -> np.ndarray:
    """A unit vector at the given cosine from the unit vector ``base``."""
    other = np.random.default_rng(seed).normal(size=base.shape)
    other -= other.dot(base) * base
    # A seed that drew ``base`` itself leaves only rounding error here, whose direction differs
    # between machines: the vector would not be at ``cosine`` from ``base``.
    assert np.linalg.norm(other) > 0.5, "the seed draws a vector parallel to base: pick another seed"
    other /= np.linalg.norm(other)
    return cosine * base + np.sqrt(1 - cosine ** 2) * other


def test_name_and_cosine_rule_needs_both_a_shared_name_and_a_close_row():
    rng = np.random.default_rng(0)
    a = rng.normal(size=64)
    a /= np.linalg.norm(a)
    b = rng.normal(size=64)
    b /= np.linalg.norm(b)
    rows = {
        "AB-001": [a],
        "AB-001A": [_near(a, 0.99, 1)],              # same name, looks alike: merged
        "AB-001B": [_near(a, 0.50, 2)],              # same name, looks different: kept apart
        "AB-002": [a],                               # identical row, another name: kept apart
        "AB-003": [b, _near(b, 0.2, 3)],
        "AB-003A": [_near(b, 0.1, 4), _near(b, 0.98, 5)],  # one close pair of views is enough
        "ZZ-404": [a],                               # not in the catalog
    }
    names = {"AB-001": "Kite", "AB-001A": "Kite", "AB-001B": "Kite", "AB-002": "Lamp",
             "AB-003": "Moth", "AB-003A": "Moth"}
    ids = [card for card, views in rows.items() for _ in views]
    matrix = np.stack([view for views in rows.values() for view in views])
    _, stored = B.stored_vectors(B.encode_bank(ids, matrix))
    assert B.groups_by_name_and_cosine(ids, stored, names) == [["AB-001", "AB-001A"], ["AB-003", "AB-003A"]]
    # With no bar to clear it reduces to the name rule.
    assert B.groups_by_name_and_cosine(ids, stored, names, threshold=-1.0) == B.groups_by_name(ids, names)
    # A stricter bar separates the 0.98 pair.
    assert B.groups_by_name_and_cosine(ids, stored, names, threshold=0.985) == [["AB-001", "AB-001A"]]


def test_name_and_cosine_rule_merges_transitively():
    rng = np.random.default_rng(1)
    a = rng.normal(size=64)
    a /= np.linalg.norm(a)
    b = _near(a, 0.97, 2)  # not seed 1: that draws a again (see _near)
    # c is close to b and, by construction, further than the bar from a.
    away = b - b.dot(a) * a
    away /= np.linalg.norm(away)
    c = 0.90 * a + np.sqrt(1 - 0.90 ** 2) * away
    assert a.dot(c) < 0.94 < 0.96 < b.dot(c)
    ids = ["AB-001", "AB-001A", "AB-001B"]
    _, stored = B.stored_vectors(B.encode_bank(ids, np.stack([a, b, c])))
    names = dict.fromkeys(ids, "Kite")
    assert B.groups_by_name_and_cosine(ids, stored, names) == [ids]
