# SPDX-License-Identifier: Apache-2.0
"""The packed bank: its bytes, its refusals, ``pack-bank``, and releases that pin one.

Every packed file is written under ``tmp_path``: the repository holds no binary file.
"""

from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest
from test_release_document import ACCEPT, COMPONENTS, components, write_tiny_detector

from cardvision import cli
from cardvision.bank import build as BB
from cardvision.bank import release as R
from cardvision.bank.documents import canonical_bytes, read_json, sha256_hex
from cardvision.core import bank as B
from cardvision.core.catalog import load_catalog
from cardvision.core.onnx import create_session
from cardvision.recognize import AcceptRule
from cardvision.recognize.looks import bank_transposed
from cardvision.recognize.recognizer import Recognizer

SOURCE_SHA = "ab" * 32


def unit_rows(seed: int, count: int, width: int = 16) -> np.ndarray:
    return B.l2_normalize(np.random.default_rng(seed).normal(size=(count, width)))


def json_bank(ids=("A-1", "A-2", "Bé-1", "A-1"), seed: int = 1, width: int = 16) -> dict:
    """A JSON bank document: its rows as fp16, read back and renormalized, as build-bank stores them."""
    header = {"schemaVersion": 1, "kind": "embedding-bank", "game": "toy", "locale": "en", "version": 3,
              "model": "tiny", "dim": width, "dtype": "fp16", "preprocessing": "full128", "views": 16,
              "aggregation": "mean", "cardCount": len(set(ids)), "entryCount": len(ids)}
    return B.encode_bank(list(ids), B.fp16_roundtrip(unit_rows(seed, len(ids), width)), header)


def layout(body: bytes) -> tuple[dict, bytes]:
    """A packed file's header and the bytes after it."""
    (length,) = struct.unpack_from("<I", body, 8)
    return json.loads(body[12:12 + length]), body[12 + length:]


def assemble(header: dict, values: bytes) -> bytes:
    text = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return B.PACKED_MAGIC + struct.pack("<I", len(text)) + text + values


# --------------------------------------------------------------------------- the bytes

def test_the_file_is_magic_length_header_and_rows():
    doc = json_bank()
    body = B.pack_bank(doc, SOURCE_SHA, levels=63)
    assert body[:8] == bytes([0x89, 0x63, 0x76, 0x62, 0x61, 0x6E, 0x6B, 0x0A])
    header, values = layout(body)
    assert list(header) == [*[key for key in doc if key != "entries"], "levels", "source", "ids"]
    assert header["kind"] == "embedding-bank-packed" and header["schemaVersion"] == 1 and header["dtype"] == "int8"
    assert header["levels"] == 63 and header["source"] == {"sha256": SOURCE_SHA, "dtype": "fp16"}
    assert header["ids"] == ["A-1", "A-2", "Bé-1", "A-1"] and header["version"] == 3
    assert "Bé-1".encode() in body  # written as itself, not escaped
    _, rows = B.decode_bank(doc)
    v = rows.astype(np.float64)
    want = np.rint(v * 63 / np.abs(v).max(axis=1, keepdims=True)).astype(np.int8)
    assert values == want.tobytes()
    assert (np.abs(want).max(axis=1) == 63).all()


def test_packing_is_deterministic_and_the_levels_choose_the_values():
    doc = json_bank()
    assert B.pack_bank(doc, SOURCE_SHA) == B.pack_bank(json.loads(json.dumps(doc)), SOURCE_SHA)
    assert layout(B.pack_bank(doc, SOURCE_SHA))[0]["levels"] == B.PACKED_LEVELS == 63
    for levels in (1, 7, 127):
        _, _, values = B.unpack_bank(B.pack_bank(doc, SOURCE_SHA, levels))
        assert np.abs(values).max() == levels


@pytest.mark.parametrize("levels", [0, 128, 1.5, True, -3])
def test_packing_refuses_levels_out_of_range(levels):
    with pytest.raises(ValueError, match="levels"):
        B.pack_bank(json_bank(), SOURCE_SHA, levels)


def test_packing_refuses_a_bad_source_hash_and_an_empty_bank():
    with pytest.raises(ValueError, match="source_sha256"):
        B.pack_bank(json_bank(), "AB" * 32)
    empty = {**json_bank(), "entries": []}
    with pytest.raises(ValueError, match="empty"):
        B.pack_bank(empty, SOURCE_SHA)


@pytest.mark.parametrize("levels", [1, 63, 127])
def test_the_decode_is_the_same_arithmetic_as_for_fp16(levels):
    doc = json_bank(ids=[f"C-{i}" for i in range(40)], seed=levels, width=128)
    header, _, rows = B.read_bank_bytes(B.pack_bank(doc, SOURCE_SHA, levels))
    _, _, values = B.unpack_bank(B.pack_bank(doc, SOURCE_SHA, levels))
    for row, stored in zip(rows, values):
        v = stored.astype(np.float64)
        assert row.dtype == np.float32
        assert row.tobytes() == (v / np.sqrt(float(np.sum(v * v)))).astype(np.float32).tobytes()
    assert header["kind"] == B.PACKED_KIND and "entries" not in header
    cosine = (rows.astype(np.float64) * B.decode_bank(doc)[1].astype(np.float64)).sum(axis=1)
    assert cosine.min() > {1: 0.5, 63: 0.999, 127: 0.9999}[levels]


def test_either_kind_of_file_reads_to_its_rows(tmp_path):
    doc = json_bank()
    (tmp_path / "bank.json").write_bytes(canonical_bytes(doc))
    (tmp_path / "bank.bin").write_bytes(B.pack_bank(doc, SOURCE_SHA))
    ids, rows, read = B.load_bank(tmp_path / "bank.json")
    assert read == doc and (ids, rows.tobytes()) == (B.decode_bank(doc)[0], B.decode_bank(doc)[1].tobytes())
    ids, rows, header = B.load_bank(tmp_path / "bank.bin")
    assert ids == doc_ids(doc) and header["dtype"] == "int8" and rows.shape == (4, 16)
    (tmp_path / "other.bin").write_bytes(b"\x00\x01 not a bank")
    with pytest.raises(ValueError, match="other.bin: neither a packed bank nor JSON"):
        B.load_bank(tmp_path / "other.bin")


def doc_ids(doc: dict) -> list[str]:
    return [entry["cardId"] for entry in doc["entries"]]


# --------------------------------------------------------------------------- the refusals

def _good() -> tuple[dict, bytes]:
    return layout(B.pack_bank(json_bank(), SOURCE_SHA, levels=63))


def _with(change) -> bytes:
    header, values = _good()
    change(header)
    return assemble(header, values)


def _values(change) -> bytes:
    header, values = _good()
    array = np.frombuffer(values, np.int8).copy()
    change(array)
    return assemble(header, array.tobytes())


BAD = {
    "another magic": lambda: b"\x89cvbanK\n" + B.pack_bank(json_bank(), SOURCE_SHA)[8:],
    "a short file": lambda: B.PACKED_MAGIC + b"\x01\x00",
    "a header of no bytes": lambda: B.PACKED_MAGIC + struct.pack("<I", 0) + bytes(64),
    "a header past the end": lambda: B.PACKED_MAGIC + struct.pack("<I", 1000) + b"{}",
    "a header that is not UTF-8": lambda: B.PACKED_MAGIC + struct.pack("<I", 2) + b"\xff\xfe",
    "a header that is not JSON": lambda: B.PACKED_MAGIC + struct.pack("<I", 3) + b"{x}",
    "a header that is an array": lambda: B.PACKED_MAGIC + struct.pack("<I", 2) + b"[]",
    "another kind": lambda: _with(lambda h: h.update(kind="embedding-bank")),
    "another schema version": lambda: _with(lambda h: h.update(schemaVersion=2)),
    "fp16 values": lambda: _with(lambda h: h.update(dtype="fp16")),
    "entries in the header": lambda: _with(lambda h: h.update(entries=[])),
    "a dim of 0": lambda: _with(lambda h: h.update(dim=0)),
    "a dim given as true": lambda: _with(lambda h: h.update(dim=True)),
    "levels of 0": lambda: _with(lambda h: h.update(levels=0)),
    "levels of 128": lambda: _with(lambda h: h.update(levels=128)),
    "levels of 1.5": lambda: _with(lambda h: h.update(levels=1.5)),
    "levels given as true": lambda: _with(lambda h: h.update(levels=True)),
    "no ids": lambda: _with(lambda h: h.update(ids=[])),
    "an id that is not a string": lambda: _with(lambda h: h["ids"].__setitem__(1, 7)),
    "no source": lambda: _with(lambda h: h.pop("source")),
    "a source hash in capitals": lambda: _with(lambda h: h["source"].update(sha256="AB" * 32)),
    "a source without its dtype": lambda: _with(lambda h: h["source"].pop("dtype")),
    "one trailing byte": lambda: B.pack_bank(json_bank(), SOURCE_SHA) + b"\x00",
    "one byte missing": lambda: B.pack_bank(json_bank(), SOURCE_SHA)[:-1],
    "a row too many for the ids": lambda: _with(lambda h: h["ids"].pop()),
    "a value of -128": lambda: _values(lambda a: a.__setitem__(5, -128)),
    "a value beyond the levels": lambda: _values(lambda a: a.__setitem__(5, 64)),
}


@pytest.mark.parametrize("case", sorted(BAD))
def test_a_malformed_packed_bank_is_refused(case):
    with pytest.raises(ValueError):
        B.read_bank_bytes(BAD[case]())


def test_a_zero_row_is_refused_naming_its_card():
    with pytest.raises(ValueError, match="'A-2' has a stored vector that cannot be normalized"):
        B.read_bank_bytes(_values(lambda a: a.__setitem__(slice(16, 32), 0)))


def test_a_json_bank_stores_fp16_only():
    doc = {**json_bank(), "dtype": "int8"}
    with pytest.raises(ValueError, match="JSON bank stores fp16"):
        B.decode_bank(doc)
    no_dtype = {key: value for key, value in json_bank().items() if key != "dtype"}
    assert B.decode_bank(no_dtype)[0] == doc_ids(json_bank())


def test_tools_that_read_a_bank_as_json_refuse_a_packed_one_by_name(tmp_path):
    (tmp_path / "bank.bin").write_bytes(B.pack_bank(json_bank(), SOURCE_SHA))
    with pytest.raises(ValueError, match="bank.bin is a packed bank.*JSON bank it was packed from"):
        read_json(tmp_path / "bank.bin")


def test_a_packed_banks_groups_do_not_take_the_place_of_its_sources():
    assert BB.groups_path_for("out/bank.json") == Path("out/bank.identity-groups.json")
    assert BB.groups_path_for("out/bank.bin") == Path("out/bank.bin.identity-groups.json")


# --------------------------------------------------------------------------- pack-bank

@pytest.fixture()
def source(toy, tiny_model, tmp_path) -> Path:
    """A JSON bank of the toy game and its identity groups, in ``tmp_path``."""
    catalog = load_catalog(toy[0])
    bank = BB.build_bank(catalog, toy[1], create_session(tiny_model), model="tiny").bank
    BB.write_bank(bank, catalog, tmp_path / "bank.json")
    return tmp_path / "bank.json"


def test_pack_bank_writes_the_bank_and_its_groups(source, capsys):
    out = source.with_name("bank.bin")
    assert cli.main(["pack-bank", str(source), "--out", str(out)]) == 0
    lines = capsys.readouterr().out.splitlines()
    body = out.read_bytes()
    header, ids, rows = B.read_bank_bytes(body)
    doc = read_json(source)
    assert header["source"]["sha256"] == sha256_hex(source.read_bytes()) and header["levels"] == 63
    assert lines[0].startswith(f"packed bank: {out} ({len(ids)} rows of 128 values, int8 within ±63; {len(body)} bytes, ")
    assert lines[1] == f"identity groups: {BB.groups_path_for(out)}"
    cosine = (rows.astype(np.float64) * B.decode_bank(doc)[1].astype(np.float64)).sum(axis=1).min()
    assert lines[2] == f"smallest cosine of a packed row to its JSON bank's row: {cosine:.6f}"
    groups, original = read_json(BB.groups_path_for(out)), read_json(BB.groups_path_for(source))
    assert groups == {**original, "bank": {**original["bank"], "sha256": sha256_hex(body)}}
    # The same bytes again: nothing to replace, so no --overwrite is needed.
    assert cli.main(["pack-bank", str(source), "--out", str(out)]) == 0
    assert out.read_bytes() == body
    # Other levels are other bytes, written only with --overwrite.
    assert cli.main(["pack-bank", str(source), "--out", str(out), "--levels", "127"]) == 1
    assert "already exists" in capsys.readouterr().err and out.read_bytes() == body
    assert cli.main(["pack-bank", str(source), "--out", str(out), "--levels", "127", "--overwrite"]) == 0
    assert B.unpack_bank(out.read_bytes())[0]["levels"] == 127


def test_pack_bank_refuses_what_it_cannot_pack(source, capsys):
    out = source.with_name("bank.bin")
    assert cli.main(["pack-bank", str(source), "--out", str(out)]) == 0
    capsys.readouterr()
    cases = [
        (["pack-bank", str(out), "--out", str(source.with_name("again.bin"))], "already a packed bank"),
        (["pack-bank", str(source), "--out", str(source)], "both an input and an output"),
        (["pack-bank", str(source), "--out", str(source.with_name("x.bin")), "--groups-out",
          str(BB.groups_path_for(source))], "both an input and an output"),
        (["pack-bank", str(source), "--out", str(source.with_name("x.bin")), "--groups",
          str(BB.groups_path_for(out))], "was not made for"),
        (["pack-bank", str(source), "--out", str(source.with_name("x.bin")), "--levels", "0"], "levels"),
    ]
    for argv, message in cases:
        assert cli.main(argv) == 1, argv
        assert message in capsys.readouterr().err, argv
    assert not source.with_name("x.bin").exists()


def test_commands_that_need_the_json_bank_refuse_a_packed_one(source, toy, tiny_model, capsys):
    out = source.with_name("bank.bin")
    assert cli.main(["pack-bank", str(source), "--out", str(out)]) == 0
    capsys.readouterr()
    for argv in (
        ["build-bank", "--catalog", str(toy[0]), "--art", str(toy[1]), "--recognizer", str(tiny_model),
         "--out", str(source.with_name("next.json")), "--base", str(out)],
        ["check-refresh", str(out), str(source)],
        ["check-bank-art", str(out), "--catalog", str(toy[0]), "--art", str(toy[1])],
    ):
        assert cli.main(argv) != 0, argv
        assert "bank.bin is a packed bank" in capsys.readouterr().err, argv


# --------------------------------------------------------------------------- releases

@pytest.fixture()
def packed_release(source, tiny_model, toy) -> Path:
    """The toy release of test_release_document, with a packed bank in place of the JSON one."""
    root = source.parent
    shutil.copy(toy[0], root / "catalog.json")
    write_tiny_detector(root / COMPONENTS["detector"])
    shutil.copy(tiny_model, root / COMPONENTS["recognizer"])
    assert cli.main(["pack-bank", str(source), "--out", str(root / "bank.bin")]) == 0
    R.build_release(root / "release.json", **components(root, bank=root / "bank.bin"))
    return root


def test_a_release_pins_a_packed_bank_and_verifies(packed_release, capsys):
    doc = R.verify_release(packed_release / "release.json")
    body = (packed_release / "bank.bin").read_bytes()
    header = B.unpack_bank(body)[0]
    assert doc["bank"]["path"] == "bank.bin" and doc["bank"]["sha256"] == sha256_hex(body)
    assert doc["bank"]["version"] == header["version"] and doc["bank"]["entryCount"] == len(header["ids"])
    assert doc["identity"]["groups"]["path"] == "bank.bin.identity-groups.json"
    assert R.bank_image_records(doc, packed_release) == (0, len(header["ids"]))
    assert R.packed_bank_header(doc, packed_release) == header
    capsys.readouterr()
    assert cli.main(["verify-release", str(packed_release / "release.json")]) == 0
    last = capsys.readouterr().out.splitlines()[-1]
    assert last == (f"bank: packed, int8 values within ±63, from the JSON bank {header['source']['sha256'][:16]} "
                    f"(version {header['version']}); its rows record no source image (artV)")


def test_a_damaged_packed_bank_fails_the_release(packed_release):
    doc = read_json(packed_release / "release.json")
    path = packed_release / "bank.bin"
    header, values = layout(path.read_bytes())
    for change, error in ((lambda h: h.update(model="other"), "bank: the file's model 'other'"),
                          (lambda h: h["ids"].__setitem__(0, "NOT-IN-CATALOG"), "bank id(s) are not in the catalog"),
                          (lambda h: h.update(levels=0), 'bank: "levels" must be')):
        changed = json.loads(json.dumps(header))
        change(changed)
        path.write_bytes(assemble(changed, values))
        body = path.read_bytes()
        doc["bank"].update({"sha256": sha256_hex(body), "bytes": len(body)})
        groups_path = packed_release / doc["identity"]["groups"]["path"]
        groups = read_json(groups_path)
        groups["bank"]["sha256"] = sha256_hex(body)
        groups_path.write_bytes(canonical_bytes(groups))
        doc["identity"]["groups"].update({"sha256": sha256_hex(groups_path.read_bytes()),
                                          "bytes": groups_path.stat().st_size})
        (packed_release / "release.json").write_bytes(canonical_bytes(R.with_id(doc)))
        with pytest.raises(R.ReleaseError) as raised:
            R.verify_release(packed_release / "release.json")
        assert any(error in found for found in raised.value.errors), raised.value.errors


def test_the_recognizer_scores_against_the_packed_rows(packed_release, toy):
    catalog = load_catalog(toy[0])
    recognizer = Recognizer.open(packed_release / COMPONENTS["recognizer"], packed_release / "bank.bin", catalog,
                                 AcceptRule(tau=ACCEPT["tau"], delta=ACCEPT["delta"]))
    _, ids, rows = B.read_bank_bytes((packed_release / "bank.bin").read_bytes())
    assert recognizer.ids == ids
    assert recognizer._bank_t.tobytes() == bank_transposed(rows).tobytes()
