// SPDX-License-Identifier: Apache-2.0
import { describe, expect, it } from "vitest";

import { decodeFp16Base64, type EmbeddingBank, isPackedBank, parseBank, parsePackedBank, readBankFile } from "../src/bank.js";
import { groupsByName, IdentityMap, parseIdentityGroups } from "../src/identity.js";
import { DEFAULT_ACCEPT, EmbeddingRecognizer, NO_OTHER_IDENTITY } from "../src/recognizer.js";
import { axis, blend, contract, float32FromHex, sha } from "./support.js";

const fixture = contract.bank;
const FULL128 = { id: "full128", size: 128, centerMask: false, poolVersion: 2, crop: "truncate" };

function bankDoc(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return { schemaVersion: 1, kind: "embedding-bank", version: 3, model: "some-recognizer", dim: fixture.dim, preprocessing: FULL128, centerMask: false, entries: fixture.entries, ...overrides };
}

function groupsDoc(groups: string[][], representative: Record<string, string>): unknown {
  return { schemaVersion: 1, kind: "identity-groups", version: 1, groups, representative };
}

/** A bank of unit rows along chosen axes, for margins worked by hand. */
function handBank(ids: string[], rows: Float32Array[]): EmbeddingBank {
  const dim = rows[0]!.length;
  const matrix = new Float32Array(ids.length * dim);
  rows.forEach((row, i) => matrix.set(row, i * dim));
  return { ids, matrix, dim, version: 1, model: "hand", preprocessing: "full128", tau: null };
}

describe("the bank file", () => {
  it("decodes half floats: zero, one, a fraction, a negative, the smallest subnormal", () => {
    const bytes = Uint8Array.from([0x00, 0x00, 0x00, 0x3c, 0x00, 0x38, 0x00, 0xc0, 0x01, 0x00]);
    const text = Buffer.from(bytes).toString("base64");
    expect(Array.from(decodeFp16Base64(text, 5)!)).toEqual([0, 1, 0.5, -2, 2 ** -24]);
    expect(decodeFp16Base64(text, 4)).toBeNull();
    expect(decodeFp16Base64("not base64!", 5)).toBeNull();
  });

  it("reads the Python package's bank to the same float32 matrix, bit for bit", () => {
    const bank = parseBank(bankDoc())!;
    expect(bank.ids).toEqual(fixture.ids);
    expect([bank.dim, bank.version, bank.model, bank.preprocessing, bank.tau]).toEqual([128, 3, "some-recognizer", "full128", null]);
    expect(sha(bank.matrix)).toBe(fixture.matrixSha256);
  });

  it("renormalizes every row to unit length", () => {
    const bank = parseBank(bankDoc())!;
    for (let r = 0; r < bank.ids.length; r++) {
      let norm = 0;
      for (let k = 0; k < bank.dim; k++) norm += bank.matrix[r * bank.dim + k]! ** 2;
      expect(Math.sqrt(norm)).toBeCloseTo(1, 6);
    }
  });

  it("takes the contract as an id or written out, and records tau when the file does", () => {
    expect(parseBank(bankDoc({ preprocessing: "full128" }))!.preprocessing).toBe("full128");
    expect(parseBank(bankDoc({ tau: 0.62 }))!.tau).toBe(0.62);
  });

  it("refuses the whole file on anything it cannot read", () => {
    const entries = fixture.entries as Array<{ cardId: string; vec: string }>;
    const zero = Buffer.alloc(fixture.dim * 2).toString("base64");
    const bad: Array<Record<string, unknown>> = [
      { kind: "something-else" },
      { version: "3" },
      { model: 7 },
      { dim: 0 },
      { dim: 64 },
      { entries: [] },
      { preprocessing: "another-contract" },
      { preprocessing: { ...FULL128, size: 96 } },
      { preprocessing: undefined },
      { centerMask: true },
      { tau: "0.62" },
      { entries: [...entries.slice(0, 2), { cardId: 5, vec: entries[0]!.vec }] },
      { entries: [...entries.slice(0, 2), { cardId: "X", vec: entries[0]!.vec.slice(4) }] },
      { entries: [...entries.slice(0, 2), { cardId: "X", vec: zero }] },
      { entries: [entries[0], null] },
    ];
    for (const change of bad) expect(parseBank(bankDoc(change)), JSON.stringify(change).slice(0, 80)).toBeNull();
    expect(parseBank(null)).toBeNull();
    expect(parseBank(bankDoc())).not.toBeNull();
  });
});

describe("the packed bank file", () => {
  const packed = contract.packedBank;
  const file = (): Uint8Array => new Uint8Array(Buffer.from(packed.file, "base64"));

  /** The file's header and the bytes after it. */
  function split(bytes: Uint8Array): { header: Record<string, unknown>; values: Uint8Array } {
    const length = new DataView(bytes.buffer, bytes.byteOffset).getUint32(8, true);
    return { header: JSON.parse(new TextDecoder().decode(bytes.subarray(12, 12 + length))), values: bytes.slice(12 + length) };
  }

  /** A packed file of this header and these value bytes. */
  function assemble(header: unknown, values: Uint8Array): Uint8Array {
    const text = new TextEncoder().encode(JSON.stringify(header));
    const out = new Uint8Array(12 + text.length + values.length);
    out.set(file().subarray(0, 8));
    new DataView(out.buffer).setUint32(8, text.length, true);
    out.set(text, 12);
    out.set(values, 12 + text.length);
    return out;
  }

  it("reads the Python package's packed bank to the same float32 matrix, bit for bit", () => {
    const bytes = file();
    expect(sha(bytes)).toBe(packed.fileSha256);
    expect(isPackedBank(bytes)).toBe(true);
    const bank = parsePackedBank(bytes)!;
    expect(bank.ids).toEqual(packed.ids);
    expect([bank.dim, bank.version, bank.model, bank.preprocessing, bank.tau]).toEqual([128, 3, "récognizer ü", "full128", null]);
    expect(sha(bank.matrix)).toBe(packed.matrixSha256);
    expect(Buffer.from(split(bytes).values).toString("hex")).toBe(packed.values);
    expect(readBankFile(bytes)).toEqual(bank);
  });

  it("reads a JSON bank as parseBank does, and tells the two kinds apart by the magic number", () => {
    const json = new TextEncoder().encode(JSON.stringify(bankDoc()));
    expect(isPackedBank(json)).toBe(false);
    expect(readBankFile(json)).toEqual(parseBank(bankDoc()));
    expect(readBankFile(new TextEncoder().encode("not json"))).toBeNull();
    expect(parseBank(bankDoc({ dtype: "int8" }))).toBeNull();
    expect(parseBank(bankDoc({ dtype: "fp16" }))).not.toBeNull();
  });

  it("refuses the whole file on anything it cannot read", () => {
    const { header, values } = split(file());
    const withHeader = (change: (h: Record<string, any>) => void): Uint8Array => {
      const h = structuredClone(header) as Record<string, any>;
      change(h);
      return assemble(h, values);
    };
    const withValue = (at: number, value: number): Uint8Array => {
      const v = values.slice();
      new Int8Array(v.buffer)[at] = value;
      return assemble(header, v);
    };
    const whole = file();
    const prefix = (length: number, body: number[]): Uint8Array => {
      const out = new Uint8Array(12 + body.length);
      out.set(whole.subarray(0, 8));
      new DataView(out.buffer).setUint32(8, length, true);
      out.set(body, 12);
      return out;
    };
    const otherMagic = whole.slice();
    otherMagic[6] = 0x4b;
    const bad: Record<string, Uint8Array> = {
      "another magic": otherMagic,
      "a short file": whole.subarray(0, 10),
      "a header of no bytes": prefix(0, Array(64).fill(0)),
      "a header past the end": prefix(1000, [0x7b, 0x7d]),
      "a header that is not UTF-8": prefix(2, [0xff, 0xfe]),
      "a header that is not JSON": prefix(3, [0x7b, 0x78, 0x7d]),
      "a header that is an array": prefix(2, [0x5b, 0x5d]),
      "another kind": withHeader((h) => (h.kind = "embedding-bank")),
      "another schema version": withHeader((h) => (h.schemaVersion = 2)),
      "fp16 values": withHeader((h) => (h.dtype = "fp16")),
      "entries in the header": withHeader((h) => (h.entries = [])),
      "a dim of 0": withHeader((h) => (h.dim = 0)),
      "a dim given as true": withHeader((h) => (h.dim = true)),
      "levels of 0": withHeader((h) => (h.levels = 0)),
      "levels of 128": withHeader((h) => (h.levels = 128)),
      "levels of 1.5": withHeader((h) => (h.levels = 1.5)),
      "levels given as true": withHeader((h) => (h.levels = true)),
      "no ids": withHeader((h) => (h.ids = [])),
      "an id that is not a string": withHeader((h) => (h.ids[1] = 7)),
      "no source": withHeader((h) => delete h.source),
      "a source hash in capitals": withHeader((h) => (h.source.sha256 = h.source.sha256.toUpperCase())),
      "a source without its dtype": withHeader((h) => delete h.source.dtype),
      "another preprocessing": withHeader((h) => (h.preprocessing = "another-contract")),
      "one trailing byte": assemble(header, Uint8Array.from([...values, 0])),
      "one byte missing": assemble(header, values.subarray(0, values.length - 1)),
      "a row too many for the ids": withHeader((h) => h.ids.pop()),
      "a value of -128": withValue(5, -128),
      "a value beyond the levels": withValue(5, 64),
      "a row of zeros": assemble(header, Uint8Array.from(values, (v, i) => (i >= 128 && i < 256 ? 0 : v))),
    };
    expect(parsePackedBank(assemble(header, values))).toEqual(parsePackedBank(whole));
    for (const [name, bytes] of Object.entries(bad)) expect(readBankFile(bytes), name).toBeNull();
  });
});

describe("identity groups", () => {
  it("groups by exact name as the Python package does", () => {
    const names = new Map<string, string>(Object.entries(fixture.names));
    expect(groupsByName(fixture.ids, names)).toEqual(fixture.groups);
    // An id without a name is left alone; names differing in case are different names.
    expect(groupsByName(["B-2", "A-1", "A-1", "C-3", "D-4"], new Map([["A-1", "x"], ["B-2", "x"], ["C-3", "X"]]))).toEqual([["A-1", "B-2"]]);
  });

  it("reads a groups document and maps members to their representative", () => {
    const groups = parseIdentityGroups(groupsDoc(fixture.groups, fixture.representative))!;
    expect(groups.version).toBe(1);
    const map = new IdentityMap(groups);
    expect(map.collapsedCount).toBe(Object.keys(fixture.representative).length);
    // The identities, in the order of first appearance among the bank's rows.
    const identities: string[] = [];
    for (const id of fixture.ids as string[]) if (!identities.includes(map.of(id))) identities.push(map.of(id));
    expect(identities).toEqual(fixture.identityIds);
    expect(new IdentityMap().of("ANY-1")).toBe("ANY-1");
  });

  it("refuses the whole document on any inconsistency", () => {
    const ok = [["A-1", "A-1A"], ["B-1", "B-1A", "B-1B"]];
    const rep = { "A-1A": "A-1", "B-1A": "B-1", "B-1B": "B-1" };
    expect(parseIdentityGroups(groupsDoc(ok, rep))).not.toBeNull();
    const bad: unknown[] = [
      { ...(groupsDoc(ok, rep) as object), kind: "other" },
      { ...(groupsDoc(ok, rep) as object), schemaVersion: 2 },
      { ...(groupsDoc(ok, rep) as object), version: 1.5 },
      groupsDoc([["A-1"]], {}), // a group of one
      groupsDoc([["A-1", "A-1A"], ["A-1", "B-1"]], { "A-1A": "A-1", "B-1": "A-1" }), // an id in two groups
      groupsDoc(ok, { "A-1A": "A-1", "B-1A": "B-1" }), // a group with two representatives
      groupsDoc(ok, { ...rep, "A-1": "A-1A" }), // a group with none
      groupsDoc(ok, { ...rep, "B-1B": "B-1A" }), // a chain
      groupsDoc(ok, { ...rep, "A-1A": "B-1" }), // a representative from another group
      groupsDoc(ok, { ...rep, "Z-9": "A-1" }), // a member of no group
      null,
    ];
    for (const doc of bad) expect(parseIdentityGroups(doc)).toBeNull();
  });

  it("what is shown: the representative, or the printing that matched", () => {
    const groups = parseIdentityGroups(groupsDoc([["A-1", "A-1A"]], { "A-1A": "A-1" }));
    expect(new IdentityMap(groups).shown("A-1A")).toBe("A-1");
    const best = new IdentityMap(groups, undefined, "best-printing");
    expect([best.shown("A-1A"), best.of("A-1A")]).toEqual(["A-1A", "A-1"]);
  });

  it("sets aside a group whose representative cannot be shown, under the representative display only", () => {
    const groups = parseIdentityGroups(groupsDoc([["A-1", "A-1A"]], { "A-1A": "A-1" }));
    const known = (id: string) => id !== "A-1";
    expect(new IdentityMap(groups, known).of("A-1A")).toBe("A-1A");
    expect(new IdentityMap(groups, known, "best-printing").of("A-1A")).toBe("A-1");
  });
});

describe("the recognizer's looks", () => {
  it("gives the Python package's best row, score and margin for generated queries", () => {
    const bank = parseBank(bankDoc())!;
    const identity = new IdentityMap(parseIdentityGroups(groupsDoc(fixture.groups, fixture.representative)), undefined, "best-printing");
    const recognizer = new EmbeddingRecognizer(bank, DEFAULT_ACCEPT, identity);
    const queries = float32FromHex(fixture.queries);
    const looks = fixture.looks as Array<{ top: number; score: number; margin: number }>;
    expect(queries.length).toBe(looks.length * bank.dim);
    looks.forEach((want, i) => {
      const got = recognizer.matchScored(queries, i * bank.dim);
      expect(got.topId, `query ${i}`).toBe(bank.ids[want.top]);
      // The two sides add the same 128 products in a different order.
      expect(got.similarity).toBeCloseTo(want.score, 12);
      expect(got.margin).toBeCloseTo(want.margin, 12);
      expect(got.displayId).toBe(bank.ids[want.top]);
      expect(got.identity).toBe(identity.of(bank.ids[want.top]!));
    });
  });

  it("the published accept rule is tau 0.62, delta 0.05, with the rotation retry", () => {
    expect(DEFAULT_ACCEPT).toEqual({ tau: 0.62, delta: 0.05, rotationRetry: true });
  });

  it("two printings of one identity do not narrow each other's margin", () => {
    // Rows: A-1 along axis 0, A-1A almost the same, B-1 along axis 2.
    const bank = handBank(["A-1", "A-1A", "B-1"], [axis(8, 0), blend(8, 0, 1, 0.999), axis(8, 2)]);
    const grouped = new IdentityMap(parseIdentityGroups(groupsDoc([["A-1", "A-1A"]], { "A-1A": "A-1" })));
    const query = blend(8, 0, 1, 0.9995);
    const one = new EmbeddingRecognizer(bank, DEFAULT_ACCEPT, grouped).matchScored(query, 0);
    expect([one.identity, one.displayId]).toEqual(["A-1", "A-1"]);
    expect(one.margin).toBeGreaterThan(0.9); // measured against B-1, not against the other printing
    const apart = new EmbeddingRecognizer(bank, DEFAULT_ACCEPT).matchScored(query, 0);
    expect(apart.margin).toBeLessThan(0.01); // without groups the two printings rival each other
  });

  it("the first row wins an exact tie, and tau alone decides cardId", () => {
    const bank = handBank(["A-1", "B-1"], [axis(4, 0), axis(4, 0)]);
    const recognizer = new EmbeddingRecognizer(bank, { tau: 0.9, delta: 0.05, rotationRetry: true });
    const hit = recognizer.matchScored(axis(4, 0), 0);
    expect([hit.topId, hit.cardId, hit.similarity, hit.margin]).toEqual(["A-1", "A-1", 1, 0]);
    const miss = recognizer.matchScored(blend(4, 0, 1, 0.8), 0);
    expect([miss.topId, miss.cardId]).toEqual(["A-1", null]);
  });

  it("reads the embedding at an offset", () => {
    const bank = handBank(["A-1", "B-1"], [axis(4, 0), axis(4, 1)]);
    const recognizer = new EmbeddingRecognizer(bank, DEFAULT_ACCEPT);
    const two = new Float32Array(8);
    two.set(axis(4, 0), 0);
    two.set(axis(4, 1), 4);
    expect([recognizer.matchScored(two, 0).topId, recognizer.matchScored(two, 4).topId]).toEqual(["A-1", "B-1"]);
  });

  it("a bank of one identity has a margin that never refuses", () => {
    const bank = handBank(["A-1"], [axis(4, 0)]);
    const look = new EmbeddingRecognizer(bank, DEFAULT_ACCEPT).matchScored(axis(4, 0), 0);
    expect(look.margin).toBe(1 - NO_OTHER_IDENTITY);
  });

  it("a pooled match needs both tau and the margin, and shows the display id", () => {
    const bank = handBank(["A-1", "A-1A", "B-1"], [axis(8, 0), blend(8, 0, 1, 0.999), axis(8, 2)]);
    const groups = parseIdentityGroups(groupsDoc([["A-1", "A-1A"]], { "A-1A": "A-1" }));
    const rule = { tau: 0.62, delta: 0.05, rotationRetry: true };
    const representative = new EmbeddingRecognizer(bank, rule, new IdentityMap(groups));
    expect(representative.matchPooled(blend(8, 0, 1, 0.999)).cardId).toBe("A-1");
    const printing = new EmbeddingRecognizer(bank, rule, new IdentityMap(groups, undefined, "best-printing"));
    expect(printing.matchPooled(blend(8, 0, 1, 0.999)).cardId).toBe("A-1A");
    // Torn between A and B: the score reaches tau, the margin does not.
    const torn = representative.matchPooled(blend(8, 0, 2, Math.SQRT1_2));
    expect(torn.cardId).toBeNull();
    expect(torn.similarity).toBeGreaterThan(0.62);
    expect(torn.margin).toBeLessThan(0.05);
    // Below tau.
    expect(representative.matchPooled(axis(8, 5)).cardId).toBeNull();
    // The margin test is inclusive, and the rule's delta is the default.
    expect(representative.matchPooled(blend(8, 0, 2, Math.SQRT1_2), 0).cardId).toBe("A-1");
  });
});
