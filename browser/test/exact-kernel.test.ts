// SPDX-License-Identifier: Apache-2.0
// The bank search in WebAssembly (src/exact-scorer.ts, src/exact-kernel.c): the same best row,
// score and best other identity as the recognizer's loop, bit for bit, on generated banks.
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";

import { describe, expect, it } from "vitest";

import type { EmbeddingBank } from "../src/bank.js";
import { rectOf } from "../src/boxes.js";
import { EXACT_KERNEL_BASE64, EXACT_KERNEL_COMPILER, EXACT_KERNEL_SHA256 } from "../src/exact-kernel-wasm.js";
import { type BankMatch, ExactBankScorer, KERNEL_DIM, KERNEL_MAX_LOOKS } from "../src/exact-scorer.js";
import { EmbeddingRecognizer } from "../src/recognizer.js";
import { Tracker } from "../src/tracker.js";

const D = KERNEL_DIM;

/** A small deterministic generator (mulberry32), for values in [-1, 1). */
function generator(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (((t ^ (t >>> 14)) >>> 0) / 4294967296) * 2 - 1;
  };
}

function unit(values: Float32Array): Float32Array {
  const norm = Math.hypot(...values);
  return values.map((v) => v / norm);
}

/** A bank of `rows` rows, row `r` of card `ids[r]`; rows of one card id share an identity. */
function bankOf(rows: Float32Array[], ids: string[]): EmbeddingBank {
  const matrix = new Float32Array(rows.length * D);
  rows.forEach((row, r) => matrix.set(row, r * D));
  return { ids, matrix, dim: D, version: 1, model: "test", preprocessing: "full128", tau: null };
}

function randomRows(count: number, seed: number, scale: (k: number) => number = () => 1): Float32Array[] {
  const next = generator(seed);
  return Array.from({ length: count }, () => unit(Float32Array.from({ length: D }, (_, k) => next() * scale(k))));
}

/** The loop's answer for every look: topAndOther, the reference. */
function loopSearch(bank: EmbeddingBank, embs: Float32Array, offsets: readonly number[]): BankMatch[] {
  const copy = { ...bank, matrix: bank.matrix.slice() };
  const loop = new EmbeddingRecognizer(copy) as unknown as { topAndOther(vec: Float32Array, offset: number): BankMatch };
  return offsets.map((offset) => loop.topAndOther(embs, offset));
}

/** The kernel's answer for every look, through a recognizer, so its row identities are the
 *  recognizer's own. */
function kernelSearch(bank: EmbeddingBank, embs: Float32Array, offsets: readonly number[]): BankMatch[] {
  const copy = { ...bank, matrix: bank.matrix.slice() };
  const scorer = ExactBankScorer.create(copy);
  expect(scorer).not.toBeNull();
  new EmbeddingRecognizer(copy, undefined, undefined, scorer);
  return scorer!.search(embs, offsets);
}

function looksOf(vectors: Float32Array[]): { embs: Float32Array; offsets: number[] } {
  const embs = new Float32Array(vectors.length * D);
  vectors.forEach((v, i) => embs.set(v, i * D));
  return { embs, offsets: vectors.map((_, i) => i * D) };
}

/** Kernel and loop agree on every look, compared with Object.is (toEqual on numbers). */
function agree(bank: EmbeddingBank, vectors: Float32Array[]): BankMatch[] {
  const { embs, offsets } = looksOf(vectors);
  const want = loopSearch(bank, embs, offsets);
  const got = kernelSearch(bank, embs, offsets);
  expect(got).toEqual(want);
  return got;
}

describe("the kept kernel", () => {
  const bytes = Buffer.from(EXACT_KERNEL_BASE64, "base64");

  it("is the module its SHA-256 names", () => {
    expect(createHash("sha256").update(bytes).digest("hex")).toBe(EXACT_KERNEL_SHA256);
  });

  it("is what the recorded compiler builds from the source, where that compiler is installed", async (context) => {
    let version: string;
    try {
      version = execFileSync("clang", ["--version"], { encoding: "utf8" }).split("\n")[0]!.trim();
      execFileSync("wasm-ld", ["--version"], { encoding: "utf8" });
    } catch {
      context.skip();
      return;
    }
    if (version !== EXACT_KERNEL_COMPILER) {
      context.skip();
      return;
    }
    // @ts-expect-error: a plain JavaScript module without types
    const { compileKernel, moduleText, OUTPUT } = await import("../scripts/build-kernel.mjs");
    const { readFileSync } = await import("node:fs");
    const fresh: Buffer = compileKernel();
    expect(fresh.equals(bytes)).toBe(true);
    expect(moduleText(fresh, version)).toBe(readFileSync(OUTPUT, "utf8"));
  });
});

describe("ExactBankScorer", () => {
  it("agrees with the loop on a bank of scattered identities, an odd number of rows, any number of looks", () => {
    const rows = randomRows(1001, 1);
    const next = generator(2);
    const ids = rows.map(() => `card-${Math.floor((next() + 1) * 18.5)}`);
    const bank = bankOf(rows, ids);
    for (const count of [1, 7, 8, 9, 50]) agree(bank, randomRows(count, 100 + count));
    // Looks near bank rows: a clear best, rivals close behind.
    const near = rows.slice(0, 20).map((row, i) => unit(row.map((v, k) => v + 0.05 * Math.sin(i + k))));
    agree(bank, near);
  });

  it("agrees with the loop on an even number of rows", () => {
    const rows = randomRows(64, 3);
    agree(bankOf(rows, rows.map((_, r) => `card-${r % 5}`)), randomRows(13, 4));
  });

  it("cuts a long batch into calls and keeps the order", () => {
    const rows = randomRows(41, 5);
    const bank = bankOf(rows, rows.map((_, r) => `card-${r % 7}`));
    const got = agree(bank, randomRows(KERNEL_MAX_LOOKS + 45, 6));
    expect(got).toHaveLength(KERNEL_MAX_LOOKS + 45);
  });

  it("keeps the loop's tie rule: the first strict maximum, and a tied row of another identity as the other", () => {
    const rows = randomRows(30, 7);
    rows[9] = rows[5]!.slice(); // a tie within one identity
    rows[20] = rows[12]!.slice(); // a tie across two identities
    rows[25] = rows[12]!.slice(); // and within the second one again
    const ids = rows.map((_, r) => `card-${r % 4}`);
    ids[9] = ids[5]!;
    ids[20] = "card-twin";
    ids[25] = "card-twin";
    const bank = bankOf(rows, ids);
    const got = agree(bank, [rows[5]!, rows[12]!, rows[20]!, rows[9]!]);
    expect(got.map((m) => m.bestRow)).toEqual([5, 12, 12, 5]);
    expect(got[1]!.other).toBe(got[1]!.best);
  });

  it("sums in the loop's order: rows and looks whose sums depend on it", () => {
    // Values of very different sizes, so that another order of the additions rounds otherwise.
    const scale = (k: number): number => (k % 3 === 0 ? 1e6 : k % 3 === 1 ? 1 : 1e-6);
    const rows = randomRows(201, 8, scale);
    const bank = bankOf(rows, rows.map((_, r) => `card-${r % 11}`));
    agree(bank, [...randomRows(16, 9, scale), ...randomRows(16, 10, (k) => scale(D - 1 - k))]);
  });

  it("agrees on a bank of one identity (no other identity) and of one row", () => {
    const rows = randomRows(17, 11);
    const one = agree(bankOf(rows, rows.map(() => "only")), randomRows(5, 12));
    expect(one.every((m) => m.other === -2)).toBe(true);
    const single = agree(bankOf([rows[0]!], ["only"]), [rows[0]!, ...randomRows(3, 13)]);
    expect(single.every((m) => m.bestRow === 0)).toBe(true);
  });

  it("agrees on a zero look and a look with no number in it", () => {
    const rows = randomRows(33, 14);
    const bank = bankOf(rows, rows.map((_, r) => `card-${r % 3}`));
    const got = agree(bank, [new Float32Array(D), new Float32Array(D).fill(Number.NaN), new Float32Array(D).fill(-0)]);
    expect(got[0]).toEqual({ bestRow: 0, best: 0, other: 0 });
    expect(got[1]).toEqual({ bestRow: -1, best: -2, other: -2 });
  });

  it("holds the bank: bank.matrix becomes its copy, with the same values", () => {
    const rows = randomRows(10, 15);
    const bank = bankOf(rows, rows.map((_, r) => `card-${r}`));
    const before = bank.matrix.slice();
    const old = bank.matrix;
    const scorer = ExactBankScorer.create(bank);
    expect(scorer).not.toBeNull();
    expect(bank.matrix).not.toBe(old);
    expect(bank.matrix).toEqual(before);
  });

  it("is not made for rows of another length, or for an empty bank", () => {
    const short: EmbeddingBank = { ids: ["a"], matrix: new Float32Array(64), dim: 64, version: 1, model: "test", preprocessing: "full128", tau: null };
    expect(ExactBankScorer.create(short)).toBeNull();
    expect(ExactBankScorer.create(bankOf([], []))).toBeNull();
  });

  it("will not search before the identities are set", () => {
    const rows = randomRows(4, 16);
    const scorer = ExactBankScorer.create(bankOf(rows, rows.map((_, r) => `card-${r}`)));
    expect(() => scorer!.search(rows[0]!, [0])).toThrow(/identities/);
  });
});

describe("EmbeddingRecognizer with the kernel", () => {
  it("gives the same looks and pooled matches as with the loop", () => {
    const rows = randomRows(301, 17);
    const ids = rows.map((_, r) => `card-${r % 29}`);
    const plain = new EmbeddingRecognizer(bankOf(rows, ids));
    const bank = bankOf(rows, ids);
    const fast = new EmbeddingRecognizer(bank, undefined, undefined, ExactBankScorer.create(bank));
    expect([plain.search, fast.search]).toEqual(["loop", "wasm-simd"]);
    const near = rows.slice(0, 24).map((row, i) => unit(row.map((v, k) => v + 0.3 * Math.cos(3 * i + k))));
    const { embs, offsets } = looksOf(near);
    expect(fast.matchScoredBatch(embs, offsets)).toEqual(plain.matchScoredBatch(embs, offsets));
    expect(fast.matchScored(embs, 5 * D)).toEqual(plain.matchScored(embs, 5 * D));
    expect(fast.matchPooledBatch(near)).toEqual(near.map((v) => plain.matchPooled(v)));
    expect(fast.scoreRow(embs, 0)).toEqual(plain.scoreRow(embs, 0));
  });
});

describe("Tracker with a batch resolver", () => {
  it("asks once per frame, for the tracks old enough, in order, and names them as the single resolver would", () => {
    const calls: number[] = [];
    const single = (pooled: Float32Array): string => `AXIS-${pooled.indexOf(Math.max(...pooled))}`;
    const tracker = new Tracker(single, (pooled) => {
      calls.push(pooled.length);
      return pooled.map(single);
    });
    const plain = new Tracker(single);
    const axis = (k: number): Float32Array => Float32Array.from({ length: 4 }, (_, i) => (i === k ? 1 : 0));
    const frame = (xs: number[]) =>
      xs.map((x, i) => {
        const aabb = { x, y: 0, width: 100, height: 100 };
        return { aabb, rect: rectOf(aabb), cardId: null, embedding: axis(i % 4) };
      });
    for (const xs of [[0, 300], [0, 300, 600], [0, 300, 600]]) {
      expect(tracker.update(frame(xs))).toEqual(plain.update(frame(xs)));
    }
    expect(calls).toEqual([2, 3]);
  });
});
