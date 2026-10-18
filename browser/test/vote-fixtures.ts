// SPDX-License-Identifier: Apache-2.0
// Descriptor sets for the vote's tests (test/vote.test.ts on the CPU, test/webgpu/ on a GPU).
// Generated, so nothing is stored: random bits, and sets full of equal distances.

import { DESCRIPTOR_BYTES } from "../src/geometric/vote.js";

/** A small deterministic generator (xorshift32), the same in Node and in a browser. */
export function random(seed: number): () => number {
  let x = seed >>> 0 || 1;
  return () => {
    x ^= x << 13;
    x >>>= 0;
    x ^= x >>> 17;
    x ^= x << 5;
    x >>>= 0;
    return x;
  };
}

/** `n` descriptors of random bits. */
export function randomDescriptors(n: number, seed: number): Uint8Array {
  const next = random(seed);
  const words = new Uint32Array(n * (DESCRIPTOR_BYTES / 4));
  for (let i = 0; i < words.length; i += 1) words[i] = next();
  return new Uint8Array(words.buffer);
}

/**
 * Descriptors where equal distances are everywhere: each row is one of `distinct` patterns, and
 * every pattern differs from the first by a few flipped bits, so many rows lie at the same
 * distance from a query and the tie rule decides the result.
 */
export function tiedDescriptors(n: number, distinct: number, seed: number): Uint8Array {
  const next = random(seed);
  const patterns: Uint8Array[] = [];
  const root = randomDescriptors(1, seed + 1);
  for (let p = 0; p < distinct; p += 1) {
    const pattern = root.slice();
    const flips = next() % 4;
    for (let f = 0; f < flips; f += 1) {
      const bit = next() % 256;
      pattern[bit >> 3]! ^= 1 << (bit & 7);
    }
    patterns.push(pattern);
  }
  const out = new Uint8Array(n * DESCRIPTOR_BYTES);
  for (let r = 0; r < n; r += 1) out.set(patterns[next() % distinct]!, r * DESCRIPTOR_BYTES);
  return out;
}

/** Queries drawn from the rows of `desc` (exact matches, distance 0) mixed with random ones. */
export function queriesFrom(desc: Uint8Array, n: number, seed: number): Uint8Array {
  const next = random(seed);
  const rows = desc.length / DESCRIPTOR_BYTES;
  const random_ = randomDescriptors(n, seed + 7);
  const out = new Uint8Array(n * DESCRIPTOR_BYTES);
  for (let q = 0; q < n; q += 1) {
    if (rows > 0 && next() % 2 === 0) {
      const r = next() % rows;
      out.set(desc.subarray(r * DESCRIPTOR_BYTES, (r + 1) * DESCRIPTOR_BYTES), q * DESCRIPTOR_BYTES);
    } else {
      out.set(random_.subarray(q * DESCRIPTOR_BYTES, (q + 1) * DESCRIPTOR_BYTES), q * DESCRIPTOR_BYTES);
    }
  }
  return out;
}
