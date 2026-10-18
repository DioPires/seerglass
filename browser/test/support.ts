// SPDX-License-Identifier: Apache-2.0
// Helpers shared by the tests: the fixtures, generated inputs and hashes. Nothing here is
// imported by the runtime.

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

export const FIXTURES = fileURLToPath(new URL("./fixtures/", import.meta.url));

/** `fixtures/contract.json`: inputs and the answers the Python package gives for them. It is
 *  written and checked by `tests/test_browser_fixtures.py` at the top of the repository. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const contract: any = JSON.parse(readFileSync(`${FIXTURES}contract.json`, "utf-8"));

export function fixtureBytes(name: string): Uint8Array {
  return new Uint8Array(readFileSync(`${FIXTURES}${name}`));
}

const TWO32 = 2 ** 32;

/** The generated RGBA image the contract file describes under "synthetic". */
export function syntheticRgba(seed: number, w: number, h: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(w * h * 4);
  const rowLen = w * 4;
  for (let y = 0; y < h; y++) {
    let s = ((seed + y * 0x9e3779b9) % TWO32) >>> 0;
    if (s === 0) s = 1;
    const base = y * rowLen;
    for (let k = 0; k < rowLen; k++) {
      s = (s ^ (s << 13)) >>> 0;
      s = (s ^ (s >>> 17)) >>> 0;
      s = (s ^ (s << 5)) >>> 0;
      out[base + k] = s >>> 24;
    }
  }
  return out;
}

/** A xorshift32 stream of numbers in [0, 1) with 24 bits each. */
export class Stream {
  private s: number;

  constructor(seed: number) {
    this.s = (seed % TWO32) >>> 0 || 1;
  }

  unit(): number {
    let s = this.s;
    s = (s ^ (s << 13)) >>> 0;
    s = (s ^ (s >>> 17)) >>> 0;
    s = (s ^ (s << 5)) >>> 0;
    this.s = s;
    return (s >>> 8) / 2 ** 24;
  }

  between(low: number, high: number): number {
    return low + (high - low) * this.unit();
  }

  below(n: number): number {
    return Math.floor(this.unit() * n);
  }
}

/** SHA-256 of a typed array's bytes. */
export function sha(bytes: ArrayBufferView): string {
  return createHash("sha256").update(Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength)).digest("hex");
}

/** The float32 values stored as hexadecimal text of their little-endian bytes. */
export function float32FromHex(hex: string): Float32Array {
  const bytes = Buffer.from(hex, "hex");
  return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

/** A unit vector along one axis. */
export function axis(dim: number, k: number): Float32Array {
  const v = new Float32Array(dim);
  v[k] = 1;
  return v;
}

/** A unit vector that is `cos` along axis `a` and the rest along axis `b`. */
export function blend(dim: number, a: number, b: number, cos: number): Float32Array {
  const v = new Float32Array(dim);
  v[a] = cos;
  v[b] = Math.sqrt(1 - cos * cos);
  return v;
}

/** A plain RGBA frame of one color. */
export function solidFrame(width: number, height: number, rgb: [number, number, number] = [0, 0, 0]): { data: Uint8ClampedArray; width: number; height: number } {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < data.length; i += 4) {
    data[i] = rgb[0];
    data[i + 1] = rgb[1];
    data[i + 2] = rgb[2];
    data[i + 3] = 255;
  }
  return { data, width, height };
}
