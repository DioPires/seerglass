// SPDX-License-Identifier: Apache-2.0
// The geometric stage's debug record of one crop: what it took its keypoints from and what the
// vote counted. In Python: cardvision/geometric/stage.py (`debug_record`, `row_hashes`,
// `resize_direction`), which records the same fields, so that a run of this runtime and a run of
// the Python package can be compared crop by crop.
//
// The record is asked for (`GeometricStage.debug`) by a scoring run, never by a session that
// shows cards. It decides nothing: the keypoints, the vote and every fit are the same with it on
// and off.

/** A keypoint row's id: this many hexadecimal digits of its SHA-256. */
export const ROW_HASH_DIGITS = 16;

/** What the stage records about one crop when asked to. */
export interface StageDebug {
  /** The post-resize grey image the keypoints were taken from: its SHA-256 and size. */
  grey: { sha256: string; width: number; height: number };
  /** How the crop was brought to `crop.longSide`: bilinear up, area averaging down, or not. */
  resize: "up" | "down" | "none";
  /** The crop's own size, `[width, height]`. */
  source: [number, number];
  /** The crop's keypoint count, and each row's id (`rowHashes`), strongest first. */
  keypoints: number;
  rows: string[];
  /** The whole vote, card id to count, before any card is left out of it. */
  votes: Record<string, number>;
  /** The stage's time on the crop, in ms (the result's `ms`). */
  ms: number;
}

/** How a `width` x `height` image is brought to `longSide` (Python: `resize_direction`). */
export function resizeDirection(width: number, height: number, longSide: number): StageDebug["resize"] {
  const scale = longSide / Math.max(height, width);
  return scale === 1 ? "none" : scale < 1 ? "down" : "up";
}

// ------------------------------------------------------------------------------- SHA-256
const K = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
  0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
  0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
  0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
  0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
  0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);

/**
 * The SHA-256 of `bytes` as lowercase hexadecimal text, computed in this thread. A record of a
 * frame's crops needs thousands of digests, and the Web Crypto API gives them only one promise at
 * a time.
 */
export function sha256Hex(bytes: Uint8Array): string {
  const length = bytes.length;
  const blocks = Math.ceil((length + 9) / 64);
  const padded = new Uint8Array(blocks * 64);
  padded.set(bytes);
  padded[length] = 0x80;
  const view = new DataView(padded.buffer);
  const bits = length * 8;
  view.setUint32(padded.length - 8, Math.floor(bits / 2 ** 32));
  view.setUint32(padded.length - 4, bits >>> 0);
  const h = new Uint32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
  const w = new Uint32Array(64);
  for (let block = 0; block < blocks; block++) {
    for (let t = 0; t < 16; t++) w[t] = view.getUint32(block * 64 + t * 4);
    for (let t = 16; t < 64; t++) {
      const a = w[t - 15]!;
      const b = w[t - 2]!;
      const s0 = ((a >>> 7) | (a << 25)) ^ ((a >>> 18) | (a << 14)) ^ (a >>> 3);
      const s1 = ((b >>> 17) | (b << 15)) ^ ((b >>> 19) | (b << 13)) ^ (b >>> 10);
      w[t] = (w[t - 16]! + s0 + w[t - 7]! + s1) >>> 0;
    }
    let [a, b, c, d, e, f, g, hh] = [h[0]!, h[1]!, h[2]!, h[3]!, h[4]!, h[5]!, h[6]!, h[7]!];
    for (let t = 0; t < 64; t++) {
      const S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
      const ch = (e & f) ^ (~e & g);
      const t1 = (hh + S1 + ch + K[t]! + w[t]!) >>> 0;
      const S0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
      const maj = (a & b) ^ (a & c) ^ (b & c);
      const t2 = (S0 + maj) >>> 0;
      hh = g;
      g = f;
      f = e;
      e = (d + t1) >>> 0;
      d = c;
      c = b;
      b = a;
      a = (t1 + t2) >>> 0;
    }
    h[0] = (h[0]! + a) >>> 0;
    h[1] = (h[1]! + b) >>> 0;
    h[2] = (h[2]! + c) >>> 0;
    h[3] = (h[3]! + d) >>> 0;
    h[4] = (h[4]! + e) >>> 0;
    h[5] = (h[5]! + f) >>> 0;
    h[6] = (h[6]! + g) >>> 0;
    h[7] = (h[7]! + hh) >>> 0;
  }
  return Array.from(h, (word) => word.toString(16).padStart(8, "0")).join("");
}

/**
 * One id per keypoint row, in order (Python: `row_hashes`): the SHA-256 of the row's position as
 * two little-endian float32 and its descriptor bytes, to `ROW_HASH_DIGITS` hexadecimal digits.
 */
export function rowHashes(n: number, xy: Float32Array, desc: Uint8Array, width = 32): string[] {
  const row = new Uint8Array(8 + width);
  const view = new DataView(row.buffer);
  const out: string[] = [];
  for (let i = 0; i < n; i++) {
    view.setFloat32(0, xy[2 * i]!, true);
    view.setFloat32(4, xy[2 * i + 1]!, true);
    row.set(desc.subarray(i * width, (i + 1) * width), 8);
    out.push(sha256Hex(row).slice(0, ROW_HASH_DIGITS));
  }
  return out;
}
