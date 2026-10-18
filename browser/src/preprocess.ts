// SPDX-License-Identifier: Apache-2.0
// Crop preprocessing: how the pixels of a crop become the recognizer's input tensor.
//
// The recognizer was trained on tensors made by exactly these operations and the bank was
// built with them (in Python: cardvision/core/preprocess.py). Any change to the arithmetic or
// to the order of the steps changes the embeddings and no longer matches a published bank.
//
// The pool. A crop of H x W pixels is reduced to a grid of outH x outW cells. Source pixel
// (x, y) belongs to cell (min(outW - 1, floor(x * outW / W)), min(outH - 1, floor(y * outH /
// H))), and a cell's value is the mean of its pixels per channel, summed in double precision
// and divided once. Two versions differ only when the crop is smaller than the grid along an
// axis, so that some cells receive no pixel:
//   version 1 leaves those cells at 0;
//   version 2 copies, per axis, the nearest row (column) that did receive pixels. On an exact
//             tie the lower index wins. Values are copied, never interpolated.
// For a crop at least as large as the grid on both axes the two versions agree bit for bit.
//
// Normalization. Each cell mean is narrowed to float32 and every later step is a float32
// operation, in this order: divide by 255, subtract 0.5, divide by 0.5. The tensor is laid out
// channel first: the red plane, then green, then blue.

export type PoolVersion = 1 | 2;

/** How a fractional box becomes whole-pixel crop edges. Every contract truncates: each edge is
 *  clipped to the frame and cut down to a whole number (see `cropRgba`). */
export type CropEdges = "truncate";

export type PreprocessingId = "full128";

/** A preprocessing contract, as a release document and a bank file write it. */
export interface PreprocessingContract {
  id: PreprocessingId;
  /** Side of the square model input, in pixels. */
  size: number;
  /** Whether a border is masked out before pooling. No contract here does. */
  centerMask: boolean;
  poolVersion: PoolVersion;
  crop: CropEdges;
}

/**
 * The contracts this runtime implements. `full128` is the whole crop pooled to 128 x 128 with
 * pool version 2 and normalized; the aspect ratio is not preserved.
 */
export const PREPROCESSING: Readonly<Record<PreprocessingId, Readonly<PreprocessingContract>>> = {
  full128: { id: "full128", size: 128, centerMask: false, poolVersion: 2, crop: "truncate" },
};

export function isPreprocessingId(v: unknown): v is PreprocessingId {
  return typeof v === "string" && Object.prototype.hasOwnProperty.call(PREPROCESSING, v);
}

/**
 * The contract a document's `preprocessing` object names, or null when this runtime cannot
 * reproduce it: the id must be known and every field must equal the contract's own, with no
 * field more or fewer.
 */
export function contractOf(raw: unknown): PreprocessingId | null {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) return null;
  const o = raw as Record<string, unknown>;
  if (!isPreprocessingId(o.id)) return null;
  const want = PREPROCESSING[o.id] as unknown as Record<string, unknown>;
  const keys = Object.keys(o);
  const same = keys.length === Object.keys(want).length && keys.every((k) => want[k] === o[k]);
  return same ? o.id : null;
}

/**
 * Pool an RGBA crop (alpha ignored) to `outW` x `outH` cell means per color channel, on the
 * 0 to 255 scale, as a row-major Float64Array of `outW * outH * 3` values (R, G, B per cell).
 */
export function boxAverageRgb(
  rgba: Uint8ClampedArray | Uint8Array,
  srcW: number,
  srcH: number,
  outW: number,
  outH: number,
  version: PoolVersion,
): Float64Array {
  const grid = new Float64Array(outW * outH * 3);
  const counts = new Uint32Array(outW * outH);
  for (let y = 0; y < srcH; y++) {
    const oy = Math.min(outH - 1, Math.floor((y * outH) / srcH));
    for (let x = 0; x < srcW; x++) {
      const ox = Math.min(outW - 1, Math.floor((x * outW) / srcW));
      const p = (y * srcW + x) * 4;
      const cell = (oy * outW + ox) * 3;
      grid[cell] = grid[cell]! + rgba[p]!;
      grid[cell + 1] = grid[cell + 1]! + rgba[p + 1]!;
      grid[cell + 2] = grid[cell + 2]! + rgba[p + 2]!;
      counts[oy * outW + ox] = counts[oy * outW + ox]! + 1;
    }
  }
  for (let i = 0; i < counts.length; i++) {
    const n = counts[i]!;
    if (n > 0) {
      grid[i * 3] = grid[i * 3]! / n;
      grid[i * 3 + 1] = grid[i * 3 + 1]! / n;
      grid[i * 3 + 2] = grid[i * 3 + 2]! / n;
    }
  }
  // Nothing to fill: version 1, a crop that reaches every cell, or an empty crop.
  if (version === 1 || (srcW >= outW && srcH >= outH) || srcW <= 0 || srcH <= 0) return grid;
  // The cell mapping is separable, so the fill is one index map per axis.
  const rowOf = nearestFilledIndex(srcH, outH);
  const colOf = nearestFilledIndex(srcW, outW);
  const filled = new Float64Array(grid.length);
  for (let oy = 0; oy < outH; oy++) {
    for (let ox = 0; ox < outW; ox++) {
      const from = (rowOf[oy]! * outW + colOf[ox]!) * 3;
      const to = (oy * outW + ox) * 3;
      filled[to] = grid[from]!;
      filled[to + 1] = grid[from + 1]!;
      filled[to + 2] = grid[from + 2]!;
    }
  }
  return filled;
}

/** Along one axis: for each of `nOut` output indices, itself when it receives a source index
 *  under floor(i * nOut / nSrc), otherwise the nearest output index that does. An exact tie
 *  goes to the lower index. Needs nSrc >= 1, so that index 0 always receives one. */
function nearestFilledIndex(nSrc: number, nOut: number): Int32Array {
  const filled = new Uint8Array(nOut);
  for (let i = 0; i < nSrc; i++) filled[Math.min(nOut - 1, Math.floor((i * nOut) / nSrc))] = 1;
  const map = new Int32Array(nOut);
  for (let o = 0; o < nOut; o++) {
    if (filled[o]) {
      map[o] = o;
      continue;
    }
    let lo = o - 1;
    while (lo >= 0 && !filled[lo]) lo--;
    let hi = o + 1;
    while (hi < nOut && !filled[hi]) hi++;
    map[o] = lo >= 0 && (hi >= nOut || o - lo <= hi - o) ? lo : hi;
  }
  return map;
}

/**
 * An RGBA crop as the model tensor of a contract: `3 * size * size` float32 values, channel
 * first.
 */
export function toContractTensor(
  rgba: Uint8ClampedArray | Uint8Array,
  srcW: number,
  srcH: number,
  contract: Pick<PreprocessingContract, "size" | "poolVersion">,
): Float32Array {
  const n = contract.size;
  const pooled = boxAverageRgb(rgba, srcW, srcH, n, n, contract.poolVersion);
  const out = new Float32Array(3 * n * n);
  const plane = n * n;
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) {
      let v = Math.fround(pooled[i * 3 + c]!);
      v = Math.fround(v / 255);
      v = Math.fround(v - 0.5);
      v = Math.fround(v / 0.5);
      out[c * plane + i] = v;
    }
  }
  return out;
}

/**
 * A square channel-first tensor turned `k` quarter turns counter-clockwise in the image plane
 * (numpy.rot90 over the two spatial axes). `k = 0` returns the tensor itself. The recognizer
 * looks at a crop in up to four orientations; the turns are applied to the pooled tensor, not
 * to the crop, on both the Python side and this one, so the two agree exactly.
 */
export function rotateTensor90(t: Float32Array, k: number, n: number): Float32Array {
  const plane = n * n;
  if (t.length !== 3 * plane) throw new Error(`rotateTensor90: a tensor of ${t.length} values is not 3 x ${n} x ${n}`);
  const r = ((k % 4) + 4) % 4;
  if (r === 0) return t;
  const out = new Float32Array(t.length);
  for (let c = 0; c < 3; c++) {
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        // One counter-clockwise quarter turn sends (y, x) to (n - 1 - x, y).
        let oy: number;
        let ox: number;
        if (r === 1) {
          oy = n - 1 - x;
          ox = y;
        } else if (r === 2) {
          oy = n - 1 - y;
          ox = n - 1 - x;
        } else {
          oy = x;
          ox = n - 1 - y;
        }
        out[c * plane + oy * n + ox] = t[c * plane + y * n + x]!;
      }
    }
  }
  return out;
}
