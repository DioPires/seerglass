// SPDX-License-Identifier: Apache-2.0
// Bilinear resize that gives the same bytes as Pillow's `Image.resize(size, Image.BILINEAR)`.
//
// The detector was trained and measured on frames resized by Pillow, and Pillow's bilinear
// filter is not the one a canvas or a GPU applies: when it shrinks an image it widens the
// filter to cover every source pixel under an output pixel. To feed the detector the same
// pixels, this file follows Pillow's 8-bit resampler step for step (ImagingResample in
// libImaging/Resample.c; see the repository's NOTICE file): filter weights computed in double
// precision per output pixel over a support of max(1, scale), normalized to 22-bit fixed
// point, a horizontal pass into an 8-bit intermediate, then a vertical pass. Every sum is
// integer arithmetic, so the result does not depend on the machine.

/** Fixed-point precision of the 8-bit resampler: 32 - 8 - 2 bits. */
const PRECISION_BITS = 22;
const ONE = 1 << PRECISION_BITS;
const HALF = 1 << (PRECISION_BITS - 1);

interface Coeffs {
  /** Weights stored per output pixel. */
  ksize: number;
  /** Per output pixel: [first source index, number of weights used]. */
  bounds: Int32Array;
  /** `outSize * ksize` fixed-point weights. */
  kk: Int32Array;
}

function bilinearFilter(x: number): number {
  if (x < 0) x = -x;
  return x < 1 ? 1 - x : 0;
}

/** The weights of one axis, for a resize of the whole axis from `inSize` to `outSize`. */
function precomputeCoeffs(inSize: number, outSize: number): Coeffs {
  const scale = inSize / outSize;
  const filterscale = scale < 1 ? 1 : scale;
  const support = filterscale; // the bilinear filter's own support is 1
  const ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2);
  const kk = new Int32Array(outSize * ksize);
  const k = new Float64Array(ksize);
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    let ww = 0;
    const ss = 1 / filterscale;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    for (let x = 0; x < xmax; x++) {
      const w = bilinearFilter((x + xmin - center + 0.5) * ss);
      k[x] = w;
      ww += w;
    }
    for (let x = 0; x < xmax; x++) if (ww !== 0) k[x] = k[x]! / ww;
    for (let x = xmax; x < ksize; x++) k[x] = 0;
    for (let x = 0; x < ksize; x++) {
      const v = k[x]!;
      kk[xx * ksize + x] = v < 0 ? Math.trunc(-0.5 + v * ONE) : Math.trunc(0.5 + v * ONE);
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return { ksize, bounds, kk };
}

/** Back from fixed point to a byte. The sums stay below 2^31 (the weights of a pixel add up to
 *  about 2^22 and a channel is at most 255), so the shift is exact. */
function clip8(sum: number): number {
  const v = sum >> PRECISION_BITS;
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

// A video keeps its size from frame to frame, so the last pair of weight tables is kept.
let cachedKey = "";
let cachedH: Coeffs | null = null;
let cachedV: Coeffs | null = null;

function coeffsFor(srcW: number, srcH: number, outW: number, outH: number): { h: Coeffs; v: Coeffs } {
  const key = `${srcW}x${srcH}>${outW}x${outH}`;
  if (key !== cachedKey || !cachedH || !cachedV) {
    cachedH = precomputeCoeffs(srcW, outW);
    cachedV = precomputeCoeffs(srcH, outH);
    cachedKey = key;
  }
  return { h: cachedH, v: cachedV };
}

/**
 * Resize the RGB channels of an RGBA image (alpha is ignored) to `outW` x `outH` and return
 * packed RGB bytes (`outW * outH * 3`). Any target size, smaller or larger; an axis whose size
 * does not change is copied, as Pillow skips that pass.
 */
export function resizeBilinearRgb(
  rgba: Uint8ClampedArray | Uint8Array,
  srcW: number,
  srcH: number,
  outW: number,
  outH: number,
): Uint8Array {
  const { h: ch, v: cv } = coeffsFor(srcW, srcH, outW, outH);
  const needH = outW !== srcW;
  const needV = outH !== srcH;

  // Horizontal pass, over only the source rows the vertical pass will read; or a plain repack
  // to RGB when the width is unchanged.
  let temp: Uint8Array;
  let rowShift: number;
  if (needH) {
    const yFirst = needV ? cv.bounds[0]! : 0;
    const yLast = needV ? cv.bounds[outH * 2 - 2]! + cv.bounds[outH * 2 - 1]! : srcH;
    const rows = yLast - yFirst;
    rowShift = yFirst;
    temp = new Uint8Array(outW * rows * 3);
    const { ksize, bounds, kk } = ch;
    for (let yy = 0; yy < rows; yy++) {
      const srcRow = (yy + yFirst) * srcW * 4;
      const outRow = yy * outW * 3;
      for (let xx = 0; xx < outW; xx++) {
        const xmin = bounds[xx * 2]!;
        const xmax = bounds[xx * 2 + 1]!;
        const kOff = xx * ksize;
        let s0 = HALF;
        let s1 = HALF;
        let s2 = HALF;
        let p = srcRow + xmin * 4;
        for (let x = 0; x < xmax; x++, p += 4) {
          const kv = kk[kOff + x]!;
          s0 += rgba[p]! * kv;
          s1 += rgba[p + 1]! * kv;
          s2 += rgba[p + 2]! * kv;
        }
        const o = outRow + xx * 3;
        temp[o] = clip8(s0);
        temp[o + 1] = clip8(s1);
        temp[o + 2] = clip8(s2);
      }
    }
  } else {
    rowShift = 0;
    temp = new Uint8Array(srcW * srcH * 3);
    for (let i = 0, j = 0; i < temp.length; i += 3, j += 4) {
      temp[i] = rgba[j]!;
      temp[i + 1] = rgba[j + 1]!;
      temp[i + 2] = rgba[j + 2]!;
    }
  }
  if (!needV) return temp;

  // Vertical pass over the intermediate, one output row at a time. The sums are integers, so
  // accumulating a row at once gives the same bytes as accumulating pixel by pixel.
  const out = new Uint8Array(outW * outH * 3);
  const rowLen = outW * 3;
  const acc = new Int32Array(rowLen);
  const { ksize, bounds, kk } = cv;
  for (let yy = 0; yy < outH; yy++) {
    const ymin = bounds[yy * 2]! - rowShift;
    const ymax = bounds[yy * 2 + 1]!;
    const kOff = yy * ksize;
    acc.fill(HALF);
    for (let y = 0; y < ymax; y++) {
      const kv = kk[kOff + y]!;
      const rowOff = (y + ymin) * rowLen;
      for (let i = 0; i < rowLen; i++) acc[i] = acc[i]! + temp[rowOff + i]! * kv;
    }
    const o = yy * rowLen;
    for (let i = 0; i < rowLen; i++) out[o + i] = clip8(acc[i]!);
  }
  return out;
}
