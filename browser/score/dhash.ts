// SPDX-License-Identifier: Apache-2.0
// The frame fingerprint: a 64-bit difference hash, to tell whether two pictures show the same
// frame. In Python: cardvision/data/dhash.py, which holds the definition. The two give the same
// 16 hexadecimal digits for the same RGB pixels, bit for bit.
//
//   1. Grayscale. Each pixel becomes round(0.299 R + 0.587 G + 0.114 B): the sum in double
//      precision, added in that order, rounded to the nearest whole number, halves to the even
//      one.
//   2. A grid of 9 columns by 8 rows. The pixel at column x and row y of a frame W wide and H
//      high lies in cell (min(8, floor(9x / W)), min(7, floor(8y / H))). A cell takes the mean
//      gray of its pixels, 0 when it has none.
//   3. Bits. For each row from the top and each of its first 8 cells from the left, the bit is
//      1 when the cell to its right is brighter. The first bit is the most significant.
//
// A browser's video decoder and its canvas do not give the pixels the Python package reads
// from the same video. The fingerprint keeps only which coarse regions are brighter than their
// neighbors, so two decodes of one frame differ by a few bits at most. That is what the
// threshold is for.

export const DHASH_COLUMNS = 9;
export const DHASH_ROWS = 8;

/** Two fingerprints at most this many bits apart are taken to be of the same frame. */
export const SAME_FRAME_MAX_HAMMING = 6;

const FORM = /^[0-9a-f]{16}$/;

/** Whether `text` is a fingerprint in its written form: 16 lower-case hexadecimal digits. */
export function isDhash(text: unknown): text is string {
  return typeof text === "string" && FORM.test(text);
}

/** Round to the nearest whole number, a half to the even one. `value` is 0 or more. */
function roundHalfEven(value: number): number {
  const floor = Math.floor(value);
  const rest = value - floor;
  if (rest < 0.5) return floor;
  if (rest > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

/**
 * The fingerprint of a frame. `data` holds the pixels row by row, `channels` bytes per pixel
 * (4 for a canvas's RGBA, 3 for RGB), red first; any further byte of a pixel is ignored.
 */
export function dhash(data: ArrayLike<number>, width: number, height: number, channels: 3 | 4 = 4): string {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) throw new Error("dhash: the frame is empty");
  if (data.length < width * height * channels) throw new Error(`dhash: expected ${width * height * channels} bytes for ${width} x ${height}, got ${data.length}`);
  const cells = DHASH_COLUMNS * DHASH_ROWS;
  // Sums of whole numbers are exact in double precision, whatever the order of addition.
  const sums = new Float64Array(cells);
  const counts = new Float64Array(cells);
  const columnOf = new Uint8Array(width);
  for (let x = 0; x < width; x += 1) columnOf[x] = Math.min(DHASH_COLUMNS - 1, Math.floor((x * DHASH_COLUMNS) / width));
  for (let y = 0; y < height; y += 1) {
    const base = Math.min(DHASH_ROWS - 1, Math.floor((y * DHASH_ROWS) / height)) * DHASH_COLUMNS;
    let p = y * width * channels;
    for (let x = 0; x < width; x += 1, p += channels) {
      const cell = base + columnOf[x]!;
      sums[cell]! += roundHalfEven(0.299 * data[p]! + 0.587 * data[p + 1]! + 0.114 * data[p + 2]!);
      counts[cell]! += 1;
    }
  }
  const mean = (cell: number): number => (counts[cell]! > 0 ? sums[cell]! / counts[cell]! : 0);
  let hex = "";
  for (let row = 0; row < DHASH_ROWS; row += 1) {
    for (let half = 0; half < 2; half += 1) {
      let nibble = 0;
      for (let k = 0; k < 4; k += 1) {
        const cell = row * DHASH_COLUMNS + half * 4 + k;
        nibble = (nibble << 1) | (mean(cell + 1) > mean(cell) ? 1 : 0);
      }
      hex += nibble.toString(16);
    }
  }
  return hex;
}

const NIBBLE_BITS = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4];

/** The number of bits in which two fingerprints differ, 0 to 64. */
export function hamming(a: string, b: string): number {
  if (!isDhash(a) || !isDhash(b)) throw new Error(`hamming: ${JSON.stringify(isDhash(a) ? b : a)} is not 16 lower-case hexadecimal digits`);
  let bits = 0;
  for (let i = 0; i < 16; i += 1) bits += NIBBLE_BITS[parseInt(a[i]!, 16) ^ parseInt(b[i]!, 16)]!;
  return bits;
}
