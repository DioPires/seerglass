// SPDX-License-Identifier: Apache-2.0
import { describe, expect, it } from "vitest";

import { boxAverageRgb, contractOf, PREPROCESSING, rotateTensor90, toContractTensor } from "../src/preprocess.js";
import { cropRgba } from "../src/recognition.js";
import { contract, sha, syntheticRgba } from "./support.js";

const FULL128 = PREPROCESSING.full128;

/** An RGBA image whose red channel is `value(x, y)`. */
function image(w: number, h: number, value: (x: number, y: number) => number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[(y * w + x) * 4] = value(x, y);
  return out;
}

describe("the pool", () => {
  it("a cell is the mean of the pixels that fall in it", () => {
    // 4 x 2 pixels into 2 x 1 cells: the left cell holds columns 0 and 1, the right one 2 and 3.
    const grid = boxAverageRgb(image(4, 2, (x, y) => x * 10 + y), 4, 2, 2, 1, 1);
    expect(Array.from(grid)).toEqual([5.5, 0, 0, 25.5, 0, 0]);
  });

  it("version 1 leaves a cell that receives no pixel at 0; version 2 copies the nearest filled one", () => {
    // 3 source columns into 6 cells: columns 0, 2 and 4 receive a pixel.
    const src = image(3, 1, (x) => 100 + x);
    const red = (grid: Float64Array) => Array.from({ length: 6 }, (_, i) => grid[i * 3]);
    expect(red(boxAverageRgb(src, 3, 1, 6, 1, 1))).toEqual([100, 0, 101, 0, 102, 0]);
    // Cell 1 sits exactly between 0 and 2: the lower index wins. Cell 5 has one neighbor.
    expect(red(boxAverageRgb(src, 3, 1, 6, 1, 2))).toEqual([100, 100, 101, 101, 102, 102]);
  });

  it("version 2 fills rows the same way, and copies values without interpolating", () => {
    const src = image(1, 2, (_, y) => (y === 0 ? 10 : 250));
    const grid = boxAverageRgb(src, 1, 2, 3, 5, 2);
    const rows = Array.from({ length: 5 }, (_, r) => [grid[r * 9], grid[r * 9 + 3], grid[r * 9 + 6]]);
    // Rows 0 and 2 receive a pixel; row 1 is a tie (lower index), rows 3 and 4 copy row 2.
    expect(rows).toEqual([[10, 10, 10], [10, 10, 10], [250, 250, 250], [250, 250, 250], [250, 250, 250]]);
    expect(new Set(grid).size).toBeLessThanOrEqual(3); // 10, 250 and the 0 of the other channels
  });

  it("a crop at least as large as the grid on both axes gives the same bytes under both versions", () => {
    for (const [w, h] of [[16, 12], [40, 37], [16, 100]] as const) {
      const src = syntheticRgba(9, w, h);
      expect(sha(boxAverageRgb(src, w, h, 16, 12, 2))).toBe(sha(boxAverageRgb(src, w, h, 16, 12, 1)));
    }
  });

  it("gives the Python package's grids for generated images, under both versions", () => {
    for (const item of contract.preprocess.images) {
      const src = syntheticRgba(item.seed, item.width, item.height);
      expect(sha(boxAverageRgb(src, item.width, item.height, 16, 12, 1)), `v1 ${item.width}x${item.height}`).toBe(item.poolV1Sha256);
      expect(sha(boxAverageRgb(src, item.width, item.height, 16, 12, 2)), `v2 ${item.width}x${item.height}`).toBe(item.poolV2Sha256);
    }
  });
});

describe("the full128 contract", () => {
  it("is the whole crop pooled to 128 x 128 with version 2, no mask, truncated edges", () => {
    expect(FULL128).toEqual({ id: "full128", size: 128, centerMask: false, poolVersion: 2, crop: "truncate" });
  });

  it("normalizes in float32: divide by 255, subtract 0.5, divide by 0.5", () => {
    const t = toContractTensor(image(1, 1, () => 200), 1, 1, { size: 2, poolVersion: 2 });
    const f = Math.fround;
    const red = f(f(f(200 / 255) - 0.5) / 0.5);
    // Channel first: four red cells, then green and blue at -1.
    expect(Array.from(t)).toEqual([red, red, red, red, -1, -1, -1, -1, -1, -1, -1, -1]);
  });

  it("a small crop has no cell left at -1 (black) by the pool", () => {
    const t = toContractTensor(new Uint8ClampedArray(30 * 40 * 4).fill(255), 30, 40, FULL128);
    expect(t.every((v) => v === 1)).toBe(true);
  });

  it("gives the Python package's tensor for every crop of the generated frame", () => {
    const { frame, crops } = contract.preprocess;
    const data = syntheticRgba(frame.seed, frame.width, frame.height);
    expect(sha(data)).toBe(frame.rgbaSha256);
    for (const item of crops) {
      const [x, y, width, height] = item.box as [number, number, number, number];
      const crop = cropRgba({ data, width: frame.width, height: frame.height }, { x, y, width, height });
      expect([crop.width, crop.height], `edges of ${item.box}`).toEqual([item.edges[2] - item.edges[0], item.edges[3] - item.edges[1]]);
      const tensor = toContractTensor(crop.rgba, crop.width, crop.height, FULL128);
      expect(sha(tensor), `tensor of ${item.box}`).toBe(item.tensorSha256);
      for (let k = 0; k < 4; k++) expect(sha(rotateTensor90(tensor, k, 128)), `look ${k} of ${item.box}`).toBe(item.rot4Sha256[k]);
    }
  });

  it("gives the Python package's tensor for whole small images", () => {
    for (const item of contract.preprocess.images) {
      const src = syntheticRgba(item.seed, item.width, item.height);
      expect(sha(toContractTensor(src, item.width, item.height, FULL128)), `${item.width}x${item.height}`).toBe(item.tensorSha256);
    }
  });
});

describe("rotateTensor90", () => {
  it("turns counter-clockwise: the top-right cell goes to the top-left", () => {
    // One 2 x 2 plane per channel: [[1, 2], [3, 4]] -> [[2, 4], [1, 3]].
    const t = Float32Array.from([1, 2, 3, 4, 10, 20, 30, 40, 5, 6, 7, 8]);
    expect(Array.from(rotateTensor90(t, 1, 2))).toEqual([2, 4, 1, 3, 20, 40, 10, 30, 6, 8, 5, 7]);
    expect(Array.from(rotateTensor90(t, 2, 2)).slice(0, 4)).toEqual([4, 3, 2, 1]);
    expect(Array.from(rotateTensor90(t, 3, 2)).slice(0, 4)).toEqual([3, 1, 4, 2]);
  });

  it("zero turns return the tensor itself, and four quarter turns bring it back", () => {
    const t = Float32Array.from({ length: 3 * 9 }, (_, i) => i);
    expect(rotateTensor90(t, 0, 3)).toBe(t);
    expect(rotateTensor90(t, 4, 3)).toBe(t);
    let turned: Float32Array = t;
    for (let i = 0; i < 4; i++) turned = rotateTensor90(turned, 1, 3);
    expect(Array.from(turned)).toEqual(Array.from(t));
    expect(Array.from(rotateTensor90(t, -1, 3))).toEqual(Array.from(rotateTensor90(t, 3, 3)));
  });

  it("refuses a tensor of another size", () => {
    expect(() => rotateTensor90(new Float32Array(10), 1, 2)).toThrow();
  });
});

describe("contractOf", () => {
  it("reads a contract written out field for field", () => {
    expect(contractOf({ id: "full128", size: 128, centerMask: false, poolVersion: 2, crop: "truncate" })).toBe("full128");
  });

  it("refuses an unknown id, a changed field, a missing field and an extra one", () => {
    const good = { id: "full128", size: 128, centerMask: false, poolVersion: 2, crop: "truncate" };
    expect(contractOf({ ...good, id: "other" })).toBeNull();
    expect(contractOf({ ...good, size: 96 })).toBeNull();
    expect(contractOf({ ...good, poolVersion: 1 })).toBeNull();
    expect(contractOf({ id: "full128", size: 128 })).toBeNull();
    expect(contractOf({ ...good, extra: 1 })).toBeNull();
    expect(contractOf("full128")).toBeNull();
    expect(contractOf(null)).toBeNull();
  });
});
