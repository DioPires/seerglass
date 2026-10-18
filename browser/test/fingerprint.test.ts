// SPDX-License-Identifier: Apache-2.0
import { describe, expect, it } from "vitest";

import { dhash, hamming, isDhash, SAME_FRAME_MAX_HAMMING } from "../score/dhash.js";
import { contract, syntheticRgba } from "./support.js";

const fixture = contract.dhash;

/** The picture the contract file describes under "ramps". */
function ramp(seed: number, width: number, height: number, across: number, down: number): Uint8ClampedArray {
  const data = syntheticRgba(seed, width, height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const level = 96 + across * 12 * Math.floor((9 * x) / width) + down * 12 * Math.floor((8 * y) / height);
      const p = (y * width + x) * 4;
      for (let c = 0; c < 3; c++) data[p + c] = Math.max(0, Math.min(255, Math.floor(data[p + c]! / 4) + level));
    }
  }
  return data;
}

describe("the frame fingerprint", () => {
  it("equals the Python package's on every generated image of the contract file", () => {
    expect(fixture.images.length).toBeGreaterThanOrEqual(12);
    for (const image of fixture.images) {
      expect(dhash(syntheticRgba(image.seed, image.width, image.height), image.width, image.height), `seed ${image.seed}`).toBe(image.dhash);
    }
  });

  it("equals the Python package's on pictures with a brightness ramp", () => {
    for (const item of fixture.ramps) {
      expect(dhash(ramp(item.seed, item.width, item.height, item.across, item.down), item.width, item.height), `seed ${item.seed}`).toBe(item.dhash);
    }
    expect(fixture.ramps[0].dhash).toBe("ffffffffffffffff"); // every cell brighter than the one on its left
  });

  it("rounds a gray value of a whole number and a half to the even number, as Python does", () => {
    const { width, height, rgb } = fixture.halves;
    expect(dhash(rgb, width, height, 3)).toBe(fixture.halves.dhash);
    expect(fixture.halves.dhash).toBe("0000000000000000");
    // The first pixel is such a color: 0.114 * 250 is exactly 28.5, and its neighbor cell is gray 28.
    expect(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]).toBe(28.5);
    expect([rgb[6], rgb[7], rgb[8]]).toEqual([28, 28, 28]);
  });

  it("reads RGB and RGBA alike, and ignores the fourth byte", () => {
    const rgba = syntheticRgba(7001, 333, 251);
    const rgb = new Uint8Array(333 * 251 * 3);
    for (let i = 0, p = 0; i < rgb.length; i += 3, p += 4) rgb.set([rgba[p]!, rgba[p + 1]!, rgba[p + 2]!], i);
    expect(dhash(rgb, 333, 251, 3)).toBe(fixture.images[0].dhash);
  });

  it("is 16 lower-case hexadecimal digits, and refuses an empty or short frame", () => {
    const print = dhash(syntheticRgba(1, 20, 20), 20, 20);
    expect(isDhash(print)).toBe(true);
    expect(isDhash("ABCDEF0123456789")).toBe(false);
    expect(isDhash("0123")).toBe(false);
    expect(isDhash(12)).toBe(false);
    expect(() => dhash(new Uint8Array(0), 0, 0)).toThrow("empty");
    expect(() => dhash(new Uint8Array(10), 4, 4)).toThrow("expected 64 bytes");
  });
});

describe("the distance between two fingerprints", () => {
  it("equals the Python package's on the pairs of the contract file", () => {
    expect(fixture.distances.length).toBeGreaterThanOrEqual(10);
    for (const [a, b, bits] of fixture.distances) expect(hamming(a, b), `${a} ${b}`).toBe(bits);
  });

  it("uses the Python package's threshold, and refuses anything that is not a fingerprint", () => {
    expect(SAME_FRAME_MAX_HAMMING).toBe(fixture.maxHamming);
    expect(() => hamming("0000000000000000", "xyz")).toThrow("not 16 lower-case hexadecimal digits");
  });
});
