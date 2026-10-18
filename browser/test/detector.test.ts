// SPDX-License-Identifier: Apache-2.0
import { describe, expect, it } from "vitest";

import { rectOf } from "../src/boxes.js";
import { clipToFrame, decodeDetections, DEFAULT_DECODE, splitByClass, toFrameCorners, validBox, validDetections } from "../src/detector/decode.js";
import { DEFAULT_LETTERBOX, letterboxBgr, letterboxGeometry, roundHalfEven, toDetectorTensor } from "../src/detector/letterbox.js";
import { resizeBilinearRgb } from "../src/resize.js";
import { contract, float32FromHex, sha, syntheticRgba } from "./support.js";

describe("the letterbox", () => {
  it("rounds halves to the even neighbor", () => {
    expect([0.5, 1.5, 2.5, 3.5, 2.4, 2.6, 404.5, 405.5].map(roundHalfEven)).toEqual([0, 2, 2, 4, 2, 3, 404, 406]);
  });

  it("scales by the largest ratio that fits both axes; a 1280 x 768 frame is halved exactly", () => {
    expect(letterboxGeometry(1280, 768)).toEqual({ ratio: 0.5, width: 640, height: 384 });
    expect(letterboxGeometry(1920, 1080)).toEqual({ ratio: 1 / 3, width: 640, height: 360 });
    expect(letterboxGeometry(500, 900).height).toBe(384); // a tall frame is limited by its height
  });

  it("puts the frame top-left on a canvas of the pad value, blue first", () => {
    // A 2 x 1 frame on a 2 x 2 canvas: it fills the top row at ratio 1 and the bottom row is padding.
    const rgba = Uint8ClampedArray.from([200, 100, 50, 255, 200, 100, 50, 255]);
    const { canvas, ratio, width, height } = letterboxBgr(rgba, 2, 1, { width: 2, height: 2, pad: 7 });
    expect([ratio, width, height]).toEqual([1, 2, 1]);
    expect(Array.from(canvas)).toEqual([50, 100, 200, 50, 100, 200, 7, 7, 7, 7, 7, 7]);
    const { tensor } = toDetectorTensor(rgba, 2, 1, { width: 2, height: 2, pad: 7 });
    expect(Array.from(tensor)).toEqual([50, 50, 7, 7, 100, 100, 7, 7, 200, 200, 7, 7]);
  });

  it("gives the Python package's canvas and tensor for generated frames", () => {
    for (const item of contract.letterbox.cases) {
      const rgba = syntheticRgba(item.seed, item.width, item.height);
      const lb = letterboxBgr(rgba, item.width, item.height);
      expect(lb.ratio, `ratio ${item.width}x${item.height}`).toBe(item.ratio);
      expect([lb.width, lb.height]).toEqual(item.size);
      expect(sha(lb.canvas), `canvas ${item.width}x${item.height}`).toBe(item.canvasSha256);
      expect(sha(toDetectorTensor(rgba, item.width, item.height).tensor), `tensor ${item.width}x${item.height}`).toBe(item.tensorSha256);
    }
  });

  it("follows another canvas size and pad value when a release states them", () => {
    const item = contract.letterbox.custom;
    const rgba = syntheticRgba(item.seed, item.width, item.height);
    const lb = letterboxBgr(rgba, item.width, item.height, item.input);
    expect([lb.ratio, lb.width, lb.height]).toEqual([item.ratio, ...item.size]);
    expect(sha(lb.canvas)).toBe(item.canvasSha256);
    expect(sha(toDetectorTensor(rgba, item.width, item.height, item.input).tensor)).toBe(item.tensorSha256);
  });

  it("the published detector's canvas is 640 x 384 padded with 114", () => {
    expect(DEFAULT_LETTERBOX).toEqual({ width: 640, height: 384, pad: 114 });
  });
});

describe("resizeBilinearRgb", () => {
  it("copies when the size does not change, dropping alpha", () => {
    const rgba = syntheticRgba(3, 5, 4);
    const rgb = resizeBilinearRgb(rgba, 5, 4, 5, 4);
    expect(rgb.length).toBe(60);
    expect([rgb[0], rgb[1], rgb[2], rgb[3]]).toEqual([rgba[0], rgba[1], rgba[2], rgba[4]]);
  });

  it("widens the filter when it shrinks: halving weighs three source pixels 3, 3 and 1 at each end", () => {
    // 4 x 1 to 2 x 1. The filter's support is the scale (2), so output pixel 0, centered on
    // x = 1, covers source pixels 0 to 2 with weights 0.75, 0.75 and 0.25, normalized to sum 1.
    const rgba = new Uint8ClampedArray(4 * 4);
    [0, 70, 140, 210].forEach((v, i) => (rgba[i * 4] = v));
    const out = resizeBilinearRgb(rgba, 4, 1, 2, 1);
    expect([out[0], out[3]]).toEqual([(0 * 3 + 70 * 3 + 140) / 7, (70 + 140 * 3 + 210 * 3) / 7]);
  });

  it("a flat image stays flat at any size", () => {
    const rgba = new Uint8ClampedArray(7 * 5 * 4).fill(93);
    for (const [w, h] of [[3, 2], [14, 10], [7, 9], [20, 5]] as const) {
      expect(resizeBilinearRgb(rgba, 7, 5, w, h).every((v) => v === 93)).toBe(true);
    }
  });
});

describe("decodeDetections", () => {
  const rows = (...list: number[][]): Float32Array => Float32Array.from(list.flat());

  it("scores a row by objectness times its best class, and names that class", () => {
    const out = rows([100, 100, 40, 60, 0.9, 0.2, 0.8], [300, 100, 40, 60, 0.5, 1, 0]);
    const found = decodeDetections(out, 7, 0.5);
    expect(found.map((d) => d.cls)).toEqual([1, 0]);
    expect(found[0]!.score).toBe(Math.fround(Math.fround(0.9) * Math.fround(0.8)));
    // Canvas (80..120, 70..130) divided by the ratio 0.5.
    expect(found[0]!.aabb).toEqual({ x: 160, y: 140, width: 80, height: 120 });
    expect(found[0]!.rect).toEqual(rectOf(found[0]!.aabb));
  });

  it("the first class wins a tie, and a score exactly at the threshold passes", () => {
    const out = rows([100, 100, 40, 60, 0.6, 0.5, 0.5], [300, 100, 40, 60, 0.3, 1, 0], [100, 300, 40, 60, 0.29, 1, 0]);
    const found = decodeDetections(out, 7, 1);
    expect(found.map((d) => [d.cls, d.score])).toEqual([[0, Math.fround(0.3)], [0, Math.fround(0.3)]]);
  });

  it("suppresses by overlap whatever the class, keeping the better box; equal scores keep row order", () => {
    const out = rows([100, 100, 40, 60, 0.8, 1, 0], [102, 100, 40, 60, 0.9, 0, 1], [300, 100, 40, 60, 0.7, 1, 0], [300, 100, 40, 60, 0.7, 0, 1]);
    const found = decodeDetections(out, 7, 1);
    expect(found.map((d) => [d.aabb.x, d.cls])).toEqual([[82, 1], [280, 0]]);
  });

  it("keeps at most maxProposals, best first", () => {
    const out = rows(...Array.from({ length: 10 }, (_, i) => [50 + i * 60, 100, 20, 20, 0.5 + i * 0.04, 1, 0]));
    const found = decodeDetections(out, 7, 1, { ...DEFAULT_DECODE, maxProposals: 3 });
    expect(found.map((d) => d.aabb.x)).toEqual([580, 520, 460]);
  });

  it("refuses rows that do not fit the stated layout", () => {
    expect(() => decodeDetections(new Float32Array(15), 7, 1)).toThrow();
    expect(() => decodeDetections(new Float32Array(10), 5, 1)).toThrow();
    expect(decodeDetections(new Float32Array(0), 7, 1)).toEqual([]);
  });

  it("gives the Python package's detections, value for value, for generated rows", () => {
    for (const item of contract.decode.cases) {
      const found = decodeDetections(float32FromHex(item.rows), item.columns, item.ratio, {
        threshold: item.threshold, nmsIou: item.nmsIou, maxProposals: item.maxProposals,
      });
      const got = found.map((d) => ({ x: d.aabb.x, y: d.aabb.y, width: d.aabb.width, height: d.aabb.height, score: d.score, cls: d.cls }));
      expect(got, item.name).toEqual(item.detections);
    }
  });

  it("the published operating point is 0.3, 0.45 and 80", () => {
    expect(DEFAULT_DECODE).toEqual({ threshold: 0.3, nmsIou: 0.45, maxProposals: 80 });
  });
});

describe("valid boxes, clipping and classes", () => {
  it("a box is valid when its crop is at least the minimum on both sides", () => {
    expect(validBox({ x: 10, y: 10, width: 24, height: 24 }, 100, 100)).toBe(true);
    expect(validBox({ x: 10.9, y: 10, width: 24, height: 24 }, 100, 100)).toBe(true); // 10..34 after truncation
    expect(validBox({ x: 10.5, y: 10, width: 23.4, height: 24 }, 100, 100)).toBe(false); // 10..33
    expect(validBox({ x: 90, y: 10, width: 40, height: 40 }, 100, 100)).toBe(false); // 10 px remain inside the frame
    expect(validBox({ x: -30, y: 10, width: 60, height: 40 }, 100, 100)).toBe(true);
    expect(validBox({ x: 10, y: 10, width: 30, height: 30 }, 100, 100, 32)).toBe(false);
  });

  it("validDetections drops the boxes that are not valid on the analyzed frame", () => {
    const out = Float32Array.from([50, 50, 60, 60, 0.9, 1, 0, 200, 50, 10, 60, 0.8, 1, 0]);
    expect(validDetections(out, 7, 1, 400, 300).map((d) => d.aabb.x)).toEqual([20]);
  });

  it("clipToFrame leaves a box inside the frame untouched and cuts one that reaches outside", () => {
    const inside = { aabb: { x: 10, y: 10, width: 50, height: 50 }, rect: rectOf({ x: 10, y: 10, width: 50, height: 50 }), score: 0.9, cls: 0 };
    expect(clipToFrame(inside, { width: 100, height: 100 })).toBe(inside);
    const over = { ...inside, aabb: { x: -20, y: 80, width: 50, height: 50 } };
    const clipped = clipToFrame(over, { width: 100, height: 100 });
    expect(clipped.aabb).toEqual({ x: 0, y: 80, width: 30, height: 20 });
    expect(clipped.rect).toEqual({ cx: 15, cy: 90, w: 30, h: 20, angle: 0 });
    expect([clipped.score, clipped.cls]).toEqual([0.9, 0]);
  });

  it("toFrameCorners scales to another frame of the same picture and clips", () => {
    expect(toFrameCorners({ x: 100, y: 50, width: 700, height: 100 }, 720, 405, 1440, 810)).toEqual([200, 100, 1440, 300]);
  });

  it("only the face class goes on to the recognizer", () => {
    const { faces, others } = splitByClass([{ cls: 0 }, { cls: 1 }, { cls: 0 }, { cls: 2 }]);
    expect([faces.length, others.map((o) => o.cls)]).toEqual([2, [1, 2]]);
  });
});
