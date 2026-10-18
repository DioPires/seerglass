// SPDX-License-Identifier: Apache-2.0
// OpenCV.js as the geometric stage calls it: a toy image and a shifted copy, through grey, ORB,
// a k=2 Hamming match with the ratio test and estimateAffine2D, and the plausibility helpers.
//
// The module is not kept in the repository. These tests run when it has been built
// (`node scripts/build-opencv.mjs --work FOLDER`, which writes vendor/opencv/) and are skipped,
// with that said in their name, when it has not.
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";

import { describe, expect, it } from "vitest";

import { loadOpenCV, opencvVersion, type CvMat, type OpenCV } from "../src/geometric/opencv.js";

const MODULE = fileURLToPath(new URL("../vendor/opencv/opencv.mjs", import.meta.url));
const BUILT = existsSync(MODULE);
const TITLE = BUILT ? "OpenCV.js" : "OpenCV.js (skipped: not built; run node scripts/build-opencv.mjs --work FOLDER)";

const W = 160;
const H = 120;
const SHIFT = { x: 7, y: 4 };

/** A toy RGBA picture: blocks of grey levels from a fixed generator, shifted by `dx`, `dy`. */
function toyImage(dx: number, dy: number): Uint8Array {
  let state = 12345;
  const next = (n: number): number => {
    state = (Math.imul(state, 1103515245) + 12345) >>> 0;
    return (state >>> 8) % n;
  };
  const grey = new Uint8Array(W * H).fill(90);
  for (let k = 0; k < 60; k++) {
    const x0 = next(W), y0 = next(H), w = 4 + next(18), h = 4 + next(18), tone = next(256);
    for (let y = y0; y < Math.min(H, y0 + h); y++) for (let x = x0; x < Math.min(W, x0 + w); x++) grey[y * W + x] = tone;
  }
  const rgba = new Uint8Array(W * H * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const sx = x - dx, sy = y - dy;
      const v = sx >= 0 && sy >= 0 && sx < W && sy < H ? grey[sy * W + sx]! : 90;
      rgba.set([v, v, v, 255], (y * W + x) * 4);
    }
  }
  return rgba;
}

function features(cv: OpenCV, rgba: Uint8Array): { xy: Float32Array; desc: CvMat; n: number } {
  const src = cv.matFromArray(H, W, cv.CV_8UC4, rgba);
  const grey = new cv.Mat();
  cv.cvtColor(src, grey, cv.COLOR_RGBA2GRAY);
  // The stage's settings (cardvision.geometric.config FEATURE_PARAMS["orb"], 500 for a crop).
  const orb = new cv.ORB(500, 1.2, 8, 15, 0, 2, cv.ORB_HARRIS_SCORE, 15, 10);
  const keypoints = new cv.KeyPointVector();
  const desc = new cv.Mat();
  const mask = new cv.Mat();
  orb.detectAndCompute(grey, mask, keypoints, desc, false);
  const n = keypoints.size();
  const xy = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    const point = keypoints.get(i).pt;
    xy[2 * i] = point.x;
    xy[2 * i + 1] = point.y;
  }
  for (const done of [src, grey, orb, keypoints, mask]) done.delete();
  return { xy, desc, n };
}

describe.skipIf(!BUILT)(TITLE, () => {
  it("loads the pinned version, once per URL", async () => {
    const url = pathToFileURL(MODULE).href;
    const cv = await loadOpenCV(url);
    expect(opencvVersion(cv)).toBe("5.0.0");
    expect(await loadOpenCV(url)).toBe(cv);
  });

  it("finds the shift between a toy image and its copy with ORB, knnMatch and estimateAffine2D", async () => {
    const cv = await loadOpenCV(pathToFileURL(MODULE).href);
    const card = features(cv, toyImage(0, 0));
    const crop = features(cv, toyImage(SHIFT.x, SHIFT.y));
    expect(card.n).toBeGreaterThan(30);
    expect(crop.desc.cols).toBe(32);

    const matcher = new cv.BFMatcher(cv.NORM_HAMMING, false);
    const pairs = new cv.DMatchVectorVector();
    matcher.knnMatch(crop.desc, card.desc, pairs, 2);
    const src: number[] = [];
    const dst: number[] = [];
    for (let i = 0; i < pairs.size(); i++) {
      const pair = pairs.get(i);
      if (pair.size() === 2 && pair.get(0).distance < 0.8 * pair.get(1).distance) {
        const match = pair.get(0);
        src.push(card.xy[2 * match.trainIdx]!, card.xy[2 * match.trainIdx + 1]!);
        dst.push(crop.xy[2 * match.queryIdx]!, crop.xy[2 * match.queryIdx + 1]!);
      }
      pair.delete();
    }
    const good = src.length / 2;
    expect(good).toBeGreaterThan(15);

    const from = cv.matFromArray(good, 1, cv.CV_32FC2, src);
    const to = cv.matFromArray(good, 1, cv.CV_32FC2, dst);
    const inliers = new cv.Mat();
    const model = cv.estimateAffine2D(from, to, inliers, cv.RANSAC, 6, 2000, 0.995, 10);
    expect(model.empty()).toBe(false);
    const [a, b, tx, c, d, ty] = Array.from({ length: 6 }, (_, k) => model.doubleAt(k));
    [a, b, c, d].forEach((value, k) => expect(Math.abs(value! - [1, 0, 0, 1][k]!)).toBeLessThan(0.01));
    expect(Math.abs(tx! - SHIFT.x)).toBeLessThan(0.5);
    expect(Math.abs(ty! - SHIFT.y)).toBeLessThan(0.5);
    expect(inliers.data.reduce((sum, v) => sum + v, 0)).toBeGreaterThanOrEqual(15);

    // The plausibility check: the card's corners mapped by the model, convex, of the card's area.
    const corners = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, W, 0, W, H, 0, H]);
    const mapped = new cv.Mat();
    cv.transform(corners, mapped, model);
    expect(cv.isContourConvex(mapped)).toBe(true);
    expect(Math.abs(cv.contourArea(mapped, false)) / (W * H)).toBeCloseTo(1, 2);

    for (const done of [card.desc, crop.desc, matcher, pairs, from, to, inliers, model, corners, mapped]) done.delete();
  });

  it("resizes with INTER_AREA and INTER_LINEAR to the size asked", async () => {
    const cv = await loadOpenCV(pathToFileURL(MODULE).href);
    const grey = cv.matFromArray(H, W, cv.CV_8UC1, new Uint8Array(W * H).fill(77));
    for (const [interpolation, size] of [[cv.INTER_AREA, [80, 60]], [cv.INTER_LINEAR, [213, 160]]] as const) {
      const out = new cv.Mat();
      cv.resize(grey, out, new cv.Size(size[0], size[1]), 0, 0, interpolation);
      expect([out.cols, out.rows]).toEqual(size);
      expect(out.data.every((v) => v === 77)).toBe(true);
      out.delete();
    }
    grey.delete();
  });
});

describe("scripts/build-opencv.mjs", () => {
  const SCRIPT = fileURLToPath(new URL("../scripts/build-opencv.mjs", import.meta.url));
  const run = (...args: string[]) => spawnSync(process.execPath, [SCRIPT, ...args], { encoding: "utf-8" });

  it("prints its options for --help, and refuses a run with no --work, an unknown option or a work folder in the repository", () => {
    const help = run("--help");
    expect(help.status).toBe(0);
    expect(help.stdout).toContain("Usage: node scripts/build-opencv.mjs --work FOLDER [--out FOLDER] [--update-lock]");
    expect(run().stderr).toMatch(/^--work is required\n/);
    expect(run("--work").stderr).toMatch(/^--work needs a value\n/);
    expect(run("--fast").stderr).toMatch(/^unknown option --fast\n/);
    const inside = run("--work", fileURLToPath(new URL("../vendor/opencv-build", import.meta.url)));
    expect([inside.status, inside.stderr]).toEqual([2, expect.stringContaining("choose a folder outside the repository")]);
  });
});
