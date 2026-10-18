// SPDX-License-Identifier: Apache-2.0
// Fitting a model to a candidate's matches with OpenCV's RANSAC, and the plausibility check on
// the fitted model. In Python: cardvision/geometric/stage.py (`_fit`, `_plausible`).
//
// The fit is OpenCV's own (`estimateAffine2D`, `findHomography`, `estimateAffinePartial2D`),
// which draws its samples from a generator with a fixed seed: there is no RANSAC of our own.

import type { GeometricSettings } from "./config.js";
import type { Cv, CvMat } from "./features.js";

/** `estimateAffine2D`'s refinement steps: OpenCV's default, as in Python. */
const REFINE_ITERS = 10;

/** The fewest matched points each model can be fitted to. */
export const NEEDED: Readonly<Record<string, number>> = { homography: 4, affine: 3, similarity: 2 };

/** Why this OpenCV cannot fit the settings' model, or null when it can. */
export function unsupportedFit(cv: Cv, ransac: GeometricSettings["ransac"]): string | null {
  if (ransac.model === "homography" && (typeof cv.findHomography !== "function" || typeof cv.perspectiveTransform !== "function")) {
    return 'this OpenCV build has no findHomography (the "homography" model)';
  }
  if (ransac.model === "similarity" && typeof cv.estimateAffinePartial2D !== "function") {
    return 'this OpenCV build has no estimateAffinePartial2D (the "similarity" model)';
  }
  if (ransac.method === "usac" && typeof cv.USAC_MAGSAC !== "number") return 'this OpenCV build has no USAC ("usac" method)';
  return null;
}

/** The corners of a `[width, height]` image, as OpenCV points. */
function corners(cv: Cv, size: readonly [number, number]): CvMat {
  const [w, h] = size;
  return cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, w, 0, w, h, 0, h]);
}

/** Whether the model maps the card image onto a convex shape of a plausible area of the crop. */
export function plausible(
  cv: Cv,
  matrix: CvMat,
  cropSize: readonly [number, number],
  cardSize: readonly [number, number],
  ransac: GeometricSettings["ransac"],
): boolean {
  const points = corners(cv, cardSize);
  const mapped = new cv.Mat();
  try {
    if (matrix.rows === 3 && matrix.cols === 3) cv.perspectiveTransform!(points, mapped, matrix);
    else cv.transform(points, mapped, matrix);
    // The four mapped corners, as float32 pairs (OpenCV's buffers are aligned for them).
    const values = new Float32Array(mapped.data.buffer, mapped.data.byteOffset, 8);
    for (let i = 0; i < 8; i++) if (!Number.isFinite(values[i]!)) return false;
    if (!cv.isContourConvex(mapped)) return false;
    const ratio = Math.abs(cv.contourArea(mapped, false)) / (cropSize[0] * cropSize[1]);
    return ransac.minAreaRatio <= ratio && ratio <= ransac.maxAreaRatio;
  } finally {
    points.delete();
    mapped.delete();
  }
}

/**
 * The inliers of the model fitted to matched points `src` (card) to `dst` (crop), each a
 * float32 array of x, y pairs in match order (Python: `_fit`). 0 when there are too few points,
 * no model is found, or the model is not plausible.
 */
export function fitInliers(
  cv: Cv,
  src: Float32Array,
  dst: Float32Array,
  cropSize: readonly [number, number],
  cardSize: readonly [number, number],
  ransac: GeometricSettings["ransac"],
): number {
  const count = src.length / 2;
  if (count < NEEDED[ransac.model]!) return 0;
  const method = ransac.method === "ransac" ? cv.RANSAC : cv.USAC_MAGSAC!;
  const from = cv.matFromArray(count, 1, cv.CV_32FC2, src);
  const to = cv.matFromArray(count, 1, cv.CV_32FC2, dst);
  const mask = new cv.Mat();
  let matrix: CvMat | null = null;
  try {
    const { threshold, maxIters, confidence } = ransac;
    if (ransac.model === "homography") matrix = cv.findHomography!(from, to, method, threshold, mask, maxIters, confidence);
    // refineIters 10 is OpenCV's default, which Python leaves in place; given, as embind wants every argument.
    else if (ransac.model === "affine") matrix = cv.estimateAffine2D(from, to, mask, method, threshold, maxIters, confidence, REFINE_ITERS);
    else matrix = cv.estimateAffinePartial2D!(from, to, mask, method, threshold, maxIters, confidence);
    if (!matrix || matrix.empty() || matrix.rows === 0 || mask.empty()) return 0;
    if (ransac.maxAreaRatio > 0 && !plausible(cv, matrix, cropSize, cardSize, ransac)) return 0;
    let inliers = 0;
    const flags = mask.data;
    for (let i = 0; i < flags.length; i++) if (flags[i] !== 0) inliers += 1;
    return inliers;
  } finally {
    from.delete();
    to.delete();
    mask.delete();
    matrix?.delete();
  }
}
