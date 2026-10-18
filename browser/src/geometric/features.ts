// SPDX-License-Identifier: Apache-2.0
// Local features of a crop: keypoints and descriptors from OpenCV.js.
// In Python: cardvision/geometric/features.py (`scaled_grey`, `make_detector`, `extract`).
//
// A crop is converted to grey and scaled so that its longer side is a set number of pixels
// (area averaging to shrink, bilinear to enlarge), ORB takes its keypoints, and the strongest
// are kept by OpenCV's response, ties in the order found. The keypoints are in the scaled
// image's pixels.
//
// OpenCV itself is passed in (`Cv`): in the runtime, the build of scripts/build-opencv.mjs that
// opencv.ts loads; in a test, any build that has the functions below.

/* eslint-disable @typescript-eslint/no-explicit-any */

/** An OpenCV.js matrix, as far as the stage uses it. */
export interface CvMat {
  rows: number;
  cols: number;
  data: Uint8Array;
  empty(): boolean;
  type(): number;
  delete(): void;
}

/** The part of the OpenCV.js module the stage uses. */
export interface Cv {
  Mat: new (...args: any[]) => CvMat;
  matFromArray(rows: number, cols: number, type: number, data: ArrayLike<number>): CvMat;
  cvtColor(src: CvMat, dst: CvMat, code: number): void;
  resize(src: CvMat, dst: CvMat, size: any, fx: number, fy: number, interpolation: number): void;
  Size: new (width: number, height: number) => any;
  ORB: new (...args: any[]) => any;
  BFMatcher: new (normType: number, crossCheck: boolean) => any;
  KeyPointVector: new () => any;
  DMatchVectorVector: new () => any;
  estimateAffine2D: (...args: any[]) => CvMat;
  /** Not in this runtime's build (the "similarity" model is refused without it). */
  estimateAffinePartial2D?: (...args: any[]) => CvMat;
  /** Not in this runtime's build (the "homography" model is refused without them). */
  findHomography?: (...args: any[]) => CvMat;
  perspectiveTransform?: (src: CvMat, dst: CvMat, m: CvMat) => void;
  transform(src: CvMat, dst: CvMat, m: CvMat): void;
  isContourConvex(contour: CvMat): boolean;
  contourArea(contour: CvMat, oriented: boolean): number;
  setRNGSeed?: (seed: number) => void;
  /** OpenCV's build information, which names its version. */
  getBuildInformation?: () => string;
  CV_8UC1: number;
  CV_8UC4: number;
  CV_32FC2: number;
  COLOR_RGBA2GRAY: number;
  INTER_AREA: number;
  INTER_LINEAR: number;
  NORM_HAMMING: number;
  RANSAC: number;
  USAC_MAGSAC?: number;
  ORB_HARRIS_SCORE: number;
}

/** The keypoints of one crop: positions in the scaled image as float32 pairs, ORB descriptors
 *  (32 bytes each), and the scaled image's size. */
export interface CropFeatures {
  n: number;
  xy: Float32Array;
  desc: Uint8Array;
  size: [number, number];
}

/** Python's `round(x)` to a whole number: a tie goes to the even neighbour. */
export function pyRoundHalfEven(x: number): number {
  const r = Math.round(x);
  return r - x === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** The size a `width` x `height` image is scaled to so that its longer side is `longSide`, and
 *  whether it is scaled at all (Python: `scaled_grey`). */
export function scaledSize(width: number, height: number, longSide: number): { size: [number, number]; scale: number } {
  const scale = longSide / Math.max(height, width);
  if (scale === 1) return { size: [width, height], scale };
  return { size: [Math.max(1, pyRoundHalfEven(width * scale)), Math.max(1, pyRoundHalfEven(height * scale))], scale };
}

/** An RGBA crop in grey, scaled so its longer side is `longSide`. The caller deletes it. */
export function scaledGrey(cv: Cv, rgba: Uint8Array | Uint8ClampedArray, width: number, height: number, longSide: number): CvMat {
  const src = new cv.Mat(height, width, cv.CV_8UC4);
  const grey = new cv.Mat();
  try {
    src.data.set(rgba.subarray(0, width * height * 4));
    cv.cvtColor(src, grey, cv.COLOR_RGBA2GRAY);
  } finally {
    src.delete();
  }
  const { size, scale } = scaledSize(width, height, longSide);
  if (scale === 1) return grey;
  const out = new cv.Mat();
  try {
    cv.resize(grey, out, new cv.Size(size[0], size[1]), 0, 0, scale < 1 ? cv.INTER_AREA : cv.INTER_LINEAR);
  } catch (error) {
    out.delete();
    throw error;
  } finally {
    grey.delete();
  }
  return out;
}

/** The ORB detector with the settings of `features.params` (Python: `make_detector`), every
 *  argument of `cv::ORB::create` given; the ones Python leaves out take OpenCV's defaults
 *  (first level 0, WTA_K 2, Harris score). */
export function makeOrb(cv: Cv, maxFeatures: number, params: Readonly<Record<string, number>>): any {
  return new cv.ORB(
    maxFeatures, params.scaleFactor!, params.nLevels!, params.edgeThreshold!, 0, 2, cv.ORB_HARRIS_SCORE,
    params.patchSize!, params.fastThreshold!,
  );
}

/** Order of keypoints by response, strongest first, ties in the order found, at most `max`. */
export function strongestFirst(responses: ArrayLike<number>, max: number): number[] {
  const order = Array.from({ length: responses.length }, (_, i) => i);
  order.sort((a, b) => responses[b]! - responses[a]!);
  return order.slice(0, max);
}

/**
 * The strongest `maxFeatures` ORB keypoints of an RGBA image scaled to `longSide` (Python:
 * `extract` for `orb` with no mask).
 */
export function extractOrb(
  cv: Cv,
  rgba: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
  longSide: number,
  maxFeatures: number,
  params: Readonly<Record<string, number>>,
): CropFeatures {
  const grey = scaledGrey(cv, rgba, width, height, longSide);
  try {
    return orbOfGrey(cv, grey, maxFeatures, params);
  } finally {
    grey.delete();
  }
}

/** `extractOrb` from the scaled grey image itself (`scaledGrey`'s output, which the caller
 *  still owns and deletes) (Python: `extract_grey`). */
export function orbOfGrey(cv: Cv, grey: CvMat, maxFeatures: number, params: Readonly<Record<string, number>>): CropFeatures {
  const size: [number, number] = [grey.cols, grey.rows];
  const orb = makeOrb(cv, maxFeatures, params);
  const keypoints = new cv.KeyPointVector();
  const descriptors = new cv.Mat();
  const noMask = new cv.Mat();
  try {
    orb.detectAndCompute(grey, noMask, keypoints, descriptors, false);
    const count: number = keypoints.size();
    if (count === 0 || descriptors.empty() || descriptors.rows === 0) return { n: 0, xy: new Float32Array(0), desc: new Uint8Array(0), size };
    const width32 = descriptors.cols;
    const responses = new Float64Array(count);
    const points = new Float32Array(count * 2);
    for (let i = 0; i < count; i++) {
      const kp = keypoints.get(i);
      responses[i] = kp.response;
      points[2 * i] = kp.pt.x;
      points[2 * i + 1] = kp.pt.y;
    }
    const order = strongestFirst(responses, maxFeatures);
    const xy = new Float32Array(order.length * 2);
    const desc = new Uint8Array(order.length * width32);
    const raw = descriptors.data;
    order.forEach((k, i) => {
      xy[2 * i] = points[2 * k]!;
      xy[2 * i + 1] = points[2 * k + 1]!;
      desc.set(raw.subarray(k * width32, (k + 1) * width32), i * width32);
    });
    return { n: order.length, xy, desc, size };
  } finally {
    orb.delete();
    keypoints.delete();
    descriptors.delete();
    noMask.delete();
  }
}
