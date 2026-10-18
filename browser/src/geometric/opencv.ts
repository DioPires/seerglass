// SPDX-License-Identifier: Apache-2.0
// The one place OpenCV.js is loaded, for the geometric stage.
//
// The module is the build of scripts/build-opencv.mjs (pinned in opencv.lock.json: OpenCV
// 5.0.0, the version the Python package's descriptor index records): an ES module whose default
// export is OpenCV's factory, with its WebAssembly (opencv_js.wasm) beside it. It is imported
// at run time from a URL, not bundled, so a page or worker that never runs the stage never
// fetches it. The same call works in a worker, on a page and under Node (the tests).
//
// The build exposes only what the stage calls (scripts/opencv-whitelist.py); the interface
// below names that surface. Every Mat and vector made through it lives in the WebAssembly heap
// and must be freed with delete().

/** A matrix in the WebAssembly heap. `data` (its bytes) aliases the heap: copy before keeping. */
export interface CvMat {
  readonly rows: number;
  readonly cols: number;
  readonly data: Uint8Array;
  /** Element `index` (row-major, counting channels) of a CV_32F matrix. */
  floatAt(index: number): number;
  /** Element `index` (row-major, counting channels) of a CV_64F matrix. */
  doubleAt(index: number): number;
  type(): number;
  empty(): boolean;
  delete(): void;
}

export interface CvVector<T> {
  size(): number;
  get(index: number): T;
  delete(): void;
}

export interface CvKeyPoint {
  pt: { x: number; y: number };
  size: number;
  angle: number;
  response: number;
  octave: number;
  class_id: number;
}

export interface CvDMatch {
  queryIdx: number;
  trainIdx: number;
  imgIdx: number;
  distance: number;
}

export interface CvORB {
  detectAndCompute(image: CvMat, mask: CvMat, keypoints: CvVector<CvKeyPoint>, descriptors: CvMat, useProvidedKeypoints: boolean): void;
  descriptorSize(): number;
  delete(): void;
}

export interface CvBFMatcher {
  knnMatch(queryDescriptors: CvMat, trainDescriptors: CvMat, matches: CvVector<CvVector<CvDMatch>>, k: number): void;
  delete(): void;
}

export interface CvSize {
  width: number;
  height: number;
}

/** The OpenCV.js surface the geometric stage uses (see scripts/opencv-whitelist.py). */
export interface OpenCV {
  Mat: { new (): CvMat; new (rows: number, cols: number, type: number): CvMat };
  Size: new (width: number, height: number) => CvSize;
  KeyPointVector: new () => CvVector<CvKeyPoint>;
  DMatchVectorVector: new () => CvVector<CvVector<CvDMatch>>;
  /** `cv::ORB::create`, every argument given: nfeatures, scaleFactor, nlevels, edgeThreshold,
   * firstLevel, WTA_K, scoreType, patchSize, fastThreshold. */
  ORB: new (
    nfeatures: number, scaleFactor: number, nlevels: number, edgeThreshold: number, firstLevel: number,
    WTA_K: number, scoreType: number, patchSize: number, fastThreshold: number,
  ) => CvORB;
  BFMatcher: new (normType: number, crossCheck: boolean) => CvBFMatcher;
  matFromArray(rows: number, cols: number, type: number, array: ArrayLike<number>): CvMat;
  cvtColor(src: CvMat, dst: CvMat, code: number): void;
  resize(src: CvMat, dst: CvMat, size: CvSize, fx: number, fy: number, interpolation: number): void;
  /** Returns the 2×3 CV_64F matrix (empty when no model was found); `inliers` gets the mask. */
  estimateAffine2D(
    from: CvMat, to: CvMat, inliers: CvMat, method: number, ransacReprojThreshold: number,
    maxIters: number, confidence: number, refineIters: number,
  ): CvMat;
  transform(src: CvMat, dst: CvMat, m: CvMat): void;
  isContourConvex(contour: CvMat): boolean;
  contourArea(contour: CvMat, oriented: boolean): number;
  setRNGSeed(seed: number): void;
  getBuildInformation(): string;
  readonly CV_8UC1: number;
  readonly CV_8UC4: number;
  readonly CV_32FC2: number;
  readonly CV_64FC1: number;
  readonly COLOR_RGBA2GRAY: number;
  readonly INTER_AREA: number;
  readonly INTER_LINEAR: number;
  readonly NORM_HAMMING: number;
  readonly RANSAC: number;
  readonly ORB_HARRIS_SCORE: number;
  readonly ORB_FAST_SCORE: number;
}

export interface LoadOpenCVOptions {
  /** The WebAssembly's bytes, when the caller has fetched (and checked) them itself; otherwise
   * the module fetches `opencv_js.wasm` from its own folder. */
  wasmBinary?: ArrayBuffer | Uint8Array;
}

type Factory = (moduleArg?: Record<string, unknown>) => Promise<OpenCV>;

const loaded = new Map<string, Promise<OpenCV>>();

/**
 * OpenCV.js from the ES module at `moduleUrl` (scripts/build-opencv.mjs's `opencv.mjs`),
 * initialized. One module per URL is loaded and kept; a failed load is forgotten so that a later
 * call tries again.
 */
export function loadOpenCV(moduleUrl: string | URL, options: LoadOpenCVOptions = {}): Promise<OpenCV> {
  const key = String(moduleUrl);
  let pending = loaded.get(key);
  if (pending === undefined) {
    pending = (async () => {
      const imported = (await import(/* @vite-ignore */ key)) as { default?: unknown };
      if (typeof imported.default !== "function") throw new Error(`${key}: not an OpenCV.js ES module (no default export)`);
      const moduleArg: Record<string, unknown> = {};
      if (options.wasmBinary !== undefined) moduleArg.wasmBinary = options.wasmBinary;
      const cv = await (imported.default as Factory)(moduleArg);
      if (typeof cv.ORB !== "function" || typeof cv.estimateAffine2D !== "function") {
        throw new Error(`${key}: this OpenCV.js lacks the functions the geometric stage needs`);
      }
      return cv;
    })();
    pending.catch(() => loaded.delete(key));
    loaded.set(key, pending);
  }
  return pending;
}

/** The OpenCV version a loaded module was built from ("5.0.0"), read from its build information. */
export function opencvVersion(cv: OpenCV): string {
  return /Version control:\s*(\S+)/.exec(cv.getBuildInformation())?.[1] ?? "";
}
