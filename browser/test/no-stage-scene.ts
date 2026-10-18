// SPDX-License-Identifier: Apache-2.0
// A fixed scene for the pipeline without a geometric stage: a frame of coloured boxes, a fake
// detector that proposes them, and a fake embedder whose embedding of a crop is the mean of
// each of its three channels. `fixtures/no-stage-golden.json` holds what the pipeline gave for
// it before the stage existed (written with this file by the code of commit 2d1d775); the test
// checks that the pipeline, with no stage set, still gives exactly that.

import { rectOf } from "../src/boxes.js";
import type { FrameDetector } from "../src/detector/detector.js";
import type { BatchEmbedder } from "../src/embedder.js";
import { Pipeline } from "../src/pipeline.js";

const WIDTH = 320;
const HEIGHT = 240;

/** Boxes: x, y, width, height, colour, class (0 a face, 1 a back), detector score. */
const BOXES: Array<[number, number, number, number, [number, number, number], number, number]> = [
  [10, 10, 60, 84, [230, 20, 20], 0, 0.95],
  [80, 10, 60, 84, [20, 220, 30], 0, 0.9],
  [150, 10, 60, 84, [200, 180, 20], 0, 0.85],
  [155, 15, 60, 84, [200, 180, 20], 0, 0.8],
  [220, 10, 60, 84, [30, 30, 210], 0, 0.75],
  [10, 120, 84, 60, [120, 120, 120], 0, 0.7],
  [120, 120, 60, 84, [90, 40, 160], 1, 0.65],
  [200, 130, 50, 70, [240, 240, 240], 0, 0.6],
];

/** The scene's frame: grey, with each box painted its colour. */
export function sceneFrame(): { data: Uint8ClampedArray; width: number; height: number } {
  const data = new Uint8ClampedArray(WIDTH * HEIGHT * 4);
  for (let i = 0; i < data.length; i += 4) data.set([60, 60, 60, 255], i);
  for (const [x, y, w, h, rgb] of BOXES) {
    for (let r = y; r < y + h; r++) for (let c = x; c < x + w; c++) data.set([...rgb, 255], (r * WIDTH + c) * 4);
  }
  return { data, width: WIDTH, height: HEIGHT };
}

/** The pipeline of the scene, with no geometric stage given. */
export function scenePipeline(): Pipeline {
  const detector = {
    lastTimings: null,
    propose: async () =>
      BOXES.map(([x, y, width, height, , cls, score]) => {
        const aabb = { x, y, width, height };
        return { aabb, rect: rectOf(aabb), score, cls };
      }),
  } as unknown as FrameDetector;
  const embedder: BatchEmbedder = {
    embedBatch: async (tensors: Float32Array, n: number) => {
      const plane = tensors.length / n / 3;
      const out = new Float32Array(n * 4);
      for (let i = 0; i < n; i++) {
        for (let ch = 0; ch < 3; ch++) {
          let sum = 0;
          for (let k = 0; k < plane; k++) sum += tensors[(i * 3 + ch) * plane + k]!;
          out[i * 4 + ch] = sum / plane;
        }
        out[i * 4 + 3] = 0.6;
      }
      return out;
    },
  };
  const ids = ["A", "B", "C", "D"];
  const matrix = new Float32Array(16);
  for (let r = 0; r < 4; r++) matrix[r * 4 + r] = 1;
  return new Pipeline({
    detector,
    embedder,
    bank: { ids, matrix, dim: 4, version: 1, model: "fake", preprocessing: "full128", tau: null },
    names: new Map([["A", "Alder"], ["B", "Birch"], ["C", "Cedar"], ["D", "Dogwood"]]),
    accept: { tau: 0.62, delta: 0.05, rotationRetry: true },
    classNames: ["face", "back"],
    now: () => 0,
    minCropPx: 1,
  });
}

/** The times of the frames analyzed, in seconds: a run, then a seek. */
export const SCENE_TIMES = [0, 0.5, 1, 1.5, 2, 12];

/** Every frame's analysis, as JSON text, for the scene. */
export async function runScene(): Promise<string[]> {
  const pipeline = scenePipeline();
  const frame = sceneFrame();
  const out: string[] = [];
  for (const t of SCENE_TIMES) out.push(JSON.stringify(await pipeline.analyze(frame, t)));
  return out;
}
