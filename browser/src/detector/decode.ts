// SPDX-License-Identifier: Apache-2.0
// The detector's output contract: from the model's raw rows to a short list of boxes.
// In Python: cardvision/detect/decode.py. The two follow the same steps in the same precision,
// so the same rows give the same boxes bit for bit.
//
// The model emits one row per anchor, already decoded to canvas pixels and with the sigmoid
// applied:   cx, cy, w, h, objectness, class score 0, class score 1, ...
//
//   1. score of a row = float32(objectness * class score), for its best class; the class is
//      that best class (the first one on a tie);
//   2. keep the rows whose score is at least float32(threshold), in row order;
//   3. corners, each step in float32: x0 = cx - w / 2, y0 = cy - h / 2, x1 = cx + w / 2,
//      y1 = cy + h / 2;
//   4. greedy suppression in descending score order (equal scores keep row order), ignoring
//      the class: a box survives while its overlap with every kept box, computed in float32 as
//      inter / (area + area2 - inter + 1e-9), is at most float32(iou);
//   5. keep at most the first `maxProposals` survivors;
//   6. divide by the letterbox ratio in double precision to return to frame pixels:
//      x = x0 / ratio, y = y0 / ratio, width = float32(x1 - x0) / ratio, likewise the height.

import { type AABB, type Candidate, rectOf } from "../boxes.js";

/** The class whose boxes go to the recognizer. A back is reported as a card and never named. */
export const FACE_CLASS = 0;
export const BACK_CLASS = 1;

/** Columns before the class scores: cx, cy, w, h, objectness. */
export const BOX_COLUMNS = 5;

/** The operating point of the detector's decode. */
export interface DecodeSettings {
  threshold: number;
  nmsIou: number;
  maxProposals: number;
}

/** The operating point of the published detector. A release document states its own. */
export const DEFAULT_DECODE: Readonly<DecodeSettings> = { threshold: 0.3, nmsIou: 0.45, maxProposals: 80 };

/** A box whose crop is shorter than this on either side, in the analyzed frame, is not a valid
 *  proposal. A release document states its own value (`detector.minCropPx`). */
export const DEFAULT_MIN_CROP_PX = 24;

const f32 = Math.fround;
const EPS_F32 = f32(1e-9);

/**
 * Every box that survives the threshold, the suppression and the cut to `maxProposals`, in
 * frame pixels, best first. `out` is one frame's rows, `columns` values each (5 + the number of
 * classes), and `ratio` the letterbox ratio. No box is filtered for size or clipped here.
 */
export function decodeDetections(
  out: Float32Array,
  columns: number,
  ratio: number,
  settings: DecodeSettings = DEFAULT_DECODE,
): Candidate[] {
  if (!Number.isInteger(columns) || columns <= BOX_COLUMNS) throw new Error(`decodeDetections: ${columns} columns leave no class score`);
  if (out.length % columns !== 0) throw new Error(`decodeDetections: ${out.length} values is not a multiple of ${columns}`);
  const rows = out.length / columns;
  const thr = f32(settings.threshold);
  const iouThr = f32(settings.nmsIou);

  const idx: number[] = [];
  const score: number[] = [];
  const cls: number[] = [];
  for (let r = 0; r < rows; r++) {
    const o = r * columns;
    const obj = out[o + 4]!;
    let best = 0;
    let s = f32(obj * out[o + BOX_COLUMNS]!);
    for (let c = 1; c < columns - BOX_COLUMNS; c++) {
      const sc = f32(obj * out[o + BOX_COLUMNS + c]!);
      if (sc > s) {
        s = sc;
        best = c;
      }
    }
    if (s >= thr) {
      idx.push(r);
      score.push(s);
      cls.push(best);
    }
  }
  const n = idx.length;
  if (n === 0) return [];
  const bx0 = new Float32Array(n);
  const by0 = new Float32Array(n);
  const bx1 = new Float32Array(n);
  const by1 = new Float32Array(n);
  for (let k = 0; k < n; k++) {
    const o = idx[k]! * columns;
    const cx = out[o]!;
    const cy = out[o + 1]!;
    const hw = f32(out[o + 2]! / 2);
    const hh = f32(out[o + 3]! / 2);
    bx0[k] = cx - hw;
    by0[k] = cy - hh;
    bx1[k] = cx + hw;
    by1[k] = cy + hh;
  }

  let order = Array.from({ length: n }, (_, k) => k).sort((a, b) => score[b]! - score[a]! || a - b);
  const keep: number[] = [];
  while (order.length > 0) {
    const i = order[0]!;
    keep.push(i);
    if (order.length === 1) break;
    const a = f32(f32(bx1[i]! - bx0[i]!) * f32(by1[i]! - by0[i]!));
    const rest: number[] = [];
    for (let q = 1; q < order.length; q++) {
      const j = order[q]!;
      const iw = Math.max(f32(Math.min(bx1[i]!, bx1[j]!) - Math.max(bx0[i]!, bx0[j]!)), 0);
      const ih = Math.max(f32(Math.min(by1[i]!, by1[j]!) - Math.max(by0[i]!, by0[j]!)), 0);
      const inter = f32(iw * ih);
      const a2 = f32(f32(bx1[j]! - bx0[j]!) * f32(by1[j]! - by0[j]!));
      const overlap = f32(inter / f32(f32(f32(a + a2) - inter) + EPS_F32));
      if (overlap <= iouThr) rest.push(j);
    }
    order = rest;
  }

  return keep.slice(0, settings.maxProposals).map((k) => {
    const aabb = {
      x: bx0[k]! / ratio,
      y: by0[k]! / ratio,
      width: f32(bx1[k]! - bx0[k]!) / ratio,
      height: f32(by1[k]! - by0[k]!) / ratio,
    };
    return { aabb, rect: rectOf(aabb), score: score[k]!, cls: cls[k]! };
  });
}

/**
 * Whether a box's crop is at least `minPx` pixels on both sides. The test is made on the crop:
 * the box clipped to the frame, each edge cut down to a whole number.
 */
export function validBox(b: AABB, frameW: number, frameH: number, minPx: number = DEFAULT_MIN_CROP_PX): boolean {
  const x0 = Math.trunc(Math.max(0, b.x));
  const y0 = Math.trunc(Math.max(0, b.y));
  const x1 = Math.trunc(Math.min(frameW, b.x + b.width));
  const y1 = Math.trunc(Math.min(frameH, b.y + b.height));
  return x1 - x0 >= minPx && y1 - y0 >= minPx;
}

/** One frame's detections: `decodeDetections`, then only the boxes that pass `validBox`. */
export function validDetections(
  out: Float32Array,
  columns: number,
  ratio: number,
  frameW: number,
  frameH: number,
  settings: DecodeSettings = DEFAULT_DECODE,
  minPx: number = DEFAULT_MIN_CROP_PX,
): Candidate[] {
  return decodeDetections(out, columns, ratio, settings).filter((d) => validBox(d.aabb, frameW, frameH, minPx));
}

/** A box scaled from the analyzed frame to another frame of the same picture and clipped to
 *  it, as corners `[x0, y0, x1, y1]`. */
export function toFrameCorners(
  b: AABB,
  analysisW: number,
  analysisH: number,
  frameW: number,
  frameH: number,
): [number, number, number, number] {
  const sx = frameW / analysisW;
  const sy = frameH / analysisH;
  return [Math.max(0, b.x * sx), Math.max(0, b.y * sy), Math.min(frameW, (b.x + b.width) * sx), Math.min(frameH, (b.y + b.height) * sy)];
}

/** A detection with its box clipped to the frame; one that lies inside is returned untouched. */
export function clipToFrame(c: Candidate, frame: { width: number; height: number }): Candidate {
  const b = c.aabb;
  if (b.x >= 0 && b.y >= 0 && b.x + b.width <= frame.width && b.y + b.height <= frame.height) return c;
  const [x0, y0, x1, y1] = toFrameCorners(b, frame.width, frame.height, frame.width, frame.height);
  const aabb = { x: x0, y: y0, width: Math.max(0, x1 - x0), height: Math.max(0, y1 - y0) };
  return { ...c, aabb, rect: rectOf(aabb) };
}

/** Split a frame's detections: the faces, which go to the recognizer, and everything else (the
 *  backs, or a class this runtime does not know), which is reported and never named. */
export function splitByClass<T extends { cls: number }>(detections: readonly T[], faceClass: number = FACE_CLASS): { faces: T[]; others: T[] } {
  const faces: T[] = [];
  const others: T[] = [];
  for (const d of detections) (d.cls === faceClass ? faces : others).push(d);
  return { faces, others };
}
