// SPDX-License-Identifier: Apache-2.0
// Boxes in analysis-frame pixels, their overlap, and greedy suppression.

/** An axis-aligned box: top-left corner and size, in pixels of the analyzed frame. */
export interface AABB {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** A box as a center, a size and an angle in degrees. The detector's boxes are axis-aligned, so
 *  the angle is always 0; the shape is kept so that a detector with an angle output can fill it. */
export interface Rect {
  cx: number;
  cy: number;
  w: number;
  h: number;
  angle: number;
}

/** One box proposed by the detector, with the detector's score and class. */
export interface Candidate {
  aabb: AABB;
  rect: Rect;
  score: number;
  cls: number;
}

/** An axis-aligned box as a `Rect` (angle 0). */
export function rectOf(b: AABB): Rect {
  return { cx: b.x + b.width / 2, cy: b.y + b.height / 2, w: b.width, h: b.height, angle: 0 };
}

/** Intersection over union of two boxes; 0 when they do not overlap. */
export function iou(a: AABB, b: AABB): number {
  const ix0 = Math.max(a.x, b.x);
  const iy0 = Math.max(a.y, b.y);
  const ix1 = Math.min(a.x + a.width, b.x + b.width);
  const iy1 = Math.min(a.y + a.height, b.y + b.height);
  if (ix1 <= ix0 || iy1 <= iy0) return 0;
  const inter = (ix1 - ix0) * (iy1 - iy0);
  return inter / (a.width * a.height + b.width * b.height - inter);
}

/**
 * Greedy suppression: items are visited by descending score (equal scores keep their input
 * order) and an item is kept unless it overlaps an already kept one by `iouThreshold` or more.
 * Returns the kept items, best first.
 */
export function nonMaxSuppression<T>(
  items: readonly T[],
  aabbOf: (t: T) => AABB,
  scoreOf: (t: T) => number,
  iouThreshold: number,
): T[] {
  const order = [...items].sort((a, b) => scoreOf(b) - scoreOf(a));
  const kept: T[] = [];
  for (const item of order) {
    const box = aabbOf(item);
    if (kept.every((k) => iou(box, aabbOf(k)) < iouThreshold)) kept.push(item);
  }
  return kept;
}
