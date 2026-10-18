// SPDX-License-Identifier: Apache-2.0
// Pure geometry helpers for the box editor: polygon IoU, row distribution (hand fans and rows
// of cards on a board), insert/remove-and-redistribute, "same layout as…" group matching, and
// the prefill / carry-over merge. No DOM, no state: shared by the page (plain <script> tag,
// window.LabelerTools) and the Node tests (test_labeltools.js).
//
// rect = {cx, cy, w, h, angle} in frame pixels, angle in degrees, y-down (a positive angle is
// clockwise on screen), the same convention as boxgeom.js. A landscape card is a portrait w/h
// box at angle ±90.
(function (root) {
  "use strict";

  // ---- geometry --------------------------------------------------------------

  function rotatePoint(dx, dy, angleDeg) {
    const a = (angleDeg * Math.PI) / 180;
    const c = Math.cos(a), s = Math.sin(a);
    return { x: dx * c - dy * s, y: dx * s + dy * c };
  }

  // Corners clockwise from the card's top-left as rendered: [[x,y] × 4].
  function rectCorners(rect) {
    const hw = rect.w / 2, hh = rect.h / 2;
    return [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]].map(([x, y]) => {
      const r = rotatePoint(x, y, rect.angle || 0);
      return [rect.cx + r.x, rect.cy + r.y];
    });
  }

  function polygonArea(p) {
    let a = 0;
    for (let i = 0; i < p.length; i++) {
      const [x0, y0] = p[i], [x1, y1] = p[(i + 1) % p.length];
      a += x0 * y1 - x1 * y0;
    }
    return Math.abs(a) / 2;
  }

  // Sutherland–Hodgman clip of `subject` by the convex polygon `clip` (either winding).
  function clipPolygon(subject, clip) {
    const sign = (() => {
      let a = 0;
      for (let i = 0; i < clip.length; i++) {
        const [x0, y0] = clip[i], [x1, y1] = clip[(i + 1) % clip.length];
        a += x0 * y1 - x1 * y0;
      }
      return a >= 0 ? 1 : -1;
    })();
    let out = subject;
    for (let i = 0; i < clip.length && out.length; i++) {
      const [ax, ay] = clip[i], [bx, by] = clip[(i + 1) % clip.length];
      const inside = ([x, y]) => sign * ((bx - ax) * (y - ay) - (by - ay) * (x - ax)) >= -1e-9;
      const cross = ([px, py], [qx, qy]) => {
        const d1x = qx - px, d1y = qy - py, d2x = bx - ax, d2y = by - ay;
        const den = d1x * d2y - d1y * d2x;
        const t = den === 0 ? 0 : ((ax - px) * d2y - (ay - py) * d2x) / den;
        return [px + t * d1x, py + t * d1y];
      };
      const input = out;
      out = [];
      for (let j = 0; j < input.length; j++) {
        const cur = input[j], prev = input[(j + input.length - 1) % input.length];
        if (inside(cur)) {
          if (!inside(prev)) out.push(cross(prev, cur));
          out.push(cur);
        } else if (inside(prev)) {
          out.push(cross(prev, cur));
        }
      }
    }
    return out;
  }

  // IoU of two convex polygons (rotated boxes: pass rectCorners()).
  function polygonIoU(p, q) {
    const inter = clipPolygon(p, q);
    const ia = inter.length >= 3 ? polygonArea(inter) : 0;
    const u = polygonArea(p) + polygonArea(q) - ia;
    return u > 0 ? ia / u : 0;
  }

  function rectIoU(a, b) {
    return polygonIoU(rectCorners(a), rectCorners(b));
  }

  function aabbOf(rect) {
    const c = rectCorners(rect);
    const xs = c.map((p) => p[0]), ys = c.map((p) => p[1]);
    const x = Math.min(...xs), y = Math.min(...ys);
    return { x, y, w: Math.max(...xs) - x, h: Math.max(...ys) - y };
  }

  // Axis-aligned IoU of the two boxes' bounding boxes (the scorer's IoU).
  function aabbIoU(a, b) {
    const A = aabbOf(a), B = aabbOf(b);
    const x0 = Math.max(A.x, B.x), y0 = Math.max(A.y, B.y);
    const x1 = Math.min(A.x + A.w, B.x + B.w), y1 = Math.min(A.y + A.h, B.y + B.h);
    if (x1 <= x0 || y1 <= y0) return 0;
    const inter = (x1 - x0) * (y1 - y0);
    const u = A.w * A.h + B.w * B.h - inter;
    return u > 0 ? inter / u : 0;
  }

  function normalizeAngle(a) {
    a = a % 360;
    if (a > 180) a -= 360;
    if (a < -180) a += 360;
    return a;
  }

  // Shortest-way interpolation between two angles in degrees.
  function lerpAngle(a, b, t) {
    const d = normalizeAngle(b - a);
    return normalizeAngle(a + d * t);
  }

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  // ---- ordering ----------------------------------------------------------------

  // Indices sorted along the axis of larger spread (x for a row, y for a column).
  function mainAxisOrder(rects) {
    if (!rects.length) return [];
    const xs = rects.map((r) => r.cx), ys = rects.map((r) => r.cy);
    const spreadX = Math.max(...xs) - Math.min(...xs), spreadY = Math.max(...ys) - Math.min(...ys);
    const key = spreadX >= spreadY ? "cx" : "cy";
    return rects.map((_, i) => i).sort((i, j) => rects[i][key] - rects[j][key] || i - j);
  }

  // Reading order: rows top to bottom (a new row starts when the center drops by more than half
  // the median card height), left to right within a row. Used to pair two groups card by card.
  function readingOrder(rects) {
    if (!rects.length) return [];
    const hs = rects.map((r) => Math.min(r.w, r.h)).sort((a, b) => a - b);
    const tol = 0.5 * Math.max(1, hs[Math.floor(hs.length / 2)] * 1.4);
    const byY = rects.map((_, i) => i).sort((i, j) => rects[i].cy - rects[j].cy || i - j);
    const rows = [];
    for (const i of byY) {
      const row = rows[rows.length - 1];
      if (row && rects[i].cy - row.cy <= tol) {
        row.items.push(i);
        row.cy = row.items.reduce((s, k) => s + rects[k].cy, 0) / row.items.length;
      } else {
        rows.push({ cy: rects[i].cy, items: [i] });
      }
    }
    return rows.flatMap((r) => r.items.sort((i, j) => rects[i].cx - rects[j].cx || i - j));
  }

  // ---- distribution (hand fans, rows on a board) ---------------------------------

  const FAN_MIN_DEG = 1.5; // end angles closer than this: a straight row
  const FAN_MAX_DEG = 25; // further apart: not a fan (e.g. a tapped card at one end of a row)

  function isFan(first, last) {
    const da = Math.abs(normalizeAngle((last.angle || 0) - (first.angle || 0)));
    return da >= FAN_MIN_DEG && da <= FAN_MAX_DEG;
  }

  // The pivot of a fan: where the two cards' local vertical axes meet. Returns null when the
  // row is not a fan or the pivot is implausibly far (then the row is straight).
  function fanPivot(first, last) {
    if (!isFan(first, last)) return null;
    const u0 = rotatePoint(0, 1, first.angle || 0), u1 = rotatePoint(0, 1, last.angle || 0);
    // first.c + s0*u0 = last.c + s1*u1
    const det = u0.x * -u1.y - -u1.x * u0.y;
    if (Math.abs(det) < 1e-9) return null;
    const dx = last.cx - first.cx, dy = last.cy - first.cy;
    const s0 = (dx * -u1.y - -u1.x * dy) / det;
    const s1 = (u0.x * dy - u0.y * dx) / det;
    const px = first.cx + s0 * u0.x, py = first.cy + s0 * u0.y;
    const maxR = 40 * Math.max(first.h, last.h, first.w, last.w);
    if (!Number.isFinite(s0) || !Number.isFinite(s1) || Math.abs(s0) > maxR || Math.abs(s1) > maxR) return null;
    if (Math.sign(s0) !== Math.sign(s1)) return null;
    return { x: px, y: py, r0: s0, r1: s1 };
  }

  // The row through `first` (t = 0) and `last` (t = 1) evaluated at parameters `ts` (any real
  // t: values outside [0, 1] extend the row past its ends at the same spacing). A fan (end
  // angles 1.5–25° apart): size and angle interpolated, centers on the arc about the common
  // pivot. Any other row: centers on the straight line; a slot with an entry in `own` (the card
  // that takes it) keeps that card's own size and angle (a tapped unit stays tapped), a new
  // slot takes interpolated ones.
  function rowAt(first, last, ts, own) {
    const pivot = fanPivot(first, last);
    const fan = isFan(first, last);
    return ts.map((t, i) => {
      const mine = !fan && own && own[i] ? own[i] : null;
      const da = normalizeAngle((last.angle || 0) - (first.angle || 0));
      const angle = mine ? mine.angle || 0 : normalizeAngle((first.angle || 0) + da * t);
      const w = mine ? mine.w : lerp(first.w, last.w, t), h = mine ? mine.h : lerp(first.h, last.h, t);
      let cx, cy;
      if (pivot) {
        const r = lerp(pivot.r0, pivot.r1, t);
        const u = rotatePoint(0, 1, normalizeAngle((first.angle || 0) + da * t));
        cx = pivot.x - r * u.x;
        cy = pivot.y - r * u.y;
      } else {
        cx = lerp(first.cx, last.cx, t);
        cy = lerp(first.cy, last.cy, t);
      }
      return { cx, cy, w, h, angle };
    });
  }

  // n rects evenly spaced from `first` to `last` inclusive (n ≥ 2); see rowAt.
  function distributeBetween(first, last, n, own) {
    if (n <= 0) return [];
    if (n === 1) return [{ ...first }];
    const ts = [];
    for (let i = 0; i < n; i++) ts.push(i / (n - 1));
    return rowAt(first, last, ts, own);
  }

  // `g`: re-space the given rects evenly between the first and last along the row. Returns
  // {order, rects}: order[k] is the input index that takes slot k (left to right), rects[k] its
  // new geometry. Each card keeps its identity (its slot follows its current order).
  function distribute(rects) {
    const order = mainAxisOrder(rects);
    if (order.length < 3) return { order, rects: order.map((i) => ({ ...rects[i] })) };
    const first = rects[order[0]], last = rects[order[order.length - 1]];
    return { order, rects: distributeBetween(first, last, order.length, order.map((i) => rects[i])) };
  }

  // Slot parameters when a row of n cards gains (+1) or loses (−1) one. "grow" (the default):
  // the spacing and the row's center stay, both ends move out (in) by half a spacing, which is
  // what a hand of cards in a digital client usually does when a card is added or played.
  // "ends": the first and last positions stay and the spacing changes.
  function slotParams(n, delta, mode) {
    const m = n + delta;
    const ts = [];
    for (let k = 0; k < m; k++) {
      if (mode === "ends" || n < 2) ts.push(m === 1 ? 0 : k / (m - 1));
      else ts.push((k - delta / 2) / (n - 1));
    }
    return ts;
  }

  // `+`: one more slot right after input index `afterIndex` (or at the right end when it is
  // not in the row), the row re-spaced (see slotParams). Returns {order, rects, newSlot}: order
  // has n+1 entries with -1 at newSlot (the new, unassigned box).
  function insertSlot(rects, afterIndex, mode) {
    const order = mainAxisOrder(rects);
    let pos = order.indexOf(afterIndex);
    if (pos < 0) pos = order.length - 1;
    const newOrder = order.slice(0, pos + 1).concat([-1], order.slice(pos + 1));
    if (order.length === 1) {
      // a single card: the new slot goes one card-width (plus a small gap) to its right
      const a = rects[order[0]];
      const step = rotatePoint(a.w * 1.05, 0, a.angle || 0);
      const b = { ...a, cx: a.cx + step.x, cy: a.cy + step.y };
      return { order: newOrder, rects: [{ ...a }, b], newSlot: 1 };
    }
    const first = rects[order[0]], last = rects[order[order.length - 1]];
    const own = newOrder.map((i) => (i >= 0 ? rects[i] : null));
    return { order: newOrder, rects: rowAt(first, last, slotParams(order.length, 1, mode), own), newSlot: pos + 1 };
  }

  // `-`: drop input index `removeIndex` and re-space the rest (see slotParams).
  function removeSlot(rects, removeIndex, mode) {
    const order = mainAxisOrder(rects);
    const pos = order.indexOf(removeIndex);
    if (pos < 0) return { order, rects: order.map((i) => ({ ...rects[i] })) };
    const rest = order.filter((i) => i !== removeIndex);
    if (rest.length < 2) return { order: rest, rects: rest.map((i) => ({ ...rects[i] })) };
    const first = rects[order[0]], last = rects[order[order.length - 1]];
    return { order: rest, rects: rowAt(first, last, slotParams(order.length, -1, mode), rest.map((i) => rects[i])) };
  }

  // ---- groups and "same layout as…" --------------------------------------------------

  // A board group: zone plus board half (own = bottom, opponent = top).
  function groupKey(row, frameH) {
    const side = row.rect.cy < (frameH || 1080) / 2 ? "top" : "bottom";
    return `${row.zone || "other"}:${side}`;
  }

  function groupsOf(rows, frameH) {
    const m = new Map();
    rows.forEach((r, i) => {
      const k = groupKey(r, frameH);
      if (!m.has(k)) m.set(k, []);
      m.get(k).push(i);
    });
    return m;
  }

  // Mean IoU of two equally sized rect lists paired in reading order.
  function layoutSimilarity(a, b) {
    if (a.length !== b.length || !a.length) return 0;
    const oa = readingOrder(a), ob = readingOrder(b);
    let s = 0;
    for (let k = 0; k < oa.length; k++) s += rectIoU(a[oa[k]], b[ob[k]]);
    return s / oa.length;
  }

  // Candidate frames whose group `key` has exactly the current count and a different layout
  // (mean IoU < `same`), nearest in time first (earlier frames win ties). frames: [{index, t,
  // rows}] of CONFIRMED frames other than the current one.
  function layoutMatches(currentRows, key, frames, frameH, opts) {
    const o = Object.assign({ same: 0.9, limit: 4 }, opts || {});
    const cur = currentRows.filter((r) => groupKey(r, frameH) === key).map((r) => r.rect);
    if (cur.length < 2) return [];
    const out = [];
    for (const f of frames) {
      const src = f.rows.filter((r) => groupKey(r, frameH) === key).map((r) => r.rect);
      if (src.length !== cur.length) continue;
      const sim = layoutSimilarity(cur, src);
      if (sim >= o.same) continue;
      out.push({ index: f.index, t: f.t, dt: f.dt, similarity: sim });
    }
    out.sort((a, b) => Math.abs(a.dt) - Math.abs(b.dt) || a.dt - b.dt);
    return out.slice(0, o.limit);
  }

  // Pair the target group's rects with the source group's (both in reading order) and return
  // [{targetIndex, rect}] — the target keeps identities, takes the source geometry.
  function applyLayout(targetRects, sourceRects) {
    if (targetRects.length !== sourceRects.length) return null;
    const ot = readingOrder(targetRects), os = readingOrder(sourceRects);
    return ot.map((ti, k) => ({ targetIndex: ti, rect: { ...sourceRects[os[k]] } }));
  }

  // ---- prefill -----------------------------------------------------------------------

  // A prefill box [x, y, w, h] (axis-aligned, frame px) as a tool rect: a box wider than tall
  // is a sideways card, stored the tool's way (portrait w/h at angle 90).
  function rectFromBox(box) {
    const [x, y, w, h] = box;
    if (w > h * 1.05) return { cx: x + w / 2, cy: y + h / 2, w: h, h: w, angle: 90 };
    return { cx: x + w / 2, cy: y + h / 2, w, h, angle: 0 };
  }

  // Prefill boxes that carry-over does not already cover: IoU < `thr` (axis-aligned, as the
  // scorer measures) against EVERY carried box. Returns the kept prefill entries.
  function prefillSuggestions(carriedRects, prefillBoxes, thr) {
    const t = typeof thr === "number" ? thr : 0.3;
    return prefillBoxes.filter((pb) => {
      const r = rectFromBox(pb.box);
      return carriedRects.every((c) => aabbIoU(c, r) < t);
    });
  }

  // Flag a prefill entry for the annotator's eye: abstained / below τ / NMS-suppressed rows
  // have no identity; a low margin is identified but close to its runner-up.
  function prefillNeedsLook(p) {
    return !p || !p.cardId || !!p.flag;
  }

  const api = {
    rotatePoint, rectCorners, polygonArea, clipPolygon, polygonIoU, rectIoU, aabbOf, aabbIoU,
    normalizeAngle, lerpAngle, mainAxisOrder, readingOrder, isFan, fanPivot, rowAt, distributeBetween, slotParams,
    distribute, insertSlot, removeSlot, groupKey, groupsOf, layoutSimilarity, layoutMatches,
    applyLayout, rectFromBox, prefillSuggestions, prefillNeedsLook,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.LabelerTools = api;
})(typeof window !== "undefined" ? window : globalThis);
