// SPDX-License-Identifier: Apache-2.0
// Tests for quadsnap.js on synthetic tables: a card = light border + textured art, drawn as an
// arbitrary quad (rotated, upside-down, strongly foreshortened) on a noisy table. Stdlib only.
// Run: node test_quadsnap.js  (also run by test_serve.py)
"use strict";
const assert = require("assert");
const QS = require("./quadsnap.js");

let seedState = 4242;
function rnd() { seedState = (seedState * 1103515245 + 12345) & 0x7fffffff; return seedState / 0x7fffffff; }

// bilinear (u, v) ∈ [0,1]² → point inside quad TL, TR, BR, BL
function inside(q, x, y) {
  // point-in-convex-quad (clockwise, y-down): all cross products ≥ 0
  for (let i = 0; i < 4; i++) {
    const a = q[i], b = q[(i + 1) % 4];
    if ((b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]) < 0) return false;
  }
  return true;
}

function distToQuadEdge(q, x, y) {
  let m = Infinity;
  for (let i = 0; i < 4; i++) {
    const a = q[i], b = q[(i + 1) % 4];
    const dx = b[0] - a[0], dy = b[1] - a[1], L = dx * dx + dy * dy;
    let t = ((x - a[0]) * dx + (y - a[1]) * dy) / L; t = Math.max(0, Math.min(1, t));
    m = Math.min(m, Math.hypot(x - a[0] - t * dx, y - a[1] - t * dy));
  }
  return m;
}

// Table image W×H with the given quads drawn (screen-clockwise corner lists).
function table(W, H, quads) {
  const data = new Uint8Array(W * H * 3);
  for (let i = 0; i < W * H; i++) {
    const n = rnd() * 10;
    data[i * 3] = 70 + n; data[i * 3 + 1] = 55 + n; data[i * 3 + 2] = 40 + n; // wood-ish
  }
  for (const q of quads) {
    const xs = q.map((p) => p[0]), ys = q.map((p) => p[1]);
    for (let y = Math.floor(Math.min(...ys)); y <= Math.ceil(Math.max(...ys)); y++) {
      for (let x = Math.floor(Math.min(...xs)); x <= Math.ceil(Math.max(...xs)); x++) {
        if (x < 0 || y < 0 || x >= W || y >= H || !inside(q, x + 0.5, y + 0.5)) continue;
        const k = (y * W + x) * 3, d = distToQuadEdge(q, x + 0.5, y + 0.5);
        if (d < 4) { data[k] = 225; data[k + 1] = 222; data[k + 2] = 215; continue; } // border
        const v = 120 + 60 * Math.sin(x * 0.31 + y * 0.17) * Math.cos(y * 0.23) + 12 * rnd();
        data[k] = v; data[k + 1] = v * 0.7; data[k + 2] = v * 0.9;
      }
    }
  }
  // 3×3 box blur: video edges are 2–3 px wide
  const out = new Uint8Array(data.length);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) for (let c = 0; c < 3; c++) {
    let s = 0, n = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const xx = x + dx, yy = y + dy;
      if (xx >= 0 && yy >= 0 && xx < W && yy < H) { s += data[(yy * W + xx) * 3 + c]; n++; }
    }
    out[(y * W + x) * 3 + c] = s / n;
  }
  return { data: out, width: W, height: H, channels: 3 };
}

function rotQuad(cx, cy, w, h, deg) {
  const r = (deg * Math.PI) / 180, c = Math.cos(r), s = Math.sin(r);
  return [[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]
    .map(([x, y]) => [cx + x * c - y * s, cy + x * s + y * c]);
}

function jitter(q, amp) { return q.map((p) => [p[0] + (rnd() * 2 - 1) * amp, p[1] + (rnd() * 2 - 1) * amp]); }
function maxErr(a, b) { return Math.max(...a.map((p, i) => Math.hypot(p[0] - b[i][0], p[1] - b[i][1]))); }

// Truth quads (card order, clockwise on screen). The card's drawn outer edge IS the quad.
const upright = rotQuad(150, 160, 90, 126, 7);
const upsideDown = rotQuad(360, 170, 90, 126, 183); // card order starts at screen bottom-right
const foreshortened = [[480, 80], [600, 90], [640, 250], [440, 240]]; // strong taper
const img = table(700, 320, [upright, upsideDown, foreshortened]);

let worst = 0, n = 0;
for (const truth of [upright, upsideDown, foreshortened]) {
  for (let trial = 0; trial < 6; trial++) {
    const hand = jitter(truth, 4);
    const res = QS.refineQuad(img, hand, null, { outerMargin: 0 });
    const err = maxErr(res.corners, truth);
    const before = maxErr(hand, truth);
    assert.ok(err < 2.5, `corner error ${err.toFixed(2)} px after snap (before ${before.toFixed(2)})`);
    worst = Math.max(worst, err); n++;
  }
}

// Order is kept: the upside-down card's first corner stays the screen bottom-right one.
{
  const res = QS.refineQuad(img, jitter(upsideDown, 3), null, { outerMargin: 0 });
  assert.ok(res.corners[0][1] > res.corners[2][1], "first corner of an upside-down card stays at the bottom");
  assert.ok(res.corners[0][0] > res.corners[2][0], "… and on the right");
}

// A covered corner (estimated) is still the intersection of its two visible sides.
{
  const hand = jitter(upright, 3);
  hand[2] = [hand[2][0] + 6, hand[2][1] - 5]; // a guessed covered corner
  const res = QS.refineQuad(img, hand, [false, false, true, false], { outerMargin: 0 });
  assert.ok(maxErr(res.corners, upright) < 3, `estimated corner recovered, err ${maxErr(res.corners, upright).toFixed(2)}`);
}

// Guards: a counter-clockwise quad is returned unchanged; a quad far from any card is not dragged.
{
  const ccw = upright.slice().reverse();
  const res = QS.refineQuad(img, ccw, null);
  assert.strictEqual(res.snapped, false);
  assert.deepStrictEqual(res.corners, ccw);
  const empty = rotQuad(620, 290, 20, 28, 0).map((p) => [p[0] - 10, p[1] - 2]);
  const r2 = QS.refineQuad(img, empty, null);
  assert.ok(maxErr(r2.corners, empty) <= Math.max(QS.DEFAULTS.maxMoveMin, 20 * QS.DEFAULTS.maxMoveFrac) + 1e-6);
}

// the default outer margin puts each side ≈ 1.5 px outside the found edge
{
  const res = QS.refineQuad(img, upright, null);
  const ss = QS.shortSide(res.corners), ss0 = QS.shortSide(upright);
  assert.ok(ss - ss0 > 1.5 && ss - ss0 < 5, `margin widened the card by ${(ss - ss0).toFixed(2)} px`);
}

// rotateOrder: four presses return to the start; flags travel with their corners.
{
  let c = upright, e = [true, false, false, false];
  const first = QS.rotateOrder(c, e);
  assert.deepStrictEqual(first.corners[3], upright[0]);
  assert.deepStrictEqual(first.estimated, [false, false, false, true]);
  for (let i = 0; i < 4; i++) ({ corners: c, estimated: e } = QS.rotateOrder(c, e));
  assert.deepStrictEqual(c, upright);
  assert.deepStrictEqual(e, [true, false, false, false]);
}

// Timing budget (loose: shared machine)
{
  const t0 = Date.now();
  for (let i = 0; i < 10; i++) QS.refineQuad(img, jitter(foreshortened, 4), null);
  const per = (Date.now() - t0) / 10;
  assert.ok(per < 150, `refineQuad ${per} ms`);
}

console.log(`test_quadsnap.js: all assertions passed (${n} snaps, worst corner error ${worst.toFixed(2)} px)`);
