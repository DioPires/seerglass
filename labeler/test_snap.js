// SPDX-License-Identifier: Apache-2.0
// Regression tests for snap.js on synthetic board images: a card = dark outer border + textured
// art on a navy board, the look of a card in a digital client. Pure Node, stdlib assert only.
// Run: node test_snap.js  (also run by test_serve.py)
"use strict";
const assert = require("assert");
const Snap = require("./snap.js");
const { rectIoU } = require("./labeltools.js");

// deterministic PRNG
let seedState = 12345;
function rnd() {
  seedState = (seedState * 1103515245 + 12345) & 0x7fffffff;
  return seedState / 0x7fffffff;
}

function board(W, H, cards, innerFrame) {
  const data = new Uint8Array(W * H * 3);
  for (let i = 0; i < W * H; i++) {
    data[i * 3] = 30 + rnd() * 6; data[i * 3 + 1] = 45 + rnd() * 6; data[i * 3 + 2] = 80 + rnd() * 6;
  }
  for (const c of cards) {
    // axis-aligned in world space (angle 0 portrait, or 90 = landscape footprint)
    const land = Math.abs(c.angle || 0) === 90;
    const ww = land ? c.h : c.w, hh = land ? c.w : c.h;
    const x0 = Math.round(c.cx - ww / 2), y0 = Math.round(c.cy - hh / 2);
    for (let y = y0; y < y0 + hh; y++) {
      for (let x = x0; x < x0 + ww; x++) {
        const k = (y * W + x) * 3;
        const border = x < x0 + 5 || x >= x0 + ww - 5 || y < y0 + 5 || y >= y0 + hh - 5;
        if (border) { data[k] = 12; data[k + 1] = 12; data[k + 2] = 14; continue; }
        // art: bright concentric rings (edges in every direction, none along the card's axes),
        // optionally with an inner frame line 12 px in
        const frame = innerFrame && (x === x0 + 12 || x === x0 + ww - 13 || y === y0 + 12 || y === y0 + hh - 13);
        const v = frame ? 220 : 140 + 70 * Math.sin(Math.hypot(x - c.cx - 9, y - c.cy + 13) * 0.35) + 10 * rnd();
        data[k] = v; data[k + 1] = v * 0.8; data[k + 2] = v * 0.6;
      }
    }
  }
  // a 3×3 box blur: real frames are scaled, JPEG-compressed video, edges are 2–3 px wide
  const out = new Uint8Array(data.length);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      for (let c = 0; c < 3; c++) {
        let s = 0, n = 0;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx, yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
          s += data[(yy * W + xx) * 3 + c]; n++;
        }
        out[(y * W + x) * 3 + c] = s / n;
      }
    }
  }
  return { data: out, width: W, height: H, channels: 3 };
}

const truth = { cx: 200, cy: 170, w: 100, h: 140, angle: 0 };
const neighbor = { cx: 330, cy: 170, w: 100, h: 140, angle: 0 };
const img = board(520, 360, [truth, neighbor]);

// clicks jittered up to 20 % of the card width, seed 10 % smaller than the card
Snap.snapRect(img, Snap.seedFromClick(truth.cx, truth.cy, { w: 90, h: 126 }, false)); // JIT warm-up
const worst = [], times = [];
for (const [dx, dy] of [[0, 0], [18, 0], [-18, 5], [0, -19], [12, 14], [-14, -12]]) {
  const seed = Snap.seedFromClick(truth.cx + dx, truth.cy + dy, { w: 90, h: 126 }, false);
  const res = Snap.snapRect(img, seed);
  const iou = rectIoU(res.rect, truth);
  worst.push(iou);
  assert.ok(iou > 0.9, `click (${dx},${dy}) snapped IoU ${iou.toFixed(3)} rect ${JSON.stringify(res.rect)}`);
  times.push(res.ms);
}
// budget ~150 ms per snap (median, so one scheduler hiccup on a loaded machine does not fail it)
times.sort((a, b) => a - b);
assert.ok(times[Math.floor(times.length / 2)] < 150, `median snap ${times[Math.floor(times.length / 2)]} ms`);
// the seed alone (no snap) is clearly worse — the test would catch a snap that does nothing
const seedOnly = rectIoU(Snap.seedFromClick(truth.cx + 18, truth.cy, { w: 90, h: 126 }, false), truth);
assert.ok(seedOnly < 0.75, `seed IoU ${seedOnly}`);

// a card with a strong inner frame line 12 px in (the art frame): a center click still boxes
// the card, and the outer-edge refinement moves the sides out from the border's inner edge
{
  const imgF = board(520, 360, [truth, neighbor], true);
  const seed0 = Snap.seedFromClick(truth.cx + 4, truth.cy - 3, { w: 90, h: 126 }, false);
  const iouF = rectIoU(Snap.snapRect(imgF, seed0).rect, truth);
  assert.ok(iouF > 0.9, `inner-frame card IoU ${iouF.toFixed(3)}`);
  const seed = Snap.seedFromClick(truth.cx, truth.cy, { w: 90, h: 126 }, false);
  const inner = Snap.snapRect(img, seed, { outer: false }).rect;
  const outer = Snap.snapRect(img, seed).rect;
  assert.ok(outer.w >= inner.w && outer.h >= inner.h, "refinement moves sides outward");
  assert.ok(rectIoU(outer, truth) >= rectIoU(inner, truth), "refinement does not hurt");
}

// landscape (Shift+click): a sideways card, stored portrait at angle 90
{
  const side = { cx: 260, cy: 150, w: 100, h: 140, angle: 90 };
  const img2 = board(520, 320, [side]);
  const seed = Snap.seedFromClick(270, 158, { w: 92, h: 128 }, true);
  assert.strictEqual(seed.angle, 90);
  const res = Snap.snapRect(img2, seed);
  const iou = rectIoU(res.rect, side);
  assert.ok(iou > 0.9, `landscape IoU ${iou.toFixed(3)}`);
  assert.strictEqual(res.rect.angle, 90);
}

// scoring sanity: the true placement outscores a shifted one
{
  const g = Snap.gradientROI(img, 0, 0, 519, 359);
  const opts = Snap.DEFAULTS;
  const s0 = Snap.scoreRect(g, truth, truth, opts);
  const s1 = Snap.scoreRect(g, { ...truth, cx: truth.cx + 9 }, truth, opts);
  assert.ok(s0 > s1, `score ${s0} vs shifted ${s1}`);
}

console.log(`test_snap.js: all assertions passed (worst click IoU ${Math.min(...worst).toFixed(3)})`);
