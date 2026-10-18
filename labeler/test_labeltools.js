// SPDX-License-Identifier: Apache-2.0
// Regression tests for labeltools.js (distribute / insert / remove, fan arc, layout
// matching, prefill merge with carry-over, polygon IoU). Pure Node, stdlib assert only.
// Run: node test_labeltools.js  (also run by test_serve.py)
"use strict";
const assert = require("assert");
const T = require("./labeltools.js");

const close = (a, b, eps, msg) => assert.ok(Math.abs(a - b) <= (eps || 1e-6), `${msg || ""} ${a} vs ${b}`);
const card = (cx, cy, angle = 0, w = 100, h = 140) => ({ cx, cy, w, h, angle });

// ---- polygon IoU ---------------------------------------------------------------------
close(T.rectIoU(card(0, 0), card(0, 0)), 1, 1e-9, "identical");
close(T.rectIoU(card(0, 0), card(1000, 0)), 0, 1e-9, "disjoint");
close(T.rectIoU(card(0, 0), card(50, 0)), (50 * 140) / (2 * 100 * 140 - 50 * 140), 1e-9, "half-overlap");
// a 90°-rotated portrait box is the landscape box of the same footprint
close(T.rectIoU({ cx: 0, cy: 0, w: 100, h: 140, angle: 90 }, { cx: 0, cy: 0, w: 140, h: 100, angle: 0 }), 1, 1e-9, "rot90");
// a 45° square vs the axis-aligned one: intersection is the regular octagon
{
  const iou = T.rectIoU({ cx: 0, cy: 0, w: 100, h: 100, angle: 45 }, { cx: 0, cy: 0, w: 100, h: 100, angle: 0 });
  const oct = 2 * (1 + Math.SQRT2) * (100 / (1 + Math.SQRT2)) ** 2; // area of the octagon, side 100/(1+√2)
  close(iou, oct / (2 * 10000 - oct), 1e-6, "45°");
}
close(T.aabbIoU(card(0, 0), card(50, 0)), T.rectIoU(card(0, 0), card(50, 0)), 1e-9, "aabb == poly when upright");

// ---- lerpAngle takes the short way ----------------------------------------------------
close(T.lerpAngle(170, -170, 0.5), 180, 1e-9, "wrap");
close(T.lerpAngle(-6, 6, 0.25), -3, 1e-9);

// ---- distribute: straight row -----------------------------------------------------------
{
  const rects = [card(100, 500), card(420, 505), card(250, 490), card(900, 500), card(500, 510)];
  const d = T.distribute(rects);
  assert.deepStrictEqual(d.order, [0, 2, 1, 4, 3], "left-to-right order");
  const xs = d.rects.map((r) => r.cx);
  assert.deepStrictEqual(xs.map((x) => Math.round(x)), [100, 300, 500, 700, 900], "even spacing");
  close(d.rects[0].cx, 100); close(d.rects[4].cx, 900); // ends fixed
  close(d.rects[2].cy, 500, 1e-9, "linear y");
}
// non-fan row keeps each middle card's own size and angle (a tapped unit stays tapped)
{
  const rects = [card(100, 500), card(300, 500, 90), card(420, 500), card(700, 500)];
  const d = T.distribute(rects);
  assert.strictEqual(d.rects[1].angle, 90, "tapped card keeps its angle");
  close(d.rects[1].cx, 300, 1e-9);
}

// ---- distribute: hand fan (arc about a common pivot) ---------------------------------------
{
  // a synthetic fan: pivot (960, 3000), radius 2000, angles -8..8 in 4° steps
  const P = { x: 960, y: 3000 }, R = 2000;
  const fan = [-8, -4, 0, 4, 8].map((a) => {
    const u = T.rotatePoint(0, 1, a);
    return { cx: P.x - R * u.x, cy: P.y - R * u.y, w: 100, h: 140, angle: a };
  });
  const piv = T.fanPivot(fan[0], fan[4]);
  close(piv.x, P.x, 1e-6, "pivot x"); close(piv.y, P.y, 1e-6, "pivot y");
  // scramble the middle cards, distribute must restore them exactly
  const messy = fan.map((r, i) => (i === 0 || i === 4 ? r : { ...r, cx: r.cx + 37 * (i - 2), cy: r.cy + 11, angle: 0 }));
  const d = T.distribute(messy);
  d.rects.forEach((r, k) => {
    const want = fan[d.order[k]];
    assert.ok(T.rectIoU(r, want) > 0.999, `fan slot ${k} iou ${T.rectIoU(r, want)}`);
    close(r.angle, [-8, -4, 0, 4, 8][k], 1e-6, `fan angle ${k}`);
  });
  // the middle of the arc sits higher than the straight chord between the ends
  assert.ok(d.rects[2].cy < (fan[0].cy + fan[4].cy) / 2 - 5, "arc bulges up");
}

// ---- insert / remove ---------------------------------------------------------------------
{
  const rects = [card(100, 500), card(300, 500), card(500, 500)];
  // default "grow": spacing (200) and center (300) kept, both ends move out by half a spacing
  const ins = T.insertSlot(rects, 1); // after the middle card
  assert.deepStrictEqual(ins.order, [0, 1, -1, 2]);
  assert.strictEqual(ins.newSlot, 2);
  assert.deepStrictEqual(ins.rects.map((r) => Math.round(r.cx)), [0, 200, 400, 600]);
  // "ends": first and last stay, spacing shrinks
  const insEnds = T.insertSlot(rects, 1, "ends");
  assert.deepStrictEqual(insEnds.rects.map((r) => Math.round(r.cx)), [100, 233, 367, 500]);
  const insEnd = T.insertSlot(rects, 99); // not in the row → right end
  assert.deepStrictEqual(insEnd.order, [0, 1, 2, -1]);
  const one = T.insertSlot([card(100, 500)], 0);
  assert.deepStrictEqual(one.order, [0, -1]);
  close(one.rects[1].cx, 205, 1e-9, "single card: next slot one width to the right");
  const four = [card(100, 500), card(300, 500), card(500, 500), card(700, 500)];
  const rem = T.removeSlot(four, 1);
  assert.deepStrictEqual(rem.order, [0, 2, 3]);
  assert.deepStrictEqual(rem.rects.map((r) => Math.round(r.cx)), [200, 400, 600], "shrink: ends move in by ½ spacing");
  const remEnds = T.removeSlot(four, 1, "ends");
  assert.deepStrictEqual(remEnds.rects.map((r) => Math.round(r.cx)), [100, 400, 700]);
  const remEnd = T.removeSlot([card(100, 500), card(300, 500), card(500, 500)], 2);
  assert.deepStrictEqual(remEnd.order, [0, 1]);
  // a fan grows along its arc with the same angle step
  const P = { x: 960, y: 3000 }, R = 2000;
  const fanAt = (a) => { const u = T.rotatePoint(0, 1, a); return { cx: P.x - R * u.x, cy: P.y - R * u.y, w: 100, h: 140, angle: a }; };
  const grown = T.insertSlot([fanAt(-4), fanAt(0), fanAt(4)], 2);
  [-6, -2, 2, 6].forEach((a, k) => assert.ok(T.rectIoU(grown.rects[k], fanAt(a)) > 0.999, `grown fan slot ${k}`));
}

// ---- reading order / layout matching ---------------------------------------------------------
{
  const grid = [card(300, 200), card(100, 210), card(200, 190), card(150, 400), card(50, 395)];
  assert.deepStrictEqual(T.readingOrder(grid), [1, 2, 0, 4, 3]);
  const row = (xs, zone = "hand", y = 1000) => xs.map((x) => ({ zone, rect: card(x, y) }));
  const current = row([500, 600, 700]).concat(row([900], "side", 800));
  const frames = [
    { index: 3, t: 30, dt: -10, rows: row([400, 520, 640]) }, // same count, other layout
    { index: 1, t: 10, dt: -30, rows: row([500, 600, 700]) }, // same layout: not offered
    { index: 5, t: 50, dt: 10, rows: row([400, 500, 600, 700]) }, // different count
    { index: 6, t: 60, dt: 20, rows: row([300, 420, 540]) },
  ];
  const m = T.layoutMatches(current, "hand:bottom", frames, 1080);
  assert.deepStrictEqual(m.map((x) => x.index), [3, 6], "same count, different layout, nearest first");
  const tgt = current.filter((r) => r.zone === "hand").map((r) => r.rect).reverse();
  const applied = T.applyLayout(tgt, frames[0].rows.map((r) => r.rect));
  // target index 2 is the leftmost current card → gets the leftmost source geometry
  const byTarget = new Map(applied.map((a) => [a.targetIndex, a.rect.cx]));
  assert.strictEqual(byTarget.get(2), 400);
  assert.strictEqual(byTarget.get(0), 640);
  assert.strictEqual(T.applyLayout([card(0, 0)], []), null, "count mismatch refused");
  assert.strictEqual(T.groupKey({ zone: "hand", rect: card(0, 100) }, 1080), "hand:top");
}

// ---- prefill merge with carry-over ---------------------------------------------------------------
{
  const carried = [card(100, 500), card(300, 500)];
  const prefill = [
    { box: [50, 430, 100, 140], cardId: "A" }, // exactly the first carried card → dropped
    { box: [215, 430, 100, 140], cardId: "B" }, // IoU with the second ≈ 0.54 → dropped
    { box: [280, 430, 100, 140], cardId: "C" }, // IoU with the second ≈ 0.54 → dropped
    { box: [335, 430, 100, 140], cardId: "D" }, // IoU ≈ 0.08 → suggested
    { box: [900, 100, 140, 100], cardId: null, flag: "below-tau" }, // nothing there → suggested
  ];
  const s = T.prefillSuggestions(carried, prefill, 0.3);
  assert.deepStrictEqual(s.map((p) => p.cardId), ["D", null]);
  assert.deepStrictEqual(T.prefillSuggestions([], prefill).length, 5, "no carry-over: everything");
  // a wide prefill box is a sideways card: portrait w/h at angle 90
  assert.deepStrictEqual(T.rectFromBox([900, 100, 140, 100]), { cx: 970, cy: 150, w: 100, h: 140, angle: 90 });
  assert.strictEqual(T.prefillNeedsLook({ cardId: "A", flag: null }), false);
  assert.strictEqual(T.prefillNeedsLook({ cardId: "A", flag: "low-margin" }), true);
  assert.strictEqual(T.prefillNeedsLook({ cardId: null, flag: "abstained" }), true);
}

console.log("test_labeltools.js: all assertions passed");
