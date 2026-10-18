// SPDX-License-Identifier: Apache-2.0
// Tests for boxsize.js: the size of a new box in the Boxes pass, where it comes from, and what a
// click on the frame where there is no box does.
// Run: node test_boxsize.js (also run by test_serve.py)
"use strict";
const assert = require("assert");
const B = require("./boxsize.js");

// the middle value
assert.strictEqual(B.median([]), null);
assert.strictEqual(B.median([3]), 3);
assert.strictEqual(B.median([5, 1, 3]), 3);
assert.strictEqual(B.median([4, 1, 3, 2]), 2.5);
assert.strictEqual(B.median([1, NaN, "x", 3]), 2, "only numbers count");

// a box as it lies on screen: one turned about a quarter is as wide as its own height
assert.deepStrictEqual(B.screenSize({ w: 80, h: 120, angle: 0 }), { w: 80, h: 120 });
assert.deepStrictEqual(B.screenSize({ w: 80, h: 120, angle: 90 }), { w: 120, h: 80 });
assert.deepStrictEqual(B.screenSize({ w: 80, h: 120, angle: -85 }), { w: 120, h: 80 });
assert.deepStrictEqual(B.screenSize({ w: 80, h: 120, angle: 180 }), { w: 80, h: 120 });
assert.deepStrictEqual(B.screenSize({ w: 80, h: 120, angle: 30 }), { w: 80, h: 120 });
assert.deepStrictEqual(B.screenSize({ w: 80, h: 120 }), { w: 80, h: 120 });

// where the size comes from, in order: a size set for this set, the model's boxes on the set,
// the boxes finished so far, a size set last on another set, the tool's default
const model = [{ w: 84, h: 120 }, { w: 90, h: 130 }, { w: 86, h: 126 }, { w: 200, h: 40 }];
const confirmed = [{ w: 70, h: 100 }, { w: 72, h: 104 }];
assert.deepStrictEqual(B.newBoxSize({}), { w: 110, h: 150, from: "default" });
assert.deepStrictEqual(B.newBoxSize({ last: { w: 60, h: 90 } }), { w: 60, h: 90, from: "last" });
assert.deepStrictEqual(B.newBoxSize({ last: { w: 60, h: 90 }, confirmed }), { w: 71, h: 102, from: "confirmed" });
assert.deepStrictEqual(B.newBoxSize({ last: { w: 60, h: 90 }, confirmed, model }), { w: 88, h: 123, from: "model" }, "the median of each side");
assert.deepStrictEqual(B.newBoxSize({ chosen: { w: 99.6, h: 140.2 }, confirmed, model }), { w: 100, h: 140, from: "chosen" }, "a size you set wins");
// what is not a size is passed over
assert.deepStrictEqual(B.newBoxSize({ chosen: { w: 1, h: 140 }, model: [{ w: 0, h: 0 }, { w: NaN, h: 5 }], confirmed, min: 3 }), { w: 71, h: 102, from: "confirmed" });
assert.deepStrictEqual(B.newBoxSize({ chosen: null, model: [], confirmed: [], last: { w: "x", h: 2 } }), { w: 110, h: 150, from: "default" });

// the side panel's line, and where the size comes from in words
assert.deepStrictEqual(B.sizeText({ w: 86, h: 126, from: "model" }),
  { text: "New box: 86 × 126 px", from: "N, then a click on a card, puts a box of this size on it: the middle size of the model's boxes on this set." });
for (const from of ["chosen", "model", "confirmed", "last", "default", "?"]) assert.ok(B.sizeText({ w: 1, h: 2, from }).from.endsWith("."), from);

// a click on the frame where there is no box
const click = (o) => B.clickOnFrame(Object.assign({ editable: true, placing: false, snap: false, selected: false }, o));
assert.strictEqual(click({}), "deselect", "a plain click never adds a box: N, then a click, does");
assert.strictEqual(click({ selected: true }), "deselect", "the click that ends work on a box never adds one");
assert.strictEqual(click({ snap: true }), "snap", "Snap to card edges stays as it was");
assert.strictEqual(click({ snap: true, selected: true }), "snap");
assert.strictEqual(click({ placing: true, selected: true }), "place", "N, then a click, as before");
assert.strictEqual(click({ placing: true, snap: true }), "place");
assert.strictEqual(click({ editable: false }), "deselect", "a set that takes no change gets no box");
assert.strictEqual(click({ editable: false, placing: true, snap: true }), "deselect");

console.log("test_boxsize.js: all assertions passed");
