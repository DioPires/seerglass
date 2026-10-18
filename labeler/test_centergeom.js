// SPDX-License-Identifier: Apache-2.0
// Tests for centergeom.js (center mode, the center census). Run: node test_centergeom.js
// (also run by test_serve.py). Stdlib assert only.
"use strict";
const assert = require("assert");
const C = require("./centergeom.js");
const W = 1920, H = 1080;

// ids: C001…, from the frame's sequence, never reused (the sequence outlives deletes)
{
  assert.deepStrictEqual(C.nextId(0), { id: "C001", seq: 1 });
  assert.deepStrictEqual(C.nextId(41), { id: "C042", seq: 42 });
  assert.strictEqual(C.nextId(999).id, "C1000");
  assert.strictEqual(C.idNumber("C007"), 7);
  assert.strictEqual(C.idNumber("Q007"), null);
  assert.strictEqual(C.seqOf({ centers: [{ id: "C001" }], centerSeq: 5 }), 5, "the stored sequence wins (C002…C005 were deleted)");
  assert.strictEqual(C.seqOf({ centers: [{ id: "C003" }] }), 3, "an older row: the largest id");
  assert.strictEqual(C.seqOf({ centers: [{ id: "C009" }], centerSeq: 2 }), 9, "never below an id on the row");
  assert.strictEqual(C.seqOf(null), 0);
}

// a new center: normalized, clamped, rounded to 1e-6, face unless back
{
  const m = C.makeCenter(4, { x: 960, y: 540 }, W, H, "face");
  assert.deepStrictEqual(m, { center: { id: "C005", x: 0.5, y: 0.5, kind: "face" }, seq: 5 });
  const b = C.makeCenter(0, { x: -3, y: 2000 }, W, H, "back").center;
  assert.deepStrictEqual([b.x, b.y, b.kind], [0, 1, "back"]);
  assert.strictEqual(C.makeCenter(0, { x: 1, y: 1 }, W, H, "weird").center.kind, "face");
  const q = C.fromFrame({ x: 1234.5678, y: 1 }, W, H);
  assert.strictEqual(q.x, Math.round((1234.5678 / W) * 1e6) / 1e6);
}

// hit test: within tol frame px, the nearest; a tie goes to the later one (drawn on top)
{
  const cs = [{ id: "C001", x: 0.5, y: 0.5, kind: "face" }, { id: "C002", x: 0.5 + 6 / W, y: 0.5, kind: "back" }];
  assert.strictEqual(C.hitCenter(cs, { x: 960 + 1, y: 540 }, W, H, 8), "C001");
  assert.strictEqual(C.hitCenter(cs, { x: 966 + 1, y: 540 }, W, H, 8), "C002");
  const ex = [{ id: "C001", x: 0.5, y: 0.5, kind: "face" }, { id: "C002", x: 0.5 + 6 / 1024, y: 0.5, kind: "back" }]; // exact in binary
  assert.strictEqual(C.hitCenter(ex, { x: 515, y: 512 }, 1024, 1024, 8), "C002", "equidistant: the later one");
  assert.strictEqual(C.hitCenter(cs, { x: 990, y: 540 }, W, H, 8), null);
  assert.strictEqual(C.hitCenter([], { x: 0, y: 0 }, W, H, 8), null);
}

// counts, kind toggle, cycle in click order
{
  const cs = [{ id: "C003", kind: "back" }, { id: "C001", kind: "face" }, { id: "C002", kind: "face" }];
  assert.deepStrictEqual(C.counts(cs), { faces: 2, backs: 1 });
  assert.deepStrictEqual(C.counts([]), { faces: 0, backs: 0 });
  assert.strictEqual(C.otherKind("face"), "back");
  assert.strictEqual(C.otherKind("back"), "face");
  assert.strictEqual(C.cycle(cs, null, 1), "C001");
  assert.strictEqual(C.cycle(cs, null, -1), "C003");
  assert.strictEqual(C.cycle(cs, "C003", 1), "C001");
  assert.strictEqual(C.cycle(cs, "C001", -1), "C003");
  assert.strictEqual(C.cycle([], null, 1), null);
}

// diff (the timing events an undo / redo logs); extra keys (a later verify pass) survive
{
  const before = [{ id: "C001", x: 0.1, y: 0.1, kind: "face" }, { id: "C002", x: 0.2, y: 0.2, kind: "face", identity: "TOY-001" }];
  const after = [{ id: "C002", x: 0.25, y: 0.2, kind: "back", identity: "TOY-001" }, { id: "C003", x: 0.3, y: 0.3, kind: "back" }];
  assert.deepStrictEqual(C.diff(before, after), [
    { ev: "center-deleted", center: "C001", kind: "face" },
    { ev: "center-moved", center: "C002", kind: "back" },
    { ev: "center-kind", center: "C002", kind: "back" },
    { ev: "center-added", center: "C003", kind: "back" },
  ]);
  assert.deepStrictEqual(C.diff(before, JSON.parse(JSON.stringify(before))), []);
}

console.log("test_centergeom.js: all passed");
