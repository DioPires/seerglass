// SPDX-License-Identifier: Apache-2.0
// Center-mode helpers for the labeling tool (a "center census": a click on the center of every
// card). Pure: no DOM, no fetch. Shared by the page (window.LabelerCenterGeom) and Node
// (test_centergeom.js).
//
// A frame row in center mode: `centers: [{id, x, y, kind}]` (x/y normalized to [0, 1] of the
// frame, kind "face" | "back"), `centerSeq` (the last id number issued on the frame: ids are
// C001, C002, … and never reused, not even after a delete or an undo), `frameW`, `frameH`,
// `done`. A center may carry other keys (the later verify pass's answers); every helper here
// keeps them.
(function (root) {
  "use strict";

  const KINDS = ["face", "back"];
  const HIT_PX = 8; // screen px: a click this near a marker selects it instead of adding one
  const ID_RE = /^C(\d{3,})$/;

  const clamp01 = (v) => Math.max(0, Math.min(1, v));
  const r6 = (v) => Math.round(v * 1e6) / 1e6;

  function idNumber(id) {
    const m = typeof id === "string" ? ID_RE.exec(id) : null;
    return m ? parseInt(m[1], 10) : null;
  }

  // The frame's sequence: the stored one, else (an older row) the largest id number on it.
  function seqOf(row) {
    const top = Math.max(0, ...((row && row.centers) || []).map((c) => idNumber(c.id) || 0));
    return row && Number.isInteger(row.centerSeq) ? Math.max(row.centerSeq, top) : top;
  }

  // The next id of a frame, and the sequence after it.
  function nextId(seq) {
    const n = (seq || 0) + 1;
    return { id: `C${String(n).padStart(3, "0")}`, seq: n };
  }

  // frame px ↔ normalized
  function toFrame(c, W, H) { return { x: c.x * W, y: c.y * H }; }
  function fromFrame(p, W, H) { return { x: r6(clamp01(p.x / W)), y: r6(clamp01(p.y / H)) }; }

  // The center nearest frame point p within `tol` FRAME px (the caller converts its screen px), or
  // null. On a tie the later one wins (drawn on top).
  function hitCenter(centers, p, W, H, tol) {
    let best = null, bd = tol;
    for (const c of centers || []) {
      const q = toFrame(c, W, H);
      const d = Math.hypot(q.x - p.x, q.y - p.y);
      if (d <= bd) { bd = d; best = c.id; }
    }
    return best;
  }

  function counts(centers) {
    let faces = 0, backs = 0;
    for (const c of centers || []) { if (c.kind === "back") backs++; else faces++; }
    return { faces, backs };
  }

  // A new center at frame point p: {center, seq}. Nothing else on the row changes.
  function makeCenter(seq, p, W, H, kind) {
    const n = nextId(seq);
    const q = fromFrame(p, W, H);
    return { center: { id: n.id, x: q.x, y: q.y, kind: kind === "back" ? "back" : "face" }, seq: n.seq };
  }

  function otherKind(kind) { return kind === "back" ? "face" : "back"; }

  // The id `dir` steps from `id` in click order (id order), wrapping; the first/last when id is null.
  function cycle(centers, id, dir) {
    const list = (centers || []).slice().sort((a, b) => (idNumber(a.id) || 0) - (idNumber(b.id) || 0));
    if (!list.length) return null;
    const i = list.findIndex((c) => c.id === id);
    if (i < 0) return (dir < 0 ? list[list.length - 1] : list[0]).id;
    return list[(i + dir + list.length) % list.length].id;
  }

  // What changed between two versions of a frame's centers, as timing events (an undo / redo
  // logs these with its `via`): added, deleted, moved (x/y), kind changed.
  function diff(before, after) {
    const out = [];
    const old = new Map((before || []).map((c) => [c.id, c]));
    const now = new Map((after || []).map((c) => [c.id, c]));
    for (const c of before || []) if (!now.has(c.id)) out.push({ ev: "center-deleted", center: c.id, kind: c.kind });
    for (const c of after || []) {
      const o = old.get(c.id);
      if (!o) { out.push({ ev: "center-added", center: c.id, kind: c.kind }); continue; }
      if (o.x !== c.x || o.y !== c.y) out.push({ ev: "center-moved", center: c.id, kind: c.kind });
      if (o.kind !== c.kind) out.push({ ev: "center-kind", center: c.id, kind: c.kind });
    }
    return out;
  }

  const api = { KINDS, HIT_PX, idNumber, seqOf, nextId, toFrame, fromFrame, hitCenter, counts, makeCenter, otherKind, cycle, diff, clamp01, r6 };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.LabelerCenterGeom = api;
})(typeof window !== "undefined" ? window : globalThis);
