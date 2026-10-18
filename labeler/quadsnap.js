// SPDX-License-Identifier: Apache-2.0
// Four-corner edge snapping for the labeling tool. Classical, no model.
// Reuses snap.js's color Sobel (`gradientROI`). Shared by the page (window.LabelerQuadSnap) and
// Node (test_quadsnap.js).
//
// A hand-placed quad (corners in the card's own order: TL, TR, BR, BL, clockwise on screen) is
// refined side by side: each side's two end points are moved independently along the side's
// normal (±R px) to the placement whose line lies on the strongest image edge, with the
// texture just outside the side subtracted (as snap.js scores a rect). The refined corners are
// the intersections of adjacent refined side lines, so a snapped corner sits where two card
// edges meet, even under perspective. A side whose score does not clearly beat the hand
// placement keeps the hand line. A corner flagged "estimated" (covered) is still the
// intersection of its two sides, but those sides are sampled only on their visible half.
// Guards: every corner moves ≤ maxMove, the result stays convex and clockwise; otherwise the
// hand quad is returned unchanged (snapped: false).
//
// Snapping has an error floor of a few pixels of its own, so it helps rough clicks and hurts
// careful ones; the page therefore starts with snap OFF (`f` / --snap on) and records per target
// whether snap placed the corners.
(function (root) {
  "use strict";

  const Snap = typeof module !== "undefined" && module.exports ? require("./snap.js") : root.LabelerSnap;

  const DEFAULTS = {
    radiusFrac: 0.1, // search radius as a fraction of the quad's short side …
    radiusMin: 3, // … clamped to [radiusMin, radiusMax] px
    radiusMax: 16,
    step: 1, // px
    inset: 0.15, // skip this fraction of each side at both ends
    samples: 24,
    ring: [3, 5], // px outside the side for the texture term
    ringWeight: 0.5,
    improve: 1.08, // a side moves only if its score beats the hand line's by this factor
    maxMoveFrac: 0.15, // guard: a corner moves at most this fraction of the short side …
    maxMoveMin: 5, // … and at least this many px are always allowed
    maxTiltFrac: 0.06, // a side's two ends may shift by different amounts up to this × its length
    maxTiltMin: 2,
    // outer-edge walk (snap.js's refineOuter, per side): a dark card border on a dark table
    // gives a weak outer step and a strong inner one (border → art); walk each side outward while
    // the pixels still differ from the table's color, as annotators box the outer edge
    outer: true,
    outerMax: 0.1, // fraction of the short side
    outerBgSpread: 30, // the table band must be this uniform (median color distance) to trust it
    outerMinContrast: 20,
    outerMargin: 1.5, // px added outside every snapped side (annotators box just outside the edge)
  };

  function pixel(img, x, y) {
    const C = img.channels || 4;
    const xi = Math.min(img.width - 1, Math.max(0, Math.round(x)));
    const yi = Math.min(img.height - 1, Math.max(0, Math.round(y)));
    const k = (yi * img.width + xi) * C;
    return [img.data[k], img.data[k + 1], img.data[k + 2]];
  }

  function median(a) {
    if (!a.length) return 0;
    const s = a.slice().sort((x, y) => x - y);
    return s[Math.floor(s.length / 2)];
  }

  // Outward offset (px) for the side a → b: 0 when the table beyond it is not uniform or the
  // side does not stand out from it; else the first offset where the median color distance to
  // the table falls under half the distance at the side itself.
  function outerOffset(img, a, b, range, kmax, o) {
    const dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy) || 1;
    const nx = dy / len, ny = -dx / len;
    const pts = [];
    for (let i = 0; i < o.samples; i++) {
      const t = range[0] + ((range[1] - range[0]) * (i + 0.5)) / o.samples;
      pts.push([a[0] + dx * t, a[1] + dy * t]);
    }
    const at = (p, k) => pixel(img, p[0] + nx * k, p[1] + ny * k);
    const band = [];
    for (const p of pts) for (let k = kmax + 2; k <= kmax + 6; k++) band.push(at(p, k));
    const bg = [0, 1, 2].map((c) => median(band.map((q) => q[c])));
    const dist = (q) => Math.hypot(q[0] - bg[0], q[1] - bg[1], q[2] - bg[2]);
    if (median(band.map(dist)) > o.outerBgSpread) return 0;
    const prof = [];
    for (let k = 0; k <= kmax; k++) prof.push(median(pts.map((p) => dist(at(p, k)))));
    const ref = Math.max(prof[0], prof[1] || 0);
    if (ref < o.outerMinContrast) return 0;
    let k = 0;
    while (k <= kmax && prof[k] >= 0.5 * ref) k++;
    return Math.max(0, k - 0.5);
  }

  function dist(a, b) { return Math.hypot(b[0] - a[0], b[1] - a[1]); }

  function shortSide(q) {
    return Math.min((dist(q[0], q[1]) + dist(q[2], q[3])) / 2, (dist(q[1], q[2]) + dist(q[3], q[0])) / 2);
  }

  function signedArea(q) {
    let s = 0;
    for (let i = 0; i < 4; i++) { const a = q[i], b = q[(i + 1) % 4]; s += a[0] * b[1] - b[0] * a[1]; }
    return s / 2;
  }

  function isConvex(q) {
    let sign = 0;
    for (let i = 0; i < 4; i++) {
      const a = q[i], b = q[(i + 1) % 4], c = q[(i + 2) % 4];
      const z = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
      if (Math.abs(z) < 1e-9) return false;
      const s = z > 0 ? 1 : -1;
      if (!sign) sign = s; else if (s !== sign) return false;
    }
    return true;
  }

  // Line score for a → b: mean |grad · n| on the line minus ringWeight × mean |grad| outside.
  // tRange [t0, t1] limits sampling to part of the side (visible half next to a covered corner).
  function lineScore(grad, a, b, tRange, o) {
    const dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy) || 1;
    const nx = dy / len, ny = -dx / len; // outward normal for a clockwise (y-down) quad
    const [t0, t1] = tRange;
    let on = 0, onN = 0, out = 0, outN = 0;
    for (let i = 0; i < o.samples; i++) {
      const t = t0 + ((t1 - t0) * (i + 0.5)) / o.samples;
      const x = a[0] + dx * t, y = a[1] + dy * t;
      const k = at(grad, x, y);
      if (k >= 0) { on += Math.abs(grad.gx[k] * nx + grad.gy[k] * ny); onN++; }
      for (const r of o.ring) {
        const kk = at(grad, x + nx * r, y + ny * r);
        if (kk >= 0) { out += Math.hypot(grad.gx[kk], grad.gy[kk]); outN++; }
      }
    }
    if (!onN) return -Infinity;
    return on / onN - o.ringWeight * (outN ? out / outN : 0);
  }

  function at(grad, x, y) {
    const px = Math.round(x) - grad.x0, py = Math.round(y) - grad.y0;
    if (px < 0 || py < 0 || px >= grad.w || py >= grad.h) return -1;
    return py * grad.w + px;
  }

  // Intersection of line p1 + s·d1 and p2 + t·d2 (null when parallel).
  function intersect(p1, p2, q1, q2) {
    const d1x = p2[0] - p1[0], d1y = p2[1] - p1[1], d2x = q2[0] - q1[0], d2y = q2[1] - q1[1];
    const den = d1x * d2y - d1y * d2x;
    if (Math.abs(den) < 1e-9) return null;
    const s = ((q1[0] - p1[0]) * d2y - (q1[1] - p1[1]) * d2x) / den;
    return [p1[0] + s * d1x, p1[1] + s * d1y];
  }

  // img = {data, width, height, channels}; corners = 4 [x, y]; estimated = 4 booleans.
  // → { corners, snapped, moved: [px × 4], sides: [{before, after, moved}] , ms }
  function refineQuad(img, corners, estimated, userOpts) {
    const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
    const o = Object.assign({}, DEFAULTS, userOpts || {});
    const q = corners.map((p) => [p[0], p[1]]);
    const est = estimated || [false, false, false, false];
    const fail = (why) => ({ corners: q.map((p) => p.slice()), snapped: false, reason: why, moved: [0, 0, 0, 0], sides: [] });
    if (q.length !== 4 || signedArea(q) <= 0 || !isConvex(q)) return fail("hand quad not convex/clockwise");
    const ss = shortSide(q);
    const R = Math.max(o.radiusMin, Math.min(o.radiusMax, ss * o.radiusFrac));
    const pad = R + Math.max(...o.ring) + 3;
    const xs = q.map((p) => p[0]), ys = q.map((p) => p[1]);
    const grad = Snap.gradientROI(img, Math.min(...xs) - pad, Math.min(...ys) - pad, Math.max(...xs) + pad, Math.max(...ys) + pad);
    const lines = [], sides = [];
    for (let s = 0; s < 4; s++) {
      const a = q[s], b = q[(s + 1) % 4];
      // sample the visible part only when one end is a covered corner
      let range = [o.inset, 1 - o.inset];
      if (est[s] && !est[(s + 1) % 4]) range = [0.5, 1 - o.inset];
      else if (!est[s] && est[(s + 1) % 4]) range = [o.inset, 0.5];
      const dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy) || 1;
      const nx = dy / len, ny = -dx / len;
      const base = lineScore(grad, a, b, range, o);
      const tilt = Math.max(o.maxTiltMin, o.maxTiltFrac * len);
      let best = base, bestA = a, bestB = b;
      for (let da = -R; da <= R + 1e-9; da += o.step) {
        const aa = [a[0] + nx * da, a[1] + ny * da];
        for (let db = -R; db <= R + 1e-9; db += o.step) {
          if (Math.abs(da - db) > tilt) continue;
          const bb = [b[0] + nx * db, b[1] + ny * db];
          const sc = lineScore(grad, aa, bb, range, o);
          if (sc > best) { best = sc; bestA = aa; bestB = bb; }
        }
      }
      let moved = best > 0 && best > base * o.improve + 1e-6;
      let la = moved ? bestA : a, lb = moved ? bestB : b;
      let outerPx = 0;
      if (o.outer) {
        outerPx = outerOffset(img, la, lb, range, Math.max(3, Math.round(o.outerMax * ss)), o) + o.outerMargin;
        if (outerPx > 0) { la = [la[0] + nx * outerPx, la[1] + ny * outerPx]; lb = [lb[0] + nx * outerPx, lb[1] + ny * outerPx]; moved = true; }
      }
      lines.push([la, lb]);
      sides.push({ before: base, after: best, moved, outerPx });
    }
    const out = [];
    for (let c = 0; c < 4; c++) {
      // corner c joins side c-1 (ends at c) and side c (starts at c)
      const prev = lines[(c + 3) % 4], next = lines[c];
      const p = intersect(prev[0], prev[1], next[0], next[1]);
      if (!p) return fail("parallel sides");
      out.push(p);
    }
    const maxMove = Math.max(o.maxMoveMin, ss * o.maxMoveFrac);
    const moved = out.map((p, i) => dist(p, q[i]));
    if (moved.some((m) => m > maxMove)) return fail("a corner would move too far");
    if (signedArea(out) <= 0 || !isConvex(out)) return fail("snapped quad not convex/clockwise");
    const ms = (typeof performance !== "undefined" ? performance.now() : Date.now()) - t0;
    return {
      corners: out.map((p) => [Math.round(p[0] * 10) / 10, Math.round(p[1] * 10) / 10]),
      snapped: sides.some((s) => s.moved), moved, sides, ms,
    };
  }

  // Rotate the corner order by one step: the corner that was second becomes the first (fixes a
  // quad started at the wrong corner, e.g. an upside-down card labeled as upright; four presses
  // return to the start). Same for the estimated flags.
  function rotateOrder(corners, estimated) {
    const c = corners.slice(1).concat(corners.slice(0, 1));
    const e = (estimated || [false, false, false, false]);
    return { corners: c, estimated: e.slice(1).concat(e.slice(0, 1)) };
  }

  const api = { DEFAULTS, refineQuad, rotateOrder, signedArea, isConvex, shortSide, intersect };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.LabelerQuadSnap = api;
})(typeof window !== "undefined" ? window : globalThis);
