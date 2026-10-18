// SPDX-License-Identifier: Apache-2.0
// Click-to-box edge snapping. Classical and detector-free: a seed box (the
// default box centered on the click, or the selected box for a re-snap) is moved and scaled
// within ±25 % of its size, keeping the card's aspect ratio, to the placement whose four sides
// lie on the strongest image edges. Pure functions over raw pixels, shared by the browser
// (sources.js, via a plain <script> tag) and Node (test_snap.js) so the tested algorithm is the
// shipped one.
//
// Geometry follows boxgeom.js: rect = {cx, cy, w, h, angle} in frame pixels, angle in degrees,
// y-down, a landscape (tapped) card is w/h portrait with angle 90.
(function (root) {
  "use strict";

  const DEFAULTS = {
    posRange: 0.25, // center search, fraction of the seed's own w (local x) and h (local y)
    scaleRange: 0.25, // size search, fraction of the seed size (aspect kept)
    coarseSteps: 12, // coarse grid steps across each range (odd count of samples = steps + 1)
    fineAspect: 0.04, // fine stage: w and h may each move ±4 % independently
    angleRange: 0, // degrees searched either side of the seed's angle in the fine stage (0 = keep)
    angleStep: 1,
    sideInset: 0.14, // skip this fraction of each side at both ends (rounded corners, neighbors)
    samplesPerSide: 28,
    keepCoarse: 4, // best coarse candidates refined
    sizePrior: 0.15, // penalty per unit of |log scale| (ties go to the seed size)
    ringOffsets: [3, 5, 7], // px outside each side sampled for the ring term
    blur: 0, // [1 2 1] blur passes on the gradient before scoring
    ringWeight: 1,
    centerPrior: 0.35, // penalty at the edge of the center search range, × the mean edge strength
    minWeight: 0.5,
    outer: true, // move each side out from the inner border edge to the card's outer edge
    outerMax: 0.1, // outward search per side, fraction of the card's short side
    outerMargin: 0, // px added outside the found edge (annotator convention)
    outerBgSpread: 30, // board band must be this uniform (median color distance) to trust it
    outerMinContrast: 20, // the snapped line must differ from the board by at least this much
  };

  // ---- gradients -----------------------------------------------------------

  // Color Sobel over a region of interest: per pixel, the gx/gy of the channel with the
  // largest gradient magnitude (a card border often differs from the board more in hue than in
  // luminance). img = {data, width, height, channels (3 = RGB, 4 = RGBA)}.
  function gradientROI(img, x0, y0, x1, y1) {
    const W = img.width, H = img.height, C = img.channels || 4, d = img.data;
    x0 = Math.max(1, Math.floor(x0)); y0 = Math.max(1, Math.floor(y0));
    x1 = Math.min(W - 2, Math.ceil(x1)); y1 = Math.min(H - 2, Math.ceil(y1));
    const rw = Math.max(0, x1 - x0 + 1), rh = Math.max(0, y1 - y0 + 1);
    const gx = new Float32Array(rw * rh), gy = new Float32Array(rw * rh);
    for (let y = y0; y <= y1; y++) {
      const rowUp = (y - 1) * W, row = y * W, rowDn = (y + 1) * W;
      for (let x = x0; x <= x1; x++) {
        let bestM = -1, bx = 0, by = 0;
        for (let c = 0; c < 3; c++) {
          const a = d[(rowUp + x - 1) * C + c], b = d[(rowUp + x) * C + c], e = d[(rowUp + x + 1) * C + c];
          const l = d[(row + x - 1) * C + c], r = d[(row + x + 1) * C + c];
          const f = d[(rowDn + x - 1) * C + c], g = d[(rowDn + x) * C + c], h = d[(rowDn + x + 1) * C + c];
          const sx = (e + 2 * r + h) - (a + 2 * l + f);
          const sy = (f + 2 * g + h) - (a + 2 * b + e);
          const m = sx * sx + sy * sy;
          if (m > bestM) { bestM = m; bx = sx; by = sy; }
        }
        const k = (y - y0) * rw + (x - x0);
        gx[k] = bx; gy[k] = by;
      }
    }
    return { x0, y0, w: rw, h: rh, gx, gy };
  }

  // Separable [1 2 1]-binomial blur, applied `passes` times to gx and gy: a 1–2 px edge becomes
  // a few px wide, so the coarse grid (≈4 px steps) lands in its basin.
  function blurGrad(grad, passes) {
    const { w, h } = grad;
    const tmp = new Float32Array(w * h);
    for (const arr of [grad.gx, grad.gy]) {
      for (let p = 0; p < passes; p++) {
        for (let y = 0; y < h; y++) {
          const r = y * w;
          for (let x = 0; x < w; x++) {
            const l = arr[r + Math.max(0, x - 1)], c = arr[r + x], rr = arr[r + Math.min(w - 1, x + 1)];
            tmp[r + x] = (l + 2 * c + rr) / 4;
          }
        }
        for (let y = 0; y < h; y++) {
          const u = Math.max(0, y - 1) * w, r = y * w, d = Math.min(h - 1, y + 1) * w;
          for (let x = 0; x < w; x++) arr[r + x] = (tmp[u + x] + 2 * tmp[r + x] + tmp[d + x]) / 4;
        }
      }
    }
    return grad;
  }

  // ---- scoring ---------------------------------------------------------------

  function rad(a) { return (a * Math.PI) / 180; }

  // Per side: the mean |gradient · side normal| ON the side line, and the mean gradient
  // magnitude on lines a few px OUTSIDE it (the "ring"). Sides are inset at both ends.
  function sideStats(grad, rect, opts) {
    const cos = Math.cos(rad(rect.angle || 0)), sin = Math.sin(rad(rect.angle || 0));
    const hw = rect.w / 2, hh = rect.h / 2;
    const n = opts.samplesPerSide, inset = opts.sideInset;
    const ring = opts.ringOffsets || [];
    const edge = [0, 0, 0, 0], outside = [0, 0, 0, 0];
    // [start, end, outward normal] in local coords: top, right, bottom, left
    const sides = [
      [-hw, -hh, hw, -hh, 0, -1], [hw, -hh, hw, hh, 1, 0], [-hw, hh, hw, hh, 0, 1], [-hw, -hh, -hw, hh, -1, 0],
    ];
    const look = (lx, ly) => {
      const wx = rect.cx + lx * cos - ly * sin, wy = rect.cy + lx * sin + ly * cos;
      const px = Math.round(wx) - grad.x0, py = Math.round(wy) - grad.y0;
      if (px < 0 || py < 0 || px >= grad.w || py >= grad.h) return -1;
      return py * grad.w + px;
    };
    for (let s = 0; s < 4; s++) {
      const [ax, ay, bx, by, nlx, nly] = sides[s];
      const nx = nlx * cos - nly * sin, ny = nlx * sin + nly * cos;
      let sum = 0, cnt = 0, rsum = 0, rcnt = 0;
      for (let i = 0; i < n; i++) {
        const t = inset + ((1 - 2 * inset) * (i + 0.5)) / n;
        const lx = ax + (bx - ax) * t, ly = ay + (by - ay) * t;
        const k = look(lx, ly);
        if (k >= 0) { sum += Math.abs(grad.gx[k] * nx + grad.gy[k] * ny); cnt++; }
        for (const d of ring) {
          const kr = look(lx + nlx * d, ly + nly * d);
          if (kr >= 0) { rsum += Math.abs(grad.gx[kr]) + Math.abs(grad.gy[kr]); rcnt++; }
        }
      }
      edge[s] = cnt ? sum / cnt : 0;
      outside[s] = rcnt ? rsum / rcnt : 0;
    }
    return { edge, outside };
  }

  function sideMeans(grad, rect, opts) {
    return sideStats(grad, rect, opts).edge;
  }

  // Score of a placement: per side, edge strength on the line minus `ringWeight` × the texture
  // just outside it (a card's outer edge has plain board beyond it; its inner art frame has
  // the card's own border beyond it); then half the mean plus half the weakest side (a full
  // outline beats three strong sides), minus a small prior on the size change.
  function scoreRect(grad, rect, seed, opts) {
    const st = sideStats(grad, rect, opts);
    const m = st.edge.map((e, i) => e - opts.ringWeight * st.outside[i]);
    const mean = (m[0] + m[1] + m[2] + m[3]) / 4;
    const minSide = Math.min(m[0], m[1], m[2], m[3]);
    const scale = Math.sqrt((rect.w * rect.h) / (seed.w * seed.h));
    const base = (st.edge[0] + st.edge[1] + st.edge[2] + st.edge[3]) / 4;
    // center prior: the click is on the card, near its middle (a neighbor of the same size
    // in a stack or fan is the typical wrong answer)
    const a = rad(seed.angle || 0);
    const dx = rect.cx - seed.cx, dy = rect.cy - seed.cy;
    const du = (dx * Math.cos(a) + dy * Math.sin(a)) / seed.w, dv = (-dx * Math.sin(a) + dy * Math.cos(a)) / seed.h;
    const off = (du * du + dv * dv) / (opts.posRange * opts.posRange);
    return 0.5 * mean + opts.minWeight * minSide
      - opts.sizePrior * base * Math.abs(Math.log(scale)) - opts.centerPrior * base * off;
  }

  // ---- search ------------------------------------------------------------------

  function withLocalOffset(seed, du, dv, w, h, angle) {
    const a = rad(seed.angle || 0);
    return {
      cx: seed.cx + du * Math.cos(a) - dv * Math.sin(a),
      cy: seed.cy + du * Math.sin(a) + dv * Math.cos(a),
      w, h, angle,
    };
  }

  function snapRect(img, seed, userOpts) {
    const opts = Object.assign({}, DEFAULTS, userOpts || {});
    const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
    const maxS = 1 + opts.scaleRange + opts.fineAspect;
    const reach = Math.hypot(seed.w, seed.h) * (maxS / 2 + opts.posRange) + 4;
    const grad = blurGrad(gradientROI(img, seed.cx - reach, seed.cy - reach, seed.cx + reach, seed.cy + reach), opts.blur);
    const angle0 = seed.angle || 0;

    // Coarse: center (local du/dv) × uniform scale, seed angle and aspect.
    const N = opts.coarseSteps;
    const coarse = [];
    for (let i = 0; i <= N; i++) {
      const du = seed.w * opts.posRange * (2 * i / N - 1);
      for (let j = 0; j <= N; j++) {
        const dv = seed.h * opts.posRange * (2 * j / N - 1);
        for (let k = 0; k <= N; k++) {
          const s = 1 + opts.scaleRange * (2 * k / N - 1);
          const r = withLocalOffset(seed, du, dv, seed.w * s, seed.h * s, angle0);
          coarse.push({ du, dv, s, score: scoreRect(grad, r, seed, opts) });
        }
      }
    }
    coarse.sort((a, b) => b.score - a.score);

    // Fine: around each of the best coarse candidates, 1-px center steps, 1 % size steps with
    // w and h free within ±fineAspect, optional small angle search.
    const stepU = (seed.w * opts.posRange * 2) / N, stepV = (seed.h * opts.posRange * 2) / N;
    let best = null;
    const seen = [];
    for (const c of coarse) {
      if (seen.length >= opts.keepCoarse) break;
      if (seen.some((p) => Math.abs(p.du - c.du) < stepU * 1.5 && Math.abs(p.dv - c.dv) < stepV * 1.5 && Math.abs(p.s - c.s) < 0.06)) continue;
      seen.push(c);
      // center + uniform scale at 1 px / 1 %
      let cur = { du: c.du, dv: c.dv, sw: c.s, sh: c.s, angle: angle0 };
      const evalP = (p) => scoreRect(grad, withLocalOffset(seed, p.du, p.dv, seed.w * p.sw, seed.h * p.sh, p.angle), seed, opts);
      let curScore = evalP(cur);
      // coordinate ascent: 2 px / 2 % steps, then 1 px / 1 % (cheap, converges in a few passes)
      for (let pass = 0; pass < 3; pass++) {
        const scaleStep = pass === 0 ? 0.02 : 0.01;
        const posStep = pass === 0 ? 2 : 1;
        let improved = true, guard = 0;
        while (improved && guard++ < 60) {
          improved = false;
          const moves = [
            { du: posStep }, { du: -posStep }, { dv: posStep }, { dv: -posStep },
            { sw: scaleStep, sh: scaleStep }, { sw: -scaleStep, sh: -scaleStep },
            { sw: scaleStep }, { sw: -scaleStep }, { sh: scaleStep }, { sh: -scaleStep },
          ];
          if (opts.angleRange > 0) moves.push({ angle: opts.angleStep }, { angle: -opts.angleStep });
          for (const mv of moves) {
            const p = {
              du: cur.du + (mv.du || 0), dv: cur.dv + (mv.dv || 0),
              sw: cur.sw + (mv.sw || 0), sh: cur.sh + (mv.sh || 0), angle: cur.angle + (mv.angle || 0),
            };
            if (Math.abs(p.du) > seed.w * opts.posRange + 1 || Math.abs(p.dv) > seed.h * opts.posRange + 1) continue;
            const us = Math.sqrt(p.sw * p.sh);
            if (us < 1 - opts.scaleRange - 1e-9 || us > 1 + opts.scaleRange + 1e-9) continue;
            if (Math.abs(p.sw / p.sh - 1) > opts.fineAspect + 1e-9) continue;
            if (Math.abs(p.angle - angle0) > opts.angleRange + 1e-9) continue;
            const sc = evalP(p);
            if (sc > curScore + 1e-9) { cur = p; curScore = sc; improved = true; }
          }
        }
      }
      if (!best || curScore > best.score) best = { p: cur, score: curScore };
    }
    const p = best ? best.p : { du: 0, dv: 0, sw: 1, sh: 1, angle: angle0 };
    const inner = withLocalOffset(seed, p.du, p.dv, seed.w * p.sw, seed.h * p.sh, p.angle);
    const rect = opts.outer ? refineOuter(img, inner, opts) : inner;
    const t1 = typeof performance !== "undefined" ? performance.now() : Date.now();
    return { rect, score: best ? best.score : 0, ms: t1 - t0, sides: sideMeans(grad, rect, opts) };
  }

  // ---- outer-boundary refinement ---------------------------------------------------
  //
  // The strongest rectangle on a card drawn by a digital client is usually the INNER edge of its
  // dark border (border → art/frame), a few px inside the card's outer edge, which is a weak
  // dark-on-dark step. Annotators box the outer edge. Per side: estimate the board color in a band just
  // outside, walk outward from the snapped line and stop where the color becomes board-like
  // (distance below half of what it was at the snapped line). A side whose outside band is not
  // plain board (a neighboring card, UI) keeps the snapped line plus the default margin.

  function pixel(img, x, y) {
    const W = img.width, C = img.channels || 4;
    const xi = Math.min(img.width - 1, Math.max(0, Math.round(x)));
    const yi = Math.min(img.height - 1, Math.max(0, Math.round(y)));
    const k = (yi * W + xi) * C;
    return [img.data[k], img.data[k + 1], img.data[k + 2]];
  }

  function median(a) {
    if (!a.length) return 0;
    const s = [...a].sort((x, y) => x - y);
    return s[Math.floor(s.length / 2)];
  }

  function refineOuter(img, rect, opts) {
    const a = rad(rect.angle || 0), cos = Math.cos(a), sin = Math.sin(a);
    const hw = rect.w / 2, hh = rect.h / 2;
    const kmax = Math.max(3, Math.round(opts.outerMax * Math.min(rect.w, rect.h)));
    const n = opts.samplesPerSide, inset = opts.sideInset;
    // [start, end, outward normal] in local coords: top, right, bottom, left
    const sides = [
      [-hw, -hh, hw, -hh, 0, -1], [hw, -hh, hw, hh, 1, 0], [-hw, hh, hw, hh, 0, 1], [-hw, -hh, -hw, hh, -1, 0],
    ];
    const offsets = [0, 0, 0, 0];
    for (let s = 0; s < 4; s++) {
      const [ax, ay, bx, by, nlx, nly] = sides[s];
      const pts = [];
      for (let i = 0; i < n; i++) {
        const t = inset + ((1 - 2 * inset) * (i + 0.5)) / n;
        pts.push([ax + (bx - ax) * t, ay + (by - ay) * t]);
      }
      const at = (lx, ly, k) => {
        const px = lx + nlx * k, py = ly + nly * k;
        return pixel(img, rect.cx + px * cos - py * sin, rect.cy + px * sin + py * cos);
      };
      // board color: per-channel median of a band just beyond the search range
      const band = [];
      for (const [lx, ly] of pts) for (let k = kmax + 2; k <= kmax + 6; k++) band.push(at(lx, ly, k));
      const bg = [0, 1, 2].map((c) => median(band.map((p) => p[c])));
      const dist = (p) => Math.hypot(p[0] - bg[0], p[1] - bg[1], p[2] - bg[2]);
      const spread = median(band.map(dist));
      let off = opts.outerMargin;
      if (spread <= opts.outerBgSpread) {
        const prof = [];
        for (let k = 0; k <= kmax; k++) prof.push(median(pts.map(([lx, ly]) => dist(at(lx, ly, k)))));
        const ref = Math.max(prof[0], prof[1] || 0);
        if (ref >= opts.outerMinContrast) {
          let k = 0;
          while (k <= kmax && prof[k] >= 0.5 * ref) k++;
          off = k - 0.5 + opts.outerMargin;
        }
      }
      offsets[s] = off;
    }
    const [oT, oR, oB, oL] = offsets;
    const x0 = -hw - oL, x1 = hw + oR, y0 = -hh - oT, y1 = hh + oB;
    const mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
    return {
      cx: rect.cx + mx * cos - my * sin,
      cy: rect.cy + mx * sin + my * cos,
      w: x1 - x0, h: y1 - y0, angle: rect.angle || 0,
    };
  }

  // A click seed: the default box (portrait w < h) centered on the click; landscape = the same
  // card tapped (angle 90, the tool's convention for a sideways card).
  function seedFromClick(x, y, defaultBox, landscape) {
    const w = Math.min(defaultBox.w, defaultBox.h), h = Math.max(defaultBox.w, defaultBox.h);
    return { cx: x, cy: y, w, h, angle: landscape ? 90 : 0 };
  }

  const api = { DEFAULTS, gradientROI, blurGrad, sideMeans, scoreRect, snapRect, seedFromClick };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.LabelerSnap = api;
})(typeof window !== "undefined" ? window : globalThis);
