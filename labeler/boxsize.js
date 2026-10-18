// SPDX-License-Identifier: Apache-2.0
// The size of a new box in the Boxes pass, and what a click on the frame does. Pure: no DOM, no
// fetch. Shared by the page (window.LabelerBoxSize) and Node (test_boxsize.js).
//
//   median(values)          the middle value (the mean of the two middle ones), or null
//   screenSize(rect)        a box's width and height as it lies on screen: {w, h}
//   newBoxSize(o)           the size a new box gets, and where that size comes from
//   sizeText(size)          "New box: 86 × 126 px", and a line on where the size comes from
//   clickOnFrame(o)         what a click on the frame where there is no box does
//
// The size is a preference of the page (kept in the browser, as the other options are): it is
// never written into a label file.
(function (root) {
  "use strict";

  const DEFAULT = { w: 110, h: 150 };
  const FROM = {
    chosen: "the size you set for this set",
    model: "the middle size of the model's boxes on this set",
    confirmed: "the middle size of the boxes of this set so far",
    last: "the size you set last, on another set",
    default: "the tool's default",
  };

  function median(values) {
    const v = (values || []).filter((x) => typeof x === "number" && isFinite(x)).sort((a, b) => a - b);
    if (!v.length) return null;
    const m = v.length >> 1;
    return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
  }

  // A box's own rectangle {w, h, angle} as it lies on screen: a box turned about a quarter (a
  // card lying sideways) is as wide as its own height.
  function screenSize(r) {
    const a = ((((r && r.angle) || 0) % 180) + 180) % 180;
    return Math.abs(a - 90) < 45 ? { w: r.h, h: r.w } : { w: r.w, h: r.h };
  }

  function valid(s, min) {
    return !!s && isFinite(s.w) && isFinite(s.h) && s.w >= (min || 3) && s.h >= (min || 3);
  }

  function middle(sizes, min) {
    const ok = (sizes || []).filter((s) => valid(s, min));
    if (!ok.length) return null;
    return { w: Math.round(median(ok.map((s) => s.w))), h: Math.round(median(ok.map((s) => s.h))) };
  }

  // o: {chosen: the size set for this set (Change, or S on a box), model: [{w, h}] the model's
  // boxes on this set, confirmed: [{w, h}] the boxes of its finished frames, last: the size set
  // last on another set, min: the smallest box} → {w, h, from}
  function newBoxSize(o) {
    const x = o || {};
    if (valid(x.chosen, x.min)) return { w: Math.round(x.chosen.w), h: Math.round(x.chosen.h), from: "chosen" };
    const model = middle(x.model, x.min);
    if (model) return Object.assign(model, { from: "model" });
    const confirmed = middle(x.confirmed, x.min);
    if (confirmed) return Object.assign(confirmed, { from: "confirmed" });
    if (valid(x.last, x.min)) return { w: Math.round(x.last.w), h: Math.round(x.last.h), from: "last" };
    return Object.assign({}, DEFAULT, { from: "default" });
  }

  function sizeText(size) {
    return { text: `New box: ${size.w} × ${size.h} px`, from: `N, then a click on a card, puts a box of this size on it: ${FROM[size.from] || FROM.default}.` };
  }

  // What a click (a press and a release without a drag) on the frame where there is no box does.
  // o: {editable, placing (N was pressed), snap (Snap to card edges is on), selected (a box is
  // selected)} → "place" (N, then a click: a box of the new-box size, centered on the click),
  // "snap" (a box fitted to the card's edges), "deselect" (otherwise: a plain click never adds a
  // box, so a stray click or the click that ends work on a box costs nothing).
  function clickOnFrame(o) {
    const x = o || {};
    if (x.editable && x.placing) return "place";
    if (x.editable && x.snap) return "snap";
    return "deselect";
  }

  const api = { DEFAULT, FROM, median, screenSize, newBoxSize, sizeText, clickOnFrame };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.LabelerBoxSize = api;
})(typeof window !== "undefined" ? window : globalThis);
