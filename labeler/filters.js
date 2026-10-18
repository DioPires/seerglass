// SPDX-License-Identifier: Apache-2.0
// Catalog-filter logic, shared between the browser (sources.js, via a plain <script> tag: see
// index.html's load order) and Node (test_filters.js). Pure, no fetch and no state beyond what is
// passed in, so it is unit-testable and cannot drift from what the page filters with; the one
// function that draws (renderChips) is handed its document and element.
//
// Nothing about any game is built in. The first filter is Type: the `kind` values the catalog's
// own cards carry. After it come the catalog's facets (a top-level "facets" list of {key, label,
// values?, type?}, and per card "facets": {key: value or list}; the format is the repository's,
// cardvision/core/catalog.py), one row each. One value per filter at a time; a card passes when
// it matches every active filter, and a card with several values for a facet matches any of them.
// An alternate printing is a card whose `printingOf` names another card.
(function (root) {
  "use strict";

  // The kind chips: every distinct `kind` of the catalog, in order of first appearance.
  function kindChips(catalog) {
    const out = [];
    for (const c of catalog || []) {
      if (c && typeof c.kind === "string" && c.kind && !out.includes(c.kind)) out.push(c.kind);
    }
    return out;
  }

  // A facet's values in chip order: its declared `values`, else those the cards carry (numbers
  // ascending, text in order of first appearance).
  function facetValues(catalog, facet) {
    if (Array.isArray(facet.values)) return facet.values.slice();
    const out = [];
    for (const c of catalog || []) for (const v of cardValues(c, facet.key)) if (!out.includes(v)) out.push(v);
    return out.every((v) => typeof v === "number") ? out.sort((a, b) => a - b) : out;
  }

  // The filter rows: Type (the kinds), then each declared facet; a row without values is left out.
  // → [{ key, label, values }]
  function chipRows(catalog, facets) {
    const rows = [{ key: "kind", label: "Type", values: kindChips(catalog) }];
    for (const f of Array.isArray(facets) ? facets : []) {
      if (f && typeof f.key === "string" && f.key !== "kind") rows.push({ key: f.key, label: String(f.label || f.key), values: facetValues(catalog, f) });
    }
    return rows.filter((r) => r.values.length);
  }

  // A card's values for a filter key, as a list (none: []).
  function cardValues(card, key) {
    if (!card) return [];
    if (key === "kind") return typeof card.kind === "string" && card.kind ? [card.kind] : [];
    const own = card.facets && typeof card.facets === "object" && Object.prototype.hasOwnProperty.call(card.facets, key);
    const v = own ? card.facets[key] : undefined;
    return v === undefined || v === null ? [] : Array.isArray(v) ? v : [v];
  }

  function isActive(v) { return v !== null && v !== undefined && v !== ""; }

  // filterState: { kind, <facet key>: value } (null, "" or absent: no constraint).
  function passesFilters(card, filterState) {
    for (const [k, v] of Object.entries(filterState || {})) {
      if (isActive(v) && !cardValues(card, k).includes(v)) return false;
    }
    return true;
  }

  function anyFilter(f) { return !!f && Object.values(f).some(isActive); }

  // The filters to use at `place` (a key naming the set, the frame and the box shown), coming
  // from `prev`: every filter clears when the place changes, since the next card can be any card.
  function filtersAt(prev, place, filters) { return prev === place ? filters : {}; }

  // Rows with more values than this are a drop-down, so the panel stays small.
  const WIDE = 12;

  // Draws the rows into `el` (emptied first): a label, then chips (a click on the active chip
  // clears it) or, for a wide row, a select. onPick(key, value or null).
  function renderChips(doc, el, rows, filters, onPick) {
    el.innerHTML = "";
    const f = filters || {};
    for (const r of rows) {
      const row = doc.createElement("div");
      row.className = "chip-row";
      const label = doc.createElement("span");
      label.className = "muted facet-label";
      label.textContent = r.label;
      label.title = r.label; // cut with … when long
      row.appendChild(label);
      if (r.values.length > WIDE) {
        const sel = doc.createElement("select");
        sel.setAttribute("aria-label", r.label);
        sel.add(new doc.defaultView.Option(`Any ${r.label.toLowerCase()}`, ""));
        r.values.forEach((v, n) => sel.add(new doc.defaultView.Option(String(v), String(n), false, f[r.key] === v)));
        sel.addEventListener("change", () => {
          onPick(r.key, sel.value === "" ? null : r.values[Number(sel.value)]);
          const back = doc.getElementById && doc.getElementById("pickerInput"); // the search keeps the keyboard
          if (back) back.focus();
        });
        sel.addEventListener("keydown", (ev) => ev.stopPropagation());
        row.appendChild(sel);
      } else {
        for (const v of r.values) {
          const c = doc.createElement("span");
          c.className = `chip plain${f[r.key] === v ? " active" : ""}`;
          c.textContent = String(v);
          c.title = String(v);
          c.addEventListener("mousedown", (ev) => { ev.preventDefault(); onPick(r.key, f[r.key] === v ? null : v); });
          row.appendChild(c);
        }
      }
      el.appendChild(row);
    }
  }

  // A typed id that names an alternate printing: the catalog card with exactly that id (case
  // aside) whose `printingOf` names another card of the catalog.
  // → { card (the printing's own entry), base (the card it is a printing of) } or null.
  function findPrinting(catalog, raw) {
    const q = String(raw || "").trim().toLowerCase();
    if (!q) return null;
    const card = (catalog || []).find((c) => c && String(c.id || "").toLowerCase() === q);
    if (!card || !card.printingOf) return null;
    const base = catalog.find((c) => c && c.id === card.printingOf);
    return base ? { card, base } : null;
  }

  // The alternate printings of a card: the catalog entries whose `printingOf` is its id.
  function printingsOf(catalog, id) {
    return (catalog || []).filter((c) => c && c.printingOf === id);
  }

  // A printing id as typed, in the catalog's own spelling when the catalog has it (case
  // aside); any other text is kept as typed; empty → null (no printing).
  function canonicalPrinting(catalog, raw) {
    const typed = String(raw || "").trim();
    if (!typed) return null;
    const hit = (catalog || []).find((c) => c && String(c.id || "").toLowerCase() === typed.toLowerCase());
    return hit ? hit.id : typed;
  }

  const api = { kindChips, facetValues, chipRows, cardValues, passesFilters, anyFilter, filtersAt, renderChips, WIDE,
    findPrinting, printingsOf, canonicalPrinting };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    root.LabelerFilters = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
