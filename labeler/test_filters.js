// SPDX-License-Identifier: Apache-2.0
// Tests for the picker's catalog filters (filters.js). Pure Node, stdlib `assert` only. Run
// directly (`node test_filters.js`) or through the Python tests, which shell out to it.
//
// Nothing about any game is built in: the chips are the kinds the catalog's own cards carry,
// and an alternate printing is a card whose `printingOf` names another card.
"use strict";

const assert = require("assert");
const F = require("./filters.js");

const CARDS = [
  { id: "TOY-001", name: "Amber Kitemaker", kind: "Unit", printingOf: null },
  { id: "TOY-002", name: "Brass Bell", kind: "Unit", printingOf: null },
  { id: "TOY-120", name: "Kite", kind: "Spell", printingOf: null },
  { id: "TOY-290", name: "Windy Hill", kind: "Site", printingOf: null },
  { id: "TOY-001A", name: "Amber Kitemaker", kind: "Unit", printingOf: "TOY-001" },
  { id: "PROMO 7/x", name: "Kite", kind: "Spell", printingOf: "TOY-120" }, // any id spelling is fine
  { id: "TOY-900", name: "No Kind Card", kind: null, printingOf: null },
];

function ids(filterState) {
  return CARDS.filter((c) => F.passesFilters(c, filterState)).map((c) => c.id);
}

// The chips come from the catalog: its distinct kinds, in order of first appearance; a card
// without a kind adds none.
assert.deepStrictEqual(F.kindChips(CARDS), ["Unit", "Spell", "Site"]);
assert.deepStrictEqual(F.kindChips([]), []);
assert.deepStrictEqual(F.kindChips([{ id: "x", name: "x" }, { id: "y", name: "y", kind: "" }]), [], "a catalog without kinds has no chips");
assert.deepStrictEqual(F.kindChips(undefined), []);

// No filter: everyone survives.
for (const none of [undefined, null, {}, { kind: null }, { kind: "" }]) {
  assert.strictEqual(ids(none).length, CARDS.length, "no filter keeps all cards");
  assert.strictEqual(F.anyFilter(none), false);
}

// A kind filter keeps exactly the cards of that kind.
assert.deepStrictEqual(ids({ kind: "Unit" }), ["TOY-001", "TOY-002", "TOY-001A"]);
assert.deepStrictEqual(ids({ kind: "Spell" }), ["TOY-120", "PROMO 7/x"]);
assert.deepStrictEqual(ids({ kind: "Site" }), ["TOY-290"]);
assert.deepStrictEqual(ids({ kind: "Nothing" }), [], "a kind no card has matches none");
assert.strictEqual(F.anyFilter({ kind: "Unit" }), true);

// An alternate printing typed in full: the catalog card with that id (case aside) whose
// printingOf names another card.
{
  const alt = F.findPrinting(CARDS, "toy-001a");
  assert.deepStrictEqual([alt.card.id, alt.base.id], ["TOY-001A", "TOY-001"]);
  assert.deepStrictEqual(F.findPrinting(CARDS, "  promo 7/x ").base.id, "TOY-120");
  assert.strictEqual(F.findPrinting(CARDS, "TOY-001"), null, "a base card is not a printing");
  assert.strictEqual(F.findPrinting(CARDS, "TOY-001B"), null, "an id the catalog lacks");
  assert.strictEqual(F.findPrinting(CARDS, ""), null);
  assert.strictEqual(F.findPrinting([{ id: "A1", printingOf: "GONE" }], "a1"), null, "printingOf must name a card");
}

// The printings of a card, and a typed printing id in the catalog's own spelling.
assert.deepStrictEqual(F.printingsOf(CARDS, "TOY-001").map((c) => c.id), ["TOY-001A"]);
assert.deepStrictEqual(F.printingsOf(CARDS, "TOY-290"), []);
assert.strictEqual(F.canonicalPrinting(CARDS, " toy-001a "), "TOY-001A");
assert.strictEqual(F.canonicalPrinting(CARDS, "Some Other Printing"), "Some Other Printing", "kept as typed when the catalog lacks it");
assert.strictEqual(F.canonicalPrinting(CARDS, "   "), null, "empty clears the printing");

// ---- facets: the catalog's other filters, after Type -------------------------------------------
const FACETED = [
  { id: "F-1", name: "One", kind: "unit", facets: { color: ["Red", "Blue"], cost: 3, set: "BETA" } },
  { id: "F-2", name: "Two", kind: "spell", facets: { color: "Green", cost: 1, set: "ALPHA" } },
  { id: "F-3", name: "Three", kind: "unit", facets: { color: "Red", cost: 10 } },
  { id: "F-4", name: "Four", kind: null },
  { id: "F-5", name: "Five", kind: "site", facets: { cost: 0 } },
];
const DECLS = [
  { key: "color", label: "Color", values: ["Red", "Blue", "Green", "Black"] },
  { key: "cost", label: "Cost", type: "number" },
  { key: "set", label: "Set" },
  { key: "rarity", label: "Rarity" }, // no card carries it: no row
];
const fids = (f) => FACETED.filter((c) => F.passesFilters(c, f)).map((c) => c.id);

// The rows: Type first, then each facet in declared order; values declared, else numbers
// ascending and text in order of first appearance; a row without values is left out.
assert.deepStrictEqual(F.chipRows(FACETED, DECLS), [
  { key: "kind", label: "Type", values: ["unit", "spell", "site"] },
  { key: "color", label: "Color", values: ["Red", "Blue", "Green", "Black"] },
  { key: "cost", label: "Cost", values: [0, 1, 3, 10] },
  { key: "set", label: "Set", values: ["BETA", "ALPHA"] },
]);
assert.deepStrictEqual(F.chipRows(CARDS, undefined).map((r) => r.key), ["kind"], "no facets: Type alone");
assert.deepStrictEqual(F.chipRows([{ id: "x", name: "x" }], []), [], "no kinds and no facets: nothing to filter by");
assert.deepStrictEqual(F.chipRows(FACETED, [{ key: "kind", label: "Kind" }]).map((r) => r.key), ["kind"], "kind is never a facet");

// A card's values: a single value as a list, several as they are, none as [].
assert.deepStrictEqual(F.cardValues(FACETED[0], "color"), ["Red", "Blue"]);
assert.deepStrictEqual(F.cardValues(FACETED[1], "color"), ["Green"]);
assert.deepStrictEqual(F.cardValues(FACETED[3], "color"), []);
assert.deepStrictEqual(F.cardValues(FACETED[3], "kind"), []);

// Every active filter must match; a card with several values matches any of them; 0 is a value.
assert.deepStrictEqual(fids({ color: "Red" }), ["F-1", "F-3"]);
assert.deepStrictEqual(fids({ color: "Blue" }), ["F-1"], "a two-colored card is found by either color");
assert.deepStrictEqual(fids({ color: "Red", kind: "unit", cost: 10 }), ["F-3"]);
assert.deepStrictEqual(fids({ color: "Red", set: "ALPHA" }), []);
assert.deepStrictEqual(fids({ cost: 0 }), ["F-5"], "0 is a value, not no filter");
assert.deepStrictEqual(fids({ cost: "3" }), [], "a number facet is matched as a number");
assert.deepStrictEqual(fids({ color: null, cost: null, kind: "" }), FACETED.map((c) => c.id));
assert.strictEqual(F.anyFilter({ cost: 0 }), true);
assert.strictEqual(F.anyFilter({ color: null, kind: null }), false);

// The reset: the same place keeps the filters, any other place (another box, frame or set)
// starts with none.
{
  const kept = { color: "Red" };
  assert.strictEqual(F.filtersAt("s,f1,B1", "s,f1,B1", kept), kept);
  assert.deepStrictEqual(F.filtersAt("s,f1,B1", "s,f1,B2", kept), {}, "another box");
  assert.deepStrictEqual(F.filtersAt("s,f1,B1", "s,f2,B1", kept), {}, "another frame");
  assert.deepStrictEqual(F.filtersAt(undefined, "s,f1,B1", kept), {}, "the first card of a page load");
}

// Drawing: a label then chips per row; a click on a chip picks it, on the active chip clears it;
// a row of more than WIDE values is a select whose first option is "any".
{
  const make = (tag) => {
    const el = { tag, children: [], className: "", textContent: "", listeners: {}, attrs: {}, options: [], value: "" };
    el.appendChild = (c) => el.children.push(c);
    el.addEventListener = (type, fn) => { el.listeners[type] = fn; };
    el.setAttribute = (k, v) => { el.attrs[k] = v; };
    el.add = (o) => { el.options.push(o); if (o.selected) el.value = o.value; };
    Object.defineProperty(el, "innerHTML", { set() { el.children = []; } });
    return el;
  };
  function Option(text, value, _d, selected) { this.text = text; this.value = value; this.selected = !!selected; }
  const doc = { createElement: make, defaultView: { Option } };
  const picks = [];
  const el = make("div");
  const wide = Array.from({ length: F.WIDE + 1 }, (_, n) => n + 1);
  const rows = [{ key: "kind", label: "Type", values: ["unit", "spell"] }, { key: "cost", label: "Cost", values: wide }];
  F.renderChips(doc, el, rows, { kind: "unit", cost: 4 }, (k, v) => picks.push([k, v]));
  assert.strictEqual(el.children.length, 2);
  const [type, cost] = el.children;
  assert.deepStrictEqual(type.children.map((c) => c.textContent), ["Type", "unit", "spell"]);
  assert.ok(type.children[1].className.includes("active") && !type.children[2].className.includes("active"));
  const down = { preventDefault() {} };
  type.children[1].listeners.mousedown(down);
  type.children[2].listeners.mousedown(down);
  const sel = cost.children[1];
  assert.strictEqual(sel.tag, "select");
  assert.strictEqual(sel.options[0].text, "Any cost");
  assert.strictEqual(sel.options.length, wide.length + 1);
  assert.strictEqual(sel.value, "3", "the active value is selected");
  sel.value = "0"; sel.listeners.change();
  sel.value = ""; sel.listeners.change();
  assert.deepStrictEqual(picks, [["kind", null], ["kind", "spell"], ["cost", 1], ["cost", null]]);
  F.renderChips(doc, el, [], {}, () => {});
  assert.strictEqual(el.children.length, 0, "the element is emptied first");
  // a game with many facets, long labels and long values (the page cuts them with …, sources.css):
  // every row is drawn, a long value and label keep their whole text in the title, rows of more
  // than WIDE values are a select whatever their length
  const long = (n) => `A remarkably long facet value number ${n} that will not fit in one chip`;
  const many = Array.from({ length: 8 }, (_, k) => ({ key: `f${k}`, label: `Facet ${k} with a label longer than forty characters`,
    values: Array.from({ length: k % 2 ? 35 : 10 }, (_, n) => long(n)) }));
  F.renderChips(doc, el, many, {}, () => {});
  assert.strictEqual(el.children.length, 8);
  for (const [k, row] of el.children.entries()) {
    assert.strictEqual(row.children[0].title, many[k].label, "the label's whole text");
    if (k % 2) assert.strictEqual(row.children[1].tag, "select", "35 values: a drop-down");
    else {
      assert.strictEqual(row.children.length, 11);
      assert.ok(row.children.slice(1).every((c, n) => c.title === long(n) && c.textContent === long(n)), "each chip's whole value");
    }
  }
}

console.log("test_filters.js: all assertions passed");

// A facet key named like a property every object has is read only from the card's own facets.
{
  const card = { id: "X", kind: "unit", facets: { cost: 1 } };
  assert.deepStrictEqual(F.cardValues(card, "constructor"), [], "no inherited value");
  assert.deepStrictEqual(F.cardValues(card, "toString"), []);
}
