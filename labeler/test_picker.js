// SPDX-License-Identifier: Apache-2.0
// Tests for picker.js, on made-up cards. Run: node test_picker.js (also run by test_serve.py)
"use strict";
const assert = require("assert");
const P = require("./picker.js");

const catalog = [
  { id: "TOY-001", name: "Amber Kitemaker", kind: "Unit" },
  { id: "TOY-002", name: "Brass Bell", kind: "Unit" },
  { id: "TOY-120", name: "Kite", kind: "Spell" },
  { id: "NEW-082", name: "Jade Lantern, Lit", kind: "Unit" },
  { id: "EXT-R01", name: "Red Marker", kind: "Marker" },
];

function hasSpecials(res) {
  const kinds = res.specials.map((s) => s.kind);
  assert.deepStrictEqual(kinds, ["not-in-list", "unknown"], "both specials, always, in this order");
}

// every combination of decklist / scope / query offers "not in the list" and "unknown"
for (const deck of [[], ["TOY-002", "NEW-082"]]) {
  for (const scope of ["decklist", "all", undefined]) {
    for (const query of ["", "kit", "zzz-no-match", "new-082a", "OGN"]) {
      hasSpecials(P.buildList(catalog, { decklistIds: deck, scope, query }));
    }
  }
}
hasSpecials(P.buildList([], {}));

// with a decklist, the first page is the decklist only (in list order), identitySource → decklist
{
  const r = P.buildList(catalog, { decklistIds: ["NEW-082", "TOY-002"], query: "" });
  assert.strictEqual(r.scope, "decklist");
  assert.deepStrictEqual(r.entries.map((e) => e.card.id), ["NEW-082", "TOY-002"]);
  assert.strictEqual(r.suggestedSource, "decklist");
  assert.strictEqual(r.specials[0].active, false);
  const pick = P.applyPick(r.entries[0], r.scope);
  assert.deepStrictEqual(pick, { outcome: "card", identity: "NEW-082", printing: null, identitySource: "decklist" });
  // a query that matches nothing in the decklist leaves an empty page, specials still there
  const none = P.buildList(catalog, { decklistIds: ["NEW-082"], query: "kite" });
  assert.strictEqual(none.entries.length, 0);
  hasSpecials(none);
}

// "not in the list" = the unrestricted search; picks from it suggest current-frame
{
  const r = P.buildList(catalog, { decklistIds: ["NEW-082"], scope: "all", query: "kit" });
  assert.strictEqual(r.scope, "all");
  assert.deepStrictEqual(r.entries.map((e) => e.card.id), ["TOY-120", "TOY-001"], "name-prefix ranked first");
  assert.strictEqual(r.specials[0].active, true);
  assert.strictEqual(P.applyPick(r.entries[0], r.scope).identitySource, "current-frame");
  assert.strictEqual(P.applyPick(r.specials[0], r.scope), null, "not-in-list only switches scope");
}

// no decklist: always the unrestricted search
{
  const r = P.buildList(catalog, { decklistIds: [], scope: "decklist", query: "toy-00" });
  assert.strictEqual(r.scope, "all");
  assert.deepStrictEqual(r.entries.map((e) => e.card.id), ["TOY-001", "TOY-002"]);
}

// unknown
assert.deepStrictEqual(P.applyPick({ kind: "unknown" }, "decklist"),
  { outcome: "card", identity: "unknown", printing: null, identitySource: null });

// an id the catalog does not list is just a query with no match (no id pattern is built in)
{
  const r = P.buildList(catalog, { decklistIds: [], query: "NEW-082A" });
  assert.deepStrictEqual(r.entries, []);
  hasSpecials(r);
}

// two decklists (one per player): list order kept, labeled, a shared card once with both labels
{
  const decklists = [{ label: "Player A", ids: ["TOY-001", "TOY-120"] }, { label: "Player B", ids: ["NEW-082", "TOY-120"] }];
  const r = P.buildList(catalog, { decklists, query: "" });
  assert.strictEqual(r.scope, "decklist");
  assert.deepStrictEqual(r.entries.map((e) => e.card.id), ["TOY-001", "TOY-120", "NEW-082"]);
  assert.deepStrictEqual(r.entries.map((e) => e.deckLabels), [["Player A"], ["Player A", "Player B"], ["Player B"]]);
  hasSpecials(r);
  // the unrestricted search still marks decklist cards and their list
  const all = P.buildList(catalog, { decklists, scope: "all", query: "jade" });
  assert.deepStrictEqual(all.entries.map((e) => [e.card.id, e.fromDecklist, e.deckLabels]), [["NEW-082", true, ["Player B"]]]);
  // an empty second list does not hide the first; no lists at all → the full search
  assert.deepStrictEqual(P.buildList(catalog, { decklists: [{ label: "A", ids: ["TOY-002"] }, { label: "B", ids: [] }] }).entries.map((e) => e.card.id), ["TOY-002"]);
  assert.strictEqual(P.buildList(catalog, { decklists: [] }).scope, "all");
  assert.strictEqual(P.applyPick(r.entries[2], r.scope).identitySource, "decklist");
}

// the kind filter narrows the deck-list page and the full list (ANDed with the query); the
// specials stay; the id of an alternate printing typed in full bypasses the filter: that card
// first, then the card it is a printing of, with the printing recorded
{
  const cat = [
    { id: "TOY-001", name: "Amber Kitemaker", kind: "Unit", printingOf: null },
    { id: "TOY-002", name: "Brass Bell", kind: "Unit", printingOf: null },
    { id: "TOY-120", name: "Kite", kind: "Spell", printingOf: null },
    { id: "NEW-082", name: "Lilac Fan, Folded", kind: "Unit", printingOf: null },
    { id: "NEW-082A", name: "Lilac Fan, Folded", kind: "Unit", printingOf: "NEW-082" },
    { id: "EXT-R01", name: "Red Marker", kind: "Marker", printingOf: null },
  ];
  const deck = ["TOY-120", "TOY-002", "TOY-001", "EXT-R01"];
  const ids = (r) => r.entries.map((e) => e.card.id);
  assert.deepStrictEqual(ids(P.buildList(cat, { decklistIds: deck, filters: { kind: "Unit" } })), ["TOY-002", "TOY-001"], "deck-list page, list order kept");
  assert.deepStrictEqual(ids(P.buildList(cat, { decklistIds: deck, filters: { kind: "Unit" }, query: "kit" })), ["TOY-001"]);
  assert.deepStrictEqual(ids(P.buildList(cat, { decklistIds: deck, filters: { kind: "Spell" }, query: "kit" })), ["TOY-120"]);
  const lilac = P.buildList(cat, { decklistIds: deck, scope: "all", filters: { kind: "Unit" }, query: "lilac" });
  assert.deepStrictEqual(lilac.entries.map((e) => [e.card.id, e.printings.map((c) => c.id)]), [["NEW-082", ["NEW-082A"]]], "a printing is folded into its card");
  assert.deepStrictEqual(ids(P.buildList(cat, { scope: "all", filters: { kind: "Marker" } })), ["EXT-R01"]);
  const none = P.buildList(cat, { decklistIds: deck, filters: { kind: "Site" } });
  assert.strictEqual(none.entries.length, 0);
  hasSpecials(none);
  assert.strictEqual(none.filtered, true);
  assert.strictEqual(P.buildList(cat, { filters: { kind: null } }).filtered, false, "an empty filter is no filter");
  // the alternate-printing path: its own id first, then the base card with the printing; the
  // filter does not hide it
  const alt = P.buildList(cat, { decklistIds: deck, query: "new-082a", filters: { kind: "Marker" } });
  assert.deepStrictEqual(alt.entries.map((e) => [e.card.id, e.printing || null]), [["NEW-082A", null], ["NEW-082", "NEW-082A"]]);
  const picked = P.applyPick(alt.entries[1], alt.scope);
  assert.deepStrictEqual([picked.outcome, picked.identity, picked.printing], ["card", "NEW-082", "NEW-082A"]);
  hasSpecials(alt);
  // the base card's own id is an ordinary query
  assert.deepStrictEqual(P.buildList(cat, { scope: "all", query: "new-082" }).entries.map((e) => [e.card.id, e.printings.map((c) => c.id)]), [["NEW-082", ["NEW-082A"]]]);
}

// different cards may share one name: every one is a row of its own, the search finds each by
// its id as well as by the name, and sameName counts them so the page can say so
{
  const cat = [
    { id: "TOY-021", name: "Jade Ring", kind: "site" },
    { id: "TOY-022", name: "Jade Ring", kind: "unit" },
    { id: "TOY-023", name: "Jade Ring" },
    { id: "TOY-024", name: "Jade Ringer", kind: "unit" },
    { id: "TOY-025", name: "Plum Ring", kind: "unit" },
  ];
  const ids = (res) => res.entries.map((e) => e.card.id);
  const all = P.buildList(cat, { scope: "all", query: "jade ring" });
  assert.deepStrictEqual(ids(all), ["TOY-021", "TOY-022", "TOY-023", "TOY-024"], "three cards of one name are three rows");
  assert.strictEqual(P.sameName(all.entries, cat[0]), 3);
  assert.strictEqual(P.sameName(all.entries, cat[3]), 1, "a longer name is another name");
  assert.strictEqual(P.sameName(all.entries, { id: "X", name: "" }), 0);
  assert.strictEqual(P.sameName(all.entries, { id: "TOY-022", name: " jade RING " }), 3, "the same name, whatever its case or the spaces around it");
  // the id picks one of them
  assert.deepStrictEqual(ids(P.buildList(cat, { scope: "all", query: "toy-022" })), ["TOY-022"]);
  // each pick writes its own id: the label stores the card, not the name
  assert.deepStrictEqual(all.entries.slice(0, 3).map((e) => P.applyPick(e, "all").identity), ["TOY-021", "TOY-022", "TOY-023"]);
  // in a deck list too, with two of the three on it
  const deck = P.buildList(cat, { decklistIds: ["TOY-023", "TOY-021", "TOY-025"], query: "ring" });
  assert.deepStrictEqual(ids(deck), ["TOY-023", "TOY-021", "TOY-025"]);
  assert.strictEqual(P.sameName(deck.entries, cat[2]), 2);
}

// one row per card: the alternate printings (`printingOf`, followed to its root) fold into the
// card's row, in both scopes; a search, a filter and the deck list see the printings too
{
  const cat = [
    { id: "OP-010", name: "Sock Thief", kind: "Unit", facets: { color: "red" } },
    { id: "OP-010A", name: "Sock Thief", kind: "Unit", printingOf: "OP-010", facets: { color: "red" } },
    { id: "PR-777", name: "Sock Thief (promo)", kind: "Unit", printingOf: "OP-010A", facets: { color: "gold" } },
    { id: "OP-011", name: "Sock Thief", kind: "Spell", facets: { color: "blue" } },
    { id: "OP-012", name: "Thimble", kind: "Unit", facets: { color: "red" } },
    { id: "OP-012B", name: "Thimble", kind: "Unit", printingOf: "OP-012", facets: { color: "red" } },
  ];
  const ids = (r) => r.entries.map((e) => e.card.id);
  const all = P.buildList(cat, { scope: "all", query: "sock" });
  assert.deepStrictEqual(ids(all), ["OP-010", "OP-011"], "one row per card; two cards of one name stay two rows");
  assert.deepStrictEqual(all.entries[0].printings.map((c) => c.id), ["OP-010A", "PR-777"], "a printing of a printing has the root");
  assert.deepStrictEqual(all.entries[1].printings, []);
  assert.strictEqual(P.sameName(all.entries, cat[0]), 2);
  // a printing's id or a name only a printing carries finds its card
  assert.deepStrictEqual(ids(P.buildList(cat, { scope: "all", query: "pr-7" })), ["OP-010"]);
  assert.deepStrictEqual(ids(P.buildList(cat, { scope: "all", query: "(promo)" })), ["OP-010"]);
  // ranked by its best match: an id prefix of a printing outranks a name substring
  assert.deepStrictEqual(ids(P.buildList(cat, { scope: "all", query: "op-01" })), ["OP-010", "OP-011", "OP-012"]);
  // a filter passes the card when it or any of its printings passes
  assert.deepStrictEqual(ids(P.buildList(cat, { scope: "all", query: "sock", filters: { color: "gold" } })), ["OP-010"]);
  assert.deepStrictEqual(ids(P.buildList(cat, { scope: "all", query: "sock", filters: { color: "blue" } })), ["OP-011"]);
  // the limit counts rows
  assert.deepStrictEqual(ids(P.buildList(cat, { scope: "all", limit: 2 })), ["OP-010", "OP-011"]);
  assert.strictEqual(P.buildList(cat, { scope: "all" }).entries.length, 3);
  // the total: the cards that match before the limit, so the page can say more matched
  assert.strictEqual(P.buildList(cat, { scope: "all", limit: 2 }).total, 3);
  assert.strictEqual(P.buildList(cat, { scope: "all", limit: 2, query: "sock" }).total, 2);
  assert.strictEqual(P.buildList(cat, { scope: "all", query: "zzz" }).total, 0);
  const many = Array.from({ length: 300 }, (_, i) => ({ id: `M-${String(i).padStart(3, "0")}`, name: "Same Name" }));
  const cut = P.buildList(many, { scope: "all", query: "same" });
  assert.deepStrictEqual([cut.entries.length, cut.total], [60, 300], "the default limit is 60; the total counts them all");
  assert.deepStrictEqual(P.buildList(many, { scope: "all", query: "same", limit: 120 }).entries.slice(0, 60), cut.entries, "a larger limit keeps the first ones, in order");
  assert.strictEqual(P.buildList(cat, { decklistIds: ["OP-010", "OP-012"] }).total, 2, "the deck list: every match is shown");
  assert.strictEqual(P.buildList(cat, { query: "PR-777" }).total, 1, "a printing typed in full: one card");
  // a deck list naming printings shows their card once, labels merged
  const deck = P.buildList(cat, { decklists: [{ label: "A", ids: ["PR-777", "OP-012B"] }, { label: "B", ids: ["OP-010"] }] });
  assert.deepStrictEqual(deck.entries.map((e) => [e.card.id, e.deckLabels]), [["OP-010", ["A", "B"]], ["OP-012", ["A"]]]);
  assert.deepStrictEqual(ids(P.buildList(cat, { decklistIds: ["OP-010A"], query: "promo" })), ["OP-010"]);
  // a printing typed in full: that printing, then its root with the printing recorded
  const typed = P.buildList(cat, { query: "PR-777" });
  assert.deepStrictEqual(typed.entries.map((e) => [e.card.id, e.printing || null]), [["PR-777", null], ["OP-010", "PR-777"]]);
  // picking a printing: the root, with the printing; the card itself: the root, no printing
  assert.deepStrictEqual(P.applyPick(P.stripPick(all.entries[0], cat[1]), "all"),
    { outcome: "card", identity: "OP-010", printing: "OP-010A", identitySource: "current-frame" });
  assert.strictEqual(P.applyPick(all.entries[0], "all").printing, null);
  assert.strictEqual(P.applyPick(all.entries[0], "all").identity, "OP-010");
  // ... and that printing's own row is marked as on the deck list through its root
  const typedDeck = P.buildList(cat, { decklists: [{ label: "A", ids: ["OP-010"] }], query: "pr-777" });
  assert.deepStrictEqual(typedDeck.entries.map((e) => [e.card.id, e.fromDecklist, e.deckLabels]), [["PR-777", true, ["A"]], ["OP-010", true, ["A"]]]);
  // the scope buttons count rows: cards in the catalog, distinct cards in the deck list
  assert.deepStrictEqual(deck.counts, { all: 3, deck: 2 });
  // a ring of printings (no catalog should have one) is one card, the lowest id
  const ring = [{ id: "R-2", name: "Ring", printingOf: "R-1" }, { id: "R-1", name: "Ring", printingOf: "R-2" }];
  assert.deepStrictEqual(P.buildList(ring, {}).entries.map((e) => [e.card.id, e.printings.map((c) => c.id)]), [["R-1", ["R-2"]]]);
  // typing the ring's root id in full is an ordinary query: never a card recorded as its own printing
  const self = P.buildList(ring, { query: "r-1" });
  assert.deepStrictEqual(self.entries.map((e) => [e.card.id, e.printing || null]), [["R-1", null]]);
  const loop = P.buildList([{ id: "S-1", name: "Loop", printingOf: "S-1" }], { query: "s-1" });
  assert.deepStrictEqual(loop.entries.map((e) => [e.card.id, e.printing || null]), [["S-1", null]]);
  assert.strictEqual(P.applyPick(loop.entries[0], loop.scope).printing, null);
  // typing the other ring member: it, then the root with it recorded
  assert.deepStrictEqual(P.buildList(ring, { query: "r-2" }).entries.map((e) => [e.card.id, e.printing || null]), [["R-2", null], ["R-1", "R-2"]]);
}

console.log("test_picker.js: all assertions passed");
