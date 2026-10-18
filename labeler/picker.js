// SPDX-License-Identifier: Apache-2.0
// Identity picker logic for the labeling tool. Pure: no DOM, no fetch. Shared by the page
// (window.LabelerPicker) and Node (test_picker.js).
//
// With a deck list, the first page is the deck list's cards; the unrestricted search is one step
// away ("not in the list"). "Unknown" is always offered. Both specials are returned on EVERY
// call, whatever the scope, query or deck list, so the page can never render a picker without
// them. A pick from the deck-list scope suggests identitySource "decklist"; a pick from the
// unrestricted search suggests "current-frame" (the labeler can change either).
//
// The card filters (the catalog's kinds; filters.js) narrow both the deck-list page and the
// full list, ANDed with the query. The specials stay, and the id of an alternate printing typed
// in full bypasses the filters (it names one card).
(function (root) {
  "use strict";

  const node = typeof module !== "undefined" && module.exports;
  const Filters = node ? require("./filters.js") : root.LabelerFilters;
  // Card id → its root card (verifycore.js, the one copy in JS; looked up at call time, since the
  // page loads it after this file)
  const printingRoots = (catalog) => (node ? require("./verifycore.js") : root.LabelerVerifyCore).printingRoots(catalog);

  function matches(card, q) {
    if (!q) return true;
    const name = (card.name || "").toLowerCase();
    const id = (card.id || "").toLowerCase();
    return name.includes(q) || id.startsWith(q);
  }

  // Rank: exact id, id prefix, name prefix, name substring; then by name.
  function rank(card, q) {
    if (!q) return 3;
    const id = (card.id || "").toLowerCase(), name = (card.name || "").toLowerCase();
    if (id === q) return 0;
    if (id.startsWith(q)) return 1;
    if (name.startsWith(q)) return 2;
    return 3;
  }

  // Two decklists (one per player) are given as decklists: [{ label, ids }]. In the decklist scope
  // the first list's cards come first, then the second's; a card in both lists appears once,
  // with both labels. Without `decklists`, decklistIds is one unlabeled list.
  function deckGroups(o) {
    if (Array.isArray(o.decklists) && o.decklists.length) {
      return o.decklists.map((d, n) => ({ label: d.label || `Decklist ${n + 1}`, ids: Array.isArray(d.ids) ? d.ids : [] }));
    }
    return [{ label: null, ids: Array.isArray(o.decklistIds) ? o.decklistIds : [] }];
  }

  function anyFilter(f) { return !!(Filters && Filters.anyFilter(f)); }

  // The catalog as one group per card: { roots (id → root id), groups: Map root id → { card, printings } }, in
  // catalog order of each group's first entry. Kept per catalog (it is the same array per page).
  const groupCache = new WeakMap();
  function cardGroups(catalog) {
    if (groupCache.has(catalog)) return groupCache.get(catalog);
    const roots = printingRoots(catalog), byId = new Map(catalog.map((c) => [c && c.id, c])), groups = new Map();
    for (const c of catalog) {
      if (!c || typeof c.id !== "string") continue;
      const r = roots[c.id] || c.id;
      if (!groups.has(r)) groups.set(r, { card: byId.get(r), printings: [] });
      if (r !== c.id) groups.get(r).printings.push(c);
    }
    const out = { roots, groups };
    groupCache.set(catalog, out);
    return out;
  }

  // opts: { query, decklistIds (array, may be empty) or decklists [{label, ids}], scope:
  //         "decklist" | "all", limit, filters: {kind} }
  // → { scope, entries: [{ kind: "card", card, printings, fromDecklist, deckLabels, printing? }],
  //     specials: [{ kind: "not-in-list", active }, { kind: "unknown" }], suggestedSource, total }
  // One entry per card: its alternate printings are folded into their root card (`printings`).
  // A card matches when it or any of its printings matches (the query, and each filter), ranked
  // by its best match; `limit` counts these rows. Deck-list ids are taken as their root card.
  function buildList(catalog, opts) {
    const o = opts || {};
    const q = (o.query || "").trim().toLowerCase();
    const { roots, groups } = cardGroups(catalog);
    const rootOf = (id) => roots[id] || id;
    const groupsOf = deckGroups(o);
    const labelsOf = new Map();
    for (const g of groupsOf) for (const raw of g.ids) {
      const id = rootOf(raw);
      if (!labelsOf.has(id)) labelsOf.set(id, []);
      if (g.label && !labelsOf.get(id).includes(g.label)) labelsOf.get(id).push(g.label);
    }
    const deck = Array.from(labelsOf.keys());
    const hasDeck = deck.length > 0;
    const scope = hasDeck && o.scope !== "all" ? "decklist" : "all";
    const limit = o.limit || 60;
    const f = anyFilter(o.filters) && Filters ? o.filters : null;
    const members = (g) => [g.card].concat(g.printings);
    const hit = (g) => members(g).some((c) => matches(c, q)) && (!f || members(g).some((c) => Filters.passesFilters(c, f)));
    const row = (g, extra) => Object.assign({ kind: "card", card: g.card, printings: g.printings, fromDecklist: labelsOf.has(g.card.id),
      deckLabels: labelsOf.get(g.card.id) || [] }, extra);
    let entries, total;
    // the id of an alternate printing typed in full (a card whose `printingOf` names another):
    // that card first, then its root card, with the printing recorded
    // (a ring of printings, whose root is the typed card itself, is an ordinary query)
    const alt = Filters ? Filters.findPrinting(catalog, q) : null;
    if (alt && rootOf(alt.card.id) !== alt.card.id) {
      const own = alt.card, r = rootOf(own.id), g = groups.get(r);
      entries = [
        { kind: "card", card: own, printings: [], fromDecklist: labelsOf.has(r), deckLabels: labelsOf.get(r) || [] },
        row(g, { printing: own.id }),
      ];
      total = 1; // one card: the typed printing's
    } else if (scope === "decklist") {
      entries = deck.map((id) => groups.get(id)).filter((g) => g && g.card && hit(g)).map((g) => row(g));
      total = entries.length;
    } else {
      const best = (g) => Math.min(...members(g).filter((c) => matches(c, q)).map((c) => rank(c, q)));
      const ranked = Array.from(groups.values()).filter((g) => g.card && hit(g)).map((g) => [g, best(g)])
        .sort((a, b) => a[1] - b[1] || (a[0].card.name || "").localeCompare(b[0].card.name || ""));
      total = ranked.length;
      entries = ranked.slice(0, limit).map(([g]) => row(g));
    }
    return {
      scope,
      entries,
      specials: [{ kind: "not-in-list", active: scope === "all" }, { kind: "unknown" }],
      suggestedSource: scope === "decklist" ? "decklist" : "current-frame",
      filtered: !!f,
      counts: { all: groups.size, deck: deck.length }, // cards (rows), not catalog entries
      total, // the cards that match, before the limit (the grid says when it shows fewer)
    };
  }

  // How many cards of the list carry the same name as `card`, itself included. Different cards
  // may share a name: the id and the picture then tell them apart, and the page says so.
  function sameName(entries, card) {
    const key = (c) => String((c && c.name) || "").trim().toLowerCase();
    const name = key(card);
    if (!name) return 0;
    const ids = new Set((entries || []).filter((e) => e.kind === "card" && key(e.card) === name).map((e) => e.card.id));
    return ids.size;
  }

  // What a pick writes onto the target. entry: a list entry or a special.
  function applyPick(entry, scope) {
    if (entry.kind === "unknown") return { outcome: "card", identity: "unknown", printing: null, identitySource: null };
    if (entry.kind !== "card") return null; // "not-in-list" only switches scope
    return {
      outcome: "card",
      identity: entry.card.id,
      printing: entry.printing || null,
      identitySource: scope === "decklist" ? "decklist" : "current-frame",
    };
  }

  // A pick of one of a card's printings (its tile in the grid, names.js): its root card, with
  // that printing recorded.
  function stripPick(entry, printing) { return Object.assign({}, entry, { printing: printing.id }); }

  const api = { buildList, applyPick, matches, anyFilter, sameName, stripPick };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.LabelerPicker = api;
})(typeof window !== "undefined" ? window : globalThis);
