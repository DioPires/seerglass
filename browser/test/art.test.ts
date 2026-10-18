// SPDX-License-Identifier: Apache-2.0
// Cards without art ("art": null): a card known by its id and name that has no image yet. The
// runtime accepts and refuses catalogs exactly as the Python package does, gives such a card
// its identity, and groups a bank built from the catalog, which has no row for it, as Python
// does. The cases and the answers come from the "art" section of fixtures/contract.json,
// written by tests/test_browser_fixtures.py at the top of the repository.

import { describe, expect, it } from "vitest";

import { groupsByKey, IdentityMap, identityOf, parseIdentityGroups } from "../src/identity.js";
import { parseCatalog } from "../src/release.js";
import { contract } from "./support.js";

interface Bank { ids: string[]; groups: string[][]; representative: Record<string, string>; identityIds: string[] }
interface Accepted { what: string; doc: unknown; art: Record<string, string | null>; keys: Record<string, string>; withoutArt: string[]; bank: Bank }
interface Refused { what: string; doc: unknown; mentions: string[] }

const fixture = contract.art as { accepted: Accepted[]; refused: Refused[] };

describe("cards without art", () => {
  it("accepts the catalogs the Python package accepts, with each card's art, null for a card without art", () => {
    expect(fixture.accepted.length).toBeGreaterThan(0);
    for (const accepted of fixture.accepted) {
      const parsed = parseCatalog(accepted.doc);
      if (typeof parsed === "string") throw new Error(`${accepted.what}: ${parsed}`);
      expect(Object.fromEntries(parsed.cards.map((card) => [card.id, card.art])), accepted.what).toEqual(accepted.art);
      expect(parsed.cards.filter((card) => card.art === null).map((card) => card.id), accepted.what).toEqual(accepted.withoutArt);
      expect(accepted.withoutArt.length, accepted.what).toBeGreaterThan(0);
    }
  });

  it("gives a card without art its identity, and groups a bank without it as Python does", () => {
    for (const accepted of fixture.accepted) {
      const parsed = parseCatalog(accepted.doc);
      if (typeof parsed === "string") throw new Error(`${accepted.what}: ${parsed}`);
      const keys = new Map(parsed.cards.map((card) => [card.id, identityOf(card, parsed.identity)]));
      expect(Object.fromEntries(keys), accepted.what).toEqual(accepted.keys);
      // the bank holds the cards with art only
      expect(accepted.bank.ids, accepted.what).toEqual(parsed.cards.filter((card) => card.art !== null).map((card) => card.id));
      expect(groupsByKey(accepted.bank.ids, keys), accepted.what).toEqual(accepted.bank.groups);
      const groups = parseIdentityGroups({ schemaVersion: 1, kind: "identity-groups", version: 1, groups: accepted.bank.groups, representative: accepted.bank.representative })!;
      const map = new IdentityMap(groups);
      const identities: string[] = [];
      for (const id of accepted.bank.ids) if (!identities.includes(map.of(id))) identities.push(map.of(id));
      expect(identities, accepted.what).toEqual(accepted.bank.identityIds);
    }
  });

  it("refuses the catalogs the Python package refuses, naming the card and its art", () => {
    expect(fixture.refused.length).toBeGreaterThan(0);
    for (const refused of fixture.refused) {
      const reason = parseCatalog(refused.doc);
      expect(typeof reason, refused.what).toBe("string");
      for (const fragment of refused.mentions) expect(reason, refused.what).toContain(fragment);
    }
  });
});
