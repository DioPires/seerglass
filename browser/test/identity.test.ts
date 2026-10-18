// SPDX-License-Identifier: Apache-2.0
// The identity rule a catalog states, as the runtime follows it: which cards are the same
// card, the groups that gives for a bank, and a release whose rule is not its catalog's.
//
// The answers come from the Python package (the "identity" section of fixtures/contract.json,
// written by tests/test_browser_fixtures.py at the top of the repository). The release is the
// toy release of fixtures/toy-release, whose catalog states no rule, rewritten here to state
// one; its two model files are stand-in byte strings, as in release.test.ts.

import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import { groupsByKey, IDENTITY_RULES, IdentityMap, identityOf, type IdentityRule, parseIdentityGroups } from "../src/identity.js";
import { loadReleaseData, parseCatalog, parseRelease, type ReleaseDeps, releaseIdOf, sha256HexWebCrypto } from "../src/release.js";
import { contract, FIXTURES } from "./support.js";

const fixture = contract.identity;

interface FixtureCard {
  id: string;
  name: string;
  printingOf: string | null;
  art: string;
}

describe("the identity rule of a catalog", () => {
  it("gives every card the identity key the Python package gives it, under each rule", () => {
    expect(Object.keys(fixture.rules)).toEqual([...IDENTITY_RULES]);
    for (const rule of IDENTITY_RULES) {
      const keys = Object.fromEntries((fixture.cards as FixtureCard[]).map((card) => [card.id, identityOf(card, rule)]));
      expect(keys).toEqual(fixture.rules[rule].keys);
    }
    // Two different cards of one name: one identity by name, two by printing.
    const [alder, , otherAlder] = fixture.cards as FixtureCard[];
    expect(identityOf(alder!, "name")).toBe(identityOf(otherAlder!, "name"));
    expect(identityOf(alder!, "printing")).not.toBe(identityOf(otherAlder!, "printing"));
  });

  it("groups a bank's ids as the Python package does, under each rule", () => {
    for (const rule of IDENTITY_RULES) {
      const keys = new Map<string, string>(Object.entries(fixture.rules[rule].keys));
      for (const bank of fixture.rules[rule].banks) {
        expect(groupsByKey(bank.ids, keys)).toEqual(bank.groups);
        // The representatives are read from the groups file, and give the bank its identities.
        const groups = parseIdentityGroups({ schemaVersion: 1, kind: "identity-groups", version: 1, groups: bank.groups, representative: bank.representative })!;
        const map = new IdentityMap(groups);
        const identities: string[] = [];
        for (const id of bank.ids as string[]) if (!identities.includes(map.of(id))) identities.push(map.of(id));
        expect(identities).toEqual(bank.identityIds);
      }
    }
    // By printing, the card the others reprint represents its group, whatever the form of its id.
    expect(fixture.rules.printing.banks[0].representative["GEN-009"]).toBe("GEN-003X");
    expect(fixture.rules.name.banks[0].representative["GEN-003X"]).toBe("GEN-009");
  });

  it("reads the rule a catalog states, and name when it states none", () => {
    const doc = (identity?: unknown, cards: unknown = fixture.cards) => ({ game: "generated", locale: "en", ...(identity === undefined ? {} : { identity }), cards });
    for (const [stated, rule] of [[undefined, "name"], ["name", "name"], ["printing", "printing"]] as Array<[string | undefined, IdentityRule]>) {
      const parsed = parseCatalog(doc(stated));
      if (typeof parsed === "string") throw new Error(parsed);
      expect(parsed.identity).toBe(rule);
      expect(parsed.cards.map((card) => [card.id, card.printingOf])).toEqual((fixture.cards as FixtureCard[]).map((card) => [card.id, card.printingOf]));
    }
    expect(parseCatalog(doc("art"))).toContain('"identity" must be "name" or "printing"');
    expect(parseCatalog(doc(null))).toContain('"identity" must be "name" or "printing"');
    // A card without the key is a printing of none.
    const bare = parseCatalog(doc("printing", [{ id: "A-1", name: "a", art: "a.png" }]));
    expect(typeof bare === "string" ? bare : bare.cards[0]!.printingOf).toBeNull();
  });

  it("refuses, under the printing rule only, a printing that does not point at a card of its own", () => {
    for (const refused of fixture.refused) {
      const document = { game: "generated", locale: "en", cards: refused.cards };
      const reason = parseCatalog({ ...document, identity: "printing" });
      expect(typeof reason).toBe("string");
      expect(reason).toContain(refused.card);
      expect(typeof parseCatalog({ ...document, identity: "name" })).toBe("object");
      expect(typeof parseCatalog(document)).toBe("object");
    }
    const cards = [{ id: "A-1", name: "a", art: "a.png", printingOf: "A-9" }];
    expect(parseCatalog({ game: "g", locale: "en", identity: "printing", cards })).toContain("card A-1");
    expect(parseCatalog({ game: "g", locale: "en", identity: "printing", cards: [{ ...cards[0], printingOf: "A-1" }] })).toContain("card A-1");
    expect(parseCatalog({ game: "g", locale: "en", cards: [{ ...cards[0], printingOf: 7 }] })).toContain('"printingOf" must be a card id or null');
  });
});

// --------------------------------------------------------------- a release under each rule
const FOLDER = `${FIXTURES}toy-release/`;
const BASE = "https://example.test/releases/toy/";
type Files = Map<string, Uint8Array>;

const text = (value: unknown): Uint8Array => new TextEncoder().encode(typeof value === "string" ? value : `${JSON.stringify(value, null, 1)}\n`);
const buffer = (bytes: Uint8Array): ArrayBuffer => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
const jsonOf = (files: Files, name: string) => JSON.parse(new TextDecoder().decode(files.get(name)!));

async function pin(block: { path: string; sha256: string; bytes: number }, files: Files): Promise<void> {
  const body = files.get(block.path)!;
  block.sha256 = await sha256HexWebCrypto(buffer(body));
  block.bytes = body.length;
}

/**
 * The toy release after `change` has edited its catalog, its groups file and its document.
 * Every file is pinned again and the document gets the id of its new content, so that what a
 * test sees refused is the disagreement it staged and nothing else.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function toyRelease(change: (parts: { doc: any; catalog: any; groups: any }) => void): Promise<Files> {
  const files: Files = new Map();
  for (const name of ["bank.json", "bank.identity-groups.json", "catalog.json"]) files.set(name, new Uint8Array(readFileSync(`${FOLDER}${name}`)));
  files.set("detector.onnx", text("stand-in detector bytes"));
  files.set("recognizer.onnx", text("stand-in recognizer bytes, a little longer"));
  const doc = JSON.parse(readFileSync(`${FOLDER}release.json`, "utf-8"));
  const parts = { doc, catalog: jsonOf(files, "catalog.json"), groups: jsonOf(files, "bank.identity-groups.json") };
  change(parts);
  files.set("catalog.json", text(parts.catalog));
  files.set("bank.identity-groups.json", text(parts.groups));
  for (const block of [doc.detector, doc.recognizer, doc.identity.groups, doc.catalog]) await pin(block, files);
  doc.id = await releaseIdOf(JSON.stringify(doc), sha256HexWebCrypto);
  files.set("release.json", text(doc));
  return files;
}

async function load(files: Files) {
  const deps: ReleaseDeps = {
    digestSha256Hex: sha256HexWebCrypto,
    fetchFn: async (url) => {
      const body = url.startsWith(BASE) ? files.get(url.slice(BASE.length)) : undefined;
      return { ok: body !== undefined, status: body ? 200 : 404, arrayBuffer: async () => (body ? buffer(body) : new ArrayBuffer(0)) };
    },
  };
  return loadReleaseData(`${BASE}release.json`, deps);
}

async function reasonOf(files: Files): Promise<string> {
  const result = await load(files);
  return result.ok ? "loaded" : result.reason;
}

// The toy game's two alternate printings share their card's name and point at it, so the same
// groups hold under either rule, and the release can be restated under "printing" as it is.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const byPrinting = ({ doc, catalog, groups }: { doc: any; catalog: any; groups: any }): void => {
  doc.identity.level = "printing";
  catalog.identity = "printing";
  groups.rule = { level: "printing" };
};
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const shareAName = (catalog: any): void => {
  // TOY-002 takes the name of TOY-001: two different cards, one name.
  catalog.cards.find((card: { id: string }) => card.id === "TOY-002").name = catalog.cards.find((card: { id: string }) => card.id === "TOY-001").name;
};

describe("a release under the identity rule its catalog states", () => {
  it("accepts the level printing in the document, and no other word", () => {
    const doc = JSON.parse(readFileSync(`${FOLDER}release.json`, "utf-8"));
    const named = parseRelease(doc);
    if ("error" in named) throw new Error(named.error);
    expect(named.release.identity.level).toBe("name");
    doc.identity.level = "printing";
    const printed = parseRelease(doc);
    if ("error" in printed) throw new Error(printed.error);
    expect(printed.release.identity.level).toBe("printing");
    doc.identity.level = "art";
    expect(parseRelease(doc)).toEqual({ error: 'identity.level must be "name" or "printing"' });
  });

  it("loads a printing release, with the groups of a card and its printings", async () => {
    const result = await load(await toyRelease(byPrinting));
    if (!result.ok) throw new Error(result.reason);
    expect(result.data.release.identity.level).toBe("printing");
    expect(result.data.identityGroups.groups).toEqual([["TOY-003", "TOY-003A"], ["TOY-010", "TOY-P01"]]);
    expect(result.data.cards.find((card) => card.id === "TOY-003A")!.printingOf).toBe("TOY-003");
  });

  it("keeps two cards of one name apart under printing, where the name rule would join them", async () => {
    // Under "printing" the groups stay a card and its printings: the release loads as it is.
    expect(await reasonOf(await toyRelease((parts) => (byPrinting(parts), shareAName(parts.catalog))))).toBe("loaded");
    // Under "name" the same catalog makes the two cards one identity, which the groups file does not say.
    expect(await reasonOf(await toyRelease(({ catalog }) => shareAName(catalog)))).toContain("not the ones the catalog's names give for the bank's ids");
    // And groups that join the two under "printing" are not the ones that rule gives.
    const joined = await toyRelease((parts) => {
      byPrinting(parts);
      shareAName(parts.catalog);
      parts.groups.groups.unshift(["TOY-001", "TOY-002"]);
      parts.groups.representative["TOY-002"] = "TOY-001";
    });
    expect(await reasonOf(joined)).toContain("not the ones the catalog's identity rule gives for the bank's ids");
  });

  it("refuses a release whose rule is not its catalog's", async () => {
    // The document says printing, the catalog states nothing, which is name.
    expect(await reasonOf(await toyRelease(({ doc, groups }) => ((doc.identity.level = "printing"), (groups.rule = { level: "printing" }))))).toBe(
      'identity.level is "printing", and the catalog states the identity rule "name": a release is used with the rule its catalog states',
    );
    // The catalog says printing, the document still says name.
    expect(await reasonOf(await toyRelease(({ catalog }) => (catalog.identity = "printing")))).toContain('identity.level is "name", and the catalog states the identity rule "printing"');
    // Document and catalog agree, and the groups file was made by the other rule.
    expect(await reasonOf(await toyRelease(({ doc, catalog }) => ((doc.identity.level = "printing"), (catalog.identity = "printing"))))).toBe("identity.groups: the file's groups are not by printing");
    expect(await reasonOf(await toyRelease(({ groups }) => (groups.rule = { level: "printing" })))).toBe("identity.groups: the file's groups are not by name");
    // A rule the runtime does not know, and a printing of a printing, are refused with the catalog.
    expect(await reasonOf(await toyRelease(({ catalog }) => (catalog.identity = "art")))).toContain('catalog: "identity" must be "name" or "printing"');
    const chain = await toyRelease((parts) => {
      byPrinting(parts);
      parts.catalog.cards.find((card: { id: string }) => card.id === "TOY-010").printingOf = "TOY-003A";
    });
    expect(await reasonOf(chain)).toContain("card TOY-010");
  });
});
