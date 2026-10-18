// SPDX-License-Identifier: Apache-2.0
// The optional facets of a catalog (the labeling tool's filters): the runtime reads none of
// them, but accepts and refuses catalogs exactly as the Python package does. The cases and the
// answers come from the "facets" section of fixtures/contract.json, written by
// tests/test_browser_fixtures.py at the top of the repository.

import { describe, expect, it } from "vitest";

import { parseCatalog } from "../src/release.js";
import { contract } from "./support.js";

interface Accepted { what: string; doc: unknown; facets: unknown[] }
interface Refused { what: string; doc: unknown; mentions: string[] }

const fixture = contract.facets as { accepted: Accepted[]; refused: Refused[] };

describe("the facets of a catalog", () => {
  it("accepts the catalogs the Python package accepts, with the same facets and values in the same order", () => {
    expect(fixture.accepted.length).toBeGreaterThan(0);
    for (const accepted of fixture.accepted) {
      const parsed = parseCatalog(accepted.doc);
      if (typeof parsed === "string") throw new Error(`${accepted.what}: ${parsed}`);
      expect(parsed.facets, accepted.what).toEqual(accepted.facets);
    }
  });

  it("refuses the catalogs the Python package refuses, naming the same place", () => {
    expect(fixture.refused.length).toBeGreaterThan(0);
    for (const refused of fixture.refused) {
      const reason = parseCatalog(refused.doc);
      expect(typeof reason, refused.what).toBe("string");
      for (const fragment of refused.mentions) expect(reason, refused.what).toContain(fragment);
    }
  });
});
