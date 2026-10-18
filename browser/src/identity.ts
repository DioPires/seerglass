// SPDX-License-Identifier: Apache-2.0
// Identity groups: which card ids count as the same card when a match is judged.
// In Python: cardvision/core/bank.py (IdentityGroups) and cardvision/core/catalog.py
// (Catalog.identity_of, identity_groups).
//
// Cards of one identity must not count against each other in the margin rule (a crop that sits
// between two printings of one card is not a doubtful match), and a release says which id is
// shown for a match: the printing whose bank row matched best, or the group's representative.
// Which cards are one identity is the rule the catalog states (`identityOf` below): by
// default, printings that share a name. The groups are a file published with the bank:
//
//   { "schemaVersion": 1, "kind": "identity-groups", "version": <int>,
//     "groups": [["ABC-003", "ABC-003A"], ...],
//     "representative": { "ABC-003A": "ABC-003", ... } }
//
// `representative` lists every member that is not its group's representative; the
// representative is the one member of a group that is not a key.

export const GROUPS_KIND = "identity-groups";

export interface IdentityGroups {
  version: number;
  groups: string[][];
  /** Member id to its group's representative, for every member that is not the representative. */
  representative: Map<string, string>;
}

/** Check a parsed identity-groups document. The whole document is refused (null) on any
 *  inconsistency: a partly read map would change which matches count as doubtful. */
export function parseIdentityGroups(raw: unknown): IdentityGroups | null {
  if (typeof raw !== "object" || raw === null) return null;
  const obj = raw as Record<string, unknown>;
  if (
    obj.schemaVersion !== 1 ||
    obj.kind !== GROUPS_KIND ||
    typeof obj.version !== "number" ||
    !Number.isInteger(obj.version) ||
    !Array.isArray(obj.groups) ||
    typeof obj.representative !== "object" ||
    obj.representative === null ||
    Array.isArray(obj.representative)
  ) {
    return null;
  }
  const groupOf = new Map<string, number>();
  const groups: string[][] = [];
  for (const g of obj.groups) {
    if (!Array.isArray(g) || g.length < 2) return null;
    const members: string[] = [];
    for (const id of g) {
      if (typeof id !== "string" || id.length === 0 || groupOf.has(id)) return null; // an id is in one group only
      groupOf.set(id, groups.length);
      members.push(id);
    }
    groups.push(members);
  }
  const representative = new Map<string, string>();
  for (const [member, rep] of Object.entries(obj.representative as Record<string, unknown>)) {
    if (typeof rep !== "string" || member === rep) return null;
    const g = groupOf.get(member);
    if (g === undefined || groupOf.get(rep) !== g) return null; // both in the same group
    representative.set(member, rep);
  }
  // Every group has exactly one representative, and every other member points straight at it.
  for (const members of groups) {
    const reps = members.filter((m) => !representative.has(m));
    if (reps.length !== 1) return null;
    if (members.some((m) => representative.has(m) && representative.get(m) !== reps[0])) return null;
  }
  return { version: obj.version, groups, representative };
}

/**
 * The identity rules a catalog can state, in its top-level `identity` (absent means "name"):
 *  - "name": entries with exactly the same name are one identity.
 *  - "printing": an entry whose `printingOf` is null is a card of its own, and every entry
 *    whose `printingOf` is that entry's id is the same card. Equal names join nothing.
 */
export const IDENTITY_RULES = ["name", "printing"] as const;
export type IdentityRule = (typeof IDENTITY_RULES)[number];

export function isIdentityRule(value: unknown): value is IdentityRule {
  return (IDENTITY_RULES as readonly unknown[]).includes(value);
}

/**
 * The identity a catalog card belongs to, as a key: two cards are the same card exactly when
 * their keys are equal. This is the one place the rule is read (Python: Catalog.identity_of).
 */
export function identityOf(card: { id: string; name: string; printingOf: string | null }, rule: IdentityRule): string {
  if (rule === "printing") return card.printingOf ?? card.id;
  return card.name;
}

/**
 * The ids that share a key, as groups of two or more: `keys` maps an id to its key (its
 * identity, from `identityOf`). Keys are compared as exact strings and ids without a key are
 * left alone. Each group is sorted, and the list is ordered by each group's first id.
 */
export function groupsByKey(ids: Iterable<string>, keys: ReadonlyMap<string, string>): string[][] {
  const byKey = new Map<string, string[]>();
  for (const id of [...new Set(ids)].sort(compareCodeUnits)) {
    const key = keys.get(id);
    if (key === undefined) continue;
    const group = byKey.get(key);
    if (group) group.push(id);
    else byKey.set(key, [id]);
  }
  return [...byKey.values()].filter((g) => g.length >= 2).sort((a, b) => compareCodeUnits(a[0]!, b[0]!));
}

/** `groupsByKey` with names as the keys: the groups the "name" rule gives. */
export function groupsByName(ids: Iterable<string>, names: ReadonlyMap<string, string>): string[][] {
  return groupsByKey(ids, names);
}

function compareCodeUnits(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

/**
 * What a recognized card shows:
 *  - "best-printing": the id of the bank row that matched best. The groups still decide the
 *    margin (a different identity must be beaten), but the printing whose image matched is shown.
 *  - "representative": the group's representative, whichever printing matched.
 */
export type IdentityDisplay = "representative" | "best-printing";

/**
 * Card id to identity (its group's representative; the id itself when it is in no group).
 * `known`, under the "representative" display, sets aside any group whose representative it
 * rejects (one that could not be shown); that group's members stay separate identities. Under
 * "best-printing" the representative is never shown, so nothing is set aside.
 */
export class IdentityMap {
  private readonly rep = new Map<string, string>();
  readonly display: IdentityDisplay;

  constructor(groups: IdentityGroups | null = null, known?: (cardId: string) => boolean, display: IdentityDisplay = "representative") {
    this.display = display;
    if (!groups) return;
    for (const [member, rep] of groups.representative) {
      if (display === "representative" && known && !known(rep)) continue;
      this.rep.set(member, rep);
    }
  }

  /** The identity a card id belongs to. */
  of(cardId: string): string {
    return this.rep.get(cardId) ?? cardId;
  }

  /** The id shown for a look whose best bank row is `topId`. */
  shown(topId: string): string {
    return this.display === "best-printing" ? topId : this.of(topId);
  }

  /** How many ids map to another id. */
  get collapsedCount(): number {
    return this.rep.size;
  }
}
