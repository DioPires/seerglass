// SPDX-License-Identifier: Apache-2.0
// Verify a crop's candidate cards by fitting a geometric model to matched local features.
// In Python: cardvision/geometric/stage.py (`GeometricStage.verify`), whose description holds.
//
// For each crop:
//   1. ORB keypoints are taken from the crop, scaled to `crop.longSide` (features.ts);
//   2. the candidates are the `embeddingTopK` best of the ranked cards, together with the
//      `voteTopN` cards whose index descriptors are nearest to the most crop descriptors (the
//      vote, every crop of a frame in one call to the voter);
//   3. each candidate's descriptors are matched with the crop's (ratio test), a model is fitted
//      to the matched points with OpenCV's RANSAC, and the inliers are counted;
//   4. the inlier count is fused with the cosine (fusion.ts).
//
// The stage decides nothing itself: recognition.ts turns the fused scores into the answer.
// This runtime ports the configuration that was measured: ORB features, the brute-force vote,
// and the models this OpenCV build can fit. Settings outside that are refused when the stage
// is made, so the runtime never runs a variant nobody measured.

import { type GeometricConfig, parseGeometricText } from "./config.js";
import { resizeDirection, rowHashes, sha256Hex, type StageDebug } from "./debug.js";
import { type CropFeatures, type Cv, extractOrb, orbOfGrey, scaledGrey } from "./features.js";
import {
  baselineOf,
  BOTH,
  EMBEDDING,
  evidenceOf,
  type FoundBy,
  fusedScore,
  type StageResult,
  type Verified,
  VOTE,
} from "./fusion.js";
import { countVotes, type KnnPairs, ratioMatches } from "./match.js";
import { fitInliers, unsupportedFit } from "./ransac.js";

/** Bytes per ORB descriptor. */
const WIDTH = 32;

/** One card of the packed index (geometric/index.ts). */
export interface IndexCard {
  cardId: string;
  artV: string;
  width: number;
  height: number;
  /** The card's first row in the whole index. */
  start: number;
  n: number;
}

/** The packed descriptor index, as geometric/index.ts loads it. */
export interface StageIndex {
  cards: IndexCard[];
  /** rows x 32 descriptor bytes. */
  desc: Uint8Array;
  /** rows x 2 keypoint positions. */
  xy: Float32Array;
  /** Index row to its card's position in `cards`. */
  rowCard: Uint32Array;
  rows: number;
  sourceSha256: string;
  bankSha256: string;
  opencv: string;
  /** The `features` block the index's keypoints were taken with. */
  features: Readonly<Record<string, unknown>>;
}

/** The whole-catalog nearest-neighbour search (geometric/vote.ts). */
export interface StageVoter {
  knn2(queries: Uint8Array): Promise<KnnPairs>;
  dispose(): void;
}

/** A card the recognizer ranked, with its cosine (its best bank row's). */
export interface RankedCard {
  cardId: string;
  cosine: number;
}

/** One crop to verify: RGBA pixels, upright as cut from the frame, and the recognizer's
 *  ranking of the cards for it. */
export interface VerifyItem {
  rgba: Uint8Array | Uint8ClampedArray;
  width: number;
  height: number;
  ranked: readonly RankedCard[];
}

const now = (): number => (typeof performance !== "undefined" ? performance.now() : Date.now());

/** Why the stage cannot run these settings with this OpenCV, or null when it can. */
export function unsupportedSettings(config: GeometricConfig, cv: Cv): string | null {
  const s = config.settings;
  if (s.features.type !== "orb") return `features.type ${JSON.stringify(s.features.type)} is not ported (only "orb")`;
  if (s.candidates.voteTopN > 0 && s.vote.matcher !== "bf") return `vote.matcher ${JSON.stringify(s.vote.matcher)} is not ported (only "bf")`;
  return unsupportedFit(cv, s.ransac);
}

export class GeometricStage {
  private readonly position = new Map<string, number>();
  /** Each index card's identity and image, by position, for the vote's ratio test. */
  private readonly identity: string[];
  private readonly image: string[];
  /** With debug on, each result carries the stage's record of its crop (debug.ts). It decides
   *  nothing: a scoring run asks for it, a session never does. */
  debug = false;

  /**
   * `identities` maps card id to identity key, as Python's `Catalog.identities()` with the
   * release's identity groups over it (`voteIdentities`); a card it lacks is its own identity.
   * Throws when the settings are outside what this runtime ports.
   */
  constructor(
    readonly config: GeometricConfig,
    readonly index: StageIndex,
    private readonly voter: StageVoter,
    private readonly cv: Cv,
    identities: ReadonlyMap<string, string> = new Map(),
  ) {
    const problem = unsupportedSettings(config, cv);
    if (problem) throw new Error(problem);
    index.cards.forEach((card, i) => this.position.set(card.cardId, i));
    this.identity = index.cards.map((card) => identities.get(card.cardId) ?? card.cardId);
    this.image = index.cards.map((card) => card.artV ?? card.cardId);
  }

  /** Whether the settings re-rank (`fusion.reorder`) or only boost the embedding's answer. */
  get reorder(): boolean {
    return this.config.settings.fusion.reorder;
  }

  /**
   * Whether the stage runs on a crop from `sourceType` (null when unknown, as in a browser)
   * whose embedding look the accept rule `accepted` or not (Python: `applies`).
   */
  applies(sourceType: string | null, accepted: boolean): boolean {
    const gate = this.config.settings.gate;
    if (gate.belowAccept && accepted) return false;
    if (gate.sources === null) return true;
    return sourceType !== null && gate.sources.some((pattern) => fnmatchcase(sourceType, pattern));
  }

  /** Free the voter. */
  dispose(): void {
    this.voter.dispose();
  }

  /** The crop's keypoints. A test may replace it to give the stage known keypoints. */
  protected cropFeatures(item: VerifyItem): CropFeatures {
    const { crop, features } = this.config.settings;
    return extractOrb(this.cv, item.rgba, item.width, item.height, crop.longSide, crop.maxFeatures, features.params);
  }

  /** `[matches, inliers]` of one index card against a crop's keypoints (Python: `inliers`). */
  private inliers(found: CropFeatures, position: number): [number, number] {
    const card = this.index.cards[position]!;
    if (found.n < 2 || card.n < 2) return [0, 0];
    const train = this.index.desc.subarray(card.start * WIDTH, (card.start + card.n) * WIDTH);
    const good = ratioMatches(this.cv, found.desc, found.n, train, card.n, this.config.settings.match.ratio, WIDTH);
    if (good.length === 0) return [0, 0];
    const src = new Float32Array(good.length * 2);
    const dst = new Float32Array(good.length * 2);
    const xy = this.index.xy;
    good.forEach((m, i) => {
      src[2 * i] = xy[2 * (card.start + m.trainIdx)]!;
      src[2 * i + 1] = xy[2 * (card.start + m.trainIdx) + 1]!;
      dst[2 * i] = found.xy[2 * m.queryIdx]!;
      dst[2 * i + 1] = found.xy[2 * m.queryIdx + 1]!;
    });
    return [good.length, fitInliers(this.cv, src, dst, found.size, [card.width, card.height], this.config.settings.ransac)];
  }

  /** Verify one crop. `known`, when given, holds the cards the vote may add (the bank's). */
  async verify(item: VerifyItem, known?: ReadonlySet<string>): Promise<StageResult> {
    return (await this.verifyBatch([item], known))[0]!;
  }

  /**
   * Verify the crops of one frame (Python: `verify`, crop by crop). The vote of every crop is
   * one call to the voter. Each crop's `ranked` is ranked here by cosine, best first (in the
   * given order on a tie), a card given twice keeping its first pair; a card the vote adds that
   * `ranked` lacks takes the lowest cosine given.
   */
  async verifyBatch(items: readonly VerifyItem[], known?: ReadonlySet<string>): Promise<StageResult[]> {
    const settings = this.config.settings;
    const spent = items.map(() => 0);
    const greys: Array<StageDebug["grey"] | null> = items.map(() => null);
    const found = items.map((item, i) => {
      this.cv.setRNGSeed?.(settings.seed);
      const t = now();
      const features = this.debug ? this.debugFeatures(item, (grey) => (greys[i] = grey)) : this.cropFeatures(item);
      spent[i] = spent[i]! + now() - t;
      return features;
    });

    // The vote: every crop's descriptors in one call.
    const votes: Array<Map<number, number>> = items.map(() => new Map());
    if (settings.candidates.voteTopN > 0 && this.index.rows > 0) {
      const total = found.reduce((sum, f) => sum + f.n, 0);
      if (total > 0) {
        const queries = new Uint8Array(total * WIDTH);
        const firsts: number[] = [];
        let at = 0;
        for (const f of found) {
          firsts.push(at);
          queries.set(f.desc.subarray(0, f.n * WIDTH), at * WIDTH);
          at += f.n;
        }
        const knn = await this.voter.knn2(queries);
        found.forEach((f, i) => {
          if (f.n < 1) return;
          votes[i] = countVotes(knn, f.n, firsts[i]!, this.index.rowCard, this.identity, this.image, settings.vote.ratio);
        });
      }
    }

    return items.map((item, i) => {
      const t = now();
      const result = this.finish(item, found[i]!, votes[i]!, known);
      spent[i] = spent[i]! + now() - t;
      const ms = Math.round(spent[i]! * 1000) / 1000;
      if (!this.debug) return { ...result, ms };
      return { ...result, ms, debug: this.debugRecord(item, greys[i]!, found[i]!, votes[i]!, ms) };
    });
  }

  /** The crop's keypoints as `cropFeatures` takes them, from a grey image whose hash and size
   *  are handed to `seen` before it is freed. */
  private debugFeatures(item: VerifyItem, seen: (grey: StageDebug["grey"]) => void): CropFeatures {
    const { crop, features } = this.config.settings;
    const grey = scaledGrey(this.cv, item.rgba, item.width, item.height, crop.longSide);
    try {
      seen({ sha256: sha256Hex(grey.data.subarray(0, grey.rows * grey.cols)), width: grey.cols, height: grey.rows });
      return orbOfGrey(this.cv, grey, crop.maxFeatures, features.params);
    } finally {
      grey.delete();
    }
  }

  /** The debug record of one crop (Python: `debug_record`). */
  private debugRecord(item: VerifyItem, grey: StageDebug["grey"], found: CropFeatures, votes: Map<number, number>, ms: number): StageDebug {
    const counted: Record<string, number> = {};
    const ids = [...votes.keys()].map((position) => this.index.cards[position]!.cardId).sort();
    for (const position of votes.keys()) counted[this.index.cards[position]!.cardId] = votes.get(position)!;
    const sorted: Record<string, number> = {};
    for (const id of ids) sorted[id] = counted[id]!;
    return {
      grey,
      resize: resizeDirection(item.width, item.height, this.config.settings.crop.longSide),
      source: [item.width, item.height],
      keypoints: found.n,
      rows: rowHashes(found.n, found.xy, found.desc),
      votes: sorted,
      ms,
    };
  }

  private finish(item: VerifyItem, found: CropFeatures, votes: Map<number, number>, known?: ReadonlySet<string>): StageResult {
    const settings = this.config.settings;
    const seen = new Map<string, number>();
    for (const { cardId, cosine } of item.ranked) if (!seen.has(cardId)) seen.set(cardId, cosine);
    const order = [...seen.keys()];
    order.sort((a, b) => seen.get(b)! - seen.get(a)!); // stable: ties keep the given order
    let floor = 0;
    let first = true;
    for (const cosine of seen.values()) {
      if (first || cosine < floor) floor = cosine;
      first = false;
    }
    const embedded = order.slice(0, settings.candidates.embeddingTopK);
    const voteOf = new Map<string, number>();
    for (const [position, count] of votes) voteOf.set(this.index.cards[position]!.cardId, count);
    const voted = [...voteOf.keys()]
      .filter((cardId) => known === undefined || known.has(cardId))
      .sort((a, b) => voteOf.get(b)! - voteOf.get(a)! || this.position.get(a)! - this.position.get(b)!)
      .slice(0, settings.candidates.voteTopN);
    const chosen = [...new Set([...embedded, ...voted])];
    const fits = new Map<string, [number, number]>();
    for (const cardId of chosen) {
      const position = this.position.get(cardId);
      if (position !== undefined) fits.set(cardId, this.inliers(found, position));
    }
    const baseline = baselineOf([...fits.values()].map(([, inliers]) => inliers), settings.fusion);
    const embeddedSet = new Set(embedded);
    const votedSet = new Set(voted);
    const candidates: Verified[] = chosen.map((cardId) => {
      const by: FoundBy = embeddedSet.has(cardId) && votedSet.has(cardId) ? BOTH : embeddedSet.has(cardId) ? EMBEDDING : VOTE;
      const cosine = seen.get(cardId) ?? floor;
      const fit = fits.get(cardId);
      if (fit === undefined) {
        return { cardId, cosine, foundBy: by, votes: 0, matches: 0, inliers: 0, evidence: 0, fused: cosine, indexed: false };
      }
      const [matches, inliers] = fit;
      const evidence = evidenceOf(inliers, baseline);
      return {
        cardId, cosine, foundBy: by, votes: voteOf.get(cardId) ?? 0, matches, inliers, evidence,
        fused: fusedScore(cosine, evidence, settings.fusion), indexed: true,
      };
    });
    return { candidates, cropKeypoints: found.n, ms: 0 };
  }
}

/** Python's `fnmatch.fnmatchcase`: `*`, `?`, `[seq]` and `[!seq]`, case-sensitive. */
export function fnmatchcase(name: string, pattern: string): boolean {
  let re = "";
  for (let i = 0; i < pattern.length; i++) {
    const c = pattern[i]!;
    if (c === "*") re += "[\\s\\S]*";
    else if (c === "?") re += "[\\s\\S]";
    else if (c === "[") {
      let j = i + 1;
      if (pattern[j] === "!") j++;
      if (pattern[j] === "]") j++;
      while (j < pattern.length && pattern[j] !== "]") j++;
      if (j >= pattern.length) re += "\\[";
      else {
        let body = pattern.slice(i + 1, j).replace(/\\/g, "\\\\");
        if (body.startsWith("!")) body = "^" + body.slice(1);
        else if (body.startsWith("^")) body = "\\" + body;
        re += `[${body}]`;
        i = j;
      }
    } else re += c.replace(/[.*+?^${}()|[\]\\/]/g, "\\$&");
  }
  return new RegExp(`^(?:${re})$`).test(name);
}

/**
 * Card id to identity key for the vote, as Python's evaluation builds it: the catalog's own
 * rule for every card (`identityOf`), and the release's identity groups over it, each member
 * keyed `group:` and its representative (Python: `groups_identities`).
 */
export function voteIdentities(
  cards: ReadonlyArray<{ id: string; name: string; printingOf: string | null }>,
  rule: "name" | "printing",
  groups: { groups: string[][]; representative: ReadonlyMap<string, string> } | null,
): Map<string, string> {
  const out = new Map<string, string>();
  for (const card of cards) out.set(card.id, rule === "printing" ? (card.printingOf ?? card.id) : card.name);
  if (groups) {
    for (const group of groups.groups) {
      const lead = group.find((member) => !groups.representative.has(member)) ?? [...group].sort()[0]!;
      for (const member of group) out.set(member, `group:${groups.representative.get(member) ?? lead}`);
    }
  }
  return out;
}

/**
 * The watchdog: the stage is turned off for the session when its share of a frame is over
 * `limitMs` on `frames` frames in a row.
 */
export class GeometricWatchdog {
  private run = 0;

  constructor(readonly limitMs = 1500, readonly frames = 2) {}

  /** Record one frame's stage time; true when the stage must be turned off. */
  observe(stageMs: number): boolean {
    this.run = stageMs > this.limitMs ? this.run + 1 : 0;
    return this.run >= this.frames;
  }

  reset(): void {
    this.run = 0;
  }
}

/** Whether two JSON values are equal, as Python's `==` compares the parsed documents. */
function jsonEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (Array.isArray(a) || Array.isArray(b)) {
    return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((x, i) => jsonEqual(x, b[i]));
  }
  if (typeof a !== "object" || typeof b !== "object" || a === null || b === null) return false;
  const ka = Object.keys(a);
  const kb = Object.keys(b);
  return ka.length === kb.length && ka.every((k) => Object.hasOwn(b, k) && jsonEqual((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k]));
}

/** The keys of two `features` blocks that differ, sorted; empty when they agree (Python:
 *  `cardvision.geometric.index.same_features`). */
export function sameFeatures(a: Readonly<Record<string, unknown>>, b: Readonly<Record<string, unknown>>): string[] {
  const keys = [...new Set([...Object.keys(a), ...Object.keys(b)])];
  return keys.filter((key) => !jsonEqual(a[key], b[key])).sort();
}

/** A warning when the index's keypoints were taken by another OpenCV version than the one
 *  running, else null (Python: `GeometricStage.opencv_note`). */
export function opencvNote(built: string, cv: Cv): string | null {
  const running = /Version control:\s*(\S+)/.exec(cv.getBuildInformation?.() ?? "")?.[1] ?? "";
  if (built === running) return null;
  return (
    `the descriptor index was built with OpenCV ${built || "of an unrecorded version"} and OpenCV ` +
    `${running || "of an unknown version"} is running; keypoints may differ slightly from the index's`
  );
}

/** What opening a stage needs from the runtime: the loaders of its three parts, and the
 *  release it serves. */
export interface StageOpenDeps {
  fetchText(url: string): Promise<string>;
  loadIndex(manifestUrl: string, opts: { bankSha256: string; bankIds: string[] }): Promise<StageIndex>;
  loadCv(): Promise<Cv>;
  makeVoter(desc: Uint8Array): Promise<StageVoter>;
  bankSha256: string;
  bankIds: string[];
  identities: ReadonlyMap<string, string>;
}

/**
 * Open the stage whose settings document is at `settingsUrl`: read the settings, load the
 * packed index its `index.path` names (relative to the settings), check that it is a repack of
 * the index the settings pin (`index.sha256`) and that it serves the bank, then load OpenCV and
 * the voter. Rejects with an Error whose message says why; nothing is left running.
 */
export async function openGeometricStage(
  settingsUrl: string,
  deps: StageOpenDeps,
): Promise<{ stage: GeometricStage; bytes: number; note: string | null }> {
  const config = parseGeometricText(await deps.fetchText(settingsUrl), settingsUrl, settingsUrl.split("/").pop() || "geometric settings");
  if (config.indexPath === null || config.indexSha256 === null) throw new Error("index.path and index.sha256 are both needed");
  const index = await deps.loadIndex(config.indexPath, { bankSha256: deps.bankSha256, bankIds: deps.bankIds });
  if (index.sourceSha256 !== config.indexSha256) {
    throw new Error(`the packed index repacks ${index.sourceSha256}, not the ${config.indexSha256} the settings pin`);
  }
  const differ = sameFeatures(config.settings.features, index.features);
  if (differ.length > 0) {
    throw new Error(`the index was built with other features settings (differing: ${differ.join(", ")}); build it again with cardvision build-features`);
  }
  const cv = await deps.loadCv();
  const problem = unsupportedSettings(config, cv);
  if (problem) throw new Error(problem);
  const voter = await deps.makeVoter(index.desc);
  try {
    const stage = new GeometricStage(config, index, voter, cv, deps.identities);
    return { stage, bytes: index.desc.byteLength + index.xy.byteLength, note: opencvNote(index.opencv, cv) };
  } catch (error) {
    voter.dispose();
    throw error;
  }
}
