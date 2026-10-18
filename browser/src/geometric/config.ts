// SPDX-License-Identifier: Apache-2.0
// The geometric stage's settings: one JSON file, every setting in it.
// In Python: cardvision/geometric/config.py, which describes each setting. This file reads the
// same documents the same way: the same defaults, the same checks and the same messages, an
// unknown key an error, and `toJson` the full settings in Python's key order.
//
// One difference comes from JSON in JavaScript: `300.0` and `300` parse to the same number, so a
// whole-number setting written with a decimal point, which Python refuses, is accepted here.

export const FEATURE_TYPES = ["orb", "akaze", "sift"] as const;
export const BASELINES = ["none", "median"] as const;
export const MATCHERS = ["bf", "flann"] as const;
export const MODELS = ["homography", "affine", "similarity"] as const;
export const METHODS = ["ransac", "usac"] as const;

export type FeatureType = (typeof FEATURE_TYPES)[number];

/** The detector settings each feature type takes, with their defaults, in Python's order.
 *  `whole` marks a setting whose default is a whole number in Python (an `int`): such a
 *  setting must be a whole number too. */
const PARAM_DEFAULTS: Record<FeatureType, ReadonlyArray<readonly [string, number, boolean]>> = {
  orb: [["scaleFactor", 1.2, false], ["nLevels", 8, true], ["edgeThreshold", 15, true], ["patchSize", 15, true], ["fastThreshold", 10, true]],
  akaze: [["threshold", 0.001, false], ["nOctaves", 4, true], ["nOctaveLayers", 4, true]],
  sift: [["nOctaveLayers", 3, true], ["contrastThreshold", 0.04, false], ["edgeThreshold", 10.0, false], ["sigma", 1.6, false]],
};

/** The detector settings of each feature type, with their defaults. */
export const FEATURE_PARAMS: Readonly<Record<FeatureType, Readonly<Record<string, number>>>> = Object.fromEntries(
  FEATURE_TYPES.map((kind) => [kind, Object.fromEntries(PARAM_DEFAULTS[kind].map(([key, value]) => [key, value]))]),
) as Record<FeatureType, Record<string, number>>;

export interface GeometricSettings {
  seed: number;
  features: { type: string; maxFeatures: number; artLongSide: number; mask: number[][]; params: Record<string, number> };
  crop: { longSide: number; maxFeatures: number };
  candidates: { embeddingTopK: number; voteTopN: number };
  vote: { matcher: string; ratio: number; lshTables: number; lshKeySize: number; lshProbes: number; kdTrees: number; checks: number };
  match: { ratio: number };
  ransac: { model: string; method: string; threshold: number; maxIters: number; confidence: number; minAreaRatio: number; maxAreaRatio: number };
  fusion: { weight: number; minInliers: number; fullInliers: number; penalty: number; baseline: string; reorder: boolean };
  gate: { belowAccept: boolean; sources: string[] | null };
}

/** Every setting's default, in Python's order. */
export const DEFAULTS: Readonly<GeometricSettings> = Object.freeze({
  seed: 0,
  features: { type: "orb", maxFeatures: 300, artLongSide: 320, mask: [], params: {} },
  crop: { longSide: 320, maxFeatures: 500 },
  candidates: { embeddingTopK: 5, voteTopN: 0 },
  vote: { matcher: "bf", ratio: 0.8, lshTables: 12, lshKeySize: 20, lshProbes: 2, kdTrees: 4, checks: 64 },
  match: { ratio: 0.8 },
  ransac: { model: "homography", method: "ransac", threshold: 6.0, maxIters: 2000, confidence: 0.995, minAreaRatio: 0.25, maxAreaRatio: 4.0 },
  fusion: { weight: 0.1, minInliers: 8, fullInliers: 30, penalty: 0.0, baseline: "none", reorder: true },
  gate: { belowAccept: false, sources: null },
});

const BLOCKS = ["features", "crop", "candidates", "vote", "match", "ransac", "fusion", "gate"] as const;
const TOP_KEYS = ["index", "seed", ...BLOCKS];
const SHA256 = /^[0-9a-f]{64}$/;

/** A geometric settings document is not valid. `errors` lists every reason. */
export class GeometricConfigError extends Error {
  readonly errors: string[];

  constructor(errors: string[]) {
    super(errors.join("; "));
    this.name = "GeometricConfigError";
    this.errors = [...errors];
  }
}

/** Loaded settings: the full settings, defaults filled in, without the index; where the index
 *  is and what its bytes must hash to (null when the document does not say). */
export class GeometricConfig {
  /** The settings, a frozen copy: nothing that holds the config can change what the stage runs. */
  readonly settings: Readonly<GeometricSettings>;

  constructor(
    settings: Readonly<GeometricSettings>,
    readonly indexPath: string | null = null,
    readonly indexSha256: string | null = null,
  ) {
    this.settings = deepFreeze(structuredClone(settings) as GeometricSettings);
  }

  get seed(): number {
    return this.settings.seed;
  }

  get features(): GeometricSettings["features"] {
    return this.settings.features;
  }

  /** The full settings, as a report records them (the index is reported by its hash). */
  toJson(): GeometricSettings {
    return structuredClone(this.settings) as GeometricSettings;
  }
}

/** `value`, and every object and array inside it, frozen. */
function deepFreeze<T>(value: T): T {
  if (typeof value === "object" && value !== null && !Object.isFrozen(value)) {
    Object.freeze(value);
    for (const inner of Object.values(value as object)) deepFreeze(inner);
  }
  return value;
}

// ------------------------------------------------------------------------- Python's types
type Obj = Record<string, unknown>;

/** A Python `Mapping`: a JSON object. */
function isMapping(value: unknown): value is Obj {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Python `int` and not `bool`. */
function isWhole(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value);
}

/** Python `int` or `float` and not `bool`. */
function isNumber(value: unknown): value is number {
  return typeof value === "number";
}

/** Python's `repr` of a string. */
function reprString(text: string): string {
  const quote = text.includes("'") && !text.includes('"') ? '"' : "'";
  let out = "";
  for (const ch of text) {
    const code = ch.codePointAt(0)!;
    if (ch === "\\") out += "\\\\";
    else if (ch === quote) out += `\\${quote}`;
    else if (ch === "\n") out += "\\n";
    else if (ch === "\r") out += "\\r";
    else if (ch === "\t") out += "\\t";
    else if (code < 0x20 || code === 0x7f) out += `\\x${code.toString(16).padStart(2, "0")}`;
    else out += ch;
  }
  return quote + out + quote;
}

/** Python's `repr` of the values these messages show: strings, numbers, booleans, None, lists. */
function repr(value: unknown): string {
  if (typeof value === "string") return reprString(value);
  if (value === null || value === undefined) return "None";
  if (typeof value === "boolean") return value ? "True" : "False";
  if (Array.isArray(value)) return `[${value.map(repr).join(", ")}]`;
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : String(value);
  return JSON.stringify(value);
}

/** Python's `sorted` of strings (by code point). */
function sortedKeys(keys: Iterable<string>): string[] {
  return [...keys].sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
}

function keysOf(obj: Obj): string[] {
  return Object.keys(obj);
}

function has(obj: Obj, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(obj, key);
}

// ------------------------------------------------------------------------------- reading
function merge(doc: Obj, errors: string[]): Obj {
  const settings = structuredClone(DEFAULTS) as unknown as Obj;
  if (has(doc, "seed")) settings.seed = doc.seed;
  for (const block of BLOCKS) {
    const given = has(doc, block) ? doc[block] : {};
    if (!isMapping(given)) {
      errors.push(`${block} must be an object`);
      continue;
    }
    const known = keysOf(DEFAULTS[block] as unknown as Obj);
    const unknown = sortedKeys(keysOf(given).filter((key) => !known.includes(key)));
    if (unknown.length > 0) errors.push(`${block} has unknown key(s) ${repr(unknown)} (expected ${repr(sortedKeys(known))})`);
    const target = settings[block] as Obj;
    for (const key of keysOf(given)) if (known.includes(key)) target[key] = structuredClone(given[key]);
  }
  return settings;
}

function checkFeatures(features: Obj, errors: string[]): void {
  const kind = features.type;
  if (typeof kind !== "string" || !(FEATURE_TYPES as readonly string[]).includes(kind)) {
    errors.push(`features.type must be one of ${repr([...FEATURE_TYPES])}, got ${repr(kind)}`);
    return;
  }
  for (const key of ["maxFeatures", "artLongSide"]) {
    const value = features[key];
    if (!isWhole(value) || value < 1) errors.push(`features.${key} must be a whole number of at least 1`);
  }
  const mask = features.mask;
  const rectOk = (rect: unknown): boolean =>
    Array.isArray(rect) &&
    rect.length === 4 &&
    rect.every((v) => isNumber(v) && 0 <= v && v <= 1) &&
    (rect[0] as number) < (rect[2] as number) &&
    (rect[1] as number) < (rect[3] as number);
  if (!Array.isArray(mask) || !mask.every(rectOk)) {
    errors.push("features.mask must be a list of [x0, y0, x1, y1] fractions from 0 to 1, x0 < x1 and y0 < y1");
  }
  const params = features.params;
  if (!isMapping(params)) {
    errors.push("features.params must be an object");
    return;
  }
  const defaults = PARAM_DEFAULTS[kind as FeatureType];
  const known = defaults.map(([key]) => key);
  const unknown = sortedKeys(keysOf(params).filter((key) => !known.includes(key)));
  if (unknown.length > 0) {
    errors.push(`features.params has unknown key(s) ${repr(unknown)} for ${kind} (expected ${repr(sortedKeys(known))})`);
  }
  const full: Obj = {};
  for (const [key, value] of defaults) full[key] = has(params, key) ? params[key] : value;
  for (const [key, , whole] of defaults) {
    const value = full[key];
    if ((whole && !isWhole(value)) || !isNumber(value) || value <= 0) {
      errors.push(`features.params.${key} must be a ${whole ? "whole " : ""}number above 0`);
    }
  }
  features.params = full;
}

function check(settings: Obj, errors: string[]): void {
  const seed = settings.seed;
  if (!isWhole(seed) || seed < 0) errors.push("seed must be a whole number of 0 or more");
  checkFeatures(settings.features as Obj, errors);
  const crop = settings.crop as Obj;
  const candidates = settings.candidates as Obj;
  const vote = settings.vote as Obj;
  const match = settings.match as Obj;
  const ransac = settings.ransac as Obj;
  const fusion = settings.fusion as Obj;
  const wholes: Array<[string, Obj, string, number]> = [
    ["crop", crop, "longSide", 1], ["crop", crop, "maxFeatures", 1],
    ["candidates", candidates, "embeddingTopK", 0], ["candidates", candidates, "voteTopN", 0], ["ransac", ransac, "maxIters", 1],
    ["vote", vote, "lshTables", 1], ["vote", vote, "lshKeySize", 1], ["vote", vote, "lshProbes", 0], ["vote", vote, "kdTrees", 1],
    ["vote", vote, "checks", 1],
    ["fusion", fusion, "minInliers", 0], ["fusion", fusion, "fullInliers", 1],
  ];
  for (const [where, block, key, low] of wholes) {
    const value = block[key];
    if (!isWhole(value) || value < low) errors.push(`${where}.${key} must be a whole number of at least ${low}`);
  }
  if (isWhole(candidates.embeddingTopK) && isWhole(candidates.voteTopN) && candidates.embeddingTopK + candidates.voteTopN === 0) {
    errors.push("candidates: embeddingTopK and voteTopN cannot both be 0");
  }
  if (!(MATCHERS as readonly unknown[]).includes(vote.matcher)) errors.push(`vote.matcher must be one of ${repr([...MATCHERS])}`);
  for (const [where, block] of [["vote", vote], ["match", match]] as const) {
    const ratio = block.ratio;
    if (!isNumber(ratio) || !(0 < ratio && ratio <= 1)) errors.push(`${where}.ratio must be a number above 0 and at most 1`);
  }
  if (!(MODELS as readonly unknown[]).includes(ransac.model)) errors.push(`ransac.model must be one of ${repr([...MODELS])}`);
  if (!(METHODS as readonly unknown[]).includes(ransac.method)) errors.push(`ransac.method must be one of ${repr([...METHODS])}`);
  else if (ransac.method === "usac" && ransac.model === "similarity") {
    errors.push('ransac.method "usac" is not available for the "similarity" model in OpenCV; use "ransac"');
  }
  if (!isNumber(ransac.threshold) || ransac.threshold <= 0) errors.push("ransac.threshold must be a number above 0");
  if (!isNumber(ransac.confidence) || !(0 < ransac.confidence && ransac.confidence < 1)) {
    errors.push("ransac.confidence must be a number above 0 and below 1");
  }
  const low = ransac.minAreaRatio;
  const high = ransac.maxAreaRatio;
  if (!isNumber(low) || !isNumber(high) || low < 0 || high < 0 || (high !== 0 && low > high)) {
    errors.push("ransac.minAreaRatio and maxAreaRatio must be numbers of 0 or more, the first at most the second");
  }
  for (const key of ["weight", "penalty"]) {
    const value = fusion[key];
    if (!isNumber(value) || value < 0) errors.push(`fusion.${key} must be a number of 0 or more`);
  }
  const gate = settings.gate as Obj;
  if (typeof gate.belowAccept !== "boolean") errors.push("gate.belowAccept must be true or false");
  const sources = gate.sources;
  if (sources !== null && (!Array.isArray(sources) || sources.length === 0 || !sources.every((item) => typeof item === "string" && item.length > 0))) {
    errors.push('gate.sources must be null or a non-empty list of source types or patterns, e.g. ["physical/*"]');
  }
  if (!(BASELINES as readonly unknown[]).includes(fusion.baseline)) errors.push(`fusion.baseline must be one of ${repr([...BASELINES])}`);
  if (typeof fusion.reorder !== "boolean") errors.push("fusion.reorder must be true or false");
}

/**
 * Read settings from a parsed document. `base` is the URL a relative index path is resolved
 * against; without it the path is kept as written. Throws a `GeometricConfigError` listing
 * every problem, each prefixed with `source`.
 */
export function parseGeometric(doc: unknown, base: string | URL | null = null, source = "geometric settings"): GeometricConfig {
  if (!isMapping(doc)) throw new GeometricConfigError([`${source}: must be a JSON object`]);
  const errors: string[] = [];
  const unknown = sortedKeys(keysOf(doc).filter((key) => !TOP_KEYS.includes(key)));
  if (unknown.length > 0) errors.push(`unknown key(s) ${repr(unknown)} (expected ${repr(TOP_KEYS)})`);
  const index = has(doc, "index") ? doc.index : {};
  let path: string | null = null;
  let sha: string | null = null;
  if (!isMapping(index) || keysOf(index).some((key) => key !== "path" && key !== "sha256")) {
    errors.push('index must be an object with the keys "path" and "sha256"');
  } else {
    const given = has(index, "path") ? index.path : null;
    if (given !== null && given !== undefined) {
      if (typeof given !== "string" || given.length === 0) errors.push("index.path must be a file path");
      else path = base === null ? given : new URL(given, base).href;
    }
    const pinned = has(index, "sha256") ? index.sha256 : null;
    if (pinned !== null && pinned !== undefined) {
      if (typeof pinned !== "string" || !SHA256.test(pinned)) errors.push("index.sha256 must be 64 lowercase hex digits");
      else sha = pinned;
    }
  }
  const settings = merge(doc, errors);
  check(settings, errors);
  if (errors.length > 0) throw new GeometricConfigError(errors.map((error) => `${source}: ${error}`));
  return new GeometricConfig(Object.freeze(settings) as unknown as GeometricSettings, path, sha);
}

/** Read a settings document's text; `source` names it in messages (Python: load_geometric). */
export function parseGeometricText(text: string, base: string | URL | null = null, source = "geometric settings"): GeometricConfig {
  let doc: unknown;
  try {
    doc = JSON.parse(text);
  } catch (error) {
    throw new GeometricConfigError([`${source}: not valid JSON: ${error instanceof Error ? error.message : String(error)}`]);
  }
  return parseGeometric(doc, base, source);
}
