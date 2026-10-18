// SPDX-License-Identifier: Apache-2.0
// The geometric stage's fusion: a verified candidate's evidence and fused score, and the record
// of what the stage found. In Python: cardvision/geometric/stage.py (`fused_score`, `Verified`,
// `GeometricOutcome`) and cardvision/recognize/recognizer.py (`_geometric_recognition`).
//
// Every number is a double computed in Python's order of operations, so the fused scores are
// the same bits as Python's for the same cosines and inlier counts.

import type { GeometricSettings } from "./config.js";
import type { StageDebug } from "./debug.js";

export const EMBEDDING = "embedding";
export const VOTE = "vote";
export const BOTH = "both";
export type FoundBy = typeof EMBEDDING | typeof VOTE | typeof BOTH;

/** One verified candidate (Python: `Verified`). */
export interface Verified {
  cardId: string;
  /** Its cosine as the caller gave it; for a card the vote added that the caller did not rank,
   *  the lowest cosine the caller gave. */
  cosine: number;
  foundBy: FoundBy;
  /** Crop keypoints that voted for it (0 when the vote is off). */
  votes: number;
  /** Matched keypoints that passed the ratio test. */
  matches: number;
  /** Matches the fitted model explains (0 with no fit). */
  inliers: number;
  /** The inliers less the baseline, at least 0. */
  evidence: number;
  fused: number;
  /** Whether the index has the card; a card it lacks keeps its cosine. */
  indexed: boolean;
}

/** What the stage found for one crop (Python: `StageResult`). */
export interface StageResult {
  /** The verified candidates in the order they were chosen: embedding rank, then vote rank. */
  candidates: Verified[];
  cropKeypoints: number;
  ms: number;
  /** The stage's debug record of the crop, when it was asked for one (debug.ts). */
  debug?: StageDebug;
}

/** A verified candidate as a report records it (Python: `Verified.to_json`). */
export interface VerifiedJson {
  cardId: string;
  cosine: number;
  foundBy: FoundBy;
  votes: number;
  matches: number;
  inliers: number;
  evidence: number;
  fused: number;
}

/** The stage's part in one recognition (Python: `GeometricOutcome.to_json`): what the
 *  recognizer would have answered without it, and the verified candidates. */
export interface GeometricEvidence {
  before: { topId: string; similarity: number; margin: number; accepted: boolean };
  candidates: VerifiedJson[];
  /** The stage's debug record, when it was asked for one (Python: the same key). */
  debug?: StageDebug;
}

/** A verified candidate's fused score from its evidence (Python: `fused_score`). */
export function fusedScore(cosine: number, evidence: number, fusion: GeometricSettings["fusion"]): number {
  if (evidence >= fusion.minInliers) {
    return cosine + (fusion.weight * Math.min(evidence, fusion.fullInliers)) / fusion.fullInliers;
  }
  return cosine - fusion.penalty;
}

/** NumPy's median of whole numbers: the middle value, or the mean of the two middle ones. */
export function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  if (n === 0) return Number.NaN;
  const mid = n >> 1;
  return n % 2 === 1 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2;
}

/** The baseline the inliers are measured from (`fusion.baseline`): 0, or the median inlier
 *  count of the crop's verified candidates. */
export function baselineOf(inliers: readonly number[], fusion: GeometricSettings["fusion"]): number {
  return fusion.baseline === "median" && inliers.length > 0 ? median(inliers) : 0;
}

/** A candidate's evidence: its inliers less the baseline, never below 0. */
export function evidenceOf(inliers: number, baseline: number): number {
  return Math.max(0, inliers - baseline);
}

/**
 * A row of similarities with each verified card's rows moved by its fused score less its
 * cosine (Python: `fused[column] += found.fused - found.cosine`). `columnsOf` gives a card's
 * bank rows. The row given is not changed.
 */
export function fuseRow(row: Float64Array, candidates: readonly Verified[], columnsOf: (cardId: string) => readonly number[]): Float64Array {
  const fused = new Float64Array(row);
  for (const found of candidates) {
    const shift = found.fused - found.cosine;
    for (const column of columnsOf(found.cardId)) fused[column] = fused[column]! + shift;
  }
  return fused;
}

// ------------------------------------------------------------------------ Python's round
const TEN = 10n;

/** The exact value of a finite double as `mantissa * 2 ** exponent`. */
function exactParts(x: number): { mantissa: bigint; exponent: number } {
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, x);
  const hi = view.getUint32(0);
  const lo = view.getUint32(4);
  const sign = hi >>> 31 ? -1n : 1n;
  const biased = (hi >>> 20) & 0x7ff;
  let mantissa = (BigInt(hi & 0xfffff) << 32n) | BigInt(lo);
  let exponent: number;
  if (biased === 0) exponent = -1074;
  else {
    mantissa |= 1n << 52n;
    exponent = biased - 1075;
  }
  return { mantissa: sign * mantissa, exponent };
}

/**
 * Python's `round(x, digits)` for a float: the exact value rounded to `digits` decimals, a tie
 * to the even neighbour, the result the nearest double to that decimal.
 */
export function pyRound(x: number, digits: number): number {
  if (!Number.isFinite(x) || x === 0) return x;
  const { mantissa, exponent } = exactParts(x);
  const negative = mantissa < 0n;
  const m = negative ? -mantissa : mantissa;
  // x * 10**digits = m * 2**exponent * 10**digits = num / den.
  let num = m * TEN ** BigInt(digits);
  let den = 1n;
  if (exponent >= 0) num <<= BigInt(exponent);
  else den <<= BigInt(-exponent);
  let q = num / den;
  const r = num - q * den;
  const twice = 2n * r;
  if (twice > den || (twice === den && q % 2n === 1n)) q += 1n;
  if (q === 0n) return negative ? -0 : 0;
  const text = `${negative ? "-" : ""}${q.toString()}e-${digits}`;
  return Number(text);
}

/** A verified candidate as a report records it. */
export function verifiedToJson(found: Verified): VerifiedJson {
  return {
    cardId: found.cardId,
    cosine: pyRound(found.cosine, 6),
    foundBy: found.foundBy,
    votes: found.votes,
    matches: found.matches,
    inliers: found.inliers,
    evidence: pyRound(found.evidence, 6),
    fused: pyRound(found.fused, 6),
  };
}

/** The stage's record for one recognition. */
export function evidenceToJson(
  before: { topId: string; similarity: number; margin: number; accepted: boolean },
  result: StageResult,
): GeometricEvidence {
  return {
    before: { topId: before.topId, similarity: pyRound(before.similarity, 6), margin: pyRound(before.margin, 6), accepted: before.accepted },
    candidates: result.candidates.map(verifiedToJson),
    ...(result.debug ? { debug: result.debug } : {}),
  };
}
