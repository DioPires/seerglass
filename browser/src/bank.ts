// SPDX-License-Identifier: Apache-2.0
// The embedding bank: the reference embeddings a crop is compared against.
// In Python: cardvision/core/bank.py (the storage format) and cardvision/bank/build.py.
//
// A bank file is one JSON document. Each entry holds a card id and its embedding as base64 of
// little-endian IEEE half floats (fp16). A card may have several entries. Half floats keep the
// file small, at the cost that a stored row is no longer exactly unit length, so every reader
// renormalizes: each stored row is widened to double precision, its norm is taken in double
// precision, and the quotient is narrowed to float32. Every runtime then scores against the
// same matrix.

import { contractOf, isPreprocessingId, PREPROCESSING, type PreprocessingId } from "./preprocess.js";

export const BANK_KIND = "embedding-bank";

/** A parsed bank: row `i` of `matrix` (`dim` values, unit length) belongs to `ids[i]`. */
export interface EmbeddingBank {
  ids: string[];
  /** `ids.length * dim` values, row-major. */
  matrix: Float32Array;
  dim: number;
  /** The bank's own version number. */
  version: number;
  /** The label of the recognizer that made the rows. */
  model: string;
  /** The preprocessing contract the rows were made with. */
  preprocessing: PreprocessingId;
  /** The accept threshold the file records, when it records one. */
  tau: number | null;
}

/** Decode base64 text of `dim` little-endian half floats; null when it is anything else. */
export function decodeFp16Base64(b64: string, dim: number): Float32Array | null {
  let raw: string;
  try {
    raw = atob(b64);
  } catch {
    return null;
  }
  if (raw.length !== dim * 2) return null;
  const out = new Float32Array(dim);
  for (let i = 0; i < dim; i++) {
    const h = raw.charCodeAt(i * 2) | (raw.charCodeAt(i * 2 + 1) << 8);
    const sign = h & 0x8000 ? -1 : 1;
    const exp = (h >> 10) & 0x1f;
    const frac = h & 0x3ff;
    if (exp === 0) out[i] = sign * frac * 2 ** -24;
    else if (exp === 31) out[i] = frac ? NaN : sign * Infinity;
    else out[i] = sign * (1 + frac / 1024) * 2 ** (exp - 15);
  }
  return out;
}

/** The contract a bank's `preprocessing` field names: an id, or the contract written out as an
 *  object that equals it field for field. Null for anything this runtime cannot reproduce. */
function bankContract(raw: unknown): PreprocessingId | null {
  return isPreprocessingId(raw) ? raw : contractOf(raw);
}

/**
 * Check and unpack a bank document. The whole document is refused (null) on any bad entry: a
 * bank with rows missing would name the wrong cards, which is worse than naming none.
 */
export function parseBank(raw: unknown): EmbeddingBank | null {
  if (typeof raw !== "object" || raw === null) return null;
  const obj = raw as Record<string, unknown>;
  if (
    obj.kind !== BANK_KIND ||
    typeof obj.version !== "number" ||
    !Number.isInteger(obj.version) ||
    typeof obj.model !== "string" ||
    typeof obj.dim !== "number" ||
    !Number.isInteger(obj.dim) ||
    obj.dim <= 0 ||
    !Array.isArray(obj.entries) ||
    obj.entries.length === 0
  ) {
    return null;
  }
  const preprocessing = bankContract(obj.preprocessing);
  if (preprocessing === null) return null;
  if (obj.centerMask !== undefined && obj.centerMask !== PREPROCESSING[preprocessing].centerMask) return null;
  if (obj.tau !== undefined && typeof obj.tau !== "number") return null;
  const dim = obj.dim;
  const ids: string[] = [];
  const matrix = new Float32Array(obj.entries.length * dim);
  for (let i = 0; i < obj.entries.length; i++) {
    const e = obj.entries[i] as Record<string, unknown> | null;
    if (typeof e !== "object" || e === null || typeof e.cardId !== "string" || typeof e.vec !== "string") return null;
    const v = decodeFp16Base64(e.vec, dim);
    if (!v) return null;
    let norm = 0;
    for (let k = 0; k < dim; k++) norm += v[k]! * v[k]!;
    norm = Math.sqrt(norm);
    if (!(norm > 0) || !Number.isFinite(norm)) return null;
    for (let k = 0; k < dim; k++) matrix[i * dim + k] = v[k]! / norm;
    ids.push(e.cardId);
  }
  return { ids, matrix, dim, version: obj.version, model: obj.model, preprocessing, tau: typeof obj.tau === "number" ? obj.tau : null };
}
