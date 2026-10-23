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
//
// A packed bank (`cardvision pack-bank`) is the same bank in one binary file, every value a
// signed 8-bit integer: the magic `89 63 76 62 61 6E 6B 0A`, the header's length as an unsigned
// 32-bit little-endian integer, the header (UTF-8 JSON: the bank's keys, with `kind`
// "embedding-bank-packed", `dtype` "int8", `levels`, `source` and the row `ids`), then the rows,
// row-major. It decodes the same way; its squares and sums are small integers, so the norm is
// exact and the matrix is Python's bit for bit. `readBankFile` reads either kind of file.

import { contractOf, isPreprocessingId, PREPROCESSING, type PreprocessingId } from "./preprocess.js";

export const BANK_KIND = "embedding-bank";
export const PACKED_BANK_KIND = "embedding-bank-packed";
/** The first bytes of a packed bank. */
const PACKED_MAGIC = [0x89, 0x63, 0x76, 0x62, 0x61, 0x6e, 0x6b, 0x0a];
const PACKED_PREFIX = PACKED_MAGIC.length + 4;

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

/** The fields every bank's document or header gives, checked; null when one is wrong. */
function bankFields(obj: Record<string, unknown>): Omit<EmbeddingBank, "ids" | "matrix"> | null {
  if (typeof obj.version !== "number" || !Number.isInteger(obj.version) || typeof obj.model !== "string") return null;
  if (typeof obj.dim !== "number" || !Number.isInteger(obj.dim) || obj.dim <= 0) return null;
  const preprocessing = bankContract(obj.preprocessing);
  if (preprocessing === null) return null;
  if (obj.centerMask !== undefined && obj.centerMask !== PREPROCESSING[preprocessing].centerMask) return null;
  if (obj.tau !== undefined && typeof obj.tau !== "number") return null;
  return { dim: obj.dim, version: obj.version, model: obj.model, preprocessing, tau: typeof obj.tau === "number" ? obj.tau : null };
}

/** Write `values` (one stored row) into `matrix` at row `row`, divided by their norm taken in
 *  double precision. False when the row has no finite, non-zero norm. */
function putUnitRow(values: ArrayLike<number>, matrix: Float32Array, row: number, dim: number): boolean {
  let norm = 0;
  for (let k = 0; k < dim; k++) norm += values[k]! * values[k]!;
  norm = Math.sqrt(norm);
  if (!(norm > 0) || !Number.isFinite(norm)) return false;
  for (let k = 0; k < dim; k++) matrix[row * dim + k] = values[k]! / norm;
  return true;
}

/**
 * Check and unpack a bank document. The whole document is refused (null) on any bad entry: a
 * bank with rows missing would name the wrong cards, which is worse than naming none.
 */
export function parseBank(raw: unknown): EmbeddingBank | null {
  if (typeof raw !== "object" || raw === null) return null;
  const obj = raw as Record<string, unknown>;
  if (obj.kind !== BANK_KIND || (obj.dtype !== undefined && obj.dtype !== "fp16") || !Array.isArray(obj.entries) || obj.entries.length === 0) return null;
  const fields = bankFields(obj);
  if (fields === null) return null;
  const dim = fields.dim;
  const ids: string[] = [];
  const matrix = new Float32Array(obj.entries.length * dim);
  for (let i = 0; i < obj.entries.length; i++) {
    const e = obj.entries[i] as Record<string, unknown> | null;
    if (typeof e !== "object" || e === null || typeof e.cardId !== "string" || typeof e.vec !== "string") return null;
    const v = decodeFp16Base64(e.vec, dim);
    if (!v || !putUnitRow(v, matrix, i, dim)) return null;
    ids.push(e.cardId);
  }
  return { ids, matrix, ...fields };
}

/** Whether a file's bytes are a packed bank (they start with its magic number). */
export function isPackedBank(bytes: Uint8Array): boolean {
  return bytes.length >= PACKED_MAGIC.length && PACKED_MAGIC.every((b, i) => bytes[i] === b);
}

/** Check and unpack a packed bank, all or nothing, as `parseBank` does a JSON one. */
export function parsePackedBank(bytes: Uint8Array): EmbeddingBank | null {
  if (!isPackedBank(bytes) || bytes.length < PACKED_PREFIX) return null;
  const length = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(PACKED_MAGIC.length, true);
  const end = PACKED_PREFIX + length;
  if (length === 0 || end > bytes.length) return null;
  let header: unknown;
  try {
    header = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes.subarray(PACKED_PREFIX, end)));
  } catch {
    return null;
  }
  if (typeof header !== "object" || header === null || Array.isArray(header)) return null;
  const obj = header as Record<string, unknown>;
  if (obj.kind !== PACKED_BANK_KIND || obj.schemaVersion !== 1 || obj.dtype !== "int8" || "entries" in obj) return null;
  const { levels, ids, source } = obj;
  if (typeof levels !== "number" || !Number.isInteger(levels) || levels < 1 || levels > 127) return null;
  if (!Array.isArray(ids) || ids.length === 0 || !ids.every((id) => typeof id === "string")) return null;
  if (typeof source !== "object" || source === null) return null;
  const src = source as Record<string, unknown>;
  if (typeof src.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(src.sha256) || typeof src.dtype !== "string") return null;
  const fields = bankFields(obj);
  if (fields === null) return null;
  const dim = fields.dim;
  if (bytes.length - end !== ids.length * dim) return null;
  const values = new Int8Array(bytes.buffer, bytes.byteOffset + end, ids.length * dim);
  for (let i = 0; i < values.length; i++) if (values[i]! === -128 || Math.abs(values[i]!) > levels) return null;
  const matrix = new Float32Array(ids.length * dim);
  for (let r = 0; r < ids.length; r++) if (!putUnitRow(values.subarray(r * dim, (r + 1) * dim), matrix, r, dim)) return null;
  return { ids: ids as string[], matrix, ...fields };
}

/** A bank file of either kind, packed or JSON; null when it is neither, or is refused. */
export function readBankFile(bytes: Uint8Array): EmbeddingBank | null {
  if (isPackedBank(bytes)) return parsePackedBank(bytes);
  try {
    return parseBank(JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes)));
  } catch {
    return null;
  }
}
