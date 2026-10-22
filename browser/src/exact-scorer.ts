// SPDX-License-Identifier: Apache-2.0
// The bank search for many looks at once, in WebAssembly: the same answers as
// EmbeddingRecognizer.topAndOther, bit for bit, several times faster.
//
// The kernel (exact-kernel.c, compiled by scripts/build-kernel.mjs into exact-kernel-wasm.ts)
// computes every similarity with the loop's own float64 arithmetic and keeps the best row, its
// score and the best score of another identity with the loop's own tie rule, so nothing is
// approximated and nothing is checked afterwards. Its SIMD lanes are looks: one pass over the
// bank serves eight looks, and a frame's looks go in one call.
//
// The bank lives once, in the kernel's memory: `create` copies the bank's matrix there and points
// `bank.matrix` at that copy, so the values stay where every other reader (scoreRow, the
// geometric stage) finds them. The memory is sized once and never grows, so the view stays valid.
// Without WebAssembly SIMD, or for a bank whose rows are not 128 values long, `create` gives null
// and the recognizer keeps its loop.

import type { EmbeddingBank } from "./bank.js";
import { EXACT_KERNEL_BASE64 } from "./exact-kernel-wasm.js";

/** Values per bank row the kernel is compiled for. */
export const KERNEL_DIM = 128;
/** Looks per SIMD group; a call is padded to a whole number of groups. */
export const KERNEL_GROUP = 8;
/** Looks per kernel call; a longer batch is cut into calls of this many. */
export const KERNEL_MAX_LOOKS = 256;

const PAGE = 65536;
/** NO_OTHER_IDENTITY in recognizer.ts: the starting best and other, below any cosine. */
const START = -2;

/** One look's result, as topAndOther returns it. */
export interface BankMatch {
  bestRow: number;
  best: number;
  other: number;
}

function decodeBase64(text: string): Uint8Array {
  const raw = atob(text);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
  return bytes;
}

let compiled: Promise<WebAssembly.Module | null> | null = null;

/** The compiled kernel, or null where this engine cannot compile it (no SIMD). Compiled once. */
function kernelModule(): Promise<WebAssembly.Module | null> {
  compiled ??= WebAssembly.compile(decodeBase64(EXACT_KERNEL_BASE64) as BufferSource).catch(() => null);
  return compiled;
}

export class ExactBankScorer {
  private readonly rowIdent: Int32Array;
  private readonly queries: Float64Array;
  private readonly best: Float64Array;
  private readonly other: Float64Array;
  private readonly bestRow: Int32Array;
  private readonly bestIdent: Int32Array;
  private identitiesSet = false;

  private constructor(
    private readonly score: (...args: number[]) => void,
    memory: WebAssembly.Memory,
    private readonly at: { bank: number; ident: number; queries: number; best: number; other: number; bestRow: number; bestIdent: number },
    private readonly rows: number,
  ) {
    const buffer = memory.buffer;
    this.rowIdent = new Int32Array(buffer, at.ident, rows);
    this.queries = new Float64Array(buffer, at.queries, KERNEL_MAX_LOOKS * KERNEL_DIM);
    this.best = new Float64Array(buffer, at.best, KERNEL_MAX_LOOKS);
    this.other = new Float64Array(buffer, at.other, KERNEL_MAX_LOOKS);
    this.bestRow = new Int32Array(buffer, at.bestRow, KERNEL_MAX_LOOKS);
    this.bestIdent = new Int32Array(buffer, at.bestIdent, KERNEL_MAX_LOOKS);
  }

  /**
   * A scorer for `bank`, or null when this engine cannot run the kernel or the bank's rows are
   * not `KERNEL_DIM` values long. On success `bank.matrix` is the scorer's copy of the same
   * values, and the matrix it held before is no longer referenced by the bank.
   */
  static async create(bank: EmbeddingBank): Promise<ExactBankScorer | null> {
    if (bank.dim !== KERNEL_DIM || bank.ids.length === 0 || bank.matrix.length !== bank.ids.length * KERNEL_DIM) return null;
    const module = await kernelModule();
    if (module === null) return null;
    const rows = bank.ids.length;
    try {
      // The kernel's own stack and data come first, up to its __heap_base; the buffers follow.
      const probe = await WebAssembly.instantiate(module, { env: { memory: new WebAssembly.Memory({ initial: 2 }) } });
      let top = Number((probe.exports.__heap_base as WebAssembly.Global).value);
      const take = (bytes: number): number => {
        const start = Math.ceil(top / 16) * 16;
        top = start + bytes;
        return start;
      };
      const at = {
        bank: take(rows * KERNEL_DIM * 4),
        ident: take(rows * 4),
        queries: take(KERNEL_MAX_LOOKS * KERNEL_DIM * 8),
        best: take(KERNEL_MAX_LOOKS * 8),
        other: take(KERNEL_MAX_LOOKS * 8),
        bestRow: take(KERNEL_MAX_LOOKS * 4),
        bestIdent: take(KERNEL_MAX_LOOKS * 4),
      };
      const pages = Math.ceil(top / PAGE);
      const memory = new WebAssembly.Memory({ initial: pages, maximum: pages });
      const instance = await WebAssembly.instantiate(module, { env: { memory } });
      const matrix = new Float32Array(memory.buffer, at.bank, rows * KERNEL_DIM);
      matrix.set(bank.matrix);
      bank.matrix = matrix;
      return new ExactBankScorer(instance.exports.score as (...args: number[]) => void, memory, at, rows);
    } catch {
      // Not enough memory for the copy, or an engine that will not run the module: the loop.
      return null;
    }
  }

  /** Set the identity index of every bank row (rows of one identity share one, 0 or more).
   *  The recognizer calls it once, before any search. */
  setIdentities(rowIdent: Int32Array): void {
    if (rowIdent.length !== this.rows) throw new Error(`the bank has ${this.rows} rows, the identities ${rowIdent.length}`);
    this.rowIdent.set(rowIdent);
    this.identitiesSet = true;
  }

  /** The search for the look of `KERNEL_DIM` values at each of `offsets` in `embs`, in order. */
  search(embs: Float32Array, offsets: readonly number[]): BankMatch[] {
    if (!this.identitiesSet) throw new Error("the scorer's identities are not set");
    const out: BankMatch[] = [];
    for (let first = 0; first < offsets.length; first += KERNEL_MAX_LOOKS) {
      const count = Math.min(KERNEL_MAX_LOOKS, offsets.length - first);
      const groups = Math.ceil(count / KERNEL_GROUP);
      const q = this.queries;
      q.fill(0, 0, groups * KERNEL_GROUP * KERNEL_DIM);
      for (let l = 0; l < count; l++) {
        const base = Math.floor(l / KERNEL_GROUP) * KERNEL_DIM * KERNEL_GROUP + (l % KERNEL_GROUP);
        const off = offsets[first + l]!;
        for (let k = 0; k < KERNEL_DIM; k++) q[base + k * KERNEL_GROUP] = embs[off + k]!;
      }
      const lanes = groups * KERNEL_GROUP;
      this.best.fill(START, 0, lanes);
      this.other.fill(START, 0, lanes);
      this.bestRow.fill(-1, 0, lanes);
      this.bestIdent.fill(-1, 0, lanes);
      const at = this.at;
      this.score(at.bank, this.rows, at.ident, at.queries, groups, at.best, at.other, at.bestRow, at.bestIdent);
      for (let l = 0; l < count; l++) out.push({ bestRow: this.bestRow[l]!, best: this.best[l]!, other: this.other[l]! });
    }
    return out;
  }
}
