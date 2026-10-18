// SPDX-License-Identifier: Apache-2.0
// Score an embedding against the bank: the best match and its margin.
// In Python: cardvision/recognize/looks.py.
//
// The similarity of an embedding to a bank row is their dot product (both are unit vectors, so
// this is the cosine), accumulated in double precision from the single-precision values. Two
// numbers summarize one look at a crop:
//   the score:  the highest similarity, and the bank row that has it (the first on a tie);
//   the margin: the score minus the best similarity among rows of a different identity.
// Printings of one card share an identity, so only a rival of another identity narrows the
// margin.

import type { EmbeddingBank } from "./bank.js";
import { IdentityMap } from "./identity.js";
import { PREPROCESSING, type PreprocessingContract } from "./preprocess.js";

/** The accept rule's settings, as a release document's `accept` block states them. */
export interface AcceptRule {
  /** The similarity the best match must reach. */
  tau: number;
  /** The margin it must keep over the best match of any other identity. */
  delta: number;
  /** Whether a crop that is not accepted upright is tried in its three other orientations. */
  rotationRetry: boolean;
}

/** The values calibrated for the published model and bank. A release document states its own,
 *  and another model or another game needs its own. */
export const DEFAULT_ACCEPT: Readonly<AcceptRule> = { tau: 0.62, delta: 0.05, rotationRetry: true };

/** Stands in for "the best other identity" when the bank holds a single identity. It is below
 *  any cosine, so the margin is then large and never the reason a look is refused. */
export const NO_OTHER_IDENTITY = -2;

/** One look at an embedding. */
export interface ScoredMatch {
  /** The best row's card id when the score reaches tau, else null. The margin plays no part. */
  cardId: string | null;
  /** The score: the best row's similarity. */
  similarity: number;
  /** The best row's card id, whether or not the score reaches tau. */
  topId: string;
  /** The identity `topId` belongs to. */
  identity: string;
  /** The id shown for this look when it is accepted (see IdentityMap.shown). */
  displayId: string;
  /** The score minus the best similarity of a row of a different identity. */
  margin: number;
}

/** A match of a track's pooled embedding: accepted only with both tau and the margin. */
export interface PooledMatch {
  cardId: string | null;
  similarity: number;
  margin: number;
}

export class EmbeddingRecognizer {
  /** Identity index of each bank row: rows of one identity share an index. */
  private readonly rowIdent: Int32Array;
  /** Identity index to the identity's representative card id. */
  private readonly identRep: string[] = [];
  /** The id shown for each bank row. */
  private readonly rowShown: string[];
  /** One similarity per row, reused between calls (calls do not overlap). */
  private readonly scratch: Float64Array;
  private readonly contract: PreprocessingContract;

  constructor(
    private readonly bank: EmbeddingBank,
    readonly rule: AcceptRule = DEFAULT_ACCEPT,
    identity: IdentityMap = new IdentityMap(),
  ) {
    this.contract = PREPROCESSING[bank.preprocessing];
    this.rowShown = bank.ids.map((id) => identity.shown(id));
    const index = new Map<string, number>();
    this.rowIdent = new Int32Array(bank.ids.length);
    for (let r = 0; r < bank.ids.length; r++) {
      const rep = identity.of(bank.ids[r]!);
      let k = index.get(rep);
      if (k === undefined) {
        k = this.identRep.length;
        index.set(rep, k);
        this.identRep.push(rep);
      }
      this.rowIdent[r] = k;
    }
    this.scratch = new Float64Array(bank.ids.length);
  }

  /** Rows in the bank. */
  get size(): number {
    return this.bank.ids.length;
  }

  /** Length of one embedding. */
  get dim(): number {
    return this.bank.dim;
  }

  get tau(): number {
    return this.rule.tau;
  }

  /** The preprocessing contract a crop goes through before it is embedded. */
  get preprocessing(): PreprocessingContract {
    return this.contract;
  }

  /** Score every row for the vector at `offset`: the best row, its score, and the best score of
   *  a row of a different identity. */
  private topAndOther(vec: Float32Array, offset: number): { bestRow: number; best: number; other: number } {
    const { matrix, dim, ids } = this.bank;
    const sims = this.scratch;
    let best = NO_OTHER_IDENTITY;
    let bestRow = -1;
    for (let r = 0; r < ids.length; r++) {
      let dot = 0;
      const rowBase = r * dim;
      for (let k = 0; k < dim; k++) dot += matrix[rowBase + k]! * vec[offset + k]!;
      sims[r] = dot;
      if (dot > best) {
        best = dot;
        bestRow = r;
      }
    }
    let other = NO_OTHER_IDENTITY;
    const topIdent = bestRow >= 0 ? this.rowIdent[bestRow]! : -1;
    for (let r = 0; r < ids.length; r++) {
      if (this.rowIdent[r] !== topIdent && sims[r]! > other) other = sims[r]!;
    }
    return { bestRow, best, other };
  }

  /** One look at the embedding stored at `offset` in `embs`. */
  matchScored(embs: Float32Array, offset: number): ScoredMatch {
    const { bestRow, best, other } = this.topAndOther(embs, offset);
    const topId = this.bank.ids[bestRow]!;
    return {
      cardId: best >= this.rule.tau ? topId : null,
      similarity: best,
      topId,
      identity: this.identRep[this.rowIdent[bestRow]!]!,
      displayId: this.rowShown[bestRow]!,
      margin: best - other,
    };
  }

  // ----------------------------------------------------------- the geometric stage's seams

  /** The similarity of the embedding at `offset` to every bank row, a new array. The values
   *  are those `matchScored` compares, computed the same way. */
  scoreRow(embs: Float32Array, offset: number): Float64Array {
    const { matrix, dim, ids } = this.bank;
    const sims = new Float64Array(ids.length);
    for (let r = 0; r < ids.length; r++) {
      let dot = 0;
      const rowBase = r * dim;
      for (let k = 0; k < dim; k++) dot += matrix[rowBase + k]! * embs[offset + k]!;
      sims[r] = dot;
    }
    return sims;
  }

  /**
   * The cards of a row of similarities, best first, each once with its best row's similarity
   * (Python: `_geometric_recognition`'s ranking: a stable argsort of the row, best first, each
   * card id kept at its first row).
   */
  rankCards(row: Float64Array): Array<{ cardId: string; cosine: number }> {
    const order = Array.from({ length: row.length }, (_, i) => i);
    order.sort((a, b) => row[b]! - row[a]!);
    const seen = new Set<string>();
    const ranked: Array<{ cardId: string; cosine: number }> = [];
    for (const r of order) {
      const cardId = this.bank.ids[r]!;
      if (seen.has(cardId)) continue;
      seen.add(cardId);
      ranked.push({ cardId, cosine: row[r]! });
    }
    return ranked;
  }

  /** The bank rows of each card id. */
  private columns: Map<string, number[]> | null = null;

  /** The bank rows of a card id (empty for an id the bank lacks). */
  columnsOf(cardId: string): readonly number[] {
    if (this.columns === null) {
      this.columns = new Map();
      this.bank.ids.forEach((id, r) => {
        const list = this.columns!.get(id);
        if (list) list.push(r);
        else this.columns!.set(id, [r]);
      });
    }
    return this.columns.get(cardId) ?? [];
  }

  private bankIdSet: ReadonlySet<string> | null = null;

  /** The card ids the bank holds. */
  get bankIds(): ReadonlySet<string> {
    this.columnsOf("");
    this.bankIdSet ??= new Set(this.columns!.keys());
    return this.bankIdSet;
  }

  private scored(top: number, score: number, other: number): ScoredMatch & { row: number } {
    const topId = this.bank.ids[top]!;
    return {
      cardId: score >= this.rule.tau ? topId : null,
      similarity: score,
      topId,
      identity: this.identRep[this.rowIdent[top]!]!,
      displayId: this.rowShown[top]!,
      margin: score - other,
      row: top,
    };
  }

  /**
   * One look from a full row of scores (Python: `score_look`): the first row with the highest
   * score, and the margin over the best row of a different identity (`NO_OTHER_IDENTITY` when
   * there is none). `row` in the result is the best row.
   */
  lookOf(row: Float64Array): ScoredMatch & { row: number } {
    let top = 0;
    for (let r = 1; r < row.length; r++) if (row[r]! > row[top]!) top = r;
    const ident = this.rowIdent[top]!;
    let other = NO_OTHER_IDENTITY;
    let any = false;
    for (let r = 0; r < row.length; r++) {
      if (this.rowIdent[r] === ident) continue;
      if (!any || row[r]! > other) other = row[r]!;
      any = true;
    }
    return this.scored(top, row[top]!, other);
  }

  /**
   * The accept booster's look (Python: `_geometric_recognition` without `fusion.reorder`): the
   * identity of bank row `keep` stays the answer, named by that identity's first highest row,
   * and the margin is taken over the best row of any other identity.
   */
  lookWithin(row: Float64Array, keep: number): ScoredMatch & { row: number } {
    const ident = this.rowIdent[keep]!;
    let top = -1;
    let other = NO_OTHER_IDENTITY;
    let any = false;
    for (let r = 0; r < row.length; r++) {
      if (this.rowIdent[r] === ident) {
        if (top < 0 || row[r]! > row[top]!) top = r;
      } else {
        if (!any || row[r]! > other) other = row[r]!;
        any = true;
      }
    }
    return this.scored(top, row[top]!, other);
  }

  /**
   * Match a track's pooled embedding (the unit-length mean of its recent accepted embeddings,
   * see `poolEmbeddings`). Accepted only when the score reaches tau and the margin reaches
   * delta, so a track torn between two identities names nothing instead of switching between
   * them, while two printings of one identity never hold each other back. Returns the id shown
   * for the best row.
   */
  matchPooled(vec: Float32Array, delta: number = this.rule.delta): PooledMatch {
    const { bestRow, best, other } = this.topAndOther(vec, 0);
    const margin = best - other;
    const cardId = best >= this.rule.tau && margin >= delta ? this.rowShown[bestRow]! : null;
    return { cardId, similarity: best, margin };
  }
}
