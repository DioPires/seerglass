// SPDX-License-Identifier: Apache-2.0
// Bank scoring, timed two ways on the same bank and the same looks: the runtime's own loop
// (EmbeddingRecognizer.matchScored, which calls topAndOther) and the scoring graph that
// cardvision/mtg/scorer.py writes (MatMul, TopK, the best row of another identity) in an ONNX
// Runtime session. Shared by the Node run (scoring.mjs) and the browser page
// (scoring-webgpu.mjs); `ort` is passed in, so each side brings its own build of ONNX Runtime.
//
// The bench data comes from `python -m cardvision.mtg scorer --rows N --out DIR`: a bank in the
// runtime's JSON format, its identity groups, a frame of looks, the graphs, and the float64
// reference answer.

import { parseBank } from "../src/bank.js";
import { parseIdentityGroups, IdentityMap } from "../src/identity.js";
import { DEFAULT_ACCEPT, EmbeddingRecognizer } from "../src/recognizer.js";

export interface BenchData {
  meta: { rows: number; dim: number; looks: number; topK: number };
  bankJson: string;
  bankF16: ArrayBuffer;
  groupsJson: string;
  queries: Float32Array;
  reference: { bestRow: number[]; best: number[]; other: number[] };
  models: Record<string, Uint8Array>;
}

export interface BenchOptions {
  providers: string[];
  runs: number;
  /** Looks per frame to time: each is the first L looks of the data. */
  frames: number[];
  now: () => number;
}

export interface Row {
  what: string;
  rows: number;
  looks: number;
  msPerFrame: number;
  msPerLook: number;
  note?: string;
}

async function sha256(text: string): Promise<void> {
  await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
}

/** The fp16 bank as stored in a binary file, decoded and renormalized as parseBank does. */
function decodeBinary(buffer: ArrayBuffer, rows: number, dim: number): Float32Array {
  const half = new Uint16Array(buffer);
  const out = new Float32Array(rows * dim);
  const row = new Float64Array(dim);
  for (let r = 0; r < rows; r++) {
    let norm = 0;
    for (let k = 0; k < dim; k++) {
      const h = half[r * dim + k]!;
      const exp = (h >> 10) & 0x1f;
      const frac = h & 0x3ff;
      const sign = h & 0x8000 ? -1 : 1;
      const v = exp === 0 ? sign * frac * 2 ** -24 : sign * (1 + frac / 1024) * 2 ** (exp - 15);
      row[k] = v;
      norm += v * v;
    }
    norm = Math.sqrt(norm);
    for (let k = 0; k < dim; k++) out[r * dim + k] = row[k]! / norm;
  }
  return out;
}

export async function runBench(ort: any, data: BenchData, options: BenchOptions): Promise<{ rows: Row[]; checks: Record<string, unknown> }> {
  const { now } = options;
  const { rows: n, dim } = data.meta;
  const out: Row[] = [];
  const checks: Record<string, unknown> = {};

  // Loading: the JSON bank as the runtime reads it, and the same rows from a binary file.
  let t = now();
  await sha256(data.bankJson);
  const tHash = now() - t;
  t = now();
  const doc = JSON.parse(data.bankJson);
  const tParse = now() - t;
  t = now();
  const bank = parseBank(doc);
  const tDecode = now() - t;
  if (!bank) throw new Error("the bench bank does not parse");
  t = now();
  const binary = decodeBinary(data.bankF16, n, dim);
  const tBinary = now() - t;
  let maxDiff = 0;
  for (let i = 0; i < binary.length; i += 97) maxDiff = Math.max(maxDiff, Math.abs(binary[i]! - bank.matrix[i]!));
  checks.load = {
    rows: n,
    jsonMB: +(data.bankJson.length / 1e6).toFixed(1),
    sha256Ms: +tHash.toFixed(1),
    jsonParseMs: +tParse.toFixed(1),
    parseBankMs: +tDecode.toFixed(1),
    binaryMB: +(data.bankF16.byteLength / 1e6).toFixed(1),
    binaryDecodeMs: +tBinary.toFixed(1),
    binaryMaxDiffFromJson: maxDiff,
  };

  // The runtime's loop, one look at a time, as the pipeline calls it.
  const groups = parseIdentityGroups(JSON.parse(data.groupsJson));
  const recognizer = new EmbeddingRecognizer(bank, DEFAULT_ACCEPT, new IdentityMap(groups));
  const rowOf = new Map(bank.ids.map((id, r) => [id, r]));
  const jsBest: number[] = [];
  const jsMargin: number[] = [];
  for (let l = 0; l < data.meta.looks; l++) {
    const m = recognizer.matchScored(data.queries, l * dim);
    jsBest.push(rowOf.get(m.topId)!);
    jsMargin.push(m.margin);
  }
  let refAgree = 0;
  for (let l = 0; l < data.meta.looks; l++) if (jsBest[l] === data.reference.bestRow[l]) refAgree++;
  checks.loopAgreesWithReference = `${refAgree} / ${data.meta.looks}`;
  for (const looks of options.frames) {
    const runs = Math.max(1, Math.round(options.runs / Math.max(1, looks / 8)));
    recognizer.matchScored(data.queries, 0);
    t = now();
    for (let r = 0; r < runs; r++) for (let l = 0; l < looks; l++) recognizer.matchScored(data.queries, l * dim);
    const ms = (now() - t) / runs;
    out.push({ what: "js loop (matchScored)", rows: n, looks, msPerFrame: ms, msPerLook: ms / looks });
  }

  // The graph, on each provider.
  for (const provider of options.providers) {
    for (const [name, bytes] of Object.entries(data.models)) {
      if (name.includes("bank-input")) continue;
      let session: any;
      t = now();
      try {
        session = await ort.InferenceSession.create(bytes, { executionProviders: [provider], graphOptimizationLevel: "all" });
      } catch (error) {
        out.push({ what: `${provider} ${name}`, rows: n, looks: 0, msPerFrame: NaN, msPerLook: NaN, note: `failed: ${String(error).slice(0, 200)}` });
        continue;
      }
      const tCreate = now() - t;
      let agree = 0;
      let worst = 0;
      for (const looks of options.frames) {
        const feeds = { queries: new ort.Tensor("float32", data.queries.subarray(0, looks * dim), [looks, dim]) };
        let result = await session.run(feeds);
        await session.run(feeds);
        if (looks === data.meta.looks) {
          const top = result.topRows.data as BigInt64Array;
          const scores = result.topScores.data as Float32Array;
          const other = result.other.data as Float32Array;
          const k = result.topRows.dims[1];
          for (let l = 0; l < looks; l++) {
            if (Number(top[l * k]) === jsBest[l]) agree++;
            worst = Math.max(worst, Math.abs(scores[l * k]! - other[l]! - jsMargin[l]!));
          }
        }
        const runs = Math.max(2, Math.round(options.runs / Math.max(1, looks / 8)));
        t = now();
        for (let r = 0; r < runs; r++) result = await session.run(feeds);
        const ms = (now() - t) / runs;
        out.push({ what: `${provider} ${name}`, rows: n, looks, msPerFrame: ms, msPerLook: ms / looks,
          note: looks === options.frames[0] ? `session created in ${tCreate.toFixed(0)} ms` : undefined });
      }
      checks[`${provider} ${name}`] = { bestRowAgreesWithLoop: `${agree} / ${data.meta.looks}`, maxMarginDiff: worst };
      await session.release?.();
    }
  }
  return { rows: out, checks };
}
