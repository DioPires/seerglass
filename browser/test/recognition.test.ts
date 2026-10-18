// SPDX-License-Identifier: Apache-2.0
import { describe, expect, it } from "vitest";

import type { EmbeddingBank } from "../src/bank.js";
import { rectOf } from "../src/boxes.js";
import { type BatchEmbedder, EMBED_MAX_BATCH, embedInChunks } from "../src/embedder.js";
import { rotateTensor90 } from "../src/preprocess.js";
import { ACCEPTED_NMS_IOU, chooseLook, cropRgba, keepAfterNms, lookAccepted, nmsPreOrder, recognizeFrame } from "../src/recognition.js";
import { type AcceptRule, EmbeddingRecognizer, type ScoredMatch } from "../src/recognizer.js";
import { axis, contract, solidFrame, syntheticRgba } from "./support.js";

function look(similarity: number, margin: number, topId = "A-1"): ScoredMatch {
  return { cardId: null, similarity, topId, identity: topId, displayId: topId, margin };
}

const RULE: AcceptRule = { tau: 0.62, delta: 0.05, rotationRetry: true };

describe("cropRgba", () => {
  const frame = { data: syntheticRgba(5, 20, 10), width: 20, height: 10 };

  it("cuts each edge down to a whole number after clipping to the frame", () => {
    const crop = cropRgba(frame, { x: 2.9, y: 1.2, width: 5.3, height: 4.9 });
    // x 2.9..8.2 -> 2..8, y 1.2..6.1 -> 1..6.
    expect([crop.width, crop.height]).toEqual([6, 5]);
    expect(Array.from(crop.rgba.subarray(0, 4))).toEqual(Array.from(frame.data.subarray((1 * 20 + 2) * 4, (1 * 20 + 2) * 4 + 4)));
    expect(Array.from(crop.rgba.subarray(crop.rgba.length - 4))).toEqual(Array.from(frame.data.subarray((5 * 20 + 7) * 4, (5 * 20 + 7) * 4 + 4)));
  });

  it("clips a box that reaches outside, and gives an empty crop for one that misses the frame", () => {
    expect(cropRgba(frame, { x: -3.5, y: -2, width: 8, height: 6 })).toMatchObject({ width: 4, height: 4 });
    expect(cropRgba(frame, { x: 15.5, y: 6.5, width: 30, height: 30 })).toMatchObject({ width: 5, height: 4 });
    expect(cropRgba(frame, { x: 40, y: 2, width: 5, height: 5 })).toMatchObject({ width: 0, height: 5 });
  });
});

describe("the accept rule and the orientation rule", () => {
  it("a look is accepted when its score reaches tau and its margin reaches delta, both inclusive", () => {
    expect(lookAccepted(look(0.62, 0.05), RULE)).toBe(true);
    expect(lookAccepted(look(0.6199, 0.5), RULE)).toBe(false);
    expect(lookAccepted(look(0.9, 0.0499), RULE)).toBe(false);
  });

  it("an accepted upright look is used and the turned looks are never asked for", () => {
    let asked = 0;
    const choice = chooseLook(look(0.9, 0.2), () => (asked++, [look(1, 1), look(1, 1), look(1, 1)]), RULE);
    expect([choice.orientation, choice.retried, asked]).toEqual([0, false, 0]);
  });

  it("otherwise the accepted look with the highest score wins, the lower orientation on a tie", () => {
    const upright = look(0.9, 0.01); // fails the margin
    const choice = chooseLook(upright, () => [look(0.7, 0.2), look(0.8, 0.2), look(0.8, 0.3)], RULE);
    expect([choice.orientation, choice.retried]).toEqual([2, true]);
    // An accepted look beats a look that scores higher and is not accepted.
    expect(chooseLook(upright, () => [look(0.95, 0.0), look(0.63, 0.06), look(0.5, 0.5)], RULE).orientation).toBe(2);
  });

  it("with no accepted look the highest score is reported, a turned look only when strictly higher", () => {
    const upright = look(0.5, 0.2);
    expect(chooseLook(upright, () => [look(0.5, 0.2), look(0.4, 0.2), look(0.5, 0.2)], RULE).orientation).toBe(0);
    expect(chooseLook(upright, () => [look(0.55, 0.0), look(0.61, 0.0), look(0.61, 0.0)], RULE).orientation).toBe(2);
  });

  it("asks for the turned looks at most once", () => {
    let asked = 0;
    chooseLook(look(0.1, 0.1), () => (asked++, [look(0.2, 0), look(0.3, 0), look(0.4, 0)]), RULE);
    expect(asked).toBe(1);
  });

  it("without the rotation retry the upright look is the answer, accepted or not", () => {
    let asked = 0;
    const choice = chooseLook(look(0.3, 0.0), () => (asked++, [look(1, 1), look(1, 1), look(1, 1)]), { ...RULE, rotationRetry: false });
    expect([choice.orientation, choice.retried, asked]).toEqual([0, false, 0]);
  });

  it("chooses the look the Python package chooses, case by case", () => {
    for (const [index, item] of (contract.looks.cases as Array<Record<string, unknown>>).entries()) {
      const rule = { tau: item.tau as number, delta: item.delta as number, rotationRetry: item.rotationRetry as boolean };
      const looks = (item.looks as Array<[number, number]>).map(([score, margin], k) => look(score, margin, `ROW-${k}`));
      const choice = chooseLook(looks[0]!, () => [looks[1]!, looks[2]!, looks[3]!], rule);
      const got = {
        orientation: choice.orientation,
        retried: choice.retried,
        tauPassed: choice.match.similarity >= rule.tau,
        accepted: lookAccepted(choice.match, rule),
      };
      expect(got, `case ${index}`).toEqual({ orientation: item.orientation, retried: item.retried, tauPassed: item.tauPassed, accepted: item.accepted });
    }
  });
});

describe("suppression among the proposals that reach tau", () => {
  const box = (x: number, y: number, width = 100, height = 100) => ({ x, y, width, height });

  it("enters upright looks first, then turned ones, each in proposal order", () => {
    const order = nmsPreOrder([
      { tauPassed: true, orientation: 2 },
      { tauPassed: false, orientation: 0 },
      { tauPassed: true, orientation: 0 },
      { tauPassed: true, orientation: 1 },
      { tauPassed: true, orientation: 0 },
    ]);
    expect(order).toEqual([2, 4, 0, 3]);
  });

  it("keeps the best of each overlapping cluster, whatever the identity, and drops what is under tau", () => {
    const proposals = [
      { aabb: box(0, 0), score: 0.7, tauPassed: true },
      { aabb: box(10, 0), score: 0.9, tauPassed: true }, // overlaps the first: wins
      { aabb: box(300, 0), score: 0.8, tauPassed: true },
      { aabb: box(600, 0), score: 0.99, tauPassed: false },
    ];
    expect(keepAfterNms(proposals)).toEqual([1, 2]);
  });

  it("drops at an overlap of exactly the limit, keeps just under it", () => {
    expect(ACCEPTED_NMS_IOU).toBe(0.3);
    // Two 100 x 100 boxes shifted by d overlap by (100 - d) / (100 + d): 0.3 at d = 700 / 13.
    const at = [{ aabb: box(0, 0), score: 0.9, tauPassed: true }, { aabb: box(53, 0), score: 0.8, tauPassed: true }];
    const under = [{ aabb: box(0, 0), score: 0.9, tauPassed: true }, { aabb: box(54, 0), score: 0.8, tauPassed: true }];
    expect(keepAfterNms(at)).toEqual([0]);
    expect(keepAfterNms(under)).toEqual([0, 1]);
  });

  it("on equal scores the order given decides", () => {
    const proposals = [{ aabb: box(0, 0), score: 0.8, tauPassed: true }, { aabb: box(5, 0), score: 0.8, tauPassed: true }];
    expect(keepAfterNms(proposals)).toEqual([0]);
    expect(keepAfterNms(proposals, [1, 0])).toEqual([1]);
  });
});

describe("embedInChunks", () => {
  it("runs at most the maximum at a time and joins the outputs in order", async () => {
    expect(EMBED_MAX_BATCH).toBe(32);
    const sizes: number[] = [];
    // A stand-in model: each crop's embedding is [its first value, its last value].
    const run = async (data: Float32Array, n: number): Promise<Float32Array> => {
      sizes.push(n);
      const plane = data.length / n;
      return Float32Array.from({ length: n * 2 }, (_, i) => data[Math.floor(i / 2) * plane + (i % 2 === 0 ? 0 : plane - 1)]!);
    };
    const data = Float32Array.from({ length: 70 * 3 }, (_, i) => i);
    const out = await embedInChunks(run, data, 70, 2);
    expect(sizes).toEqual([32, 32, 6]);
    expect(Array.from(out)).toEqual(Array.from({ length: 140 }, (_, i) => Math.floor(i / 2) * 3 + (i % 2 === 0 ? 0 : 2)));
  });

  it("a batch that fits is one run of the caller's own buffer", async () => {
    const data = new Float32Array(12);
    let seen: Float32Array | null = null;
    await embedInChunks(async (d, n) => ((seen = d), new Float32Array(n * 2)), data, 4, 2);
    expect(seen).toBe(data);
  });

  it("a part with the wrong output size is an error, not a misaligned result", async () => {
    await expect(embedInChunks(async () => new Float32Array(5), new Float32Array(80), 40, 2)).rejects.toThrow();
  });
});

describe("recognizeFrame", () => {
  const DIM = 4;
  const SIZE = 128;
  const PLANE = 3 * SIZE * SIZE;

  /** Bank: RED along axis 0, GREEN along axis 1, TURNED along axis 2. */
  function recognizer(rule: Partial<AcceptRule> = {}): EmbeddingRecognizer {
    const ids = ["RED-1", "GREEN-1", "TURNED-1"];
    const matrix = new Float32Array(ids.length * DIM);
    ids.forEach((_, i) => matrix.set(axis(DIM, i), i * DIM));
    const bank: EmbeddingBank = { ids, matrix, dim: DIM, version: 1, model: "fake", preprocessing: "full128", tau: null };
    return new EmbeddingRecognizer(bank, { tau: 0.62, delta: 0.05, rotationRetry: true, ...rule });
  }

  /**
   * A stand-in model that reads a tensor's corners. A solid red crop embeds as RED, a solid
   * green one as GREEN. A crop whose top-left cell alone is blue embeds as nothing in the bank
   * (axis 3) while upright, and as TURNED once the blue cell has moved to the bottom-left, which
   * is where one counter-clockwise quarter turn puts it.
   */
  class FakeEmbedder implements BatchEmbedder {
    batches: number[] = [];

    async embedBatch(data: Float32Array, n: number): Promise<Float32Array> {
      this.batches.push(n);
      const out = new Float32Array(n * DIM);
      for (let i = 0; i < n; i++) {
        const t = data.subarray(i * PLANE, (i + 1) * PLANE);
        const blueTopLeft = t[2 * SIZE * SIZE]! > 0;
        const blueBottomLeft = t[2 * SIZE * SIZE + (SIZE - 1) * SIZE]! > 0;
        let k = 3;
        if (blueBottomLeft) k = 2;
        else if (!blueTopLeft && t[0]! > 0) k = 0;
        else if (!blueTopLeft && t[SIZE * SIZE]! > 0) k = 1;
        out[i * DIM + k] = 1;
      }
      return out;
    }
  }

  /** A 400 x 300 frame: red at (0, 0), green at (150, 0), a card with a blue top-left at (0, 150). */
  function scene(): { data: Uint8ClampedArray; width: number; height: number } {
    const frame = solidFrame(400, 300);
    const paint = (x0: number, y0: number, w: number, h: number, rgb: [number, number, number]): void => {
      for (let y = y0; y < y0 + h; y++) {
        for (let x = x0; x < x0 + w; x++) frame.data.set(rgb, (y * 400 + x) * 4);
      }
    };
    paint(0, 0, 128, 128, [255, 0, 0]);
    paint(150, 0, 128, 128, [0, 255, 0]);
    paint(0, 150, 1, 1, [0, 0, 255]);
    return frame;
  }

  const candidate = (x: number, y: number, score: number, size = 128) => {
    const aabb = { x, y, width: size, height: size };
    return { aabb, rect: rectOf(aabb), score, cls: 0 };
  };

  it("embeds every crop upright in one batch, and the turned looks of the unaccepted ones in a second", async () => {
    const embedder = new FakeEmbedder();
    const frame = await recognizeFrame(scene(), embedder, recognizer(), [candidate(0, 0, 0.9), candidate(150, 0, 0.8), candidate(0, 150, 0.7)]);
    expect(embedder.batches).toEqual([3, 3]); // three crops upright, then three turns of the third
    expect([frame.retried, frame.embeddings]).toEqual([1, 6]);
    expect(frame.proposals.map((p) => [p.topId, p.orientation, p.retried, p.tauPassed, p.accepted])).toEqual([
      ["RED-1", 0, false, true, true],
      ["GREEN-1", 0, false, true, true],
      ["TURNED-1", 1, true, true, true],
    ]);
    expect(frame.proposals.map((p) => p.detectorScore)).toEqual([0.9, 0.8, 0.7]);
    expect(frame.kept).toEqual([0, 1, 2]);
    expect(Array.from(frame.proposals[2]!.embedding!)).toEqual([0, 0, 1, 0]);
  });

  it("the stand-in's turn is the runtime's: one counter-clockwise quarter turn moves the top-left cell to the bottom-left", () => {
    const t = new Float32Array(PLANE);
    t[2 * SIZE * SIZE] = 1;
    expect(rotateTensor90(t, 1, SIZE)[2 * SIZE * SIZE + (SIZE - 1) * SIZE]).toBe(1);
  });

  it("without the rotation retry nothing is embedded a second time", async () => {
    const embedder = new FakeEmbedder();
    const frame = await recognizeFrame(scene(), embedder, recognizer({ rotationRetry: false }), [candidate(0, 0, 0.9), candidate(0, 150, 0.7)]);
    expect(embedder.batches).toEqual([2]);
    expect(frame.proposals.map((p) => [p.orientation, p.retried, p.tauPassed, p.accepted])).toEqual([[0, false, true, true], [0, false, false, false]]);
    expect([frame.retried, frame.embeddings, frame.kept]).toEqual([0, 2, [0]]);
    expect(frame.proposals[1]!.embedding).toBeNull();
  });

  it("a look that reaches tau and fails the margin is kept for the suppression and names nothing", async () => {
    const embedder = new FakeEmbedder();
    const frame = await recognizeFrame(scene(), embedder, recognizer({ delta: 1.5 }), [candidate(0, 0, 0.9)]);
    const [p] = frame.proposals;
    expect([p!.tauPassed, p!.accepted, p!.retried, frame.kept]).toEqual([true, false, true, [0]]);
    expect(p!.embedding).not.toBeNull();
  });

  it("skips a box whose crop is under the minimum, and embeds nothing for a frame without faces", async () => {
    const embedder = new FakeEmbedder();
    const frame = await recognizeFrame(scene(), embedder, recognizer(), [candidate(0, 0, 0.9, 20), candidate(390, 0, 0.8)]);
    expect([frame.proposals.length, frame.embeddings, embedder.batches]).toEqual([0, 0, []]);
    expect((await recognizeFrame(scene(), embedder, recognizer(), [])).proposals).toEqual([]);
  });

  it("the embedding kept for the tracker is a copy, not a view of the batch", async () => {
    let batch: Float32Array | null = null;
    const embedder: BatchEmbedder = {
      embedBatch: async (_data, n) => {
        batch = new Float32Array(n * DIM);
        for (let i = 0; i < n; i++) batch[i * DIM] = 1;
        return batch;
      },
    };
    const frame = await recognizeFrame(scene(), embedder, recognizer(), [candidate(0, 0, 0.9)]);
    batch!.fill(9);
    expect(Array.from(frame.proposals[0]!.embedding!)).toEqual([1, 0, 0, 0]);
  });
});
