// SPDX-License-Identifier: Apache-2.0
import { describe, expect, it } from "vitest";

import type { EmbeddingBank } from "../src/bank.js";
import { type Candidate, rectOf } from "../src/boxes.js";
import type { Frame, FrameDetector } from "../src/detector/detector.js";
import type { BatchEmbedder } from "../src/embedder.js";
import { HOVER_ID_PREFIX, HOVER_SELECTIONS } from "../src/hover-layer.js";
import { parseIdentityGroups } from "../src/identity.js";
import { DISCONTINUITY_BACK_SEC, DISCONTINUITY_FORWARD_SEC, hoverAt, hoverCardsOf, isDiscontinuity, Pipeline, type PipelineOptions } from "../src/pipeline.js";
import { TRACK_ID_PREFIX } from "../src/tracker.js";
import { axis, blend, solidFrame } from "./support.js";

const DIM = 8;
const SIZE = 128;
const PLANE = 3 * SIZE * SIZE;

/**
 * A stand-in model: the red value of a crop's first cell says which embedding it gets. The
 * scenes below paint each card one flat color, so a card's red value is its "picture".
 *   250 -> A-1 exactly          200 -> A-1A (another printing of the same name)
 *   150 -> between A and B      100 -> B-1 exactly          anything else -> nothing in the bank
 */
class FakeEmbedder implements BatchEmbedder {
  crops = 0;

  async embedBatch(data: Float32Array, n: number): Promise<Float32Array> {
    this.crops += n;
    const out = new Float32Array(n * DIM);
    for (let i = 0; i < n; i++) {
      const red = Math.round(((data[i * PLANE]! + 1) / 2) * 255);
      const v = red === 250 ? axis(DIM, 0) : red === 200 ? blend(DIM, 0, 1, 0.999) : red === 150 ? blend(DIM, 0, 2, Math.SQRT1_2) : red === 100 ? axis(DIM, 2) : axis(DIM, 7);
      out.set(v, i * DIM);
    }
    return out;
  }
}

class FakeDetector implements FrameDetector {
  lastTimings = null;
  boxes: Candidate[] = [];
  fail = false;

  async propose(): Promise<Candidate[]> {
    if (this.fail) throw new Error("detector down");
    return this.boxes;
  }
}

const NAMES = new Map([["A-1", "Alder"], ["A-1A", "Alder"], ["B-1", "Birch"]]);
const GROUPS = parseIdentityGroups({ schemaVersion: 1, kind: "identity-groups", version: 1, groups: [["A-1", "A-1A"]], representative: { "A-1A": "A-1" } });

function bank(): EmbeddingBank {
  const ids = ["A-1", "A-1A", "B-1"];
  const rows = [axis(DIM, 0), blend(DIM, 0, 1, 0.999), axis(DIM, 2)];
  const matrix = new Float32Array(ids.length * DIM);
  rows.forEach((row, i) => matrix.set(row, i * DIM));
  return { ids, matrix, dim: DIM, version: 1, model: "fake", preprocessing: "full128", tau: null };
}

function box(x: number, y: number, score: number, cls = 0, width = 100, height = 140): Candidate {
  const aabb = { x, y, width, height };
  return { aabb, rect: rectOf(aabb), score, cls };
}

/** An 800 x 450 frame with flat-colored cards painted where the boxes are. */
function scene(cards: Array<{ box: Candidate; red: number }>): Frame {
  const frame = solidFrame(800, 450, [5, 5, 5]);
  for (const { box: b, red } of cards) {
    const { x, y, width, height } = b.aabb;
    for (let py = Math.max(0, y); py < Math.min(450, y + height); py++) {
      for (let px = Math.max(0, x); px < Math.min(800, x + width); px++) frame.data[(py * 800 + px) * 4] = red;
    }
  }
  return frame;
}

function setup(options: Partial<PipelineOptions> = {}): { pipeline: Pipeline; detector: FakeDetector; embedder: FakeEmbedder } {
  const detector = new FakeDetector();
  const embedder = new FakeEmbedder();
  const pipeline = new Pipeline({
    detector, embedder, bank: bank(), names: NAMES, identityGroups: GROUPS, identityDisplay: "best-printing",
    accept: { tau: 0.62, delta: 0.05, rotationRetry: true }, now: () => 0, ...options,
  });
  return { pipeline, detector, embedder };
}

describe("Pipeline.analyze", () => {
  it("reports every valid box with its class, and what the recognizer concluded about each face", async () => {
    const { pipeline, detector, embedder } = setup();
    const cards = [
      { box: box(20, 20, 0.95), red: 250 },
      { box: box(200, 20, 0.9), red: 200 },
      { box: box(380, 20, 0.85), red: 150 },
      { box: box(560, 20, 0.8, 1), red: 250 }, // a back that looks exactly like A-1
      { box: box(20, 250, 0.75), red: 30 },
    ];
    detector.boxes = [...cards.map((c) => c.box), box(700, 300, 0.7, 0, 20, 20)]; // and one too small
    const analysis = await pipeline.analyze(scene(cards));

    expect(analysis.frame).toEqual({ width: 800, height: 450 });
    expect(analysis.boxes.map((b) => [b.className, b.score])).toEqual([["face", 0.95], ["face", 0.9], ["face", 0.85], ["back", 0.8], ["face", 0.75]]);
    expect(analysis.boxes[0]!.box).toEqual([20, 20, 100, 140]);
    const faces = analysis.boxes.map((b) => b.face);
    expect(faces[3]).toBeNull(); // the back is reported and never looked at
    expect(faces[0]).toMatchObject({ topId: "A-1", topName: "Alder", look: 0, retried: false, tauPassed: true, accepted: true, cardId: "A-1", name: "Alder", kept: true });
    expect(faces[0]!.similarity).toBeCloseTo(1, 6);
    // The other printing matched best: it is what is shown, and the printings do not rival each other.
    expect(faces[1]).toMatchObject({ topId: "A-1A", accepted: true, cardId: "A-1A", name: "Alder" });
    expect(faces[1]!.margin).toBeGreaterThan(0.9);
    // Between two names: the score reaches tau, the margin does not; nothing is named.
    expect(faces[2]).toMatchObject({ tauPassed: true, accepted: false, retried: true, cardId: null, name: null, kept: true });
    // Nothing in the bank resembles it.
    expect(faces[4]).toMatchObject({ tauPassed: false, accepted: false, retried: true, cardId: null, kept: false });
    // Four faces upright, then three turns for each of the two that were not accepted.
    expect([analysis.retried, analysis.embeddings, embedder.crops]).toEqual([2, 10, 10]);
  });

  it("builds the hover layer from this frame alone, from the first frame on", async () => {
    const { pipeline, detector } = setup();
    const cards = [{ box: box(20, 20, 0.9), red: 250 }, { box: box(380, 20, 0.8), red: 150 }];
    detector.boxes = cards.map((c) => c.box);
    const analysis = await pipeline.analyze(scene(cards));
    expect(analysis.hover.selection).toBe("accepted-nms");
    expect(analysis.hover.cards.map((c) => [c.cardId, c.name, c.boxIndex])).toEqual(expect.arrayContaining([["A-1", "Alder", 0], [null, null, 1]]));
    expect(hoverAt(analysis, 70, 90)).toMatchObject({ boxIndex: 0, cardId: "A-1", name: "Alder", box: { x: 20, y: 20, width: 100, height: 140 } });
    expect(hoverAt(analysis, 430, 90)).toMatchObject({ boxIndex: 1, cardId: null, name: null }); // hit, and nothing shown
    expect(hoverAt(analysis, 700, 400)).toMatchObject({ boxIndex: null, card: null });
    // The track layer has not been confirmed yet.
    expect(analysis.tracks).toEqual({ shown: false, cards: [] });
  });

  it("shows the track layer from the third frame in a row, with pooled identities", async () => {
    const { pipeline, detector } = setup();
    const cards = [{ box: box(20, 20, 0.9), red: 250 }, { box: box(380, 20, 0.8), red: 150 }];
    detector.boxes = cards.map((c) => c.box);
    const frame = scene(cards);
    const shown = [];
    for (let i = 0; i < 3; i++) shown.push((await pipeline.analyze(frame, i)).tracks);
    expect(shown.map((t) => t.shown)).toEqual([false, false, true]);
    expect(shown[2]!.cards.map((c) => [c.instanceId, c.cardId])).toEqual([[`${TRACK_ID_PREFIX}0001`, "A-1"], [`${TRACK_ID_PREFIX}0002`, null]]);
  });

  it("holds the last track layer through a frame that recognized nothing, and stops after three", async () => {
    const { pipeline, detector } = setup();
    const cards = [{ box: box(20, 20, 0.9), red: 250 }];
    detector.boxes = cards.map((c) => c.box);
    const frame = scene(cards);
    for (let i = 0; i < 3; i++) await pipeline.analyze(frame, i);
    detector.boxes = [];
    const empty = solidFrame(800, 450);
    const first = await pipeline.analyze(empty, 3);
    expect([first.tracks.shown, first.tracks.cards.length, first.hover.cards.length]).toEqual([true, 1, 0]); // the hover layer holds nothing back
    await pipeline.analyze(empty, 4);
    const third = await pipeline.analyze(empty, 5);
    expect(third.tracks).toEqual({ shown: false, cards: [] });
  });

  it("keeps hover ids from frame to frame, apart from track ids", async () => {
    const { pipeline, detector } = setup();
    const cards = [{ box: box(20, 20, 0.9), red: 250 }];
    detector.boxes = cards.map((c) => c.box);
    const frame = scene(cards);
    const first = await pipeline.analyze(frame, 0);
    const second = await pipeline.analyze(frame, 1);
    expect(first.hover.cards[0]!.instanceId).toBe(`${HOVER_ID_PREFIX}0001`);
    expect(second.hover.cards[0]!.instanceId).toBe(first.hover.cards[0]!.instanceId);
  });

  it("a jump in video time resets what is carried between frames", async () => {
    const { pipeline, detector } = setup();
    const cards = [{ box: box(20, 20, 0.9), red: 250 }];
    detector.boxes = cards.map((c) => c.box);
    const frame = scene(cards);
    for (const t of [10, 11, 12]) await pipeline.analyze(frame, t);
    const back = await pipeline.analyze(frame, 3);
    expect([back.discontinuity, back.tracks.shown]).toEqual([true, false]);
    expect(back.hover.cards[0]!.instanceId).toBe(`${HOVER_ID_PREFIX}0002`);
    expect((await pipeline.analyze(frame, 3.5)).discontinuity).toBe(false);
    expect((await pipeline.analyze(frame, 9)).discontinuity).toBe(true);
  });

  it("clips a face that reaches outside the frame, and reports the clipped box", async () => {
    const { pipeline, detector } = setup();
    const over = box(760, 20, 0.9);
    detector.boxes = [over];
    const analysis = await pipeline.analyze(scene([{ box: over, red: 250 }]));
    expect(analysis.boxes[0]!.box).toEqual([760, 20, 40, 140]);
    expect(analysis.boxes[0]!.face).toMatchObject({ accepted: true, cardId: "A-1" });
  });

  it("a frame of backs only embeds nothing and shows nothing", async () => {
    const { pipeline, detector, embedder } = setup();
    detector.boxes = [box(20, 20, 0.9, 1), box(200, 20, 0.8, 1)];
    const analysis = await pipeline.analyze(solidFrame(800, 450, [250, 0, 0]));
    expect([analysis.boxes.length, analysis.hover.cards.length, embedder.crops]).toEqual([2, 0, 0]);
    expect(analysis.boxes.every((b) => b.face === null && b.className === "back")).toBe(true);
  });

  it("a detector error rejects the frame", async () => {
    const { pipeline, detector } = setup();
    detector.fail = true;
    await expect(pipeline.analyze(solidFrame(800, 450))).rejects.toThrow("detector down");
  });

  it("under the representative display the base printing is shown for either printing", async () => {
    const { pipeline, detector } = setup({ identityDisplay: "representative" });
    const cards = [{ box: box(200, 20, 0.9), red: 200 }];
    detector.boxes = cards.map((c) => c.box);
    const analysis = await pipeline.analyze(scene(cards));
    expect(analysis.boxes[0]!.face).toMatchObject({ topId: "A-1A", cardId: "A-1", name: "Alder" });
  });

  it("without identity groups two printings of one name rival each other and nothing is named", async () => {
    const { pipeline, detector } = setup({ identityGroups: null });
    const cards = [{ box: box(200, 20, 0.9), red: 200 }];
    detector.boxes = cards.map((c) => c.box);
    const analysis = await pipeline.analyze(scene(cards));
    expect(analysis.boxes[0]!.face).toMatchObject({ tauPassed: true, accepted: false, cardId: null });
  });
});

describe("the hover layer under each selection", () => {
  it("is rebuilt from an analyzed frame's boxes, and matches the frame's own layer for its selection", async () => {
    for (const selection of HOVER_SELECTIONS) {
      const { pipeline, detector } = setup({ hoverSelection: selection });
      // A large accepted card, a small unaccepted card on top of it, a far card under tau.
      const big = box(100, 100, 0.6, 0, 300, 300);
      const small = box(200, 200, 0.9, 0, 100, 100);
      const far = box(600, 100, 0.95);
      detector.boxes = [far, small, big];
      const frame = scene([{ box: big, red: 250 }, { box: small, red: 150 }, { box: far, red: 30 }]);
      const analysis = await pipeline.analyze(frame);
      const rebuilt = hoverCardsOf(analysis, selection);
      const strip = (cards: typeof rebuilt) => cards.map((c) => ({ ...c, instanceId: "" }));
      expect(strip(rebuilt), selection).toEqual(strip(analysis.hover.cards));
      for (const other of HOVER_SELECTIONS) {
        const again = setup({ hoverSelection: other });
        again.detector.boxes = detector.boxes;
        const direct = await again.pipeline.analyze(frame);
        for (const [x, y] of [[250, 250], [120, 120], [650, 150], [10, 10]] as const) {
          const a = hoverAt(analysis, x, y, other);
          const b = hoverAt(direct, x, y);
          expect([a.boxIndex, a.cardId], `${selection} as ${other} at ${x},${y}`).toEqual([b.boxIndex, b.cardId]);
        }
      }
    }
  });

  it("the three selections differ where they should", async () => {
    const { pipeline, detector } = setup();
    const big = box(100, 100, 0.99, 0, 300, 300);
    const small = box(200, 200, 0.5, 0, 100, 100);
    detector.boxes = [big, small];
    const analysis = await pipeline.analyze(scene([{ box: big, red: 250 }, { box: small, red: 150 }]));
    // On the small card, which reaches tau without being accepted and lies inside the big one.
    expect(hoverAt(analysis, 250, 250, "accepted-nms")).toMatchObject({ boxIndex: 1, cardId: null }); // the smaller box is on top
    expect(hoverAt(analysis, 250, 250, "best-score")).toMatchObject({ boxIndex: 0, cardId: "A-1" }); // the higher similarity
    expect(hoverAt(analysis, 250, 250, "detector-score")).toMatchObject({ boxIndex: 0, cardId: "A-1" }); // the higher detector score
  });
});

describe("isDiscontinuity", () => {
  it("is a step back of more than a quarter second or forward of more than five", () => {
    expect([DISCONTINUITY_BACK_SEC, DISCONTINUITY_FORWARD_SEC]).toEqual([0.25, 5]);
    expect([isDiscontinuity(10, 9.75), isDiscontinuity(10, 9.74), isDiscontinuity(10, 15), isDiscontinuity(10, 15.01)]).toEqual([false, true, false, true]);
    expect([isDiscontinuity(null, 3), isDiscontinuity(3, undefined)]).toEqual([false, false]);
  });
});
