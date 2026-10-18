// SPDX-License-Identifier: Apache-2.0
import { describe, expect, it } from "vitest";

import { rectOf } from "../src/boxes.js";
import { CONFIRM_FRAMES, TemporalGate } from "../src/temporal-gate.js";
import { type Detection, EMBED_POOL_WINDOW, MAX_MISSES, MIN_IDENTITY_FRAMES, poolEmbeddings, TRACK_ID_PREFIX, TRACK_IOU, Tracker } from "../src/tracker.js";
import { axis, blend } from "./support.js";

function detection(x: number, embedding: Float32Array = axis(4, 0), y = 0): Detection {
  const aabb = { x, y, width: 100, height: 100 };
  return { aabb, rect: rectOf(aabb), cardId: "A-1", embedding };
}

/** A resolver that names the axis the pooled embedding points along most. */
const byAxis = (pooled: Float32Array): string => `AXIS-${pooled.indexOf(Math.max(...pooled))}`;

describe("poolEmbeddings", () => {
  it("is the mean brought back to unit length", () => {
    const pooled = poolEmbeddings([axis(4, 0), axis(4, 1)])!;
    expect(pooled[0]).toBeCloseTo(Math.SQRT1_2, 6);
    expect(pooled[1]).toBeCloseTo(Math.SQRT1_2, 6);
    expect(Math.hypot(...pooled)).toBeCloseTo(1, 6);
  });

  it("leaves a single unit vector as it is", () => {
    expect(Array.from(poolEmbeddings([axis(4, 2)])!)).toEqual([0, 0, 1, 0]);
  });

  it("is null when the mean has no direction", () => {
    expect(poolEmbeddings([axis(4, 0), axis(4, 0).map((v) => -v)])).toBeNull();
  });
});

describe("Tracker", () => {
  it("states its constants", () => {
    expect([TRACK_IOU, MAX_MISSES, EMBED_POOL_WINDOW, MIN_IDENTITY_FRAMES]).toEqual([0.4, 2, 5, 2]);
  });

  it("keeps one id for a card that stays put, and gives its identity from the second frame", () => {
    const tracker = new Tracker(byAxis);
    const [first] = tracker.update([detection(0)]);
    expect([first!.instanceId, first!.identity]).toEqual([`${TRACK_ID_PREFIX}0001`, null]);
    const [second] = tracker.update([detection(5)]);
    expect([second!.instanceId, second!.identity]).toEqual([`${TRACK_ID_PREFIX}0001`, "AXIS-0"]);
  });

  it("gives separate cards separate tracks, and returns only the tracks matched on this frame", () => {
    const tracker = new Tracker(byAxis);
    expect(tracker.update([detection(0), detection(500)]).map((t) => t.instanceId)).toEqual([`${TRACK_ID_PREFIX}0001`, `${TRACK_ID_PREFIX}0002`]);
    expect(tracker.update([detection(500)]).map((t) => t.instanceId)).toEqual([`${TRACK_ID_PREFIX}0002`]);
  });

  it("survives a gap of two frames and is dropped after a third", () => {
    const tracker = new Tracker(byAxis);
    tracker.update([detection(0)]);
    tracker.update([]);
    tracker.update([]);
    expect(tracker.update([detection(0)])[0]!.instanceId).toBe(`${TRACK_ID_PREFIX}0001`);
    tracker.update([]);
    tracker.update([]);
    tracker.update([]);
    expect(tracker.update([detection(0)])[0]!.instanceId).toBe(`${TRACK_ID_PREFIX}0002`);
  });

  it("joins a detection to a track only above the overlap limit", () => {
    const tracker = new Tracker(byAxis);
    tracker.update([detection(0)]);
    // Shifted by 43 px: overlap 57 / 143 < 0.4, a new track. Shifted by 42: 58 / 142 > 0.4.
    expect(tracker.update([detection(43)])[0]!.instanceId).toBe(`${TRACK_ID_PREFIX}0002`);
    expect(tracker.update([detection(85)])[0]!.instanceId).toBe(`${TRACK_ID_PREFIX}0002`);
  });

  it("an older track takes the detection it overlaps most", () => {
    const tracker = new Tracker(byAxis);
    tracker.update([detection(0)]);
    const tracks = tracker.update([detection(30), detection(10)]);
    expect(tracks.find((t) => t.instanceId === `${TRACK_ID_PREFIX}0001`)!.detection.aabb.x).toBe(10);
  });

  it("reads identity from the pooled embeddings, so one odd frame does not rename a card", () => {
    const tracker = new Tracker(byAxis);
    for (let i = 0; i < 3; i++) tracker.update([detection(0, axis(4, 0))]);
    expect(tracker.update([detection(0, axis(4, 1))])[0]!.identity).toBe("AXIS-0");
  });

  it("pools only the most recent five embeddings", () => {
    const seen: Float32Array[] = [];
    const tracker = new Tracker((pooled) => (seen.push(pooled.slice()), "X"));
    for (let i = 0; i < 5; i++) tracker.update([detection(0, axis(8, 0))]);
    for (let i = 0; i < 5; i++) tracker.update([detection(0, axis(8, 1))]);
    const last = seen[seen.length - 1]!;
    expect([last[0], last[1]]).toEqual([0, 1]); // every axis-0 look has left the window
    const mixed = seen[seen.length - 3]!; // three of axis 1, two of axis 0
    expect(mixed[1]! / mixed[0]!).toBeCloseTo(1.5, 5);
  });

  it("names nothing when the resolver does not, or when the pooled mean has no direction", () => {
    const silent = new Tracker(() => null);
    silent.update([detection(0)]);
    expect(silent.update([detection(0)])[0]!.identity).toBeNull();
    let calls = 0;
    const opposed = new Tracker(() => (calls++, "X"));
    opposed.update([detection(0, axis(4, 0))]);
    expect(opposed.update([detection(0, axis(4, 0).map((v) => -v))])[0]!.identity).toBeNull();
    expect(calls).toBe(0);
  });

  it("reset forgets every track and its embeddings, and never reuses an id", () => {
    const seen: Float32Array[] = [];
    const tracker = new Tracker((pooled) => (seen.push(pooled.slice()), "X"));
    tracker.update([detection(0, axis(4, 0))]);
    tracker.update([detection(0, axis(4, 0))]);
    tracker.reset();
    const [fresh] = tracker.update([detection(0, axis(4, 1))]);
    expect([fresh!.instanceId, fresh!.identity]).toEqual([`${TRACK_ID_PREFIX}0002`, null]);
    tracker.update([detection(0, blend(4, 1, 2, 0.6))]);
    expect(seen[seen.length - 1]![0]).toBe(0); // nothing of the old track is pooled
  });
});

describe("TemporalGate", () => {
  const feed = (gate: TemporalGate, pattern: string): boolean[] => [...pattern].map((c) => gate.observe(c === "1").confirmed);

  it("starts on the third consecutive frame that recognized a card", () => {
    expect(CONFIRM_FRAMES).toBe(3);
    expect(feed(new TemporalGate(), "110111")).toEqual([false, false, false, false, false, true]);
  });

  it("holds through one or two empty frames and stops on the third in a row", () => {
    expect(feed(new TemporalGate(), "111001000")).toEqual([false, false, true, true, true, true, true, true, false]);
  });

  it("after stopping it takes three in a row again", () => {
    const gate = new TemporalGate();
    feed(gate, "111000");
    expect(feed(gate, "10110111")).toEqual([false, false, false, false, false, false, false, true]);
  });

  it("counts the current run", () => {
    const gate = new TemporalGate();
    expect([..."1101"].map((c) => gate.observe(c === "1").consecutive)).toEqual([1, 2, 0, 1]);
  });

  it("reset starts over", () => {
    const gate = new TemporalGate();
    feed(gate, "111");
    gate.reset();
    expect(feed(gate, "11")).toEqual([false, false]);
  });

  it("takes another k, and refuses one that is not a positive whole number", () => {
    expect(feed(new TemporalGate(1), "10")).toEqual([true, false]);
    expect(() => new TemporalGate(0)).toThrow(RangeError);
    expect(() => new TemporalGate(1.5)).toThrow(RangeError);
  });
});
