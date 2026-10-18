// SPDX-License-Identifier: Apache-2.0
// The scoring page's logic, driven with made-up videos: which frame stands for a row, what is
// recorded at each position, and the shape of the file. No <video> element is involved.
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import type { EmbeddingBank } from "../src/bank.js";
import { type Candidate, rectOf } from "../src/boxes.js";
import type { ReleaseStatus } from "../src/messages.js";
import { type FrameAnalysis, hoverCardsOf, Pipeline } from "../src/pipeline.js";
import { RUNTIME_NAME, RUNTIME_VERSION } from "../src/version.js";
import { fileStem, matchVideos, parseManifest, parseVideoIndex, parseVideoName, readChosen, rowsOfRole, type ScoreRow, shapeMatches, VIDEO_INDEX_KIND } from "../score/manifest.js";
import { eventLines, EVENTS_KIND, onVideoTimeline, positionOutcome, type RowLine, runHeader, scoreRows, type ScoringVideo, summarize } from "../score/run.js";
import {
  chooseFrame,
  type DecodedFrame,
  FINGERPRINT_WINDOW_SECONDS,
  framesBetween,
  MAX_STEP_SEC,
  measureFrameInterval,
  nearestFrame,
  reportsFrameTimes,
  type VideoFrames,
  windowStep,
} from "../score/select.js";
import { axis, contract, solidFrame } from "./support.js";

// ----------------------------------------------------------------------------- made-up video
/** A fingerprint with the lowest `bits` bits set: `bits` away from `ZERO`. */
function print(bits: number): string {
  return (bits >= 64 ? 0xffffffffffffffffn : (1n << BigInt(bits)) - 1n).toString(16).padStart(16, "0");
}
const ZERO = print(0);
const FAR = print(40);

/** A video of frames with given times: `frameAt` shows the last frame at or before the time
 *  asked for, as a browser does, and counts how often it was asked. */
class FakeVideo implements VideoFrames {
  seeks = 0;
  readonly exactTimes: boolean = true;
  readonly durationSec: number;
  shown: number | null = null;

  constructor(
    readonly times: readonly number[],
    readonly prints: (index: number) => string = () => FAR,
    durationSec?: number,
  ) {
    this.durationSec = durationSec ?? times[times.length - 1]! + (times[1]! - times[0]!);
  }

  async frameAt(timeSec: number): Promise<DecodedFrame> {
    this.seeks += 1;
    const clamped = Math.min(Math.max(0, timeSec), this.durationSec);
    let index = 0;
    for (let i = 0; i < this.times.length; i++) if (this.times[i]! <= clamped + 1e-9) index = i;
    this.shown = index;
    return { timeSec: this.times[index]!, seekSec: timeSec, fingerprint: this.prints(index) };
  }
}

/** The same video in a browser that cannot say which frame it shows: the time it reports for
 *  a frame is the time it was asked for. */
class BlindVideo extends FakeVideo {
  override readonly exactTimes = false;

  override async frameAt(timeSec: number): Promise<DecodedFrame> {
    const frame = await super.frameAt(timeSec);
    return { ...frame, timeSec: Math.min(Math.max(0, timeSec), this.durationSec) };
  }
}

const steady = (rate: number, count: number): number[] => Array.from({ length: count }, (_, i) => i / rate);

/** What `cardvision fetch` reads for a window: the frame on screen at its start and each later
 *  frame whose time is not after its end (cardvision/data/extract.py, `frames_between`). */
function pythonWindow(times: readonly number[], start: number, end: number): number[] {
  const out: number[] = [];
  for (const time of times) {
    if (time > end + 1e-6) break;
    if (time <= Math.max(0, start) + 1e-6) out.length = 0;
    out.push(time);
  }
  return out;
}

describe("the frame interval", () => {
  it("is measured from one frame to the next, at any frame rate", async () => {
    for (const rate of [24, 25, 30, 30000 / 1001, 50, 60, 120]) {
      const interval = await measureFrameInterval(new FakeVideo(steady(rate, 400)), 1.2345);
      expect(interval, `${rate} frames a second`).toBeCloseTo(1 / rate, 9);
    }
  });

  it("takes a handful of looks, not one per millisecond", async () => {
    const video = new FakeVideo(steady(30, 300));
    await measureFrameInterval(video, 2);
    expect(video.seeks).toBeLessThanOrEqual(8);
  });

  it("is null when no other frame shows", async () => {
    expect(await measureFrameInterval(new FakeVideo([0], undefined, 0.04), 0)).toBeNull();
    expect(await measureFrameInterval(new FakeVideo(steady(30, 30)), 5)).toBeNull(); // asked past the end
  });
});

describe("the frames of a window", () => {
  it("are the frames cardvision fetch reads: the one on screen at the start, and each later one up to the end", async () => {
    for (const rate of [24, 30, 30000 / 1001, 60]) {
      const times = steady(rate, Math.round(rate * 12));
      for (const t of [0.2, 0.9, 1.0, 3.517, 5.0, 7.25, 11.3, 11.99]) {
        const frames = await framesBetween(new FakeVideo(times), t - 1, t + 1, 1 / rate);
        expect(frames.map((f) => f.timeSec), `${rate} fps, t=${t}`).toEqual(pythonWindow(times, t - 1, t + 1));
      }
    }
  });

  it("is about 60 frames at 30 frames a second, each looked at once", async () => {
    const video = new FakeVideo(steady(30, 600));
    const frames = await framesBetween(video, 9.017, 11.017, 1 / 30);
    expect(frames.length).toBe(61);
    expect(video.seeks).toBeLessThanOrEqual(63);
  });

  it("goes on past a frame that stays on screen for longer than one interval", async () => {
    // 30 frames a second with a frame held for a third of a second at 5.0 s.
    const times = steady(30, 300).filter((time) => time <= 5.0 + 1e-9 || time >= 5.3333);
    const frames = await framesBetween(new FakeVideo(times), 4.5, 6.5, 1 / 30);
    expect(frames.map((f) => f.timeSec)).toEqual(pythonWindow(times, 4.5, 6.5));
  });

  it("stops at the end of the video, and a window past the end holds the last frame", async () => {
    const times = steady(30, 150); // five seconds
    const atEnd = await framesBetween(new FakeVideo(times), 4.2, 6.2, 1 / 30);
    expect(atEnd.map((f) => f.timeSec)).toEqual(pythonWindow(times, 4.2, 6.2));
    const video = new FakeVideo(times);
    const past = await framesBetween(video, 8, 10, 1 / 30);
    expect(past.map((f) => f.timeSec)).toEqual([times[149]]);
    expect(video.seeks).toBeLessThanOrEqual(3);
  });

  it("steps through the window where the browser cannot say which frame it shows", async () => {
    const video = new BlindVideo(steady(30, 300));
    const frames = await framesBetween(video, 4, 6, 1 / 60);
    expect(frames.length).toBe(121);
    expect(frames[0]!.timeSec).toBe(4);
    expect(frames[frames.length - 1]!.timeSec).toBeCloseTo(6, 9);
  });
});

describe("the nearest frame", () => {
  const frame = (timeSec: number, bits: number): DecodedFrame => ({ timeSec, seekSec: timeSec, fingerprint: print(bits) });

  it("is the one fewest bits away", () => {
    expect(nearestFrame([frame(1, 9), frame(2, 3), frame(3, 5)], ZERO, 1)).toEqual({ frame: frame(2, 3), distance: 3 });
  });

  it("on equal distance is the one nearest the row's time, then the earlier one", () => {
    expect(nearestFrame([frame(1, 2), frame(2, 2), frame(4, 2)], ZERO, 2.4).frame.timeSec).toBe(2);
    expect(nearestFrame([frame(3, 2), frame(1, 2)], ZERO, 2).frame.timeSec).toBe(1);
    expect(nearestFrame([frame(1, 2), frame(3, 2)], ZERO, 2).frame.timeSec).toBe(1);
  });

  it("needs a frame", () => {
    expect(() => nearestFrame([], ZERO, 0)).toThrow("no frames");
  });
});

describe("choosing the frame for a row", () => {
  const times = steady(30, 600);
  // The labeled frame is number 150 (5.0 s). Its neighbors look different.
  const labeled = (bits: number) => (index: number) => (index === 150 ? print(bits) : FAR);

  it("keeps the frame within the threshold, wherever it is in the window", async () => {
    for (const t of [5.017, 4.2, 5.95]) {
      const choice = await chooseFrame(new FakeVideo(times, labeled(2)), t, ZERO);
      expect(choice, `t=${t}`).toMatchObject({ status: "kept", distance: 2, frame: { timeSec: times[150] } });
    }
  });

  it("keeps a frame exactly at the threshold and not one bit beyond it", async () => {
    expect((await chooseFrame(new FakeVideo(times, labeled(6)), 5.017, ZERO)).status).toBe("kept");
    const beyond = await chooseFrame(new FakeVideo(times, labeled(7)), 5.017, ZERO);
    expect(beyond).toMatchObject({ status: "unreachable", distance: 7, looked: 61 });
  });

  it("is unreachable when the labeled frame lies outside the window, with the best distance seen", async () => {
    const choice = await chooseFrame(new FakeVideo(times, labeled(0)), 7.517, ZERO);
    expect(choice).toMatchObject({ status: "unreachable", distance: 40 });
    expect(FINGERPRINT_WINDOW_SECONDS).toBe(1);
  });

  it("takes the frame on screen at the row's time when the row has no fingerprint", async () => {
    const video = new FakeVideo(times);
    expect(await chooseFrame(video, 5.017, null)).toMatchObject({ status: "kept", distance: null, looked: 1, frame: { timeSec: times[150] } });
    expect(video.seeks).toBe(1);
    expect(await chooseFrame(video, 25, null)).toMatchObject({ status: "failed", reason: "the video ends before t=25" });
  });

  it("has failed, with the reason, when the row's whole window lies past the end of the video", async () => {
    // Twenty seconds of video. The window of a row at 21.5 s starts at 20.5 s.
    const video = new FakeVideo(times, () => ZERO);
    expect(await chooseFrame(video, 21.5, ZERO)).toEqual({ status: "failed", reason: "the video ends before t=20.5, where the window of the row at t=21.5 starts" });
    // The last frame was not compared with the row, though it would have matched.
    expect(video.seeks).toBe(0);
  });

  it("still compares the frames of a window that the end of the video cuts short", async () => {
    // The window of a row at 20.9 s starts at 19.9 s: the last three frames are in it.
    const last = (bits: number) => (index: number) => (index === 599 ? print(bits) : FAR);
    expect(await chooseFrame(new FakeVideo(times, last(2)), 20.9, ZERO)).toMatchObject({ status: "kept", distance: 2, looked: 3, frame: { timeSec: times[599] } });
    expect(await chooseFrame(new FakeVideo(times, last(9)), 20.9, ZERO)).toMatchObject({ status: "unreachable", distance: 9, looked: 3 });
  });
});

describe("the step of a window", () => {
  it("is at most 16 ms, and the frame interval of a video whose frames are shorter than that", async () => {
    expect(MAX_STEP_SEC).toBe(0.016);
    for (const rate of [15, 24, 25, 30, 30000 / 1001, 50, 60]) {
      expect(await windowStep(new FakeVideo(steady(rate, 400)), 1.2345), `${rate} frames a second`).toBe(MAX_STEP_SEC);
    }
    for (const rate of [90, 120, 240]) {
      expect(await windowStep(new FakeVideo(steady(rate, 2400)), 1.2345), `${rate} frames a second`).toBeCloseTo(1 / rate, 9);
    }
  });

  it("takes four looks at most, and none where the browser does not report frame times", async () => {
    const video = new FakeVideo(steady(30, 300));
    await windowStep(video, 2);
    expect(video.seeks).toBeLessThanOrEqual(4);
    const blind = new BlindVideo(steady(30, 300));
    expect(await windowStep(blind, 2)).toBe(MAX_STEP_SEC);
    expect(blind.seeks).toBe(0);
  });

  it("sees every frame of a video of one frame rate, up to 60 frames a second and beyond", async () => {
    for (const rate of [24, 30, 30000 / 1001, 60, 120]) {
      const times = steady(rate, Math.round(rate * 12));
      for (const t of [0.2, 1.0, 3.517, 7.25, 11.3]) {
        const video = new FakeVideo(times);
        const frames = await framesBetween(video, t - 1, t + 1, await windowStep(video, t - 1));
        expect(frames.map((f) => f.timeSec), `${rate} fps, t=${t}`).toEqual(pythonWindow(times, t - 1, t + 1));
      }
    }
  });

  it("costs about two looks a frame at 30 frames a second", async () => {
    const video = new FakeVideo(steady(30, 600));
    const choice = await chooseFrame(video, 10.017, ZERO);
    expect(choice).toMatchObject({ status: "unreachable", looked: 61 });
    expect(video.seeks).toBeLessThanOrEqual(4 + 2 * 61 + 4);
  });
});

// A recording whose frames do not last equally long, as a phone or a screen recorder makes:
// the first frame stays on screen for 3.2 seconds, and after it short frames (a sixtieth of a
// second) follow long ones.
const UNEVEN_DURATIONS = [1 / 60, 1 / 60, 1 / 30, 0.1, 1 / 60, 1 / 24, 0.25, 1 / 60, 1 / 60, 1 / 60, 1 / 30, 0.5, 1 / 30, 1 / 60];
const UNEVEN: number[] = (() => {
  const times = [0];
  let time = 3.2;
  for (let index = 0; time < 14; index += 1) {
    times.push(Math.round(time * 1e6) / 1e6);
    time += UNEVEN_DURATIONS[index % UNEVEN_DURATIONS.length]!;
  }
  return times;
})();
/** A frame that lasts a sixtieth of a second and follows another such frame, near `nearSec`. */
function shortFrame(nearSec: number): number {
  const short = (index: number): boolean => UNEVEN[index + 1]! - UNEVEN[index]! < 0.017;
  const found = UNEVEN.findIndex((time, index) => time >= nearSec && index > 1 && index + 1 < UNEVEN.length && short(index - 1) && short(index));
  expect(found).toBeGreaterThan(0);
  return found;
}

describe("a recording whose frames do not last equally long", () => {
  it("has every frame of a window seen, whichever frame the window starts on", async () => {
    // 1.5 s and 3.9 s: the window starts on the long first frame. The others start anywhere.
    for (const t of [1.5, 3.9, 4.3, 5.0, 6.77, 8.123, 10.5, 12.9]) {
      const video = new FakeVideo(UNEVEN, undefined, 14.5);
      const frames = await framesBetween(video, t - 1, t + 1, await windowStep(video, t - 1));
      expect(frames.map((f) => f.timeSec), `t=${t}`).toEqual(pythonWindow(UNEVEN, t - 1, t + 1));
    }
  });

  it("would lose its short frames to a step as long as a common frame interval", async () => {
    const frames = await framesBetween(new FakeVideo(UNEVEN, undefined, 14.5), 4, 6, 1 / 30);
    expect(frames.length).toBeLessThan(pythonWindow(UNEVEN, 4, 6).length);
  });

  it("has the labeled frame found, though it is a short one", async () => {
    for (const near of [3.3, 5, 8.5, 11]) {
      const labeled = shortFrame(near);
      const video = new FakeVideo(UNEVEN, (index) => (index === labeled ? print(2) : FAR), 14.5);
      for (const t of [UNEVEN[labeled]! + 0.004, UNEVEN[labeled]! - 0.9, UNEVEN[labeled]! + 0.95]) {
        expect(await chooseFrame(video, t, ZERO), `frame ${labeled}, t=${t}`).toMatchObject({ status: "kept", distance: 2, frame: { timeSec: UNEVEN[labeled] } });
      }
    }
  });

  it("looks at a window a bounded number of times", async () => {
    const video = new FakeVideo(UNEVEN, undefined, 14.5);
    await chooseFrame(video, 5, ZERO);
    // Two seconds in steps of 16 ms, a look more for each new frame, and the measurement.
    expect(video.seeks).toBeLessThanOrEqual(Math.ceil(2 / MAX_STEP_SEC) + pythonWindow(UNEVEN, 4, 6).length + 8);
  });
});

describe("a browser that reports the time it was asked for", () => {
  const frameTimes = (video: FakeVideo) => async (timeSec: number): Promise<number> => (await video.frameAt(timeSec)).timeSec;

  it("is told from one that reports each frame's own time, in three looks", async () => {
    for (const rate of [24, 30, 30000 / 1001, 60, 120]) {
      const exact = new FakeVideo(steady(rate, Math.round(rate * 10)));
      expect(await reportsFrameTimes(frameTimes(exact), exact.durationSec), `${rate} frames a second`).toBe(true);
      expect(exact.seeks).toBe(3);
      const echo = new BlindVideo(steady(rate, Math.round(rate * 10)));
      expect(await reportsFrameTimes(frameTimes(echo), echo.durationSec), `${rate} frames a second, times asked for`).toBe(false);
    }
  });

  it("is not mistaken for one when a new frame starts between the looks", async () => {
    // Ten seconds; the looks are at 4.0012, 4.0024 and 4.0036 s. A frame starts between them.
    for (const startsAt of [4.0006, 4.0018, 4.003]) {
      const video = new FakeVideo([0, startsAt, 9], undefined, 10);
      expect(await reportsFrameTimes(frameTimes(video), 10), `a frame at ${startsAt}`).toBe(true);
    }
  });

  it("is assumed where the video is too short to tell", async () => {
    expect(await reportsFrameTimes(async () => 0, 0.005)).toBe(false);
  });

  // Each frame has a fingerprint of its own: its number.
  const numbered = (index: number): string => index.toString(16).padStart(16, "0");

  it("has its window walked in steps of 16 ms, which shows every frame of up to 60 frames a second", async () => {
    for (const rate of [24, 30, 30000 / 1001, 60]) {
      const times = steady(rate, Math.round(rate * 12));
      const video = new BlindVideo(times, numbered);
      const frames = await framesBetween(video, 4.017, 6.017, await windowStep(video, 4.017));
      // 126 looks in the window, and the one past its end that ends the walk.
      expect(frames.length, `${rate} frames a second`).toBe(126);
      expect(video.seeks).toBe(127);
      const shown = [...new Set(frames.map((frame) => times[parseInt(frame.fingerprint, 16)]!))];
      expect(shown, `${rate} frames a second`).toEqual(pythonWindow(times, 4.017, 6.017));
    }
  });

  it("has the labeled frame kept, with the time asked for as its time", async () => {
    const times = steady(30, 600);
    const video = new BlindVideo(times, (index) => (index === 149 ? print(3) : index === 150 ? print(1) : FAR));
    const choice = await chooseFrame(video, 5.017, ZERO);
    expect(choice).toMatchObject({ status: "kept", distance: 1, looked: 126 });
    if (choice.status !== "kept") throw new Error("not kept");
    // Frame 150 is on screen from 5 s for a thirtieth of a second. The time recorded is a time
    // in that span at which the page looked, and asking for it again shows the frame again.
    expect(choice.frame.timeSec).toBeGreaterThanOrEqual(5);
    expect(choice.frame.timeSec).toBeLessThan(5 + 1 / 30);
    expect(choice.frame.seekSec).toBe(choice.frame.timeSec);
    expect(video.seeks).toBe(127);
  });
});

// ------------------------------------------------------------------------------ the manifest
const quad = (x: number, y: number, w: number, h: number): number[][] => [[x, y], [x + w, y], [x + w, y + h], [x, y + h]];
const rowDoc = (over: Record<string, unknown> = {}): Record<string, unknown> => ({
  manifest_version: 1, platform: "youtube", video_id: "EXAMPLE0001", t: 5.017, frame_width: 800, frame_height: 450,
  source_type: "physical/tournament", group: "g-1", role: "evaluation", completeness: "exhaustive-boxes",
  attributes: { frame_dhash: ZERO },
  points: [{ x: 0.1, y: 0.2, kind: "not-card" }, { x: 0.7, y: 0.6, kind: "back" }],
  boxes: [{ kind: "face", quad: quad(0.125, 0.1, 0.125, 0.3) }, { kind: "back", quad: quad(0.5, 0.5, 0.2, 0.2) }, { kind: "face", ignore: true, quad: quad(0.8, 0.1, 0.1, 0.1) }],
  ...over,
});
const manifestText = (...docs: Array<Record<string, unknown>>): string => docs.map((doc) => JSON.stringify(doc)).join("\n") + "\n";

describe("reading a manifest", () => {
  it("takes every point and the center of every box, points first, in manifest order", () => {
    const [row] = parseManifest(manifestText(rowDoc()), "labels.jsonl");
    expect(row).toMatchObject({ platform: "youtube", videoId: "EXAMPLE0001", t: 5.017, role: "evaluation", frameWidth: 800, frameHeight: 450, fingerprint: ZERO, points: 2, boxes: 3 });
    expect(row!.positions.map((p) => [p.kind, p.index])).toEqual([["point", 0], ["point", 1], ["center", 0], ["center", 1], ["center", 2]]);
    expect(row!.positions[0]).toMatchObject({ x: 0.1, y: 0.2 });
    expect(row!.positions[2]!.x).toBe((0.125 + 0.25 + 0.25 + 0.125) / 4);
    expect(row!.positions[2]!.y).toBe((0.1 + 0.1 + 0.4 + 0.4) / 4);
  });

  it("reads a row without a fingerprint, without points or without boxes, and skips blank lines", () => {
    const text = `\n${JSON.stringify(rowDoc({ attributes: undefined, boxes: undefined }))}\r\n\n${JSON.stringify(rowDoc({ t: 6, points: undefined }))}\n`;
    const rows = parseManifest(text, "labels.jsonl");
    expect(rows.map((row) => [row.fingerprint, row.points, row.boxes])).toEqual([[null, 2, 0], [ZERO, 0, 3]]);
  });

  it("names the file and the line of anything it cannot read", () => {
    const text = `${JSON.stringify(rowDoc())}\n{not json\n${JSON.stringify(rowDoc({ attributes: { frame_dhash: "XYZ" } }))}\n${JSON.stringify(rowDoc({ boxes: [{ quad: [[0, 0]] }] }))}\n`;
    expect(() => parseManifest(text, "labels.jsonl")).toThrow(
      "labels.jsonl: line 2: not valid JSON; line 3: attributes.frame_dhash must be 16 lower-case hexadecimal digits; line 4: box 0 must give a quad of four corners",
    );
  });

  it("keeps the rows of one role and refuses a row given twice", () => {
    const first = { name: "a.jsonl", rows: parseManifest(manifestText(rowDoc(), rowDoc({ t: 6, role: "train" })), "a.jsonl") };
    const second = { name: "b.jsonl", rows: parseManifest(manifestText(rowDoc({ video_id: "EXAMPLE0002" })), "b.jsonl") };
    expect(rowsOfRole([first, second], "evaluation").map((row) => row.videoId)).toEqual(["EXAMPLE0001", "EXAMPLE0002"]);
    expect(rowsOfRole([first, second], "train").map((row) => row.t)).toEqual([6]);
    expect(() => rowsOfRole([first, first], "evaluation")).toThrow("a.jsonl: the row youtube EXAMPLE0001 t=5.017 is given twice");
  });

  it("accepts a video of the labeled shape at any size, within one percent", () => {
    const row = { frameWidth: 1920, frameHeight: 1080 };
    expect(shapeMatches(row, 1280, 720)).toBe(true);
    expect(shapeMatches(row, 1280, 726)).toBe(true);
    expect(shapeMatches(row, 1280, 740)).toBe(false);
    expect(shapeMatches(row, 1080, 1920)).toBe(false);
    expect(shapeMatches(row, 0, 0)).toBe(false);
  });
});

describe("reading a chosen file", () => {
  it("names the file when the browser cannot read it", async () => {
    // What a browser throws for a file that was moved or changed after it was chosen: no file name in it.
    const unreadable = { name: "evaluation.manifest.jsonl", text: () => Promise.reject(new DOMException("The requested file could not be read, typically due to permission problems.", "NotReadableError")) };
    await expect(readChosen(unreadable, () => unreadable.text())).rejects.toThrow(
      "evaluation.manifest.jsonl: the file could not be read (The requested file could not be read, typically due to permission problems.). Choose it again.",
    );
    await expect(readChosen({ name: "video-files.json" }, () => Promise.reject("gone"))).rejects.toThrow("video-files.json: the file could not be read (gone). Choose it again.");
  });

  it("gives what was read when the file can be read", async () => {
    expect(await readChosen({ name: "a.jsonl" }, () => Promise.resolve("text"))).toBe("text");
  });
});

describe("matching video files to rows", () => {
  const rows = parseManifest(manifestText(rowDoc(), rowDoc({ video_id: "EXAMPLE0002" }), rowDoc({ platform: "twitch", video_id: "0000000001" }), rowDoc({ video_id: "EXAMPLE0003" })), "m.jsonl");

  it("goes by the file's name: the video id, or the platform and the video id", () => {
    expect(fileStem("folder/EXAMPLE0001.mp4")).toBe("EXAMPLE0001");
    expect(fileStem("a.b.webm")).toBe("a.b");
    const { fileOf, unused } = matchVideos(rows, ["EXAMPLE0002.webm", "notes.mp4", "twitch-0000000001.mp4", "EXAMPLE0001.mp4"]);
    expect(fileOf).toEqual([3, 0, 2, null]);
    expect(unused).toEqual(["notes.mp4"]);
  });

  it("prefers the name with the platform, and refuses two files for one video", () => {
    expect(matchVideos(rows, ["0000000001.mp4", "twitch-0000000001.mp4"]).fileOf[2]).toBe(1);
    expect(() => matchVideos(rows, ["EXAMPLE0001.mp4", "EXAMPLE0001.webm"])).toThrow("EXAMPLE0001.mp4 and EXAMPLE0001.webm name the same video");
  });

  it("gives no start to a whole video", () => {
    expect(matchVideos(rows, ["EXAMPLE0002.webm", "twitch-0000000001.mp4"]).startOf).toEqual([null, null, null, null]);
  });

  it("matches a recording of your own (platform local) the same two ways", () => {
    // The video id of such a row is a name chosen by whoever made the recording; it may hold hyphens and underscores.
    const own = parseManifest(manifestText(rowDoc({ platform: "local", video_id: "kitchen-table_01" }), rowDoc({ platform: "local", video_id: "second-game" }), rowDoc()), "own.jsonl");
    expect(own.map((row) => row.platform)).toEqual(["local", "local", "youtube"]);
    const named = matchVideos(own, ["local-kitchen-table_01.mp4", "second-game.mov", "local-EXAMPLE0001.mp4"]);
    expect(named.fileOf).toEqual([0, 1, null]);
    expect(named.unused).toEqual(["local-EXAMPLE0001.mp4"]);
    // With the platform is preferred to without, and a section of such a recording is read like any other.
    expect(matchVideos(own, ["kitchen-table_01.mp4", "local-kitchen-table_01.mp4"]).fileOf[0]).toBe(1);
    expect(matchVideos(own, ["local-second-game+2.mp4"])).toMatchObject({ fileOf: [null, 0, null], startOf: [null, 2, null] });
  });
});

describe("a file that holds one section of a video", () => {
  // Three rows of one video, at 5.017 s, 80.017 s and 500.5 s, and one row of another video.
  const rows = parseManifest(manifestText(rowDoc(), rowDoc({ t: 80.017 }), rowDoc({ t: 500.5 }), rowDoc({ video_id: "EXAMPLE0002", t: 80.017 })), "m.jsonl");
  const index = (...files: Array<Record<string, unknown>>): string => JSON.stringify({ kind: VIDEO_INDEX_KIND, version: 1, files });
  const entry = (file: string, start: number | null, end: number | null, over: Record<string, unknown> = {}): Record<string, unknown> => ({
    file, platform: "youtube", video_id: "EXAMPLE0001", start, end, rows: [], ...over,
  });

  it("says in its name where it starts in the video", () => {
    expect(parseVideoName("youtube-EXAMPLE0001+77.mp4")).toEqual({ video: "youtube-EXAMPLE0001", start: 77 });
    expect(parseVideoName("folder/EXAMPLE0001+4578.webm")).toEqual({ video: "EXAMPLE0001", start: 4578 });
    expect(parseVideoName("EXAMPLE0001+0.mp4")).toEqual({ video: "EXAMPLE0001", start: 0 });
    expect(parseVideoName("twitch-0000000001+12.5.mkv")).toEqual({ video: "twitch-0000000001", start: 12.5 });
    // A whole video has no start, and a name that only looks like a section is a whole video's.
    expect(parseVideoName("youtube-EXAMPLE0001.mp4")).toEqual({ video: "youtube-EXAMPLE0001", start: null });
    for (const name of ["EXAMPLE0001+.mp4", "EXAMPLE0001+late.mp4", "EXAMPLE0001+-3.mp4", "+77.mp4", "EXAMPLE0001+1e3.mp4"]) {
      expect(parseVideoName(name), name).toEqual({ video: fileStem(name), start: null });
    }
  });

  it("serves the rows whose time lies inside it", () => {
    const names = ["youtube-EXAMPLE0001+77.mp4", "EXAMPLE0001+2.mp4", "EXAMPLE0002+77.mp4", "EXAMPLE0001+300.mp4"];
    // By name alone a section's end is not known: the section that starts last at or before the
    // row's time is taken. A name with the platform is preferred to one without.
    const { fileOf, startOf, unused } = matchVideos(rows, names);
    expect(fileOf).toEqual([1, 0, 0, 2]);
    expect(startOf).toEqual([2, 77, 77, 77]);
    expect(unused).toEqual(["EXAMPLE0001+300.mp4"]);
    // A row before every section has no file. A row at the very start of a section is inside it.
    expect(matchVideos(rows, ["EXAMPLE0001+6.mp4"]).fileOf).toEqual([null, 0, 0, null]);
    expect(matchVideos(rows, ["EXAMPLE0001+5.017.mp4"]).startOf[0]).toBe(5.017);
  });

  it("serves no row past its end, where the index gives the end", () => {
    const names = ["youtube-EXAMPLE0001+2.mp4", "youtube-EXAMPLE0001+77.mp4"];
    const listed = parseVideoIndex(index(entry(names[0]!, 2, 9), entry(names[1]!, 77, 84)), "video-files.json");
    expect(listed).toEqual([
      { file: names[0], platform: "youtube", videoId: "EXAMPLE0001", start: 2, end: 9 },
      { file: names[1], platform: "youtube", videoId: "EXAMPLE0001", start: 77, end: 84 },
    ]);
    const { fileOf, startOf } = matchVideos(rows, names, listed);
    expect(fileOf).toEqual([0, 1, null, null]);
    expect(startOf).toEqual([2, 77, null, null]);
    // The end itself is inside the section.
    expect(matchVideos(rows, names, parseVideoIndex(index(entry(names[1]!, 77, 80.017)), "i.json")).fileOf[1]).toBe(1);
    expect(matchVideos(rows, names, parseVideoIndex(index(entry(names[1]!, 77, 80.016)), "i.json")).fileOf[1]).toBe(0);
  });

  it("is known by its index entry, whatever its name says", () => {
    const listed = parseVideoIndex(index(entry("clip-a.mp4", 77, 84), entry("EXAMPLE0001+2.mp4", null, null, { video_id: "EXAMPLE0002" })), "video-files.json");
    const { fileOf, startOf } = matchVideos(rows, ["folder/clip-a.mp4", "EXAMPLE0001+2.mp4"], listed);
    expect(fileOf).toEqual([null, 0, null, 1]);
    expect(startOf).toEqual([null, 77, null, null]);
    // An entry for a file that was not chosen changes nothing.
    expect(matchVideos(rows, ["EXAMPLE0001.mp4"], listed).fileOf).toEqual([0, 0, 0, null]);
  });

  it("gives way to a whole-video file of the same video", () => {
    const { fileOf, startOf, unused } = matchVideos(rows, ["youtube-EXAMPLE0001+77.mp4", "EXAMPLE0001.mp4", "EXAMPLE0002+77.mp4"]);
    expect(fileOf).toEqual([1, 1, 1, 2]);
    expect(startOf).toEqual([null, null, null, 77]);
    expect(unused).toEqual(["youtube-EXAMPLE0001+77.mp4"]);
  });

  it("must be the only file for its section", () => {
    expect(() => matchVideos(rows, ["EXAMPLE0001+77.mp4", "EXAMPLE0001+77.webm"])).toThrow("EXAMPLE0001+77.mp4 and EXAMPLE0001+77.webm name the same section of a video");
    expect(() => matchVideos(rows, ["EXAMPLE0001+77.mp4", "EXAMPLE0001+77.0.mp4"])).toThrow("name the same section");
    expect(matchVideos(rows, ["EXAMPLE0001+77.mp4", "youtube-EXAMPLE0001+77.mp4"]).fileOf[1]).toBe(1);
  });

  it("has an index that is refused when it is not one", () => {
    expect(parseVideoIndex(index(), "video-files.json")).toEqual([]);
    expect(() => parseVideoIndex("{", "video-files.json")).toThrow("video-files.json: not valid JSON");
    for (const text of ["[]", JSON.stringify({ kind: "other", version: 1, files: [] }), JSON.stringify({ kind: VIDEO_INDEX_KIND, version: 2, files: [] }), JSON.stringify({ kind: VIDEO_INDEX_KIND, version: 1 })]) {
      expect(() => parseVideoIndex(text, "video-files.json"), text).toThrow("video-files.json: not an index of kept video files");
    }
    expect(() => parseVideoIndex(index({ file: "a.mp4" }), "i.json")).toThrow("i.json: files[0] must give file, platform and video_id");
    for (const [start, end] of [[77, null], [null, 84], [84, 77], [77, 77], [-1, 5], ["77", 84]]) {
      expect(() => parseVideoIndex(index(entry("a.mp4", start as number, end as number)), "i.json"), `${start}, ${end}`).toThrow("i.json: files[0]: start and end");
    }
  });
});

describe("a section on the video's own timeline", () => {
  // A video of 20 seconds at 30 frames a second, and the file that holds its frames 60 to 269:
  // the section from 2 s to 9 s. Time x of the file is time 2 + x of the video.
  const START = 2;
  const whole = steady(30, 600);
  const fileTimes = steady(30, 210);

  it("shows, at a time of the video, the file's frame at that time less its start", async () => {
    const file = new FakeScoringVideo("EXAMPLE0001+2.mp4", 1600, 900, null, () => FAR, fileTimes);
    const section = onVideoTimeline(file, START);
    expect(section).toMatchObject({ file: "EXAMPLE0001+2.mp4", width: 1600, height: 900, sectionStart: 2, exactTimes: true });
    expect(section.durationSec).toBeCloseTo(9, 9);
    const frame = await section.frameAt(5.017);
    expect(file.shown).toBe(90);                                // 3.017 s of the file
    expect(frame.timeSec).toBeCloseTo(5, 9);                    // a time of the video
    expect(frame.seekSec).toBe(5.017);                          // asking for it again shows it again
    expect((await section.frameAt(frame.seekSec)).timeSec).toBe(frame.timeSec);
    // Before the section's start there is its first frame; past its end, its last.
    expect((await section.frameAt(0.5)).timeSec).toBe(2);
    expect(file.shown).toBe(0);
    expect((await section.frameAt(15)).timeSec).toBeCloseTo(2 + fileTimes[209]!, 6);
  });

  it("gives every frame a time to the microsecond, in step with the whole video", async () => {
    const section = onVideoTimeline(new FakeScoringVideo("EXAMPLE0001+2.mp4", 1600, 900, null, () => FAR, fileTimes), START);
    for (const index of [60, 61, 150, 269]) {
      const time = (await section.frameAt(whole[index]! + 0.01)).timeSec;
      expect(Math.abs(time - whole[index]!), `frame ${index}`).toBeLessThanOrEqual(5e-7);
      expect(Math.round(time * 1e6) / 1e6).toBe(time);
    }
  });

  it("has, for a row, the window cardvision fetch reads from that file: clipped where the file starts and ends", async () => {
    for (const rate of [24, 30, 30000 / 1001, 60]) {
      for (const start of [2, 77, 4578]) {
        const times = steady(rate, Math.round(rate * 7));       // a file of seven seconds
        const section = onVideoTimeline(new FakeScoringVideo("f.mp4", 1600, 900, null, () => FAR, times), start);
        for (const local of [0.2, 0.9, 1.0, 3.517, 5.0, 6.3, 6.99]) {
          const t = start + local;
          const frames = await framesBetween(section, t - 1, t + 1, 1 / rate);
          // cardvision/data/extract.py, `frames_between` with `offset`: the window less the offset, on the file's own times.
          const expected = pythonWindow(times, local - 1, local + 1).map((time) => time + start);
          expect(frames.length, `${rate} fps, start ${start}, t=${t}`).toBe(expected.length);
          frames.forEach((frame, i) => expect(Math.abs(frame.timeSec - expected[i]!)).toBeLessThanOrEqual(1e-6));
        }
      }
    }
  });

  it("looks at fewer frames where the window is cut by the start or the end of the file", async () => {
    const section = () => onVideoTimeline(new FakeScoringVideo("f.mp4", 1600, 900, null, () => FAR, fileTimes), START);
    expect((await framesBetween(section(), 4.017, 6.017, 1 / 30)).length).toBe(61);      // the whole window
    const atStart = await framesBetween(section(), 1.517, 3.517, 1 / 30);                // cut at 2 s
    expect([atStart.length, atStart[0]!.timeSec]).toEqual([46, 2]);                      // frames 0 to 45 of the file
    const atEnd = await framesBetween(section(), 7.517, 9.517, 1 / 30);                  // the file ends before 9 s
    expect(atEnd.length).toBe(45);                                                       // frames 165 to 209 of the file
    expect(atEnd[atEnd.length - 1]!.timeSec).toBeCloseTo(2 + fileTimes[209]!, 6);
  });

  it("measures the frame interval as in the whole video", async () => {
    const section = onVideoTimeline(new FakeScoringVideo("f.mp4", 1600, 900, null, () => FAR, steady(30000 / 1001, 210)), 4578);
    expect(await measureFrameInterval(section, 4581.2)).toBeCloseTo(1001 / 30000, 5);
  });
});

// -------------------------------------------------------------------------------- a whole run
const DIM = 8;
const NAMES = new Map([["A-1", "Alder"], ["B-1", "Birch"]]);

function candidate(x: number, y: number, width: number, height: number, score: number, cls = 0): Candidate {
  const aabb = { x, y, width, height };
  return { aabb, rect: rectOf(aabb), score, cls };
}

/** An analysis of an 800 x 450 frame by the real pipeline, over stand-in models: a card named
 *  Alder at (100, 45), a card that matches two names equally at (400, 45), and a card back. */
async function analysis(): Promise<FrameAnalysis> {
  const frame = solidFrame(800, 450, [5, 5, 5]);
  const paint = (x0: number, y0: number, red: number): void => {
    for (let y = y0; y < y0 + 135; y++) for (let x = x0; x < x0 + 100; x++) frame.data[(y * 800 + x) * 4] = red;
  };
  paint(100, 45, 250);
  paint(400, 45, 150);
  const matrix = new Float32Array(2 * DIM);
  matrix.set(axis(DIM, 0), 0);
  matrix.set(axis(DIM, 2), DIM);
  const bank: EmbeddingBank = { ids: ["A-1", "B-1"], matrix, dim: DIM, version: 1, model: "fake", preprocessing: "full128", tau: null };
  const half = new Float32Array(DIM);
  half[0] = Math.SQRT1_2;
  half[2] = Math.SQRT1_2;
  const pipeline = new Pipeline({
    detector: { lastTimings: null, propose: async () => [candidate(100, 45, 100, 135, 0.9), candidate(400, 45, 100, 135, 0.8), candidate(600, 250, 100, 135, 0.7, 1)] },
    embedder: {
      embedBatch: async (data: Float32Array, n: number) => {
        const out = new Float32Array(n * DIM);
        for (let i = 0; i < n; i++) out.set(Math.round(((data[i * 3 * 128 * 128]! + 1) / 2) * 255) === 250 ? axis(DIM, 0) : half, i * DIM);
        return out;
      },
    },
    bank, names: NAMES, accept: { tau: 0.62, delta: 0.05, rotationRetry: false }, now: () => 0,
  });
  return pipeline.analyze(frame);
}

describe("what is recorded at a position", () => {
  it("is the card shown with its box, score and margin; a box that shows nothing; or nothing", async () => {
    const found = await analysis();
    const at = (x: number, y: number) => positionOutcome(found, { kind: "point", index: 3, x, y });
    expect(at(150 / 800, 100 / 450)).toEqual({ index: 3, x: 150 / 800, y: 100 / 450, cardId: "A-1", name: "Alder", box: [100, 45, 100, 135], similarity: 1, margin: 1, accepted: true });
    // Two names match equally: the box wins the pointer and shows nothing.
    const tie = at(450 / 800, 100 / 450);
    expect(tie).toMatchObject({ cardId: null, name: null, box: [400, 45, 100, 135], accepted: false });
    expect(tie.similarity).toBeCloseTo(Math.SQRT1_2, 6);
    expect(tie.margin).toBeCloseTo(0, 6);
    // A card back is never a candidate, and the bare table has no box.
    const nothing = { cardId: null, name: null, box: null, similarity: null, margin: null, accepted: null };
    expect(at(650 / 800, 300 / 450)).toMatchObject(nothing);
    expect(at(0.01, 0.99)).toMatchObject(nothing);
  });
});

/** A made-up video that can also be analyzed: it records which frame was on screen then. */
class FakeScoringVideo extends FakeVideo implements ScoringVideo {
  analyzed: number[] = [];
  failAnalysis = false;

  constructor(
    readonly file: string,
    readonly width: number,
    readonly height: number,
    private readonly found: FrameAnalysis | null,
    prints: (index: number) => string,
    times: readonly number[] = steady(30, 600),
  ) {
    super(times, prints);
  }

  async analyzeShown(): Promise<FrameAnalysis> {
    if (this.failAnalysis || this.found === null) throw new Error("the frame was not analyzed: analysis-failed");
    this.analyzed.push(this.shown!);
    return this.found;
  }
}

/** The same in a browser that reports, for a frame, the time it was asked for. */
class EchoScoringVideo extends FakeScoringVideo {
  override readonly exactTimes = false;

  override async frameAt(timeSec: number): Promise<DecodedFrame> {
    const frame = await super.frameAt(timeSec);
    return { ...frame, timeSec: Math.min(Math.max(0, timeSec), this.durationSec) };
  }
}

const RELEASE: NonNullable<ReleaseStatus["release"]> = {
  id: "0123456789abcdef", game: "toy", locale: "en",
  detector: { version: "toy-detector", ep: "wasm" }, recognizer: { version: "toy-recognizer", ep: "wasm" },
  bankCards: 2, catalogCards: 2, identityLevel: "name", decode: { threshold: 0.3, nmsIou: 0.45, maxProposals: 80, minCropPx: 24 },
  accept: { tau: 0.62, delta: 0.05, rotationRetry: true }, hoverSelection: "accepted-nms", hoverMaxAgeMs: 2000,
  analysisWidth: { min: 720, max: 1920 }, ortWeb: "1.29.0",
};
const BACKEND = { detector: "wasm", recognizer: "wasm" } as const;

describe("a scoring run", () => {
  const manifest = manifestText(
    rowDoc(),                                                    // frame 150, found
    rowDoc({ t: 9.017 }),                                        // its frame is not in the video
    rowDoc({ video_id: "EXAMPLE0002" }),                         // no video file
    rowDoc({ video_id: "EXAMPLE0003" }),                         // a video of another shape
    rowDoc({ t: 12.017, attributes: undefined, boxes: undefined }),   // no fingerprint: by time
  );
  const rows: ScoreRow[] = parseManifest(manifest, "m.jsonl");

  async function run(change: (video: FakeScoringVideo) => void = () => undefined): Promise<{ lines: RowLine[]; video: FakeScoringVideo; seen: RowLine[] }> {
    const found = await analysis();
    const video = new FakeScoringVideo("EXAMPLE0001.mp4", 1600, 900, found, (index) => (index === 149 ? print(3) : index === 150 ? print(1) : FAR));
    const square = new FakeScoringVideo("EXAMPLE0003.mp4", 900, 900, found, () => ZERO);
    change(video);
    const seen: RowLine[] = [];
    const lines = await scoreRows({
      rows,
      backend: BACKEND,
      videoOf: async (row) => (row.videoId === "EXAMPLE0001" ? video : row.videoId === "EXAMPLE0003" ? square : null),
      onRow: (line, done, total) => {
        expect([done, total]).toEqual([seen.length + 1, rows.length]);
        seen.push(line);
      },
    });
    return { lines, video, seen };
  }

  it("ends every row in exactly one place", async () => {
    const { lines, seen } = await run();
    expect(lines.map((line) => line.status)).toEqual(["kept", "unreachable", "no-video", "frame-shape", "kept"]);
    expect(seen).toEqual(lines);
    expect(lines.every((line, i) => line.platform === rows[i]!.platform && line.video_id === rows[i]!.videoId && line.t === rows[i]!.t)).toBe(true);
    expect(lines.map((line) => line.labels)).toEqual([{ points: 2, boxes: 3 }, { points: 2, boxes: 3 }, { points: 2, boxes: 3 }, { points: 2, boxes: 3 }, { points: 2, boxes: 0 }]);
  });

  it("keeps the nearest frame within the threshold and records how it was found", async () => {
    const { lines, video } = await run();
    const kept = lines[0]!;
    expect(kept.frame).toEqual({ mediaTime: 5, offset: -0.017, distance: 1, looked: 61, exactTime: true });
    expect(kept.video).toEqual({ file: "EXAMPLE0001.mp4", width: 1600, height: 900 });
    expect(kept.analyzed).toEqual({ width: 800, height: 450 });
    expect(kept.backend).toEqual(BACKEND);
    // The chosen frame was on screen when it was analyzed: the walk had moved on, so it was shown again.
    expect(video.analyzed[0]).toBe(150);
  });

  it("records one outcome per point and per box center, in manifest order", async () => {
    const { lines } = await run();
    const kept = lines[0]!;
    expect(kept.points!.map((o) => [o.index, o.x, o.y, o.name])).toEqual([[0, 0.1, 0.2, null], [1, 0.7, 0.6, null]]);
    expect(kept.centers!.map((o) => [o.index, o.name])).toEqual([[0, "Alder"], [1, null], [2, null]]);
    expect(kept.centers![0]).toMatchObject({ cardId: "A-1", box: [100, 45, 100, 135], accepted: true });
  });

  it("never scores an unreachable row on another frame", async () => {
    const { lines, video } = await run();
    const unreachable = lines[1]!;
    expect(unreachable.frame).toMatchObject({ distance: 40, looked: 61 });
    expect(unreachable.points).toBeUndefined();
    expect(unreachable.centers).toBeUndefined();
    expect(video.analyzed).toEqual([150, 360]); // the two kept rows, and nothing else
  });

  it("does not open or read a video of another shape, and says which rows had no video", async () => {
    const { lines } = await run();
    expect(lines[2]).toEqual({ platform: "youtube", video_id: "EXAMPLE0002", t: 5.017, status: "no-video", labels: { points: 2, boxes: 3 } });
    expect(lines[3]).toMatchObject({ status: "frame-shape", video: { file: "EXAMPLE0003.mp4", width: 900, height: 900 } });
    expect(lines[3]!.frame).toBeUndefined();
  });

  it("takes a row without a fingerprint by time, with no distance", async () => {
    const { lines } = await run();
    expect(lines[4]!.frame).toMatchObject({ mediaTime: 12, distance: null, looked: 1 });
    expect(lines[4]!.centers).toEqual([]);
  });

  it("gives the same outcomes whatever was scored before", async () => {
    const forward = (await run()).lines[0]!;
    const found = await analysis();
    const video = new FakeScoringVideo("EXAMPLE0001.mp4", 1600, 900, found, (index) => (index === 150 ? print(1) : FAR));
    const alone = await scoreRows({ rows: [rows[4]!, rows[0]!], backend: BACKEND, videoOf: async () => video });
    expect(alone[1]!.points).toEqual(forward.points);
    expect(alone[1]!.centers).toEqual(forward.centers);
  });

  it("records a row that could not be analyzed as failed, with the reason, and goes on", async () => {
    const { lines } = await run((video) => {
      video.failAnalysis = true;
    });
    expect(lines.map((line) => line.status)).toEqual(["failed", "unreachable", "no-video", "frame-shape", "failed"]);
    expect(lines[0]).toMatchObject({ reason: "the frame was not analyzed: analysis-failed", labels: { points: 2, boxes: 3 } });
    expect(lines[0]!.points).toBeUndefined();
  });

  it("chooses the step for each row's window, and looks a bounded number of times", async () => {
    const { video } = await run();
    // Two windows of 61 frames at two looks a frame with four looks to choose the step, one
    // look by time, and the two kept frames shown again.
    expect(video.seeks).toBeGreaterThan(2 * 61);
    expect(video.seeks).toBeLessThanOrEqual(2 * (4 + 2 * 61 + 4) + 1 + 2);
  });

  it("is not misled by a long first frame: a later row's short frame is still found", async () => {
    // The first row sits on the frame that stays on screen for 3.2 s. The second is labeled
    // on a frame that lasts a sixtieth of a second.
    const labeled = shortFrame(8.5);
    const t = Math.round((UNEVEN[labeled]! + 0.3) * 1000) / 1000;
    const pair = parseManifest(manifestText(rowDoc({ t: 1.5 }), rowDoc({ t })), "m.jsonl");
    const video = new FakeScoringVideo("EXAMPLE0001.mp4", 1600, 900, await analysis(), (index) => (index === labeled ? print(1) : FAR), UNEVEN);
    const lines = await scoreRows({ rows: pair, backend: BACKEND, videoOf: async () => video });
    expect(lines.map((line) => line.status)).toEqual(["unreachable", "kept"]);
    expect(lines[0]!.frame).toMatchObject({ mediaTime: 0, looked: 1 });
    expect(lines[1]!.frame).toMatchObject({ mediaTime: UNEVEN[labeled], distance: 1, exactTime: true });
    expect(lines[1]!.frame!.looked).toBe(pythonWindow(UNEVEN, t - 1, t + 1).length);
    expect(video.analyzed).toEqual([labeled]);
  });

  it("records, where the browser reports the time it was asked for, that the time is not the frame's own", async () => {
    const found = await analysis();
    const video = new EchoScoringVideo("EXAMPLE0001.mp4", 1600, 900, found, (index) => (index === 149 ? print(3) : index === 150 ? print(1) : FAR));
    const lines = await scoreRows({ rows: [rows[0]!], backend: BACKEND, videoOf: async () => video });
    expect(lines[0]).toMatchObject({ status: "kept", frame: { distance: 1, looked: 126, exactTime: false } });
    // The labeled frame is on screen from 5 s for a thirtieth of a second.
    const { mediaTime, offset } = lines[0]!.frame!;
    expect(mediaTime).toBeGreaterThanOrEqual(5);
    expect(mediaTime).toBeLessThan(5 + 1 / 30);
    expect(offset).toBe(Math.round((mediaTime - 5.017) * 1000) / 1000);
    expect(video.analyzed).toEqual([150]);
    expect(lines[0]!.centers!.map((o) => o.name)).toEqual(["Alder", null, null]);
  });
});

describe("a scoring run on sections of a video", () => {
  // The video of "a scoring run": 20 seconds at 30 frames a second, frame 150 (5 s) labeled.
  // Two files hold sections of it: frames 60 to 269 (2 s to 9 s) and frames 300 to 479 (10 s to 16 s).
  const prints = (index: number): string => (index === 149 ? print(3) : index === 150 ? print(1) : FAR);
  const manifest = manifestText(
    rowDoc(),                                                         // frame 150, in the first section
    rowDoc({ t: 2.517 }),                                             // its window is cut by the section's start
    rowDoc({ t: 12.017, attributes: undefined, boxes: undefined }),   // no fingerprint: by time, in the second section
    rowDoc({ t: 18.017 }),                                            // past the end of the last section
  );
  const rows: ScoreRow[] = parseManifest(manifest, "m.jsonl");

  async function runs(): Promise<{ whole: RowLine[]; sections: RowLine[]; files: FakeScoringVideo[] }> {
    const found = await analysis();
    const video = new FakeScoringVideo("EXAMPLE0001.mp4", 1600, 900, found, prints);
    const whole = await scoreRows({ rows, backend: BACKEND, videoOf: async () => video });
    const first = new FakeScoringVideo("EXAMPLE0001+2.mp4", 1600, 900, found, (index) => prints(index + 60), steady(30, 210));
    const second = new FakeScoringVideo("EXAMPLE0001+10.mp4", 1600, 900, found, (index) => prints(index + 300), steady(30, 180));
    const { fileOf, startOf } = matchVideos(rows, [first.file, second.file]);
    expect(fileOf).toEqual([0, 0, 1, 1]);
    const files = [first, second];
    const seen = files.map((file, index) => onVideoTimeline(file, [2, 10][index]!));
    expect(startOf).toEqual([2, 2, 10, 10]);
    const sections = await scoreRows({ rows, backend: BACKEND, videoOf: async (row) => seen[fileOf[rows.indexOf(row)]!]! });
    return { whole, sections, files };
  }

  it("lands on the frames the whole video gives, with the same outcomes", async () => {
    const { whole, sections } = await runs();
    expect(sections.slice(0, 3).map((line) => line.status)).toEqual(["kept", "unreachable", "kept"]);
    for (const at of [0, 1, 2]) {
      expect(sections[at]!.status, `row ${at}`).toBe(whole[at]!.status);
      expect(sections[at]!.frame!.mediaTime, `row ${at}`).toBeCloseTo(whole[at]!.frame!.mediaTime, 6);
      expect(sections[at]!.frame!.offset).toBe(whole[at]!.frame!.offset);
      expect(sections[at]!.frame!.distance).toBe(whole[at]!.frame!.distance);
      expect(sections[at]!.points).toEqual(whole[at]!.points);
      expect(sections[at]!.centers).toEqual(whole[at]!.centers);
      expect(sections[at]!.analyzed).toEqual(whole[at]!.analyzed);
    }
  });

  it("records the file that served the row and where that file starts", async () => {
    const { whole, sections, files } = await runs();
    expect(sections[0]!.video).toEqual({ file: "EXAMPLE0001+2.mp4", width: 1600, height: 900, start: 2 });
    expect(sections[2]!.video).toEqual({ file: "EXAMPLE0001+10.mp4", width: 1600, height: 900, start: 10 });
    expect(sections[0]!.frame).toEqual({ mediaTime: 5, offset: -0.017, distance: 1, looked: 61, exactTime: true });
    // The frame analyzed is the file's frame at the row's time less the start: 3 s and 2 s in.
    expect(files[0]!.analyzed).toEqual([90]);
    expect(files[1]!.analyzed).toEqual([60]);
    // A whole video has no start.
    expect(whole[0]!.video).toEqual({ file: "EXAMPLE0001.mp4", width: 1600, height: 900 });
    expect(Object.keys(JSON.parse(JSON.stringify(sections[0]))).slice(0, 7)).toEqual(["platform", "video_id", "t", "status", "labels", "video", "frame"]);
  });

  it("looks only at the frames the file holds when the window is cut by its start", async () => {
    const { whole, sections } = await runs();
    // The window of the row at 2.517 s runs from 1.517 s; the file starts at 2 s.
    expect([whole[1]!.frame!.looked, sections[1]!.frame!.looked]).toEqual([61, 46]);
    expect(sections[1]!.frame!.distance).toBe(40);
  });

  it("gives a row past the end of its section no video, and reads no frame for it", async () => {
    const { sections, files } = await runs();
    expect(sections[3]).toEqual({ platform: "youtube", video_id: "EXAMPLE0001", t: 18.017, status: "no-video", labels: { points: 2, boxes: 3 } });
    const seeks = files[1]!.seeks;
    const again = await scoreRows({ rows: [rows[3]!], backend: BACKEND, videoOf: async () => onVideoTimeline(files[1]!, 10) });
    expect(again[0]!.status).toBe("no-video");
    expect(files[1]!.seeks).toBe(seeks);
    // A whole video that ends before the row's window starts is another matter: the row has failed in it.
    const short = new FakeScoringVideo("EXAMPLE0001.mp4", 1600, 900, await analysis(), () => FAR, steady(30, 180));
    const past = (await scoreRows({ rows: [rows[3]!], backend: BACKEND, videoOf: async () => short }))[0]!;
    expect(past).toMatchObject({ status: "failed", reason: "the video ends before t=17.017, where the window of the row at t=18.017 starts" });
    expect(past.frame).toBeUndefined();
    expect(short.seeks).toBe(0);
  });

  it("checks the shape of a section as it does a whole video's", async () => {
    const square = onVideoTimeline(new FakeScoringVideo("EXAMPLE0001+2.mp4", 900, 900, null, () => ZERO, steady(30, 210)), 2);
    const lines = await scoreRows({ rows: [rows[0]!], backend: BACKEND, videoOf: async () => square });
    expect(lines[0]).toMatchObject({ status: "frame-shape", video: { file: "EXAMPLE0001+2.mp4", width: 900, height: 900, start: 2 } });
  });
});

describe("the events file", () => {
  it("starts with a line that says what ran, then has one line per row", async () => {
    const found = await analysis();
    const video = new FakeScoringVideo("EXAMPLE0001.mp4", 800, 450, found, (index) => (index === 150 ? ZERO : FAR));
    const rows = parseManifest(manifestText(rowDoc(), rowDoc({ t: 9.017 })), "m.jsonl");
    const lines = await scoreRows({ rows, backend: BACKEND, videoOf: async () => video });
    const header = runHeader(RELEASE, { userAgent: "Test/1.0", asked: "wasm", role: "evaluation", manifests: [{ file: "m.jsonl", sha256: "0".repeat(64), rows: 2 }] });
    const text = eventLines(header, lines);
    expect(text.length).toBe(3);
    expect(text.every((line) => !line.includes("\n"))).toBe(true);
    const [first, kept, unreachable] = text.map((line) => JSON.parse(line));
    expect(first).toEqual({
      kind: EVENTS_KIND, version: 1,
      release: {
        id: "0123456789abcdef", game: "toy", locale: "en",
        detector: { version: "toy-detector", threshold: 0.3, nmsIou: 0.45, maxProposals: 80, minCropPx: 24 },
        recognizer: { version: "toy-recognizer" },
        identity: { level: "name" },
        accept: { tau: 0.62, delta: 0.05, rotationRetry: true },
        hover: { selection: "accepted-nms", nmsIou: 0.3 },
        analysisWidth: { min: 720, max: 1920 }, ortWeb: "1.29.0",
      },
      runtime: { name: RUNTIME_NAME, version: RUNTIME_VERSION },
      userAgent: "Test/1.0",
      backend: { asked: "wasm", detector: "wasm", recognizer: "wasm" },
      role: "evaluation",
      manifests: [{ file: "m.jsonl", sha256: "0".repeat(64), rows: 2 }],
      frameSelection: { windowSeconds: 1, maxHamming: 6 },
    });
    expect(Object.keys(kept)).toEqual(["platform", "video_id", "t", "status", "labels", "video", "frame", "analyzed", "backend", "points", "centers"]);
    expect(Object.keys(kept.points[0])).toEqual(["index", "x", "y", "cardId", "name", "box", "similarity", "margin", "accepted"]);
    expect(Object.keys(unreachable)).toEqual(["platform", "video_id", "t", "status", "labels", "video", "frame"]);
    expect([kept.status, unreachable.status]).toEqual(["kept", "unreachable"]);
  });

  it("is, for the made-up run of the fixture, line for line the file the Python package reads", async () => {
    // fixtures/contract.json, "events": the inputs of a run and the lines it must write. The
    // lines were built by a model of the format in Python (tests/test_browser_fixtures.py),
    // and pytest reads them with cardvision.evaluate.browser. A field that is renamed, removed
    // or changes type here fails this test.
    const fixture = contract.events;
    const text = (fixture.manifest.lines as string[]).join("\n") + "\n";
    const manifest = { name: fixture.manifest.file as string, rows: parseManifest(text, fixture.manifest.file) };
    const rows = rowsOfRole([manifest], fixture.run.role);
    const selection = fixture.release.hoverSelection;
    const bare: FrameAnalysis = {
      frame: fixture.analysis.frame, boxes: fixture.analysis.boxes, hover: { selection, cards: [] }, tracks: { shown: false, cards: [] },
      discontinuity: false, retried: 0, embeddings: 0, timings: { detectorMs: 0, recognizeMs: 0, analyzeMs: 0 },
    };
    const found: FrameAnalysis = { ...bare, hover: { selection, cards: hoverCardsOf(bare, selection) } };
    const videos = (fixture.videos as Array<Record<string, any>>).map((video) => {
      const Video = video.reportsFrameTimes ? FakeScoringVideo : EchoScoringVideo;
      return new Video(video.file, video.width, video.height, found, (index) => video.prints[String(index)] ?? video.otherPrint, steady(video.rate, video.frames));
    });
    const { fileOf, startOf } = matchVideos(rows, videos.map((video) => video.file));
    const backend = { detector: fixture.release.detector.ep, recognizer: fixture.release.recognizer.ep };
    const lines = await scoreRows({
      rows,
      backend,
      videoOf: async (row) => {
        const at = rows.indexOf(row);
        const file = fileOf[at];
        if (file === null || file === undefined) return null;
        return startOf[at] === null ? videos[file]! : onVideoTimeline(videos[file]!, startOf[at]!);
      },
    });
    const header = runHeader(fixture.release, {
      userAgent: fixture.run.userAgent, asked: fixture.run.asked, role: fixture.run.role,
      manifests: [{ file: manifest.name, sha256: createHash("sha256").update(text).digest("hex"), rows: manifest.rows.length }],
    });
    expect(lines.map((line) => line.status)).toEqual(["kept", "unreachable", "kept", "failed", "no-video", "frame-shape", "kept", "kept"]);
    const written = eventLines(header, lines);
    written.forEach((line, index) => expect(line, `line ${index + 1}`).toBe(fixture.lines[index]));
    expect(written.length).toBe(fixture.lines.length);
  });

  it("records the identity rule of the release, so that the file is scored under the rule it was recorded under", () => {
    const details = { userAgent: "Test/1.0", asked: "release" as const, role: "evaluation", manifests: [] };
    expect(runHeader(RELEASE, details).release.identity).toEqual({ level: "name" });
    const header = runHeader({ ...RELEASE, identityLevel: "printing" }, details);
    expect(header.release.identity).toEqual({ level: "printing" });
    // In the line as written: after the recognizer, where the Python model of the format puts it.
    expect(eventLines(header, [])[0]).toContain('"recognizer":{"version":"toy-recognizer"},"identity":{"level":"printing"},"accept":');
  });

  it("leaves the suppression overlap out for a selection that has none", () => {
    const header = runHeader({ ...RELEASE, hoverSelection: "best-score" }, { userAgent: "Test/1.0", asked: "release", role: "train", manifests: [] });
    expect(header.release.hover).toEqual({ selection: "best-score" });
    expect(header.backend.asked).toBe("release");
  });

  it("is summed up in counts", async () => {
    const found = await analysis();
    const video = new FakeScoringVideo("EXAMPLE0001.mp4", 800, 450, found, (index) => (index === 150 ? print(2) : FAR));
    const rows = parseManifest(manifestText(rowDoc(), rowDoc({ t: 9.017 }), rowDoc({ video_id: "EXAMPLE0002" })), "m.jsonl");
    const lines = await scoreRows({ rows, backend: BACKEND, videoOf: async (row) => (row.videoId === "EXAMPLE0001" ? video : null) });
    expect(summarize(lines)).toEqual({ rows: 3, kept: 1, unreachable: 1, noVideo: 1, frameShape: 0, failed: 0, positions: 5, shown: 1, distances: { "2": 1 } });
  });

  it("names the runtime by the version in package.json", () => {
    const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf-8"));
    expect([RUNTIME_NAME, RUNTIME_VERSION]).toEqual([pkg.name, pkg.version]);
  });
});
