// SPDX-License-Identifier: Apache-2.0
// Which decoded frame of a video stands for a manifest row. In Python: `cardvision fetch`
// (cardvision/data/extract.py and fetch.py), whose rule this follows.
//
// A row with a fingerprint is matched by content. Every frame on screen from one second before
// the row's time to one second after it is fingerprinted: the frame on screen at the start of
// that window, and each later frame whose own time is not after its end. The frame nearest the
// row's fingerprint is kept if it is within `SAME_FRAME_MAX_HAMMING` bits. A tie goes to the
// frame nearest the row's time, then to the earlier one. If no frame is that near, the row is
// unreachable: it is reported with the smallest distance seen, and it is never scored on
// another frame. A row whose window starts after the end of the video has failed.
//
// A row without a fingerprint is matched by time alone: the frame on screen at its time.
//
// The step. A browser shows the frame on screen at a time; it has no "next frame". So a window
// is walked in steps of time, and a step that is too long passes over a frame. The step is
// chosen for each window (`windowStep`) and is never longer than `MAX_STEP_SEC`, whatever the
// video's frame rate seems to be: a recording need not have one frame rate, and phones and
// screen recorders hold some frames much longer than others. A step that is shorter than a
// frame shows that frame twice, which costs a look and nothing else.
//
// The times. Where a browser reports each frame's own time, a frame shown twice is known by
// that time and counted once. Some browsers report the time that was asked for instead
// (`reportsFrameTimes` tells the two apart). There the page cannot tell one frame from the
// next, every look counts as a frame, and a frame's recorded time is the time that was asked
// for: a time at which the frame is on screen, up to one step after the frame's start.
//
// Nothing here touches a page. A video is seen through `VideoFrames`, which video.ts
// implements on a <video> element and the tests implement on made-up frames.

import { hamming, SAME_FRAME_MAX_HAMMING } from "./dhash.js";

/** A row with a fingerprint is matched against the frames this far either side of its time. */
export const FINGERPRINT_WINDOW_SECONDS = 1;

/** The longest step between two looks at a video: 16 ms. A frame that is on screen for at
 *  least this long is never stepped over, so no frame of a video of up to 60 frames a second
 *  is. A longer step would look less often and could miss a short frame of a recording whose
 *  frame rate varies; a shorter one would cost more looks for videos faster than 60 frames a
 *  second, and a look is a seek. */
export const MAX_STEP_SEC = 0.016;

const EPSILON = 1e-6;
/** The shortest frame interval the measurement can find: 250 frames a second. */
const SHORTEST_INTERVAL_SEC = 1 / 250;
/** How far apart `reportsFrameTimes` asks: far less than any frame lasts. */
const PROBE_GAP_SEC = 0.0012;

/** One frame of a video, as it was shown. */
export interface DecodedFrame {
  /** The frame's own time in the video, in seconds. Where the browser cannot tell (`exactTimes`
   *  false), the time that was asked for. */
  timeSec: number;
  /** The time that was asked for: asking for it again shows this frame again. */
  seekSec: number;
  fingerprint: string;
}

/** A video that can show the frame on screen at a time. */
export interface VideoFrames {
  readonly durationSec: number;
  /** Whether `timeSec` of a frame is the frame's own time. */
  readonly exactTimes: boolean;
  /** Show the frame on screen at `timeSec` and report it. A time past the end shows the last
   *  frame. Rejects when the browser shows no frame. */
  frameAt(timeSec: number): Promise<DecodedFrame>;
}

/**
 * Whether the times a browser reports for the frames it shows are the frames' own.
 * `reportedAt(t)` shows the frame on screen at `t` and gives the time the browser reports for
 * it. Three times are asked for, each 1.2 ms after the one before, which is far less than a
 * frame lasts. They lie in one frame, or in two that follow each other. So a browser that
 * reports a frame's own time gives the same time at least twice. A browser that reports the
 * time it was asked for gives three different times, and its times are not taken as frame
 * times. A video too short for the three is not taken to have them either.
 */
export async function reportsFrameTimes(reportedAt: (timeSec: number) => Promise<number>, durationSec: number): Promise<boolean> {
  const from = durationSec * 0.4;
  if (!(from + 3 * PROBE_GAP_SEC < durationSec)) return false;
  const reported: number[] = [];
  for (let probe = 1; probe <= 3; probe += 1) reported.push(await reportedAt(from + probe * PROBE_GAP_SEC));
  const same = (a: number, b: number): boolean => Math.abs(a - b) <= EPSILON;
  return same(reported[0]!, reported[1]!) || same(reported[1]!, reported[2]!);
}

/**
 * The time from one frame to the next, measured at `fromSec`: steps of doubling length, up to
 * `longestSec`, are tried from the frame on screen there until another frame shows. In a video
 * of one frame rate the first such step is at least one interval and under two, so it lands on
 * the very next frame. Null when no other frame is found: a video of one frame, the end of the
 * video, or a frame that stays on screen for longer than `longestSec`.
 */
export async function measureFrameInterval(video: VideoFrames, fromSec: number, longestSec = 2): Promise<number | null> {
  const first = await video.frameAt(fromSec);
  for (let step = SHORTEST_INTERVAL_SEC; step <= longestSec; step *= 2) {
    const next = await video.frameAt(first.timeSec + step);
    if (next.timeSec > first.timeSec + EPSILON) return next.timeSec - first.timeSec;
    if (first.timeSec + step >= video.durationSec) break;
  }
  return null;
}

/**
 * The step for the window that starts at `startSec`: the time from the frame on screen there
 * to the next, and never more than `MAX_STEP_SEC`. It is measured for each window and not once
 * for a video, because a frame at the start of one window says nothing about the frames of
 * another. The measurement only matters for a video faster than 60 frames a second, whose
 * frames are shorter than `MAX_STEP_SEC`; it looks no further than that. Where the browser
 * does not report frame times nothing can be measured, and the step is `MAX_STEP_SEC`.
 */
export async function windowStep(video: VideoFrames, startSec: number): Promise<number> {
  if (!video.exactTimes) return MAX_STEP_SEC;
  const from = Math.max(0, Math.min(startSec, video.durationSec));
  return Math.min((await measureFrameInterval(video, from, MAX_STEP_SEC)) ?? MAX_STEP_SEC, MAX_STEP_SEC);
}

/**
 * Every frame on screen at some moment from `startSec` to `endSec`, in order, walked in steps
 * of `stepSec`. No frame that is on screen for `stepSec` or longer is passed over.
 *
 * Where the browser reports frame times: after a new frame the next look is one and a half
 * steps after that frame's own time, which is the middle of the next frame when `stepSec` is
 * the frame interval. If the frame was already on screen for a step or more when it was first
 * seen (the frame at the start of the window can be), the next look is half a step after that
 * look instead. A look that shows the same frame again is followed by one a step later.
 *
 * Where the browser does not report frame times, the window is walked by `stepSec` and each
 * look counts as a frame.
 */
export async function framesBetween(video: VideoFrames, startSec: number, endSec: number, stepSec: number): Promise<DecodedFrame[]> {
  const frames = [await video.frameAt(Math.max(0, startSec))];
  const most = Math.ceil((endSec - startSec) / stepSec) * 2 + 16;
  const stride = video.exactTimes ? 1.5 * stepSec : stepSec;
  let target = Math.max(frames[0]!.timeSec + stride, Math.max(0, startSec) + stepSec / 2);
  for (let looks = 0; looks < most; looks += 1) {
    const last = frames[frames.length - 1]!;
    const frame = await video.frameAt(target);
    if (frame.timeSec > endSec + EPSILON) break;
    if (frame.timeSec <= last.timeSec + EPSILON) {
      // The same frame again: it is on screen for longer than one step, or the video ended.
      if (target >= video.durationSec || target > endSec + stepSec) break;
      target += stepSec;
      continue;
    }
    frames.push(frame);
    target = Math.max(frame.timeSec + stride, target + stepSec / 2);
  }
  return frames;
}

/**
 * The frame whose fingerprint is nearest `fingerprint`, with its distance in bits. Ties go to
 * the frame nearest `t` in time, then to the earlier one.
 */
export function nearestFrame(frames: readonly DecodedFrame[], fingerprint: string, t: number): { frame: DecodedFrame; distance: number } {
  if (frames.length === 0) throw new Error("no frames to compare");
  let best: { frame: DecodedFrame; distance: number } | null = null;
  for (const frame of frames) {
    const distance = hamming(frame.fingerprint, fingerprint);
    if (best === null || distance < best.distance) {
      best = { frame, distance };
      continue;
    }
    if (distance > best.distance) continue;
    const apart = Math.abs(frame.timeSec - t);
    const bestApart = Math.abs(best.frame.timeSec - t);
    if (apart < bestApart || (apart === bestApart && frame.timeSec < best.frame.timeSec)) best = { frame, distance };
  }
  return best!;
}

/** What became of the search for a row's frame. */
export type FrameChoice =
  | {
      status: "kept";
      frame: DecodedFrame;
      /** Bits between the row's fingerprint and the frame's; null for a row without one. */
      distance: number | null;
      /** How many frames were fingerprinted. */
      looked: number;
    }
  | {
      status: "unreachable";
      /** The nearest frame found and its distance: more than `SAME_FRAME_MAX_HAMMING`. */
      frame: DecodedFrame;
      distance: number;
      looked: number;
    }
  | { status: "failed"; reason: string };

/**
 * Find the frame for a row at time `t` with the given fingerprint (null: by time alone). A
 * row whose time (without a fingerprint) or whose whole window (with one) lies past the end of
 * the video has failed: the last frame is not compared with it.
 */
export async function chooseFrame(video: VideoFrames, t: number, fingerprint: string | null): Promise<FrameChoice> {
  if (fingerprint === null) {
    if (t > video.durationSec + EPSILON) return { status: "failed", reason: `the video ends before t=${t}` };
    return { status: "kept", frame: await video.frameAt(t), distance: null, looked: 1 };
  }
  const start = t - FINGERPRINT_WINDOW_SECONDS;
  if (start > video.durationSec + EPSILON) {
    return { status: "failed", reason: `the video ends before t=${Math.round(start * 1e6) / 1e6}, where the window of the row at t=${t} starts` };
  }
  const frames = await framesBetween(video, start, t + FINGERPRINT_WINDOW_SECONDS, await windowStep(video, start));
  const { frame, distance } = nearestFrame(frames, fingerprint, t);
  return { status: distance <= SAME_FRAME_MAX_HAMMING ? "kept" : "unreachable", frame, distance, looked: frames.length };
}
