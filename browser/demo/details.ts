// SPDX-License-Identifier: Apache-2.0
// What the demo page's Details shows, as a function of what it knows. Like view.ts, nothing
// here touches the page, so every state can be tested without a browser
// (test/demo-details.test.ts).
//
// Details is a diagnostics view: one line that says what state the page is in, four figures of
// the newest analyzed frame, the detections of that frame in a table, and what the model and
// the file are. Every figure is one the session reports: where it reports nothing, Details
// shows a dash or leaves the figure out. The text panel the page has always printed
// (`panelLines` in view.ts) stays under it, as "Raw text", for scripts and bug reports.

import type { FrameAnalysis } from "../src/pipeline.js";
import type { ExecutionProvider } from "../src/providers.js";
import type { AcceptRule } from "../src/recognizer.js";
import { BACKEND, type DemoView, type MediaState, type ModelState, type Tone } from "./view.js";

/** What a session reports, as far as Details shows it (a part of `SessionStatus`). */
export interface DetailsStatus {
  release: {
    id: string;
    game: string;
    bankCards: number;
    catalogCards: number;
    detector: { version: string; ep: ExecutionProvider };
    recognizer: { version: string; ep: ExecutionProvider };
    accept: Pick<AcceptRule, "tau" | "delta">;
  } | null;
  schedule: string;
  frames: number;
  lastWorkerMs: number | null;
}

/** The file on show or chosen, as the page knows it. Sizes are null until the browser has
 *  decoded the file; the duration is that of a video, null for an image. */
export interface DetailsFile {
  name: string;
  servedAt: string | null;
  video: boolean;
  width: number | null;
  height: number | null;
  durationSec: number | null;
}

/** A figure of the header row: its label, a large value, its unit, and one line under it. */
export interface Tile {
  key: "cards" | "time" | "backend" | "frames";
  label: string;
  value: string;
  unit: string;
  note: string;
}

/** The worker's time for the last frames, newest last, for the sparkline. */
export interface Sparkline {
  points: number[];
  latest: number;
  max: number;
  min: number;
}

/** What one detected box was found to be. "named" faces show a name under the pointer;
 *  "unnamed" faces were refused by the accept rule; anything else is another class (a back). */
export type RowKind = "named" | "unnamed" | "other";

export interface DetectionRow {
  /** Its place in the frame's list of boxes, from 1. */
  number: number;
  kind: RowKind;
  /** The card's name, "Not named", or what the detector called the box ("Card back"). */
  card: string;
  /** The card's id when named; for a face that is not named, the nearest card in the bank. */
  id: string | null;
  /** Whether `id` is the nearest card of a face that is not named. */
  nearest: boolean;
  /** The look the recognizer chose when it is not upright, in words. */
  turned: string | null;
  /** The score and the margin the accept rule compared, for a face. */
  score: string | null;
  margin: string | null;
  scoreLow: boolean;
  marginLow: boolean;
  /** The accept rule's answer, in words. */
  result: string;
  /** Whether the pointer is on this box's card. */
  pointer: boolean;
}

export interface Fact {
  label: string;
  value: string;
  /** The whole value, where `value` is a shortened one. */
  full?: string;
}

export interface DetailsView {
  /** The state of the page, in one line, with the tone of the page's state line. */
  tone: Tone;
  line: string;
  /** The header row; null while no model has loaded. */
  tiles: Tile[] | null;
  spark: Sparkline | null;
  /** The detections of the newest frame; null when the page is not ready. */
  frame: {
    /** What the table is about: the size the frame was analyzed at, and the accept rule. */
    caption: string;
    rows: DetectionRow[];
    /** Said in place of the table when it has no row. */
    empty: string | null;
  } | null;
  model: Fact[] | null;
  file: Fact[] | null;
}

/** How many frames the sparkline shows. */
export const SPARK_FRAMES = 30;

const DASH = "—";

const ms = (value: number): string => value.toFixed(1);
const fixed3 = (value: number): string => value.toFixed(3);
const plural = (n: number, one: string, many = `${one}s`): string => (n === 1 ? one : many);

/** The schedule a session reports, in words. */
const SCHEDULE: Record<string, string> = {
  fast: "analyzing while the pointer moves",
  reading: "slower while a name shows",
  idle: "paused until the pointer moves",
  "always-on": "analyzing continuously",
};

/** A release id as the Model group shows it: the first eight characters. */
export function shortId(id: string): string {
  return id.length > 8 ? `${id.slice(0, 8)}…` : id;
}

/** A duration in seconds as minutes and seconds, to a tenth under a minute. */
export function durationText(sec: number): string {
  if (sec < 60) return `${sec.toFixed(1)} s`;
  const whole = Math.round(sec);
  return `${Math.floor(whole / 60)} min ${String(whole % 60).padStart(2, "0")} s`;
}

/** The sparkline of the worker's times, or null with no time yet. */
export function sparkOf(times: readonly number[]): Sparkline | null {
  const points = times.slice(-SPARK_FRAMES);
  if (points.length === 0) return null;
  return { points, latest: points[points.length - 1]!, max: Math.max(...points), min: Math.min(...points) };
}

/** The one line that says what Details shows, for each state of the page. */
function lineOf(view: DemoView, model: ModelState, media: MediaState, status: DetailsStatus | null, analysis: FrameAnalysis | null, frameError: string | null): string {
  if (model.kind === "none") return "Nothing loaded yet. Load a model and choose a file: what it finds in each frame shows here.";
  if (model.kind === "loading") return "Loading the model. What it is, and then each frame's figures, show here once it has loaded.";
  if (model.kind === "failed") return `The model did not load (${model.reason}), so nothing is analyzed.`;
  if (media.kind === "none") return "Model loaded, no file yet. Choose a file to see what each frame holds.";
  if (media.kind === "opening") return `Model loaded. Opening ${media.name}…`;
  if (media.kind !== "shown") return "The file did not open, so nothing was analyzed.";
  if (frameError !== null) return `The last frame was not analyzed (${frameError}).`;
  if (!view.ready || status === null || analysis === null) {
    // In pointer mode nothing is analyzed until the pointer moves over the picture.
    return status?.schedule === "always-on" ? "Running. No analyzed frame matches the picture on screen yet." : "Running. No analyzed frame matches the picture on screen yet: move the pointer over it.";
  }
  return "Running. The figures are those of the newest analyzed frame.";
}

/** The words of the accept rule's answer for one face, and which of its two figures fell short. */
function resultOf(face: NonNullable<FrameAnalysis["boxes"][number]["face"]>, accept: Pick<AcceptRule, "tau" | "delta"> | null): { result: string; scoreLow: boolean; marginLow: boolean } {
  if (face.cardId !== null) return { result: "Named", scoreLow: false, marginLow: false };
  const scoreLow = accept !== null && face.similarity < accept.tau;
  const marginLow = accept !== null && face.margin < accept.delta;
  if (face.accepted) return { result: "Accepted, not in the catalog", scoreLow, marginLow };
  const short = [scoreLow ? `score under ${accept!.tau}` : null, marginLow ? `margin under ${accept!.delta}` : null].filter((part) => part !== null);
  return { result: short.length > 0 ? `Not named: ${short.join(", ")}` : "Not named", scoreLow, marginLow };
}

/** The rows of the detections table: one per detected box, in the order the session reports
 *  them (best detector score first). `pointerBox` is the index of the box under the pointer. */
export function detectionRows(analysis: FrameAnalysis, accept: Pick<AcceptRule, "tau" | "delta"> | null, pointerBox: number | null): DetectionRow[] {
  return analysis.boxes.map((item, i) => {
    const pointer = pointerBox === i;
    const face = item.face;
    if (face === null) {
      return {
        number: i + 1, kind: "other", card: item.className === "back" ? "Card back" : `Not a face (${item.className})`, id: null, nearest: false, turned: null,
        score: null, margin: null, scoreLow: false, marginLow: false, result: "Not looked up", pointer,
      };
    }
    const named = face.cardId !== null;
    return {
      number: i + 1,
      kind: named ? "named" : "unnamed",
      card: named ? (face.name ?? face.cardId!) : "Not named",
      id: named ? face.cardId : face.topId,
      nearest: !named,
      turned: face.look === 0 ? null : `turned ${face.look * 90}°`,
      score: fixed3(face.similarity),
      margin: fixed3(face.margin),
      ...resultOf(face, accept),
      pointer,
    };
  });
}

/**
 * The box under a point of the analyzed frame, for the row Details highlights. `hoverBox` is
 * the box of the card the hover layer resolves the point to (`hoverAt`), which settles which of
 * two overlapping faces is on top; it holds faces only, so a point on no face falls back to the
 * smallest box that contains it, such as a card back.
 */
export function boxAtPoint(analysis: FrameAnalysis, x: number, y: number, hoverBox: number | null): number | null {
  if (hoverBox !== null) return hoverBox;
  let best: number | null = null;
  let bestArea = Infinity;
  analysis.boxes.forEach((item, i) => {
    const [left, top, width, height] = item.box;
    if (x < left || y < top || x > left + width || y > top + height) return;
    if (width * height < bestArea) {
      best = i;
      bestArea = width * height;
    }
  });
  return best;
}

function backendTile(detector: ExecutionProvider, recognizer: ExecutionProvider): Tile {
  if (detector === recognizer) return { key: "backend", label: "Backend", value: BACKEND[detector], unit: "", note: "detector and recognizer" };
  return { key: "backend", label: "Backend", value: "Mixed", unit: "", note: `detector on ${BACKEND[detector]}, recognizer on ${BACKEND[recognizer]}` };
}

/**
 * Everything Details shows for the page as it is now.
 *
 * `analysis` is the session's newest result and `times` the worker's times of the frames it
 * analyzed, newest last; both count only while the view is ready, so that no figure of an
 * earlier model or file stays on screen. `pointerBox` is the index of the box under the pointer
 * in `analysis`, or null.
 */
export function detailsView(
  view: DemoView,
  model: ModelState,
  media: MediaState,
  file: DetailsFile | null,
  status: DetailsStatus | null,
  frameError: string | null,
  analysis: FrameAnalysis | null,
  times: readonly number[],
  pointerBox: number | null,
): DetailsView {
  const release = model.kind === "loaded" ? status?.release ?? null : null;
  const running = view.ready && status !== null;
  const frame = running ? analysis : null;

  let tiles: Tile[] | null = null;
  if (release) {
    const faces = frame ? frame.boxes.filter((item) => item.face !== null) : [];
    const named = faces.filter((item) => item.face?.cardId !== null).length;
    const others = frame ? frame.boxes.length - faces.length : 0;
    const backs = frame ? frame.boxes.filter((item) => item.face === null && item.className === "back").length : 0;
    const otherWords = others === backs ? `${others} ${plural(others, "back")}` : `${others} other`;
    tiles = [
      frame
        ? { key: "cards", label: "Cards found", value: String(frame.boxes.length), unit: plural(frame.boxes.length, "card"), note: `${named} named · ${faces.length - named} not named · ${otherWords}` }
        : { key: "cards", label: "Cards found", value: DASH, unit: "", note: "in the newest frame" },
      running && status.lastWorkerMs !== null
        ? { key: "time", label: "Time per frame", value: ms(status.lastWorkerMs), unit: "ms", note: frame ? `detect ${ms(frame.timings.detectorMs)} · recognize ${ms(frame.timings.recognizeMs)} ms` : "in the worker" }
        : { key: "time", label: "Time per frame", value: DASH, unit: "", note: "in the worker" },
      backendTile(release.detector.ep, release.recognizer.ep),
      running
        ? { key: "frames", label: "Frames analyzed", value: String(status.frames), unit: plural(status.frames, "frame"), note: SCHEDULE[status.schedule] ?? status.schedule }
        : { key: "frames", label: "Frames analyzed", value: DASH, unit: "", note: "since the file opened" },
    ];
  }

  let frameView: DetailsView["frame"] = null;
  if (running) {
    const accept = release?.accept ?? null;
    const rule = accept ? `named when the score is at least ${accept.tau} and the margin at least ${accept.delta}` : "";
    if (frame) {
      const rows = detectionRows(frame, accept, pointerBox);
      frameView = { caption: [`analyzed at ${frame.frame.width} × ${frame.frame.height} px`, rule].filter((part) => part !== "").join(" · "), rows, empty: rows.length === 0 ? "No card was found in this frame." : null };
    } else {
      frameView = { caption: rule, rows: [], empty: "No analyzed frame matches the picture on screen yet." };
    }
  }

  let modelFacts: Fact[] | null = null;
  if (model.kind !== "none") {
    modelFacts = [];
    if (release) {
      modelFacts.push(
        { label: "Release", value: shortId(release.id), full: release.id },
        { label: "Game", value: release.game },
        { label: "Cards", value: `${release.bankCards} in the bank, ${release.catalogCards} in the catalog` },
        { label: "Detector", value: release.detector.version },
        { label: "Recognizer", value: release.recognizer.version },
      );
    }
    modelFacts.push({ label: "Address", value: model.address });
  }

  let fileFacts: Fact[] | null = null;
  if (file !== null) {
    fileFacts = [{ label: "Name", value: file.name }];
    if (file.width !== null && file.height !== null) fileFacts.push({ label: "Size", value: `${file.width} × ${file.height} px` });
    if (file.video && file.durationSec !== null && Number.isFinite(file.durationSec)) fileFacts.push({ label: "Duration", value: durationText(file.durationSec) });
    if (file.servedAt !== null) fileFacts.push({ label: "From", value: `the local server's ${file.servedAt}` });
  }

  return {
    tone: view.state.tone,
    line: lineOf(view, model, media, status, frame, frameError),
    tiles,
    spark: running ? sparkOf(times) : null,
    frame: frameView,
    model: modelFacts,
    file: fileFacts,
  };
}
