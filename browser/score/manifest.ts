// SPDX-License-Identifier: Apache-2.0
// What the scoring page reads from a label manifest (data/SCHEMA.md): for each row its key,
// its role, the size of the labeled frame, its fingerprint, and the positions to ask the hover
// rule about. It does not judge the labels. The Python package validates the manifest in full
// when it scores the page's output (cardvision/data/schema.py).
//
// Positions. Every labeled point of a row, and the center of every box (the mean of its four
// corners), in the order the manifest lists them. Which of them count as hover events, and
// what each should show, is decided in Python (cardvision/evaluate/events.py). The page asks
// about all of them, so that it needs no rule of its own.

import { isDhash } from "./dhash.js";

/** Width over height of a video may differ from the row's by this fraction of it. In Python:
 *  `RATIO_TOLERANCE` in cardvision/data/extract.py. */
export const RATIO_TOLERANCE = 0.01;

/** One position of a frame: a labeled point, or the center of a labeled box. */
export interface Position {
  kind: "point" | "center";
  /** Its place in the row's `points` or `boxes`, from 0. */
  index: number;
  /** Fractions of the frame. */
  x: number;
  y: number;
}

/** One row of a manifest, narrowed to what the page uses. */
export interface ScoreRow {
  platform: string;
  videoId: string;
  /** Seconds from the start of the video at which the labeled frame is on screen. */
  t: number;
  role: string;
  frameWidth: number;
  frameHeight: number;
  /** `attributes.frame_dhash`, or null: the row is matched by time alone. */
  fingerprint: string | null;
  /** How many points and boxes the row has. */
  points: number;
  boxes: number;
  /** Points first, then box centers. */
  positions: Position[];
}

type Obj = Record<string, unknown>;
const isObj = (v: unknown): v is Obj => typeof v === "object" && v !== null && !Array.isArray(v);
const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const isText = (v: unknown): v is string => typeof v === "string" && v !== "";

function parseRow(doc: unknown): ScoreRow | string {
  if (!isObj(doc)) return "not a JSON object";
  if (!isText(doc.platform) || !isText(doc.video_id)) return "platform and video_id must be text";
  if (!isNum(doc.t) || doc.t < 0) return "t must be a number of 0 or more";
  if (!isText(doc.role)) return "role must be text";
  if (!isNum(doc.frame_width) || !isNum(doc.frame_height) || doc.frame_width <= 0 || doc.frame_height <= 0) return "frame_width and frame_height must be above 0";
  let fingerprint: string | null = null;
  if (isObj(doc.attributes) && doc.attributes.frame_dhash !== undefined) {
    if (!isDhash(doc.attributes.frame_dhash)) return "attributes.frame_dhash must be 16 lower-case hexadecimal digits";
    fingerprint = doc.attributes.frame_dhash;
  }
  const positions: Position[] = [];
  const points = doc.points ?? [];
  const boxes = doc.boxes ?? [];
  if (!Array.isArray(points) || !Array.isArray(boxes)) return "points and boxes must be lists";
  for (const [index, point] of points.entries()) {
    if (!isObj(point) || !isNum(point.x) || !isNum(point.y)) return `point ${index} must give x and y`;
    positions.push({ kind: "point", index, x: point.x, y: point.y });
  }
  for (const [index, box] of boxes.entries()) {
    const quad = isObj(box) ? box.quad : null;
    const corners = Array.isArray(quad) && quad.length === 4 && quad.every((c) => Array.isArray(c) && c.length === 2 && isNum(c[0]) && isNum(c[1])) ? (quad as number[][]) : null;
    if (corners === null) return `box ${index} must give a quad of four corners`;
    // Added in corner order and then divided, as the Python package does.
    const x = (corners[0]![0]! + corners[1]![0]! + corners[2]![0]! + corners[3]![0]!) / 4;
    const y = (corners[0]![1]! + corners[1]![1]! + corners[2]![1]! + corners[3]![1]!) / 4;
    positions.push({ kind: "center", index, x, y });
  }
  return {
    platform: doc.platform,
    videoId: doc.video_id,
    t: doc.t,
    role: doc.role,
    frameWidth: doc.frame_width,
    frameHeight: doc.frame_height,
    fingerprint,
    points: points.length,
    boxes: boxes.length,
    positions,
  };
}

/** A row's key as text: unique across a set of manifests. */
export function rowKey(row: Pick<ScoreRow, "platform" | "videoId" | "t">): string {
  return `${row.platform} ${row.videoId} t=${row.t}`;
}

/**
 * Every row of a manifest file's text, in file order. Blank lines are skipped. Throws an error
 * that names the file and each line it could not read.
 */
export function parseManifest(text: string, fileName: string): ScoreRow[] {
  const rows: ScoreRow[] = [];
  const problems: string[] = [];
  for (const [at, line] of text.split("\n").entries()) {
    if (line.trim() === "") continue;
    let doc: unknown;
    try {
      doc = JSON.parse(line);
    } catch {
      problems.push(`line ${at + 1}: not valid JSON`);
      continue;
    }
    const row = parseRow(doc);
    if (typeof row === "string") problems.push(`line ${at + 1}: ${row}`);
    else rows.push(row);
  }
  if (problems.length > 0) throw new Error(`${fileName}: ${problems.join("; ")}`);
  return rows;
}

/**
 * The rows of several manifests that have the wanted role, in order. Throws when two rows share
 * a platform, a video id and a time, as the Python package does.
 */
export function rowsOfRole(manifests: ReadonlyArray<{ name: string; rows: readonly ScoreRow[] }>, role: string): ScoreRow[] {
  const seen = new Set<string>();
  const out: ScoreRow[] = [];
  for (const manifest of manifests) {
    for (const row of manifest.rows) {
      const key = rowKey(row);
      if (seen.has(key)) throw new Error(`${manifest.name}: the row ${key} is given twice`);
      seen.add(key);
      if (row.role === role) out.push(row);
    }
  }
  return out;
}

/** Whether a video of this size can stand for the row's frame: the test is on shape, not on
 *  size, because labels are fractions of the frame. */
export function shapeMatches(row: Pick<ScoreRow, "frameWidth" | "frameHeight">, width: number, height: number): boolean {
  if (!(width > 0) || !(height > 0)) return false;
  const expected = row.frameWidth / row.frameHeight;
  return Math.abs(width / height / expected - 1) <= RATIO_TOLERANCE;
}

// ------------------------------------------------------------------------- chosen files
/**
 * Read a file the person chose, with `read`. A browser's own error for a file it cannot read
 * (one that was moved, changed or deleted after it was chosen, for instance) names no file.
 * This one does.
 */
export async function readChosen<T>(file: { readonly name: string }, read: () => Promise<T>): Promise<T> {
  try {
    return await read();
  } catch (error) {
    const reason = error instanceof Error && error.message !== "" ? error.message : String(error);
    throw new Error(`${file.name}: the file could not be read (${reason}). Choose it again.`);
  }
}

// ------------------------------------------------------------------------- video files
/** A file name without its folder. */
function baseName(name: string): string {
  return name.split(/[/\\]/).pop() ?? "";
}

/** A file name without its folder and its last extension. */
export function fileStem(name: string): string {
  const base = baseName(name);
  const dot = base.lastIndexOf(".");
  return dot > 0 ? base.slice(0, dot) : base;
}

/** What a video file's name says. */
export interface VideoName {
  /** The video: `<platform>-<video id>`, or the video id alone. */
  video: string;
  /** For a file that holds one section of the video: the time in the video, in seconds, at
   *  which the file starts. Null for a whole video. */
  start: number | null;
}

/**
 * Read a video file's name. Without its extension it is `<video>` for a whole video, or
 * `<video>+<start>` for a section that starts `<start>` seconds into the video:
 * `youtube-EXAMPLE0001+77.mp4`. These are the names `cardvision fetch --keep-video` gives
 * (cardvision/data/fetch.py, `kept_video_name`).
 */
export function parseVideoName(name: string): VideoName {
  const stem = fileStem(name);
  const section = /^(.+)\+(\d+(?:\.\d+)?)$/.exec(stem);
  return section ? { video: section[1]!, start: Number(section[2]) } : { video: stem, start: null };
}

/** The `kind` of the index `cardvision fetch --keep-video` writes beside the files it keeps. */
export const VIDEO_INDEX_KIND = "cardvision-video-files";
export const VIDEO_INDEX_VERSION = 1;

/** One file of that index. */
export interface IndexedVideo {
  /** The file's name, without a folder. */
  file: string;
  platform: string;
  videoId: string;
  /** Where the section starts and ends in the video, in seconds. Null twice for a whole video. */
  start: number | null;
  end: number | null;
}

/**
 * The files an index of kept video files lists (`video-files.json`). Throws an error that
 * names the file when it is not such an index.
 */
export function parseVideoIndex(text: string, fileName: string): IndexedVideo[] {
  let doc: unknown;
  try {
    doc = JSON.parse(text);
  } catch {
    throw new Error(`${fileName}: not valid JSON`);
  }
  if (!isObj(doc) || doc.kind !== VIDEO_INDEX_KIND || doc.version !== VIDEO_INDEX_VERSION || !Array.isArray(doc.files)) {
    throw new Error(`${fileName}: not an index of kept video files (kind ${VIDEO_INDEX_KIND}, version ${VIDEO_INDEX_VERSION})`);
  }
  return doc.files.map((entry: unknown, at: number): IndexedVideo => {
    const where = `${fileName}: files[${at}]`;
    if (!isObj(entry) || !isText(entry.file) || !isText(entry.platform) || !isText(entry.video_id)) throw new Error(`${where} must give file, platform and video_id`);
    const { start, end } = entry;
    if (start === null && end === null) return { file: baseName(entry.file), platform: entry.platform, videoId: entry.video_id, start: null, end: null };
    if (!isNum(start) || !isNum(end) || start < 0 || end <= start) throw new Error(`${where}: start and end must be null twice, or two times with 0 <= start < end`);
    return { file: baseName(entry.file), platform: entry.platform, videoId: entry.video_id, start, end };
  });
}

/** The file that serves a row. */
interface Candidate {
  index: number;
  start: number | null;
  end: number | null;
}

/**
 * Which video file serves which row.
 *
 * A file is known by its name (`parseVideoName`): `<platform>-<video id>` (the start of the
 * name `cardvision fetch` gives a frame) or the video id alone, and for a section the same
 * followed by `+<start seconds>`. `cardvision fetch --video-file ID=PATH` names a video the
 * same two ways, with a colon where this has a hyphen. A file that `indexed` lists is known by
 * its entry there instead, which also gives the section's end. A recording of your own, whose
 * platform is `local`, is matched the same way: `local-<video id>`, or the video id alone.
 *
 * - A whole-video file serves every row of its video.
 * - A section serves the rows whose time lies inside it: from its start to its end. Where the
 *   end is not known (no index), the section that starts last at or before the row's time is
 *   taken, and the run finds out whether the file reaches that far when it opens it.
 * - A whole-video file wins: where one is given, the sections of that video are not used.
 * - A name with the platform is preferred to a name without.
 *
 * Returns, per row, the index of its file in `fileNames` or null, with the start of that file
 * in the video (null for a whole video, and for no file), and the files that serve no row.
 * Throws when two files name the same video, or the same section of a video.
 */
export function matchVideos(
  rows: readonly ScoreRow[],
  fileNames: readonly string[],
  indexed: readonly IndexedVideo[] = [],
): { fileOf: Array<number | null>; startOf: Array<number | null>; unused: string[] } {
  const listed = new Map(indexed.map((entry) => [entry.file, entry]));
  const byVideo = new Map<string, Candidate[]>();
  for (const [index, name] of fileNames.entries()) {
    const entry = listed.get(baseName(name));
    const named = parseVideoName(name);
    const video = entry ? `${entry.platform}-${entry.videoId}` : named.video;
    const candidate: Candidate = entry ? { index, start: entry.start, end: entry.end } : { index, start: named.start, end: null };
    const others = byVideo.get(video) ?? [];
    const twin = others.find((other) => other.start === candidate.start);
    if (twin) throw new Error(`${fileNames[twin.index]} and ${name} name the same ${candidate.start === null ? "video" : "section of a video"}`);
    others.push(candidate);
    byVideo.set(video, others);
  }
  const whole = (video: string): Candidate | undefined => byVideo.get(video)?.find((candidate) => candidate.start === null);
  const section = (video: string, t: number): Candidate | undefined => {
    let best: Candidate | undefined;
    for (const candidate of byVideo.get(video) ?? []) {
      if (candidate.start === null || candidate.start > t || (candidate.end !== null && t > candidate.end)) continue;
      if (best === undefined || candidate.start > best.start!) best = candidate;
    }
    return best;
  };
  const used = new Set<number>();
  const served = rows.map((row) => {
    const withPlatform = `${row.platform}-${row.videoId}`;
    const found = whole(withPlatform) ?? whole(row.videoId) ?? section(withPlatform, row.t) ?? section(row.videoId, row.t) ?? null;
    if (found !== null) used.add(found.index);
    return found;
  });
  return {
    fileOf: served.map((found) => found?.index ?? null),
    startOf: served.map((found) => found?.start ?? null),
    unused: fileNames.filter((_, index) => !used.has(index)),
  };
}
