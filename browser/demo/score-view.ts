// SPDX-License-Identifier: Apache-2.0
// What the scoring page says before a run and after it, as a function of what was chosen and
// of what the run gave. Nothing here touches the page or runs anything, so every case can be
// tested without a browser (test/score-view.test.ts).
//
// Before a run the page counts: how many rows of the chosen role the manifests hold, and how
// many of those have a video file that matches them. With no such row, or with no matched
// video, Start does not start a run: the page says what is wrong and what to do, beside the
// input it concerns. With some rows matched and some not, a run may start, and the number
// left out is said before it and after it.
//
// The check is the page's, for a person. `window.cardvisionScore()` is not held back by it: a
// tool that calls it gets the lines of the events file for whatever was chosen, as before.

import { type IndexedVideo, matchVideos, rowsOfRole, type ScoreRow } from "../score/manifest.js";
import type { RunSummary } from "../score/run.js";

/** The three inputs of the page, in their order. A message is shown beside the one it concerns. */
export type Input = "model" | "labels" | "videos";

function plural(count: number, one: string, many = `${one}s`): string {
  return `${count} ${count === 1 ? one : many}`;
}

/** A list in words: "a", "a and b", "a, b and c"; past `most` items, "a, b and 3 more". */
function listed(items: readonly string[], most = 3): string {
  const named = items.length > most ? [...items.slice(0, most - 1), `${items.length - (most - 1)} more`] : [...items];
  if (named.length <= 1) return named.join("");
  return `${named.slice(0, -1).join(", ")} and ${named[named.length - 1]}`;
}

/** What the chosen manifests and video files hold for one role. */
export interface Coverage {
  role: string;
  /** The manifest files, by name. */
  manifests: string[];
  /** Rows of the chosen role. */
  rows: number;
  /** How many of them have a video file. */
  matched: number;
  /** How many rows each role has, over all the manifests. */
  roles: Record<string, number>;
  /** The videos of the rows without a file, as `<platform>-<video id>`, each once. */
  wanted: string[];
  /** The video files that serve no row. */
  unused: string[];
}

/** Count the rows of a role and the videos that match them, with the rules a run uses
 *  (score/manifest.ts). It throws what they throw: a row given twice, two files for one video. */
export function coverage(
  manifests: ReadonlyArray<{ name: string; rows: readonly ScoreRow[] }>,
  role: string,
  videoNames: readonly string[],
  indexed: readonly IndexedVideo[] = [],
): Coverage {
  const rows = rowsOfRole(manifests, role);
  const roles: Record<string, number> = {};
  for (const manifest of manifests) for (const row of manifest.rows) roles[row.role] = (roles[row.role] ?? 0) + 1;
  const { fileOf, unused } = matchVideos(rows, videoNames, indexed);
  const wanted = new Set<string>();
  rows.forEach((row, index) => {
    if (fileOf[index] === null) wanted.add(`${row.platform}-${row.videoId}`);
  });
  return {
    role, manifests: manifests.map((manifest) => manifest.name), rows: rows.length,
    matched: fileOf.filter((file) => file !== null).length, roles, wanted: [...wanted], unused,
  };
}

/** What was chosen on the page, as far as the check needs it. */
export interface Chosen {
  release: string;
  manifests: number;
  videos: number;
}

export interface PreRun {
  /** Whether Start starts a run. */
  start: boolean;
  /** The line beside Start: what a run would read, or what is still to choose. */
  summary: string;
  /** Rows of the role that a run would leave out for want of a video. */
  leftOut: number;
  /** What is wrong, by the input it concerns. Shown as soon as it is known. */
  problems: Partial<Record<Input, string>>;
  /** What is not chosen yet, by input. Shown only when Start is pressed without it. */
  missing: Partial<Record<Input, string>>;
}

/**
 * What the page says before a run. `found` is the count for what was chosen, or null while
 * there is nothing to count; `unreadable` is a chosen file that could not be read as what it
 * should be, with the input it was chosen in.
 */
export function preRun(chosen: Chosen, found: Coverage | null, unreadable: { where: Input; text: string } | null = null): PreRun {
  const missing: PreRun["missing"] = {};
  if (chosen.release === "") missing.model = "Give the address of a release document.";
  if (chosen.manifests === 0) missing.labels = "Choose one or more manifest files.";
  if (chosen.videos === 0) missing.videos = "Choose the video files of those manifests.";
  const stopped = (summary: string, problems: PreRun["problems"] = {}): PreRun => ({ start: false, summary, leftOut: 0, problems, missing });

  if (unreadable !== null) return stopped("A run cannot start yet: see the note above.", { [unreadable.where]: unreadable.text });
  if (Object.keys(missing).length > 0 || found === null) {
    const todo = [...(missing.model ? ["a model"] : []), ...(missing.labels ? ["the manifest files"] : []), ...(missing.videos ? ["their videos"] : [])];
    return stopped(todo.length > 0 ? `Choose ${listed(todo)}, then press Start.` : "Reading what was chosen…");
  }

  const files = listed(found.manifests);
  if (found.rows === 0) {
    const held = Object.entries(found.roles).sort(([a], [b]) => a.localeCompare(b)).map(([role, rows]) => `${plural(rows, "row")} of the role ${role}`);
    const holds = held.length > 0 ? `${files} ${found.manifests.length === 1 ? "holds" : "hold"} ${listed(held, 4)}` : `${files} ${found.manifests.length === 1 ? "holds" : "hold"} no rows`;
    return stopped("A run cannot start yet: see the note above.", {
      labels: `No row has the role ${found.role}: ${holds}. Choose another role under Options, or other manifest files.`,
    });
  }
  const unused = found.unused.length > 0 ? ` Not used: ${listed(found.unused)}.` : "";
  if (found.matched === 0) {
    const names = found.wanted.map((video) => `the video ${video}`);
    return stopped("A run cannot start yet: see the note above.", {
      videos: `No video file matches a row. The ${plural(found.rows, "row")} of the role ${found.role} ${found.rows === 1 ? "is" : "are"} of ${listed(names)}. ` +
        `Give each video file the name of its video, as below, and choose it again.${unused}`,
    });
  }
  const leftOut = found.rows - found.matched;
  const rows = `${plural(found.rows, "row")} of the role ${found.role} in ${files}`;
  const summary = leftOut === 0
    ? `${rows}. ${found.rows === 1 ? "It has" : `All ${found.rows} have`} a matching video.${unused}`
    : `${rows}. ${found.matched} of them ${found.matched === 1 ? "has" : "have"} a matching video. ` +
      `${plural(leftOut, "row")} will be left out: no video file matches ${listed(found.wanted.map((video) => `the video ${video}`))}.${unused}`;
  return { start: true, summary, leftOut, problems: {}, missing };
}

export interface ResultView {
  /** Whether the events file holds a row to score. */
  usable: boolean;
  title: string;
  /** The counts shown large: rows kept, unreachable and failed. */
  counts: Array<{ label: string; rows: number }>;
  /** What was left out, and what an empty result means. */
  notes: string[];
  /** The words of the download button. */
  download: string;
}

/** The name the events file is downloaded under. */
export const EVENTS_FILE = "browser-events.jsonl";

/** What the page says when a run has ended: a file to score, or no usable rows. */
export function resultView(summary: RunSummary): ResultView {
  const usable = summary.kept > 0;
  const notes: string[] = [];
  if (summary.noVideo > 0) notes.push(`${plural(summary.noVideo, "row")} had no matching video and ${summary.noVideo === 1 ? "was" : "were"} left out.`);
  if (summary.frameShape > 0) notes.push(`${plural(summary.frameShape, "row")} had a video of another shape than the labeled frame and ${summary.frameShape === 1 ? "was" : "were"} left out.`);
  if (!usable) notes.push(`No row was kept, so this file has nothing to score. It says why each row was left out: download it to find out what went wrong, not to score it.`);
  return {
    usable,
    title: usable ? "Events ready for scoring" : "No usable rows",
    counts: [{ label: "Kept", rows: summary.kept }, { label: "Unreachable", rows: summary.unreachable }, { label: "Failed", rows: summary.failed }],
    notes,
    download: usable ? `Download ${EVENTS_FILE}` : `Download ${EVENTS_FILE} (for diagnosis only)`,
  };
}

/** A piece of the scoring command: words to copy as they are, or a placeholder to replace. */
export interface CommandPart {
  text: string;
  placeholder: boolean;
}

/** The folders the page cannot know: a browser gives a page the name of a chosen file, not its folder. */
export const PLACEHOLDERS = { downloads: "DOWNLOADS-FOLDER", labels: "LABELS-FOLDER", release: "RELEASE-FOLDER" } as const;

const PLAIN_NAME = /^[A-Za-z0-9._+@%,=-]+$/;

/** One file inside a folder the page cannot know, as an argument a shell reads as one word. */
function inFolder(folder: string, name: string): CommandPart[] {
  if (PLAIN_NAME.test(name)) return [{ text: folder, placeholder: true }, { text: `/${name}`, placeholder: false }];
  return [{ text: '"', placeholder: false }, { text: folder, placeholder: true }, { text: `/${name.replace(/["$`\\]/g, "\\$&")}"`, placeholder: false }];
}

/** The role `cardvision evaluate hover` scores when it is given none. */
export const DEFAULT_ROLE = "evaluation";

/**
 * The command that scores the events file, with what the page knows filled in: the names of
 * the manifest files and of the release document, and the role the run scored. The folders
 * they are in, and the folder the browser saved the events file to, are placeholders in
 * capitals. `role` is the role of the rows the run read: the command scores the evaluation
 * rows unless it is told otherwise, so any other role is named with `--role`.
 */
export function scoringCommand(manifests: readonly string[], releaseAddress: string, role: string = DEFAULT_ROLE): CommandPart[] {
  const plain = (text: string): CommandPart => ({ text, placeholder: false });
  let document = "release.json";
  try {
    document = decodeURIComponent(new URL(releaseAddress, "http://localhost/").pathname.split("/").pop() ?? "") || document;
  } catch {
    // An address that cannot be read as one: the usual name stands in.
  }
  return [
    plain("cardvision evaluate hover --browser-events "),
    ...inFolder(PLACEHOLDERS.downloads, EVENTS_FILE),
    plain(" \\\n    --manifest"),
    ...manifests.flatMap((name) => [plain(" "), ...inFolder(PLACEHOLDERS.labels, name)]),
    plain(" \\\n    --release "),
    ...inFolder(PLACEHOLDERS.release, document),
    ...(role === DEFAULT_ROLE ? [] : [plain(` \\\n    --role ${role}`)]),
  ];
}

/** The command as text, as it is copied. */
export function commandText(parts: readonly CommandPart[]): string {
  return parts.map((part) => part.text).join("");
}
