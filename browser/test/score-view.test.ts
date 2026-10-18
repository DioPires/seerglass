// SPDX-License-Identifier: Apache-2.0
// What the scoring page says before a run and after it (demo/score-view.ts): the count of rows
// and matched videos, when Start starts nothing, the result card and the command. No page and
// no run is started here.
import { describe, expect, it } from "vitest";

import { commandText, coverage, DEFAULT_ROLE, EVENTS_FILE, PLACEHOLDERS, preRun, resultView, scoringCommand } from "../demo/score-view.js";
import { parseManifest } from "../score/manifest.js";
import type { RunSummary } from "../score/run.js";

const rowDoc = (over: Record<string, unknown> = {}): Record<string, unknown> => ({
  manifest_version: 1, platform: "local", video_id: "table", t: 1.5, frame_width: 1280, frame_height: 768,
  source_type: "digital", group: "g-toy-table", role: "evaluation", completeness: "points",
  points: [{ x: 0.1, y: 0.2, kind: "not-card" }],
  ...over,
});
const manifest = (name: string, ...docs: Array<Record<string, unknown>>) => ({ name, rows: parseManifest(docs.map((doc) => JSON.stringify(doc)).join("\n") + "\n", name) });

/** Three rows of one video, as the toy manifest has them. */
const TABLE = manifest("table.manifest.jsonl", rowDoc(), rowDoc({ t: 3.5 }), rowDoc({ t: 4.5 }));
/** Two videos and two roles. */
const MIXED = manifest("mixed.manifest.jsonl", rowDoc(), rowDoc({ video_id: "second-game" }), rowDoc({ video_id: "second-game", t: 9 }), rowDoc({ t: 7, role: "train" }));
const CHOSEN = { release: "/release/release.json", manifests: 1, videos: 1 };
const STOPPED = "A run cannot start yet: see the note above.";

describe("counting what a run would read", () => {
  it("counts the rows of the role, the rows with a video, and what is left over", () => {
    expect(coverage([TABLE], "evaluation", ["table.webm"])).toEqual({
      role: "evaluation", manifests: ["table.manifest.jsonl"], rows: 3, matched: 3, roles: { evaluation: 3 }, wanted: [], unused: [],
    });
    expect(coverage([TABLE], "train", ["table.webm"])).toMatchObject({ rows: 0, matched: 0, roles: { evaluation: 3 }, unused: ["table.webm"] });
    expect(coverage([TABLE], "evaluation", ["other-video.webm"])).toMatchObject({ rows: 3, matched: 0, wanted: ["local-table"], unused: ["other-video.webm"] });
    expect(coverage([MIXED, manifest("b.jsonl", rowDoc({ video_id: "third" }))], "evaluation", ["local-table.webm", "notes.mp4"])).toEqual({
      role: "evaluation", manifests: ["mixed.manifest.jsonl", "b.jsonl"], rows: 4, matched: 1, roles: { evaluation: 4, train: 1 },
      wanted: ["local-second-game", "local-third"], unused: ["notes.mp4"],
    });
    // A section serves the rows at or after its start, as in a run.
    expect(coverage([TABLE], "evaluation", ["table+3.webm"])).toMatchObject({ rows: 3, matched: 2, wanted: ["local-table"] });
  });

  it("fails as a run would on a row given twice and on two files for one video", () => {
    expect(() => coverage([TABLE, TABLE], "evaluation", ["table.webm"])).toThrow("is given twice");
    expect(() => coverage([TABLE], "evaluation", ["table.webm", "table.mp4"])).toThrow("table.webm and table.mp4 name the same video");
  });
});

describe("what the scoring page says before a run", () => {
  it("says what is still to choose, and names each missing input only for a pressed Start", () => {
    const nothing = preRun({ release: "", manifests: 0, videos: 0 }, null);
    expect([nothing.start, nothing.summary, nothing.problems]).toEqual([false, "Choose a model, the manifest files and their videos, then press Start.", {}]);
    expect(nothing.missing).toEqual({
      model: "Give the address of a release document.", labels: "Choose one or more manifest files.", videos: "Choose the video files of those manifests.",
    });
    const noVideos = preRun({ ...CHOSEN, videos: 0 }, null);
    expect([noVideos.start, noVideos.summary, noVideos.missing]).toEqual([false, "Choose their videos, then press Start.", { videos: "Choose the video files of those manifests." }]);
    expect(preRun({ ...CHOSEN, manifests: 0, videos: 0 }, null).summary).toBe("Choose the manifest files and their videos, then press Start.");
    // The count is there and the model's address is not: still no run.
    const noModel = preRun({ ...CHOSEN, release: "" }, coverage([TABLE], "evaluation", ["table.webm"]));
    expect([noModel.start, noModel.summary, noModel.missing]).toEqual([false, "Choose a model, then press Start.", { model: "Give the address of a release document." }]);
  });

  it("starts a run when every row of the role has a video, and says how many", () => {
    const check = preRun(CHOSEN, coverage([TABLE], "evaluation", ["table.webm"]));
    expect(check).toEqual({
      start: true, leftOut: 0, problems: {}, missing: {},
      summary: "3 rows of the role evaluation in table.manifest.jsonl. All 3 have a matching video.",
    });
    const one = preRun(CHOSEN, coverage([manifest("one.jsonl", rowDoc())], "evaluation", ["table.webm", "spare.webm"]));
    expect(one.summary).toBe("1 row of the role evaluation in one.jsonl. It has a matching video. Not used: spare.webm.");
  });

  it("does not start a run with no row of the role, and says what the manifests hold", () => {
    const check = preRun(CHOSEN, coverage([TABLE], "train", ["table.webm"]));
    expect([check.start, check.summary, check.leftOut]).toEqual([false, STOPPED, 0]);
    expect(check.problems).toEqual({
      labels: "No row has the role train: table.manifest.jsonl holds 3 rows of the role evaluation. Choose another role under Options, or other manifest files.",
    });
    const two = preRun(CHOSEN, coverage([MIXED, manifest("b.jsonl", rowDoc({ video_id: "third" }))], "extra", ["table.webm"]));
    expect(two.problems.labels).toBe(
      "No row has the role extra: mixed.manifest.jsonl and b.jsonl hold 4 rows of the role evaluation and 1 row of the role train. Choose another role under Options, or other manifest files.",
    );
    expect(preRun(CHOSEN, coverage([{ name: "empty.jsonl", rows: [] }], "evaluation", ["table.webm"])).problems.labels).toContain("empty.jsonl holds no rows");
  });

  it("does not start a run when no video matches a row, and says which videos the rows are of", () => {
    const check = preRun(CHOSEN, coverage([TABLE], "evaluation", ["other-video.webm"]));
    expect([check.start, check.summary]).toEqual([false, STOPPED]);
    expect(check.problems).toEqual({
      videos: "No video file matches a row. The 3 rows of the role evaluation are of the video local-table. " +
        "Give each video file the name of its video, as below, and choose it again. Not used: other-video.webm.",
    });
    const many = preRun(CHOSEN, coverage([MIXED, manifest("b.jsonl", rowDoc({ video_id: "third" }), rowDoc({ video_id: "fourth" }))], "evaluation", ["x.mp4"]));
    expect(many.problems.videos).toContain("The 5 rows of the role evaluation are of the video local-table, the video local-second-game and 2 more.");
  });

  it("starts a run on part of the rows, and says how many are left out", () => {
    const check = preRun(CHOSEN, coverage([MIXED], "evaluation", ["table.webm"]));
    expect([check.start, check.leftOut, check.problems]).toEqual([true, 2, {}]);
    expect(check.summary).toBe(
      "3 rows of the role evaluation in mixed.manifest.jsonl. 1 of them has a matching video. 2 rows will be left out: no video file matches the video local-second-game.",
    );
    expect(preRun(CHOSEN, coverage([TABLE], "evaluation", ["table+3.webm"])).summary).toContain("2 of them have a matching video. 1 row will be left out");
  });

  it("puts a file that cannot be read beside the input it was chosen in, and starts nothing", () => {
    const check = preRun(CHOSEN, null, { where: "labels", text: "notpic.txt: line 1: not valid JSON." });
    expect([check.start, check.summary, check.problems]).toEqual([false, STOPPED, { labels: "notpic.txt: line 1: not valid JSON." }]);
    expect(preRun(CHOSEN, null, { where: "videos", text: "a.mp4 and a.webm name the same video." }).problems).toEqual({ videos: "a.mp4 and a.webm name the same video." });
  });
});

describe("what the scoring page says when a run has ended", () => {
  const summary = (over: Partial<RunSummary>): RunSummary => ({ rows: 3, kept: 3, unreachable: 0, noVideo: 0, frameShape: 0, failed: 0, positions: 18, shown: 12, distances: { "0": 3 }, ...over });

  it("offers the events file for scoring when a row was kept", () => {
    expect(resultView(summary({}))).toEqual({
      usable: true, title: "Events ready for scoring", notes: [], download: "Download browser-events.jsonl",
      counts: [{ label: "Kept", rows: 3 }, { label: "Unreachable", rows: 0 }, { label: "Failed", rows: 0 }],
    });
    expect(EVENTS_FILE).toBe("browser-events.jsonl");
  });

  it("says after a run on part of the rows how many were left out", () => {
    const part = resultView(summary({ kept: 1, unreachable: 1, noVideo: 2, frameShape: 1, failed: 1, rows: 6 }));
    expect([part.usable, part.title]).toEqual([true, "Events ready for scoring"]);
    expect(part.counts.map((count) => count.rows)).toEqual([1, 1, 1]);
    expect(part.notes).toEqual([
      "2 rows had no matching video and were left out.",
      "1 row had a video of another shape than the labeled frame and was left out.",
    ]);
  });

  it("calls a result with no kept row what it is, and offers its file for diagnosis only", () => {
    const empty = resultView(summary({ kept: 0, noVideo: 3 }));
    expect([empty.usable, empty.title, empty.download]).toEqual([false, "No usable rows", "Download browser-events.jsonl (for diagnosis only)"]);
    expect(empty.notes).toEqual([
      "3 rows had no matching video and were left out.",
      "No row was kept, so this file has nothing to score. It says why each row was left out: download it to find out what went wrong, not to score it.",
    ]);
    expect(resultView(summary({ rows: 0, kept: 0 })).title).toBe("No usable rows");
  });
});

describe("the scoring command the page shows", () => {
  it("fills in the names it knows and marks every folder as a placeholder", () => {
    const parts = scoringCommand(["table.manifest.jsonl"], "/release/release.json");
    expect(commandText(parts)).toBe(
      "cardvision evaluate hover --browser-events DOWNLOADS-FOLDER/browser-events.jsonl \\\n" +
        "    --manifest LABELS-FOLDER/table.manifest.jsonl \\\n" +
        "    --release RELEASE-FOLDER/release.json",
    );
    expect(parts.filter((part) => part.placeholder).map((part) => part.text)).toEqual([PLACEHOLDERS.downloads, PLACEHOLDERS.labels, PLACEHOLDERS.release]);
    // No path that the page cannot know is given as if it were known.
    expect(parts.filter((part) => !part.placeholder).every((part) => !/[A-Z]{3,}-FOLDER/.test(part.text))).toBe(true);
  });

  it("names every manifest, and the release document by its own name", () => {
    const text = commandText(scoringCommand(["a.manifest.jsonl", "b.manifest.jsonl"], "http://127.0.0.1:9000/models/my%20game.json?x=1"));
    expect(text).toContain("--manifest LABELS-FOLDER/a.manifest.jsonl LABELS-FOLDER/b.manifest.jsonl \\\n");
    expect(text.endsWith('--release "RELEASE-FOLDER/my game.json"')).toBe(true);
    expect(commandText(scoringCommand(["a.jsonl"], "/release/")).endsWith("--release RELEASE-FOLDER/release.json")).toBe(true);
  });

  it("names the role the run scored, unless it is the one the command scores by itself", () => {
    const scored = "cardvision evaluate hover --browser-events DOWNLOADS-FOLDER/browser-events.jsonl \\\n" +
      "    --manifest LABELS-FOLDER/table.manifest.jsonl \\\n" +
      "    --release RELEASE-FOLDER/release.json";
    // Python scores the evaluation rows when it is given no role, and refuses an events file of another role.
    expect(commandText(scoringCommand(["table.manifest.jsonl"], "/release/release.json", "evaluation"))).toBe(scored);
    expect(commandText(scoringCommand(["table.manifest.jsonl"], "/release/release.json"))).toBe(scored);
    expect(commandText(scoringCommand(["table.manifest.jsonl"], "/release/release.json", "train"))).toBe(`${scored} \\\n    --role train`);
    expect(commandText(scoringCommand(["table.manifest.jsonl"], "/release/release.json", "extra"))).toBe(`${scored} \\\n    --role extra`);
    expect(DEFAULT_ROLE).toBe("evaluation");
    // The role is words to copy as they are, not a placeholder.
    const parts = scoringCommand(["table.manifest.jsonl"], "/release/release.json", "train");
    expect(parts[parts.length - 1]).toEqual({ text: " \\\n    --role train", placeholder: false });
    expect(parts.filter((part) => part.placeholder).map((part) => part.text)).toEqual([PLACEHOLDERS.downloads, PLACEHOLDERS.labels, PLACEHOLDERS.release]);
  });

  it("quotes a name a shell would split or expand", () => {
    const text = commandText(scoringCommand(["my labels.jsonl", 'odd"$name.jsonl'], "/release/release.json"));
    expect(text).toContain('--manifest "LABELS-FOLDER/my labels.jsonl" "LABELS-FOLDER/odd\\"\\$name.jsonl"');
  });
});
