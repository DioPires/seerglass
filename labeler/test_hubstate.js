// SPDX-License-Identifier: Apache-2.0
// Tests for hubstate.js: every sentence of the start screen and the project screen, state by
// state, from facts shaped as the server sends them (project.py; test_project.py feeds real
// ones through the same functions).
// Run: node test_hubstate.js (also run by test_project.py)
"use strict";
const assert = require("assert");
const H = require("./hubstate.js");

// ---- facts, as the server sends them ----------------------------------------------------------
const place = (path, o) => Object.assign({ path, source: "beside", state: "ok", usual: true }, o || {});
const pass = (id, o) => Object.assign({ id, allowed: true, reason: null, sets: ["table-a", "table-b"], frames: 10, done: 0, action: "start",
  button: true, running: false, command: { text: `python3 labeler/serve.py --mode ${id} --role training --sample toy/sample.json`, parts: [] } }, o || {});
const no = (id, reason, o) => pass(id, Object.assign({ allowed: false, reason, action: null }, o || {}));
const exp = (kind, o) => Object.assign({ included: 0, recordings: 0, leftOut: 10, out: kind === "coco" ? "/work/toy/boxes.json" : "/work/toy/manifest/labels.manifest.jsonl", outExists: false,
  text: kind === "coco" ? "python3 labeler/export.py coco --labels-dir toy/labels --out toy/boxes.json --catalog toy/catalog.json"
    : 'python3 labeler/export.py manifest --labels-dir toy/labels --sample toy/sample.json --credit "YOUR NAME" --out toy/manifest/labels.manifest.jsonl', parts: [] },
kind === "manifest" ? { untagged: [], unaddressed: [], unanswered: 0 } : {}, o || {});

// A toy project with nothing labeled, its purpose chosen: Boxes can be started.
function report(o) {
  const p = {
    ok: true, sample: "/work/toy/sample.json", folder: "/work/toy", name: "toy", recordings: 2,
    frames: { listed: 10, onDisk: 10, missing: 0, firstMissing: null, thumb: { asset: "table-a", name: "f00001.png" } },
    sets: [{ setId: "table-a", foreign: false }, { setId: "table-b", foreign: false }],
    locations: { labels: place("/work/toy/labels", { state: "will-create" }), frames: place("/work/toy/frames", { ownFolders: 2 }),
      catalog: place("/work/toy/catalog.json", { cards: 12 }), art: place("/work/toy/art"), decklists: place("/work/toy/decklists", { files: 2 }),
      prefill: place("/work/toy/prefill.jsonl") },
    role: { value: "training", stored: {}, fixed: false, choices: ["training", "development", "test"], refused: null, source: "asked" },
    passes: { box: pass("box"), identify: no("identify", "nothing-to-name", { sets: [], frames: 0, cards: 0, left: 0, boxes: 0 }),
      corner: pass("corner"), training: pass("training"), point: pass("point"), center: no("center", "not-for-training"),
      verify: no("verify", "not-for-training", { sets: [], frames: 0, faces: 0, named: 0 }) },
    modelBoxes: { offered: true, on: true, file: "prefill.jsonl" },
    export: { kind: "coco", coco: exp("coco"), manifest: exp("manifest"), damaged: [] },
    next: { pass: "box", startable: true }, flow: "box",
    server: { configured: false, sameProject: false, mode: null, role: null },
    restart: "python3 labeler/serve.py",
  };
  return Object.assign(p, o || {});
}
const undecided = () => {
  const p = report({ role: { value: null, stored: {}, fixed: false, choices: ["training", "development", "test"], refused: null, source: null }, next: { pass: null, startable: false } });
  for (const id of Object.keys(p.passes)) p.passes[id] = no(id, "no-role", p.passes[id].id === "identify" ? { boxes: 0, cards: 0, left: 0 } : {});
  return p;
};
// Boxes done on both recordings, 28 cards waiting for a name.
const boxed = (o) => report(Object.assign({
  role: { value: "training", stored: { training: ["table-a", "table-b"] }, fixed: true, choices: ["training"], refused: null, source: "labels" },
  next: { pass: "identify", startable: true }, export: { kind: "coco", coco: exp("coco", { included: 10, recordings: 2, leftOut: 0 }), manifest: exp("manifest", { included: 10, recordings: 2, leftOut: 0 }), damaged: [] } }, o || {}));
const withPasses = (p, passes) => { Object.assign(p.passes, passes); return p; };
const boxedPasses = { box: pass("box", { done: 10, action: "review" }), identify: pass("identify", { boxes: 34, cards: 28, left: 28, action: "start" }) };
const running = (mode) => ({ configured: true, sameProject: true, mode, role: "training" });
// labels for measuring a model: Development, nothing labeled yet
const measuring = (o) => {
  const p = report(Object.assign({ role: { value: "development", stored: {}, fixed: false, choices: ["training", "development", "test"], refused: null, source: "asked" },
    next: { pass: "center", startable: true }, flow: "center" }, o || {}));
  p.export.kind = "manifest";
  Object.assign(p.passes, { center: pass("center"), verify: no("verify", "no-centers", { sets: [], frames: 0, faces: 0, named: 0 }), training: no("training", "training-only") });
  return p;
};

const hub = (o) => Object.assign({ startDir: "/work", run: "0123456789abcdef", recent: [], resume: null, found: [], search: { cut: false, depth: 3 }, toy: { path: "/work/toy", state: "ok" },
  restart: "python3 labeler/serve.py", tools: { sample: "labeler/make_sample.py", python: "python3" },
  server: { configured: false, mode: null, role: null, project: null, name: null } }, o || {});
const found = (name) => ({ sample: `/work/${name}/sample.json`, folder: `/work/${name}`, name, recordings: 2, frames: 10 });

// ---- the start screen: three paths, the one that fits what was found first ---------------------
{
  const v = H.start(hub());
  assert.strictEqual(v.title, "Card labeler");
  assert.strictEqual(v.lead, "Label trading cards in frames of video: where each card is, which way up it lies and which card it is. Everything stays on this computer: the tool reads and writes files in folders you choose, and opens no connection.");
  assert.deepStrictEqual(v.paths.map((x) => [x.id, x.title, x.fits]), [["frames", "Label my frames", false], ["toy", "Learn on a toy game", true], ["video", "I have a video, not frames", false]]);
  assert.deepStrictEqual(v.paths.map((x) => x.promise), [
    "You have frames of play and a sample that lists them: open it and label, pass by pass.",
    "A made-up game drawn for you, to try every step in a few minutes. Nothing to prepare.",
    "Turn the video into frames and a sample first, with two commands, then come back here."]);
  assert.strictEqual(v.guide, "Nothing to label was found here. New to the tool? {{Enter}} makes the toy game. Have frames elsewhere: Label my frames. Only a video: the third path.");
  assert.deepStrictEqual([v.open, v.primary, v.resume, v.running, v.before, v.found], ["toy", "toy", null, null, [], []]);
  assert.strictEqual(v.foundNote, "No sample was found in /work or the 3 levels of folders below it. A sample is a JSON file that lists the frames to label, with “sample” in its name.");
  assert.deepStrictEqual(v.toy, { text: "A made-up card game, drawn by this tool, with a pretend model's boxes, so that you can try every step. Nothing is downloaded.",
    path: "/work/toy", where: "It will be written to a new folder: /work/toy", button: "Make the toy game and open it", disabled: false, missing: null });
  // nothing to label: what is missing, the one step back, and the guide
  assert.deepStrictEqual([v.none.text, v.none.step, v.none.guide.label], ["Missing: a sample, the list of frames to label.", "Make one from folders of frames, then open it here:", "Guide: The sample"]);
}
// the toy example cannot be made: what is missing, and the way back
{
  let v = H.start(hub({ toy: { path: "/work/toy", state: "not-writable" } }));
  assert.deepStrictEqual([v.toy.disabled, v.primary, v.open], [true, null, "toy"]);
  assert.strictEqual(v.toy.where, "It cannot be made here: /work cannot be written.");
  assert.deepStrictEqual([v.toy.missing.text, v.toy.missing.step], ["Missing: a folder this tool may write to.", "Stop the tool (Ctrl+C in its terminal) and start it in a folder you can write to."]);
  assert.strictEqual(v.guide, "Nothing to label was found here. Open a sample by its path under Label my frames, or see the third path for a video.");
  v = H.start(hub({ toy: { path: null, state: "none-free" } }));
  assert.strictEqual(v.toy.where, "It cannot be made here: /work already holds a great many toy folders.");
  assert.strictEqual(v.toy.missing.step, "Open one of the toy folders under “Label my frames”, or delete some of them.");
}
// ---- one sample found is the primary action; two are a choice -----------------------------------
{
  let v = H.start(hub({ found: [found("toy")] }));
  assert.strictEqual(v.guide, "Label my frames fits: one sample was found here. {{Enter}} opens it and shows its passes.");
  assert.deepStrictEqual([v.open, v.primary, v.foundNote, v.none], ["frames", "found", "", null]);
  assert.deepStrictEqual(v.paths.map((x) => x.fits), [true, false, false]);
  assert.deepStrictEqual(v.found, [{ sample: "/work/toy/sample.json", name: "toy", folder: "/work/toy", text: "2 recordings · 10 frames" }]);
  v = H.start(hub({ found: [found("toy"), Object.assign(found("club"), { recordings: 1, frames: 1 })] }));
  assert.strictEqual(v.guide, "Label my frames fits: choose one of the samples below.");
  assert.strictEqual(v.primary, null, "no one of them is the obvious one");
  assert.strictEqual(v.found[1].text, "1 recording · 1 frame");
  v = H.start(hub({ found: [found("toy")], search: { cut: true, depth: 3 } }));
  assert.strictEqual(v.foundNote, "The search stopped early because this folder holds a great many files, so there may be more samples: open one by its path below.");
}
// ---- the start screen: resume -----------------------------------------------------------------
{
  let v = H.start(hub({ resume: withPasses(boxed(), boxedPasses), recent: [found("toy")], found: [found("toy")] }));
  assert.strictEqual(v.primary, "resume");
  assert.deepStrictEqual(v.resume, { sample: "/work/toy/sample.json", name: "toy", folder: "/work/toy", text: "2 recordings · 10 frames · Training",
    next: "Next: Names. 28 cards left to name.", primary: "Start Names on toy", pass: "identify", does: "starts Names on toy", switches: false, secondary: "Show all passes of toy" });
  assert.strictEqual(v.guide, "Label my frames fits: carry on where you stopped. {{Enter}} starts Names on toy.");
  assert.deepStrictEqual([v.found, v.before, v.foundNote], [[], [], "No other sample was found in /work."], "the project to resume is listed once");
  const part = withPasses(report({ role: boxed().role }), { box: pass("box", { done: 4, action: "continue" }) });
  v = H.start(hub({ resume: part, recent: [found("toy")] }));
  assert.deepStrictEqual([v.resume.primary, v.resume.next], ["Continue with Boxes on toy", "Next: Boxes. 4 of 10 frames."]);
  assert.strictEqual(v.guide, "Label my frames fits: carry on where you stopped. {{Enter}} goes on with Boxes on toy.");
  v = H.start(hub({ resume: undecided(), recent: [found("toy")] }));
  assert.deepStrictEqual([v.resume.primary, v.resume.pass, v.resume.secondary, v.resume.next, v.resume.text],
    ["Open toy", null, null, "Next: say what these labels will be used for.", "2 recordings · 10 frames"]);
  assert.strictEqual(v.guide, "Label my frames fits: {{Enter}} opens toy, the project you opened last.");
  v = H.start(hub({ resume: boxed({ next: { pass: "export", startable: false } }), recent: [found("toy")] }));
  assert.deepStrictEqual([v.resume.primary, v.resume.next], ["Open toy", "Every frame is done and nothing is left to name. Next: write the boxes to a file."]);
  v = H.start(hub({ resume: report({ next: { pass: null, startable: false } }), recent: [found("toy")] }));
  assert.strictEqual(v.resume.next, "Nothing can be started right now: its screen says why.");
  const census = withPasses(report({ next: { pass: "verify", startable: true } }), { verify: pass("verify", { faces: 24, named: 3, action: "continue" }) });
  v = H.start(hub({ resume: census, recent: [found("toy")] }));
  assert.deepStrictEqual([v.resume.primary, v.resume.next], ["Continue with Check names on toy", "Next: Check names. 3 of 24 cards named."]);
  const gone = { sample: "/work/old/sample.json", folder: "/work/old", name: "old", problem: "missing" };
  v = H.start(hub({ resume: withPasses(boxed(), boxedPasses), recent: [gone, found("toy"), found("club")], found: [found("club"), found("toy"), found("new")] }));
  assert.deepStrictEqual(v.before, [
    { sample: "/work/old/sample.json", name: "old", gone: true, text: "This sample is not there any more: /work/old/sample.json. It was moved or deleted. Open it from where it is now, or remove it from this list." },
    { sample: "/work/club/sample.json", name: "club", folder: "/work/club", text: "2 recordings · 10 frames" }]);
  assert.deepStrictEqual(v.found.map((e) => e.name), ["new"]);
  assert.strictEqual(H.gone({ sample: "/work/old/sample.json", problem: "not-json" }),
    "This sample cannot be read any more: /work/old/sample.json. /work/old/sample.json is not a sample: it is not a JSON file.");
  v = H.start(hub({ recent: [gone], found: [found("toy")] }));
  assert.deepStrictEqual([v.primary, v.open], ["found", "frames"]);
}
// ---- the start screen on a tool that runs a pass -----------------------------------------------
{
  const server = { configured: true, mode: "box", role: "training", project: "/work/toy/sample.json", name: "toy" };
  let v = H.start(hub({ server, resume: boxed(), recent: [found("toy")], found: [found("toy"), found("club")] }));
  assert.deepStrictEqual(v.running, { text: "The tool is running the Boxes pass on toy, for Training.", primary: "Go to the Boxes pass", secondary: "Show the passes of toy" });
  assert.deepStrictEqual([v.primary, v.resume, v.found.map((e) => e.name), v.open], ["running", null, ["club"], "frames"]);
  assert.strictEqual(v.guide, "The tool is running the Boxes pass on toy, for Training. {{Enter}} goes to it.");
  // another project was opened last: its next pass is one button away, the tool switches to it
  v = H.start(hub({ server: Object.assign({}, server, { project: "/work/club/sample.json", name: "club" }), resume: withPasses(boxed(), boxedPasses), recent: [found("toy")] }));
  assert.deepStrictEqual([v.resume.primary, v.resume.pass, v.resume.switches, v.primary], ["Start Names on toy", "identify", true, "running"]);
}
// ---- "I have a video, not frames" ---------------------------------------------------------------
{
  const v = H.video(hub({ tools: { sample: "/opt/labeler/make_sample.py", python: "python3 -B" } }));
  assert.strictEqual(v.summary, "I have a video, not frames");
  assert.deepStrictEqual(v.commands.map((c) => c.text), [
    "cardvision infer my-video.mp4 --release release.json --every 2 --frame-format png --frames work/frames/my-video --prefill-out work/prefill.jsonl",
    "python3 -B /opt/labeler/make_sample.py --out work/sample.json --asset my-video=work/frames/my-video --all --times work/prefill.jsonl --local"]);
  assert.deepStrictEqual(v.commands.map((c) => c.parts.filter((x) => x.fill).map((x) => x.t)), [
    ["my-video.mp4", "release.json", "work/frames/my-video", "work/prefill.jsonl"], ["work/sample.json", "my-video=work/frames/my-video", "work/prefill.jsonl"]], "the parts to fill in are marked");
  assert.ok(v.needs.startsWith("Needed first:") && v.text.startsWith("The tool labels frames and not videos.") && v.after.includes("Then open work/sample.json under Label my frames."));
  assert.strictEqual(v.guide.label, "Guide: From a video to frames");
}
// ---- a path that is no project: what it is, and what to do -------------------------------------
{
  const ctx = { startDir: "/work", sampleTool: "labeler/make_sample.py", python: "python3" };
  const make = "python3 labeler/make_sample.py --out sample.json --frames-root FOLDER_OF_FRAME_FOLDERS --n 8";
  const said = (problem, path) => { const r = H.problem({ problem, path }, ctx); return `${r.text} ${r.fix}`.trim(); };
  assert.strictEqual(said("empty", null), "Type the path of a sample file, or of a folder that holds a sample.json.");
  assert.strictEqual(said("missing", "/work/toyy"), "Nothing is at /work/toyy. Check the spelling. A path that does not start with / is read from /work.");
  assert.strictEqual(said("bad-path", "a\u0000b"), "That is not a path this computer understands: a\u0000b Give the path as your file manager or terminal shows it.");
  assert.strictEqual(said("no-sample-in-folder", "/work/frames"), `There is no sample.json in /work/frames. If the sample has another name, give the file itself. To make a sample from folders of frames: ${make}`);
  assert.strictEqual(said("unreadable", "/work/s.json"), "/work/s.json cannot be read. Check that you are allowed to read the file.");
  assert.strictEqual(said("not-json", "/work/notes.txt"), `/work/notes.txt is not a sample: it is not a JSON file. A sample is a JSON file that lists the frames to label. To make one from folders of frames: ${make}`);
  assert.strictEqual(said("not-sample", "/work/toy/catalog.json"),
    `/work/toy/catalog.json is a JSON file but not a sample: it has no list of recordings ("assets"). A sample looks like {"assets": [{"assetId": "table-a", "frames": [{"t": 0, "name": "f00001.png"}]}]}. To make one from folders of frames: ${make}`);
  assert.strictEqual(said("none", null), "No project is open. Choose one on the start screen.");
  assert.strictEqual(said("something-new", "/work/x"), "/work/x could not be opened as a sample.");
  for (const code of ["empty", "missing", "bad-path", "no-sample-in-folder", "unreadable", "not-json", "not-sample", "none"]) assert.ok(H.problem({ problem: code, path: "/x" }).text, code);
}
// ---- the project screen: the header, and the first state ---------------------------------------
{
  let v = H.project(undecided());
  assert.deepStrictEqual([v.title, v.folder, v.summary], ["toy", "/work/toy", "2 recordings · 10 frames"]);
  assert.strictEqual(v.labels, "Labels are written to /work/toy/labels (the folder is made at the first save)");
  assert.strictEqual(v.guide, "Say what these labels will be used for. The passes below can be started after that.");
  assert.deepStrictEqual([v.primary, v.warnings, v.modelBoxes], [null, [], null]);
  assert.deepStrictEqual(v.rows.map((r) => [r.name, r.status, r.kind, r.reason]),
    [["Boxes", "0 of 10 frames", "disabled", "Say what the labels are for first (above)."], ["Names", "No boxes yet", "disabled", "Say what the labels are for first (above)."]]);
  assert.strictEqual(v.purpose.question, "What will these labels be used for?");
  assert.deepStrictEqual(v.purpose.options.map((o) => [o.label, o.checked, o.disabled]),
    [["Training: teach the model", false, false], ["Development: compare and tune", false, false], ["Test: final held-out check", false, false]]);
  assert.strictEqual(v.purpose.note, "Choose one. It is stored with the first label and decides what the tool may show you while you label.");

  v = H.project(report());
  assert.strictEqual(v.guide, "Next: Boxes. {{Enter}} starts it.");
  assert.strictEqual(v.primary, "box");
  assert.deepStrictEqual(v.rows[0], { id: "box", name: "Boxes", meaning: "Draw a box around every card on each frame.", status: "0 of 10 frames", button: "Start", kind: "start", reason: "", note: "" });
  assert.deepStrictEqual([v.rows[1].kind, v.rows[1].reason, v.rows[1].meaning], ["disabled", "Nothing to name yet: draw boxes first.", "Say which card each box shows."]);
  assert.deepStrictEqual([v.why, v.protocols, v.measuringGuide], ["", [], null], "Training: boxes for training, no measuring protocol");
  assert.strictEqual(v.purpose.note, "You can change this until the first label is saved.");
  assert.deepStrictEqual(v.modelBoxes, { label: "Start from the model's boxes", checked: true, disabled: false, note: "prefill.jsonl, found beside the sample. You accept, correct or remove each box." });
}
// ---- the pass list follows the purpose -----------------------------------------------------------
{
  // Development and Test: Centers, then Check names, then the manifest; Points the other protocol; Boxes under other ways
  let v = H.project(measuring());
  assert.deepStrictEqual(v.rows.map((r) => r.name), ["Centers", "Check names"]);
  assert.deepStrictEqual(v.protocols.map((r) => r.name), ["Points"]);
  assert.deepStrictEqual(v.others.map((r) => r.name), ["Boxes", "Names", "Corners"]);
  assert.strictEqual(v.why, "Centers, then Check names: fast (one click a card, then one name each), and the way this project's own figures were measured, as a census of card centers together with seeded random points.");
  assert.strictEqual(v.protocolsWhy, "Points: random points per frame, each answered. Together with a census they give the off-card positions a measurement needs.");
  assert.strictEqual(v.measuringGuide.label, "Guide: Labels for measuring a model");
  assert.deepStrictEqual([v.primary, v.guide], ["center", "Next: Centers. {{Enter}} starts it."]);
  assert.deepStrictEqual([v.rows[0].meaning, v.rows[1].meaning], ["Click the center of every card on each frame.", "Name each clicked center, with no model's proposal shown."]);
  assert.strictEqual(v.rows[1].reason, "Nothing to check yet: click the centers first, in the Centers pass.");
  assert.strictEqual(v.exportRow.meaning, "Write the labels to a manifest, the file cardvision evaluate reads to measure a model.");
  // the same for Test
  const test = measuring(); test.role.value = "test";
  assert.deepStrictEqual(H.lists(test).main, ["center", "verify"]);
  // labels already made another way: the list goes on that way
  assert.deepStrictEqual(H.lists(measuring({ flow: "box" })).main, ["box", "identify"]);
  assert.deepStrictEqual(H.lists(measuring({ flow: "box" })).protocols, ["center", "verify", "point"]);
  assert.deepStrictEqual(H.lists(measuring({ flow: "point" })).main, ["point"]);
  assert.deepStrictEqual(H.lists(report()), { main: ["box", "identify"], protocols: [], others: ["corner", "training", "point", "center", "verify"], exportKind: "coco", why: "", protocolsWhy: "" });
  // Check names and Points: what they run with, in one line each
  const named = withPasses(measuring(), { verify: pass("verify", { faces: 24, named: 0 }) });
  v = H.project(named);
  assert.strictEqual(v.rows[1].note, "Every center is named from the frame, with no model's proposal shown. To show the proposals of models beside each center, start this pass with flags (the guide says how).");
  assert.strictEqual(v.protocols[0].note, "12 points a frame, placed by seed 1, fixed at the first save. Other numbers: the command for scripts, with --k and --seed.");
}
// ---- the model's boxes: Training only, ticked when a file is found -----------------------------
{
  let p = report({ modelBoxes: { offered: false, on: false, file: "prefill.jsonl" } });
  p.locations.prefill = place("/work/toy/prefill.jsonl", { state: "absent" });
  assert.deepStrictEqual(H.modelBoxes(p), { label: "Start from the model's boxes", checked: false, disabled: true, note: "No suggestions file was found (looked for /work/toy/prefill.jsonl)." });
  p = report({ modelBoxes: { offered: true, on: false, file: "boxes.jsonl" } });
  p.locations.prefill = place("/models/boxes.jsonl", { usual: false, source: "page" });
  assert.deepStrictEqual(H.modelBoxes(p), { label: "Start from the model's boxes", checked: false, disabled: false, note: "boxes.jsonl, as given. You accept, correct or remove each box." });
  assert.strictEqual(H.modelBoxes(report({ role: Object.assign(report().role, { value: "test" }) })), null, "never offered for Development or Test");
  assert.strictEqual(H.modelBoxes(report({ role: Object.assign(report().role, { value: "development" }) })), null);
  p = report({ server: running("box"), modelBoxes: { offered: false, on: false, file: null } });
  p.locations.prefill = { path: null, source: "flag", state: "off", usual: false };
  assert.deepStrictEqual([H.modelBoxes(p).disabled, H.modelBoxes(p).note], [true, "The running pass was started without a suggestions file."]);
  assert.strictEqual(H.modelBoxes(report({ server: running("box") })).disabled, true, "a running pass keeps what it was started with");
}
// ---- what the labels are for: read from the labels, asked otherwise ----------------------------
{
  const fixed = { value: "training", stored: { training: ["table-a"] }, fixed: true, choices: ["training"], refused: null, source: "labels" };
  let q = H.purpose(report({ role: fixed }));
  assert.strictEqual(q.note, "Read from the labels in this folder: they were made for Training. A label set keeps its purpose from its first save, so it cannot be changed here. To label these frames for another purpose, give another labels folder under Change locations.");
  assert.deepStrictEqual(q.options.map((o) => [o.checked, o.disabled]), [[true, false], [false, true], [false, true]]);
  assert.deepStrictEqual(q.settled, { label: "Training: teach the model", help: "The labels a model learns from. A model's suggestions may be shown while you label, since they only save time here." });
  assert.strictEqual(H.purpose(report()).settled, null);
  q = H.purpose(report({ role: Object.assign({}, fixed, { refused: "test" }) }));
  assert.strictEqual(q.note, "You asked for Test, but the labels in this folder were made for Training, and a label set keeps its purpose from its first save. The passes below go on with the Training labels. To label these frames for Test, give another labels folder under Change locations.");
  q = H.purpose(report({ role: { value: null, stored: { training: ["table-a"], development: ["table-b", "table-c"] }, fixed: false, choices: ["training", "development"], refused: null, source: null } }));
  assert.strictEqual(q.note, "This folder holds labels made for Training (table-a) and Development (table-b, table-c). Choose which to work on; the others are left as they are.");
  assert.deepStrictEqual(q.options.map((o) => o.disabled), [false, false, true]);
  q = H.purpose(report({ server: running("box") }));
  assert.deepStrictEqual(q.options.map((o) => o.disabled), [false, true, true]);
  assert.strictEqual(q.settled.label, "Training: teach the model");
}
// ---- the status of each pass, from its counts ---------------------------------------------------
{
  const st = (id, o, base) => H.status(withPasses(base || report(), { [id]: pass(id, o) }), id);
  assert.strictEqual(st("box", { done: 0 }), "0 of 10 frames");
  assert.strictEqual(st("box", { done: 4 }), "4 of 10 frames");
  assert.strictEqual(st("box", { done: 10 }), "Done · 10 of 10 frames");
  assert.strictEqual(st("box", { done: 1, frames: 1 }), "Done · 1 of 1 frame");
  assert.strictEqual(st("box", { done: 2, frames: 5, sets: ["table-b"] }), "2 of 5 frames · 1 of 2 recordings");
  assert.strictEqual(st("box", { frames: 0, sets: [] }), "No frames");
  assert.strictEqual(st("identify", { boxes: 0, cards: 0, left: 0 }), "No boxes yet");
  assert.strictEqual(st("identify", { boxes: 34, cards: 28, left: 28 }), "28 cards left to name");
  assert.strictEqual(st("identify", { boxes: 34, cards: 28, left: 1 }), "1 card left to name · 27 named");
  assert.strictEqual(st("identify", { boxes: 34, cards: 28, left: 0 }), "Every box so far has its answer", "Boxes is not finished yet");
  assert.strictEqual(st("identify", { boxes: 34, cards: 28, left: 0 }, withPasses(report(), { box: pass("box", { done: 10 }) })), "Done · every box has its answer");
  assert.strictEqual(st("verify", { faces: 0, named: 0 }), "No centers yet");
  assert.strictEqual(st("verify", { faces: 24, named: 3 }), "3 of 24 cards named");
  assert.strictEqual(st("verify", { faces: 24, named: 24 }), "Done · 24 cards named");
  assert.strictEqual(st("center", { done: 5 }), "5 of 10 frames");
}
// ---- one button for each pass: Start, Continue, Review, Open; or disabled, with the reason ------
{
  const row = (id, o, base) => H.passRow(withPasses(base || report(), { [id]: pass(id, o) }), id);
  assert.deepStrictEqual(["start", "continue", "review"].map((action) => row("box", { action }).button), ["Start", "Continue", "Review"]);
  assert.deepStrictEqual([row("box", { running: true }, report({ server: running("box") })).button, row("box", { running: true }, report({ server: running("box") })).kind], ["Open", "open"]);
  assert.strictEqual(row("identify", { boxes: 3, cards: 3, left: 3 }, report({ server: running("box") })).kind, "switch", "on a running tool, another pass is a switch");
  const why = (id, reason, extra, base) => { const r = H.passRow(withPasses(base || report(), { [id]: no(id, reason, extra) }), id); assert.deepStrictEqual([r.kind, r.button], ["disabled", null]); return r.reason; };
  assert.strictEqual(why("box", "no-role"), "Say what the labels are for first (above).");
  assert.strictEqual(why("training", "training-only"), "Only for Training labels: it starts from a model's corners and names.");
  assert.strictEqual(why("center", "not-for-training"), "Only for Development or Test labels: they are what a model is measured on.");
  assert.strictEqual(why("corner", "refused"), "Not offered for this purpose.");
  assert.strictEqual(why("box", "labels-not-writable"), "The labels folder cannot be written (see the top of this page).");
  assert.strictEqual(why("identify", "no-catalog", { boxes: 3, cards: 3, left: 3 }), "Needs a catalog: a JSON list of the game's cards. None is at /work/toy/catalog.json.");
  let p = report();
  p.locations.catalog = { path: null, source: "flag", state: "off", usual: false };
  assert.strictEqual(why("identify", "no-catalog", { boxes: 3, cards: 3, left: 3 }, p), "Needs a catalog: a JSON list of the game's cards. The running pass was started without one.");
  p = report();
  p.locations.catalog = place("/work/toy/catalog.json", { state: "unreadable", problem: "it is not JSON" });
  assert.strictEqual(why("identify", "bad-catalog", { boxes: 3, cards: 3, left: 3 }, p), "Needs a catalog, and this one cannot be used: it is not JSON (/work/toy/catalog.json).");
  assert.strictEqual(why("identify", "nothing-to-name", { boxes: 0 }), "Nothing to name yet: draw boxes first.");
  assert.strictEqual(why("verify", "no-centers", { faces: 0, named: 0 }), "Nothing to check yet: click the centers first, in the Centers pass.");
  p = report();
  p.locations.prefill = place("/work/toy/prefill.jsonl", { state: "absent" });
  assert.strictEqual(why("training", "no-suggestions", {}, p), "Needs a model's suggestions file. None is at /work/toy/prefill.jsonl.");
  assert.strictEqual(why("box", "other-way", { have: ["corner"], sets: [] }),
    "These recordings already have Corners labels, and a recording keeps one way of labeling in a labels folder. To label them this way too, give another labels folder under Change locations.");
  assert.strictEqual(why("corner", "other-way", { have: ["box", "center"], sets: [] }, report({ recordings: 1 })),
    "This recording already has Boxes and Centers labels, and a recording keeps one way of labeling in a labels folder. To label it this way too, give another labels folder under Change locations.");
  assert.strictEqual(why("box", "no-frames", { frames: 0, sets: [] }), "This sample lists no frames.");
  assert.strictEqual(why("box", "frames-missing"), "The frames are not on disk (see the top of this page).");
  assert.deepStrictEqual(H.project(report()).others.map((r) => [r.name, r.meaning]), [
    ["Corners", "Mark the four corners of every card, for footage where a rectangle does not fit."],
    ["Corners from the model", "Check a model's corners and names instead of clicking them yourself."],
    ["Points", "Say what is under each of a fixed number of random points per frame."],
    ["Centers", "Click the center of every card on each frame."],
    ["Check names", "Name each clicked center, with no model's proposal shown."]]);
  assert.deepStrictEqual(H.scripts(report(), "box"), { lead: "For scripts, the same pass as one command:", text: "python3 labeler/serve.py --mode box --role training --sample toy/sample.json", parts: [] });
}
// ---- the running tool switches to another pass from the page: no new start ----------------------
{
  let p = withPasses(report({ server: running("box") }), { box: pass("box", { done: 4, action: "continue", running: true }) });
  let v = H.project(p);
  assert.strictEqual(v.guide, "The tool is running the Boxes pass. {{Enter}} opens it. Any other pass: its row's button switches the tool to it, here.");
  assert.deepStrictEqual([v.primary, v.placesLocked], ["box", true]);
  assert.deepStrictEqual(v.rows.map((r) => [r.kind, r.button]), [["open", "Open"], ["disabled", null]]);
  // Boxes runs and is done, Names is next: the primary action switches the tool to Names
  p = withPasses(boxed({ server: running("box") }), { box: pass("box", { done: 10, action: "review", running: true }), identify: boxedPasses.identify });
  v = H.project(p);
  assert.strictEqual(v.guide, "Boxes is done. Next: Names. {{Enter}} switches the tool to it, here: no new start.");
  assert.deepStrictEqual(v.primary, "identify");
  assert.deepStrictEqual(v.rows.map((r) => [r.kind, r.button]), [["open", "Open"], ["switch", "Start"]]);
  // Names runs and is the next pass: it opens
  const named = withPasses(boxed({ server: running("identify"), next: { pass: "identify", startable: true } }), { box: pass("box", { done: 10, action: "review" }),
    identify: pass("identify", { boxes: 34, cards: 28, left: 5, action: "continue", running: true }) });
  assert.strictEqual(H.project(named).primary, "identify");
  // Names runs and is done: what is left is the export, written from the page
  const all = withPasses(boxed({ server: running("identify"), next: { pass: "export", startable: false } }), { box: pass("box", { done: 10, action: "review" }),
    identify: pass("identify", { boxes: 34, cards: 28, left: 0, action: "review", running: true }) });
  assert.deepStrictEqual([H.project(all).primary, H.project(all).guide], ["export", "Every frame is done and nothing is left to name. Next: write boxes.json. {{Enter}} writes it."]);
  const stuck = withPasses(boxed({ server: running("box"), next: { pass: null, startable: false } }), { box: pass("box", { done: 10, action: "review", running: true }) });
  assert.strictEqual(H.project(stuck).primary, "box");
  // the tool runs a pass of another project: the switch is the same, from here
  const other = H.project(withPasses(boxed({ server: { configured: true, sameProject: false, mode: "box", role: "test" } }), boxedPasses));
  assert.strictEqual(other.guide, "The tool is running a pass of another project. {{Enter}} switches it to Names of toy.");
  assert.deepStrictEqual([other.primary, other.rows.map((r) => r.kind), other.placesLocked], ["identify", ["switch", "switch"], false]);
}
// ---- the instruction line of the other states --------------------------------------------------
{
  assert.strictEqual(H.project(withPasses(boxed(), boxedPasses)).guide, "Next: Names. {{Enter}} starts it.");
  assert.strictEqual(H.project(withPasses(report(), { box: pass("box", { done: 4, action: "continue" }) })).guide, "Next: Boxes. {{Enter}} goes on with it.");
  const done = H.project(boxed({ next: { pass: "export", startable: false } }));
  assert.deepStrictEqual([done.primary, done.guide], ["export", "Every frame is done and nothing is left to name. Next: write boxes.json. {{Enter}} writes it."]);
  assert.strictEqual(H.project(report({ next: { pass: null, startable: false } })).guide, "No pass can be started as things are: each row says what it needs.");
}
// ---- the export: what it writes, what it asks for, what it answered ---------------------------
{
  const ex = (o, form, kind) => { const p = report(); p.export.kind = kind || "coco"; Object.assign(p.export[p.export.kind], o); return H.exportRow(p, form); };
  let r = ex({});
  assert.deepStrictEqual([r.status, r.note, r.button, r.disabled, r.fields], ["Nothing to export yet", "The export writes the frames that are done, and none is yet.", "Write boxes.json", true, null]);
  assert.strictEqual(r.meaning, "Write the boxes to boxes.json, the file a model is trained on to find cards.");
  assert.deepStrictEqual(r.scripts, { lead: "For scripts, the same export as one command:", text: "python3 labeler/export.py coco --labels-dir toy/labels --out toy/boxes.json --catalog toy/catalog.json", parts: [] });
  r = ex({ included: 6, recordings: 2, leftOut: 4 });
  assert.deepStrictEqual([r.status, r.note, r.disabled], ["6 frames ready", "It writes 6 frames that are done, from 2 recordings, to /work/toy/boxes.json. 4 frames of this sample are not done and are left out.", false]);
  assert.strictEqual(ex({ included: 1, recordings: 1, leftOut: 0 }).note, "It writes 1 frame that is done, from 1 recording, to /work/toy/boxes.json. No frame of this sample is left out.");
  // what it answered
  r = ex({ included: 6, recordings: 2 }, { result: { ok: true, out: "/work/toy/boxes.json", frames: 6, boxes: 40, ignored: 2, notes: ["table-c: a center set holds no boxes, skipped"] } });
  assert.deepStrictEqual(r.result, { ok: true, text: "Wrote /work/toy/boxes.json: 6 frames, 40 boxes (2 ignored).", notes: ["table-c: a center set holds no boxes, skipped"], check: "" });
  r = ex({ included: 6 }, { result: { ok: false, problem: "exists", out: "/work/toy/boxes.json" } });
  assert.deepStrictEqual(r.confirm, { text: "/work/toy/boxes.json is there already, from an earlier export. Replace it with what the labels hold now?", yes: "Replace it", no: "Keep it" });
  assert.strictEqual(ex({ included: 6 }, { result: { ok: false, problem: "nothing" } }).result.text, "Nothing to write: no boxes are done yet.");
  // the manifest: it asks for a credit, and for the kind of footage when no recording says it
  r = ex({ included: 3, recordings: 1, untagged: ["table-a"] }, {}, "manifest");
  assert.deepStrictEqual([r.button, r.disabled, r.fields.credit.label, r.fields.sourceType.label], ["Write the manifest", "credit", "Credit: the name to show for these labels", "Kind of footage (table-a has none recorded)"]);
  assert.deepStrictEqual(r.fields.sourceType.options.map((o) => o.value), ["physical/webcam", "physical/tournament", "physical/phone", "physical", "digital"]);
  assert.strictEqual(ex({ included: 3, untagged: ["table-a"] }, { credit: "Me" }, "manifest").disabled, "source");
  assert.strictEqual(ex({ included: 3, untagged: ["table-a"] }, { credit: "Me", sourceType: "digital" }, "manifest").disabled, false);
  assert.strictEqual(ex({ included: 3 }, { credit: "Me" }, "manifest").fields.sourceType, null, "the recordings say it");
  // what it cannot do without: where each video is; a census not named in full is said
  r = ex({ included: 3, unaddressed: ["table-a", "table-b"], unanswered: 2 }, { credit: "Me" }, "manifest");
  assert.strictEqual(r.disabled, true);
  assert.deepStrictEqual(r.needs.map((x) => [x.text, x.guide && x.guide.label]), [
    ["Missing: where table-a and table-b are hosted. A manifest names videos, so sample.json must give each recording \"platform\" and \"videoId\" (for a recording of your own: \"platform\": \"local\").", "Guide: A recording of your own"],
    ["2 faces of the census have no checked name yet: written as unlabeled. Check names first to name them.", null]]);
  r = ex({ included: 3 }, { credit: "Me", result: { ok: true, out: "/m/labels.manifest.jsonl", frames: 3, boxes: 0, points: 12, ownRecording: true, notes: [] } }, "manifest");
  assert.deepStrictEqual([r.result.text, r.result.check], ["Wrote /m/labels.manifest.jsonl: 3 rows, 12 points.", "Check it: cardvision validate-manifest /m/labels.manifest.jsonl"]);
  // each refusal of export.py as one sentence that says what to fill in
  r = ex({ included: 3 }, { credit: "Me", result: { ok: false, problem: "refused", problems: [
    { code: "address", where: "table-a/f00001.png", text: "x" }, { code: "address", where: "table-a/f00002.png", text: "x" }, { code: "source-type", where: "table-b", text: "x" },
    { code: "credit", where: "", text: "x" }, { code: "same-frame", where: "table-a/f00003.png", text: "x" }, { code: "time-size", where: "table-b/f00001.png", text: "x" },
    { code: "card-id", where: "table-b/f00002.png", text: "table-b/f00002.png: box B001: card id 'A B' has a space" }, { code: "no-asset", where: "old", text: "x" }, { code: "new", where: "", text: "Something else." }] } }, "manifest");
  assert.deepStrictEqual([r.result.text, r.result.items], ["The manifest was not written:", [
    "The sample does not say where table-a is hosted. In sample.json, give that recording \"platform\" and \"videoId\"; for a recording of your own, \"platform\": \"local\" and a name of your choice (make_sample.py --local writes both).",
    "Choose the kind of footage of table-b under “Kind of footage”, or tag it under About this recording while you label.",
    "Fill in the credit: the name to show for these labels, 1 to 80 characters on one line.",
    "table-a/f00003.png: two sets label the same frame. Export them one at a time, with the command for scripts and --set.",
    "table-b/f00001.png: the frame needs its time in the video (\"t\", in seconds, in sample.json) and its size, which the tool records once the frame is labeled.",
    "table-b/f00002.png: a card id the manifest cannot hold (card id 'A B' has a space). Correct it in the catalog, then name that card again.",
    "old: this set labels a recording the sample does not list. Put that recording back in sample.json, or move the set's folder out of the labels folder.",
    "Something else."]], "one sentence per kind and recording");
  // the instruction line while the export waits for what it asks
  const wait = boxed({ next: { pass: "export", startable: false } });
  wait.export.kind = "manifest"; wait.role.value = "development";
  assert.strictEqual(H.project(wait, {}).guide, "Every frame is done. Next: the export. Fill in the credit under Export, then {{Enter}} writes the manifest.");
  assert.strictEqual(H.project(wait, { credit: "Me" }).primary, "export");
  wait.export.manifest.unaddressed = ["table-a"];
  assert.deepStrictEqual([H.project(wait, { credit: "Me" }).primary, H.project(wait, { credit: "Me" }).guide],
    [null, "Every frame is done. Next: the export, once sample.json says where each recording is hosted (the Export row says how)."]);
}
// ---- what is missing in the project's files, each with the one step back -------------------------
{
  const warn = (o, locations) => { const p = report(o); Object.assign(p.locations, locations || {}); return H.warnings(p).map((w) => [w.text, w.step, w.guide && w.guide.label]); };
  assert.deepStrictEqual(warn({}), []);
  assert.deepStrictEqual(warn({ recordings: 0, frames: { listed: 0, onDisk: 0, missing: 0, firstMissing: null, thumb: null } }),
    [["Missing: frames to label. This sample lists none.", "Make a sample from folders of frames with make_sample.py (on the start screen: I have a video, not frames), then open that one.", "Guide: The sample"]]);
  assert.deepStrictEqual(warn({ frames: { listed: 10, onDisk: 0, missing: 10, firstMissing: "/work/toy/frames/table-a/f00001.png", thumb: null } }),
    [["Missing: the frames. None of the 10 frames this sample lists is on disk; the first is looked for at /work/toy/frames/table-a/f00001.png.", "Put the frames there, or give the right frames folder under Change locations.", "Guide: The sample"]]);
  assert.deepStrictEqual(warn({ frames: { listed: 10, onDisk: 7, missing: 3, firstMissing: "/f.png", thumb: {} } })[0][0], "3 of 10 frames are not on disk, starting with /f.png. Those frames cannot be labeled.");
  assert.deepStrictEqual(warn({ frames: { listed: 10, onDisk: 9, missing: 1, firstMissing: "/f.png", thumb: {} } })[0][1], "Put it there, or give the right frames folder under Change locations.");
  assert.deepStrictEqual(warn({}, { labels: place("/locked/labels", { state: "not-writable" }) }),
    [["Missing: a labels folder this tool may write to. /locked/labels cannot be written, so no label could be saved and no pass can be started.", "Give another folder under Change locations, or make this one writable.", "Guide: Starting the tool"]]);
  assert.deepStrictEqual(warn({}, { labels: place("/work/toy/sample.json", { state: "not-folder" }) })[0][0], "The labels location is a file, not a folder: /work/toy/sample.json.");
  assert.deepStrictEqual(warn({}, { catalog: place("/work/toy/catalog.json", { state: "absent" }) }),
    [["Missing: a catalog, the list of the game's cards. Without it boxes and centers can be made, but no card can be named. It is looked for at /work/toy/catalog.json.",
      "Put the game's catalog.json beside the sample, or give its path under Change locations.", "Guide: The catalog"]]);
  assert.deepStrictEqual(warn({}, { catalog: place("/c.json", { state: "unreadable", problem: "it is not JSON" }) })[0][0], "The catalog cannot be used: it is not JSON (/c.json). Without it no card can be named.");
  assert.deepStrictEqual(H.warnings(undecided()).length, 0, "nothing is said of a catalog before the purpose is chosen");
  assert.deepStrictEqual(warn({ sets: [{ setId: "table-a", foreign: true }, { setId: "table-b", foreign: false }] })[0][0],
    "table-a: the label files in this folder were not written by this tool. They are left as they are and not counted here.");
  assert.strictEqual(H.project(report()).labels.includes("first save"), true);
  const there = report(); there.locations.labels = place("/work/toy/labels");
  assert.strictEqual(H.project(there).labels, "Labels are written to /work/toy/labels");
}
// ---- the guide: each link finds a phrase of README.md, which holds it once -----------------------
{
  const readme = require("fs").readFileSync(require("path").join(__dirname, "README.md"), "utf8").replace(/\*\*/g, "").replace(/`/g, "").replace(/\s+/g, " ");
  for (const [key, g] of Object.entries(H.GUIDE)) {
    assert.strictEqual(readme.split(g.find).length - 1, 1, `${key}: "${g.find}" is in README.md once`);
    assert.ok(!/[-,&]/.test(g.find), `${key}: a text fragment cannot hold a dash, a comma or an ampersand`);
    assert.strictEqual(H.guide(key).href, `/guide#:~:text=${encodeURIComponent(g.find)}`);
  }
  assert.strictEqual(H.guide("nothing"), null);
}
// ---- Change locations: each file, where it is looked for, whether it is there, what needs it ---
{
  let rows = H.places(report());
  assert.deepStrictEqual(rows.map((r) => [r.label, r.path, r.where, r.state, r.ok, r.need]), [
    ["Frames", "/work/toy/frames", "beside the sample", "Found.", true, "Needed to show the frames."],
    ["Labels", "/work/toy/labels", "beside the sample", "Not there yet: it is made at the first save.", true, "Every label is saved here."],
    ["Catalog", "/work/toy/catalog.json", "beside the sample", "Found: 12 cards.", true, "Needed to name cards: a JSON list of the game's cards."],
    ["Card images", "/work/toy/art", "beside the sample", "Found.", true, "Optional: without them the lists show card names only."],
    ["Deck lists", "/work/toy/decklists", "beside the sample", "Found: 2 deck lists.", true, "Optional: a deck list puts the cards in play first when you name."],
    ["Model suggestions", "/work/toy/prefill.jsonl", "beside the sample", "Found.", true, "Optional, Training only: a model's boxes to start from."]]);
  const p = report();
  Object.assign(p.locations, {
    frames: place("/work/toy/frames", { state: "unused" }), labels: place("/out/labels", { state: "not-writable", source: "page", usual: false }),
    catalog: place("/games/cards.json", { state: "unreadable", problem: "cards[3]: card id 'A B' must be 1 to 64 characters", source: "flag", usual: false }),
    art: place("https://cards.example/art/", { url: true, source: "remembered", usual: false }), decklists: place("/work/toy/decklists", { state: "absent" }),
    prefill: place("/models/boxes.jsonl", { state: "missing", source: "page", usual: false }) });
  rows = H.places(p);
  assert.deepStrictEqual(rows.map((r) => [r.state, r.ok, r.where]), [
    ["Not used: every recording names its own folder in the sample.", true, "beside the sample"],
    ["It cannot be written.", false, "changed here"],
    ["It cannot be used: cards[3]: card id 'A B' must be 1 to 64 characters.", false, "given when the tool was started"],
    ["An address: your browser loads the card images from it.", true, "remembered from last time"],
    ["Nothing is there.", true, "beside the sample"],
    ["Nothing is there.", false, "changed here"]]);
  assert.strictEqual(H.places(Object.assign({}, p, { locations: Object.assign({}, p.locations, { decklists: place("/work/toy/decklists", { state: "missing", source: "flag", usual: true }) }) }))[4].ok, true);
  assert.deepStrictEqual(rows.map((r) => r.found), [false, false, false, true, false, false]);
  Object.assign(p.locations, { frames: place("/work/toy/frames", { state: "absent" }), labels: place("/x", { state: "not-folder" }), catalog: { path: null, source: "flag", state: "off", usual: false } });
  assert.deepStrictEqual(H.places(p).slice(0, 3).map((r) => [r.state, r.ok]), [["Nothing is there.", false], ["This is a file, not a folder.", false], ["The running pass was started without one.", true]]);
  assert.strictEqual(H.project(report()).placesNote, "Each file is looked for beside the sample first. Give another path where yours is somewhere else; a path that does not start with / is read from the folder the tool was started in.");
}
// ---- words, not internal names, on both screens ------------------------------------------------
{
  const texts = [];
  const walk = (v) => { if (typeof v === "string") texts.push(v); else if (v && typeof v === "object") Object.values(v).forEach(walk); };
  const states = [undecided(), report(), withPasses(boxed(), boxedPasses), withPasses(boxed({ server: running("box") }), boxedPasses), measuring(), measuring({ flow: "point" })];
  for (const p of states) {
    const v = H.project(p, { credit: "" });
    const rowWords = (rs) => rs.map((r) => [r.name, r.meaning, r.status, r.button, r.reason, r.note]);
    walk([v.guide, v.labels, v.warnings.map((w) => [w.text, w.step]), v.places.map((r) => [r.label, r.state, r.need, r.where]), v.placesNote, v.purpose, v.modelBoxes,
      rowWords(v.rows), rowWords(v.protocols), rowWords(v.others), v.why, v.protocolsWhy,
      [v.exportRow.name, v.exportRow.meaning, v.exportRow.status, v.exportRow.note, v.exportRow.button, v.exportRow.needs.map((x) => x.text)]]);
  }
  const shown = (v) => [v.guide, v.lead, v.running, v.paths.map((x) => [x.title, x.promise]), v.resume && [v.resume.name, v.resume.text, v.resume.next, v.resume.primary, v.resume.secondary],
    v.before.map((e) => e.text), v.found.map((e) => e.text), v.foundNote, v.toy.text, v.toy.where, v.toy.button, v.none && [v.none.text, v.none.step]];
  walk(shown(H.start(hub({ resume: withPasses(boxed(), boxedPasses), recent: [found("toy")], found: [found("club")] }))));
  walk(shown(H.start(hub())));
  for (const t of texts) {
    assert.ok(!/\b(prefill|identify|verify|role|mode|box mode|meta\.json|carry-over|arms?|eligible|detector)\b/.test(t.replace(/--all-blind/g, "").replace(/\S+\.jsonl?/g, "")), t); // a file's own name is not the tool's wording
    assert.ok(!/\b(first|only) (tool|labeler)\b/i.test(t), t);
    assert.ok(!/stop the tool \(Ctrl\+C in its terminal\) and start it again(?! as it ran)/.test(t) || t.startsWith("The tool is running a pass with these locations"), `no restart hand-off: ${t}`);
  }
}

// ---- arriving at an address, and what is remembered ----------------------------------------------
{
  const none = hub();
  const own = hub({ server: { configured: false, mode: null, role: null, project: "/work/toy/sample.json", name: "toy" } });
  const at = (pathname, sample, facts) => H.arrival({ pathname, sample }, facts);
  // the address of a project: its screen, and the one request is the report (which writes nothing)
  assert.deepStrictEqual(at("/project", "/work/club/sample.json", none), { screen: "project", path: "/work/club/sample.json", ask: "/api/project" });
  // no address ever asks the server to remember: that takes an act of a person on the page
  const addresses = [["/", null], ["/", "/x/sample.json"], ["/start", null], ["/start", "/x/sample.json"], ["/project", null], ["/project", "/x/sample.json"],
    ["/index.html", "/x/sample.json"], ["/project", "../../etc/passwd"], ["/anything", "/x/sample.json"]];
  for (const facts of [none, own]) {
    for (const [pathname, sample] of addresses) {
      const a = at(pathname, sample, facts);
      assert.ok(a.ask === null || a.ask === "/api/project", `${pathname} ${sample}: ${a.ask}`);
      assert.strictEqual(a.screen === "start", a.ask === null, "the start screen needs no project");
    }
  }
  // a tool that names no project of its own: every address but a project's is the start screen
  assert.deepStrictEqual(at("/", null, none), { screen: "start", path: null, ask: null });
  assert.deepStrictEqual(at("/", "/x/sample.json", none), { screen: "start", path: null, ask: null }, "only /project reads a sample from the address");
  assert.deepStrictEqual(at("/project", null, none), { screen: "start", path: null, ask: null });
  assert.deepStrictEqual(at("/start", "/x/sample.json", none), { screen: "start", path: null, ask: null });
  // a tool started on one project shows that project, unless the start screen is asked for
  assert.deepStrictEqual(at("/", null, own), { screen: "project", path: null, ask: "/api/project" });
  assert.deepStrictEqual(at("/project", null, own), { screen: "project", path: null, ask: "/api/project" });
  assert.deepStrictEqual(at("/start", null, own), { screen: "start", path: null, ask: null });
  assert.deepStrictEqual(H.arrival(null, null), { screen: "start", path: null, ask: null });

  // what a person does that is remembered, and nothing else is
  for (const act of ["open-row", "check-files", "purpose", "model-boxes", "locations"]) assert.strictEqual(H.remembers(act), true, act);
  for (const act of ["arrive", "arrival", "", undefined, null, "show", "copy", "heartbeat", "reload"]) assert.strictEqual(H.remembers(act), false, String(act));
  // Enter after choosing the purpose starts the next pass, as the instruction line promises
  for (const f of ["none", "checkbox"]) assert.strictEqual(H.enterPressesPrimary(f), true, f);
  for (const f of ["text", "select", "button", "summary", "link"]) assert.strictEqual(H.enterPressesPrimary(f), false, f);
}

// ---- a path that is no file, or too large a one --------------------------------------------------
{
  assert.deepStrictEqual(H.problem({ problem: "not-file", path: "/dev/null" }, {}),
    { text: "/dev/null is not a file.", fix: "A sample is a JSON file. Give the path of the file itself, or of a folder that holds a sample.json." });
  const big = H.problem({ problem: "too-big", path: "/work/huge.json" }, {});
  assert.strictEqual(big.text, "/work/huge.json is too large to be a sample: the tool reads samples of up to 20 MB.");
  assert.ok(big.fix.includes("--sample"));
  assert.ok(H.gone({ sample: "/work/pipe.json", problem: "not-file" }).includes("/work/pipe.json is not a file."));
}

// ---- a label file that cannot be read in full ---------------------------------------------------
{
  const line = { file: "/work/toy/labels/table-a/frames.jsonl", line: 3, problem: "not JSON" };
  const p = report({ sets: [{ setId: "table-a", foreign: false, damaged: [line] }, { setId: "table-b", foreign: false, damaged: [] }] });
  p.export.damaged = [Object.assign({ setId: "table-a" }, line)];
  assert.deepStrictEqual(H.warnings(p).map((w) => [w.text, w.step]), [["A label file cannot be read in full: /work/toy/labels/table-a/frames.jsonl, line 3 (not JSON). Until that is corrected, table-a opens read-only on the labeling page, so nothing is written over it, and is left out of the counts below and of the export.", "Open the file in a text editor and correct or remove that line: each line is one frame."]]);
  // several places, and a set of the labels folder that this sample does not list
  p.sets[0].damaged.push({ file: line.file, line: 7, problem: "not a JSON object" });
  p.export.damaged = [Object.assign({ setId: "table-a" }, line), { setId: "old-set", file: "/work/toy/labels/old-set/meta.json", line: null, problem: "not JSON" }];
  const w = H.warnings(p)[0].text;
  assert.ok(w.startsWith("A label file cannot be read in full: /work/toy/labels/table-a/frames.jsonl, line 3 (not JSON), and 2 more places. Until that is corrected, table-a and old-set open read-only"), w);
  assert.deepStrictEqual(H.warnings(report()), [], "a project with whole files says nothing");
  // the reasons a pass cannot be started that come with it
  const q = withPasses(report(), { box: no("box", "damaged"), training: no("training", "bad-suggestions") });
  q.locations.prefill = place("/work/toy/prefill.jsonl", { state: "unreadable", problem: "it is larger than 20 MB" });
  q.modelBoxes = { offered: false, on: false, file: "prefill.jsonl" };
  assert.strictEqual(H.reason(q, "box"), "The label files of this sample cannot be read in full (see the top of this page).");
  assert.strictEqual(H.reason(q, "training"), "Needs a model's suggestions file, and this one cannot be used: it is larger than 20 MB (/work/toy/prefill.jsonl).");
  assert.strictEqual(H.modelBoxes(q).note, "The suggestions file cannot be used: it is larger than 20 MB (/work/toy/prefill.jsonl).");
  assert.deepStrictEqual(H.places(q).find((r) => r.key === "prefill").state, "It cannot be used: it is larger than 20 MB.");
}

// ---- the two commands for a video: the words as the server sends them, already fit for a shell ----
{
  const v = H.video(hub({ tools: { sample: "'/srv/our team/labeler/make_sample.py'", python: "'/opt/card tools/venv/bin/python3' -B" } }));
  assert.ok(v.commands[1].text.startsWith("'/opt/card tools/venv/bin/python3' -B '/srv/our team/labeler/make_sample.py' --out work/sample.json"), v.commands[1].text);
}

// ---- frames removed from the project: listed with a Restore button, and a project with none left ----
{
  assert.strictEqual(H.project(report(), { credit: "" }).removed, null, "nothing removed: no list");
  const p = report({ removed: { file: "/work/toy/labels/removed-frames.json", problem: null, frames: [
    { assetId: "table-a", name: "f00002.png", reason: "no-cards", at: "2026-01-01T00:00:00+00:00", replacedBy: null },
    { assetId: "table-b", name: "f00004.png", reason: null, at: null, replacedBy: "f00005.png" }] } });
  const v = H.project(p, { credit: "" });
  assert.strictEqual(v.removed.title, "Removed frames (2)");
  assert.deepStrictEqual(v.removed.items.map((x) => [x.assetId, x.name, x.text, x.button]), [
    ["table-a", "f00002.png", "table-a · f00002.png · No cards on it", "Restore"],
    ["table-b", "f00004.png", "table-b · f00004.png · f00005.png used instead", "Restore"]]);
  // every frame removed: said as such, not as a sample without frames
  const all = withPasses(report({ removed: p.removed, frames: { listed: 0, onDisk: 0, missing: 0, firstMissing: null, thumb: null } }),
    { box: no("box", "all-removed", { frames: 0 }) });
  const w = H.warnings(all);
  assert.deepStrictEqual(w.map((x) => x.text), ["Every frame of this sample was removed from the project, so there is nothing to label."]);
  assert.strictEqual(H.reason(all, "box"), "Every frame was removed from the project. Restore one under Removed frames below.");
  // a list that cannot be read is said, with its file
  const bad = report({ removed: { file: "/work/toy/labels/removed-frames.json", problem: "not-json", frames: [] } });
  assert.ok(H.warnings(bad)[0].text.startsWith("The list of removed frames cannot be used: /work/toy/labels/removed-frames.json."), H.warnings(bad)[0].text);
  for (const t of [v.removed.title, v.removed.lead, ...v.removed.items.map((x) => x.text), ...w.map((x) => [x.text, x.step]).flat(), H.reason(all, "box"), H.warnings(bad)[0].text]) {
    assert.ok(!/\b(prefill|role|mode|meta\.json|carry-over|drop(ped)?|delete[sd]? (its|the) labels)\b/.test(t), t);
  }
}

console.log("test_hubstate.js: all assertions passed");
