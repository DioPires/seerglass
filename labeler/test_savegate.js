// SPDX-License-Identifier: Apache-2.0
// Tests for savegate.js: when the labeling page may change a frame, when a change may be sent
// to be saved, and whether the tool that answers is the run the page was loaded from.
// Run: node test_savegate.js (also run by test_serve.py)
"use strict";
const assert = require("assert");
const G = require("./savegate.js");

const OPEN = { readOnly: false, noPicture: null, otherRun: false };

// ---- may the frame on the page be changed now? ----------------------------------------------
assert.deepStrictEqual(G.mayEdit(OPEN), { ok: true });
assert.deepStrictEqual(G.mayEdit({}), { ok: true });
// a frame whose picture did not load takes no change, whatever control asks
{
  const g = G.mayEdit({ ...OPEN, noPicture: "f00002.png" });
  assert.strictEqual(g.ok, false);
  assert.strictEqual(g.why, "no-picture");
  assert.ok(/picture/.test(g.text));
}
assert.strictEqual(G.mayEdit({ ...OPEN, readOnly: true }).why, "read-only");
assert.strictEqual(G.mayEdit({ ...OPEN, otherRun: true }).why, "other-run");
// the reason that cannot be mended on this page comes first
assert.strictEqual(G.mayEdit({ readOnly: true, noPicture: "f1.png", otherRun: true }).why, "other-run");
assert.strictEqual(G.mayEdit({ readOnly: true, noPicture: "f1.png" }).why, "read-only");

// ---- may this row be sent to the server now? -------------------------------------------------
assert.deepStrictEqual(G.maySave(OPEN, "f00001.png"), { ok: true });
// the frame on the page, its picture missing: never saved. This is the one place every change
// passes, so it holds for a control that did not ask `mayEdit` first.
assert.strictEqual(G.maySave({ ...OPEN, noPicture: "f00002.png" }, "f00002.png").why, "no-picture");
// a change made on ANOTHER frame that still waits to be saved goes out, whatever is on the page now
assert.deepStrictEqual(G.maySave({ ...OPEN, noPicture: "f00002.png" }, "f00001.png"), { ok: true });
// a page of another run sends nothing at all; a read-only set neither
for (const name of ["f00001.png", "f00002.png"]) {
  assert.strictEqual(G.maySave({ ...OPEN, otherRun: true }, name).why, "other-run");
  assert.strictEqual(G.maySave({ ...OPEN, readOnly: true }, name).why, "read-only");
}
// whenever a frame may not be changed, its row may not be saved either
for (const readOnly of [false, true]) for (const noPicture of [null, "a.png"]) for (const otherRun of [false, true]) {
  const s = { readOnly, noPicture, otherRun };
  if (!G.mayEdit(s).ok) assert.strictEqual(G.maySave(s, "a.png").ok, false, JSON.stringify(s));
}

// ---- is this the run the page was loaded from? -----------------------------------------------
const PAGE = { run: "aaaaaaaaaaaaaaaa", project: "1111111111111111", mode: "box", role: "training" };
assert.strictEqual(G.runIs(PAGE, { ...PAGE }), "same");
// the tool was stopped and started again with the same command: another run, the same work
assert.strictEqual(G.runIs(PAGE, { ...PAGE, run: "bbbbbbbbbbbbbbbb" }), "restarted");
// the reported defect: the same pass and the same purpose, but another project (a copy, with the
// same set ids). Pass and purpose agreeing is not enough.
assert.strictEqual(G.runIs(PAGE, { run: "bbbbbbbbbbbbbbbb", project: "2222222222222222", mode: "box", role: "training" }), "other");
assert.strictEqual(G.runIs(PAGE, { ...PAGE, run: "bbbbbbbbbbbbbbbb", mode: "identify" }), "other");
assert.strictEqual(G.runIs(PAGE, { ...PAGE, run: "bbbbbbbbbbbbbbbb", role: "test" }), "other");
// started again with no pass yet (the start screen), or an answer that says nothing
assert.strictEqual(G.runIs(PAGE, { run: "bbbbbbbbbbbbbbbb", project: null, mode: null, role: null }), "other");
assert.strictEqual(G.runIs(PAGE, {}), "other");
assert.strictEqual(G.runIs(PAGE, null), "other");
// a page that knows no run of its own is never "the same run"
assert.strictEqual(G.runIs({ mode: "box", role: "training" }, { mode: "box", role: "training" }), "other");

// what the page does about it
assert.strictEqual(G.onRun(PAGE, { ...PAGE }, false), "go-on");
assert.strictEqual(G.onRun(PAGE, { ...PAGE }, true), "go-on");
assert.strictEqual(G.onRun(PAGE, { ...PAGE, run: "b" }, false), "adopt", "the same work: go on under the new run");
assert.strictEqual(G.onRun(PAGE, { ...PAGE, run: "b" }, true), "adopt", "and what waited is saved");
const OTHER = { run: "b", project: "2222222222222222", mode: "box", role: "training" };
assert.strictEqual(G.onRun(PAGE, OTHER, false), "reload", "other work, nothing waits here: show what runs now");
assert.strictEqual(G.onRun(PAGE, OTHER, true), "stop", "other work and a change waits here: never loaded away, never saved there");

// the server's refusal of a save from a page of another run
assert.strictEqual(G.isOtherRun(409, { error: "…", problem: "other-run" }), true);
assert.strictEqual(G.isOtherRun(409, { error: "read-only" }), false, "a read-only set answers 409 too");
assert.strictEqual(G.isOtherRun(200, { ok: true }), false);
assert.strictEqual(G.isOtherRun(409, null), false);

// every save carries the run
assert.deepStrictEqual(G.saveBody("aaaa", { frame: { name: "f.png" } }), { run: "aaaa", frame: { name: "f.png" } });
assert.deepStrictEqual(Object.keys(G.saveBody("aaaa", { events: [] })), ["run", "events"]);
assert.deepStrictEqual(G.saveBody(undefined, {}), { run: null });

// ---- what a page that saves nothing more says -------------------------------------------------
{
  const text = G.stopText({ unsaved: ["f00001.png", "f00003.png"], launch: ["python3", "labeler/serve.py", "--mode", "box", "--sample", "my work/sample.json"] });
  assert.ok(text.startsWith("The tool now runs other work (it was switched to another pass or project, or started again)"), text);
  assert.ok(text.includes("switch the tool back to this pass on the project screen; this page then saves them by itself"), "the way back needs no terminal");
  assert.ok(text.includes("NOT SAVED: f00001.png · f00003.png"), "what was not saved is named");
  assert.ok(text.includes("python3 labeler/serve.py --mode box --sample 'my work/sample.json'"), "and how to save it: the command of this page's run, ready to paste");
  assert.ok(text.includes("reload this page"));
  assert.ok(!/\bSaved\b/.test(text.replace("NOT SAVED", "")), "it never says Saved");
  const one = G.stopText({ unsaved: ["f00001.png"], launch: null });
  assert.ok(one.includes("NOT SAVED: f00001.png. The page keeps this change.") && one.includes("To save it"));
  const none = G.stopText({ unsaved: [], launch: [] });
  assert.ok(none.includes("Everything on this page was saved before that") && none.includes("Reload this page"));
}
// ---- what a page tells the server, and why a switch was refused -------------------------------------
assert.strictEqual(G.unsavedCount({}), 0);
assert.strictEqual(G.unsavedCount({ failed: 0, pending: false, saving: false }), 0);
assert.strictEqual(G.unsavedCount({ pending: true }), 1);
assert.strictEqual(G.unsavedCount({ saving: true, pending: true }), 1);
assert.strictEqual(G.unsavedCount({ failed: 2, saving: true }), 3);
assert.strictEqual(G.switchText({ problem: "busy", pages: 1, saving: 0 }, "Names"),
  "The tool did not switch to Names: another tab of this tool has changes that are not saved yet. Go to it and wait until the top bar says Saved (or close it), then press again.");
assert.strictEqual(G.switchText({ problem: "busy", pages: 2, saving: 1 }, "Names"),
  "The tool did not switch to Names: 2 other tabs of this tool have changes that are not saved yet. Go to them and wait until the top bar says Saved (or close them), then press again.");
assert.strictEqual(G.switchText({ problem: "busy", pages: 0, saving: 1 }, "Names"), "The tool did not switch to Names: a save is being written. Press again in a moment.");
assert.strictEqual(G.switchText({ problem: "unsaved" }), "The tool did not switch: this page has changes that are not saved yet. Wait until the top bar says Saved, then press again.");
assert.strictEqual(G.switchText({ problem: "other-run" }, "Names"), "The tool was switched by another tab meanwhile. Reload this page to see what it runs now.");
assert.strictEqual(G.switchText({ problem: "refused", detail: "--catalog: it went away" }, "Names"), "The tool could not start Names: --catalog: it went away. It runs on as before.");
assert.strictEqual(G.switchText({ problem: "nothing-to-name" }, "Names"), "The tool did not switch to Names (nothing-to-name). The project screen says what it needs.");
for (const problem of ["busy", "unsaved", "other-run", "no-pass", "refused", "no-answer", "x"]) assert.ok(G.switchText({ problem }, "Names").length > 20, problem);

// a command, word by word
assert.strictEqual(G.commandText(["python3", "a b/serve.py", "--art", "it's"]), "python3 'a b/serve.py' --art 'it'\"'\"'s'");
assert.strictEqual(G.commandText(null), "");

console.log("test_savegate.js: all assertions passed");
