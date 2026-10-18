// SPDX-License-Identifier: Apache-2.0
// When the labeling page may change a frame, when a change may be sent to be saved, and whether
// the tool that answers is still the run this page was loaded from. Pure: no DOM, no fetch.
// Shared by the page (window.LabelerSaveGate) and Node (test_savegate.js).
//
// Every control of the page asks `mayEdit` before it changes anything, and every change passes
// `maySave` on its way to the server, in one place (sources.js, scheduleSave and saveFrame): a
// control that forgot to ask still cannot save.
//
//   mayEdit(state)          may the frame on the page be changed now?
//   maySave(state, name)    may the row of frame `name` be sent to the server now?
//   runIs(page, now)        is the tool that answers the run the page was loaded from?
//   onRun(page, now, unsaved)  what the page does about the answer
//   isOtherRun(status, body)   a save was refused because its page is of another run
//   stopText(...)           what the page says once it may save nothing more
//   unsavedCount(state)     what a page tells the server about its unsaved changes (/api/tabs)
//   switchText(answer)      why the tool did not switch to another pass, and what to do
//
// The rules:
//   - A set that is read-only takes no change.
//   - A frame whose picture did not load takes no change and is never saved: nobody saw it.
//   - Every save carries the identity of the run the page was loaded from (/api/config `run`),
//     and the server refuses a save of another run. A switch to another pass from the page is a
//     new run too, and the server refuses it while a save is in progress or a page says it holds
//     unsaved changes. When the tool was switched or started again:
//       the same work (the same sample, labels folder, pass and purpose: `project`)
//           the page goes on under the new run's identity, and what waited is saved;
//       other work, and nothing on the page is unsaved
//           the page loads again, and shows what the tool runs now;
//       other work, and something on the page is unsaved
//           the page stops saving, says that the tool was started again, names what is not
//           saved and how to save it. It never shows "Saved".
(function (root) {
  "use strict";

  const WHY = {
    "read-only": "This set is read-only.",
    "no-picture": "The picture of this frame did not load, so the frame cannot be labeled and nothing is saved for it.",
    "other-run": "The tool now runs other work, so nothing more is saved from this page (see the banner).",
  };
  const OTHER_RUN = "the tool now runs other work";

  function no(why) { return { ok: false, why, text: WHY[why] }; }

  // state: {readOnly, noPicture: the name of the frame on the page when its picture did not
  // load (else null), otherRun: the tool that answers is another run, doing other work}
  function mayEdit(state) {
    const s = state || {};
    if (s.otherRun) return no("other-run");
    if (s.readOnly) return no("read-only");
    if (s.noPicture) return no("no-picture");
    return { ok: true };
  }

  // The row of frame `name`: the frame on the page, or one that waits to be saved (a save that
  // failed is tried again from any frame).
  function maySave(state, name) {
    const s = state || {};
    if (s.otherRun) return no("other-run");
    if (s.readOnly) return no("read-only");
    if (s.noPicture && s.noPicture === name) return no("no-picture");
    return { ok: true };
  }

  // page: /api/config as the page read it when it loaded; now: the same, asked again.
  // → "same"       the run the page was loaded from
  //   "restarted"  another run of the tool, doing the same work
  //   "other"      another run, doing other work, or running no pass
  function runIs(page, now) {
    const p = page || {}, n = now || {};
    if (p.run && n.run === p.run) return "same";
    if (p.project && n.project === p.project && n.mode === p.mode && n.role === p.role) return "restarted";
    return "other";
  }

  // → "go-on"   nothing changed
  //   "adopt"   go on under the new run's identity (`now.run`); what waited is saved
  //   "reload"  load the page again: it then shows what the tool runs now
  //   "stop"    save nothing more, and say so
  function onRun(page, now, unsaved) {
    const is = runIs(page, now);
    if (is === "same") return "go-on";
    if (is === "restarted") return "adopt";
    return unsaved ? "stop" : "reload";
  }

  // The server's answer to a save from a page of another run (serve.py: 409, problem other-run).
  function isOtherRun(status, body) { return status === 409 && !!body && body.problem === "other-run"; }

  // What a save sends: the run's identity first, then the save itself.
  function saveBody(run, body) { return Object.assign({ run: run === undefined ? null : run }, body || {}); }

  // A word of a command, quoted for a shell when it needs to be.
  function word(w) {
    const s = String(w);
    return /^[A-Za-z0-9_@%+=:,./-]+$/.test(s) ? s : `'${s.replace(/'/g, "'\"'\"'")}'`;
  }
  function commandText(argv) { return (argv || []).map(word).join(" "); }

  // The banner of a page that may save nothing more. unsaved: the frames (names) with changes
  // that are not saved; launch: the command of the run the page was loaded from (argv).
  function stopText(o) {
    const names = (o && o.unsaved) || [];
    const cmd = commandText(o && o.launch);
    const lost = names.length
      ? `NOT SAVED: ${names.join(" · ")}. The page keeps ${names.length === 1 ? "this change" : "these changes"}.`
      : "Everything on this page was saved before that.";
    const it = names.length === 1 ? "it" : "them";
    const keep = names.length
      ? ` To save ${it}: keep this tab open, and in another tab switch the tool back to this pass on the project screen; this page then saves ${it} by itself.`
        + ` (A tool started again as it ran for this page does the same${cmd ? `: ${cmd}` : ""}.) To drop ${it} and work in what the tool runs now, reload this page.`
      : " Reload this page to work in what the tool runs now.";
    return `The tool now runs other work (it was switched to another pass or project, or started again), so this page belongs to the run before and nothing more is saved from it. ${lost}${keep}`;
  }

  // What a labeling page tells the server about itself: how many changes wait to be saved (0
  // when none). state: {failed: count of frames whose save failed, pending, saving}.
  function unsavedCount(state) {
    const s = state || {};
    return (s.failed || 0) + (s.pending || s.saving ? 1 : 0);
  }

  // Why the tool did not switch (/api/switch's answer), in one sentence that says what to do.
  // `name`: the pass asked for, as the page names it.
  function switchText(answer, name) {
    const a = answer || {};
    const to = name ? ` to ${name}` : "";
    switch (a.problem) {
      case "busy":
        if (a.pages) return `The tool did not switch${to}: ${a.pages === 1 ? "another tab of this tool has" : `${a.pages} other tabs of this tool have`} changes that are not saved yet. Go to ${a.pages === 1 ? "it" : "them"} and wait until the top bar says Saved (or close ${a.pages === 1 ? "it" : "them"}), then press again.`;
        return `The tool did not switch${to}: a save is being written. Press again in a moment.`;
      case "unsaved": return `The tool did not switch${to}: this page has changes that are not saved yet. Wait until the top bar says Saved, then press again.`;
      case "other-run": return "The tool was switched by another tab meanwhile. Reload this page to see what it runs now.";
      case "no-pass": return "The tool runs no pass right now. Reload this page to choose one.";
      case "refused": return `The tool could not start ${name || "that pass"}: ${a.detail || "it refused"}. It runs on as before.`;
      case "no-answer": return "The tool did not answer. Is it still running?";
      default: return `The tool did not switch${to}${a.problem ? ` (${a.problem})` : ""}. The project screen says what it needs.`;
    }
  }

  const api = { WHY, OTHER_RUN, mayEdit, maySave, runIs, onRun, isOtherRun, saveBody, word, commandText, stopText, unsavedCount, switchText };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.LabelerSaveGate = api;
})(typeof window !== "undefined" ? window : globalThis);
