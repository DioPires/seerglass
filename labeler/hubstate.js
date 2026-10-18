// SPDX-License-Identifier: Apache-2.0
// What the start screen and the project screen say, worked out from the facts the server sends
// (project.py). Pure: no DOM, no fetch. Shared by the page (window.LabelerHubState) and Node
// (test_hubstate.js).
//
//   start(facts)          the start screen: its instruction line, and its three paths (label my
//                         frames, learn on a toy game, I have a video), the one that fits first
//   video(facts)          the third path: what is needed, and the commands with the parts to fill in
//   problem(result, ...)  why a path is no project, and what to do about it
//   project(report, form) the project screen: its instruction line, header, what is missing,
//                         the locations, the purpose question, the passes in the order of the
//                         purpose, and the export
//   lists(report)         which passes a purpose puts first, which are its other protocol, which
//                         are other ways
//   exportRow(report, form)   the export: what it writes, what it asks for, what it answered
//   exportProblems(problems)  each refusal of the export as one sentence that says what to fill in
//   removed(report)       the frames removed from the project, each with its Restore button
//   arrival(where, facts) what arriving at an address does: which screen, and the one request
//   remembers(act)        which acts of a person make the tool remember a project
//
// A key in a text is written {{Enter}}; the page draws it as a key cap, as in uistate.js.
(function (root) {
  "use strict";
  const U = (typeof module !== "undefined" && module.exports) ? require("./uistate.js") : root.LabelerUiState;

  const ROLES = [
    { value: "training", label: "Training: teach the model", name: "Training",
      help: "The labels a model learns from. A model's suggestions may be shown while you label, since they only save time here." },
    { value: "development", label: "Development: compare and tune", name: "Development",
      help: "The labels you measure a model on while you work on it. No model's suggestion is shown while you draw." },
    { value: "test", label: "Test: final held-out check", name: "Test",
      help: "The labels for the final measurement. No model output of any kind is shown: every label comes from the frame alone." },
  ];
  const MEANING = {
    box: "Draw a box around every card on each frame.",
    identify: "Say which card each box shows.",
    corner: "Mark the four corners of every card, for footage where a rectangle does not fit.",
    training: "Check a model's corners and names instead of clicking them yourself.",
    point: "Say what is under each of a fixed number of random points per frame.",
    center: "Click the center of every card on each frame.",
    verify: "Name each clicked center, with no model's proposal shown.",
  };
  const BOX_FLOW = ["box", "corner", "training"];
  const ACTION = { start: "Start", continue: "Continue", review: "Review" };

  // Where the guide (README.md, served at /guide) says more: a phrase that is in it once, which the
  // browser finds and shows (a text fragment). No dash, comma or ampersand in a phrase.
  const GUIDE = {
    toy: { section: "Five-minute start", find: "No footage and no card art are needed for a first try" },
    start: { section: "Starting the tool", find: "is one sample file (the list of frames to label) and what lies beside it" },
    sample: { section: "The sample", find: "The list of frames to label:" },
    catalog: { section: "The catalog", find: "The one file that describes a game's cards." },
    video: { section: "From a video to frames", find: "The tool labels frames and not videos." },
    manifest: { section: "Offering labels: the manifest export", find: "A manifest names videos and not files" },
    own: { section: "A recording of your own", find: "A phone or screen recording of your own play can be labeled" },
    measuring: { section: "Labels for measuring a model", find: "The census of centers and the points are the two protocols" },
  };
  function guide(key) {
    const g = GUIDE[key];
    return g ? { label: `Guide: ${g.section}`, href: `/guide#:~:text=${encodeURIComponent(g.find)}` } : null;
  }

  function n(count, one, many) { return `${count} ${count === 1 ? one : many || one + "s"}`; }
  function roleName(value) { const r = ROLES.find((x) => x.value === value); return r ? r.name : String(value || ""); }
  function passName(id) { return id === "export" ? "Export" : U.passName(id); }
  function list(words) { return words.length < 2 ? words.join("") : `${words.slice(0, -1).join(", ")} and ${words[words.length - 1]}`; }
  function size(p) { return `${n(p.recordings, "recording")} · ${n(typeof p.frames === "number" ? p.frames : p.frames.listed, "frame")}`; }

  // ---- why a path is no project ---------------------------------------------------------------

  // r: {problem, path}; ctx: {startDir, sampleTool} → {text, fix}
  function problem(r, ctx) {
    const c = ctx || {};
    const at = r.path || "";
    const make = `${c.python || "python3"} ${c.sampleTool || "labeler/make_sample.py"} --out sample.json --frames-root FOLDER_OF_FRAME_FOLDERS --n 8`;
    switch (r.problem) {
      case "empty": return { text: "Type the path of a sample file, or of a folder that holds a sample.json.", fix: "" };
      case "missing": return { text: `Nothing is at ${at}.`, fix: `Check the spelling. A path that does not start with / is read from ${c.startDir || "the folder the tool was started in"}.` };
      case "bad-path": return { text: `That is not a path this computer understands: ${at}`, fix: "Give the path as your file manager or terminal shows it." };
      case "no-sample-in-folder": return { text: `There is no sample.json in ${at}.`, fix: `If the sample has another name, give the file itself. To make a sample from folders of frames: ${make}` };
      case "unreadable": return { text: `${at} cannot be read.`, fix: "Check that you are allowed to read the file." };
      case "not-file": return { text: `${at} is not a file.`, fix: "A sample is a JSON file. Give the path of the file itself, or of a folder that holds a sample.json." };
      case "too-big": return { text: `${at} is too large to be a sample: the tool reads samples of up to 20 MB.`, fix: "A sample only lists frames. If this is the right file, start the tool on it with flags (--mode, --role and --sample), which read a file of any size." };
      case "not-json": return { text: `${at} is not a sample: it is not a JSON file.`, fix: `A sample is a JSON file that lists the frames to label. To make one from folders of frames: ${make}` };
      case "not-sample": return { text: `${at} is a JSON file but not a sample: it has no list of recordings ("assets").`, fix: `A sample looks like {"assets": [{"assetId": "table-a", "frames": [{"t": 0, "name": "f00001.png"}]}]}. To make one from folders of frames: ${make}` };
      case "none": return { text: "No project is open.", fix: "Choose one on the start screen." };
      default: return { text: `${at || "That"} could not be opened as a sample.`, fix: "" };
    }
  }

  // A project that was opened before and whose sample cannot be read any more.
  function gone(entry) {
    return entry.problem === "missing"
      ? `This sample is not there any more: ${entry.sample}. It was moved or deleted. Open it from where it is now, or remove it from this list.`
      : `This sample cannot be read any more: ${entry.sample}. ${problem({ problem: entry.problem, path: entry.sample }).text}`;
  }

  // ---- what comes next in a project -------------------------------------------------------------

  // → {pass, startable, label, text}: the button that goes on, and the sentence beside it
  function next(p) {
    const step = p.next.pass;
    const name = p.name;
    if (!p.role.value) return { pass: null, startable: false, label: `Open ${name}`, text: "Next: say what these labels will be used for." };
    if (step === "export") return { pass: "export", startable: false, label: `Open ${name}`, text: `Every frame is done and nothing is left to name. Next: write the ${p.export && p.export.kind === "manifest" ? "labels" : "boxes"} to a file.` };
    if (!step) return { pass: null, startable: false, label: `Open ${name}`, text: "Nothing can be started right now: its screen says why." };
    const row = passRow(p, step);
    const verb = p.passes[step].action === "start" ? "Start" : "Continue with";
    return { pass: step, startable: !!p.next.startable, label: p.next.startable ? `${verb} ${passName(step)} on ${name}` : `Open ${name}`,
      does: `${p.passes[step].action === "start" ? "starts" : "goes on with"} ${passName(step)} on ${name}`, text: `Next: ${passName(step)}. ${row.status}.` };
  }

  // ---- the start screen -------------------------------------------------------------------------

  // facts: /api/hub → {guide, title, lead, running, paths: [{id, title, promise, fits}], open (the
  // path shown first), resume, before, found, foundNote, toy, video, primary}. `primary`: the one
  // button Enter presses: running, resume, found, toy, or null.
  function start(facts) {
    const server = facts.server || {};
    const own = server.configured ? server.project : null;
    const resume = facts.resume && facts.resume.sample !== own ? facts.resume : null;
    const shown = new Set([resume && resume.sample, own].filter(Boolean));
    const before = (facts.recent || []).filter((e) => !shown.has(e.sample)).map((e) => (e.problem
      ? { sample: e.sample, name: e.name, gone: true, text: gone(e) }
      : { sample: e.sample, name: e.name, folder: e.folder, text: size(e) }));
    for (const e of before) shown.add(e.sample);
    const found = (facts.found || []).filter((e) => !shown.has(e.sample)).map((e) => ({ sample: e.sample, name: e.name, folder: e.folder, text: size(e) }));
    const all = (facts.found || []).length;
    const where = facts.startDir;
    const depth = (facts.search || {}).depth;
    let foundNote = "";
    if (!all) foundNote = `No sample was found in ${where} or the ${n(depth, "level")} of folders below it. A sample is a JSON file that lists the frames to label, with “sample” in its name.`;
    else if (!found.length) foundNote = `No other sample was found in ${where}.`;
    if ((facts.search || {}).cut) foundNote += `${foundNote ? " " : ""}The search stopped early because this folder holds a great many files, so there may be more samples: open one by its path below.`;

    const toy = { text: "A made-up card game, drawn by this tool, with a pretend model's boxes, so that you can try every step. Nothing is downloaded.",
      path: facts.toy.path, where: "", button: "Make the toy game and open it", disabled: false, missing: null };
    if (facts.toy.state === "ok") toy.where = `It will be written to a new folder: ${facts.toy.path}`;
    else {
      toy.disabled = true;
      toy.where = facts.toy.state === "not-writable"
        ? `It cannot be made here: ${where} cannot be written.`
        : `It cannot be made here: ${where} already holds a great many toy folders.`;
      toy.missing = facts.toy.state === "not-writable"
        ? { text: `Missing: a folder this tool may write to.`, step: "Stop the tool (Ctrl+C in its terminal) and start it in a folder you can write to.", guide: guide("toy") }
        : { text: "Missing: a free name for the toy folder.", step: "Open one of the toy folders under “Label my frames”, or delete some of them.", guide: guide("toy") };
    }

    let running = null;
    if (server.configured) {
      running = { text: `The tool is running the ${passName(server.mode)} pass on ${server.name}, for ${roleName(server.role)}.`,
        primary: `Go to the ${passName(server.mode)} pass`, secondary: `Show the passes of ${server.name}` };
    }
    let resumeView = null;
    if (resume) {
      const nx = next(resume);
      // a pass of the project opened last can be started from here, or the tool switched to it
      resumeView = { sample: resume.sample, name: resume.name, folder: resume.folder,
        text: `${size(resume)}${resume.role.value ? " · " + roleName(resume.role.value) : ""}`, next: nx.text,
        primary: nx.label, pass: nx.startable ? nx.pass : null, does: nx.does, switches: !!server.configured && nx.startable,
        secondary: nx.startable ? `Show all passes of ${resume.name}` : null };
    }
    const own_ = before.some((e) => !e.gone);
    const anything = !!(resumeView || own_ || found.length);
    const primary = running ? "running" : resumeView ? "resume" : found.length === 1 && !own_ ? "found" : !anything && !toy.disabled ? "toy" : null;
    // the path that fits what was found, shown first
    const open = running || anything ? "frames" : "toy";
    const paths = [
      { id: "frames", title: "Label my frames", promise: "You have frames of play and a sample that lists them: open it and label, pass by pass.",
        fits: open === "frames" },
      { id: "toy", title: "Learn on a toy game", promise: "A made-up game drawn for you, to try every step in a few minutes. Nothing to prepare.", fits: open === "toy" },
      { id: "video", title: "I have a video, not frames", promise: "Turn the video into frames and a sample first, with two commands, then come back here.", fits: false },
    ];
    let guideText;
    if (running) guideText = `${running.text} {{Enter}} goes to it.`;
    else if (resumeView) guideText = resumeView.pass
      ? `Label my frames fits: carry on where you stopped. {{Enter}} ${resumeView.does}${resumeView.switches ? " (the tool switches to it: no new start)" : ""}.`
      : `Label my frames fits: {{Enter}} opens ${resumeView.name}, the project you opened last.`;
    else if (primary === "found") guideText = "Label my frames fits: one sample was found here. {{Enter}} opens it and shows its passes.";
    else if (found.length || own_) guideText = "Label my frames fits: choose one of the samples below.";
    else if (primary === "toy") guideText = "Nothing to label was found here. New to the tool? {{Enter}} makes the toy game. Have frames elsewhere: Label my frames. Only a video: the third path.";
    else guideText = "Nothing to label was found here. Open a sample by its path under Label my frames, or see the third path for a video.";
    return { guide: guideText, title: "Card labeler",
      lead: "Label trading cards in frames of video: where each card is, which way up it lies and which card it is. Everything stays on this computer: the tool reads and writes files in folders you choose, and opens no connection.",
      running, paths, open, resume: resumeView, before, found, foundNote, toy, video: video(facts), primary,
      none: !anything ? { text: "Missing: a sample, the list of frames to label.", step: "Make one from folders of frames, then open it here:", guide: guide("sample") } : null };
  }

  // "I have a video, not frames": what is needed, and the two commands that come before the tool
  // (README, "From a video to frames"), with the parts a person fills in marked. The server sends
  // the interpreter and the script ready to put into a command: each word quoted for a shell where
  // it needs to be (a path with a space), as every command it sends is.
  function video(facts) {
    const t = (facts && facts.tools) || {};
    const w = (text) => ({ t: text, fill: false });
    const f = (text) => ({ t: text, fill: true });
    const cmd = (parts) => ({ text: parts.map((x) => x.t).join(" "), parts });
    const first = cmd([w("cardvision infer"), f("my-video.mp4"), w("--release"), f("release.json"),
      w("--every 2 --frame-format png --frames"), f("work/frames/my-video"), w("--prefill-out"), f("work/prefill.jsonl")]);
    const second = cmd([w(`${t.python || "python3"} ${t.sample || "labeler/make_sample.py"} --out`), f("work/sample.json"), w("--asset"),
      f("my-video=work/frames/my-video"), w("--all --times"), f("work/prefill.jsonl"), w("--local")]);
    return { summary: "I have a video, not frames",
      needs: "Needed first: this repository installed, for its cardvision command, and a release of its models (release.json) for your game. Without them, any program that saves frames of a video as .png or .jpg files into a folder will do.",
      text: "The tool labels frames and not videos. Cut frames from the video, then list them in a sample. Fill in the marked parts, then run each in a terminal:",
      commands: [first, second],
      after: "The first saves one frame every 2 seconds with its time, and the model's boxes for each; the second lists those frames as a recording of your own (--local). Then open work/sample.json under Label my frames. With frames from another program: the second command alone, without --times and its file.",
      guide: guide("video") };
  }

  // ---- the project screen: locations ------------------------------------------------------------

  const PLACES = [
    { key: "frames", label: "Frames", need: "Needed to show the frames." },
    { key: "labels", label: "Labels", need: "Every label is saved here." },
    { key: "catalog", label: "Catalog", need: "Needed to name cards: a JSON list of the game's cards." },
    { key: "art", label: "Card images", need: "Optional: without them the lists show card names only." },
    { key: "decklists", label: "Deck lists", need: "Optional: a deck list puts the cards in play first when you name." },
    { key: "prefill", label: "Model suggestions", need: "Optional, Training only: a model's boxes to start from." },
  ];
  const SOURCE = { page: "changed here", flag: "given when the tool was started", remembered: "remembered from last time", beside: "beside the sample" };

  function placeState(key, l) {
    switch (l.state) {
      case "ok":
        if (key === "catalog") return { ok: true, text: `Found: ${n(l.cards, "card")}.` };
        if (key === "decklists") return { ok: true, text: `Found: ${n(l.files, "deck list")}.` };
        if (key === "art" && l.url) return { ok: true, text: "An address: your browser loads the card images from it." };
        return { ok: true, text: "Found." };
      case "will-create": return { ok: true, text: "Not there yet: it is made at the first save." };
      case "unused": return { ok: true, text: "Not used: every recording names its own folder in the sample." };
      case "off": return { ok: true, text: "The running pass was started without one." };
      case "absent": return { ok: key !== "frames", text: "Nothing is there." };
      // a place that was named and holds nothing is a fault; the conventional place holding nothing is not, however the run came to name it
      case "missing": return { ok: Boolean(l.usual) && key !== "frames", text: "Nothing is there." };
      case "unreadable": return { ok: false, text: `It cannot be used: ${l.problem}.` };
      case "not-writable": return { ok: false, text: "It cannot be written." };
      case "not-folder": return { ok: false, text: "This is a file, not a folder." };
      default: return { ok: false, text: "" };
    }
  }

  // → [{key, label, path, where, state, ok, found, need}]
  function places(p) {
    return PLACES.map((x) => {
      const l = p.locations[x.key];
      const s = placeState(x.key, l);
      // found: something usable is there. A place with nothing in it is no fault when it is optional (ok), but it is not found either.
      return { key: x.key, label: x.label, path: l.path, where: l.usual ? SOURCE.beside : SOURCE[l.source] || "", state: s.text, ok: s.ok, found: l.state === "ok", need: x.need };
    });
  }

  // ---- the project screen: what is missing, and what is wrong -------------------------------------

  // → [{text, step, guide}]: each thing that keeps the project off the usual way, with the one step
  // back to it and where the guide says more (guide may be null).
  function warnings(p) {
    const out = [];
    const add = (text, step, key) => out.push({ text, step: step || "", guide: key ? guide(key) : null });
    const f = p.frames, lab = p.locations.labels, cat = p.locations.catalog;
    const dropped = (p.removed && p.removed.frames) || [];
    if (p.recordings && !f.listed && dropped.length) {
      add("Every frame of this sample was removed from the project, so there is nothing to label.", "Restore the frames you want back, under Removed frames below.");
    } else if (!p.recordings || !f.listed) {
      add("Missing: frames to label. This sample lists none.", "Make a sample from folders of frames with make_sample.py (on the start screen: I have a video, not frames), then open that one.", "sample");
    } else if (!f.onDisk) {
      add(`Missing: the frames. None of the ${n(f.listed, "frame")} this sample lists is on disk; the first is looked for at ${f.firstMissing}.`,
        "Put the frames there, or give the right frames folder under Change locations.", "sample");
    } else if (f.missing) {
      add(`${f.missing} of ${n(f.listed, "frame")} ${f.missing === 1 ? "is" : "are"} not on disk, starting with ${f.firstMissing}. ${f.missing === 1 ? "That frame" : "Those frames"} cannot be labeled.`,
        `Put ${f.missing === 1 ? "it" : "them"} there, or give the right frames folder under Change locations.`, "sample");
    }
    if (p.removed && p.removed.problem) {
      add(`The list of removed frames cannot be used: ${p.removed.file}. Until it is corrected, every frame of the sample is labeled and no frame can be removed or restored.`,
        "Correct the file, or delete it to bring every removed frame back.");
    }
    if (lab.state === "not-writable") add(`Missing: a labels folder this tool may write to. ${lab.path} cannot be written, so no label could be saved and no pass can be started.`, "Give another folder under Change locations, or make this one writable.", "start");
    if (lab.state === "not-folder") add(`The labels location is a file, not a folder: ${lab.path}.`, "Give a folder under Change locations.", "start");
    if (cat.state === "unreadable") add(`The catalog cannot be used: ${cat.problem} (${cat.path}). Without it no card can be named.`, "Correct the file, or give another under Change locations.", "catalog");
    else if (cat.state !== "ok" && p.role.value) add(`Missing: a catalog, the list of the game's cards. Without it boxes and centers can be made, but no card can be named.${cat.path ? ` It is looked for at ${cat.path}.` : ""}`,
      "Put the game's catalog.json beside the sample, or give its path under Change locations.", "catalog");
    const foreign = p.sets.filter((s) => s.foreign).map((s) => s.setId);
    if (foreign.length) add(`${list(foreign)}: the label files in ${foreign.length === 1 ? "this folder were" : "these folders were"} not written by this tool. They are left as they are and not counted here.`);
    // a label file with a line that cannot be read: said with the file and the line, never passed over
    const seen = new Set();
    const damaged = p.sets.flatMap((s) => (s.damaged || []).map((d) => Object.assign({ setId: s.setId }, d))).concat((p.export && p.export.damaged) || [])
      .filter((d) => { const k = `${d.file}#${d.line}`; if (seen.has(k)) return false; seen.add(k); return true; });
    if (damaged.length) {
      const first = damaged[0];
      const sets = Array.from(new Set(damaged.map((d) => d.setId)));
      add(`A label file cannot be read in full: ${first.file}${first.line ? `, line ${first.line}` : ""} (${first.problem})${damaged.length > 1 ? `, and ${n(damaged.length - 1, "more place")}` : ""}. `
        + `Until that is corrected, ${list(sets)} ${sets.length === 1 ? "opens" : "open"} read-only on the labeling page, so nothing is written over ${sets.length === 1 ? "it" : "them"}, and ${sets.length === 1 ? "is" : "are"} left out of the counts below and of the export.`,
      "Open the file in a text editor and correct or remove that line: each line is one frame.");
    }
    return out;
  }

  // ---- the project screen: what the labels are for ----------------------------------------------

  // → {question, options: [{value, label, help, checked, disabled}], note}
  function purpose(p) {
    const r = p.role;
    const stored = Object.keys(r.stored);
    const running = p.server.configured && p.server.sameProject;
    let note = "";
    if (r.refused && r.fixed) {
      note = `You asked for ${roleName(r.refused)}, but the labels in this folder were made for ${roleName(r.value)}, and a label set keeps its purpose from its first save. The passes below go on with the ${roleName(r.value)} labels. To label these frames for ${roleName(r.refused)}, give another labels folder under Change locations.`;
    } else if (r.fixed) {
      note = `Read from the labels in this folder: they were made for ${roleName(r.value)}. A label set keeps its purpose from its first save, so it cannot be changed here. To label these frames for another purpose, give another labels folder under Change locations.`;
    } else if (stored.length > 1) {
      note = `This folder holds labels made for ${list(stored.map((s) => `${roleName(s)} (${r.stored[s].join(", ")})`))}. Choose which to work on; the others are left as they are.`;
    } else if (running) {
      note = `The tool is running a pass for ${roleName(r.value)}. To choose another purpose, stop the tool (Ctrl+C in its terminal) and start it again.`;
    } else if (!r.value) {
      note = "Choose one. It is stored with the first label and decides what the tool may show you while you label.";
    } else {
      note = "You can change this until the first label is saved.";
    }
    const locked = r.fixed || running;
    const now = ROLES.find((x) => x.value === r.value);
    return { question: "What will these labels be used for?", note,
      // settled (read from the labels, or the tool runs a pass): the answer, not a choice
      settled: locked && now ? { label: now.label, help: now.help } : null,
      options: ROLES.map((x) => ({ value: x.value, label: x.label, help: x.help, checked: r.value === x.value,
        disabled: locked ? r.value !== x.value : !r.choices.includes(x.value) })) };
  }

  // ---- the project screen: the passes -----------------------------------------------------------

  function reason(p, id) {
    const f = p.passes[id];
    const loc = p.locations;
    switch (f.reason) {
      case "no-role": return "Say what the labels are for first (above).";
      case "training-only": return "Only for Training labels: it starts from a model's corners and names.";
      case "not-for-training": return "Only for Development or Test labels: they are what a model is measured on.";
      case "refused": return "Not offered for this purpose.";
      case "labels-not-writable": return "The labels folder cannot be written (see the top of this page).";
      case "no-catalog": return `Needs a catalog: a JSON list of the game's cards. ${loc.catalog.path ? `None is at ${loc.catalog.path}` : "The running pass was started without one"}.`;
      case "bad-catalog": return `Needs a catalog, and this one cannot be used: ${loc.catalog.problem} (${loc.catalog.path}).`;
      case "nothing-to-name": return "Nothing to name yet: draw boxes first.";
      case "no-centers": return "Nothing to check yet: click the centers first, in the Centers pass.";
      case "no-suggestions": return `Needs a model's suggestions file. ${loc.prefill.path ? `None is at ${loc.prefill.path}` : "The running pass was started without one"}.`;
      case "bad-suggestions": return `Needs a model's suggestions file, and this one cannot be used: ${loc.prefill.problem} (${loc.prefill.path}).`;
      case "damaged": return "The label files of this sample cannot be read in full (see the top of this page).";
      case "other-way": return `${p.recordings === 1 ? "This recording already has" : "These recordings already have"} ${list((f.have || []).map(passName))} labels, and a recording keeps one way of labeling in a labels folder. To label ${p.recordings === 1 ? "it" : "them"} this way too, give another labels folder under Change locations.`;
      case "no-frames": return "This sample lists no frames.";
      case "all-removed": return "Every frame was removed from the project. Restore one under Removed frames below.";
      case "frames-missing": return "The frames are not on disk (see the top of this page).";
      default: return "";
    }
  }

  function status(p, id) {
    const f = p.passes[id];
    if (id === "identify") {
      if (!f.boxes) return "No boxes yet";
      if (f.left > 0) return `${n(f.left, "card")} left to name${f.cards - f.left > 0 ? ` · ${f.cards - f.left} named` : ""}`;
      const box = p.passes.box;
      return box.frames > box.done && box.sets.length ? "Every box so far has its answer" : "Done · every box has its answer";
    }
    if (id === "verify") {
      if (!f.faces) return "No centers yet";
      return f.named >= f.faces ? `Done · ${n(f.faces, "card")} named` : `${f.named} of ${n(f.faces, "card")} named`;
    }
    if (!f.frames) return "No frames";
    const part = f.sets.length < p.recordings ? ` · ${f.sets.length} of ${n(p.recordings, "recording")}` : "";
    return `${f.done >= f.frames ? "Done · " : ""}${f.done} of ${n(f.frames, "frame")}${part}`;
  }

  // Which passes come first for what the labels are for, and in which order. Training: Boxes and
  // Names, then the boxes for training. Development and Test: the census of centers and Check
  // names (fast, and how this project's own figures were measured, together with seeded points),
  // then the manifest; Points is the other measuring protocol. A project whose labels were made
  // another way goes on that way. → {main, protocols, others, why, protocolsWhy, exportKind}
  function lists(p) {
    const role = p.role.value;
    const measuring = role === "development" || role === "test";
    const flow = p.flow || (measuring ? "center" : "box");
    if (!measuring) {
      return { main: ["box", "identify"], protocols: [], others: ["corner", "training", "point", "center", "verify"], exportKind: "coco",
        why: "", protocolsWhy: "" };
    }
    const why = "Centers, then Check names: fast (one click a card, then one name each), and the way this project's own figures were measured, as a census of card centers together with seeded random points.";
    const pointsWhy = "Points: random points per frame, each answered. Together with a census they give the off-card positions a measurement needs.";
    if (flow === "point") {
      return { main: ["point"], protocols: ["center", "verify"], others: ["box", "identify", "corner"], exportKind: "manifest",
        why: "These labels were made with Points: each frame's random points, answered.", protocolsWhy: why };
    }
    if (BOX_FLOW.includes(flow)) {
      return { main: ["box", "identify"], protocols: ["center", "verify", "point"], others: ["corner"], exportKind: "manifest",
        why: "These labels were made as boxes, which measure a model too.", protocolsWhy: `${why} ${pointsWhy}` };
    }
    return { main: ["center", "verify"], protocols: ["point"], others: ["box", "identify", "corner"], exportKind: "manifest", why, protocolsWhy: pointsWhy };
  }

  // One row of the list. kind: "open" (the pass the tool runs), "start" (the page starts it),
  // "switch" (the page switches the running tool to it), "disabled" (with the reason).
  function passRow(p, id) {
    const f = p.passes[id];
    const row = { id, name: passName(id), meaning: MEANING[id], status: status(p, id), button: null, kind: "disabled", reason: "", note: "" };
    if (f.running) { row.kind = "open"; row.button = "Open"; return row; }
    if (!f.allowed) { row.reason = reason(p, id); return row; }
    row.button = ACTION[f.action] || "Start";
    row.kind = p.server.configured ? "switch" : "start";
    if (id === "point") row.note = "12 points a frame, placed by seed 1, fixed at the first save. Other numbers: the command for scripts, with --k and --seed.";
    if (id === "verify") row.note = "Every center is named from the frame, with no model's proposal shown. To show the proposals of models beside each center, start this pass with flags (the guide says how).";
    return row;
  }

  // The start of a pass as one command, for scripts: small, under its row.
  function scripts(p, id) {
    const c = p.passes[id] && p.passes[id].command;
    return c ? { lead: "For scripts, the same pass as one command:", text: c.text, parts: c.parts } : null;
  }

  // The "Start from the model's boxes" option under Boxes: Training only.
  function modelBoxes(p) {
    if (p.role.value !== "training") return null;
    const m = p.modelBoxes;
    const locked = p.server.configured && p.server.sameProject;
    if (!m.offered) {
      const at = p.locations.prefill;
      return { label: "Start from the model's boxes", checked: false, disabled: true,
        note: at.state === "unreadable" ? `The suggestions file cannot be used: ${at.problem} (${at.path}).`
          : at.path ? `No suggestions file was found (looked for ${at.path}).` : "The running pass was started without a suggestions file." };
    }
    return { label: "Start from the model's boxes", checked: m.on, disabled: locked, note: `${m.file}, ${p.locations.prefill.usual ? "found beside the sample" : "as given"}. You accept, correct or remove each box.` };
  }

  // ---- the export --------------------------------------------------------------------------------

  const SOURCE_TYPES = [
    { value: "physical/webcam", label: "A table filmed from above (physical/webcam)" },
    { value: "physical/tournament", label: "A tournament broadcast of a table (physical/tournament)" },
    { value: "physical/phone", label: "A phone recording of a table (physical/phone)" },
    { value: "physical", label: "Another recording of a table (physical)" },
    { value: "digital", label: "A digital client on screen (digital)" },
  ];

  // Each refusal of the manifest export (export.ManifestRefused: {code, where, text}) as one
  // sentence that says what to fill in. Problems of one kind and one recording are said once.
  function exportProblems(problems) {
    const out = [], seen = new Set();
    for (const pr of problems || []) {
      const set = String(pr.where || "").split("/")[0];
      const key = `${pr.code}#${["address", "time-size", "source-type", "bad-source-type"].includes(pr.code) ? set : pr.where}`;
      if (seen.has(key)) continue;
      seen.add(key);
      switch (pr.code) {
        case "credit": out.push("Fill in the credit: the name to show for these labels, 1 to 80 characters on one line."); break;
        case "source-type": case "bad-source-type": out.push(`Choose the kind of footage${set ? ` of ${set}` : ""} under “Kind of footage”, or tag it under About this recording while you label.`); break;
        case "address": out.push(`The sample does not say where ${set} is hosted. In sample.json, give that recording "platform" and "videoId"; for a recording of your own, "platform": "local" and a name of your choice (make_sample.py --local writes both).`); break;
        case "time-size": out.push(`${pr.where}: the frame needs its time in the video ("t", in seconds, in sample.json) and its size, which the tool records once the frame is labeled.`); break;
        case "card-id": out.push(`${pr.where}: a card id the manifest cannot hold (${pr.text.split(": ").slice(-1)[0]}). Correct it in the catalog, then name that card again.`); break;
        case "same-frame": out.push(`${pr.where}: two sets label the same frame. Export them one at a time, with the command for scripts and --set.`); break;
        case "no-asset": out.push(`${pr.where}: this set labels a recording the sample does not list. Put that recording back in sample.json, or move the set's folder out of the labels folder.`); break;
        case "group": case "group-conflict": out.push(`${pr.where}: ${pr.text}`); break;
        default: out.push(pr.text || "The export was refused.");
      }
    }
    return out;
  }

  // The Export row. form: what the page holds for it: {credit, sourceType, result (the answer of
  // the last export: {ok, out, frames, boxes, points, notes} or a refusal), confirm (the file to
  // replace, asked about)}. → {id, name, meaning, kind, status, note, needs: [{text, guide}],
  // fields: {credit, sourceType} (what to ask for), button, disabled, result, confirm, scripts}
  function exportRow(p, form) {
    const fm = form || {};
    const kind = (p.export && p.export.kind) || "coco";
    const e = p.export[kind];
    const file = kind === "coco" ? "boxes.json" : "the manifest";
    const what = kind === "coco" ? "boxes" : "labels";
    const row = { id: "export", name: "Export", kind, status: "", note: "", needs: [], fields: null, button: `Write ${file}`, disabled: false, result: null, confirm: null,
      meaning: kind === "coco" ? "Write the boxes to boxes.json, the file a model is trained on to find cards."
        : "Write the labels to a manifest, the file cardvision evaluate reads to measure a model.",
      scripts: { lead: "For scripts, the same export as one command:", text: e.text, parts: e.parts } };
    if (!e.included) {
      row.status = "Nothing to export yet";
      row.note = `The export writes the frames that are done${kind === "manifest" ? " (a census with at least one center)" : ""}, and none is yet.`;
      row.disabled = true;
    } else {
      row.status = `${n(e.included, "frame")} ready`;
      row.note = `It writes ${n(e.included, "frame")} that ${e.included === 1 ? "is" : "are"} done, from ${n(e.recordings, "recording")}, to ${e.out}.`
        + (e.leftOut ? ` ${n(e.leftOut, "frame")} of this sample ${e.leftOut === 1 ? "is" : "are"} not done and ${e.leftOut === 1 ? "is" : "are"} left out.` : " No frame of this sample is left out.");
    }
    if (kind === "manifest") {
      if ((e.unaddressed || []).length) {
        row.needs.push({ text: `Missing: where ${list(e.unaddressed)} ${e.unaddressed.length === 1 ? "is" : "are"} hosted. A manifest names videos, so sample.json must give each recording "platform" and "videoId" (for a recording of your own: "platform": "local").`,
          guide: guide("own") });
        row.disabled = true;
      }
      if (e.unanswered) row.needs.push({ text: `${n(e.unanswered, "face")} of the census ${e.unanswered === 1 ? "has" : "have"} no checked name yet: written as unlabeled. Check names first to name ${e.unanswered === 1 ? "it" : "them"}.`, guide: null });
      row.fields = { credit: { label: "Credit: the name to show for these labels", value: fm.credit || "" },
        sourceType: (e.untagged || []).length ? { label: `Kind of footage (${list(e.untagged)} ${e.untagged.length === 1 ? "has" : "have"} none recorded)`, value: fm.sourceType || "", options: SOURCE_TYPES } : null };
      if (!row.disabled && !String(fm.credit || "").trim()) row.disabled = "credit";
      else if (!row.disabled && row.fields.sourceType && !fm.sourceType) row.disabled = "source";
    }
    const r = fm.result;
    if (r && r.ok) {
      row.result = { ok: true, text: `Wrote ${r.out}: ${n(r.frames, kind === "coco" ? "frame" : "row")}, ${kind === "coco" ? `${n(r.boxes, "box", "boxes")}${r.ignored ? ` (${r.ignored} ignored)` : ""}` : `${r.boxes ? `${n(r.boxes, "box", "boxes")} and ` : ""}${n(r.points || 0, "point")}`}.`,
        notes: r.notes || [], check: kind === "manifest" ? `Check it: cardvision validate-manifest ${r.out}${r.ownRecording ? "" : " --contribution"}` : "" };
    } else if (r && r.problem === "exists") {
      row.confirm = { text: `${r.out} is there already, from an earlier export. Replace it with what the labels hold now?`, yes: "Replace it", no: "Keep it" };
    } else if (r && r.problem === "refused") {
      row.result = { ok: false, text: "The manifest was not written:", items: exportProblems(r.problems) };
    } else if (r && r.problem) {
      row.result = { ok: false, text: r.problem === "nothing" ? `Nothing to write: no ${what} are done yet.` : r.problem === "other-run" ? "The tool was switched meanwhile: reload this page." : `The export failed: ${r.detail || r.problem}.`, items: [] };
    }
    return row;
  }

  // The frames removed from the project (/api/project's `removed`), each with a Restore button.
  // → null when none was removed; else {title, lead, items: [{assetId, name, text, button}]}
  function removed(p) {
    const frames = (p.removed && p.removed.frames) || [];
    if (!frames.length) return null;
    return { title: `Removed frames (${frames.length})`,
      lead: "These frames are skipped in every pass, in the counts and in the export. Their labels were kept: Restore brings a frame back as it was.",
      items: frames.map((e) => ({ assetId: e.assetId, name: e.name, text: U.removedText(e), button: "Restore" })) };
  }

  // report: /api/project → everything the project screen shows. form: the Export row's (exportRow).
  function project(p, form) {
    const lab = p.locations.labels;
    const L = lists(p);
    const rows = L.main.map((id) => passRow(p, id));
    const protocols = L.protocols.map((id) => passRow(p, id));
    const others = L.others.map((id) => passRow(p, id));
    const nx = next(p);
    const all = [...rows, ...protocols, ...others];
    const running = all.find((r) => r.kind === "open");
    const exportRowView = exportRow(p, form);
    // The one primary action. On a tool with no pass yet: the next pass, which the page starts. On
    // a tool that runs a pass: that pass ("Open"), unless it is finished and something else comes
    // next; then the switch to it. When all that is left is the export: writing it.
    const finished = running && p.passes[running.id].action === "review" && nx.pass && nx.pass !== running.id;
    let primary = null;
    if (running && !finished) primary = running.id;
    else if (nx.pass === "export") primary = exportRowView.disabled ? null : "export";
    else if (nx.startable) primary = nx.pass;
    const name = (id) => passName(id);
    let guideText;
    if (primary && primary === (running && running.id)) guideText = `The tool is running the ${running.name} pass. {{Enter}} opens it. Any other pass: its row's button switches the tool to it, here.`;
    else if (!p.role.value) guideText = "Say what these labels will be used for. The passes below can be started after that.";
    else if (nx.pass === "export" && primary === "export") guideText = `Every frame is done and nothing is left to name. Next: write ${exportRowView.kind === "coco" ? "boxes.json" : "the manifest"}. {{Enter}} writes it.`;
    else if (nx.pass === "export" && exportRowView.disabled === "credit") guideText = "Every frame is done. Next: the export. Fill in the credit under Export, then {{Enter}} writes the manifest.";
    else if (nx.pass === "export" && exportRowView.disabled === "source") guideText = "Every frame is done. Next: the export. Choose the kind of footage under Export, then {{Enter}} writes the manifest.";
    else if (nx.pass === "export" && ((p.export[exportRowView.kind] || {}).unaddressed || []).length) guideText = "Every frame is done. Next: the export, once sample.json says where each recording is hosted (the Export row says how).";
    else if (nx.pass === "export") guideText = "Every frame is done. Next: the export, which needs what its row says first.";
    else if (running && primary) guideText = `${running.name} is done. Next: ${name(primary)}. {{Enter}} switches the tool to it, here: no new start.`;
    else if (running) guideText = `The tool is running the ${running.name} pass, which is done. Its row opens it; any other row's button switches the tool to that pass.`;
    else if (p.server.configured && primary) guideText = `The tool is running a pass of another project. {{Enter}} switches it to ${name(primary)} of ${p.name}.`;
    else if (primary) guideText = `Next: ${name(primary)}. {{Enter}} ${p.passes[primary].action === "start" ? "starts" : "goes on with"} it.`;
    else guideText = "No pass can be started as things are: each row says what it needs.";
    return {
      guide: guideText, title: p.name, folder: p.folder, summary: size(p),
      labels: `Labels are written to ${lab.path}${lab.state === "will-create" ? " (the folder is made at the first save)" : ""}`,
      warnings: warnings(p), places: places(p), placesLocked: p.server.configured && p.server.sameProject,
      placesNote: p.server.configured && p.server.sameProject
        ? "The tool is running a pass with these locations. To change them, stop the tool (Ctrl+C in its terminal), start it again and open this project."
        : "Each file is looked for beside the sample first. Give another path where yours is somewhere else; a path that does not start with / is read from the folder the tool was started in.",
      purpose: purpose(p), rows, why: L.why, protocols, protocolsWhy: L.protocolsWhy, others, modelBoxes: modelBoxes(p),
      exportRow: exportRowView, primary, measuringGuide: L.why ? guide("measuring") : null, removed: removed(p),
    };
  }

  // ---- arriving, and what is remembered ----------------------------------------------------------

  // What arriving at an address does. where: {pathname, sample (the address's `sample`, or
  // null)}; facts: /api/hub → {screen: "start" | "project", path, ask}. `ask` is the one request
  // the page sends for a project's screen, and it is always the report, which writes nothing.
  // Arriving is not a person opening a project: a link on any site, a redirect or a bookmark
  // brings a browser to an address, so nothing is remembered because of it.
  function arrival(where, facts) {
    const w = where || {};
    const own = facts && facts.server ? facts.server.project : null;
    const sample = w.sample || null;
    if (w.pathname === "/start" || (w.pathname !== "/project" && !own) || (w.pathname === "/project" && !sample && !own)) {
      return { screen: "start", path: null, ask: null };
    }
    return { screen: "project", path: sample, ask: "/api/project" };
  }

  // The acts of a person on these screens after which the tool remembers the project: opening
  // it from a row of the start screen, "Check files" on a path that is a project, and on the
  // project's screen choosing what the labels are for, the model's boxes, or a location.
  // (Starting a pass is remembered by the server, once the pass runs.) Nothing else is.
  const REMEMBERED = ["open-row", "check-files", "purpose", "model-boxes", "locations"];
  function remembers(act) { return REMEMBERED.includes(act); }

  // Whether Enter, pressed with the focus on this kind of element (keyroute's focusKind), presses
  // the primary button, as the instruction line says it does. On nothing, and on a choice just made
  // (the purpose's radio buttons, the model boxes' tick box), where Enter does nothing by itself.
  // Never on a text field, a button, a link or a disclosure: Enter acts on that control.
  function enterPressesPrimary(focus) { return focus === "none" || focus === "checkbox"; }

  const api = { ROLES, MEANING, GUIDE, SOURCE_TYPES, guide, start, video, problem, gone, next, project, lists, places, warnings, purpose, passRow, scripts, status, reason,
    modelBoxes, exportRow, exportProblems, removed, passName, roleName, arrival, remembers, REMEMBERED, enterPressesPrimary };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.LabelerHubState = api;
})(typeof window !== "undefined" ? window : globalThis);
