// SPDX-License-Identifier: Apache-2.0
// The start screen and the project screen of the labeling tool. Vanilla JS, no build step.
// Talks only to serve.py.
//
//   /start                 the start screen: three paths (label my frames, learn on a toy game,
//                          I have a video), the one that fits what was found shown first
//   /project?sample=PATH   one project: where its files are, what its labels are for, its passes
//                          in the order of that purpose, and the export
//   /                      one of the two, until a pass runs (then the labeling page is here)
//
// Every word on these screens comes from hubstate.js (test_hubstate.js), from the facts the
// server sends; this file draws them and sends the few requests a person can cause: check a
// path, remember a project, make the toy example, start a pass, switch the running tool to
// another pass, write the export. Arriving at an address causes none of them: the page then asks
// only for the report, which writes nothing (hubstate.js, `arrival` and `remembers`). They are
// ordinary pages: Tab follows the reading order, the one primary button has the focus on
// arrival, and Enter activates it. Who takes a key is decided as on the labeling pages, by
// keyroute.js.
(function (root) {
  "use strict";
  const H = root.LabelerHubState;
  const U = root.LabelerUiState;
  const SG = root.LabelerSaveGate; // why a switch was refused
  const KR = root.LabelerKeyRoute;
  const HEARTBEAT_MS = 3000;
  const $ = (id) => document.getElementById(id);

  const S = {
    screen: null, // "start" | "project"
    facts: null, // /api/hub
    report: null, // /api/project
    path: null, // the sample this page shows (project screen); null: the server's own project
    given: {}, // locations changed on this page
    role: null, modelBoxes: null, // what was chosen on this page
    placesOpen: false, othersOpen: false, connected: true, busy: false,
    note: null, // {pass, text}: what the server said when a start or a switch was refused
    check: null, // {text, fix}: the answer to "Check files"
    path3: null, // the path of the start screen that is shown (null: the one that fits)
    run: null, // the run of the tool this page knows (a switch from another tab changes it)
    exportForm: { credit: "", sourceType: "", result: null }, // the Export row's fields and its last answer
  };
  const CREDIT_PREF = "labeler-export-credit"; // the credit typed last, kept in this browser only

  // ---- small helpers ----------------------------------------------------------------------------

  // el("p", {class: "x", text: "…"}, child, …): text is always set as text, never as markup
  function el(tag, attrs, ...children) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "text") e.textContent = v;
      else if (k === "class") e.className = v;
      else if (k === "on") for (const [name, fn] of Object.entries(v)) e.addEventListener(name, fn);
      else if (k === "data") for (const [name, value] of Object.entries(v)) e.dataset[name] = value;
      else if (v === true) e.setAttribute(k, "");
      else e.setAttribute(k, v);
    }
    for (const c of children.flat(Infinity)) if (c) e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    return e;
  }

  function keyText(target, text) {
    target.textContent = "";
    for (const p of U.parts(text)) target.appendChild(p.key ? el("kbd", { text: p.key }) : document.createTextNode(p.text));
  }

  // A button: its words, the key it stands for (drawn on it), and what it does.
  function button(label, opts, onClick) {
    const o = opts || {};
    return el("button", { type: "button", class: o.primary ? "primary" : null, disabled: !!o.disabled, id: o.id, title: o.title,
      data: o.focus ? { focus: o.focus } : null, "aria-expanded": o.expanded === undefined ? null : String(o.expanded), on: { click: onClick } },
    el("span", { text: label }), o.primary ? el("kbd", { text: "Enter" }) : null);
  }

  // A command, ready to copy; the parts a person still has to fill in are marked.
  function command(cmd, small, primary) {
    const pre = el("pre");
    if (cmd.parts) {
      cmd.parts.forEach((p, i) => { if (i) pre.appendChild(document.createTextNode(" ")); pre.appendChild(p.fill ? el("mark", { text: p.t }) : document.createTextNode(p.t)); });
    } else pre.textContent = cmd.text;
    const b = el("button", { type: "button", class: primary ? "primary" : null, data: primary ? { focus: "copy" } : null, on: { click: async () => {
      const label = b.firstChild;
      try { await navigator.clipboard.writeText(cmd.text); label.textContent = "Copied"; }
      catch (e) { // no clipboard here: the command is selected instead, ready for the copy key
        const r = document.createRange(); r.selectNodeContents(pre);
        const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(r);
        label.textContent = "Selected: copy it now";
      }
    } } }, el("span", { text: primary ? "Copy the command" : "Copy" }), primary ? el("kbd", { text: "Enter" }) : null);
    return el("div", { class: small ? "cmd small" : "cmd" }, pre, b);
  }

  // A link to the guide (README.md), at the place that says more.
  function guideLink(g) {
    return g ? el("a", { class: "guide", href: g.href, target: "_blank", rel: "noopener", text: g.label }) : null;
  }

  // What is missing, and the one step back: {text, step, guide}.
  function missingBox(m, cls) {
    return el("div", { class: cls || "warn" }, el("span", { text: m.text }), m.step ? ` ${m.step}` : "", m.guide ? " " : "", guideLink(m.guide));
  }

  async function get(url) {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) throw new Error(`${url}: ${r.status}`);
    return r.json();
  }

  // Every request that changes or configures anything is a POST with a JSON body.
  async function post(url, body) {
    const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
    let j = {};
    try { j = await r.json(); } catch (e) { j = { ok: false, problem: "no-answer" }; }
    return j;
  }

  function lsGet(k) { try { return root.localStorage.getItem(k); } catch (e) { return null; } }
  function lsSet(k, v) { try { root.localStorage.setItem(k, v); } catch (e) { /* storage blocked */ } }

  function projectUrl(sample) { return `/project?sample=${encodeURIComponent(sample)}`; }
  function go(url) { root.location.href = url; }

  function asked() {
    return { path: S.path || "", role: S.role, modelBoxes: S.modelBoxes, locations: S.given };
  }

  // ---- drawing: keep the focus where it was ------------------------------------------------------

  function draw(node, guide, focusFirst) {
    const had = document.activeElement && document.activeElement.dataset ? document.activeElement.dataset.focus : null;
    const main = $("hub");
    const top = main.scrollTop;
    main.textContent = "";
    main.appendChild(node);
    main.scrollTop = top;
    keyText($("guideText"), guide);
    const again = had && Array.from(main.querySelectorAll("[data-focus]")).find((x) => x.dataset.focus === had);
    const first = again || (focusFirst && (main.querySelector("button.primary:not(:disabled)") || main.querySelector("[data-first]")));
    if (first) first.focus({ preventScroll: !!again });
  }

  // The control drawn with this `data-focus` name, or null.
  function byFocus(name) { return Array.from($("hub").querySelectorAll("[data-focus]")).find((x) => x.dataset.focus === name) || null; }

  function panel(cls, ...children) { return el("section", { class: `panel ${cls || ""}` }, ...children); }

  // ---- the start screen -------------------------------------------------------------------------

  // A person opens a project from the start screen (a row, Resume, a checked path): that is
  // remembered, here, by the click. The project's screen then loads like any address.
  async function openProject(sample, act) {
    if (H.remembers(act)) await post("/api/remember", { path: sample });
    go(projectUrl(sample));
  }

  async function openFromStart(sample, pass) {
    if (S.busy) return;
    if (!pass) return openProject(sample, "open-row");
    S.busy = true;
    const configured = S.facts.server && S.facts.server.configured;
    const r = configured ? await post("/api/switch", { run: S.run, unsaved: 0, path: sample, pass }) : await post("/api/open", { path: sample, pass });
    S.busy = false;
    go(r.ok ? "/" : projectUrl(sample)); // a start or a switch that was refused: its screen says why
  }

  async function checkFiles() {
    const text = $("pathInput").value;
    const r = await post("/api/project", { path: text });
    if (r.ok) return openProject(r.sample, "check-files");
    S.check = H.problem(r, { startDir: S.facts.startDir, sampleTool: S.facts.tools.sample, python: S.facts.tools.python });
    S.typed = text;
    renderStart(false);
    $("pathInput").focus();
  }

  async function makeToy() {
    if (S.busy) return;
    S.busy = true;
    const r = await post("/api/toy", { path: S.facts.toy.path });
    S.busy = false;
    if (r.ok) return go(projectUrl(r.sample));
    S.facts = await get("/api/hub");
    S.toyNote = r.problem === "moved" ? "That folder was taken in the meantime. The toy example would now go to the folder named here: press the button again."
      : `The toy example could not be written${r.detail ? ` (${r.detail})` : ""}.`;
    renderStart(false);
  }

  async function forget(sample) {
    await post("/api/forget", { sample });
    S.facts = await get("/api/hub");
    renderStart(false);
  }

  function sampleRow(e, primary) {
    const what = el("div", { class: "what" }, el("span", { class: "name", text: e.name }),
      e.gone ? el("span", { class: "bad", text: e.text }) : el("span", { class: "path", text: e.folder }),
      e.gone ? null : el("span", { class: "muted", text: e.text }));
    const b = e.gone ? button("Remove from this list", { focus: `forget:${e.sample}` }, () => forget(e.sample))
      : button(`Open ${e.name}`, { primary, focus: `open:${e.sample}` }, () => openProject(e.sample, "open-row"));
    return el("li", null, what, b);
  }

  // The start screen: three paths side by side, each with its promise and its own button; the
  // one shown below is the one that fits what was found, until another is chosen.
  function renderStart(arrive) {
    const v = H.start(S.facts);
    $("homeLink").hidden = true;
    $("passName").textContent = "Card labeler";
    document.title = "Card labeler";
    const col = el("div", { class: "col" }, el("h1", { text: v.title }), el("p", { class: "lead", text: v.lead }));
    if (v.running) {
      col.appendChild(panel("main", el("h2", { text: "Running now" }), el("p", { text: v.running.text }),
        el("div", { class: "row" }, button(v.running.primary, { primary: true, focus: "running" }, () => go("/")),
          button(v.running.secondary, { focus: "running-project" }, () => go("/project")))));
    }
    const shown = S.path3 || v.open;
    // a path chosen: its own first action takes the focus
    const FIRST = { frames: ["resume", "path"], toy: ["toy"], video: ["copy"] };
    const choose = (id) => {
      S.path3 = id;
      renderStart(false);
      const b = FIRST[id].map((f) => byFocus(f)).find((x) => x && !x.disabled);
      if (b) b.focus();
    };
    col.appendChild(el("div", { class: "paths", role: "group", "aria-label": "Three ways to start" }, v.paths.map((x, i) =>
      el("button", { type: "button", class: `route${x.id === shown ? " chosen" : ""}${x.fits ? " fits" : ""}`, "aria-pressed": String(x.id === shown),
        data: { focus: `path:${x.id}` }, on: { click: () => choose(x.id) } },
      el("span", { class: "step", text: String(i + 1), "aria-hidden": "true" }), el("span", { class: "title", text: x.title }),
      el("span", { class: "promise", text: x.promise }), x.fits ? el("span", { class: "fitsTag", text: "Fits what was found here" }) : null))));
    const body = el("div", { class: "pathBody", data: { pathBody: shown } });
    if (shown === "frames") body.append(...framesPath(v));
    else if (shown === "toy") body.append(...toyPath(v));
    else body.append(...videoPath(v));
    col.appendChild(body);
    draw(col, v.guide, arrive);
  }

  function framesPath(v) {
    const out = [];
    if (v.resume) {
      const r = v.resume;
      out.push(panel(v.primary === "resume" ? "main" : "", el("h2", { text: "Resume" }),
        el("div", { class: "what" }, el("p", null, el("b", { text: r.name }), ` · ${r.text}`), el("p", { class: "path", text: r.folder }), el("p", { text: r.next })),
        el("div", { class: "row" }, button(r.primary, { primary: v.primary === "resume", focus: "resume" }, () => openFromStart(r.sample, r.pass)),
          r.secondary ? button(r.secondary, { focus: "resume-project" }, () => openProject(r.sample, "open-row")) : null)));
    }
    if (v.before.length) out.push(panel("", el("h2", { text: "Opened before" }), el("ul", { class: "rows" }, v.before.map((e) => sampleRow(e, false)))));
    out.push(panel("", el("h2", { text: "Samples found here" }),
      v.found.length ? el("ul", { class: "rows" }, v.found.map((e) => sampleRow(e, v.primary === "found"))) : null,
      v.foundNote ? el("p", { class: "muted", text: v.foundNote }) : null,
      v.none ? missingBox(v.none, "need") : null,
      v.none ? command({ text: `${S.facts.tools.python} ${S.facts.tools.sample} --out sample.json --frames-root FOLDER_OF_FRAME_FOLDERS --n 8`,
        parts: [{ t: `${S.facts.tools.python} ${S.facts.tools.sample} --out`, fill: false }, { t: "sample.json", fill: true }, { t: "--frames-root", fill: false },
          { t: "FOLDER_OF_FRAME_FOLDERS", fill: true }, { t: "--n 8", fill: false }] }, true) : null));
    const input = el("input", { id: "pathInput", type: "text", autocomplete: "off", spellcheck: "false", value: S.typed || "",
      placeholder: "for example toy, or work/sample.json", "aria-describedby": "pathHelp", data: { focus: "path", first: "1" },
      on: { keydown: (ev) => { if (ev.key === "Enter") { ev.preventDefault(); checkFiles(); } } } });
    out.push(panel("", el("h2", { text: "Open a sample by its path" }),
      el("label", { class: "over", for: "pathInput", text: "The path of a sample file, or of the folder that holds a sample.json" }),
      el("div", { class: "field" }, input, button("Check files", { focus: "check" }, checkFiles)),
      el("p", { id: "pathHelp", class: "muted", text: `The tool checks the path and then shows what it found. A path that does not start with / is read from ${S.facts.startDir}.` }),
      S.check ? el("div", { class: "warn", role: "alert" }, S.check.text, S.check.fix ? ` ${S.check.fix}` : "") : null));
    return out;
  }

  function toyPath(v) {
    return [panel(v.primary === "toy" ? "main" : "", el("h2", { text: "Learn on a toy game" }), el("p", { text: v.toy.text }),
      el("p", { class: v.toy.disabled ? "bad" : "path", text: v.toy.where }),
      v.toy.missing ? missingBox(v.toy.missing, "need") : null,
      S.toyNote ? el("div", { class: "warn", role: "alert", text: S.toyNote }) : null,
      el("div", { class: "row" }, button(v.toy.button, { primary: v.primary === "toy" || !v.toy.disabled, disabled: v.toy.disabled, focus: "toy" }, makeToy)))];
  }

  function videoPath(v) {
    const x = v.video;
    return [panel("", el("h2", { text: x.summary }), el("p", { text: x.needs }), el("p", { text: x.text }),
      x.commands.map((c, i) => command(c, false, i === 0)), el("p", { class: "muted", text: x.after }), el("p", null, guideLink(x.guide)))];
  }

  // ---- the project screen -----------------------------------------------------------------------

  // A person chose something on this project's screen (`act`: what, as hubstate.js names it):
  // the server remembers the project with it, and answers with the project as it now stands.
  // Anything that is not such an act only reads the report.
  async function remember(act) {
    if (!H.remembers(act)) { S.report = await post("/api/project", asked()); return; }
    const r = await post("/api/remember", asked());
    if (r.ok) { S.report = r.project; S.remembered = r.remembered; } else S.report = r;
  }

  // Start a pass on a tool that runs none, or switch the running tool to it (no new start: the
  // server refuses while a save is in progress or a labeling page holds unsaved changes).
  async function startPass(id) {
    if (S.busy) return;
    S.busy = true; S.note = null;
    const switching = S.report.server && S.report.server.configured;
    const r = await post(switching ? "/api/switch" : "/api/open", { ...asked(), run: S.run, unsaved: 0, path: S.report.sample, pass: id });
    S.busy = false;
    if (r.ok) return go("/");
    if (switching) S.note = { pass: id, text: SG.switchText(r, H.passName(id)) };
    else if (r.problem === "refused") S.note = { pass: id, text: `The tool could not start this pass: ${r.detail}` };
    if (r.problem === "other-run") await refreshRun();
    S.report = await post("/api/project", asked()); // what changed meanwhile shows in the rows
    renderProject(false);
  }

  function scriptsBox(x) {
    return x ? el("div", { class: "small" }, el("p", { text: x.lead }), command({ text: x.text, parts: x.parts }, true)) : null;
  }

  function passItem(v, row, num) {
    const p = S.report;
    const primary = v.primary === row.id;
    const li = el("li", { class: `pass${primary ? " next" : ""}`, data: { pass: row.id } });
    li.appendChild(el("div", { class: "num", text: num || "·", "aria-hidden": "true" }));
    const body = el("div", { class: "body" },
      el("div", { class: "title" }, el("span", { class: "name", text: row.name }), el("span", { class: `status${row.status.startsWith("Done") ? " done" : ""}`, text: row.status })),
      el("p", { class: "meaning", text: row.meaning }));
    li.appendChild(body);
    const act = el("div", { class: "do" });
    li.appendChild(act);
    const more = el("div", { class: "more" });
    if (row.id === "box" && v.modelBoxes) {
      const m = v.modelBoxes;
      const box = el("input", { type: "checkbox", checked: m.checked, disabled: m.disabled, data: { focus: "modelBoxes" }, on: { change: async (ev) => {
        S.modelBoxes = ev.target.checked; await remember("model-boxes"); renderProject(false);
      } } });
      body.appendChild(el("label", { class: `opt${m.disabled ? " disabled" : ""}` }, box, el("span", null, el("b", { text: m.label }), ` · ${m.note}`)));
    }
    if (row.note) body.appendChild(el("p", { class: "meaning", text: row.note }));
    if (row.kind === "open") act.appendChild(button("Open", { primary, focus: `do:${row.id}` }, () => go("/")));
    else if (row.kind === "start" || row.kind === "switch") {
      const label = primary && row.kind === "switch" ? `Continue to ${row.name}` : row.button;
      act.appendChild(button(label, { primary, focus: `do:${row.id}`, title: row.kind === "switch" ? "The running tool switches to this pass: no new start" : null }, () => startPass(row.id)));
      if (primary) more.appendChild(scriptsBox(H.scripts(p, row.id)));
    } else {
      act.appendChild(button("Start", { disabled: true, title: row.reason }, () => {}));
      body.appendChild(el("p", { class: "reason", text: row.reason }));
    }
    if (S.note && S.note.pass === row.id) more.appendChild(el("div", { class: "warn", role: "alert", text: S.note.text }));
    if (more.childNodes.length) li.appendChild(more);
    return li;
  }

  // The Export row: what it writes, what it asks for, and what it answered.
  function exportItem(v, num) {
    const x = v.exportRow;
    const f = S.exportForm;
    const li = el("li", { class: `pass${v.primary === "export" ? " next" : ""}`, data: { pass: "export" } }, el("div", { class: "num", text: num, "aria-hidden": "true" }));
    li.appendChild(el("div", { class: "body" }, el("div", { class: "title" }, el("span", { class: "name", text: x.name }), el("span", { class: "status", text: x.status })),
      el("p", { class: "meaning", text: x.meaning })));
    const act = el("div", { class: "do" });
    li.appendChild(act);
    act.appendChild(button(x.button, { primary: v.primary === "export" || (!x.disabled && !!f.touched), disabled: !!x.disabled, focus: "do:export" }, () => writeExport(false)));
    const more = el("div", { class: "more" }, el("p", { text: x.note }));
    for (const need of x.needs) more.appendChild(missingBox(need, "need"));
    if (x.fields) {
      const credit = el("input", { id: "exportCredit", type: "text", autocomplete: "off", spellcheck: "false", value: x.fields.credit.value, maxlength: "80",
        data: { focus: "export-credit" }, on: { input: (ev) => { f.credit = ev.target.value; f.touched = true; lsSet(CREDIT_PREF, f.credit); followForm(); },
          change: () => { f.result = null; renderProject(false); },
          keydown: (ev) => { if (ev.key === "Enter" && !x.disabled) { ev.preventDefault(); writeExport(false); } } } });
      more.appendChild(el("div", { class: "form" }, el("label", { class: "over", for: "exportCredit", text: x.fields.credit.label }), credit));
      if (x.fields.sourceType) {
        const sel = el("select", { id: "exportSource", data: { focus: "export-source" }, on: { change: (ev) => { f.sourceType = ev.target.value; f.touched = true; f.result = null; renderProject(false); } } },
          el("option", { value: "", text: "Choose…" }), x.fields.sourceType.options.map((o) => el("option", { value: o.value, text: o.label, selected: o.value === f.sourceType })));
        more.appendChild(el("div", { class: "form" }, el("label", { class: "over", for: "exportSource", text: x.fields.sourceType.label }), sel));
      }
    }
    if (x.confirm) {
      more.appendChild(el("div", { class: "warn", role: "alert" }, x.confirm.text, " ",
        button(x.confirm.yes, { focus: "export-replace" }, () => writeExport(true)), " ", button(x.confirm.no, { focus: "export-keep" }, () => { f.result = null; renderProject(false); })));
    }
    if (x.result && x.result.ok) {
      more.appendChild(el("div", { class: "exportDone", role: "status" }, el("b", { text: x.result.text }), x.result.notes.map((t) => el("p", { class: "muted", text: t })),
        x.result.check ? el("p", { class: "muted", text: x.result.check }) : null));
    } else if (x.result) {
      more.appendChild(el("div", { class: "warn", role: "alert" }, el("p", { text: x.result.text }), x.result.items.length ? el("ul", null, x.result.items.map((t) => el("li", { text: t }))) : null));
    }
    more.appendChild(scriptsBox(x.scripts));
    li.appendChild(more);
    return li;
  }

  // While the credit is typed: the button and the instruction line follow, nothing is redrawn.
  function followForm() {
    const v = H.project(S.report, S.exportForm);
    const b = byFocus("do:export");
    if (b) { b.disabled = !!v.exportRow.disabled; b.classList.toggle("primary", !v.exportRow.disabled); }
    keyText($("guideText"), v.guide);
  }

  // The export, written by the server from the page (export.py's functions). An existing file is
  // replaced only after the person said so.
  async function writeExport(replace) {
    if (S.busy) return;
    const f = S.exportForm;
    S.busy = true;
    const r = await post("/api/export", { run: S.run, path: S.report.sample, kind: S.report.export.kind, credit: f.credit, sourceType: f.sourceType, replace });
    S.busy = false;
    if (r.problem === "other-run") await refreshRun();
    f.result = r;
    S.report = await post("/api/project", asked());
    renderProject(false);
    const b = byFocus("export-replace");
    if (b) b.focus();
  }

  // A removed frame comes back, with its labels as they were. The server refuses while a
  // labeling page holds unsaved changes or a save is being written (as it refuses a switch).
  async function restoreFrame(item) {
    if (S.busy) return;
    S.busy = true;
    const r = await post("/api/restore-frame", { ...asked(), run: S.run, path: S.report.sample, assetId: item.assetId, name: item.name });
    S.busy = false;
    if (r.problem === "other-run") await refreshRun();
    S.restoreNote = r.ok ? null : r.problem === "busy"
      ? (r.pages ? "The frame was not restored: a labeling tab has changes that are not saved yet. Wait until its top bar says Saved, then press Restore again."
        : "The frame was not restored: a save is being written. Press Restore again in a moment.")
      : r.problem === "other-run" ? "The tool was switched meanwhile: press Restore again." : (r.error || "The frame was not restored.");
    S.report = await post("/api/project", asked());
    renderProject(false);
  }

  function removedPanel(x) {
    if (!x) return null;
    return panel("", el("h2", { text: x.title }), el("p", { class: "muted", text: x.lead }),
      S.restoreNote ? el("div", { class: "warn", role: "alert", text: S.restoreNote }) : null,
      el("ul", { class: "removed" }, x.items.map((it) => el("li", null, el("span", { text: it.text }), " ",
        button(it.button, { focus: `restore:${it.assetId}/${it.name}`, title: "Bring this frame back, with its labels as they were" }, () => restoreFrame(it))))));
  }

  async function applyPlaces() {
    const given = {};
    for (const input of document.querySelectorAll("#places input[data-key]")) {
      const text = input.value.trim();
      if (text && text !== input.dataset.was) given[input.dataset.key] = text;
      else if (S.given[input.dataset.key] && text) given[input.dataset.key] = S.given[input.dataset.key];
    }
    S.given = given;
    await remember("locations");
    renderProject(false);
  }

  function renderProblem(r) {
    const text = H.problem(r, S.facts ? { startDir: S.facts.startDir, sampleTool: S.facts.tools.sample, python: S.facts.tools.python } : {});
    $("homeLink").hidden = false;
    $("passName").textContent = "No project";
    draw(el("div", { class: "col" }, panel("main", el("h1", { text: "This project cannot be opened" }), el("p", { text: text.text }),
      text.fix ? el("p", { text: text.fix }) : null, el("div", { class: "row" }, button("Back to the start screen", { primary: true }, () => go("/start"))))),
    "This project cannot be opened: the page says why.", true);
  }

  function renderProject(arrive) {
    const p = S.report;
    if (!p || !p.ok) return renderProblem(p || { problem: "none" });
    const v = H.project(p, S.exportForm);
    $("homeLink").hidden = false;
    $("passName").textContent = v.title;
    document.title = `${v.title} · Card labeler`;
    const col = el("div", { class: "col" });

    // the header: what the project is, and where its files are
    const about = el("div", { class: "about" }, el("h1", { text: v.title }), el("p", { class: "path", text: v.folder }),
      el("p", null, el("b", { text: v.summary })), el("p", { class: "path", text: v.labels }));
    const lockedPlaces = v.placesLocked;
    const places = el("details", { id: "places", open: S.placesOpen, on: { toggle: (ev) => { S.placesOpen = ev.target.open; } } },
      el("summary", { text: "Change locations", data: { focus: "places" } }),
      el("p", { class: "muted", text: v.placesNote }),
      el("div", { class: "places" }, v.places.map((x) => [
        el("label", { class: "k", for: `place-${x.key}`, text: x.label }),
        el("div", { class: "v" },
          el("input", { id: `place-${x.key}`, type: "text", autocomplete: "off", spellcheck: "false", value: x.path || "", disabled: lockedPlaces,
            data: { key: x.key, was: x.path || "", focus: `place:${x.key}` }, on: { keydown: (ev) => { if (ev.key === "Enter") { ev.preventDefault(); applyPlaces(); } } } }),
          el("span", { class: "state" }, el("span", { class: !x.ok ? "bad" : x.found ? "good" : "quiet", text: `${!x.ok ? "!" : x.found ? "✓" : "–"} ${x.state}` }), ` ${x.need} ${x.where ? `(${x.where})` : ""}`)),
      ])),
      lockedPlaces ? null : el("div", { class: "row" }, button("Check these locations", { focus: "places-apply" }, applyPlaces)));
    about.appendChild(places);
    col.appendChild(panel("head", p.frames.thumb ? el("img", { class: "thumb", alt: `One frame of ${v.title}`,
      src: `/api/thumb?key=${encodeURIComponent(p.frames.thumb.key)}` }) : null, about));
    for (const w of v.warnings) col.appendChild(missingBox(w));
    if (S.remembered && S.remembered.problem && !v.warnings.length) {
      col.appendChild(el("div", { class: "warn", text: "What you choose here cannot be remembered for next time: the labels folder cannot be written." }));
    }

    // what the labels are for
    const q = v.purpose;
    col.appendChild(panel("", el("h2", { id: "purposeQ", text: q.question }),
      q.settled ? el("p", null, el("b", { text: q.settled.label }), ` · ${q.settled.help}`)
        : el("fieldset", { "aria-labelledby": "purposeQ" }, q.options.map((o, i) => el("label", { class: "choice" },
          el("input", { type: "radio", name: "purpose", value: o.value, checked: o.checked, disabled: o.disabled, data: { focus: `role:${o.value}`, ...(i === 0 ? { first: "1" } : {}) },
            on: { change: async () => { S.role = o.value; await remember("purpose"); renderProject(false); } } }),
          el("span", { class: "label", text: o.label }), el("span", { class: "help", text: o.help })))),
      el("p", { class: "muted", text: q.note })));

    // the passes, in the order of the purpose, then the export
    col.appendChild(panel("", el("h2", { text: "The passes, in order" }),
      v.why ? el("p", { class: "muted" }, v.why, " ", guideLink(v.measuringGuide)) : null,
      el("ol", { class: "passes" }, v.rows.map((r, i) => passItem(v, r, String(i + 1))), exportItem(v, String(v.rows.length + 1))),
      v.protocols.length ? el("div", { class: "protocols" }, el("h3", { text: "The other measuring protocol" }), el("p", { class: "muted", text: v.protocolsWhy }),
        el("ul", { class: "passes" }, v.protocols.map((r) => passItem(v, r, "")))) : null,
      el("details", { id: "others", open: S.othersOpen || v.others.some((r) => r.kind === "open" || v.primary === r.id), on: { toggle: (ev) => { S.othersOpen = ev.target.open; } } },
        el("summary", { text: "Other ways to label", data: { focus: "others" } }),
        el("ul", { class: "passes" }, v.others.map((r) => passItem(v, r, ""))))));
    const gone = removedPanel(v.removed);
    if (gone) col.appendChild(gone);
    draw(col, v.guide, arrive);
  }

  // ---- the connection ---------------------------------------------------------------------------

  function renderConn() {
    const x = U.saveLabel({ connected: S.connected }).conn;
    const c = $("connState");
    c.className = `pill ${x.kind}`;
    c.textContent = "";
    c.append(el("span", { "aria-hidden": "true", text: x.mark }), el("span", { class: "word", text: x.text }));
    c.setAttribute("aria-label", x.text);
    const b = $("banner");
    b.hidden = S.connected;
    const restart = (S.report && S.report.restart) || (S.facts && S.facts.restart) || "python3 labeler/serve.py";
    b.textContent = S.connected ? "" : `The tool is not running, so nothing on this page works. Start it again in a terminal: ${restart}`;
  }

  // Every few seconds: does the server still answer? When it is back after a stop it may be
  // another run of the tool, so the page is loaded again.
  async function heartbeat() {
    let ok = false, now = null;
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), HEARTBEAT_MS - 500);
    try { const r = await fetch("/api/config?ping=1", { cache: "no-store", signal: ctl.signal }); ok = r.ok; if (ok) now = await r.json(); } catch (e) { ok = false; }
    clearTimeout(t);
    if (ok && now && S.run && now.run !== S.run && S.connected) {
      // the tool was switched to another pass (from another tab): this screen shows it, keeping
      // what was typed here
      S.run = now.run;
      await refreshRun();
      return;
    }
    if (ok === S.connected) return;
    S.connected = ok;
    if (ok) return root.location.reload();
    renderConn();
  }

  // What the tool runs now, after a switch: the facts and the report again, drawn in place.
  async function refreshRun() {
    S.facts = await get("/api/hub");
    S.run = S.facts.run;
    if (S.screen === "start") renderStart(false);
    else if (S.report) { S.report = await post("/api/project", asked()); renderProject(false); }
  }

  // ---- start ------------------------------------------------------------------------------------

  // Enter with the focus on no control, or on a choice just made, activates the primary button, as
  // the instruction line says.
  function onKey(ev) {
    if (ev.key !== "Enter" || ev.metaKey || ev.ctrlKey || ev.altKey) return;
    const a = document.activeElement;
    const focus = KR.focusKind(a && a !== document.body ? { tag: a.tagName, type: a.type, editable: a.isContentEditable } : null);
    if (!H.enterPressesPrimary(focus)) return;
    const b = $("hub").querySelector("button.primary:not(:disabled)");
    if (b) { ev.preventDefault(); b.click(); }
  }

  async function boot() {
    renderConn();
    const where = new URL(root.location.href);
    S.facts = await get("/api/hub");
    S.run = S.facts.run;
    S.exportForm.credit = lsGet(CREDIT_PREF) || "";
    // what arriving at this address does (hubstate.js): a screen, and at most the read-only report
    const a = H.arrival({ pathname: where.pathname, sample: where.searchParams.get("sample") }, S.facts);
    S.screen = a.screen;
    if (a.screen === "start") renderStart(true);
    else {
      S.path = a.path;
      S.report = await post(a.ask, asked());
      renderProject(true);
    }
    document.addEventListener("keydown", onKey);
    setInterval(heartbeat, HEARTBEAT_MS);
    // a page brought back by the browser's Back button may be an old one: load it again
    root.addEventListener("pageshow", (ev) => { if (ev.persisted) root.location.reload(); });
  }

  root.LabelerHub = { state: S };
  boot().catch((e) => {
    S.connected = false;
    renderConn();
    $("guideText").textContent = `The page could not start: ${e.message}. Is the tool running? Reload when it is.`;
  });
})(window);
