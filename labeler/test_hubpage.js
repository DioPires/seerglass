// SPDX-License-Identifier: Apache-2.0
// Tests for hub.js itself (the start screen and the project screen), run in Node against a
// stand-in for the page: just enough of `document` and `fetch` for the script to draw into and
// to record what it sends. What is tested is what the page REQUESTS, and when:
//
//   - arriving at the address of a project asks only for the read-only report: nothing is
//     remembered because a browser was sent to an address;
//   - what a person does on the screens (opening a row, choosing what the labels are for,
//     changing a location, the model's boxes) is what makes the tool remember;
//   - the picture on a project's screen is asked for by the key its report gave.
//
// The facts and the report are the running server's own: test_project.py starts one on the toy
// project and passes them in. Run: node test_hubpage.js < facts.json (run by test_project.py).
"use strict";
const assert = require("assert");
const fs = require("fs");
const path = require("path");

const given = JSON.parse(fs.readFileSync(0, "utf8")); // {facts, report, sample}

// ---- a stand-in for the page -------------------------------------------------------------------

function element(tag) {
  const e = {
    tagName: String(tag).toUpperCase(), children: [], attrs: {}, dataset: {}, listeners: {}, style: {}, hidden: false,
    className: "", textContent: "", value: "", checked: false, open: false, scrollTop: 0,
    setAttribute(k, v) { this.attrs[k] = v; if (k === "value") this.value = v; if (k === "checked") this.checked = true; if (k === "id") this.id = v; if (k === "disabled") this.disabled = true; },
    getAttribute(k) { return this.attrs[k]; },
    addEventListener(name, fn) { (this.listeners[name] = this.listeners[name] || []).push(fn); },
    appendChild(c) { this.children.push(c); return c; },
    append(...cs) { for (const c of cs) this.children.push(c); },
    focus() { doc.activeElement = this; },
    click() { return fire(this, "click"); },
    querySelector(sel) { return all(this).find((x) => matches(x, sel)) || null; },
    querySelectorAll(sel) { return all(this).filter((x) => matches(x, sel)); },
  };
  Object.defineProperty(e, "firstChild", { get() { return this.children[0] || null; } });
  Object.defineProperty(e, "childNodes", { get() { return this.children; } });
  return e;
}
function all(root) { const out = []; (function walk(n) { for (const c of n.children || []) { out.push(c); walk(c); } })(root); return out; }
// the few selectors hub.js uses
function matches(x, sel) {
  if (!x.tagName) return false;
  if (sel === "[data-focus]") return x.dataset.focus !== undefined;
  if (sel === "[data-first]") return x.dataset.first !== undefined;
  if (sel === "button.primary:not(:disabled)") return x.tagName === "BUTTON" && x.className === "primary" && !x.disabled;
  if (sel === "#places input[data-key]") return x.tagName === "INPUT" && x.dataset.key !== undefined;
  throw new Error(`the stand-in page does not know the selector ${sel}`);
}
async function fire(el, name, ev) { for (const fn of el.listeners[name] || []) await fn(Object.assign({ target: el, preventDefault() {} }, ev || {})); }

let doc, sent, wentTo;
function page(address, answers) {
  const ids = {};
  for (const id of ["hub", "guideText", "homeLink", "passName", "connState", "banner"]) ids[id] = Object.assign(element("div"), { id });
  doc = { activeElement: null, title: "", body: element("body"), createElement: element, createTextNode: (t) => ({ text: t }),
    getElementById: (id) => ids[id] || all(ids.hub).find((x) => x.id === id) || null,
    addEventListener() {}, querySelectorAll: (sel) => ids.hub.querySelectorAll(sel), createRange: () => ({ selectNodeContents() {} }) };
  sent = [];
  wentTo = [];
  const location = { get href() { return address; }, set href(v) { wentTo.push(v); }, reload() { wentTo.push("reload"); } };
  const win = { location, document: doc, addEventListener() {}, getSelection: () => ({ removeAllRanges() {}, addRange() {} }),
    LabelerUiState: require("./uistate.js"), LabelerKeyRoute: require("./keyroute.js"), LabelerHubState: require("./hubstate.js"),
    LabelerSaveGate: require("./savegate.js") };
  const fetch = async (url, opts) => {
    const method = (opts && opts.method) || "GET";
    const body = opts && opts.body ? JSON.parse(opts.body) : null;
    sent.push({ method, url, body });
    const answer = answers(method, url, body);
    return { ok: true, status: 200, json: async () => answer };
  };
  // hub.js as the page runs it, with the stand-ins for what it finds on a page
  const script = fs.readFileSync(path.join(__dirname, "hub.js"), "utf8");
  new Function("window", "document", "fetch", "navigator", "setInterval", script)(win, doc, fetch, {}, () => 0);
  return win;
}
const settled = () => new Promise((r) => setTimeout(r, 20));
const posts = () => sent.filter((r) => r.method === "POST").map((r) => r.url);
const answers = (method, url) => {
  if (url === "/api/hub") return given.facts;
  if (url === "/api/project") return given.report;
  if (url === "/api/remember") return { ok: true, remembered: { recent: true, projectFile: null, problem: null }, project: given.report };
  throw new Error(`not expected: ${method} ${url}`);
};

(async () => {
  const address = `http://127.0.0.1:8090/project?sample=${encodeURIComponent(given.sample)}`;

  // Arriving at a project's address: its screen is drawn from the report, and nothing else is asked.
  // (A link on any site, a redirect, a bookmark: all of them end here, with no person's act.)
  {
    const win = page(address, answers);
    await settled();
    assert.strictEqual(win.LabelerHub.state.screen, "project");
    assert.deepStrictEqual(sent.map((r) => `${r.method} ${r.url}`), ["GET /api/hub", "POST /api/project"]);
    assert.deepStrictEqual(sent[1].body, { path: given.sample, role: null, modelBoxes: null, locations: {} });
    assert.ok(!posts().includes("/api/remember"), "arriving at an address remembers nothing");
    assert.strictEqual(doc.title, `${given.report.name} · Card labeler`, "and the project's screen is shown");
    // the picture is asked for by the key the report gave: no path of the page's choosing
    const img = doc.getElementById("hub").querySelectorAll("[data-focus]").length && all(doc.getElementById("hub")).find((x) => x.tagName === "IMG");
    assert.strictEqual(img.attrs.src, `/api/thumb?key=${given.report.frames.thumb.key}`);
    assert.ok(!/sample=|frames=/.test(img.attrs.src));

    // a person chooses what the labels are for: that is remembered, with the choice
    const radio = all(doc.getElementById("hub")).find((x) => x.tagName === "INPUT" && x.attrs.type === "radio" && x.value === "development");
    await fire(radio, "change");
    assert.deepStrictEqual(posts().slice(-1), ["/api/remember"]);
    assert.deepStrictEqual(sent[sent.length - 1].body, { path: given.sample, role: "development", modelBoxes: null, locations: {} });

    // and a location changed on the screen
    const before = sent.length;
    const place = all(doc.getElementById("hub")).find((x) => x.tagName === "INPUT" && x.dataset.key === "catalog");
    place.value = "elsewhere/cards.json";
    await fire(place, "keydown", { key: "Enter" });
    await settled();
    assert.deepStrictEqual(sent.slice(before).map((r) => r.url), ["/api/remember"]);
    assert.deepStrictEqual(sent[sent.length - 1].body.locations, { catalog: "elsewhere/cards.json" });
  }

  // The other addresses: the start screen asks for its facts and nothing more.
  for (const where of ["/", "/start", "/project", "/start?sample=" + encodeURIComponent(given.sample)]) {
    const win = page(`http://127.0.0.1:8090${where}`, answers);
    await settled();
    assert.strictEqual(win.LabelerHub.state.screen, "start", where);
    assert.deepStrictEqual(sent.map((r) => `${r.method} ${r.url}`), ["GET /api/hub"], where);
  }

  // On the start screen, a person opens a sample from its row: remembered by that click, then its screen.
  {
    page("http://127.0.0.1:8090/start", answers);
    await settled();
    const open = all(doc.getElementById("hub")).find((x) => x.tagName === "BUTTON" && x.dataset.focus === `open:${given.sample}`);
    assert.ok(open, "the sample found in the start folder has a row with a button");
    await open.click();
    assert.deepStrictEqual(sent.slice(1).map((r) => [r.url, r.body]), [["/api/remember", { path: given.sample }]]);
    assert.deepStrictEqual(wentTo, [`/project?sample=${encodeURIComponent(given.sample)}`]);
  }

  // "Check files" on a typed path: a path that is a project is opened, and remembered; one that is not is neither.
  {
    page("http://127.0.0.1:8090/start", (method, url, body) => (url === "/api/project" && body.path === "nowhere" ? { ok: false, problem: "missing", path: "/x/nowhere" } : answers(method, url)));
    await settled();
    const input = doc.getElementById("pathInput");
    input.value = "nowhere";
    await fire(input, "keydown", { key: "Enter" });
    await settled();
    assert.deepStrictEqual(sent.slice(1).map((r) => r.url), ["/api/project"]);
    assert.deepStrictEqual(wentTo, []);
    doc.getElementById("pathInput").value = "toy";
    await fire(doc.getElementById("pathInput"), "keydown", { key: "Enter" });
    await settled();
    assert.deepStrictEqual(sent.slice(2).map((r) => [r.url, r.body]), [["/api/project", { path: "toy" }], ["/api/remember", { path: given.report.sample }]]);
    assert.deepStrictEqual(wentTo, [`/project?sample=${encodeURIComponent(given.report.sample)}`]);
  }

  // The start screen's three paths: choosing one asks nothing; the toy path's button makes the toy.
  {
    page("http://127.0.0.1:8090/start", (method, url, body) => (url === "/api/toy" ? { ok: false, problem: "moved", toy: given.facts.toy } : answers(method, url)));
    await settled();
    const cards = all(doc.getElementById("hub")).filter((x) => x.tagName === "BUTTON" && String(x.dataset.focus || "").startsWith("path:"));
    assert.deepStrictEqual(cards.map((x) => x.dataset.focus), ["path:frames", "path:toy", "path:video"]);
    for (const id of ["video", "toy"]) {
      await all(doc.getElementById("hub")).find((x) => x.dataset && x.dataset.focus === `path:${id}`).click();
    }
    assert.deepStrictEqual(sent.map((r) => r.url), ["/api/hub"], "choosing a path asks nothing");
    const make = all(doc.getElementById("hub")).find((x) => x.tagName === "BUTTON" && x.dataset.focus === "toy");
    await make.click();
    await settled();
    assert.deepStrictEqual(sent.filter((r) => r.method === "POST").map((r) => [r.url, r.body]), [["/api/toy", { path: given.facts.toy.path }]]);
  }

  // A tool that runs a pass: a pass's button switches the tool to it, here, as this run.
  {
    const running = JSON.parse(JSON.stringify(given.report));
    running.server = { configured: true, sameProject: false, mode: "box", role: "training" };
    page(address, (method, url, body) => (url === "/api/project" ? running : url === "/api/switch" ? { ok: true, run: "feedfacefeedface" } : answers(method, url)));
    await settled();
    const button = all(doc.getElementById("hub")).find((x) => x.tagName === "BUTTON" && x.dataset.focus === "do:box" && !x.disabled);
    assert.ok(button, "the Boxes row has a button that switches the tool");
    await button.click();
    await settled();
    const ask = sent.find((r) => r.url === "/api/switch");
    assert.deepStrictEqual([ask.body.run, ask.body.unsaved, ask.body.pass, ask.body.path], [given.facts.run, 0, "box", given.sample]);
    assert.deepStrictEqual(wentTo, ["/"]);
  }

  // The Export row writes the file from the page, and asks before it replaces one.
  {
    const done = JSON.parse(JSON.stringify(given.report));
    done.export.coco.included = 3; done.export.coco.recordings = 1;
    done.next = { pass: "export", startable: false };
    let first = true;
    page(address, (method, url, body) => {
      if (url === "/api/project") return done;
      if (url === "/api/export") { const a = first ? { ok: false, problem: "exists", out: "/x/boxes.json" } : { ok: true, kind: "coco", out: "/x/boxes.json", frames: 3, boxes: 9, ignored: 0, sets: 1, notes: [] }; first = false; return a; }
      return answers(method, url);
    });
    await settled();
    await all(doc.getElementById("hub")).find((x) => x.tagName === "BUTTON" && x.dataset.focus === "do:export").click();
    await settled();
    const replace = all(doc.getElementById("hub")).find((x) => x.tagName === "BUTTON" && x.dataset.focus === "export-replace");
    assert.ok(replace, "a file that is there already: the page asks");
    await replace.click();
    await settled();
    assert.deepStrictEqual(sent.filter((r) => r.url === "/api/export").map((r) => [r.body.kind, r.body.replace, r.body.run]),
      [["coco", false, given.facts.run], ["coco", true, given.facts.run]]);
  }

  console.log("test_hubpage.js: all assertions passed");
})().catch((e) => { console.error(e); process.exit(1); });
