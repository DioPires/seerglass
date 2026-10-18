// SPDX-License-Identifier: Apache-2.0
// Verify-mode helpers for the labeling tool (naming a center census). Pure: no DOM, no fetch.
// Shared by the page (window.LabelerVerifyCore) and Node (test_verifycore.js). Mirrors
// labelcore.py's answer contract (center_answer_errors, answer_complete, answer_stale,
// verify_walk), which the server enforces on every save.
//
// A center (from center mode: {id, x, y, kind}) gets answer fields only: `identity`, `printing`,
// `eligible` (readability y / m / n → yes / uncertain / no), `verifiedBy` (accept-<arm name> |
// accept-both | picked | unknown | not-card | back), `blind`, `verifiedAt` {x, y,
// armsManifestSha256}, `correction` (not-card | back: never a change of `kind`). The arms are the
// one or two models whose proposals the run shows; their names come from the server.
//
// Readability first: `eligible` is answered from the crop BEFORE any proposal is shown, for
// every face (named, Unknown), so that seeing a proposal cannot change which cards count as
// readable; the server stamps `readabilityAt` and, when it releases the proposals (only after a
// stored readability), `revealedAt`. Both are server-owned: the page keeps them as the server
// sent them. Once revealed, the readability is fixed.
(function (root) {
  "use strict";

  const VERIFY_KEYS = ["identity", "printing", "eligible", "verifiedBy", "blind", "verifiedAt", "correction", "readabilityAt", "revealedAt"];
  // what an answer (name, Unknown, correction) leaves in place: the readability and the stamps
  const READABILITY_KEYS = ["eligible", "readabilityAt", "revealedAt"];
  const ACCEPT_PREFIX = "accept-";
  const ACCEPT_BOTH = "accept-both";
  const CORRECTIONS = ["not-card", "back"];

  // `accept-<arm name>` or `accept-both`
  function isAccept(vb) { return typeof vb === "string" && vb.startsWith(ACCEPT_PREFIX) && vb.length > ACCEPT_PREFIX.length; }
  // a named card (an accept or a pick): its readability is required
  function isNamed(vb) { return vb === "picked" || isAccept(vb); }
  function isVerifiedBy(vb) { return isNamed(vb) || vb === "unknown" || CORRECTIONS.includes(vb); }
  function acceptOf(arm) { return ACCEPT_PREFIX + arm; }
  // The arms an accept credits in a run with these arm names (in the server's order): accept-both
  // → both arms of a two-arm run; accept-<name> → that arm; null when the run has no such arm.
  function acceptArms(how, arms) {
    const a = Array.isArray(arms) ? arms : [];
    if (how === ACCEPT_BOTH) return a.length === 2 ? a.slice() : null;
    if (!isAccept(how)) return null;
    const name = how.slice(ACCEPT_PREFIX.length);
    return a.includes(name) ? [name] : null;
  }
  const ELIGIBLE = { y: "yes", m: "uncertain", n: "no" };
  const XY_TOL = 1e-6;
  // the crop-first view: a fixed window around the center, a share of the frame's height (z cycles);
  // the same whatever the arms did
  const WINDOWS = [0.35, 0.6, 0.2];

  // labelcore.normalize_name: case, typographic quotes / dashes, runs of spaces
  function normName(s) {
    return String(s || "").replace(/[’‘]/g, "'").replace(/[“”]/g, '"').replace(/[–—]/g, "-")
      .replace(/\s+/g, " ").trim().toLowerCase();
  }

  // Which of a catalog's entries are one card: the catalog's "identity" rule (labelcore
  // identity_rule, printing_roots, same_card). "name" (the default): entries that share a name
  // are printings of one card. "printing": an entry and the entries whose printingOf names it
  // are one card, and entries that only share a name are different cards.
  // catalog: [{id, printingOf}] → card id → the card it is a printing of (itself when none)
  function printingRoots(catalog) {
    const parent = new Map();
    for (const c of catalog || []) if (c && typeof c.id === "string") parent.set(c.id, c.printingOf);
    const out = {};
    for (const id of parent.keys()) {
      const path = [id];
      let cur = id;
      while (typeof parent.get(cur) === "string" && parent.has(parent.get(cur)) && !path.includes(parent.get(cur))) { cur = parent.get(cur); path.push(cur); }
      const next = parent.get(cur);
      out[id] = typeof next === "string" && path.includes(next) ? path.slice(path.indexOf(next)).sort()[0] : cur; // a ring of printings: one card
    }
    return out;
  }
  // Two arms' cards ({name, printingId, artId}) are the same card by the rule.
  function sameCard(a, b, rule, roots) {
    if (rule === "printing") {
      const ia = a && (a.printingId || a.artId), ib = b && (b.printingId || b.artId), r = roots || {};
      return !!ia && !!ib && (r[ia] || ia) === (r[ib] || ib);
    }
    return normName(a && a.name) === normName(b && b.name);
  }

  function idNum(id) { const m = /^C(\d{3,})$/.exec(String(id || "")); return m ? parseInt(m[1], 10) : 0; }

  // Every FACE center (and every back with includeBacks), frames in the order given, ids in order.
  function walk(frames, includeBacks) {
    const out = [];
    for (const fr of frames || []) {
      const cs = (fr.centers || []).filter((c) => c && (c.kind === "face" || includeBacks));
      cs.sort((a, b) => idNum(a.id) - idNum(b.id) || String(a.id).localeCompare(String(b.id)));
      for (const c of cs) out.push({ frame: fr.name, id: c.id });
    }
    return out;
  }

  function hasAnswer(c) { return !!(c && c.verifiedBy); }
  // the readability stored for the marker's current position (readability first), else null
  function readability(c) {
    if (!c || !["yes", "uncertain", "no"].includes(c.eligible)) return null;
    const ra = c.readabilityAt;
    if (ra && typeof ra === "object" && !(Math.abs(ra.x - c.x) <= XY_TOL && Math.abs(ra.y - c.y) <= XY_TOL)) return null;
    return c.eligible;
  }
  // the proposals were shown for this position: the readability is fixed
  function isRevealed(c) { return !!(c && c.revealedAt) && !!readability(c) && !!c.readabilityAt; }
  function needsReadability(c) { return !!c && isNamed(c.verifiedBy) && !["yes", "uncertain", "no"].includes(c.eligible); }
  function isComplete(c) { return hasAnswer(c) && isVerifiedBy(c.verifiedBy) && !needsReadability(c); }

  // An answer given for another position of the marker, or for another run of the arms.
  function isStale(c, sha) {
    if (!hasAnswer(c)) return false;
    const at = c.verifiedAt;
    if (!at || typeof at !== "object") return true;
    if (!(Math.abs(at.x - c.x) <= XY_TOL && Math.abs(at.y - c.y) <= XY_TOL)) return true;
    return sha !== null && sha !== undefined && at.armsManifestSha256 !== sha;
  }
  function isVerified(c, sha) { return isComplete(c) && !isStale(c, sha); }

  function clearAnswer(c) { for (const k of VERIFY_KEYS) delete c[k]; return c; }

  // The answer fields for `how` (a verifiedBy) on center c. ctx: {sha, blind, armNames (the run's
  // arms, in order), arms (the server's view of each arm: {name, printingId, artId}; absent on a
  // blind center), pick ({identity, printing}, for picked), identity (the catalog's rule; "name"
  // when absent) and roots (printingRoots, for the rule "printing")}. → {fields} or {error}
  // (nothing is written then).
  function answerFor(how, c, ctx) {
    const o = ctx || {};
    const base = { blind: !!o.blind, verifiedAt: { x: c.x, y: c.y, armsManifestSha256: o.sha } };
    // readability first: a name or Unknown only after y / m / n (corrections x / b at any time)
    if (!CORRECTIONS.includes(how) && !readability(c)) return { error: "readable from this crop alone? y / m / n first, then the name" };
    if (isAccept(how)) {
      if (o.blind) return { error: "blind: name it yourself (/ opens the picker); no proposals here" };
      if (!o.arms) return { error: "no proposals for this center" };
      if (!o.revealed) return { error: "Space or Enter reveals the proposals first" };
      const credited = acceptArms(how, o.armNames || Object.keys(o.arms));
      if (!credited) return { error: how === ACCEPT_BOTH ? "this run has one arm: 1 accepts it" : "this run has no such arm" };
      const shown = credited.map((arm) => o.arms[arm] || {});
      const missing = credited.filter((arm, i) => !shown[i].name);
      if (missing.length) return { error: `${missing.join(" and ")} showed nothing: nothing to accept` };
      if (shown.length === 2 && !sameCard(shown[0], shown[1], o.identity, o.roots)) {
        return { error: o.identity === "printing" ? "the arms show different cards: 1 or 2, or / for another" : "the arms show different names: 1 or 2, or / for another" };
      }
      const id = shown[0].printingId || shown[0].artId;
      if (!id) return { error: `no card id for ${shown[0].name}` };
      return { fields: Object.assign(base, { verifiedBy: how, identity: id, printing: null }) };
    }
    if (how === "picked") {
      const p = o.pick || {};
      if (p.identity === "unknown") return { fields: Object.assign(base, { verifiedBy: "unknown", identity: "unknown" }) };
      if (!p.identity) return { error: "nothing picked" };
      return { fields: Object.assign(base, { verifiedBy: "picked", identity: p.identity, printing: p.printing || null }) };
    }
    if (how === "unknown") return { fields: Object.assign(base, { verifiedBy: "unknown", identity: "unknown" }) };
    if (CORRECTIONS.includes(how)) return { fields: Object.assign(base, { verifiedBy: how, correction: how }) };
    return { error: `unknown answer ${how}` };
  }

  // Replace c's answer with `fields`; the readability and the server's stamps stay (every other
  // key of c stays too).
  function applyAnswer(c, fields) {
    const keep = {};
    for (const k of READABILITY_KEYS) if (k in c) keep[k] = c[k];
    clearAnswer(c); Object.assign(c, keep, fields); return c;
  }

  // Backspace: the whole answer before the reveal; after it, the name only (the readability stays).
  function clearForRedo(c) {
    if (!isRevealed(c)) return clearAnswer(c);
    const keep = { blind: c.blind };
    for (const k of READABILITY_KEYS) if (k in c) keep[k] = c[k];
    clearAnswer(c); Object.assign(c, keep); return c;
  }

  // y / m / n from the crop, before any proposal and before the name → its eligible; else an error.
  function setReadability(c, key, blind) {
    const el = ELIGIBLE[key];
    if (!el) return { error: `not a readability key: ${key}` };
    if (!c) return { error: "no center" };
    if (isRevealed(c)) return { error: "the readability was answered before the proposals were shown; it stays" };
    if (hasAnswer(c) && !CORRECTIONS.includes(c.verifiedBy) && readability(c)) return { error: "already named: Backspace clears the answer to change the readability" };
    if (hasAnswer(c) && !readability(c) && !CORRECTIONS.includes(c.verifiedBy)) clearAnswer(c); // a stale or unstamped answer: start again
    c.eligible = el;
    c.blind = !!blind;
    return { ok: true, eligible: el };
  }

  // The next walk index after `from` (wrapping, `from` itself last) whose item is open, or -1.
  function nextOpen(items, from, isOpen) {
    const n = items.length;
    for (let k = 1; k <= n; k++) {
      const i = (((from + k) % n) + n) % n;
      if (isOpen(items[i])) return i;
    }
    return -1;
  }

  // verified / total / stale / blind over walk items; lookup(item) → {c, info: {blind, sha}}
  function counts(items, lookup) {
    const out = { total: 0, verified: 0, stale: 0, blind: 0, blindVerified: 0 };
    for (const it of items) {
      const { c, info } = lookup(it);
      if (!c) continue;
      const ok = isVerified(c, info && info.sha);
      out.total++;
      if (ok) out.verified++;
      if (isStale(c, info && info.sha)) out.stale++;
      if (info && info.blind) { out.blind++; if (ok) out.blindVerified++; }
    }
    out.open = out.total - out.verified;
    return out;
  }

  // An arm's box [x, y, w, h] in the frame's own pixels (the server has scaled it to the label
  // row's frame size), clipped to the frame; null when there is none.
  function boxToFrame(box, W, H) {
    if (!Array.isArray(box) || box.length !== 4 || box.some((v) => typeof v !== "number" || !isFinite(v))) return null;
    const x0 = Math.max(0, Math.min(W, box[0])), y0 = Math.max(0, Math.min(H, box[1]));
    const x1 = Math.max(0, Math.min(W, box[0] + box[2])), y1 = Math.max(0, Math.min(H, box[1] + box[3]));
    return x1 > x0 && y1 > y0 ? [x0, y0, x1 - x0, y1 - y0] : null;
  }

  // The crop-first view: frame px per side of the window at level k, and the zoom for a view of
  // `size` css px.
  function windowFor(level, H, size) {
    const share = WINDOWS[((level % WINDOWS.length) + WINDOWS.length) % WINDOWS.length];
    const side = Math.max(40, share * H);
    return { share, side, z: size / side };
  }

  const api = { VERIFY_KEYS, READABILITY_KEYS, ACCEPT_BOTH, CORRECTIONS, ELIGIBLE, XY_TOL, WINDOWS, normName, printingRoots, sameCard, walk,
    isAccept, isNamed, isVerifiedBy, acceptOf, acceptArms,
    hasAnswer, readability, isRevealed, needsReadability, isComplete, isStale, isVerified, clearAnswer, clearForRedo, answerFor,
    applyAnswer, setReadability, nextOpen, counts, boxToFrame, windowFor };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.LabelerVerifyCore = api;
})(typeof window !== "undefined" ? window : globalThis);
