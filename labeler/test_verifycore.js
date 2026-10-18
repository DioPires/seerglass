// SPDX-License-Identifier: Apache-2.0
// Tests for verifycore.js (verify mode, naming a center census), on made-up cards.
// Run: node test_verifycore.js (also run by test_serve.py). Stdlib assert only.
"use strict";
const assert = require("assert");
const V = require("./verifycore.js");
const SHA = "a".repeat(64);
const ARMS = ["new", "old"]; // a run's arms, in the server's order: key 1 and key 2
const arms = (first, second, firstId, secondId) => ({
  new: { name: first, printingId: firstId === undefined ? null : firstId, artId: first ? "TOY-001" : null },
  old: { name: second, printingId: secondId === undefined ? null : secondId, artId: second ? "TOY-120" : null },
});
const center = () => ({ id: "C003", x: 0.5, y: 0.25, kind: "face", note: "kept" });

// the walk: faces only (backs with the toggle), frames as given, ids in number order
{
  const frames = [{ name: "b.jpg", centers: [{ id: "C010", kind: "face" }, { id: "C002", kind: "back" }, { id: "C003", kind: "face" }] },
    { name: "a.jpg", centers: [{ id: "C001", kind: "face" }] }];
  assert.deepStrictEqual(V.walk(frames).map((e) => `${e.frame}#${e.id}`), ["b.jpg#C003", "b.jpg#C010", "a.jpg#C001"]);
  assert.deepStrictEqual(V.walk(frames, true).map((e) => e.id), ["C002", "C003", "C010", "C001"]);
  assert.deepStrictEqual(V.walk([]), []);
}

const RA = { x: 0.5, y: 0.25, at: "2030-01-02T10:00:00.000+00:00" };
const REV = "2030-01-02T10:00:05.000+00:00";
const readC = (el) => Object.assign(center(), { eligible: el || "yes", blind: false }); // y / m / n given, not yet saved
const shownC = (el) => Object.assign(readC(el), { readabilityAt: Object.assign({}, RA), revealedAt: REV }); // saved and revealed

// accepts: only after the readability and the reveal, never on a blind center, only a name the arm showed
{
  const c = shownC();
  const ctx = { sha: SHA, blind: false, armNames: ARMS, arms: arms("Amber Kitemaker", "amber  kitemaker", "TOY-001"), revealed: true };
  const a = V.answerFor("accept-new", c, ctx);
  assert.deepStrictEqual(a.fields, { blind: false, verifiedAt: { x: 0.5, y: 0.25, armsManifestSha256: SHA }, verifiedBy: "accept-new", identity: "TOY-001", printing: null });
  assert.strictEqual(V.answerFor("accept-old", c, ctx).fields.identity, "TOY-120", "no printingId: the art id (catalog by name)");
  assert.strictEqual(V.answerFor("accept-both", c, ctx).fields.verifiedBy, "accept-both", "names agree (spacing and case aside)");
  assert.ok(V.answerFor("accept-new", c, Object.assign({}, ctx, { revealed: false })).error, "crop first: no accept before the reveal");
  assert.ok(V.answerFor("accept-new", c, Object.assign({}, ctx, { blind: true })).error, "blind: no accepts");
  assert.ok(V.answerFor("accept-new", c, Object.assign({}, ctx, { arms: null })).error);
  const differ = Object.assign({}, ctx, { arms: arms("Amber Kitemaker", "Kite") });
  assert.ok(V.answerFor("accept-both", c, differ).error, "a: only when both show the same name");
  const none = Object.assign({}, ctx, { arms: arms(null, "Kite") });
  assert.ok(V.answerFor("accept-new", c, none).error, "the new arm showed nothing");
  assert.ok(V.answerFor("accept-both", c, none).error);
  assert.strictEqual(V.answerFor("accept-old", c, none).fields.identity, "TOY-120");
  // readability first: no accept without a readability, even with the arms shown
  for (const how of ["accept-new", "accept-old", "accept-both"]) {
    assert.ok(V.answerFor(how, center(), ctx).error, `${how} without a readability`);
  }
  // an accept names an arm of this run; accept-both needs two arms
  assert.ok(V.answerFor("accept-third", c, ctx).error, "no such arm in this run");
  const oneArm = Object.assign({}, ctx, { armNames: ["new"], arms: { new: ctx.arms.new } });
  assert.strictEqual(V.answerFor("accept-new", c, oneArm).fields.verifiedBy, "accept-new");
  assert.ok(V.answerFor("accept-both", c, oneArm).error, "a one-arm run has no both");
  assert.strictEqual(V.answerFor("accept-both", c, Object.assign({}, ctx, { armNames: undefined })).fields.verifiedBy, "accept-both",
    "without armNames the arms shown are the run's arms");
  const moved = Object.assign(shownC(), { x: 0.6 }); // the readability was for another position
  assert.ok(V.answerFor("accept-new", moved, ctx).error, "a readability for another position is no readability");
}

// readability first: names and Unknown need y / m / n; corrections x / b do not
{
  const bare = center();
  for (const [how, ctx] of [["unknown", { sha: SHA }], ["picked", { sha: SHA, pick: { identity: "TOY-120" } }],
    ["picked", { sha: SHA, pick: { identity: "unknown" } }], ["picked", { sha: SHA, blind: true, pick: { identity: "TOY-120" } }]]) {
    const r = V.answerFor(how, bare, ctx);
    assert.ok(r.error && !r.fields, `${how} without a readability: ${JSON.stringify(ctx.pick || {})}`);
  }
  assert.strictEqual(V.answerFor("not-card", bare, { sha: SHA }).fields.verifiedBy, "not-card", "x at any time");
  assert.strictEqual(V.answerFor("back", bare, { sha: SHA }).fields.verifiedBy, "back", "b at any time");
  assert.deepStrictEqual(bare, center(), "answerFor writes nothing");
  assert.strictEqual(V.answerFor("unknown", readC("no"), { sha: SHA }).fields.identity, "unknown", "Unknown after y / m / n");
}

// picker, unknown, corrections; apply keeps the center's own keys and the readability
{
  const c = shownC();
  const p = V.answerFor("picked", c, { sha: SHA, blind: true, pick: { identity: "TOY-120", printing: "TOY-120A" } }).fields;
  assert.deepStrictEqual([p.verifiedBy, p.identity, p.printing, p.blind], ["picked", "TOY-120", "TOY-120A", true]);
  assert.strictEqual(V.answerFor("picked", c, { sha: SHA, pick: { identity: "unknown" } }).fields.verifiedBy, "unknown", "the picker's Unknown");
  assert.ok(V.answerFor("picked", c, { sha: SHA, pick: {} }).error);
  const x = V.answerFor("not-card", c, { sha: SHA }).fields;
  assert.deepStrictEqual([x.verifiedBy, x.correction, x.identity], ["not-card", "not-card", undefined]);
  const b = V.answerFor("back", c, { sha: SHA }).fields;
  V.applyAnswer(c, p);
  assert.deepStrictEqual([c.verifiedBy, c.eligible, c.readabilityAt, c.revealedAt], ["picked", "yes", RA, REV], "the readability and stamps stay");
  V.applyAnswer(c, b);
  assert.deepStrictEqual(c, { id: "C003", x: 0.5, y: 0.25, kind: "face", note: "kept", blind: false, eligible: "yes",
    readabilityAt: RA, revealedAt: REV, verifiedAt: { x: 0.5, y: 0.25, armsManifestSha256: SHA }, verifiedBy: "back", correction: "back" },
  "kind never changes; the old name goes; the readability and the server's stamps stay");
  V.clearAnswer(c);
  assert.deepStrictEqual(c, center());
  const q = V.applyAnswer(center(), V.answerFor("back", center(), { sha: SHA }).fields);
  assert.ok(!("eligible" in q) && !("readabilityAt" in q), "nothing to keep: nothing invented");
  assert.deepStrictEqual(V.READABILITY_KEYS, ["eligible", "readabilityAt", "revealedAt"]);
  assert.ok(V.VERIFY_KEYS.includes("readabilityAt") && V.VERIFY_KEYS.includes("revealedAt"));
}

// readability(), isRevealed()
{
  assert.strictEqual(V.readability(center()), null);
  assert.strictEqual(V.readability(readC("uncertain")), "uncertain", "set by the page, not yet stamped");
  assert.strictEqual(V.readability(Object.assign(readC(), { eligible: "maybe" })), null);
  assert.strictEqual(V.readability(shownC("no")), "no");
  assert.strictEqual(V.readability(Object.assign(shownC(), { y: 0.3 })), null, "stamped for another position");
  assert.ok(V.isRevealed(shownC()));
  assert.ok(!V.isRevealed(readC()) && !V.isRevealed(Object.assign(readC(), { readabilityAt: RA })));
  assert.ok(!V.isRevealed(Object.assign(readC(), { revealedAt: REV })), "a reveal needs a stamped readability");
  assert.ok(!V.isRevealed(Object.assign(shownC(), { x: 0.6 })), "a reveal for another position is void");
}

// setReadability: before any name and any reveal; fixed once revealed or named
{
  const c = center();
  assert.ok(V.setReadability(c, "q").error);
  assert.deepStrictEqual(c, center(), "a refused key writes nothing");
  assert.deepStrictEqual(V.setReadability(c, "y"), { ok: true, eligible: "yes" });
  assert.deepStrictEqual([c.eligible, c.blind, c.verifiedBy], ["yes", false, undefined], "before any name");
  assert.deepStrictEqual(V.setReadability(c, "m"), { ok: true, eligible: "uncertain" }, "changed freely before the reveal");
  const bl = center();
  assert.deepStrictEqual(V.setReadability(bl, "n", true), { ok: true, eligible: "no" });
  assert.strictEqual(bl.blind, true, "the blind flag comes with it");
  // once revealed: fixed, a pending or a named answer alike
  const s = shownC();
  assert.ok(V.setReadability(s, "n").error, "after the reveal");
  assert.strictEqual(s.eligible, "yes");
  const sn = V.applyAnswer(shownC(), V.answerFor("picked", shownC(), { sha: SHA, pick: { identity: "TOY-001" } }).fields);
  assert.ok(V.setReadability(sn, "m").error);
  // named before any reveal (blind) with a readability: Backspace first
  const nb = V.applyAnswer(Object.assign(readC(), { readabilityAt: RA, blind: true }),
    V.answerFor("picked", readC(), { sha: SHA, blind: true, pick: { identity: "TOY-001" } }).fields);
  assert.ok(V.setReadability(nb, "n").error, "already named");
  assert.strictEqual(nb.eligible, "yes");
  const u = V.applyAnswer(readC(), V.answerFor("unknown", readC(), { sha: SHA }).fields);
  assert.ok(V.isComplete(u) && !V.needsReadability(u));
  assert.ok(V.setReadability(u, "y").error, "Unknown carries its readability: Backspace to change it");
  // a named answer with no readability (written by another tool), or one stale since a move: start again
  const unread = V.applyAnswer(center(), { blind: false, verifiedAt: { x: 0.5, y: 0.25, armsManifestSha256: SHA }, verifiedBy: "picked", identity: "TOY-001", printing: null });
  assert.deepStrictEqual(V.setReadability(unread, "y"), { ok: true, eligible: "yes" });
  assert.deepStrictEqual(unread, Object.assign(center(), { eligible: "yes", blind: false }), "the old name is cleared");
  // a correction keeps its answer; a readability may be added to it
  const corr = V.applyAnswer(center(), V.answerFor("not-card", center(), { sha: SHA }).fields);
  assert.deepStrictEqual(V.setReadability(corr, "n"), { ok: true, eligible: "no" });
  assert.deepStrictEqual([corr.verifiedBy, corr.correction, corr.eligible], ["not-card", "not-card", "no"]);
}

// clearForRedo (Backspace): the whole answer before the reveal, the name only after it
{
  const before = V.applyAnswer(Object.assign(readC(), { readabilityAt: RA, blind: true }),
    V.answerFor("picked", readC(), { sha: SHA, blind: true, pick: { identity: "TOY-001" } }).fields);
  V.clearForRedo(before);
  assert.deepStrictEqual(before, center(), "no reveal: everything goes");
  const after = V.applyAnswer(shownC("uncertain"), V.answerFor("accept-new", shownC("uncertain"),
    { sha: SHA, armNames: ARMS, arms: arms("Amber Kitemaker", null, "TOY-001"), revealed: true }).fields);
  assert.strictEqual(after.verifiedBy, "accept-new");
  V.clearForRedo(after);
  assert.deepStrictEqual(after, Object.assign(center(), { blind: false, eligible: "uncertain", readabilityAt: RA, revealedAt: REV }),
    "revealed: the name goes, the readability and both stamps stay");
  assert.ok(V.isRevealed(after) && !V.hasAnswer(after));
  assert.ok(V.setReadability(after, "y").error, "still fixed after the clear");
  const pend = shownC();
  V.clearForRedo(pend);
  assert.deepStrictEqual(pend, shownC(), "a revealed pending readability: nothing to clear");
  const stale = Object.assign(shownC(), { x: 0.6, verifiedBy: "accept-new", identity: "TOY-001", verifiedAt: { x: 0.5, y: 0.25, armsManifestSha256: SHA } });
  V.clearForRedo(stale);
  assert.deepStrictEqual(stale, Object.assign(center(), { x: 0.6 }), "a reveal for another position is void: everything goes");
}

// completeness, staleness
{
  const c = V.applyAnswer(readC("uncertain"), V.answerFor("picked", readC("uncertain"), { sha: SHA, pick: { identity: "TOY-001" } }).fields);
  assert.ok(!V.needsReadability(c));
  assert.ok(V.isComplete(c) && V.isVerified(c, SHA));
  const noRead = { id: "C003", x: 0.5, y: 0.25, kind: "face", verifiedBy: "picked", identity: "TOY-001", blind: false,
    verifiedAt: { x: 0.5, y: 0.25, armsManifestSha256: SHA } };
  assert.ok(V.needsReadability(noRead) && !V.isComplete(noRead), "a named card with no readability is not complete");
  assert.ok(V.isStale(c, "b".repeat(64)), "another run of the arms");
  c.x = 0.5005; // the marker moved in center mode
  assert.ok(V.isStale(c, SHA) && !V.isVerified(c, SHA));
  assert.ok(!V.isStale(center(), SHA), "no answer: not stale");
  assert.ok(!V.isStale(readC(), SHA) && !V.isComplete(readC()), "a readability alone: open, never stale");
}

// the walk's next open item, counts
{
  const items = [{ id: 1 }, { id: 2 }, { id: 3 }, { id: 4 }];
  const open = new Set([1, 3]);
  assert.strictEqual(V.nextOpen(items, 1, (it) => open.has(it.id)), 2);
  assert.strictEqual(V.nextOpen(items, 2, (it) => open.has(it.id)), 0, "wraps");
  assert.strictEqual(V.nextOpen(items, 0, (it) => it.id === 1), 0, "itself last");
  assert.strictEqual(V.nextOpen(items, 0, () => false), -1);
  const done = V.applyAnswer(center(), V.answerFor("back", center(), { sha: SHA }).fields);
  const stale = Object.assign(V.applyAnswer(center(), V.answerFor("back", center(), { sha: SHA }).fields), { x: 0.6 });
  const table = { 1: { c: done, info: { blind: true, sha: SHA } }, 2: { c: stale, info: { sha: SHA } }, 3: { c: center(), info: { blind: true, sha: SHA } } };
  assert.deepStrictEqual(V.counts([{ k: 1 }, { k: 2 }, { k: 3 }], (it) => table[it.k]),
    { total: 3, verified: 1, stale: 1, blind: 2, blindVerified: 1, open: 2 });
}

// accept names and who they credit
{
  assert.ok(V.isAccept("accept-new") && V.isAccept("accept-both") && !V.isAccept("accept-") && !V.isAccept("picked") && !V.isAccept(null));
  assert.ok(V.isNamed("picked") && V.isNamed("accept-old") && !V.isNamed("unknown"));
  for (const vb of ["accept-new", "accept-both", "picked", "unknown", "not-card", "back"]) assert.ok(V.isVerifiedBy(vb), vb);
  assert.ok(!V.isVerifiedBy("someone") && !V.isVerifiedBy(undefined));
  assert.strictEqual(V.acceptOf("old"), "accept-old");
  assert.deepStrictEqual(V.acceptArms("accept-both", ARMS), ["new", "old"]);
  assert.deepStrictEqual(V.acceptArms("accept-old", ARMS), ["old"]);
  assert.strictEqual(V.acceptArms("accept-both", ["new"]), null);
  assert.strictEqual(V.acceptArms("accept-third", ARMS), null);
  assert.strictEqual(V.acceptArms("accept-new", undefined), null);
}

// an arm's box (frame px, clipped to the frame: the page crops the frame by it) and the fixed window
{
  assert.deepStrictEqual(V.boxToFrame([100, 50, 63, 88], 1920, 1080), [100, 50, 63, 88]);
  assert.deepStrictEqual(V.boxToFrame([-10, 1000, 60, 200], 1920, 1080), [0, 1000, 50, 80], "clipped to the frame");
  assert.strictEqual(V.boxToFrame([2000, 50, 63, 88], 1920, 1080), null, "wholly outside");
  assert.strictEqual(V.boxToFrame(null, 1, 1), null);
  assert.strictEqual(V.boxToFrame([1, 2, 3], 1, 1), null);
  const w = V.windowFor(0, 1080, 540);
  assert.ok(Math.abs(w.side - 378) < 1e-9 && Math.abs(w.z - 540 / 378) < 1e-9);
  assert.strictEqual(V.windowFor(1, 1080, 540).share, 0.6);
  assert.strictEqual(V.windowFor(3, 1080, 540).share, 0.35, "cycles");
}

// which answers are one card: the catalog's rule. Two different cards share the name "Amber
// Circle"; each has a printing of its own.
{
  const catalog = [{ id: "TOY-201", printingOf: null }, { id: "TOY-201A", printingOf: "TOY-201" }, { id: "TOY-202", printingOf: null }, { id: "TOY-202A", printingOf: "TOY-202" }];
  const roots = V.printingRoots(catalog);
  assert.deepStrictEqual(roots, { "TOY-201": "TOY-201", "TOY-201A": "TOY-201", "TOY-202": "TOY-202", "TOY-202A": "TOY-202" });
  const view = (id, name) => ({ name, printingId: id, artId: id });
  const twoCards = { new: view("TOY-201", "Amber Circle"), old: view("TOY-202", "Amber Circle") };
  const oneCard = { new: view("TOY-201", "Amber Circle"), old: view("TOY-201A", "amber circle") };
  const c = shownC();
  const ctx = (armsShown, rule) => ({ sha: SHA, blind: false, armNames: ARMS, arms: armsShown, revealed: true, identity: rule, roots });
  // the default rule (no rule, or "name"): one name is one card, as before
  for (const rule of [undefined, "name"]) {
    assert.strictEqual(V.answerFor("accept-both", c, ctx(twoCards, rule)).fields.identity, "TOY-201", "same name: both agree");
    assert.strictEqual(V.sameCard(twoCards.new, twoCards.old, rule, roots), true);
  }
  assert.strictEqual(V.answerFor("accept-both", c, ctx({ new: view("TOY-201", "Amber Circle"), old: view("TOY-120", "Kite") })).error,
    "the arms show different names: 1 or 2, or / for another", "the default rule's words, as they were");
  // "printing": the same name is not the same card
  assert.strictEqual(V.answerFor("accept-both", c, ctx(twoCards, "printing")).error, "the arms show different cards: 1 or 2, or / for another");
  assert.strictEqual(V.sameCard(twoCards.new, twoCards.old, "printing", roots), false);
  assert.strictEqual(V.answerFor("accept-both", c, ctx(oneCard, "printing")).fields.identity, "TOY-201", "two printings of one card: both agree");
  assert.strictEqual(V.sameCard(oneCard.new, oneCard.old, "printing", roots), true);
  assert.strictEqual(V.answerFor("accept-old", c, ctx(twoCards, "printing")).fields.identity, "TOY-202", "1 / 2 store the card that model showed");
  assert.strictEqual(V.sameCard({ name: "A" }, { name: "A" }, "printing", roots), false, "no card id: no card to compare");
  // printings that name each other in a ring are one card
  assert.deepStrictEqual(V.printingRoots([{ id: "B", printingOf: "C" }, { id: "C", printingOf: "B" }, { id: "E", printingOf: "C" }, { id: "S", printingOf: "S" }]),
    { B: "B", C: "B", E: "B", S: "S" });
}

console.log("test_verifycore.js: ok");
