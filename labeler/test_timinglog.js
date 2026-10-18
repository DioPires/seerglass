// SPDX-License-Identifier: Apache-2.0
// Tests for timinglog.js. Run: node test_timinglog.js (also run by test_serve.py)
"use strict";
const assert = require("assert");
const T = require("./timinglog.js");

const ev = (i, extra) => Object.assign({ ts: 1000 + i, mode: "box", frame: "f00046.jpg", ev: "committed", target: `B${i}`,
  labelSource: "manual", outcome: "card", eligible: null, ignore: false }, extra || {});

(async () => {
  // batches: one set per batch, queue order kept, every body under the byte cap
  {
    const entries = [];
    for (let i = 0; i < 1000; i++) entries.push({ setId: i < 600 ? "A" : "B", ev: ev(i) });
    entries.push({ setId: "A", ev: ev(1000) });
    const bs = T.batches(entries);
    assert.ok(bs.every((b) => b.body.length <= T.MAX_BATCH_BYTES), "every batch fits the cap");
    assert.ok(bs.every((b) => b.events.length <= T.MAX_BATCH_EVENTS));
    assert.deepStrictEqual(bs.flatMap((b) => b.events.map((e) => e.target)), entries.map((e) => e.ev.target), "order kept");
    assert.ok(bs.every((b) => JSON.parse(b.body).events.length === b.events.length));
    assert.deepStrictEqual(bs.map((b) => b.setId).filter((s, i, a) => i === 0 || a[i - 1] !== s), ["A", "B", "A"], "set runs kept apart");
    const big = T.batches([{ setId: "A", ev: ev(0, { pad: "x".repeat(40000) }) }, { setId: "A", ev: ev(1) }]);
    assert.strictEqual(big.length, 2, "an event over the cap goes alone");
    assert.deepStrictEqual(T.batches([]), []);
    // other keys of a body (the run the page was loaded from), before the events and inside the cap
    assert.strictEqual(T.batches([{ setId: "A", ev: ev(0) }])[0].body, JSON.stringify({ events: [ev(0)] }), "without them: the events alone, as before");
    const tagged = T.batches(entries, 0, 0, { run: "aaaaaaaaaaaaaaaa" });
    assert.ok(tagged.every((b) => JSON.parse(b.body).run === "aaaaaaaaaaaaaaaa" && Object.keys(JSON.parse(b.body)).join() === "run,events"));
    assert.ok(tagged.every((b) => b.body.length <= T.MAX_BATCH_BYTES), "the cap holds with them");
    assert.deepStrictEqual(tagged.flatMap((b) => b.events.map((e) => e.target)), entries.map((e) => e.ev.target));
  }

  // every post and every beacon carries what `extra` says at the moment it is sent; a post that
  // throws (a page of another run: nothing may be written) keeps its events for later
  {
    let run = "first";
    let refuse = true;
    const posts = [], beacons = [];
    const q = T.createQueue({ extra: () => ({ run }),
      post: async (setId, body) => { if (refuse) throw new Error("the tool was started again on other work"); posts.push(JSON.parse(body)); return { status: 200 }; },
      beacon: (setId, body) => { beacons.push(JSON.parse(body)); return true; } });
    q.push("A", ev(1)); q.push("A", ev(2));
    await q.flush();
    assert.strictEqual(q.pending(), 2, "kept, not dropped");
    assert.ok(/started again/.test(q.state.problem));
    refuse = false; run = "second";
    await q.flush();
    assert.deepStrictEqual(posts.map((p) => [p.run, p.events.length]), [["second", 2]], "sent under the run the page now belongs to");
    assert.strictEqual(q.pending(), 0);
    q.push("A", ev(3));
    q.flushBeacon();
    assert.deepStrictEqual(beacons.map((b) => [b.run, b.events.length]), [["second", 1]]);
  }

  // the old failure: a long session (far more than 64 KiB in total) drains completely, in
  // batches under the cap, each posted to the set its events were logged in
  {
    const posts = [];
    const q = T.createQueue({ post: async (setId, body) => { posts.push({ setId, n: JSON.parse(body).events.length, bytes: body.length }); return { status: 200 }; } });
    for (let i = 0; i < 2000; i++) q.push(i % 3 ? "A" : "B", ev(i));
    await q.flush();
    assert.strictEqual(q.pending(), 0);
    assert.strictEqual(posts.reduce((s, p) => s + p.n, 0), 2000);
    assert.ok(posts.every((p) => p.bytes <= T.MAX_BATCH_BYTES));
    assert.strictEqual(q.state.problem, null);
  }

  // set switch: events logged in A stay A's even when B is open by the time they are flushed
  {
    const got = { A: [], B: [] };
    const q = T.createQueue({ post: async (setId, body) => { got[setId].push(...JSON.parse(body).events.map((e) => e.target)); return { status: 200 }; } });
    q.push("A", ev(1)); q.push("A", ev(2)); q.push("B", ev(3)); q.push("A", ev(4));
    await q.flush();
    assert.deepStrictEqual(got, { A: ["B1", "B2", "B4"], B: ["B3"] });
    q.push(null, ev(5)); // no set open: nothing to log to
    assert.strictEqual(q.pending(), 0);
  }

  // network error: the failed batch and the rest stay queued, in order, ahead of newer events;
  // the next flush sends them all
  {
    let down = true;
    const sent = [];
    const q = T.createQueue({ post: async (setId, body) => {
      if (down) throw new TypeError("Failed to fetch");
      sent.push(...JSON.parse(body).events.map((e) => e.target)); return { status: 200 };
    } });
    for (let i = 0; i < 450; i++) q.push("A", ev(i));
    await q.flush();
    assert.strictEqual(q.pending(), 450);
    assert.match(q.state.problem, /not written/);
    q.push("A", ev(999));
    down = false;
    await q.flush();
    assert.strictEqual(q.pending(), 0);
    assert.deepStrictEqual(sent, [...Array.from({ length: 450 }, (_, i) => `B${i}`), "B999"]);
    assert.strictEqual(q.state.problem, null);
  }

  // a failure part-way keeps only what was not sent (no duplicates)
  {
    let calls = 0;
    const sent = [];
    const q = T.createQueue({ post: async (setId, body) => {
      calls++;
      if (calls === 2) return { status: 503 };
      sent.push(...JSON.parse(body).events.map((e) => e.target)); return { status: 200 };
    } });
    for (let i = 0; i < 3 * T.MAX_BATCH_EVENTS; i++) q.push("A", ev(i));
    await q.flush();
    assert.strictEqual(sent.length, T.MAX_BATCH_EVENTS);
    assert.strictEqual(q.pending(), 2 * T.MAX_BATCH_EVENTS);
    await q.flush();
    assert.strictEqual(q.pending(), 0);
    assert.strictEqual(new Set(sent).size, 3 * T.MAX_BATCH_EVENTS, "each event sent once");
  }

  // a 4xx is a refusal: dropped (retrying would block the queue for good) and reported; later
  // batches still go
  {
    const sent = [];
    const q = T.createQueue({ post: async (setId, body) => {
      if (setId === "RO") return { status: 409, error: "read-only" };
      sent.push(setId); return { status: 200 };
    } });
    q.push("RO", ev(1)); q.push("A", ev(2));
    await q.flush();
    assert.strictEqual(q.pending(), 0);
    assert.deepStrictEqual(sent, ["A"]);
    assert.strictEqual(q.state.refused, 1);
  }

  // flushes never overlap: a flush started while one is in flight waits for it
  {
    let inFlight = 0, maxInFlight = 0;
    const q = T.createQueue({ post: async () => {
      inFlight++; maxInFlight = Math.max(maxInFlight, inFlight);
      await new Promise((r) => setTimeout(r, 5));
      inFlight--; return { status: 200 };
    } });
    q.push("A", ev(1));
    const a = q.flush();
    q.push("A", ev(2));
    const b = q.flush();
    await Promise.all([a, b]);
    assert.strictEqual(maxInFlight, 1);
    assert.strictEqual(q.pending(), 0);
  }

  // unload: beacons in small batches; one the browser refuses stays queued
  {
    const bodies = [];
    let n = 0;
    const q = T.createQueue({ post: async () => ({ status: 200 }), beacon: (setId, body) => { n++; bodies.push(body.length); return n !== 2; } });
    for (let i = 0; i < 3 * T.MAX_BATCH_EVENTS; i++) q.push("A", ev(i));
    q.flushBeacon();
    assert.ok(bodies.every((b) => b <= T.MAX_BATCH_BYTES), "every beacon under the 64 KiB budget");
    assert.strictEqual(q.pending(), T.MAX_BATCH_EVENTS);
  }

  console.log("test_timinglog.js: all assertions passed");
})().catch((e) => { console.error(e); process.exit(1); });
