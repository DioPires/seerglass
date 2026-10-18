// SPDX-License-Identifier: Apache-2.0
// The timing log's queue. Pure, no DOM: sources.js gives it a post and a beacon
// function; test_timinglog.js tests it under node.
//
// Why it exists: posting the whole queue with `fetch(..., {keepalive: true})` without reading the
// response loses events. Chrome counts a keepalive request's body against a 64 KiB budget per
// page until the response body is consumed (or garbage-collected), so after roughly 64 KiB of
// timing in one page session such posts fail, the events go back into the queue, the queue grows
// past 64 KiB (which no keepalive request or sendBeacon can carry) and nothing more is written
// until the page is reloaded. So:
// - every event is tagged with the set it was logged in and posted to that set, whatever set is
//   open when the queue is flushed;
// - posts are ordinary fetches (no keepalive) whose response is read, in batches of at most
//   MAX_BATCH_BYTES / MAX_BATCH_EVENTS, one after another (flushes are chained, never two at once);
// - a network error or a 5xx keeps that batch and everything after it for the next flush; a 4xx
//   (the server refused it: read-only set, bad request) drops that batch and reports it, since
//   retrying it would block the queue for good;
// - the unload beacon goes out in the same small batches;
// - every body may carry more than its events (`opts.extra()`, read when a batch is built): the
//   page sends the identity of the run it was loaded from, which the server asks of every save.
(function (root) {
  "use strict";

  const MAX_BATCH_BYTES = 32 * 1024; // half of Chrome's 64 KiB keepalive / sendBeacon budget
  const MAX_BATCH_EVENTS = 200;

  // `extra`: other keys of the body, before the events ({} or nothing: the events alone)
  function bodyOf(events, extra) { return JSON.stringify(Object.assign({}, extra || {}, { events })); }

  // Queue entries {setId, ev} → [{setId, events, body}], in queue order, one set per batch; a
  // batch closes when the set changes or it would pass maxBytes / maxEvents (a single event larger
  // than maxBytes goes alone).
  function batches(entries, maxBytes, maxEvents, extra) {
    maxBytes = maxBytes || MAX_BATCH_BYTES;
    maxEvents = maxEvents || MAX_BATCH_EVENTS;
    const out = [];
    let cur = null;
    let bytes = 0;
    for (const e of entries) {
      const size = JSON.stringify(e.ev).length + 1;
      if (!cur || cur.setId !== e.setId || cur.events.length >= maxEvents || (cur.events.length && bytes + size > maxBytes)) {
        cur = { setId: e.setId, events: [] };
        bytes = bodyOf([], extra).length;
        out.push(cur);
      }
      cur.events.push(e.ev);
      bytes += size;
    }
    for (const b of out) b.body = bodyOf(b.events, extra);
    return out;
  }

  // opts.post(setId, body) → Promise<{status, error?}> (rejects on a network error, or to keep
  // the batch for a later try); opts.beacon(setId, body) → boolean (sendBeacon's answer);
  // opts.onChange() after each flush; opts.extra() → the other keys of every body, or nothing.
  function createQueue(opts) {
    const q = [];
    let chain = Promise.resolve();
    const state = { sent: 0, refused: 0, problem: null };

    function push(setId, ev) { if (setId) q.push({ setId, ev }); }
    function extra() { return opts.extra ? opts.extra() : null; }

    async function run() {
      if (!q.length) return;
      const list = batches(q.splice(0), 0, 0, extra());
      let i = 0;
      try {
        for (; i < list.length; i++) {
          const b = list[i];
          const r = await opts.post(b.setId, b.body);
          const status = r && r.status;
          if (status >= 200 && status < 300) {
            state.sent += b.events.length;
            state.problem = null;
          } else if (status >= 400 && status < 500) {
            state.refused += b.events.length;
            state.problem = `the server refused ${b.events.length} timing event(s) for ${b.setId} (HTTP ${status}${r.error ? ": " + r.error : ""}); they are not in the log`;
          } else {
            throw new Error(`HTTP ${status}`);
          }
        }
      } catch (e) {
        // keep this batch and the ones after it, in order, ahead of anything queued meanwhile
        const keep = [];
        for (const b of list.slice(i)) for (const ev of b.events) keep.push({ setId: b.setId, ev });
        q.unshift(...keep);
        state.problem = `timing log not written (${e && e.message ? e.message : e}): ${q.length} event(s) waiting, retried every few seconds`;
      }
    }

    // Resolves once everything queued before the call has been tried.
    function flush() {
      chain = chain.then(run, run).then(() => { if (opts.onChange) opts.onChange(); });
      return chain;
    }

    // Page closing: what is left goes out as beacons, in small batches; a batch the browser will
    // not take stays queued (nothing else can be done during unload).
    function flushBeacon() {
      if (!q.length) return;
      for (const b of batches(q.splice(0), 0, 0, extra())) {
        let ok = false;
        try { ok = !!opts.beacon(b.setId, b.body); } catch (e) { ok = false; }
        if (ok) state.sent += b.events.length;
        else for (const ev of b.events) q.push({ setId: b.setId, ev });
      }
    }

    return { push, flush, flushBeacon, state, pending: () => q.length, entries: () => q.slice() };
  }

  const api = { MAX_BATCH_BYTES, MAX_BATCH_EVENTS, batches, createQueue };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.LabelerTimingLog = api;
})(typeof window !== "undefined" ? window : globalThis);
