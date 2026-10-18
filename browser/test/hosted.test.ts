// SPDX-License-Identifier: Apache-2.0
// The hosted demo (scripts/pages.mjs): the release address a build takes, the policy the page
// runs under, the page as the site serves it, and the built site itself: what it holds, that it
// refers to no other origin than the release's, and that the same address gives the same bytes.
// No page is opened here.
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { afterAll, beforeAll, describe, expect, it } from "vitest";

// @ts-expect-error: a script of this folder, in JavaScript with no declarations.
import * as pages from "../scripts/pages.mjs";
import { hostedRelease, hostedWorkerScript, RELEASE_META, WORKER_URL } from "../demo/page.js";
import { demoView, HOSTED, hostedNote, type ModelState } from "../demo/view.js";

const { buildPages, contentSecurityPolicy, HOSTED_LINKS, hostedPage, releaseAddress, REPOSITORY_URL, SCORE_LINK, siteFiles, VIEWPORT } = pages;
const read = (name: string): string => readFileSync(fileURLToPath(new URL(`../${name}`, import.meta.url)), "utf-8");

const HF = "https://huggingface.co/example/models/resolve/main/release.json";
const HF_FOLDER = "https://huggingface.co/example/models/resolve/main/";

describe("the release address of a hosted build", () => {
  it("is an https address of a release document, or none", () => {
    expect(releaseAddress("")).toEqual({ release: null });
    expect(releaseAddress("  ")).toEqual({ release: null });
    expect(releaseAddress(undefined)).toEqual({ release: null });
    expect(releaseAddress(HF)).toEqual({ release: { url: HF, folder: HF_FOLDER } });
    expect(releaseAddress(` ${HF} `)).toEqual({ release: { url: HF, folder: HF_FOLDER } });
    // http only on this machine, for a preview.
    expect(releaseAddress("http://127.0.0.1:8766/release.json")).toEqual({ release: { url: "http://127.0.0.1:8766/release.json", folder: "http://127.0.0.1:8766/" } });
    expect(releaseAddress("http://localhost:8766/a/release.json").release?.folder).toBe("http://localhost:8766/a/");
  });

  it("refuses anything else, and says why", () => {
    const problem = (text: string): string => releaseAddress(text).problem ?? "";
    expect(problem("release.json")).toBe("release.json is not an absolute URL");
    expect(problem("/release/release.json")).toContain("is not an absolute URL");
    expect(problem("http://example.org/release.json")).toContain("must be served over https");
    expect(problem("ftp://example.org/release.json")).toContain("must be served over https");
    const signedIn = new URL("https://example.org/release.json");
    signedIn.username = "someone";
    expect(problem(signedIn.href)).toContain("user name or a password");
    expect(problem(`${HF}?download=true`)).toContain("query or a fragment");
    expect(problem(`${HF}#x`)).toContain("query or a fragment");
    expect(problem(`${HF}?`)).toContain("query or a fragment");
    expect(problem(HF_FOLDER)).toContain("the address of the release document itself");
    for (const bad of ["https://example.org/a;b/release.json", "https://example.org/a,b/release.json", "https://example.org/it's/release.json"]) {
      expect(problem(bad), bad).toContain("a character a policy source cannot");
    }
  });
});

describe("the Content-Security-Policy of the hosted page", () => {
  it("allows the site, the release's folder and its host's storage, and nothing else", () => {
    expect(contentSecurityPolicy(releaseAddress(HF).release)).toBe([
      "default-src 'none'",
      "script-src 'self' 'wasm-unsafe-eval'",
      "worker-src 'self' blob:",
      `connect-src 'self' ${HF_FOLDER} https://*.hf.co`,
      "img-src blob: data:",
      "media-src blob:",
      "style-src 'self'",
      "base-uri 'none'",
      "form-action 'none'",
    ].join("; "));
  });

  it("names no storage for another host, and nothing to connect to without a release", () => {
    expect(contentSecurityPolicy(releaseAddress("https://models.example.org/v1/release.json").release)).toContain("; connect-src 'self' https://models.example.org/v1/; ");
    expect(contentSecurityPolicy(null)).toContain("; connect-src 'self'; ");
    for (const policy of [contentSecurityPolicy(null), contentSecurityPolicy(releaseAddress(HF).release)]) {
      // No JavaScript eval, no inline script or style, and no frame-ancestors, which a meta element cannot carry.
      expect(policy).not.toMatch(/'unsafe-eval'|'unsafe-inline'|frame-ancestors|\*(?!\.hf\.co)/);
    }
  });
});

describe("the page as the hosted site serves it", () => {
  const html = read("demo/index.html");

  it("carries the policy and the release address before anything is loaded, and links the repository", () => {
    const hosted = hostedPage(html, releaseAddress(HF).release);
    const head = hosted.slice(0, hosted.indexOf("<title>"));
    expect(head.endsWith(`${VIEWPORT}<meta http-equiv="Content-Security-Policy" content="${contentSecurityPolicy(releaseAddress(HF).release)}">\n<meta name="${RELEASE_META}" content="${HF}">\n`)).toBe(true);
    expect(hosted).not.toContain("score.html");
    expect(hosted).toContain(`<span class="page-links"><a class="quiet" href="${REPOSITORY_URL}">Source code and documentation</a>`
      + `<a class="quiet" href="THIRD-PARTY-NOTICES.txt">Third-party notices</a></span>`);
    // Nothing else of the page changes.
    expect(hosted.replace(/<meta http-equiv=.*\n<meta name="cardvision-release".*\n/, "").replace(HOSTED_LINKS, SCORE_LINK)).toBe(html);
    expect(hostedPage(html, null)).toContain(`<meta name="${RELEASE_META}" content="">`);
  });

  it("stops the build when the page no longer has the pieces it changes", () => {
    expect(() => hostedPage(html.replace(SCORE_LINK, ""), null)).toThrow(/expected one/);
    expect(() => hostedPage(html.replace(VIEWPORT, ""), null)).toThrow(/expected one/);
    expect(() => hostedPage(html + SCORE_LINK, null)).toThrow(/expected one/);
  });

  it("is told its release address by the meta element, and the local server's page has none", () => {
    const doc = (content: string | null) => ({
      querySelector: (selector: string) => (selector === `meta[name="${RELEASE_META}"]` && content !== null ? { content } : null),
    }) as unknown as Pick<Document, "querySelector">;
    expect(hostedRelease(doc(null))).toBeNull();
    expect(hostedRelease(doc(""))).toEqual({ release: null, games: [] });
    expect(hostedRelease(doc(` ${HF} `))).toEqual({ release: HF, games: [] });
  });

  it("starts its worker from a script that imports the built worker beside the page", () => {
    expect(WORKER_URL).toBe("lib/worker.js");
    expect(hostedWorkerScript("https://example.github.io/project/")).toBe('import "https://example.github.io/project/lib/worker.js";\n');
    expect(hostedWorkerScript("http://127.0.0.1:8765/?x=1")).toBe('import "http://127.0.0.1:8765/lib/worker.js";\n');
  });

  it("says where its model comes from, and what it lacks when it has none", () => {
    expect(hostedNote(HF)).toBe("The model is downloaded from huggingface.co when the page opens.");
    expect(hostedNote("http://127.0.0.1:8766/release.json")).toBe("The model is downloaded from 127.0.0.1:8766 when the page opens.");
    expect(hostedNote(null)).toBeNull();
    const none = demoView({ kind: "none" }, { kind: "none" }, undefined, false, [], true);
    expect([none.step, none.state.text]).toEqual(["model", HOSTED.noModelState]);
    const failed: ModelState = { kind: "failed", address: HF, reason: "release: HTTP 404" };
    expect(demoView(failed, { kind: "none" }, undefined, false, [], true).modelProblem).toBe(`Could not load the model from ${HF} (release: HTTP 404). Press Retry to try again.`);
    // The local server's page says what it always said.
    expect(demoView({ kind: "none" }, { kind: "none" }).state.text).toBe("No model loaded.");
    expect(demoView(failed, { kind: "none" }).modelProblem).toBe(`Could not load the model at ${HF} (release: HTTP 404). Check the address, then press Retry.`);
  });
});

/** Every absolute http(s) address in a text. */
const addresses = (text: string): string[] => [...new Set(text.match(/https?:\/\/[^\s"'`)<>;\\]+/g) ?? [])].sort();

/** Addresses that appear in the bundled code and are never fetched: an XML namespace, and a page
 *  an ONNX Runtime error message points to. A new one fails the test, to be looked at. */
const NOT_FETCHED = ["http://www.w3.org/2000/svg", "https://web.dev/cross-origin-isolation-guide/"];

describe("the built site", () => {
  let out = "";
  let again = "";
  let none = "";
  beforeAll(async () => {
    out = mkdtempSync(join(tmpdir(), "pages-"));
    again = mkdtempSync(join(tmpdir(), "pages-"));
    none = mkdtempSync(join(tmpdir(), "pages-"));
    await buildPages({ releaseUrl: HF, out });
    await buildPages({ releaseUrl: HF, out: again });
    await buildPages({ releaseUrl: "", out: none });
  }, 120_000);
  afterAll(() => {
    for (const folder of [out, again, none]) rmSync(folder, { recursive: true, force: true });
  });
  const text = (folder: string, name: string): string => readFileSync(join(folder, name), "utf-8");

  it("holds the demo page, its worker and the ONNX Runtime files the worker loads, and not the scoring page", () => {
    expect(siteFiles(out)).toEqual([
      "THIRD-PARTY-NOTICES.txt", "demo.css", "demo.js", "index.html", "lib/ort-wasm-simd-threaded.jsep.mjs", "lib/ort-wasm-simd-threaded.jsep.wasm",
      "lib/worker.js", "lib/worker.js.LEGAL.txt", "theme.css",
    ]);
    for (const name of siteFiles(out)) expect(name).not.toMatch(/score/);
    for (const name of siteFiles(out).filter((file: string) => /\.(html|js|mjs|css)$/.test(file))) {
      expect(text(out, name), name).not.toMatch(/score\.(html|js)/);
    }
    // Every ONNX Runtime file the worker names is there.
    for (const name of new Set(text(out, "lib/worker.js").match(/ort-wasm-[a-z.-]+\.(?:wasm|mjs)/g))) expect(siteFiles(out)).toContain(`lib/${name}`);
  });

  it("carries the policy, and refers to no origin but the release's, the site's own and the repository's link", () => {
    const html = text(out, "index.html");
    expect(html.match(/http-equiv="Content-Security-Policy"/g)).toHaveLength(1);
    expect(html).toContain(`<meta http-equiv="Content-Security-Policy" content="${contentSecurityPolicy(releaseAddress(HF).release)}">`);
    // Each address the page loads is relative or data:, and the one absolute link is the repository's.
    const attributes = [...html.matchAll(/\s(src|href)="([^"]*)"/g)].map(([, name, value]) => `${name}=${value}`);
    expect(attributes).toEqual(["href=data:,", "href=theme.css", "href=demo.css", `href=${REPOSITORY_URL}`, "href=THIRD-PARTY-NOTICES.txt", "src=demo.js"]);
    expect(html).toMatch(new RegExp(`<a class="quiet" href="${REPOSITORY_URL}">`));
    expect(addresses(html)).toEqual([HF_FOLDER, REPOSITORY_URL, HF, "https://*.hf.co"].sort());
    for (const sheet of ["theme.css", "demo.css"]) expect(text(out, sheet), sheet).not.toMatch(/url\(|@import|https?:/);
    for (const script of ["demo.js", "lib/worker.js", "lib/ort-wasm-simd-threaded.jsep.mjs"]) {
      expect(addresses(text(out, script)).filter((found) => !NOT_FETCHED.includes(found)), script).toEqual([]);
    }
  });

  it("carries the full license texts of the bundled third-party code, which the LEGAL file names", () => {
    const notices = text(out, "THIRD-PARTY-NOTICES.txt");
    const legal = text(out, "lib/worker.js.LEGAL.txt");
    // The code the LEGAL file names, each with its license.
    expect(legal).toContain("ONNX Runtime Web v1.29.0");
    expect(legal).toContain("Google LLC");
    expect(legal).toContain("long/index.js");
    expect(notices).toContain("ONNX Runtime Web 1.29.0 (onnxruntime-web, onnxruntime-common)\nLicense: MIT");
    expect(notices).toContain("Code by Google LLC");
    expect(notices).toContain("long.js (Copyright 2009 The Closure Library Authors");
    // The MIT License in full, and the Apache License 2.0 terms in full, without the appendix.
    expect(notices).toContain("Copyright (c) Microsoft Corporation\n\nPermission is hereby granted, free of charge");
    expect(notices).toContain("The above copyright notice and this permission notice shall be included in all\ncopies or substantial portions of the Software.");
    expect(notices).toContain("TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION");
    expect(notices.trimEnd().endsWith("END OF TERMS AND CONDITIONS")).toBe(true);
    expect(notices).not.toContain("APPENDIX");
    // The page links it from its header.
    expect(text(out, "index.html")).toContain('<a class="quiet" href="THIRD-PARTY-NOTICES.txt">Third-party notices</a>');
  });

  it("gives the same bytes for the same address", () => {
    expect(siteFiles(again)).toEqual(siteFiles(out));
    for (const name of siteFiles(out)) expect(readFileSync(join(again, name)).equals(readFileSync(join(out, name))), name).toBe(true);
  });

  it("built with no address, connects to nothing but the site, and differs only in its page", () => {
    const html = text(none, "index.html");
    expect(html).toContain(`<meta name="${RELEASE_META}" content="">`);
    expect(html).toContain(`content="${contentSecurityPolicy(null)}"`);
    expect(addresses(html)).toEqual([REPOSITORY_URL]);
    for (const name of siteFiles(none).filter((file: string) => file !== "index.html")) {
      expect(readFileSync(join(none, name)).equals(readFileSync(join(out, name))), name).toBe(true);
    }
  });
});
