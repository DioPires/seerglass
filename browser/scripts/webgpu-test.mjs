// SPDX-License-Identifier: Apache-2.0
// The vote's WebGPU checks (test/webgpu/vote.page.ts) in a headless Chrome: the GPU voter
// against the CPU voter, exactly; with --bench, a timing of the whole-index vote too.
//
//   npm run test:webgpu -- --chrome PATH [--bench] [--rows N] [--runs N] [--headed]
//
// CHROME in the environment may name the Chrome instead of --chrome. It drives Chrome with
// the package puppeteer-core, which is not a dependency of this folder:
//   npm install --no-save puppeteer-core
// The page is served from this machine (127.0.0.1) and opens nothing else.

import { createServer } from "node:http";
import { build } from "esbuild";

const args = process.argv.slice(2);
const value = (name, fallback) => {
  const at = args.indexOf(`--${name}`);
  return at >= 0 ? args[at + 1] : fallback;
};
const chrome = value("chrome", process.env.CHROME);
if (!chrome) {
  console.error("Name a Chrome with --chrome PATH or CHROME=PATH (Chrome for Testing works).");
  process.exit(2);
}
let puppeteer;
try {
  puppeteer = (await import("puppeteer-core")).default;
} catch {
  console.error("Needs the package puppeteer-core: npm install --no-save puppeteer-core");
  process.exit(2);
}

const bundle = await build({
  entryPoints: [new URL("../test/webgpu/vote.page.ts", import.meta.url).pathname],
  bundle: true,
  format: "esm",
  write: false,
  target: "es2022",
});
const script = bundle.outputFiles[0].text;
const page = `<!doctype html><meta charset="utf-8"><title>vote</title><script type="module" src="/vote.js"></script>`;
const server = createServer((request, response) => {
  if (request.url === "/vote.js") response.writeHead(200, { "content-type": "text/javascript" }).end(script);
  else response.writeHead(200, { "content-type": "text/html" }).end(page);
});
await new Promise((done) => server.listen(0, "127.0.0.1", done));
const address = `http://127.0.0.1:${server.address().port}/`;

const browser = await puppeteer.launch({
  executablePath: chrome,
  headless: !args.includes("--headed"),
  args: ["--enable-unsafe-webgpu", "--enable-gpu", "--use-angle=metal", "--no-first-run"],
});
let code = 0;
try {
  const tab = await browser.newPage();
  tab.on("console", (message) => console.log(`[page] ${message.text()}`));
  await tab.goto(address);
  await tab.waitForFunction(() => typeof window.voteChecks === "function");
  if (!args.includes("--bench-only")) {
    const results = await tab.evaluate(() => window.voteChecks());
    for (const r of results) console.log(`${r.ok ? "ok  " : "FAIL"} ${r.name}: ${r.detail}`);
    const failed = results.filter((r) => !r.ok).length;
    console.log(`${results.length - failed} of ${results.length} checks agree with the CPU voter`);
    if (failed > 0) code = 1;
  }
  if (args.includes("--bench") || args.includes("--bench-only")) {
    const rows = Number(value("rows", "1500000"));
    const runs = Number(value("runs", "10"));
    const options = JSON.parse(value("options", "{}"));
    const timing = await tab.evaluate((r, n, o) => window.voteBench(r, [500, 6000], n, o), rows, runs, options);
    console.log(JSON.stringify(timing));
  }
} catch (error) {
  console.error(error);
  code = 1;
} finally {
  await browser.close();
  server.close();
}
process.exit(code);
