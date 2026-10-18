// SPDX-License-Identifier: Apache-2.0
// An example: a run of the scoring page without a person, in a headless Chrome.
//
//   node examples/score-headless.mjs --chrome PATH --release FOLDER \
//       --manifest FILE [--manifest FILE ...] --video FILE [--video FILE ...]
//
// It starts the local server of this folder (scripts/demo-server.mjs) on a free port, opens
// the scoring page in the Chrome you name, gives the page the manifests and the video files,
// calls `window.cardvisionScore()`, writes the lines it resolves with to the events file,
// prints the page's summary, and stops the browser and the server. Everything stays on this
// machine, as on the page itself.
//
// It drives Chrome with the package puppeteer-core, which is not a dependency of this folder
// and is not installed by `npm ci`. The script says how to install it when it is missing.
// Run `npm run build` first: the server serves the built pages.
//
// This is an example to start from, not a tool the tests cover beyond its options.

import { spawn } from "node:child_process";
import { existsSync, statSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const USAGE = `Usage: node examples/score-headless.mjs --chrome PATH --release FOLDER
           --manifest FILE [--manifest FILE ...] --video FILE [--video FILE ...]
           [--video-index FILE] [--role ROLE] [--backend webgpu|wasm] [--geometric FILE]
           [--out FILE] [--chrome-arg ARG ...]

Runs the scoring page in a headless Chrome and writes the events file.

  --chrome PATH       the Chrome (or Chromium) program to run
  --release FOLDER    a release folder: a release document and the files it pins
  --manifest FILE     a label manifest; give the option once for each file
  --video FILE        a video file of those manifests, named as the scoring page
                      expects; give the option once for each file
  --video-index FILE  the video-files.json beside kept video files (optional)
  --role ROLE         evaluation (the default), train or extra
  --backend NAME      webgpu or wasm; without it, as the release lists them
  --geometric FILE    the geometric stage's settings file (its packed index inside
                      the same folder): the run uses the stage, which needs WebGPU
  --out FILE          where to write the events file (default browser-events.jsonl)
  --chrome-arg ARG    one more argument for Chrome, as --chrome-arg=ARG; may be repeated
  --help              print this and exit

Needs the package puppeteer-core:  npm install --no-save puppeteer-core
`;

const INSTALL = `This example drives Chrome with the package puppeteer-core, which is not installed.
In browser/, run:

  npm install --no-save puppeteer-core

It is not a dependency of this folder, and \`npm ci\` removes it again.`;

function fail(message) {
  console.error(`${message}\n\n${USAGE}`);
  process.exit(2);
}

/** The options given: each name with its values, in order. */
function options(args) {
  const known = ["chrome", "release", "manifest", "video", "video-index", "role", "backend", "out", "chrome-arg", "geometric"];
  const found = Object.fromEntries(known.map((name) => [name, []]));
  for (let at = 0; at < args.length; at += 1) {
    if (args[at] === "--help" || args[at] === "-h") {
      console.log(USAGE);
      process.exit(0);
    }
    // --name VALUE, or --name=VALUE.
    const [, name, joined] = /^--([a-z-]+)(?:=(.*))?$/s.exec(args[at]) ?? [];
    if (!known.includes(name)) fail(`unknown option ${args[at]}`);
    const value = joined ?? args[(at += 1)];
    if (value === undefined) fail(`--${name} needs a value`);
    found[name].push(value);
  }
  return found;
}

const given = options(process.argv.slice(2));
for (const name of ["chrome", "release", "manifest", "video"]) if (given[name].length === 0) fail(`--${name} is required`);
for (const name of ["chrome", "release", "video-index", "role", "backend", "out", "geometric"]) if (given[name].length > 1) fail(`--${name} is given more than once`);
const [chrome] = given.chrome;
const [release] = given.release;
const role = given.role[0] ?? "evaluation";
const backend = given.backend[0] ?? "";
const out = given.out[0] ?? "browser-events.jsonl";
if (!["evaluation", "train", "extra"].includes(role)) fail(`--role ${role}: choose evaluation, train or extra`);
if (!["", "webgpu", "wasm"].includes(backend)) fail(`--backend ${backend}: choose webgpu or wasm`);
if (!existsSync(release) || !statSync(release).isDirectory()) fail(`--release ${release}: no such folder`);
for (const file of [chrome, ...given.manifest, ...given.video, ...given["video-index"], ...given.geometric]) {
  if (!existsSync(file) || !statSync(file).isFile()) fail(`${file}: no such file`);
}

let puppeteer;
try {
  puppeteer = (await import("puppeteer-core")).default;
} catch {
  console.error(INSTALL);
  process.exit(2);
}

/** Start the local server on a free port; resolves with the address of the scoring page. */
function startServer() {
  const script = fileURLToPath(new URL("../scripts/demo-server.mjs", import.meta.url));
  const extra = given.geometric.length > 0 ? ["--geometric", given.geometric[0]] : [];
  const server = spawn(process.execPath, [script, "--release", release, "--port", "0", ...extra], { stdio: ["ignore", "pipe", "pipe"] });
  const address = new Promise((done, failed) => {
    let said = "";
    const read = (chunk) => {
      said += chunk;
      const found = /^scoring page: (\S+)$/m.exec(said);
      if (found) done(found[1]);
    };
    server.stdout.on("data", read);
    server.stderr.on("data", read);
    server.on("exit", () => failed(new Error(`the server did not start: ${said.trim() || "no reason given"}`)));
  });
  return { server, address };
}

const { server, address } = startServer();
let browser;
let code = 0;
try {
  const url = await address;
  // A long run answers after minutes: no time limit on the call that waits for it.
  browser = await puppeteer.launch({ executablePath: chrome, headless: true, args: given["chrome-arg"], protocolTimeout: 0 });
  console.log(`browser: ${await browser.version()}`);
  const page = await browser.newPage();
  await page.goto(url, { waitUntil: "load" });
  // The page fills in the release field once it has asked the server what it offers.
  await page.waitForFunction(() => typeof window.cardvisionScore === "function" && document.getElementById("release-url").value !== "");
  const choose = async (selector, files) => (await page.$(selector)).uploadFile(...files.map((file) => resolve(file)));
  await choose("#manifests", given.manifest);
  await choose("#videos", given.video);
  if (given["video-index"].length > 0) await choose("#video-index", given["video-index"]);
  await page.select("#role", role);
  await page.select("#backend", backend);

  const started = Date.now();
  // The page says how far it is; repeat that every ten seconds while the run goes on.
  const progress = setInterval(async () => {
    const line = await page.evaluate(() => document.getElementById("panel").textContent.split("\n").find((text) => text.startsWith("rows done"))).catch(() => undefined);
    if (line) console.log(line);
  }, 10_000);
  let lines;
  try {
    lines = await page.evaluate(() => window.cardvisionScore());
  } finally {
    clearInterval(progress);
  }
  writeFileSync(out, lines.join("\n") + "\n");
  // The panel holds the lines of the run, with no empty line among them, so this prints all
  // of it. (The advice for a person that once followed an empty line is now in the page's
  // result card.)
  const panel = await page.evaluate(() => document.getElementById("panel").textContent);
  console.log(panel.split("\n\n")[0]);
  console.log(`wrote ${out}: ${lines.length - 1} row(s), in ${((Date.now() - started) / 1000).toFixed(1)} s`);
} catch (error) {
  console.error(`The run did not finish: ${error instanceof Error ? error.message : String(error)}`);
  code = 1;
} finally {
  await browser?.close().catch(() => undefined);
  server.kill();
}
process.exit(code);
