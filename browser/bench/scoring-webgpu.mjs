// SPDX-License-Identifier: Apache-2.0
// Bank scoring benchmark in a browser: the same measurements as scoring.mjs (the runtime's loop
// against the scoring graph), run in a page, on WebGPU and on WebAssembly.
//
//   python -m cardvision.mtg scorer --rows 100000 --out DIR      (from the repository's root)
//   node bench/scoring-webgpu.mjs --chrome PATH --data DIR [--data DIR2 ...] [--runs 20] [--headed]
//       [--chrome-arg ARG ...]      (for example --chrome-arg --no-sandbox in a container run as root)
//
// It drives Chrome with the package puppeteer-core, which is not a dependency of this folder:
//   npm install --no-save puppeteer-core
// The page is served from this machine (127.0.0.1) and opens nothing else. The adapter the page
// got is printed with the figures: on a machine without a GPU, Chrome's WebGPU runs on
// SwiftShader, a software renderer, and its figures say nothing about a GPU.

import { readFileSync } from "node:fs";
import { createServer } from "node:http";
import { join } from "node:path";
import { build } from "esbuild";

const args = process.argv.slice(2);
const all = (name) => args.flatMap((a, i) => (a === `--${name}` ? [args[i + 1]] : []));
const one = (name, fallback) => all(name)[0] ?? fallback;
const chrome = one("chrome", process.env.CHROME);
const dirs = all("data");
if (!chrome || dirs.length === 0) {
  console.error("Usage: node bench/scoring-webgpu.mjs --chrome PATH --data DIR [--data DIR ...] [--runs N] [--headed]");
  process.exit(2);
}
let puppeteer;
try {
  puppeteer = (await import("puppeteer-core")).default;
} catch {
  console.error("Needs the package puppeteer-core: npm install --no-save puppeteer-core");
  process.exit(2);
}

const entry = `
import * as ort from "onnxruntime-web/webgpu";
import { runBench } from ${JSON.stringify(new URL("./scoring.ts", import.meta.url).pathname)};
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = "/ort/";
window.bench = async (dir, runs) => {
  const get = async (name) => (await fetch("/data/" + dir + "/" + name)).arrayBuffer();
  const text = async (name) => new TextDecoder().decode(await get(name));
  const adapter = await navigator.gpu?.requestAdapter();
  const data = {
    meta: JSON.parse(await text("meta.json")),
    bankJson: await text("bank.json"),
    bankF16: await get("bank.f16"),
    groupsJson: await text("identity-groups.json"),
    queries: new Float32Array(await get("queries.f32")),
    reference: JSON.parse(await text("reference.json")),
    models: { "scorer-topk.onnx": new Uint8Array(await get("scorer-topk.onnx")),
              "scorer-argmax.onnx": new Uint8Array(await get("scorer-argmax.onnx")) },
  };
  const result = await runBench(ort, data, { providers: ["webgpu", "wasm"], runs, frames: [1, data.meta.looks],
    now: () => performance.now() });
  return { adapter: adapter ? { vendor: adapter.info?.vendor, architecture: adapter.info?.architecture,
    description: adapter.info?.description } : null, ...result };
};
window.benchReady = true;
`;
const bundle = await build({ stdin: { contents: entry, resolveDir: new URL(".", import.meta.url).pathname, loader: "js" },
  bundle: true, format: "esm", write: false, target: "es2022", logLevel: "error" });
const script = bundle.outputFiles[0].text;
const ortDist = new URL("../node_modules/onnxruntime-web/dist/", import.meta.url).pathname;
const page = `<!doctype html><meta charset="utf-8"><title>scoring bench</title><script type="module" src="/bench.js"></script>`;
const server = createServer((request, response) => {
  const url = decodeURIComponent(new URL(request.url, "http://x").pathname);
  try {
    if (url === "/bench.js") return response.writeHead(200, { "content-type": "text/javascript" }).end(script);
    if (url.startsWith("/ort/")) {
      const name = url.slice(5).replace(/[^A-Za-z0-9._-]/g, "");
      const type = name.endsWith(".wasm") ? "application/wasm" : "text/javascript";
      return response.writeHead(200, { "content-type": type }).end(readFileSync(join(ortDist, name)));
    }
    if (url.startsWith("/data/")) {
      const [, , index, name] = url.split("/");
      const dir = dirs[Number(index)];
      if (dir === undefined || !/^[A-Za-z0-9._-]+$/.test(name)) throw new Error("not served");
      return response.writeHead(200).end(readFileSync(join(dir, name)));
    }
    response.writeHead(200, { "content-type": "text/html" }).end(page);
  } catch {
    response.writeHead(404).end();
  }
});
await new Promise((done) => server.listen(0, "127.0.0.1", done));
const browser = await puppeteer.launch({ executablePath: chrome, headless: !args.includes("--headed"),
  args: ["--enable-unsafe-webgpu", "--enable-gpu", "--no-first-run", ...all("chrome-arg")], protocolTimeout: 1_800_000 });
try {
  const tab = await browser.newPage();
  tab.on("console", (message) => console.error(`[page] ${message.text()}`));
  await tab.goto(`http://127.0.0.1:${server.address().port}/`);
  await tab.waitForFunction(() => window.benchReady === true);
  const results = [];
  for (let i = 0; i < dirs.length; i++) {
    results.push({ data: dirs[i], ...(await tab.evaluate((d, r) => window.bench(d, r), String(i), Number(one("runs", "20")))) });
    console.error(`done: ${dirs[i]}`);
  }
  console.log(JSON.stringify({ userAgent: await browser.userAgent(), results }, null, 1));
} finally {
  await browser.close();
  server.close();
}
