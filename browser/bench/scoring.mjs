// SPDX-License-Identifier: Apache-2.0
// Bank scoring benchmark under Node: the runtime's loop against the scoring graph on ONNX
// Runtime Web's WebAssembly backend (the browser's fallback when there is no WebGPU).
//
//   python -m cardvision.mtg scorer --rows 100000 --out DIR      (from the repository's root)
//   node bench/scoring.mjs --data DIR [--data DIR2 ...] [--runs 20] [--threads 1]
//
// The runtime runs WebAssembly on one thread (src/ort.ts), so --threads 1 is the figure that
// applies to the extension; more threads need a cross-origin isolated page. Prints one JSON
// document. The WebGPU figures come from scoring-webgpu.mjs, in a browser.

import { readFileSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { build } from "esbuild";
import * as ort from "onnxruntime-web";

const args = process.argv.slice(2);
const all = (name) => args.flatMap((a, i) => (a === `--${name}` ? [args[i + 1]] : []));
const one = (name, fallback) => all(name)[0] ?? fallback;
const dirs = all("data");
if (dirs.length === 0) {
  console.error("Usage: node bench/scoring.mjs --data DIR [--data DIR ...] [--runs N] [--threads N]");
  process.exit(2);
}
ort.env.wasm.numThreads = Number(one("threads", "1"));

const folder = mkdtempSync(join(tmpdir(), "scoring-bench-"));
const bundle = join(folder, "scoring.mjs");
await build({ entryPoints: [new URL("./scoring.ts", import.meta.url).pathname], bundle: true, format: "esm",
  platform: "node", outfile: bundle, external: ["onnxruntime-web"], logLevel: "error" });
const { runBench } = await import(pathToFileURL(bundle).href);
rmSync(folder, { recursive: true, force: true });

const results = [];
for (const dir of dirs) {
  const read = (name) => readFileSync(join(dir, name));
  const f16 = read("bank.f16");
  const q = read("queries.f32");
  const data = {
    meta: JSON.parse(read("meta.json")),
    bankJson: read("bank.json").toString("utf8"),
    bankF16: f16.buffer.slice(f16.byteOffset, f16.byteOffset + f16.byteLength),
    groupsJson: read("identity-groups.json").toString("utf8"),
    queries: new Float32Array(q.buffer.slice(q.byteOffset, q.byteOffset + q.byteLength)),
    reference: JSON.parse(read("reference.json")),
    models: { "scorer-topk.onnx": new Uint8Array(read("scorer-topk.onnx")), "scorer-argmax.onnx": new Uint8Array(read("scorer-argmax.onnx")) },
  };
  const result = await runBench(ort, data, { providers: ["wasm"], runs: Number(one("runs", "20")), frames: [1, data.meta.looks],
    now: () => performance.now() });
  results.push({ data: dir, ...result });
  console.error(`done: ${dir}`);
}
console.log(JSON.stringify({ node: process.version, ort: ort.env.versions?.web, wasmThreads: ort.env.wasm.numThreads, results }, null, 1));
