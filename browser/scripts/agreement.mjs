// SPDX-License-Identifier: Apache-2.0
// Bundle node/agreement.ts and run it: `node scripts/agreement.mjs JOB.json`.
// ONNX Runtime Web is left out of the bundle, so that under Node it finds its own `.wasm` files.

import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import { build } from "esbuild";

const root = fileURLToPath(new URL("..", import.meta.url));
const outfile = `${root}dist/agreement.mjs`;
await build({
  entryPoints: [`${root}node/agreement.ts`],
  outfile,
  bundle: true,
  platform: "node",
  format: "esm",
  target: "node22",
  external: ["onnxruntime-web", "onnxruntime-web/*"],
  logLevel: "warning",
});
const run = spawnSync(process.execPath, [outfile, ...process.argv.slice(2)], { stdio: "inherit" });
process.exit(run.status ?? 1);
