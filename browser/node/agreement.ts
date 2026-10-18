// SPDX-License-Identifier: Apache-2.0
// Run the pipeline under Node on frames from disk and print what it found as JSON, so that the
// same frames can be given to the Python package and the two answers compared
// (tests/test_browser_agreement.py). The models run on ONNX Runtime Web's WebAssembly backend,
// which works under Node, so this is the code a browser runs on its WebAssembly path.
//
// Usage: node dist/agreement.mjs JOB.json
//
//   JOB.json  {"release": "<path of a release document>",
//              "frames": [{"name": "...", "rgba": "<path of raw RGBA bytes>",
//                          "width": W, "height": H, "points": [[x, y], ...]}]}
//
// A frame's bytes are W * H * 4 values, row-major, and its width must lie within the release's
// analysis range: the frame is analyzed as it is. `points` are pointer positions in frame
// pixels. One JSON object is printed: per frame its boxes and, for each of the three hover
// selections, what each point resolves to.

import { readFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";

import { HOVER_SELECTIONS } from "../src/hover-layer.js";
import { analyzeFrame, loadRelease } from "../src/load.js";
import { hoverAt } from "../src/pipeline.js";
import { fetchFile } from "./files.js";

interface Job {
  release: string;
  frames: Array<{ name: string; rgba: string; width: number; height: number; points: Array<[number, number]> }>;
}

async function main(): Promise<void> {
  const jobPath = process.argv[2];
  if (!jobPath) throw new Error("usage: node dist/agreement.mjs JOB.json");
  const job = JSON.parse(await readFile(jobPath, "utf-8")) as Job;
  const loaded = await loadRelease(pathToFileURL(job.release), { providers: ["wasm"], deps: { fetchFn: fetchFile } });
  const range = loaded.release.runtime.analysisWidth;
  const frames = [];
  for (const frame of job.frames) {
    if (frame.width < range.min || frame.width > range.max) throw new Error(`${frame.name}: width ${frame.width} is outside the analysis range`);
    const data = new Uint8ClampedArray(await readFile(frame.rgba));
    if (data.length !== frame.width * frame.height * 4) throw new Error(`${frame.name}: expected ${frame.width * frame.height * 4} bytes, found ${data.length}`);
    loaded.pipeline.reset(); // every frame is judged alone, as the Python command judges it
    const analysis = await analyzeFrame(loaded, { data, width: frame.width, height: frame.height });
    const hover: Record<string, Array<{ boxIndex: number | null; name: string | null; cardId: string | null }>> = {};
    for (const selection of HOVER_SELECTIONS) {
      hover[selection] = frame.points.map(([x, y]) => {
        const hit = hoverAt(analysis, x, y, selection);
        return { boxIndex: hit.boxIndex, name: hit.name, cardId: hit.cardId };
      });
    }
    frames.push({ name: frame.name, boxes: analysis.boxes, hover });
  }
  await loaded.dispose();
  process.stdout.write(`${JSON.stringify({ release: loaded.release.id, backend: loaded.backend, selection: loaded.release.hover.selection, frames })}\n`);
}

main().catch((error: unknown) => {
  process.stderr.write(`agreement: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exit(1);
});
