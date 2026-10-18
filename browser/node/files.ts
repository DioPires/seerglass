// SPDX-License-Identifier: Apache-2.0
// Reading a release from the local disk under Node, where `fetch` does not open files.

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import type { ReleaseDeps } from "../src/release.js";

/** A `fetchFn` for `file:` URLs: the file's bytes, or status 404 when it cannot be read. */
export const fetchFile: ReleaseDeps["fetchFn"] = async (url) => {
  try {
    const bytes = await readFile(fileURLToPath(url));
    return { ok: true, status: 200, arrayBuffer: async () => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer };
  } catch {
    return { ok: false, status: 404, arrayBuffer: async () => new ArrayBuffer(0) };
  }
};
