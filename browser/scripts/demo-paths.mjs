// SPDX-License-Identifier: Apache-2.0
// What the local server (scripts/demo-server.mjs) serves, and at which paths. These functions
// start nothing and read no file, so the rules can be tested without a server.

import { basename, extname, normalize, resolve, sep } from "node:path";

export const TYPES = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8", ".json": "application/json; charset=utf-8", ".wasm": "application/wasm",
  ".txt": "text/plain; charset=utf-8", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
  ".gif": "image/gif", ".avif": "image/avif", ".mp4": "video/mp4", ".m4v": "video/mp4", ".webm": "video/webm",
  ".mov": "video/quicktime", ".mkv": "video/x-matroska", ".ogv": "video/ogg",
};

/** Where the --images folder is served. The demo page shows card images from it when the server
 *  has one, and takes this base, and no other, from its address. */
export const IMAGES_BASE = "/images/";

/** The file a request path names inside a folder, or null when it would leave the folder. */
export function inside(folder, relative) {
  const root = resolve(folder);
  const file = resolve(root, normalize(relative).replace(/^[/\\]+/, ""));
  return file === root || file.startsWith(root + sep) ? file : null;
}

/**
 * Where the one file given with --file is served: at one fixed path, `/file/` and the file's own
 * name. The page tells a video from an image by the type, so a file whose extension names
 * neither is refused: the result is then `{ problem }`.
 */
export function servedFile(file) {
  const name = basename(resolve(file));
  const type = TYPES[extname(name).toLowerCase()];
  if (type === undefined || !/^(image|video)\//.test(type)) {
    const known = Object.keys(TYPES).filter((ext) => /^(image|video)\//.test(TYPES[ext])).join(", ");
    return { problem: `not a video or image file by its name (the page shows ${known})` };
  }
  return { file: resolve(file), name, path: `/file/${encodeURIComponent(name)}`, type };
}

/**
 * The file a request under `/file/` gets: the served file for exactly its own path, and nothing
 * for any other. `path` is the request's path, percent-decoded. No folder is served here, so
 * no other name and no path with a further part reaches a file.
 */
export function fileAt(served, path) {
  return served && path === `/file/${served.name}` ? served.file : null;
}

/** The address that opens the served file in the demo page, with the server's card images if it has them. */
export function fileAddress(origin, served, images) {
  return `${origin}/?file=${served.path}${images ? `&images=${IMAGES_BASE}` : ""}`;
}
