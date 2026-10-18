// SPDX-License-Identifier: Apache-2.0
// Serve the demo page on this machine only (127.0.0.1). Run it with
//
//   npm run demo -- [--release RELEASE_FOLDER] [--images IMAGE_FOLDER] [--port 8765] [--file FILE]
//                   [--geometric SETTINGS_FILE]
//
// `--help` prints the options and exits. An option it does not know is an error.
//
// It serves four things and nothing else:
//   /            the built demo page, the scoring page (/score.html) and the built library (dist/)
//   /release/    the folder given with --release: a release document and the files it pins
//   /images/     the folder given with --images, if any: card images named as the catalog
//                names them
//   /file/NAME   the one video or image file given with --file, if any, under its own name
//   /geometric/  the folder of the geometric stage's settings file given with --geometric, if
//                any: the settings and the packed index they name (relative to them, inside it)
// It only answers GET requests, only for files inside those folders and for that one file, and
// it makes no requests of its own.
//
// With --file it also prints the address that opens that file in the demo page: the page then
// opens the file as if it had been chosen there. That is how a program that drives a browser
// opens the demo, since it cannot use a file chooser. The paths are in demo-paths.mjs.
//
// The pages ask /config.json what was given: with --release they load their model from
// /release/, with --images the demo page shows card images from /images/ in its tooltip, and
// with --geometric the scoring page runs with the geometric stage.

import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { basename, dirname, extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { fileAddress, fileAt, IMAGES_BASE, inside, servedFile, TYPES } from "./demo-paths.mjs";

const USAGE = `Usage: npm run demo -- [--release FOLDER] [--images FOLDER] [--port N] [--file PATH]
                       [--geometric FILE]

Serves the demo page and the scoring page on this machine only (127.0.0.1).

  --release FOLDER  a release folder: a release document and the files it pins.
                    It is served at /release/, and the pages take their model from it.
  --images FOLDER   card images, named as the catalog names them. They are served
                    at /images/, and the demo page shows them in its tooltip.
  --port N          the port to listen on (default 8765). 0 takes any free port.
  --file PATH       one video or image file to open in the demo page with no file
                    chooser. It is served at /file/ under its own name, and the
                    address that opens it is printed. Needs --release.
  --geometric FILE  the geometric stage's settings file. Its folder is served at
                    /geometric/ (the packed index it names must lie inside it), and
                    the scoring page runs with the stage.
  --help            print this and exit
`;

/** The options given, by name. Prints the usage and exits on --help and on anything unknown. */
function options(args) {
  const found = {};
  for (let at = 0; at < args.length; at += 2) {
    const name = args[at];
    if (name === "--help" || name === "-h") {
      console.log(USAGE);
      process.exit(0);
    }
    const value = args[at + 1];
    const problem = !["--release", "--images", "--port", "--file", "--geometric"].includes(name) ? `unknown option ${name}` : value === undefined ? `${name} needs a value` : null;
    if (problem) {
      console.error(`${problem}\n\n${USAGE}`);
      process.exit(2);
    }
    found[name.slice(2)] = value;
  }
  return found;
}

const dist = fileURLToPath(new URL("../dist/", import.meta.url));
const { release, images, file, geometric, port: portText = "8765" } = options(process.argv.slice(2));
const port = Number(portText);
if (!/^\d+$/.test(portText) || port > 65535) {
  console.error(`--port ${portText}: not a port number (0 to 65535)\n\n${USAGE}`);
  process.exit(2);
}
for (const [label, folder] of [["--release", release], ["--images", images]]) {
  if (folder !== undefined && !(existsSync(folder) && statSync(folder).isDirectory())) {
    console.error(`${label} ${folder}: no such folder`);
    process.exit(2);
  }
}
if (geometric !== undefined && !(existsSync(geometric) && statSync(geometric).isFile())) {
  console.error(`--geometric ${geometric}: no such file`);
  process.exit(2);
}
const geometricFolder = geometric === undefined ? undefined : dirname(resolve(geometric));
// The one file to open in the demo page, if any: served at a path of its own and nowhere else.
let served = null;
if (file !== undefined) {
  served = existsSync(file) && statSync(file).isFile() ? servedFile(file) : { problem: "no such file" };
  if (!served.problem && release === undefined) served = { problem: "needs --release, since the page cannot start on a file without a release" };
  if (served.problem) {
    console.error(`--file ${file}: ${served.problem}`);
    process.exit(2);
  }
}
if (!existsSync(join(dist, "demo", "index.html"))) {
  console.error("dist/demo is missing: run `npm run build` first (`npm run demo` does)");
  process.exit(2);
}

function locate(path) {
  if (path === "/config.json") {
    return {
      json: {
        release: release !== undefined,
        images: images !== undefined,
        ...(served ? { file: served.path } : {}),
        ...(geometric !== undefined ? { geometric: `/geometric/${encodeURIComponent(basename(geometric))}` } : {}),
      },
    };
  }
  if (path.startsWith("/geometric/")) return { file: geometricFolder ? inside(geometricFolder, path.slice("/geometric/".length)) : null };
  if (path.startsWith("/release/")) return { file: release ? inside(release, path.slice("/release/".length)) : null };
  if (path.startsWith(IMAGES_BASE)) return { file: images ? inside(images, path.slice(IMAGES_BASE.length)) : null };
  if (path.startsWith("/file/")) return { file: fileAt(served, path) };
  if (path.startsWith("/lib/")) return { file: inside(dist, path.slice("/lib/".length)) };
  return { file: inside(join(dist, "demo"), path === "/" ? "index.html" : path) };
}

const server = createServer((request, response) => {
  let path;
  try {
    path = decodeURIComponent(new URL(request.url ?? "/", "http://127.0.0.1").pathname);
  } catch {
    path = null;
  }
  const found = request.method === "GET" && path !== null ? locate(path) : {};
  if (found.json) {
    response.writeHead(200, { "content-type": TYPES[".json"], "cache-control": "no-store" });
    response.end(JSON.stringify(found.json));
    return;
  }
  if (!found.file || !existsSync(found.file) || !statSync(found.file).isFile()) {
    response.writeHead(request.method === "GET" ? 404 : 405, { "content-type": TYPES[".txt"] });
    response.end(request.method === "GET" ? "not found\n" : "only GET is answered\n");
    return;
  }
  response.writeHead(200, {
    "content-type": TYPES[extname(found.file).toLowerCase()] ?? "application/octet-stream",
    "content-length": statSync(found.file).size,
    "cache-control": "no-store",
  });
  createReadStream(found.file).pipe(response);
});

server.on("error", (error) => {
  console.error(error.code === "EADDRINUSE" ? `port ${port} is in use on 127.0.0.1: pass --port N for another, or --port 0 for any free one` : String(error));
  process.exit(2);
});
server.listen(port, "127.0.0.1", () => {
  const { port: listening } = server.address();
  console.log(`demo: http://127.0.0.1:${listening}/`);
  console.log(`scoring page: http://127.0.0.1:${listening}/score.html`);
  console.log(release ? `release folder: ${resolve(release)} (served at /release/)` : "no --release folder: the demo page asks for the address of a release document");
  if (images) console.log(`image folder: ${resolve(images)} (served at /images/; the demo page shows card images from it)`);
  if (geometric) console.log(`geometric stage: ${resolve(geometric)} (its folder served at /geometric/; the scoring page runs with it)`);
  if (served) {
    console.log(`file: ${served.file} (served at ${served.path})`);
    console.log(`open the file in the demo: ${fileAddress(`http://127.0.0.1:${listening}`, served, images !== undefined)}`);
  }
});
