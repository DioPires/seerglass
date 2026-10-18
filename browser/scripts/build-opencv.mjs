// SPDX-License-Identifier: Apache-2.0
// Build the OpenCV.js that the geometric stage runs on, from OpenCV's own sources, the same
// way every time:
//
//   node scripts/build-opencv.mjs --work FOLDER [--out FOLDER] [--update-lock]
//
// What is built, and from what, is pinned in opencv.lock.json (beside package.json):
//
//   opencv      OpenCV's git tag and the commit it must resolve to: the version the Python
//               package's index was built with (its "opencv" field), so that keypoints match
//               as closely as two builds of one version can
//   emsdk       the Emscripten SDK's git tag and commit, and the SDK version it installs
//   whitelist   scripts/opencv-whitelist.py: the only functions and classes exposed, and its
//               SHA-256
//   buildJs     the options given to OpenCV's platforms/js/build_js.py: WebAssembly in a file of
//               its own beside an ES module, wasm SIMD on, no threads (so no cross-origin
//               isolation is needed), no eval (DYNAMIC_EXECUTION=0, for a strict CSP)
//   built       the files made, with their sizes and SHA-256, and the tools that made them
//
// --work FOLDER (required) is where the sources and the toolchain go: a folder of yours outside
// the repository. OpenCV is cloned from github.com/opencv/opencv and the SDK from
// github.com/emscripten-core/emsdk; the SDK then installs its compiler from its own release
// store. An existing clone at the pinned commit is reused. CMake (3.x) and git must be on the
// PATH, and python3.
//
// The output goes to --out (default: vendor/opencv/ in this folder, which git ignores):
//
//   opencv.mjs        the ES module; its default export is OpenCV's factory (src/geometric/opencv.ts loads it)
//   opencv_js.wasm    the WebAssembly the module fetches from its own folder
//   opencv.mjs.LEGAL.txt  the licenses of what the two files contain (OpenCV, Emscripten's runtime)
//
// When opencv.lock.json already records a build, the new files are compared with it and the
// script fails on a difference; --update-lock records the new build instead.

import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { copyFileSync, existsSync, mkdirSync, readFileSync, realpathSync, statSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { isAbsolute, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const lockPath = join(root, "opencv.lock.json");
const USAGE = "Usage: node scripts/build-opencv.mjs --work FOLDER [--out FOLDER] [--update-lock]";

function fail(message, code = 1) {
  console.error(message);
  process.exit(code);
}

function parseArgs(argv) {
  const options = { work: null, out: join(root, "vendor", "opencv"), updateLock: false };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--help" || arg === "-h") {
      console.log(`${USAGE}\n\n  --work FOLDER   sources and toolchain (outside the repository)\n  --out FOLDER    where the build goes (default: vendor/opencv)\n  --update-lock   record this build in opencv.lock.json\n  --help`);
      process.exit(0);
    } else if (arg === "--update-lock") options.updateLock = true;
    else if (arg === "--work" || arg === "--out") {
      const value = argv[++i];
      if (value === undefined) fail(`${arg} needs a value\n\n${USAGE}`, 2);
      options[arg.slice(2)] = resolve(value);
    } else fail(`unknown option ${arg}\n\n${USAGE}`, 2);
  }
  if (options.work === null) fail(`--work is required\n\n${USAGE}`, 2);
  const repo = resolve(root, "..");
  const inside = (folder) => !relative(repo, folder).startsWith("..") && !isAbsolute(relative(repo, folder));
  if (inside(options.work)) fail(`--work ${options.work}: choose a folder outside the repository`, 2);
  return options;
}

function run(command, args, { cwd, env } = {}) {
  console.log(`$ ${[command, ...args].join(" ")}`);
  const done = spawnSync(command, args, { cwd, env: env ?? process.env, stdio: "inherit" });
  if (done.error) fail(`${command}: ${done.error.message}`);
  if (done.status !== 0) fail(`${command} exited with ${done.status}`);
}

function output(command, args, { cwd, env } = {}) {
  const done = spawnSync(command, args, { cwd, env: env ?? process.env, encoding: "utf-8" });
  if (done.error || done.status !== 0) return null;
  return done.stdout.trim();
}

const sha256 = (file) => createHash("sha256").update(readFileSync(file)).digest("hex");

/** A shallow clone of `pin.repository` at `pin.tag` in `folder`, checked against `pin.commit`. */
function checkout(pin, folder) {
  if (!existsSync(folder)) run("git", ["-c", "advice.detachedHead=false", "clone", "--depth", "1", "--branch", pin.tag, pin.repository, folder]);
  const head = output("git", ["rev-parse", "HEAD"], { cwd: folder });
  if (head !== pin.commit) fail(`${folder}: at ${head}, not the pinned ${pin.commit} (${pin.tag}); remove the folder to clone it again`);
  const dirty = output("git", ["status", "--porcelain"], { cwd: folder });
  if (dirty) fail(`${folder}: has local changes; remove the folder to clone it again`);
}

/** The environment of the activated SDK (what `source emsdk_env.sh` sets). */
function emsdkEnv(emsdk) {
  const printed = output("bash", ["-c", 'source ./emsdk_env.sh >/dev/null 2>&1 && env -0'], { cwd: emsdk });
  if (printed === null) fail(`${emsdk}: emsdk_env.sh failed`);
  return Object.fromEntries(printed.split("\0").filter(Boolean).map((line) => [line.slice(0, line.indexOf("=")), line.slice(line.indexOf("=") + 1)]));
}

const options = parseArgs(process.argv.slice(2));
const lock = JSON.parse(readFileSync(lockPath, "utf-8"));
for (const tool of ["git", "cmake", "python3"]) if (output(tool, ["--version"]) === null) fail(`${tool} is not on the PATH`);

const whitelist = join(root, lock.whitelist.path);
const whitelistSha = sha256(whitelist);
if (lock.whitelist.sha256 !== null && lock.whitelist.sha256 !== whitelistSha && !options.updateLock) {
  fail(`${lock.whitelist.path}: its SHA-256 is ${whitelistSha}, not the ${lock.whitelist.sha256} opencv.lock.json records (run with --update-lock after changing it)`);
}

mkdirSync(options.work, { recursive: true });
const opencvSrc = join(options.work, `opencv-${lock.opencv.tag}`);
const emsdk = join(options.work, `emsdk-${lock.emsdk.tag}`);
const buildDir = join(options.work, `build-${lock.opencv.tag}`);
checkout(lock.opencv, opencvSrc);
checkout(lock.emsdk, emsdk);
run("./emsdk", ["install", lock.emsdk.sdk], { cwd: emsdk });
run("./emsdk", ["activate", lock.emsdk.sdk], { cwd: emsdk });
// OpenCV writes the build's time into its build information (getBuildInformation), and so into
// the WebAssembly. CMake takes that time from SOURCE_DATE_EPOCH when it is set: the pinned
// commit's time makes two builds byte-identical.
const sourceDate = output("git", ["log", "-1", "--format=%ct"], { cwd: opencvSrc });
const env = { ...emsdkEnv(emsdk), OPENCV_JS_WHITELIST: whitelist, SOURCE_DATE_EPOCH: sourceDate };
const emccVersion = output("emcc", ["--version"], { env })?.split("\n")[0] ?? "unknown";
if (!emccVersion.includes(` ${lock.emsdk.sdk} `)) fail(`emcc reports "${emccVersion}", not ${lock.emsdk.sdk}`);

// No path of this machine may end up in the build. Two kinds would: the source paths that
// OpenCV's assertions carry (__FILE__), which -ffile-prefix-map rewrites to "opencv/...", and
// the compiler's path in OpenCV's build information (modules/core/version_string.inc, written
// when CMake configures), rewritten below between the configure step and the build. The work
// folder is mapped under every name it has (macOS: /tmp is /private/tmp).
const workNames = [...new Set([options.work, realpathSync(options.work)].flatMap((name) =>
  name.startsWith("/private/") ? [name, name.slice("/private".length)] : [name]))].sort((a, b) => b.length - a.length);
const prefixMaps = workNames.map((name) => `-ffile-prefix-map=${name}=opencv`).join(" ");
const buildJs = (extra) => run("emcmake", [
  "python3", join(opencvSrc, "platforms", "js", "build_js.py"), buildDir,
  "--config", whitelist,
  ...extra,
  ...lock.buildJs.cmakeOptions.map((option) => `--cmake_option=${option}`),
  `--build_flags=${lock.buildJs.buildFlags} ${prefixMaps}`,
], { cwd: options.work, env });
buildJs([...lock.buildJs.options, "--config_only"]);
const versionString = join(buildDir, "modules", "core", "version_string.inc");
let info = readFileSync(versionString, "utf-8");
for (const name of workNames) info = info.split(name).join("opencv");
writeFileSync(versionString, info);
buildJs([...lock.buildJs.options.filter((option) => option !== "--clean_build_dir"), "--skip_config"]);

// OpenCV's target wraps the module for <script> use (bin/opencv.js); the ES module is the
// file before that wrapping. Its WebAssembly keeps the name the module asks for.
const bin = join(buildDir, "bin");
mkdirSync(options.out, { recursive: true });
copyFileSync(join(bin, "opencv_js.js"), join(options.out, "opencv.mjs"));
copyFileSync(join(bin, "opencv_js.wasm"), join(options.out, "opencv_js.wasm"));

// The licenses of what the two files hold, in one file named like the bundler's legal files
// (dist/*.LEGAL.txt): OpenCV, the zlib it bundles (the only third-party library its build info
// lists for this build), and the parts of Emscripten compiled in with it (its runtime,
// musl, libc++, libc++abi, compiler-rt).
const emLib = join(emsdk, "upstream", "emscripten", "system", "lib");
const legal = [
  ["OpenCV " + lock.opencv.tag + " (Apache-2.0), " + lock.opencv.repository, [join(opencvSrc, "COPYRIGHT"), join(opencvSrc, "LICENSE")]],
  ["zlib, as bundled by OpenCV in 3rdparty/zlib (zlib license)", [join(opencvSrc, "3rdparty", "zlib", "LICENSE")]],
  ["Emscripten " + lock.emsdk.sdk + " runtime (MIT or University of Illinois/NCSA)", [join(emsdk, "upstream", "emscripten", "LICENSE")]],
  ["musl libc, as compiled in by Emscripten (MIT)", [join(emLib, "libc", "musl", "COPYRIGHT")]],
  ["libc++, as compiled in by Emscripten (Apache-2.0 WITH LLVM-exception)", [join(emLib, "libcxx", "LICENSE.TXT")]],
  ["libc++abi, as compiled in by Emscripten (Apache-2.0 WITH LLVM-exception)", [join(emLib, "libcxxabi", "LICENSE.TXT")]],
  ["compiler-rt, as compiled in by Emscripten (Apache-2.0 WITH LLVM-exception)", [join(emLib, "compiler-rt", "LICENSE.TXT")]],
];
const rule = "=".repeat(78);
writeFileSync(join(options.out, "opencv.mjs.LEGAL.txt"),
  "Third-party licenses of opencv.mjs and opencv_js.wasm (built by scripts/build-opencv.mjs).\n\n" +
  legal.map(([title, files]) => `${rule}\n${title}\n${rule}\n\n${files.map((file) => readFileSync(file, "utf-8").trimEnd()).join("\n\n")}\n`).join("\n"));

const files = Object.fromEntries(["opencv.mjs", "opencv_js.wasm"].map((name) => {
  const file = join(options.out, name);
  return [name, { bytes: statSync(file).size, sha256: sha256(file) }];
}));
const built = {
  files,
  emcc: emccVersion,
  cmake: output("cmake", ["--version"])?.split("\n")[0] ?? "unknown",
  python: output("python3", ["--version"], { env }) ?? "unknown",
  host: `${process.platform}-${process.arch}`,
};
// The check of the path rewriting above: no local path in what is shipped.
// (The home folder of Emscripten's virtual file system, web_user, is the same in every build.)
const escape = (text) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const forbidden = [...workNames, homedir()].map((path) => new RegExp(escape(path)))
  .concat([/\/Users\//, /\/home\/(?!web_user")/, /\/tmp\//, /\/private\//]);
for (const name of Object.keys(files)) {
  const text = readFileSync(join(options.out, name)).toString("latin1");
  const found = forbidden.filter((path) => path.test(text)).map(String);
  if (found.length > 0) fail(`${name} contains a local path (${found.join(", ")}); it must not be shipped`);
}
for (const [name, file] of Object.entries(files)) console.log(`${name}: ${(file.bytes / 1024).toFixed(0)} KB, sha256 ${file.sha256}`);

if (options.updateLock || lock.built === null) {
  writeFileSync(lockPath, `${JSON.stringify({ ...lock, whitelist: { ...lock.whitelist, sha256: whitelistSha }, built }, null, 2)}\n`);
  console.log(`recorded in opencv.lock.json`);
} else {
  const differ = Object.keys(files).filter((name) => lock.built.files[name]?.sha256 !== files[name].sha256);
  if (differ.length > 0) {
    fail(`differs from the build opencv.lock.json records: ${differ.join(", ")} (recorded with ${lock.built.emcc} on ${lock.built.host}; ` +
      `this one with ${built.emcc} on ${built.host}). Run with --update-lock to record this build instead.`);
  }
  console.log("identical to the build opencv.lock.json records");
}
console.log(`OpenCV.js ${lock.opencv.tag} in ${options.out}`);
