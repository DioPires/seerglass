// SPDX-License-Identifier: Apache-2.0
// The third-party notices file that every build writes beside its bundles: the full license texts
// of the third-party code that esbuild bundles into them and that its *.LEGAL.txt files name.
// Those files keep only the short comments the code carries; MIT and Apache-2.0 both ask for the
// license text itself to travel with the code.
//
// What the bundles hold, as worker.js.LEGAL.txt lists it for onnxruntime-web 1.29.0:
// - ONNX Runtime Web (onnxruntime-web, with onnxruntime-common), MIT, Microsoft Corporation. Its
//   npm package ships no license file; the text below is the MIT License as the package declares
//   it, with the copyright line of its bundle's own header.
// - Code by Google LLC inside ONNX Runtime Web's bundle, Apache-2.0.
// - long.js, inside ONNX Runtime Web's bundle, Apache-2.0.
// The Apache-2.0 text is read from the repository's LICENSE, up to the end of its terms.
// OpenCV.js keeps its own license file (opencv.mjs.LEGAL.txt), written by build-opencv.mjs.

import { readFileSync } from "node:fs";
import { join } from "node:path";

/** The file's name in a build. */
export const NOTICES_NAME = "THIRD-PARTY-NOTICES.txt";

const END_OF_TERMS = "END OF TERMS AND CONDITIONS";

const MIT = `MIT License

Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
`;

const RULE = "-".repeat(78);

/** The Apache License 2.0 terms, from the text of a LICENSE file: everything up to and including
 *  "END OF TERMS AND CONDITIONS", without the appendix that names one project's copyright. */
export function apacheTerms(licenseText) {
  const at = licenseText.indexOf(END_OF_TERMS);
  if (at < 0 || !licenseText.includes("Apache License") || !licenseText.includes("Version 2.0, January 2004")) {
    throw new Error("LICENSE is not the Apache License 2.0 text: the third-party notices cannot be written");
  }
  return `${licenseText.slice(0, at + END_OF_TERMS.length).trimEnd()}\n`;
}

/** The notices file's text, given the Apache License 2.0 terms. Reads and writes no file. */
export function thirdPartyNotices(apache, ortVersion) {
  return [
    "Third-party notices",
    "",
    "The scripts and WebAssembly files beside this file include the third-party code below. Each",
    "keeps its own license. This project's own code is Apache-2.0:",
    "https://github.com/ryangswe/riftsight-vision",
    "",
    RULE,
    `ONNX Runtime Web ${ortVersion} (onnxruntime-web, onnxruntime-common)`,
    "License: MIT",
    "Upstream: https://github.com/microsoft/onnxruntime",
    "",
    MIT,
    "ONNX Runtime publishes the notices of the libraries compiled into its WebAssembly files in",
    "ThirdPartyNotices.txt at the upstream address above.",
    "",
    RULE,
    "Code by Google LLC (Copyright 2019, 2020 and 2021 Google LLC), inside ONNX Runtime Web",
    "License: Apache-2.0, text below",
    "",
    RULE,
    "long.js (Copyright 2009 The Closure Library Authors; Copyright 2020 Daniel Wirtz / The",
    "long.js Authors), inside ONNX Runtime Web",
    "License: Apache-2.0, text below",
    "Upstream: https://github.com/dcodeIO/long.js",
    "",
    RULE,
    "Apache License 2.0",
    "",
    apache.trimEnd(),
    "",
  ].join("\n");
}

/** The notices file's text for the browser folder at `root`: the repository's LICENSE beside it
 *  gives the Apache terms, and the installed onnxruntime-web its version. */
export function noticesFor(root) {
  const apache = apacheTerms(readFileSync(join(root, "..", "LICENSE"), "utf-8"));
  const ort = JSON.parse(readFileSync(join(root, "node_modules", "onnxruntime-web", "package.json"), "utf-8"));
  if (ort.license !== "MIT") throw new Error(`onnxruntime-web declares the license ${ort.license}, not MIT: check the notices`);
  return thirdPartyNotices(apache, ort.version);
}
