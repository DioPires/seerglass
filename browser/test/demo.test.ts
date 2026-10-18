// SPDX-License-Identifier: Apache-2.0
// What the demo page's address may ask of it (demo/page.ts), and that the address the local
// server prints for a file (scripts/demo-paths.mjs) is one the page follows. No page and no
// server is started here.
import { fileURLToPath, pathToFileURL } from "node:url";

import { describe, expect, it } from "vitest";

import { addressRequest, SERVED_IMAGES_BASE, servedGeometric, type ServerConfig } from "../demo/page.js";

const ORIGIN = "http://127.0.0.1:8765";
const SERVER: ServerConfig = { release: true, images: true, file: "/file/table.png" };
const NOTHING = { file: null, images: false, refused: [] };

describe("the address of the demo page", () => {
  it("asks for nothing when it has no parameter, or others", () => {
    expect(addressRequest("", SERVER, ORIGIN)).toEqual(NOTHING);
    expect(addressRequest("?", SERVER, ORIGIN)).toEqual(NOTHING);
    expect(addressRequest("?release=/elsewhere.json&x=1", SERVER, ORIGIN)).toEqual(NOTHING);
  });

  it("opens the server's one file when it names exactly that file's path", () => {
    expect(addressRequest("?file=/file/table.png", SERVER, ORIGIN)).toEqual({
      file: { url: `${ORIGIN}/file/table.png`, path: "/file/table.png", name: "table.png" },
      images: false,
      refused: [],
    });
    const both = addressRequest("?file=/file/table.png&images=/images/", SERVER, ORIGIN);
    expect([both.file?.name, both.images, both.refused]).toEqual(["table.png", true, []]);
    // A name that needs escaping: the server reports the escaped path, the address carries it.
    const spaced = addressRequest("?file=/file/my%20table.png", { ...SERVER, file: "/file/my%20table.png" }, ORIGIN);
    expect(spaced.file).toEqual({ url: `${ORIGIN}/file/my%20table.png`, path: "/file/my table.png", name: "my table.png" });
  });

  it("follows no other value of file, and fetches nothing for it", () => {
    const others = [
      "http://elsewhere.example/table.png", "https://elsewhere.example/file/table.png", "//elsewhere.example/file/table.png",
      "/release/release.json", "/images/TOY-001.png", "/file/other.png", "/file/", "/file/table.png/", "/file/table.png/x",
      "/file/../release/release.json", "/file/table.png/../../release/release.json", "/file/table.png?x=1", "/file/table.png#x",
      "file/table.png", "table.png", "/FILE/table.png", " /file/table.png", "",
    ];
    for (const value of others) {
      const asked = addressRequest(`?file=${encodeURIComponent(value)}`, SERVER, ORIGIN);
      expect([asked.file, asked.refused.length], value).toEqual([null, 1]);
      expect(asked.refused[0], value).toBe(`file=${value}: this page's server offers one file, /file/table.png, and the address can name no other`);
    }
  });

  it("follows no file when the server offers none", () => {
    for (const config of [{}, { release: true, images: true }]) {
      const asked = addressRequest("?file=/file/table.png", config, ORIGIN);
      expect(asked.file).toBeNull();
      expect(asked.refused).toEqual(["file=/file/table.png: this page's server offers no file (it was started without --file)"]);
    }
  });

  it("takes only a path of the form /file/NAME from the server, on the page's own origin", () => {
    // What a page served by something else might be told: none of it is followed, even when the
    // address repeats it word for word.
    const offers = [
      "http://elsewhere.example/file/table.png", "//elsewhere.example/file/table.png", "/file/a/table.png", "/file/..", "/file/.",
      "/file/", "/elsewhere/table.png", "/file/table.png?x=1", "/file/table.png#x", "/file/a\\b.png", "file/table.png", "/file/%ff.png",
    ];
    for (const offered of offers) {
      const asked = addressRequest(`?file=${encodeURIComponent(offered)}`, { ...SERVER, file: offered }, ORIGIN);
      expect([asked.file, asked.refused.length], offered).toEqual([null, 1]);
    }
    const asked = addressRequest("?file=1", { ...SERVER, file: 1 as unknown as string }, ORIGIN);
    expect(asked.file).toBeNull();
    // What is fetched for a file that is followed is always on the page's origin, under /file/.
    for (const origin of [ORIGIN, "http://127.0.0.1:9000", "http://localhost:8765"]) {
      const followed = addressRequest("?file=/file/table.png", SERVER, origin).file;
      expect(followed === null ? null : [new URL(followed.url).origin, new URL(followed.url).pathname]).toEqual([origin, "/file/table.png"]);
    }
  });

  it("takes card images from the server's own image folder and from nowhere else", () => {
    expect(SERVED_IMAGES_BASE).toBe("/images/");
    expect(addressRequest("?images=/images/", SERVER, ORIGIN)).toEqual({ file: null, images: true, refused: [] });
    for (const value of ["http://elsewhere.example/", "//elsewhere.example/images/", "/release/", "/images", "/images/sub/", "images/", "/", ""]) {
      const asked = addressRequest(`?images=${encodeURIComponent(value)}`, SERVER, ORIGIN);
      expect([asked.images, asked.refused], value).toEqual([false, [`images=${value}: only /images/, the image folder of this page's own server, can be named in the address`]]);
    }
    const none = addressRequest("?images=/images/", { release: true, images: false }, ORIGIN);
    expect([none.images, none.refused]).toEqual([false, ["images=/images/: this page's server has no image folder (it was started without --images)"]]);
    // A refused image folder does not stop the file, and a refused file does not stop the images.
    const mixed = addressRequest("?file=/file/table.png&images=/elsewhere/", SERVER, ORIGIN);
    expect([mixed.file?.name, mixed.images, mixed.refused.length]).toEqual(["table.png", false, 1]);
    const other = addressRequest("?file=/file/other.png&images=/images/", SERVER, ORIGIN);
    expect([other.file, other.images, other.refused.length]).toEqual([null, true, 1]);
  });
});

describe("the address the local server prints for a file", () => {
  it("is followed by the page, whatever the file is called", async () => {
    const paths = await import(pathToFileURL(fileURLToPath(new URL("../scripts/demo-paths.mjs", import.meta.url))).href);
    for (const name of ["table.png", "my table.png", "100%.png", "a+b&c=d#e?.webm", "café (1).jpg", "..hidden.mp4"]) {
      const served = paths.servedFile(`/somewhere/on disk/${name}`);
      expect(served.name, name).toBe(name);
      for (const images of [true, false]) {
        const address = new URL(paths.fileAddress(ORIGIN, served, images));
        const asked = addressRequest(address.search, { release: true, images, file: served.path }, address.origin);
        expect(asked.refused, name).toEqual([]);
        expect([asked.file?.name, asked.file?.path, asked.images], name).toEqual([name, `/file/${name}`, images]);
        // The page fetches the path the server answers at: decoded, it is the served file's own path.
        const fetched = decodeURIComponent(new URL(asked.file!.url).pathname);
        expect(paths.fileAt(served, fetched), name).toBe(served.file);
      }
    }
  });
});

describe("the geometric stage the server offers the scoring page", () => {
  it("is taken only as a path under the server's own folder for it", () => {
    expect(servedGeometric({})).toBeNull();
    expect(servedGeometric({ geometric: "/geometric/settings.json" })).toBe("/geometric/settings.json");
    for (const path of ["https://elsewhere.example/g.json", "/release/g.json", "/geometric/../release/g.json", 42])
      expect(servedGeometric({ geometric: path } as unknown as ServerConfig), String(path)).toBeNull();
  });
});
