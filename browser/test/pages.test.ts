// SPDX-License-Identifier: Apache-2.0
// The files of the demo page and the scoring page, read as text: the shared theme, that the
// style sheet takes its colors from it, and that each page holds what its flow and the tools
// that drive it rely on. No page is opened here.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

const fileOf = (name: string): string => fileURLToPath(new URL(`../${name}`, import.meta.url));
const read = (name: string): string => readFileSync(fileOf(name), "utf-8");

/** The declarations of the n-th `:root` block of a style sheet, as `name: value` in order. */
function rootBlock(css: string, nth: number): string[] {
  const blocks = [...css.matchAll(/:root\s*\{([^}]*)\}/g)];
  return (blocks[nth]?.[1] ?? "").split(";").map((line) => line.replace(/\/\*.*?\*\//gs, "").replace(/\s+/g, " ").trim()).filter((line) => line !== "");
}

/** The shared theme: these names with these values, in this order, and nothing else. */
const THEME = [
  "--bg: #0b0f16", "--bg-deep: #070a10", "--panel: #131022", "--raised: #1a1630", "--border: #241d3a", "--border-strong: #4c1d95",
  "--ink: #e6e6e6", "--ink-strong: #ffffff", "--muted: #8b8b98", "--off: #565664", "--accent: #a78bfa", "--accent-soft: #c4b5fd",
  "--accent-strong: #7c3aed", "--accent-deep: #4c1d95", "--on-accent: #ffffff", "--success: #22c55e", "--warning: #fbbf24", "--error: #f87171",
  "--gradient-primary: linear-gradient(135deg, #7c3aed, #4c1d95)",
  '--font-ui: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
  '--font-mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace',
  "--space-1: 4px", "--space-2: 8px", "--space-3: 12px", "--space-4: 16px", "--space-5: 24px", "--space-6: 32px", "--space-7: 48px",
  "--radius-xs: 4px", "--radius-sm: 6px", "--radius-md: 8px", "--radius-lg: 10px", "--radius-xl: 14px", "--radius-pill: 999px",
];

describe("the theme of the pages", () => {
  const theme = read("demo/theme.css");
  const styles = read("demo/demo.css");

  it("starts with the shared block, whole and in order", () => {
    expect(rootBlock(theme, 0)).toEqual(THEME);
    // What the two pages add is in a block of its own, and gives no name of the shared block another value.
    const extras = rootBlock(theme, 1).map((line) => line.split(":")[0]);
    expect(extras.length).toBeGreaterThan(0);
    expect(extras.filter((name) => THEME.some((line) => line.startsWith(`${name}:`)))).toEqual([]);
  });

  it("is where every color, radius and spacing of the style sheet comes from", () => {
    const rules = styles.replace(/\/\*.*?\*\//gs, "");
    expect(rules.match(/#[0-9a-fA-F]{3,8}\b|\b(rgba?|hsla?)\(/g)).toBeNull();
    const declared = [...rules.matchAll(/(?:^|[;{\s])(border-radius|gap|margin(?:-[a-z]+)?|padding(?:-[a-z]+)?)\s*:\s*([^;}]+)/g)];
    expect(declared.length).toBeGreaterThan(40);
    for (const [, property, value] of declared) {
      // A length is a name of the theme; 0 and auto need none.
      const literal = value!.replace(/var\(--[a-z0-9-]+\)/g, "").replace(/\b(0|auto)\b/g, "").trim();
      expect(literal, `${property}: ${value}`).toBe("");
    }
    // Every name the style sheet reads is one the theme gives.
    const given = new Set([...rootBlock(theme, 0), ...rootBlock(theme, 1)].map((line) => line.split(":")[0]));
    const read = new Set([...rules.matchAll(/var\((--[a-z0-9-]+)/g)].map((found) => found[1]!));
    expect([...read].filter((name) => !given.has(name) && name !== "--ratio")).toEqual([]);
  });

  it("is linked by both pages before their style sheet, and copied by the build", () => {
    for (const page of ["demo/index.html", "demo/score.html"]) {
      const html = read(page);
      expect(html.indexOf('href="theme.css"'), page).toBeGreaterThan(0);
      expect(html.indexOf('href="theme.css"'), page).toBeLessThan(html.indexOf('href="demo.css"'));
    }
    expect(read("scripts/build.mjs")).toContain('"index.html", "score.html", "theme.css", "demo.css"');
  });
});

describe("the demo page's file", () => {
  const html = read("demo/index.html");

  it("leads with its title, one sentence and the stage, and has no Start button", () => {
    expect(html).toContain("<h1>Card recognition demo</h1>");
    expect(html).toContain("Drop a video or screenshot of a game, then hover a card to see its name. Nothing leaves this computer.");
    expect(html).toContain("Drop a video or image here");
    expect(html).not.toMatch(/id="start"|>Start</);
    // One primary button, and it is the label of a real file input.
    expect(html.match(/class="button primary"/g)).toHaveLength(1);
    expect(html).toMatch(/<input id="file"[^>]*type="file" accept="video\/\*,image\/\*">\s*<label id="choose" class="button primary" for="file">Choose a file…<\/label>/);
    expect(html).toContain('href="score.html"');
  });

  it("keeps what a first run does not need behind Options and Details, closed", () => {
    const options = /<details id="options"[^>]*>([\s\S]*?)<\/details>/.exec(html);
    const details = /<details id="details"[^>]*>([\s\S]*?)<\/details>/.exec(html);
    expect(html).not.toMatch(/<details[^>]*\bopen\b/);
    for (const id of ["backend", "image-base", "always-on"]) expect(options?.[1], id).toContain(`id="${id}"`);
    expect(details?.[1]).toContain('<pre id="panel">');
    expect(options?.[0].startsWith('<details id="options" class="disclosure">\n    <summary>Options</summary>')).toBe(true);
    expect(details?.[0]).toContain("<summary>Details</summary>");
  });

  it("keeps the text panel whole under Raw text, at the end of Details, and copies that text", () => {
    const details = html.slice(html.indexOf('<details id="details"'), html.indexOf("</main>"));
    for (const id of ["diag-text", "copy-diagnostics", "tiles", "detection-rows", "model-facts", "file-facts"]) expect(details, id).toContain(`id="${id}"`);
    // The panel is the last thing in Details, empty until the page writes its lines.
    expect(details).toMatch(/<details id="raw" class="raw">\s*<summary>Raw text<\/summary>\s*<pre id="panel"><\/pre>\s*<\/details>\s*<\/div>\s*<\/details>\s*$/);
    const script = read("demo/demo.ts");
    expect(script).toContain('const text = panel.textContent ?? "";');
    // The panel is written as it always was, from panelLines, whether or not Details is open.
    expect(script).toContain('setText(panel, panelLines(view, model, file, session?.status() ?? null, view.ready ? panelFrame() : null, shown).join("\\n"));');
  });

  it("holds what a script that drives it reads", () => {
    for (const id of ["file", "video", "image", "outlines", "box", "tooltip", "tooltip-name", "tooltip-id", "panel", "release-url", "all-boxes"]) {
      expect(html.match(new RegExp(`id="${id}"`, "g")), id).toHaveLength(1);
    }
    // A video starts without sound.
    expect(html).toMatch(/<video id="video"[^>]*\bmuted\b/);
  });
});

describe("the scoring page's file", () => {
  const html = read("demo/score.html");

  it("numbers its three inputs, with the naming rule under the field and not in its label", () => {
    const steps = /<ol class="steps">([\s\S]*?)<\/ol>/.exec(html)?.[1] ?? "";
    const names = [...steps.matchAll(/<label class="name" for="([a-z-]+)">([^<]+)<\/label>/g)].map((found) => [found[1], found[2]]);
    expect(names).toEqual([["release-url", "Model"], ["manifests", "Labels (manifest files)"], ["videos", "Videos"]]);
    expect(steps.match(/<li>/g)).toHaveLength(3);
    expect(steps).toMatch(/<p class="quiet">A video is found by its file name[\s\S]*Example: <code>local-table\+2\.webm<\/code>/);
  });

  it("keeps the role, the backend and the index under Options, closed", () => {
    const options = /<details id="options"[^>]*>([\s\S]*?)<\/details>/.exec(html)?.[1] ?? "";
    for (const id of ["role", "backend", "video-index"]) expect(options, id).toContain(`id="${id}"`);
    expect(html).not.toMatch(/<details[^>]*\bopen\b/);
  });

  it("offers the three roles, and its script gives the scoring command the role the run read", () => {
    const options = [...html.matchAll(/<option value="([a-z]+)"( selected)?>\1<\/option>/g)].map((found) => [found[1], found[2] !== undefined]);
    expect(options).toEqual([["evaluation", true], ["train", false], ["extra", false]]);
    const script = read("demo/score.ts");
    // The role is read once, when the run starts, and that value reaches the command: a
    // change of the field during a run does not change what the command says was scored.
    expect(script.match(/roleSelect\.value/g)).toHaveLength(2);   // the check before a run, and the run
    expect(script).toContain("  const role = roleSelect.value;\n");
    expect(script).toContain("showResult(lines, manifests.map((manifest) => manifest.name), releaseUrl, role);");
    expect(script).toContain("scoringCommand(manifestNames, releaseUrl, role)");
  });

  it("holds what the headless example drives and reads, and the download under its own name", () => {
    for (const id of ["release-url", "manifests", "videos", "video-index", "role", "backend", "panel", "notice", "video", "start"]) {
      expect(html.match(new RegExp(`id="${id}"`, "g")), id).toHaveLength(1);
    }
    expect(html).toContain('<a id="download" class="button primary" download="browser-events.jsonl" hidden>Download browser-events.jsonl</a>');
    expect(html).toContain("<title>cardvision: score a video in this browser</title>");
    const example = read("examples/score-headless.mjs");
    for (const used of ['getElementById("release-url")', '"#manifests"', '"#videos"', '"#video-index"', 'page.select("#role"', 'page.select("#backend"', 'getElementById("panel")', "window.cardvisionScore()"]) {
      expect(example, used).toContain(used);
    }
  });
});
