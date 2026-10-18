// SPDX-License-Identifier: Apache-2.0
// Writes what details.ts decided into the Details of the demo page. It decides nothing itself:
// every word and figure comes from the DetailsView it is given.

import type { DetailsView, DetectionRow, Fact, Sparkline, Tile } from "./details.js";
import { byId } from "./page.js";
import type { Tone } from "./view.js";

const SVG = "http://www.w3.org/2000/svg";

/** An element with a class and its text, or its children. */
function el<K extends keyof HTMLElementTagNameMap>(tag: K, className: string | null, ...content: Array<Node | string>): HTMLElementTagNameMap[K] {
  const element = document.createElement(tag);
  if (className) element.className = className;
  element.append(...content);
  return element;
}

function svg<K extends keyof SVGElementTagNameMap>(tag: K, attributes: Record<string, string>): SVGElementTagNameMap[K] {
  const element = document.createElementNS(SVG, tag);
  for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
  return element;
}

/** The width and height of the sparkline's drawing, in its own units: the drawing is stretched
 *  to the width of its tile, and its lines keep their width (`vector-effect`). */
const SPARK_W = 100;
const SPARK_H = 28;

/** The worker's time over the last frames: a line from zero to the highest time, the area under
 *  it, and a dot on the newest frame. */
function sparkline(spark: Sparkline): HTMLElement {
  const n = spark.points.length;
  const top = spark.max > 0 ? spark.max : 1;
  const xy = spark.points.map((value, i) => [n === 1 ? SPARK_W : (i / (n - 1)) * SPARK_W, SPARK_H - (value / top) * (SPARK_H - 2)] as const);
  const line = xy.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(" ");
  const words = `Time per frame over the last ${n} ${n === 1 ? "frame" : "frames"}: ${spark.min.toFixed(1)} to ${spark.max.toFixed(1)} ms, the newest ${spark.latest.toFixed(1)} ms`;
  const drawing = svg("svg", { viewBox: `0 0 ${SPARK_W} ${SPARK_H}`, preserveAspectRatio: "none", role: "img", "aria-label": words, class: "spark" });
  if (n > 1) {
    drawing.append(svg("polygon", { class: "spark-area", points: `0,${SPARK_H} ${line} ${SPARK_W},${SPARK_H}` }));
    drawing.append(svg("polyline", { class: "spark-line", points: line, "vector-effect": "non-scaling-stroke" }));
  }
  const [lx, ly] = xy[n - 1]!;
  // A line of no length with round ends draws a dot that the stretch does not turn into an oval.
  drawing.append(svg("line", { class: "spark-dot", x1: String(lx), y1: String(ly), x2: String(lx), y2: String(ly), "vector-effect": "non-scaling-stroke" }));
  const range = n === 1 ? `${spark.latest.toFixed(1)} ms` : `${spark.min.toFixed(1)}–${spark.max.toFixed(1)} ms`;
  return el("dd", "spark-box", drawing, el("span", "spark-caption", `last ${n} ${n === 1 ? "frame" : "frames"} · ${range}`));
}

function tile(item: Tile, spark: Sparkline | null): HTMLElement {
  const figure = el("dd", "figure", el("span", "value", item.value));
  if (item.unit !== "") figure.append(el("span", "unit", item.unit));
  const box = el("div", "tile", el("dt", null, item.label), figure, el("dd", "note", item.note));
  box.dataset.key = item.key;
  if (item.key === "time" && spark) box.append(sparkline(spark));
  return box;
}

/** A sample of the line the outlines of "Show detected boxes" draw for this kind of box. */
const swatch = (row: DetectionRow): HTMLElement => {
  const mark = el("span", "swatch");
  mark.setAttribute("aria-hidden", "true");
  mark.dataset.kind = row.kind;
  return mark;
};

function detectionRow(row: DetectionRow): HTMLTableRowElement {
  const card = el("td", "name", el("span", row.kind === "named" ? "strong" : "muted", row.card));
  if (row.turned) card.append(" ", el("span", "muted", row.turned));
  if (row.pointer) card.append(" ", el("span", "pill", "under the pointer"));
  const id = el("td", "id");
  if (row.id !== null) id.append(row.nearest ? el("span", "muted", "nearest ", el("code", null, row.id)) : el("code", null, row.id));
  else id.append(el("span", "muted", "—"));
  const figure = (value: string | null, low: boolean): HTMLElement => {
    const cell = el("td", "num", value ?? "—");
    if (value === null) cell.classList.add("muted");
    if (low) cell.classList.add("low");
    return cell;
  };
  const tr = el("tr", null, el("td", "num muted", String(row.number)), card, id, figure(row.score, row.scoreLow), figure(row.margin, row.marginLow), el("td", "result", swatch(row), row.result));
  tr.dataset.kind = row.kind;
  if (row.pointer) tr.dataset.pointer = "";
  return tr;
}

function facts(list: HTMLElement, items: readonly Fact[]): void {
  list.replaceChildren(
    ...items.map((item) => {
      const value = el("dd", null, item.value);
      if (item.full !== undefined) value.title = item.full;
      return el("div", null, el("dt", null, item.label), value);
    }),
  );
}

/** Write a DetailsView into the page's Details. */
export function renderDetails(view: DetailsView, marks: Record<Tone, string>): void {
  const state = byId<HTMLParagraphElement>("diag-state");
  state.dataset.tone = view.tone;
  byId("diag-mark").textContent = marks[view.tone];
  byId("diag-text").textContent = view.line;

  const tiles = byId<HTMLDListElement>("tiles");
  tiles.hidden = view.tiles === null;
  tiles.replaceChildren(...(view.tiles ?? []).map((item) => tile(item, view.spark)));

  const section = byId<HTMLElement>("detections");
  section.hidden = view.frame === null;
  const rows = view.frame?.rows ?? [];
  byId("detections-caption").textContent = view.frame?.caption ?? "";
  const empty = byId<HTMLParagraphElement>("detections-empty");
  empty.hidden = view.frame?.empty == null;
  empty.textContent = view.frame?.empty ?? "";
  byId("detection-table").hidden = rows.length === 0;
  byId("detection-rows").replaceChildren(...rows.map(detectionRow));

  byId("model-group").hidden = view.model === null;
  facts(byId("model-facts"), view.model ?? []);
  byId("file-group").hidden = view.file === null;
  facts(byId("file-facts"), view.file ?? []);
}
