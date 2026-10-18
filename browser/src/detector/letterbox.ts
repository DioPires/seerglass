// SPDX-License-Identifier: Apache-2.0
// The detector's input contract: how an analysis frame becomes the detector's input tensor.
// In Python: cardvision/detect/letterbox.py.
//
//   ratio  = min(inputHeight / H, inputWidth / W)
//   size   = round(W * ratio) x round(H * ratio)        (halves go to the even neighbor)
//   canvas = inputWidth x inputHeight, filled with the pad value
//   the frame, resized to `size` with Pillow's bilinear filter, sits in the top-left corner
//   tensor = the canvas as float32, channel first, in the order blue, green, red, 0 to 255
//
// No mean is subtracted and nothing is divided. A coordinate on the canvas divided by the ratio
// is a coordinate in the frame; there is no offset.

import { resizeBilinearRgb } from "../resize.js";

/** The canvas the frame is fitted into: the model's input size and the fill value. */
export interface LetterboxSettings {
  width: number;
  height: number;
  pad: number;
}

/** The input of the published detector. A release document states its own. */
export const DEFAULT_LETTERBOX: Readonly<LetterboxSettings> = { width: 640, height: 384, pad: 114 };

/** Round to the nearest integer, halves to the even neighbor (as Python's `round`). */
export function roundHalfEven(v: number): number {
  const f = Math.floor(v);
  const frac = v - f;
  if (frac > 0.5) return f + 1;
  if (frac < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

export interface Letterbox {
  /** Canvas pixels per frame pixel. Boxes are divided by it to return to the frame. */
  ratio: number;
  /** Size of the resized frame inside the canvas. */
  width: number;
  height: number;
}

export function letterboxGeometry(srcW: number, srcH: number, settings: LetterboxSettings = DEFAULT_LETTERBOX): Letterbox {
  const ratio = Math.min(settings.height / srcH, settings.width / srcW);
  return { ratio, width: roundHalfEven(srcW * ratio), height: roundHalfEven(srcH * ratio) };
}

/** The canvas as packed bytes, row-major, three bytes per pixel in the order blue, green, red. */
export function letterboxBgr(
  rgba: Uint8ClampedArray | Uint8Array,
  srcW: number,
  srcH: number,
  settings: LetterboxSettings = DEFAULT_LETTERBOX,
): Letterbox & { canvas: Uint8Array } {
  const lb = letterboxGeometry(srcW, srcH, settings);
  const rgb = resizeBilinearRgb(rgba, srcW, srcH, lb.width, lb.height);
  const canvas = new Uint8Array(settings.width * settings.height * 3).fill(settings.pad);
  for (let y = 0; y < lb.height; y++) {
    for (let x = 0; x < lb.width; x++) {
      const s = (y * lb.width + x) * 3;
      const d = (y * settings.width + x) * 3;
      canvas[d] = rgb[s + 2]!;
      canvas[d + 1] = rgb[s + 1]!;
      canvas[d + 2] = rgb[s]!;
    }
  }
  return { ...lb, canvas };
}

/** An RGBA analysis frame as the detector's input tensor (`3 * height * width` float32 values:
 *  the blue plane, then green, then red) and the letterbox geometry. */
export function toDetectorTensor(
  rgba: Uint8ClampedArray | Uint8Array,
  srcW: number,
  srcH: number,
  settings: LetterboxSettings = DEFAULT_LETTERBOX,
): Letterbox & { tensor: Float32Array } {
  const lb = letterboxGeometry(srcW, srcH, settings);
  const rgb = resizeBilinearRgb(rgba, srcW, srcH, lb.width, lb.height);
  const plane = settings.width * settings.height;
  const tensor = new Float32Array(3 * plane).fill(settings.pad);
  for (let y = 0; y < lb.height; y++) {
    for (let x = 0; x < lb.width; x++) {
      const s = (y * lb.width + x) * 3;
      const d = y * settings.width + x;
      tensor[d] = rgb[s + 2]!;
      tensor[plane + d] = rgb[s + 1]!;
      tensor[2 * plane + d] = rgb[s]!;
    }
  }
  return { ...lb, tensor };
}
