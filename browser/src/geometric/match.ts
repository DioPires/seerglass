// SPDX-License-Identifier: Apache-2.0
// Matching a crop's keypoints: the whole-catalog vote and the matches with one candidate.
// In Python: cardvision/geometric/stage.py (`votes`, `inliers`).

import type { Cv } from "./features.js";
import { type KnnResult, NO_ROW } from "./vote.js";

export { NO_ROW };

/** The two nearest index rows of each query descriptor, nearest first; on equal distances the
 *  lower row first. `NO_ROW` where there is no such row. (The voter's result, vote.ts.) */
export type KnnPairs = KnnResult;

/**
 * The vote (Python: `GeometricStage.votes`): index card position to the number of query
 * descriptors whose nearest index descriptor is that card's and that pass the ratio test. The
 * test is waived when the two nearest descriptors belong to one card, to cards of one identity,
 * or to cards taken from one image: such neighbours are not rivals. `rowCard` maps an index row
 * to its card's position; `identity` and `image` give each position's identity and image.
 * Positions appear in the order they first got a vote, as in Python's dictionary.
 */
export function countVotes(
  knn: KnnPairs,
  queries: number,
  first: number,
  rowCard: Uint32Array,
  identity: readonly string[],
  image: readonly string[],
  ratio: number,
): Map<number, number> {
  const counts = new Map<number, number>();
  for (let q = first; q < first + queries; q++) {
    const bestRow = knn.best[q]!;
    if (bestRow === NO_ROW) continue;
    const position = rowCard[bestRow]!;
    const secondRow = knn.second[q]!;
    if (secondRow !== NO_ROW) {
      const other = rowCard[secondRow]!;
      const rivals = other !== position && identity[other] !== identity[position] && image[other] !== image[position];
      if (rivals && knn.dBest[q]! >= ratio * knn.dSecond[q]!) continue;
    }
    counts.set(position, (counts.get(position) ?? 0) + 1);
  }
  return counts;
}

/** A crop keypoint matched with a candidate's keypoint. */
export interface Match {
  queryIdx: number;
  trainIdx: number;
}

/**
 * The matches of a crop's descriptors with one candidate's that pass the ratio test (Python:
 * the `good` list of `GeometricStage.inliers`): OpenCV's brute-force Hamming matcher, the two
 * nearest per crop keypoint, kept when the nearest is under `ratio` times the second. In the
 * order of the crop's keypoints.
 */
export function ratioMatches(
  cv: Cv,
  query: Uint8Array,
  queryCount: number,
  train: Uint8Array,
  trainCount: number,
  ratio: number,
  width = 32,
): Match[] {
  const queryMat = new cv.Mat(queryCount, width, cv.CV_8UC1);
  const trainMat = new cv.Mat(trainCount, width, cv.CV_8UC1);
  const matcher = new cv.BFMatcher(cv.NORM_HAMMING, false);
  const pairs = new cv.DMatchVectorVector();
  try {
    queryMat.data.set(query.subarray(0, queryCount * width));
    trainMat.data.set(train.subarray(0, trainCount * width));
    matcher.knnMatch(queryMat, trainMat, pairs, 2);
    const good: Match[] = [];
    const total: number = pairs.size();
    for (let i = 0; i < total; i++) {
      const pair = pairs.get(i);
      try {
        if (pair.size() !== 2) continue;
        const nearest = pair.get(0);
        const second = pair.get(1);
        if (nearest.distance < ratio * second.distance) good.push({ queryIdx: nearest.queryIdx, trainIdx: nearest.trainIdx });
      } finally {
        pair.delete?.();
      }
    }
    return good;
  } finally {
    queryMat.delete();
    trainMat.delete();
    matcher.delete();
    pairs.delete();
  }
}
