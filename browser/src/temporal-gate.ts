// SPDX-License-Identifier: Apache-2.0
// The temporal gate: when the track layer may be shown at all.
//
// One frame that happens to match a card in footage with no cards in it should not put an
// outline on screen, and one frame in which recognition finds nothing (a transition, motion
// blur) should not blank a table full of cards. So showing starts only after `k` consecutive
// analyzed frames that each recognized at least one card, and stops only after `k` consecutive
// frames that recognized none. A frame recognized a card when at least one of its detections
// reached tau (the detections are the tracker's input; the margin is not asked for here).
//
// The gate counts analyzed frames, not time, and holds no pixels, so it can be tested without
// a browser. It governs the track layer only; the hover layer answers from one frame.

/** k: consecutive frames needed to start showing, and to stop. */
export const CONFIRM_FRAMES = 3;

export interface TemporalGateStep {
  /** Whether the track layer may be shown after this frame. */
  confirmed: boolean;
  /** Length of the current run of frames that recognized a card (0 after one that did not). */
  consecutive: number;
}

export class TemporalGate {
  private run = 0;
  private misses = 0;
  private showing = false;

  constructor(private readonly k: number = CONFIRM_FRAMES) {
    if (!Number.isInteger(k) || k < 1) throw new RangeError(`k must be a positive integer, got ${k}`);
  }

  /** The picture is no longer continuous with the last one: back to not showing, with no run. */
  reset(): void {
    this.run = 0;
    this.misses = 0;
    this.showing = false;
  }

  /** Feed one analyzed frame: did it recognize at least one card? */
  observe(recognized: boolean): TemporalGateStep {
    if (recognized) {
      this.run += 1;
      this.misses = 0;
      if (this.run >= this.k) this.showing = true;
    } else {
      this.run = 0;
      this.misses += 1;
      if (this.misses >= this.k) this.showing = false;
    }
    return { confirmed: this.showing, consecutive: this.run };
  }
}
