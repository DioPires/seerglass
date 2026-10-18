// SPDX-License-Identifier: Apache-2.0
import { describe, expect, it } from "vitest";

import { type AABB, rectOf } from "../src/boxes.js";
import { computeCardQuad, computeHitboxBox, type LayerCard, pointInConvexQuad, resolveHoveredCard } from "../src/hover-geometry.js";
import {
  DEFAULT_HOVER_SELECTION,
  HOVER_ID_PREFIX,
  HOVER_SELECTIONS,
  hoverEntriesToCards,
  HoverIdAssigner,
  hoverOrder,
  type HoverSelection,
  isHoverSelection,
  resolveHover,
  selectHoverLayer,
} from "../src/hover-layer.js";
import { keepAfterNms, nmsPreOrder } from "../src/recognition.js";
import { toLayerCards } from "../src/scene.js";
import { contract } from "./support.js";

const STAGE = { width: 1000, height: 500 };

function card(bounds: AABB, extra: Partial<LayerCard> = {}): LayerCard {
  const [card] = toLayerCards([{ instanceId: "c", aabb: bounds, rect: rectOf(bounds), identity: null }], { frameWidth: STAGE.width, frameHeight: STAGE.height, names: new Map() });
  return { ...card!, ...extra };
}

interface TestProposal {
  aabb: AABB;
  rect: ReturnType<typeof rectOf>;
  score: number;
  detectorScore: number;
  accepted: boolean;
  tauPassed: boolean;
  displayId: string;
}

function proposal(x: number, y: number, width: number, height: number, score: number, extra: Partial<TestProposal> = {}): TestProposal {
  const aabb = { x, y, width, height };
  return { aabb, rect: rectOf(aabb), score, detectorScore: 0.5, accepted: true, tauPassed: true, displayId: `ID-${x}-${y}`, ...extra };
}

function frameOf(proposals: TestProposal[]): { proposals: TestProposal[]; kept: number[] } {
  return { proposals, kept: keepAfterNms(proposals) };
}

describe("the hit test", () => {
  it("turns a box into four corners in pixels: top-left, top-right, bottom-right, bottom-left", () => {
    const quad = computeCardQuad(card({ x: 100, y: 50, width: 200, height: 100 }), STAGE);
    const flat = quad.points.flatMap((p) => [p.x, p.y]);
    [100, 50, 300, 50, 300, 150, 100, 150].forEach((v, i) => expect(flat[i]).toBeCloseTo(v, 9));
  });

  it("a point on an edge or a corner is inside", () => {
    const quad = computeCardQuad(card({ x: 100, y: 50, width: 200, height: 100 }), STAGE);
    for (const [x, y] of [[200, 100], [100, 100], [300, 100], [200, 50], [200, 150], [100, 50], [300, 150]] as const) {
      expect(pointInConvexQuad({ x, y }, quad), `${x},${y}`).toBe(true);
    }
    for (const [x, y] of [[99.999, 100], [300.001, 100], [200, 49.999], [200, 150.001], [0, 0]] as const) {
      expect(pointInConvexQuad({ x, y }, quad), `${x},${y}`).toBe(false);
    }
  });

  it("clamps the four numbers of a box one by one, so a box that starts off the frame keeps its width", () => {
    const quad = computeCardQuad(card({ x: -50, y: 100, width: 200, height: 100 }), STAGE);
    const flat = quad.points.flatMap((p) => [p.x, p.y]);
    [0, 100, 200, 100, 200, 200, 0, 200].forEach((v, i) => expect(flat[i]).toBeCloseTo(v, 9));
  });

  it("uses the card's own size about the center of its box, and turns it in pixels", () => {
    // A 200 x 100 px card turned a quarter: its axis-aligned box is 100 x 200 px about the same center.
    const turned: LayerCard = { ...card({ x: 450, y: 150, width: 100, height: 200 }), rotation: 90, localWidth: 0.2, localHeight: 0.2 };
    expect(computeHitboxBox(turned)).toEqual({ left: 0.5 - 0.1, top: 0.5 - 0.1, width: 0.2, height: 0.2 });
    const quad = computeCardQuad(turned, STAGE);
    // Half sizes 100 and 50 px turned a quarter: the outline spans x 450..550 and y 150..350.
    expect(pointInConvexQuad({ x: 455, y: 340 }, quad)).toBe(true);
    expect(pointInConvexQuad({ x: 440, y: 250 }, quad)).toBe(false);
  });

  it("the highest stack rank containing the point wins; on equal rank the later one", () => {
    const low = card({ x: 0, y: 0, width: 400, height: 400 });
    const high = card({ x: 100, y: 100, width: 100, height: 100 });
    const candidates = [
      { card: "low", quad: computeCardQuad(low, STAGE), zIndex: 1 },
      { card: "high", quad: computeCardQuad(high, STAGE), zIndex: 2 },
    ];
    expect(resolveHoveredCard({ x: 150, y: 150 }, candidates)).toBe("high");
    expect(resolveHoveredCard({ x: 50, y: 50 }, candidates)).toBe("low");
    expect(resolveHoveredCard({ x: 900, y: 450 }, candidates)).toBeNull();
    expect(resolveHoveredCard({ x: 150, y: 150 }, [...candidates].reverse())).toBe("high");
    const tied = [{ ...candidates[0]!, zIndex: 5 }, { ...candidates[1]!, zIndex: 5 }];
    expect(resolveHoveredCard({ x: 150, y: 150 }, tied)).toBe("high");
    expect(resolveHoveredCard({ x: 150, y: 150 }, [tied[1]!, tied[0]!])).toBe("low");
  });
});

describe("the three selections", () => {
  it("names them", () => {
    expect(HOVER_SELECTIONS).toEqual(["accepted-nms", "best-score", "detector-score"]);
    expect(DEFAULT_HOVER_SELECTION).toBe("accepted-nms");
    expect([isHoverSelection("best-score"), isHoverSelection("other"), isHoverSelection(3)]).toEqual([true, false, false]);
  });

  it("accepted-nms holds only the kept proposals, the smaller box on top", () => {
    const big = proposal(0, 0, 400, 400, 0.9);
    const small = proposal(150, 150, 100, 100, 0.7); // inside the big one: overlap 1/16
    const under = proposal(600, 0, 100, 100, 0.5, { tauPassed: false, accepted: false });
    const frame = frameOf([big, small, under]);
    expect(frame.kept.sort()).toEqual([0, 1]);
    expect(hoverOrder(frame, "accepted-nms")).toEqual([0, 1]);
    const layer = selectHoverLayer(frame, "accepted-nms");
    expect(layer.map((e) => [e.proposal, e.zIndex, e.identity])).toEqual([[0, 1, "ID-0-0"], [1, 2, "ID-150-150"]]);
  });

  it("a kept box that fails the margin still wins the pointer and shows nothing", () => {
    const big = proposal(0, 0, 400, 400, 0.9);
    const small = proposal(150, 150, 100, 100, 0.7, { accepted: false });
    const layer = selectHoverLayer(frameOf([big, small]), "accepted-nms");
    expect(layer[1]).toMatchObject({ proposal: 1, identity: null });
    const cards = hoverEntriesToCards(layer, ["a", "b"], { frameWidth: 1000, frameHeight: 500, names: new Map([["ID-0-0", "Big"]]) });
    expect(resolveHover(cards, { x: 200, y: 200 }, STAGE)).toMatchObject({ boxIndex: 1, cardId: null, name: null });
    expect(resolveHover(cards, { x: 50, y: 50 }, STAGE)).toMatchObject({ boxIndex: 0, cardId: "ID-0-0", name: "Big" });
    expect(resolveHover(cards, { x: 900, y: 50 }, STAGE)).toEqual({ card: null, boxIndex: null, box: null, cardId: null, name: null });
  });

  it("accepted-nms: on equal area the higher score is on top, then the earlier proposal", () => {
    const frame = { proposals: [proposal(0, 0, 100, 100, 0.7), proposal(300, 0, 100, 100, 0.9), proposal(600, 0, 100, 100, 0.9)], kept: [0, 1, 2] };
    expect(hoverOrder(frame, "accepted-nms")).toEqual([0, 2, 1]);
  });

  it("best-score holds every proposal, the higher score on top whatever its size, then the smaller box", () => {
    const frame = frameOf([proposal(0, 0, 400, 400, 0.9), proposal(150, 150, 100, 100, 0.7), proposal(600, 0, 100, 100, 0.5, { tauPassed: false, accepted: false }), proposal(800, 0, 50, 50, 0.7)]);
    expect(hoverOrder(frame, "best-score")).toEqual([2, 1, 3, 0]);
    expect(selectHoverLayer(frame, "best-score").map((e) => e.identity)).toEqual([null, "ID-150-150", "ID-800-0", "ID-0-0"]);
  });

  it("detector-score holds every proposal, the higher detector score on top, then the earlier proposal", () => {
    const frame = frameOf([
      proposal(0, 0, 100, 100, 0.9, { detectorScore: 0.6 }),
      proposal(200, 0, 100, 100, 0.7, { detectorScore: 0.9 }),
      proposal(400, 0, 100, 100, 0.8, { detectorScore: 0.6 }),
    ]);
    expect(hoverOrder(frame, "detector-score")).toEqual([2, 0, 1]);
  });

  it("stack ranks run 1 to n in order, bottom first", () => {
    const layer = selectHoverLayer(frameOf([proposal(0, 0, 100, 100, 0.9), proposal(300, 0, 50, 50, 0.8), proposal(600, 0, 200, 200, 0.7)]), "accepted-nms");
    expect(layer.map((e) => e.zIndex)).toEqual([1, 2, 3]);
    expect(layer.map((e) => e.proposal)).toEqual([2, 0, 1]);
  });

  it("stacks, hits and shows what the Python package does, for generated frames under every selection", () => {
    for (const [f, frame] of (contract.hover.frames as Array<Record<string, any>>).entries()) {
      const proposals = (frame.proposals as Array<Record<string, any>>).map((p) => {
        const aabb = { x: p.box[0], y: p.box[1], width: p.box[2], height: p.box[3] };
        return { aabb, rect: rectOf(aabb), score: p.similarity, detectorScore: p.detectorScore, accepted: p.accepted, tauPassed: p.tauPassed, displayId: p.cardId ?? "none", orientation: p.look };
      });
      const names = new Map<string, string>(frame.proposals.filter((p: any) => p.cardId).map((p: any) => [p.cardId, p.name]));
      const kept = keepAfterNms(proposals, nmsPreOrder(proposals));
      expect(kept, `frame ${f}: kept`).toEqual(frame.kept);
      for (const selection of HOVER_SELECTIONS) {
        const want = frame.selections[selection];
        expect(hoverOrder({ proposals, kept }, selection as HoverSelection), `frame ${f} ${selection}: order`).toEqual(want.order);
        const entries = selectHoverLayer({ proposals, kept }, selection as HoverSelection);
        const cards = hoverEntriesToCards(entries, entries.map((_, k) => `h${k}`), { frameWidth: frame.width, frameHeight: frame.height, names });
        const hits = (frame.points as Array<[number, number]>).map(([x, y]) => {
          const hit = resolveHover(cards, { x, y }, { width: frame.width, height: frame.height });
          return [hit.boxIndex, hit.cardId];
        });
        expect(hits, `frame ${f} ${selection}: hits`).toEqual(want.hits);
      }
    }
  });
});

describe("hover ids", () => {
  const entry = (x: number, identity: string | null, zIndex = 1) => {
    const aabb = { x, y: 0, width: 100, height: 100 };
    return { proposal: 0, aabb, rect: rectOf(aabb), identity, score: 0.9, zIndex };
  };

  it("keeps an id for the same identity while the box overlaps its last position by half or more", () => {
    const ids = new HoverIdAssigner();
    const [first] = ids.assign([entry(0, "A-1")]);
    expect(first).toBe(`${HOVER_ID_PREFIX}0001`);
    expect(ids.assign([entry(30, "A-1")])).toEqual([first]); // overlap 70 / 130
    expect(ids.assign([entry(55, "A-1")])).toEqual([first]); // 75 / 125 against the box at 30: it follows the card
  });

  it("gives a new id when the box moved too far or the identity changed; unnamed boxes keep theirs", () => {
    const ids = new HoverIdAssigner();
    const [a] = ids.assign([entry(0, "A-1")]);
    const [moved] = ids.assign([entry(40, "A-1")]); // overlap 60 / 140 < 0.5
    expect(moved).not.toBe(a);
    const [renamed] = ids.assign([entry(40, "B-1")]);
    expect(renamed).not.toBe(moved);
    const [unnamed] = ids.assign([entry(40, null)]);
    expect(ids.assign([entry(45, null)])).toEqual([unnamed]);
  });

  it("the entries stacked highest choose first, and a previous id is given once", () => {
    const ids = new HoverIdAssigner();
    const [only] = ids.assign([entry(0, "A-1")]);
    const next = ids.assign([entry(10, "A-1", 1), entry(5, "A-1", 2)]);
    expect(next[1]).toBe(only);
    expect(next[0]).not.toBe(only);
  });

  it("never reuses an id after a reset", () => {
    const ids = new HoverIdAssigner();
    const [a] = ids.assign([entry(0, "A-1")]);
    ids.reset();
    expect(ids.assign([entry(0, "A-1")])).toEqual([`${HOVER_ID_PREFIX}0002`]);
    expect(a).toBe(`${HOVER_ID_PREFIX}0001`);
  });
});

describe("layer cards", () => {
  const ctx = { frameWidth: 1000, frameHeight: 500, names: new Map([["A-1", "Alder"]]) };
  const item = (identity: string | null) => {
    const aabb = { x: 100, y: 50, width: 200, height: 100 };
    return { instanceId: "t1", aabb, rect: rectOf(aabb), identity };
  };

  it("a card with an identity the catalog knows is named", () => {
    const [named] = toLayerCards([item("A-1")], ctx);
    expect(named).toEqual({
      instanceId: "t1", bounds: { x: 0.1, y: 0.1, width: 0.2, height: 0.2 }, rotation: 0, localWidth: 0.2, localHeight: 0.2,
      landscape: true, cardId: "A-1", name: "Alder",
    });
  });

  it("a card without an identity, or with one the catalog does not hold, carries neither id nor name", () => {
    for (const identity of [null, "Z-9"]) {
      const [unnamed] = toLayerCards([item(identity)], ctx);
      expect([unnamed!.cardId, unnamed!.name]).toEqual([null, null]);
    }
  });

  it("clamps to the frame and folds any angle into (-45, 45]", () => {
    const aabb = { x: -100, y: 450, width: 1500, height: 100 };
    const [clamped] = toLayerCards([{ instanceId: "t", aabb, rect: { ...rectOf(aabb), angle: 80 }, identity: null }], ctx);
    expect(clamped!.bounds).toEqual({ x: 0, y: 0.9, width: 1, height: 0.2 });
    expect(clamped!.rotation).toBe(-10);
  });
});
