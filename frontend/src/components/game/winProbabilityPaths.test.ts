import { describe, expect, it } from "vitest";
import {
  buildSeriesPathD,
  buildSplitSeriesPaths,
  xForIndex,
  yForPct,
} from "./winProbabilityPaths";

const points = [
  { awayWinPct: 56, homeWinPct: 44 },
  { awayWinPct: 46, homeWinPct: 54 },
  { awayWinPct: 40, homeWinPct: 60 },
];

describe("winProbabilityPaths", () => {
  it("maps index 0 to the left of the plot and 100% near the top", () => {
    expect(xForIndex(0, 3)).toBeLessThan(xForIndex(2, 3));
    expect(yForPct(100)).toBeLessThan(yForPct(0));
  });

  it("builds a polyline for a home series range", () => {
    const d = buildSeriesPathD(points, "home", 0, 2);
    expect(d.startsWith("M")).toBe(true);
    expect(d.includes(" L")).toBe(true);
  });

  it("returns empty string for an inverted range", () => {
    expect(buildSeriesPathD(points, "away", 2, 1)).toBe("");
  });

  it("splits vivid through scrub and muted after, sharing the scrub point", () => {
    const split = buildSplitSeriesPaths(points, 1);
    expect(split.homeVivid).toBe(buildSeriesPathD(points, "home", 0, 1));
    expect(split.homeMuted).toBe(buildSeriesPathD(points, "home", 1, 2));
    expect(split.awayVivid).toBe(buildSeriesPathD(points, "away", 0, 1));
    expect(split.awayMuted).toBe(buildSeriesPathD(points, "away", 1, 2));
  });

  it("omits muted paths when scrub is at the last point", () => {
    const split = buildSplitSeriesPaths(points, 2);
    expect(split.homeMuted).toBe("");
    expect(split.awayMuted).toBe("");
    expect(split.homeVivid).toBe(buildSeriesPathD(points, "home", 0, 2));
  });
});
