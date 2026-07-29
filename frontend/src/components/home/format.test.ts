import { describe, expect, it } from "vitest";
import {
  LIVE_NOW_SKELETON_COUNT,
  formatGamesInProgress,
  normalizeLiveGames,
} from "./format";

describe("formatGamesInProgress", () => {
  it("formats zero and plural counts", () => {
    expect(formatGamesInProgress(0)).toBe("0 games in progress");
    expect(formatGamesInProgress(3)).toBe("3 games in progress");
  });

  it("uses singular for one game", () => {
    expect(formatGamesInProgress(1)).toBe("1 game in progress");
  });

  it("rejects invalid counts", () => {
    expect(() => formatGamesInProgress(-1)).toThrow(/non-negative/);
    expect(() => formatGamesInProgress(Number.NaN)).toThrow(/non-negative/);
  });
});

describe("normalizeLiveGames", () => {
  it("treats undefined and null as empty", () => {
    expect(normalizeLiveGames(undefined)).toEqual([]);
    expect(normalizeLiveGames(null)).toEqual([]);
  });

  it("returns the same array when provided", () => {
    const games = [{ id: "1" }];
    expect(normalizeLiveGames(games)).toBe(games);
  });

  it("rejects non-arrays", () => {
    expect(() => normalizeLiveGames({} as never)).toThrow(/array/);
  });
});

describe("LIVE_NOW_SKELETON_COUNT", () => {
  it("shows three skeleton cards when empty", () => {
    expect(LIVE_NOW_SKELETON_COUNT).toBe(3);
  });
});
