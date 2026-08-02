import { describe, expect, it } from "vitest";
import { gameDetailHref } from "./gameDetailHref";

describe("gameDetailHref", () => {
  it("returns WNBA espn path", () => {
    expect(
      gameDetailHref({
        league: "wnba",
        espnEventId: "401",
        mlbGamePk: null,
      }),
    ).toBe("/games/401");
  });

  it("returns MLB stub path", () => {
    expect(
      gameDetailHref({
        league: "mlb",
        espnEventId: null,
        mlbGamePk: "824971",
      }),
    ).toBe("/mlb/games/824971");
  });

  it("returns null for NBA / missing ids", () => {
    expect(
      gameDetailHref({ league: "nba", espnEventId: null, mlbGamePk: null }),
    ).toBeNull();
  });
});
