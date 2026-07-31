import { describe, expect, it } from "vitest";
import { formatOddsPill, mergeMatchupOdds } from "./mergeMatchupOdds";
import type { MatchupGame } from "./types";

const baseGame: MatchupGame = {
  id: "1",
  espnEventId: "401",
  league: "wnba",
  status: "scheduled",
  statusLabel: "8:00 PM ET",
  away: {
    abbrev: "SEA",
    name: "Seattle Storm",
    score: null,
    logoUrl: null,
  },
  home: {
    abbrev: "ATL",
    name: "Atlanta Dream",
    score: null,
    logoUrl: null,
  },
};

describe("mergeMatchupOdds", () => {
  it("merges by home and away abbrev", () => {
    const merged = mergeMatchupOdds([baseGame], [
      {
        home_abbrev: "ATL",
        away_abbrev: "SEA",
        spread_team_abbrev: "ATL",
        spread_line: -12.5,
        total: 179.5,
      },
    ]);
    expect(merged[0].odds).toEqual({
      spreadTeamAbbrev: "ATL",
      spreadLine: -12.5,
      total: 179.5,
    });
  });

  it("leaves unmatched games without odds", () => {
    const merged = mergeMatchupOdds([baseGame], [
      {
        home_abbrev: "DAL",
        away_abbrev: "WAS",
        spread_team_abbrev: "DAL",
        spread_line: -3.5,
        total: 167.5,
      },
    ]);
    expect(merged[0].odds).toBeNull();
  });
});

describe("formatOddsPill", () => {
  it("formats full and partial pills", () => {
    expect(
      formatOddsPill({
        spreadTeamAbbrev: "ATL",
        spreadLine: -12.5,
        total: 179.5,
      }),
    ).toBe("Spread: ATL -12.5 · Total: 179.5");
    expect(
      formatOddsPill({
        spreadTeamAbbrev: "ATL",
        spreadLine: -12.5,
        total: null,
      }),
    ).toBe("Spread: ATL -12.5");
    expect(
      formatOddsPill({
        spreadTeamAbbrev: null,
        spreadLine: null,
        total: 178.5,
      }),
    ).toBe("Total: 178.5");
    expect(
      formatOddsPill({
        spreadTeamAbbrev: null,
        spreadLine: null,
        total: null,
      }),
    ).toBeNull();
  });
});
