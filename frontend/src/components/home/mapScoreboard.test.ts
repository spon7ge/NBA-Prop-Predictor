import { describe, expect, it } from "vitest";
import type { ApiWnbaGame } from "@/lib/api";
import {
  mapToLiveGames,
  mapToTickerGames,
  shouldPollScoreboard,
} from "./mapScoreboard";

function apiGame(overrides: Partial<ApiWnbaGame> = {}): ApiWnbaGame {
  return {
    id: "g1",
    league: "wnba",
    status: "live",
    status_label: "Q3 7:13",
    start_time_et: "2026-07-29T23:00:00Z",
    away: { abbrev: "ATL", name: "Atlanta Dream", score: 36 },
    home: { abbrev: "DAL", name: "Dallas Wings", score: 44 },
    ...overrides,
  };
}

describe("shouldPollScoreboard", () => {
  it("does not poll an empty slate", () => {
    expect(shouldPollScoreboard([])).toBe(false);
  });

  it("does not poll when games are undefined", () => {
    expect(shouldPollScoreboard(undefined)).toBe(false);
  });

  it("does not poll when every game is final", () => {
    expect(
      shouldPollScoreboard([apiGame({ status: "final", status_label: "Final" })]),
    ).toBe(false);
  });

  it("polls when any game is not final", () => {
    expect(
      shouldPollScoreboard([
        apiGame({ id: "a", status: "final", status_label: "Final" }),
        apiGame({ id: "b", status: "halftime", status_label: "Halftime" }),
      ]),
    ).toBe(true);
  });
});

describe("scoreboard mappers", () => {
  it("maps API games to ticker games", () => {
    expect(mapToTickerGames([apiGame()])).toEqual([
      {
        id: "g1",
        league: "wnba",
        awayAbbrev: "ATL",
        homeAbbrev: "DAL",
        statusLabel: "Q3 7:13",
        status: "live",
      },
    ]);
  });

  it("maps API games to live games and preserves null scores", () => {
    const scheduled = apiGame({
      status: "scheduled",
      status_label: "7:00 PM ET",
      away: { abbrev: "NYL", name: "New York Liberty", score: null },
      home: { abbrev: "LVA", name: "Las Vegas Aces", score: null },
    });
    expect(mapToLiveGames([scheduled])[0]).toEqual({
      id: "g1",
      league: "wnba",
      statusLabel: "7:00 PM ET",
      status: "scheduled",
      away: { abbrev: "NYL", name: "New York Liberty", score: null },
      home: { abbrev: "LVA", name: "Las Vegas Aces", score: null },
    });
  });
});
