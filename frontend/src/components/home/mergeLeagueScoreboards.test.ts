import { describe, expect, it } from "vitest";
import { mergeLeagueScoreboards } from "./mergeLeagueScoreboards";
import type { LiveGame, TickerGame } from "./types";

const wnbaTicker: TickerGame = {
  id: "wnba-1",
  league: "wnba",
  espnEventId: "401857098",
  mlbGamePk: null,
  awayAbbrev: "ATL",
  homeAbbrev: "DAL",
  statusLabel: "Q3 7:13",
  status: "live",
  awayScore: 36,
  homeScore: 44,
};

const wnbaLive: LiveGame = {
  id: "wnba-1",
  league: "wnba",
  espnEventId: "401857098",
  mlbGamePk: null,
  status: "live",
  statusLabel: "Q3 7:13",
  away: { abbrev: "ATL", name: "Atlanta Dream", score: 36, logoUrl: null },
  home: { abbrev: "DAL", name: "Dallas Wings", score: 44, logoUrl: null },
};

const mlbTicker: TickerGame = {
  id: "mlb-9",
  league: "mlb",
  espnEventId: null,
  mlbGamePk: "9",
  awayAbbrev: "BOS",
  homeAbbrev: "NYY",
  statusLabel: "Top 3rd",
  status: "live",
  awayScore: 2,
  homeScore: 3,
};

const mlbLive: LiveGame = {
  id: "mlb-9",
  league: "mlb",
  espnEventId: null,
  mlbGamePk: "9",
  status: "live",
  statusLabel: "Top 3rd",
  away: { abbrev: "BOS", name: "Boston Red Sox", score: 2, logoUrl: null },
  home: { abbrev: "NYY", name: "New York Yankees", score: 3, logoUrl: null },
};

describe("mergeLeagueScoreboards", () => {
  it("keeps WNBA games when MLB never loaded", () => {
    const merged = mergeLeagueScoreboards([
      {
        tickerGames: [wnbaTicker],
        liveGames: [wnbaLive],
        isLoading: false,
        hasNeverLoaded: false,
      },
      {
        tickerGames: [],
        liveGames: [],
        isLoading: false,
        hasNeverLoaded: true,
      },
    ]);
    expect(merged.tickerGames).toHaveLength(1);
    expect(merged.liveGames).toHaveLength(1);
    expect(merged.hasNeverLoaded).toBe(false);
  });

  it("marks hasNeverLoaded only when every part never loaded", () => {
    const merged = mergeLeagueScoreboards([
      {
        tickerGames: [],
        liveGames: [],
        isLoading: false,
        hasNeverLoaded: true,
      },
      {
        tickerGames: [],
        liveGames: [],
        isLoading: false,
        hasNeverLoaded: true,
      },
    ]);
    expect(merged.hasNeverLoaded).toBe(true);
  });

  it("concatenates ticker and live games in part order", () => {
    const merged = mergeLeagueScoreboards([
      {
        tickerGames: [wnbaTicker],
        liveGames: [wnbaLive],
        isLoading: false,
        hasNeverLoaded: false,
      },
      {
        tickerGames: [mlbTicker],
        liveGames: [mlbLive],
        isLoading: false,
        hasNeverLoaded: false,
      },
    ]);
    expect(merged.tickerGames.map((g) => g.id)).toEqual(["wnba-1", "mlb-9"]);
    expect(merged.liveGames.map((g) => g.id)).toEqual(["wnba-1", "mlb-9"]);
  });

  it("is loading when any part is loading and merged live list is empty", () => {
    const merged = mergeLeagueScoreboards([
      {
        tickerGames: [],
        liveGames: [],
        isLoading: true,
        hasNeverLoaded: false,
      },
      {
        tickerGames: [],
        liveGames: [],
        isLoading: false,
        hasNeverLoaded: false,
      },
    ]);
    expect(merged.isLoading).toBe(true);
  });

  it("is not loading when live games already exist", () => {
    const merged = mergeLeagueScoreboards([
      {
        tickerGames: [wnbaTicker],
        liveGames: [wnbaLive],
        isLoading: false,
        hasNeverLoaded: false,
      },
      {
        tickerGames: [],
        liveGames: [],
        isLoading: true,
        hasNeverLoaded: false,
      },
    ]);
    expect(merged.isLoading).toBe(false);
    expect(merged.liveGames).toHaveLength(1);
  });

  it("shouldPoll when any part wants poll", () => {
    const merged = mergeLeagueScoreboards([
      {
        tickerGames: [wnbaTicker],
        liveGames: [wnbaLive],
        isLoading: false,
        hasNeverLoaded: false,
        shouldPoll: false,
      },
      {
        tickerGames: [mlbTicker],
        liveGames: [mlbLive],
        isLoading: false,
        hasNeverLoaded: false,
        shouldPoll: true,
      },
    ]);
    expect(merged.shouldPoll).toBe(true);
  });
});
