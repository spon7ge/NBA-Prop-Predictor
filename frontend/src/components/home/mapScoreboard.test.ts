import { describe, expect, it } from "vitest";
import type { ApiMlbGame, ApiWnbaGame } from "@/lib/api";
import {
  mapToLiveGames,
  mapToMatchupGames,
  mapToTickerGames,
  shouldPollScoreboard,
} from "./mapScoreboard";

function apiGame(overrides: Partial<ApiWnbaGame> = {}): ApiWnbaGame {
  return {
    id: "g1",
    espn_event_id: null,
    league: "wnba",
    status: "live",
    status_label: "Q3 7:13",
    start_time_et: "2026-07-29T23:00:00Z",
    away: { abbrev: "ATL", name: "Atlanta Dream", score: 36, logo_url: null },
    home: { abbrev: "DAL", name: "Dallas Wings", score: 44, logo_url: null },
    ...overrides,
  };
}

function mlbApiGame(overrides: Partial<ApiMlbGame> = {}): ApiMlbGame {
  return {
    id: "mlb-9",
    mlb_game_pk: "9",
    league: "mlb",
    status: "live",
    status_label: "Top 3rd",
    start_time_et: "2026-08-02T23:00:00Z",
    venue: "Yankee Stadium",
    venue_city: "New York",
    away: {
      abbrev: "BOS",
      name: "Boston Red Sox",
      score: 2,
      record: "50-40",
      logo_url: null,
    },
    home: {
      abbrev: "NYY",
      name: "New York Yankees",
      score: 3,
      record: "55-35",
      logo_url: null,
    },
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
        espnEventId: null,
        mlbGamePk: null,
        league: "wnba",
        awayAbbrev: "ATL",
        homeAbbrev: "DAL",
        statusLabel: "Q3 7:13",
        status: "live",
        awayScore: 36,
        homeScore: 44,
      },
    ]);
  });

  it("maps null scores for scheduled ticker games", () => {
    const scheduled = apiGame({
      status: "scheduled",
      status_label: "7:00 PM ET",
      away: { abbrev: "NYL", name: "New York Liberty", score: null, logo_url: null },
      home: { abbrev: "LVA", name: "Las Vegas Aces", score: null, logo_url: null },
    });
    expect(mapToTickerGames([scheduled])[0]).toMatchObject({
      awayScore: null,
      homeScore: null,
      status: "scheduled",
    });
  });

  it("maps API games to live games and preserves null scores", () => {
    const scheduled = apiGame({
      status: "scheduled",
      status_label: "7:00 PM ET",
      away: { abbrev: "NYL", name: "New York Liberty", score: null, logo_url: null },
      home: { abbrev: "LVA", name: "Las Vegas Aces", score: null, logo_url: null },
    });
    expect(mapToLiveGames([scheduled])[0]).toEqual({
      id: "g1",
      espnEventId: null,
      mlbGamePk: null,
      league: "wnba",
      statusLabel: "7:00 PM ET",
      status: "scheduled",
      away: { abbrev: "NYL", name: "New York Liberty", score: null, logoUrl: null },
      home: { abbrev: "LVA", name: "Las Vegas Aces", score: null, logoUrl: null },
    });
  });

  it("maps espn_event_id to espnEventId on ticker and live games", () => {
    const game = apiGame({ espn_event_id: "401857098" });
    expect(mapToTickerGames([game])[0].espnEventId).toBe("401857098");
    expect(mapToLiveGames([game])[0].espnEventId).toBe("401857098");
  });

  it("maps logo_url to logoUrl for live and matchup games", () => {
    const game = apiGame({
      away: {
        abbrev: "ATL",
        name: "Atlanta Dream",
        score: 36,
        logo_url: "https://a.espncdn.com/i/teamlogos/wnba/500/atl.png",
      },
      home: {
        abbrev: "DAL",
        name: "Dallas Wings",
        score: 44,
        logo_url: "https://a.espncdn.com/i/teamlogos/wnba/500/dal.png",
      },
    });
    const live = mapToLiveGames([game])[0];
    expect(live.away.logoUrl).toBe(
      "https://a.espncdn.com/i/teamlogos/wnba/500/atl.png",
    );
    expect(live.home.logoUrl).toBe(
      "https://a.espncdn.com/i/teamlogos/wnba/500/dal.png",
    );
    const matchup = mapToMatchupGames([game])[0];
    expect(matchup.away.logoUrl).toBe(
      "https://a.espncdn.com/i/teamlogos/wnba/500/atl.png",
    );
    expect(matchup.home.logoUrl).toBe(
      "https://a.espncdn.com/i/teamlogos/wnba/500/dal.png",
    );
  });

  it("maps null logo_url to null logoUrl", () => {
    const live = mapToLiveGames([apiGame()])[0];
    expect(live.away.logoUrl).toBeNull();
    expect(live.home.logoUrl).toBeNull();
  });

  it("mapToMatchupGames maps all games with venue and records", () => {
    const games: ApiWnbaGame[] = [
      {
        id: "1",
        espn_event_id: "401",
        league: "wnba",
        status: "final",
        status_label: "Final",
        away: {
          abbrev: "ATL",
          name: "Atlanta Dream",
          score: 82,
          record: "17-10",
          logo_url: null,
        },
        home: {
          abbrev: "DAL",
          name: "Dallas Wings",
          score: 81,
          record: "18-10",
          logo_url: null,
        },
        start_time_et: "2026-07-29T23:00:00Z",
        venue: "College Park Center",
        venue_city: "Arlington",
      },
    ];
    expect(mapToMatchupGames(games)).toEqual([
      {
        id: "1",
        espnEventId: "401",
        mlbGamePk: null,
        league: "wnba",
        status: "final",
        statusLabel: "Final",
        venue: "College Park Center",
        venueCity: "Arlington",
        away: {
          abbrev: "ATL",
          name: "Atlanta Dream",
          score: 82,
          record: "17-10",
          logoUrl: null,
        },
        home: {
          abbrev: "DAL",
          name: "Dallas Wings",
          score: 81,
          record: "18-10",
          logoUrl: null,
        },
      },
    ]);
  });

  it("maps MLB mlb_game_pk and leaves espnEventId null", () => {
    const game = mlbApiGame();
    const ticker = mapToTickerGames([game])[0];
    const live = mapToLiveGames([game])[0];
    const matchup = mapToMatchupGames([game])[0];

    expect(ticker.mlbGamePk).toBe("9");
    expect(ticker.espnEventId).toBeNull();
    expect(ticker.league).toBe("mlb");

    expect(live.mlbGamePk).toBe("9");
    expect(live.espnEventId).toBeNull();
    expect(live.league).toBe("mlb");

    expect(matchup.mlbGamePk).toBe("9");
    expect(matchup.espnEventId).toBeNull();
    expect(matchup.league).toBe("mlb");
  });
});
