import { describe, expect, it } from "vitest";
import type { ApiWnbaGameDetail } from "@/lib/api";
import { mapGameDetail } from "./mapGameDetail";

function buildApiDetail(
  overrides: Partial<ApiWnbaGameDetail> = {},
): ApiWnbaGameDetail {
  return {
    espn_event_id: "401749001",
    league: "wnba",
    status: "live",
    status_label: "4:13 - 1st",
    venue: "Mortgage Matchup Center",
    away: {
      id: "away1",
      abbrev: "GS",
      name: "Golden State Valkyries",
      score: 10,
      color: "#5B2C6F",
      logo_url: "https://a.espncdn.com/i/teamlogos/wnba/500-dark/gs.png",
    },
    home: {
      id: "home1",
      abbrev: "PHX",
      name: "Phoenix Mercury",
      score: 9,
      color: "#E56020",
      logo_url: "https://a.espncdn.com/i/teamlogos/wnba/500-dark/phx.png",
    },
    fg_made: 6,
    fg_attempted: 16,
    latest_play: {
      id: "p1",
      clock: "4:29",
      period: 1,
      text: "Laeticia Amihere makes two point shot",
      team_id: "away1",
    },
    shots: [
      {
        id: "s1",
        team_id: "away1",
        player_name: "A. Player",
        made: true,
        x: 25,
        y: 5,
        period: 1,
        clock: "8:00",
      },
    ],
    plays: [
      {
        id: "pl1",
        team_id: "away1",
        period: 1,
        clock: "8:00",
        text: "A. Player makes two point shot",
        scoring: true,
        away_score: 2,
        home_score: 0,
        shooting: true,
      },
    ],
    win_probability: null,
    matchup_prediction: null,
    projected_starters: null,
    season_leaders: null,
    injuries: null,
    box_score: null,
    fetched_at: "2026-07-29T00:00:00Z",
    ...overrides,
  };
}

describe("mapGameDetail", () => {
  it("maps snake_case API fields to camelCase UI fields", () => {
    expect(mapGameDetail(buildApiDetail())).toEqual({
      espnEventId: "401749001",
      league: "wnba",
      status: "live",
      statusLabel: "4:13 - 1st",
      venue: "Mortgage Matchup Center",
      away: {
        id: "away1",
        abbrev: "GS",
        name: "Golden State Valkyries",
        score: 10,
        color: "#5B2C6F",
        logoUrl: "https://a.espncdn.com/i/teamlogos/wnba/500-dark/gs.png",
      },
      home: {
        id: "home1",
        abbrev: "PHX",
        name: "Phoenix Mercury",
        score: 9,
        color: "#E56020",
        logoUrl: "https://a.espncdn.com/i/teamlogos/wnba/500-dark/phx.png",
      },
      fgMade: 6,
      fgAttempted: 16,
      latestPlay: {
        id: "p1",
        clock: "4:29",
        period: 1,
        text: "Laeticia Amihere makes two point shot",
        teamId: "away1",
      },
      shots: [
        {
          id: "s1",
          teamId: "away1",
          playerName: "A. Player",
          made: true,
          x: 25,
          y: 5,
          period: 1,
          clock: "8:00",
        },
      ],
      plays: [
        {
          id: "pl1",
          teamId: "away1",
          period: 1,
          clock: "8:00",
          text: "A. Player makes two point shot",
          scoring: true,
          awayScore: 2,
          homeScore: 0,
          shooting: true,
        },
      ],
      winProbability: null,
      matchupPrediction: null,
      projectedStarters: null,
      seasonLeaders: null,
      injuries: null,
      boxScore: null,
    });
  });

  it("maps a null latest_play to null", () => {
    expect(
      mapGameDetail(buildApiDetail({ latest_play: null })).latestPlay,
    ).toBeNull();
  });

  it("maps a null venue through unchanged", () => {
    expect(mapGameDetail(buildApiDetail({ venue: null })).venue).toBeNull();
  });

  it("maps win probability into camelCase detail data", () => {
    const mapped = mapGameDetail(
      buildApiDetail({
        win_probability: {
          summary: "Above the midline favors PHX",
          timeline: [
            {
              id: "p-1",
              period: 1,
              clock: "4:29",
              away_score: 10,
              home_score: 8,
              away_win_pct: 46,
              home_win_pct: 54,
              team_id: "129153",
            },
          ],
          team_stats: [
            {
              key: "field_goal_pct",
              label: "Field goal %",
              away_value: 41,
              home_value: 49,
            },
          ],
        },
      }),
    );

    expect(mapped.winProbability?.summary).toBe(
      "Above the midline favors PHX",
    );
    expect(mapped.winProbability?.timeline[0]).toEqual({
      id: "p-1",
      period: 1,
      clock: "4:29",
      awayScore: 10,
      homeScore: 8,
      awayWinPct: 46,
      homeWinPct: 54,
      teamId: "129153",
    });
    expect(mapped.winProbability?.teamStats[0]).toEqual({
      key: "field_goal_pct",
      label: "Field goal %",
      awayValue: 41,
      homeValue: 49,
    });
  });

  it("maps matchup preview fields", () => {
    const mapped = mapGameDetail({
      ...buildApiDetail(),
      matchup_prediction: {
        away_win_pct: 67,
        home_win_pct: 33,
        source_label: "ESPN game projection",
      },
      projected_starters: {
        note: "from each team's last game",
        away: [{ jersey: "1", name: "Natasha Howard", position: "F" }],
        home: [{ jersey: "10", name: "Maria Conde", position: "F" }],
      },
      season_leaders: {
        away: [
          {
            stat: "points",
            label: "Points",
            name: "Olivia Miles",
            value: "19.5",
          },
        ],
        home: [],
      },
      injuries: {
        away: [],
        home: [
          {
            name: "Nyara Sabally",
            position: "F",
            status: "Out",
            detail: "Ribs",
          },
        ],
      },
    });
    expect(mapped.matchupPrediction?.awayWinPct).toBe(67);
    expect(mapped.projectedStarters?.away[0].name).toBe("Natasha Howard");
    expect(mapped.seasonLeaders?.away[0].stat).toBe("points");
    expect(mapped.injuries?.home[0].detail).toBe("Ribs");
  });

  it("maps null preview fields", () => {
    const mapped = mapGameDetail(buildApiDetail());
    expect(mapped.matchupPrediction).toBeNull();
    expect(mapped.projectedStarters).toBeNull();
    expect(mapped.seasonLeaders).toBeNull();
    expect(mapped.injuries).toBeNull();
  });

  it("maps null logo_url to null logoUrl", () => {
    const mapped = mapGameDetail(
      buildApiDetail({
        away: {
          id: "away1",
          abbrev: "GS",
          name: "Golden State Valkyries",
          score: 10,
          color: "#5B2C6F",
          logo_url: null,
        },
      }),
    );
    expect(mapped.away.logoUrl).toBeNull();
  });
});
