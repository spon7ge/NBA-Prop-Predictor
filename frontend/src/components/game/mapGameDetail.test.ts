import { describe, expect, it } from "vitest";
import type { ApiWnbaGameDetail } from "@/lib/api";
import { mapGameDetail } from "./mapGameDetail";

function apiDetail(
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
    },
    home: {
      id: "home1",
      abbrev: "PHX",
      name: "Phoenix Mercury",
      score: 9,
      color: "#E56020",
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
    fetched_at: "2026-07-29T00:00:00Z",
    ...overrides,
  };
}

describe("mapGameDetail", () => {
  it("maps snake_case API fields to camelCase UI fields", () => {
    expect(mapGameDetail(apiDetail())).toEqual({
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
      },
      home: {
        id: "home1",
        abbrev: "PHX",
        name: "Phoenix Mercury",
        score: 9,
        color: "#E56020",
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
    });
  });

  it("maps a null latest_play to null", () => {
    expect(mapGameDetail(apiDetail({ latest_play: null })).latestPlay).toBeNull();
  });

  it("maps a null venue through unchanged", () => {
    expect(mapGameDetail(apiDetail({ venue: null })).venue).toBeNull();
  });
});
