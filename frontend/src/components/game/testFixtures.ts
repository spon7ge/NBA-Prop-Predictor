import type { GameDetail, GameDetailWinProbability } from "./types";

const winProbabilityFixture: GameDetailWinProbability = {
  summary: "Above the midline favors PHX",
  timeline: [
    {
      id: "wp-1",
      period: 1,
      clock: "8:00",
      awayScore: 2,
      homeScore: 0,
      awayWinPct: 56,
      homeWinPct: 44,
      teamId: "away1",
    },
    {
      id: "wp-2",
      period: 1,
      clock: "4:29",
      awayScore: 10,
      homeScore: 8,
      awayWinPct: 46,
      homeWinPct: 54,
      teamId: "home1",
    },
  ],
  teamStats: [
    {
      key: "field_goal_pct",
      label: "Field goal %",
      awayValue: 41,
      homeValue: 49,
    },
  ],
};

export const detail: GameDetail = {
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
    logoUrl: null,
  },
  home: {
    id: "home1",
    abbrev: "PHX",
    name: "Phoenix Mercury",
    score: 9,
    color: "#E56020",
    logoUrl: null,
  },
  fgMade: 1,
  fgAttempted: 2,
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
    {
      id: "s2",
      teamId: "home1",
      playerName: "B. Player",
      made: false,
      x: 20,
      y: 10,
      period: 1,
      clock: "7:00",
    },
  ],
  // Newest-first, matching the order returned by the API / mapGameDetail.
  plays: [
    {
      id: "pl3",
      teamId: "home1",
      period: 2,
      clock: "9:00",
      text: "B. Player makes three point shot",
      scoring: true,
      awayScore: 2,
      homeScore: 3,
      shooting: true,
    },
    {
      id: "pl2",
      teamId: "away1",
      period: 1,
      clock: "8:00",
      text: "A. Player makes two point shot",
      scoring: true,
      awayScore: 2,
      homeScore: 0,
      shooting: true,
    },
    {
      id: "pl1",
      teamId: "away1",
      period: 1,
      clock: "10:00",
      text: "Tip off won by Golden State",
      scoring: false,
      awayScore: 0,
      homeScore: 0,
      shooting: false,
    },
  ],
  winProbability: null,
  matchupPrediction: null,
  projectedStarters: null,
  seasonLeaders: null,
  injuries: null,
  boxScore: null,
};

export function buildGameDetailFixture(
  overrides: Partial<GameDetail> = {},
): GameDetail {
  return {
    ...detail,
    winProbability: winProbabilityFixture,
    ...overrides,
  };
}

export function buildScheduledDetail(
  overrides: Partial<GameDetail> = {},
): GameDetail {
  return buildGameDetailFixture({
    status: "scheduled",
    statusLabel: "Sun, July 30 at 7:00 PM EDT",
    venue: "Scotiabank Arena",
    away: {
      id: "away1",
      abbrev: "MIN",
      name: "Minnesota Lynx",
      score: null,
      color: "#266092",
      logoUrl: null,
    },
    home: {
      id: "home1",
      abbrev: "TOR",
      name: "Toronto Tempo",
      score: null,
      color: "#CE1141",
      logoUrl: null,
    },
    fgMade: 0,
    fgAttempted: 0,
    latestPlay: null,
    shots: [],
    plays: [],
    winProbability: null,
    matchupPrediction: null,
    projectedStarters: null,
    seasonLeaders: null,
    injuries: null,
    boxScore: null,
    ...overrides,
  });
}
