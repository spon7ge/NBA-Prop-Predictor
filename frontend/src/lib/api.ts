import type { components } from "./api.schema";

type Schemas = components["schemas"];

export type ApiGameStatus = Schemas["WnbaGame"]["status"];
export type ApiWnbaTeam = Schemas["WnbaTeam"];
export type ApiWnbaGame = Schemas["WnbaGame"];
export type ApiGameDetailTeam = Schemas["GameDetailTeam"];
export type ApiGameDetailShot = Schemas["GameDetailShot"];
export type ApiGameDetailPlay = Schemas["GameDetailPlay"];
export type ApiGameDetailLatestPlay = Schemas["GameDetailLatestPlay"];
export type ApiGameDetailWinProbabilityPoint =
  Schemas["GameDetailWinProbabilityPoint"];
export type ApiGameDetailTeamStat = Schemas["GameDetailTeamStat"];
export type ApiGameDetailWinProbability = Schemas["GameDetailWinProbability"];
export type ApiGameDetailMatchupPrediction =
  Schemas["GameDetailMatchupPrediction"];
export type ApiGameDetailStarter = Schemas["GameDetailStarter"];
export type ApiGameDetailProjectedStarters =
  Schemas["GameDetailProjectedStarters"];
export type ApiGameDetailSeasonLeader = Schemas["GameDetailSeasonLeader"];
export type ApiGameDetailSeasonLeaders = Schemas["GameDetailSeasonLeaders"];
export type ApiGameDetailInjury = Schemas["GameDetailInjury"];
export type ApiGameDetailInjuries = Schemas["GameDetailInjuries"];
export type ApiGameDetailBoxScorePlayer = Schemas["GameDetailBoxScorePlayer"];
export type ApiGameDetailBoxScore = Schemas["GameDetailBoxScore"];
export type ApiWnbaGameDetail = Schemas["WnbaGameDetail"];
export type WnbaScoreboardResponse = Schemas["WnbaScoreboardResponse"];

export type ApiWnbaLeaderRow = Schemas["WnbaLeaderRow"];
export type ApiWnbaLeaderCategory = Schemas["WnbaLeaderCategory"];
export type ApiWnbaLeadersResponse = Schemas["WnbaLeadersResponse"];

export type ApiWnbaStandingsRow = Schemas["WnbaStandingsRow"];
export type ApiWnbaStandingsConference = Schemas["WnbaStandingsConference"];
export type ApiWnbaStandingsResponse = Schemas["WnbaStandingsResponse"];

export type ApiWnbaOddsGame = Schemas["WnbaOddsGame"];
export type ApiWnbaOddsResponse = Schemas["WnbaOddsResponse"];

export type ApiWnbaPropBookQuote = Schemas["WnbaPropBookQuote"];
export type ApiWnbaPropLine = Schemas["WnbaPropLine"];
export type ApiWnbaPropsResponse = Schemas["WnbaPropsResponse"];

export type ApiWnbaFuturesEntry = Schemas["WnbaFuturesEntry"];
export type ApiWnbaFuturesMarket = Schemas["WnbaFuturesMarket"];
export type ApiWnbaFuturesResponse = Schemas["WnbaFuturesResponse"];

export type ApiWnbaPlayerAverages = Schemas["WnbaPlayerAverages"];
export type ApiWnbaPlayerGame = Schemas["WnbaPlayerGame"];
export type ApiWnbaPlayerResponse = Schemas["WnbaPlayerResponse"];

export type ApiMlbTeam = Schemas["MlbTeam"];
export type ApiMlbGame = Schemas["MlbGame"];
export type MlbScoreboardResponse = Schemas["MlbScoreboardResponse"];
export type ApiMlbOddsGame = Schemas["MlbOddsGame"];
export type ApiMlbOddsResponse = Schemas["MlbOddsResponse"];

/** Shared shape for matchup odds merge (WNBA + MLB). */
export type ApiMatchupOddsGame = ApiWnbaOddsGame | ApiMlbOddsGame;

/**
 * Origin of the HoopVista API, without a trailing slash.
 *
 * Empty in local dev, where Vite's `/api` proxy forwards to the backend. Static
 * hosts (GitHub Pages and friends) have no proxy, so their builds must set
 * `VITE_API_BASE_URL` to the live API origin or every request 404s.
 */
const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

export async function fetchWnbaScoreboard(): Promise<WnbaScoreboardResponse> {
  const res = await fetch(`${API_BASE}/api/wnba/scoreboard/today`, {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`Scoreboard request failed: ${res.status}`);
  }
  return res.json();
}

export async function fetchWnbaScoreboardByDate(
  dateEt: string,
): Promise<WnbaScoreboardResponse> {
  const res = await fetch(
    `${API_BASE}/api/wnba/scoreboard?date=${encodeURIComponent(dateEt)}`,
    { headers: { Accept: "application/json" }, cache: "no-store" },
  );
  if (!res.ok) {
    throw new Error(`Scoreboard request failed: ${res.status}`);
  }
  return res.json();
}

export async function fetchGameDetail(
  espnEventId: string,
): Promise<ApiWnbaGameDetail> {
  const res = await fetch(`${API_BASE}/api/wnba/games/${espnEventId}`, {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`Game detail request failed: ${res.status}`);
  }
  return res.json();
}

export async function fetchWnbaLeaders(): Promise<ApiWnbaLeadersResponse> {
  const res = await fetch(`${API_BASE}/api/wnba/leaders`, {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`Leaders request failed: ${res.status}`);
  }
  return res.json();
}

export async function fetchWnbaStandings(): Promise<ApiWnbaStandingsResponse> {
  const res = await fetch(`${API_BASE}/api/wnba/standings`, {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`Standings request failed: ${res.status}`);
  }
  return res.json();
}

export async function fetchWnbaOdds(): Promise<ApiWnbaOddsResponse> {
  const res = await fetch(`${API_BASE}/api/wnba/odds/today`, {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`Odds request failed: ${res.status}`);
  }
  return res.json();
}

export async function fetchWnbaProps(): Promise<ApiWnbaPropsResponse> {
  const res = await fetch(`${API_BASE}/api/wnba/props/today`, {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`Props request failed: ${res.status}`);
  }
  return res.json();
}

export async function fetchWnbaFutures(): Promise<ApiWnbaFuturesResponse> {
  const res = await fetch(`${API_BASE}/api/wnba/futures`, {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`Futures request failed: ${res.status}`);
  }
  return res.json();
}

export async function fetchWnbaPlayer(
  playerId: string,
): Promise<ApiWnbaPlayerResponse> {
  const res = await fetch(`${API_BASE}/api/wnba/player/${playerId}`, {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`Player request failed: ${res.status}`);
  }
  return res.json();
}

export async function fetchMlbScoreboard(): Promise<MlbScoreboardResponse> {
  const res = await fetch(`${API_BASE}/api/mlb/scoreboard/today`, {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`MLB scoreboard request failed: ${res.status}`);
  }
  return res.json();
}

export async function fetchMlbScoreboardByDate(
  dateEt: string,
): Promise<MlbScoreboardResponse> {
  const res = await fetch(
    `${API_BASE}/api/mlb/scoreboard?date=${encodeURIComponent(dateEt)}`,
    { headers: { Accept: "application/json" }, cache: "no-store" },
  );
  if (!res.ok) {
    throw new Error(`MLB scoreboard request failed: ${res.status}`);
  }
  return res.json();
}

export async function fetchMlbOdds(): Promise<ApiMlbOddsResponse> {
  const res = await fetch(`${API_BASE}/api/mlb/odds/today`, {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`MLB odds request failed: ${res.status}`);
  }
  return res.json();
}
