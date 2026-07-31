export type ApiGameStatus = "scheduled" | "live" | "halftime" | "final";

export type ApiWnbaTeam = {
  abbrev: string;
  name: string;
  score: number | null;
  record?: string | null;
  logo_url: string | null;
};

export type ApiWnbaGame = {
  id: string;
  espn_event_id: string | null;
  league: "wnba";
  status: ApiGameStatus;
  status_label: string;
  away: ApiWnbaTeam;
  home: ApiWnbaTeam;
  start_time_et: string;
  venue?: string | null;
  venue_city?: string | null;
};

export type ApiGameDetailTeam = {
  id: string;
  abbrev: string;
  name: string;
  score: number | null;
  color: string;
  logo_url: string | null;
};

export type ApiGameDetailShot = {
  id: string;
  team_id: string;
  player_name: string;
  made: boolean;
  x: number;
  y: number;
  period: number;
  clock: string;
};

export type ApiGameDetailPlay = {
  id: string;
  team_id: string | null;
  period: number;
  clock: string;
  text: string;
  scoring: boolean;
  away_score: number;
  home_score: number;
  shooting: boolean;
};

export type ApiGameDetailLatestPlay = {
  id: string;
  clock: string;
  period: number;
  text: string;
  team_id: string | null;
};

export type ApiGameDetailWinProbabilityPoint = {
  id: string;
  period: number;
  clock: string;
  away_score: number;
  home_score: number;
  away_win_pct: number;
  home_win_pct: number;
  team_id: string | null;
};

export type ApiGameDetailTeamStat = {
  key: string;
  label: string;
  away_value: number;
  home_value: number;
};

export type ApiGameDetailWinProbability = {
  summary: string | null;
  timeline: ApiGameDetailWinProbabilityPoint[];
  team_stats: ApiGameDetailTeamStat[];
};

export type ApiGameDetailMatchupPrediction = {
  away_win_pct: number;
  home_win_pct: number;
  source_label: string;
};

export type ApiGameDetailStarter = {
  jersey: string | null;
  name: string;
  position: string | null;
  gtd?: boolean;
};

export type ApiGameDetailProjectedStarters = {
  note: string;
  away: ApiGameDetailStarter[];
  home: ApiGameDetailStarter[];
};

export type ApiGameDetailSeasonLeader = {
  stat: "points" | "assists" | "rebounds";
  label: string;
  name: string;
  value: string;
};

export type ApiGameDetailSeasonLeaders = {
  away: ApiGameDetailSeasonLeader[];
  home: ApiGameDetailSeasonLeader[];
};

export type ApiGameDetailInjury = {
  name: string;
  position: string | null;
  status: string;
  detail: string | null;
};

export type ApiGameDetailInjuries = {
  away: ApiGameDetailInjury[];
  home: ApiGameDetailInjury[];
};

export type ApiGameDetailBoxScorePlayer = {
  name: string;
  did_not_play: boolean;
  values: string[];
};

export type ApiGameDetailBoxScore = {
  columns: string[];
  away: ApiGameDetailBoxScorePlayer[];
  home: ApiGameDetailBoxScorePlayer[];
};

export type ApiWnbaGameDetail = {
  espn_event_id: string;
  league: "wnba";
  status: ApiGameStatus;
  status_label: string;
  venue: string | null;
  away: ApiGameDetailTeam;
  home: ApiGameDetailTeam;
  fg_made: number;
  fg_attempted: number;
  latest_play: ApiGameDetailLatestPlay | null;
  shots: ApiGameDetailShot[];
  plays: ApiGameDetailPlay[];
  win_probability: ApiGameDetailWinProbability | null;
  matchup_prediction: ApiGameDetailMatchupPrediction | null;
  projected_starters: ApiGameDetailProjectedStarters | null;
  season_leaders: ApiGameDetailSeasonLeaders | null;
  injuries: ApiGameDetailInjuries | null;
  box_score: ApiGameDetailBoxScore | null;
  fetched_at: string;
};

export type WnbaScoreboardResponse = {
  date: string;
  games: ApiWnbaGame[];
  fetched_at: string;
};

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

export type ApiWnbaLeaderRow = {
  rank: number;
  player_id: string;
  name: string;
  team_abbrev: string;
  gp: number;
  value: string;
};

export type ApiWnbaLeaderCategory = {
  key:
    | "points"
    | "rebounds"
    | "assists"
    | "steals"
    | "blocks"
    | "three_pointers";
  label: string;
  stat: string;
  leaders: ApiWnbaLeaderRow[];
};

export type ApiWnbaLeadersResponse = {
  season: number;
  pace: "per_game";
  categories: ApiWnbaLeaderCategory[];
};

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

export type ApiWnbaStandingsRow = {
  rank: number;
  team_id: string;
  abbrev: string;
  name: string;
  logo_url: string | null;
  wins: number;
  losses: number;
  wl: string;
  pct: string;
  gb: string;
  home: string;
  away: string;
  l10: string;
  diff: string;
  streak: string;
};

export type ApiWnbaStandingsConference = {
  key: "east" | "west";
  label: string;
  teams: ApiWnbaStandingsRow[];
};

export type ApiWnbaStandingsResponse = {
  season: number;
  conferences: ApiWnbaStandingsConference[];
};

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

export type ApiWnbaOddsGame = {
  home_abbrev: string;
  away_abbrev: string;
  spread_team_abbrev: string | null;
  spread_line: number | null;
  total: number | null;
};

export type ApiWnbaOddsResponse = {
  as_of: string;
  sportsbook: string;
  games: ApiWnbaOddsGame[];
  error?: string | null;
};

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
