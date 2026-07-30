export type ApiGameStatus = "scheduled" | "live" | "halftime" | "final";

export type ApiWnbaTeam = {
  abbrev: string;
  name: string;
  score: number | null;
};

export type ApiWnbaGame = {
  id: string;
  league: "wnba";
  status: ApiGameStatus;
  status_label: string;
  away: ApiWnbaTeam;
  home: ApiWnbaTeam;
  start_time_et: string;
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
