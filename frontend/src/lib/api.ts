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

export async function fetchWnbaScoreboard(): Promise<WnbaScoreboardResponse> {
  const res = await fetch("/api/wnba/scoreboard/today", {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`Scoreboard request failed: ${res.status}`);
  }
  return res.json();
}
