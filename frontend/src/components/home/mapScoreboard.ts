import type { ApiWnbaGame } from "@/lib/api";
import type { LiveGame, TickerGame } from "./types";

export function mapToTickerGames(games: ApiWnbaGame[]): TickerGame[] {
  return games.map((g) => ({
    id: g.id,
    league: g.league,
    awayAbbrev: g.away.abbrev,
    homeAbbrev: g.home.abbrev,
    statusLabel: g.status_label,
    status: g.status,
  }));
}

export function mapToLiveGames(games: ApiWnbaGame[]): LiveGame[] {
  return games.map((g) => ({
    id: g.id,
    league: g.league,
    statusLabel: g.status_label,
    status: g.status,
    away: {
      abbrev: g.away.abbrev,
      name: g.away.name,
      score: g.away.score,
    },
    home: {
      abbrev: g.home.abbrev,
      name: g.home.name,
      score: g.home.score,
    },
  }));
}

export function shouldPollScoreboard(games: ApiWnbaGame[] | undefined): boolean {
  if (!games || games.length === 0) return false;
  return games.some((g) => g.status !== "final");
}
