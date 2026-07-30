import type { ApiWnbaGame } from "@/lib/api";
import type { MatchupGame } from "@/components/league/types";
import type { GameStatus, LiveGame, TickerGame } from "./types";

export function isInProgressStatus(status: GameStatus): boolean {
  return status === "live" || status === "halftime";
}

export function mapToTickerGames(games: ApiWnbaGame[]): TickerGame[] {
  return games.filter((g) => isInProgressStatus(g.status)).map((g) => ({
    id: g.id,
    espnEventId: g.espn_event_id,
    league: g.league,
    awayAbbrev: g.away.abbrev,
    homeAbbrev: g.home.abbrev,
    statusLabel: g.status_label,
    status: g.status,
    awayScore: g.away.score,
    homeScore: g.home.score,
  }));
}

export function mapToLiveGames(games: ApiWnbaGame[]): LiveGame[] {
  return games.filter((g) => isInProgressStatus(g.status)).map((g) => ({
    id: g.id,
    espnEventId: g.espn_event_id,
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

export function mapToMatchupGames(games: ApiWnbaGame[]): MatchupGame[] {
  return games.map((g) => ({
    id: g.id,
    espnEventId: g.espn_event_id,
    league: g.league,
    statusLabel: g.status_label,
    status: g.status,
    venue: g.venue ?? null,
    venueCity: g.venue_city ?? null,
    away: {
      abbrev: g.away.abbrev,
      name: g.away.name,
      score: g.away.score,
      record: g.away.record ?? null,
    },
    home: {
      abbrev: g.home.abbrev,
      name: g.home.name,
      score: g.home.score,
      record: g.home.record ?? null,
    },
  }));
}

export function shouldPollScoreboard(games: ApiWnbaGame[] | undefined): boolean {
  if (!games || games.length === 0) return false;
  return games.some((g) => g.status !== "final");
}
