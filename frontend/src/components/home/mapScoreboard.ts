import type { MatchupGame } from "@/components/league/types";
import type { GameStatus, HomeLeague, LiveGame, TickerGame } from "./types";

type ScoreboardGame = {
  id: string;
  league: HomeLeague;
  status: GameStatus;
  status_label: string;
  espn_event_id?: string | null;
  mlb_game_pk?: string | null;
  away: {
    abbrev: string;
    name: string;
    score: number | null;
    logo_url: string | null;
    record?: string | null;
  };
  home: {
    abbrev: string;
    name: string;
    score: number | null;
    logo_url: string | null;
    record?: string | null;
  };
  venue?: string | null;
  venue_city?: string | null;
};

export function isInProgressStatus(status: GameStatus): boolean {
  return status === "live" || status === "halftime";
}

export function mapToTickerGames(games: ScoreboardGame[]): TickerGame[] {
  return games.map((g) => ({
    id: g.id,
    espnEventId: g.espn_event_id ?? null,
    mlbGamePk: g.mlb_game_pk ?? null,
    league: g.league,
    awayAbbrev: g.away.abbrev,
    homeAbbrev: g.home.abbrev,
    statusLabel: g.status_label,
    status: g.status,
    awayScore: g.away.score,
    homeScore: g.home.score,
  }));
}

export function mapToLiveGames(games: ScoreboardGame[]): LiveGame[] {
  return games.map((g) => ({
    id: g.id,
    espnEventId: g.espn_event_id ?? null,
    mlbGamePk: g.mlb_game_pk ?? null,
    league: g.league,
    statusLabel: g.status_label,
    status: g.status,
    away: {
      abbrev: g.away.abbrev,
      name: g.away.name,
      score: g.away.score,
      logoUrl: g.away.logo_url,
    },
    home: {
      abbrev: g.home.abbrev,
      name: g.home.name,
      score: g.home.score,
      logoUrl: g.home.logo_url,
    },
  }));
}

export function mapToMatchupGames(games: ScoreboardGame[]): MatchupGame[] {
  return games.map((g) => ({
    id: g.id,
    espnEventId: g.espn_event_id ?? null,
    mlbGamePk: g.mlb_game_pk ?? null,
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
      logoUrl: g.away.logo_url,
    },
    home: {
      abbrev: g.home.abbrev,
      name: g.home.name,
      score: g.home.score,
      record: g.home.record ?? null,
      logoUrl: g.home.logo_url,
    },
  }));
}

export function shouldPollScoreboard(games: ScoreboardGame[] | undefined): boolean {
  if (!games || games.length === 0) return false;
  return games.some((g) => g.status !== "final");
}
