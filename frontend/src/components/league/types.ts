import type { GameStatus, HomeLeague } from "@/components/home/types";

export type LeagueSlug = "nba" | "wnba";

export type MatchupTeam = {
  abbrev: string;
  name: string;
  score: number | null;
  record?: string | null;
};

export type MatchupGame = {
  id: string;
  espnEventId?: string | null;
  league: HomeLeague;
  status: GameStatus;
  statusLabel: string;
  venue?: string | null;
  venueCity?: string | null;
  away: MatchupTeam;
  home: MatchupTeam;
};
