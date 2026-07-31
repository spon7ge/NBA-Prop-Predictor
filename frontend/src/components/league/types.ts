import type { GameStatus, HomeLeague } from "@/components/home/types";

export type LeagueSlug = HomeLeague;

export type MatchupTeam = {
  abbrev: string;
  name: string;
  score: number | null;
  record?: string | null;
  logoUrl: string | null;
};

export type MatchupOdds = {
  spreadTeamAbbrev: string | null;
  spreadLine: number | null;
  total: number | null;
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
  odds?: MatchupOdds | null;
};
