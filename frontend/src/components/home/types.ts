export type HomeLeague = "nba" | "wnba";

export type TickerGame = {
  id: string;
  league: HomeLeague;
  awayAbbrev: string;
  homeAbbrev: string;
  statusLabel: string;
};

export type LiveGameTeam = {
  abbrev: string;
  name: string;
  score: number | null;
};

export type LiveGame = {
  id: string;
  league: HomeLeague;
  statusLabel: string;
  away: LiveGameTeam;
  home: LiveGameTeam;
};
