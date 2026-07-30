export type GameStatus = "scheduled" | "live" | "halftime" | "final";

export type GameDetailTeam = {
  id: string;
  abbrev: string;
  name: string;
  score: number | null;
  color: string;
};

export type GameDetailShot = {
  id: string;
  teamId: string;
  playerName: string;
  made: boolean;
  x: number;
  y: number;
  period: number;
  clock: string;
};

export type GameDetailPlay = {
  id: string;
  teamId: string | null;
  period: number;
  clock: string;
  text: string;
  scoring: boolean;
  awayScore: number;
  homeScore: number;
  shooting: boolean;
};

export type GameDetailLatestPlay = {
  id: string;
  clock: string;
  period: number;
  text: string;
  teamId: string | null;
};

export type GameDetail = {
  espnEventId: string;
  league: "wnba";
  status: GameStatus;
  statusLabel: string;
  venue: string | null;
  away: GameDetailTeam;
  home: GameDetailTeam;
  fgMade: number;
  fgAttempted: number;
  latestPlay: GameDetailLatestPlay | null;
  shots: GameDetailShot[];
  plays: GameDetailPlay[];
};
