export type GameStatus = "scheduled" | "live" | "halftime" | "final";

export type GameDetailTeam = {
  id: string;
  abbrev: string;
  name: string;
  score: number | null;
  color: string;
  logoUrl: string | null;
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

export type GameDetailWinProbabilityPoint = {
  id: string;
  period: number;
  clock: string;
  awayScore: number;
  homeScore: number;
  awayWinPct: number;
  homeWinPct: number;
  teamId: string | null;
};

export type GameDetailTeamStat = {
  key: string;
  label: string;
  awayValue: number;
  homeValue: number;
};

export type GameDetailWinProbability = {
  summary: string | null;
  timeline: GameDetailWinProbabilityPoint[];
  teamStats: GameDetailTeamStat[];
};

export type GameDetailMatchupPrediction = {
  awayWinPct: number;
  homeWinPct: number;
  sourceLabel: string;
};

export type GameDetailStarter = {
  jersey: string | null;
  name: string;
  position: string | null;
};

export type GameDetailProjectedStarters = {
  note: string;
  away: GameDetailStarter[];
  home: GameDetailStarter[];
};

export type GameDetailSeasonLeader = {
  stat: "points" | "assists" | "rebounds";
  label: string;
  name: string;
  value: string;
};

export type GameDetailSeasonLeaders = {
  away: GameDetailSeasonLeader[];
  home: GameDetailSeasonLeader[];
};

export type GameDetailInjury = {
  name: string;
  position: string | null;
  status: string;
  detail: string | null;
};

export type GameDetailInjuries = {
  away: GameDetailInjury[];
  home: GameDetailInjury[];
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
  winProbability: GameDetailWinProbability | null;
  matchupPrediction: GameDetailMatchupPrediction | null;
  projectedStarters: GameDetailProjectedStarters | null;
  seasonLeaders: GameDetailSeasonLeaders | null;
  injuries: GameDetailInjuries | null;
};
