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

/** Decorative graphic key rendered on the right side of a story card. */
export type StoryGraphic =
  | "bracket"
  | "crown"
  | "arc"
  | "trade"
  | "diamond"
  | "checklist";

export type Story = {
  id: string;
  league: HomeLeague;
  headline: string;
  dateLabel: string;
  summary: string;
  graphic: StoryGraphic;
  /** Optional countdown badge, e.g. "3 DAYS LEFT". */
  daysLeft?: number;
};

/** Decorative graphic key on the right side of an Explore card. */
export type ExploreGraphic =
  | "chart"
  | "bars"
  | "dots"
  | "standings"
  | "pulse"
  | "radar";

export type ExploreItem = {
  id: string;
  league: HomeLeague;
  headline: string;
  summary: string;
  graphic: ExploreGraphic;
  /** When true, card spans full width above the grid. */
  featured?: boolean;
};

/** A sport primer card inside the Learn the Game panel. */
export type LearnSport = {
  id: string;
  league: HomeLeague;
  sport: string;
  href?: string;
};
