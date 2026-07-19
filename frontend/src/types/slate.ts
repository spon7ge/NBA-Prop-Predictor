export type Book = "prizepicks" | "underdog" | "draftkings" | "betr";
export type LegCount = 2 | 3 | 5 | 6;
export type View = "pairs" | "players" | "results";
export type Tier = "sharp_verified" | "conflict" | "no_model" | string;
export type PlayersGroupSort = "edge_desc" | "props_desc" | "verified_desc" | "name_asc";

export interface FlatParlayRow {
  PARLAY_PROB?: number;
  EV?: number;
  KELLY?: number;
  KELLY_QUARTER?: number;
  N_LEGS?: number;
  LEGS?: ParlayLegRaw[];
  [key: string]: unknown;
}

export interface ParlayLegRaw {
  player?: string;
  display_name?: string;
  team_abbr?: string;
  team?: string;
  market?: string;
  dfs_line?: number;
  side?: string;
  opponent_abbr?: string;
  opponent?: string;
  avg_stat_l10?: number;
  game_context?: {
    spread?: number;
    game_total?: number;
    opp_def_rating_rank?: number;
    opp_pace_rank?: number;
  };
  vs_opp?: {
    avg_stat?: number;
    n_games?: number;
  };
  form?: {
    over_l10?: number;
  };
  model?: {
    lean?: string;
    stat_q50?: number;
    p_over?: number;
  };
}

export interface Leg {
  name: string;
  team: string;
  market: string;
  line: number;
  side: string;
  prediction: number;
  opponent: string;
  spread?: number;
  total?: number;
  defRank?: number;
  avgStatL10?: number;
  oppPaceRank?: number;
  avgVsMatchup?: number;
  matchupGames?: number;
  overRateL10?: number;
}

export interface MappedParlay {
  parlayProb: number;
  ev: number;
  kelly?: number;
  legs: Leg[];
}

export interface EnrichedPick {
  player_id?: number;
  player?: string;
  display_name?: string;
  /** Set when loaded from live-props (``nba`` / ``wnba``). */
  league?: "nba" | "wnba";
  team_abbr?: string;
  opponent_abbr?: string;
  is_home?: boolean;
  market?: string;
  dfs_line?: number;
  platform?: string;
  tier?: Tier;
  model?: {
    lean?: string;
    stat_q50?: number;
    min_q50?: number;
    p_over?: number;
  };
  sharp?: {
    lean?: string;
    no_vig_over?: number;
  };
  consensus?: {
    mean_no_vig_over_same_line?: number;
    n_books_same_line?: number;
  };
  game_context?: {
    spread?: number;
    game_total?: number;
    opp_def_rating_rank?: number;
    opp_pace_rank?: number;
    /** Raw pace from live-props (not a rank). */
    opp_pace?: number;
  };
  form?: {
    over_l5?: number;
    over_l10?: number;
    over_l15?: number;
  };
  vs_opp?: {
    avg_stat?: number;
    n_games?: number;
  };
}

export interface PlayerGroup {
  player: string;
  playerId?: number;
  displayName: string;
  team: string;
  opp: string | null;
  is_home?: boolean;
  picks: EnrichedPick[];
  tiers: {
    sharp_verified: number;
    conflict: number;
    no_model: number;
  };
  markets: Record<string, boolean>;
  platforms: Record<string, boolean>;
  bestEdge: number | null;
  bestEdgeSide: string | null;
}

export interface PickEdge {
  side: string;
  prob: number;
  edge: number;
}

export type SlatesByBook = Record<Book, FlatParlayRow[]>;
export type SlatesState = Record<LegCount, SlatesByBook>;
