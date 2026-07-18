import type { FlatParlayRow } from "@/types/slate";

export interface ApiGame {
  game_date: string;
  game_id?: string;
  event_id?: number;
  home_team_abbrev: string;
  away_team_abbrev: string;
  season_year?: string;
  source?: string;
}

export interface ApiPropLine {
  bookmaker: string;
  market_category: string;
  player_id?: number;
  player_name: string;
  player_name_raw?: string;
  normalized_name?: string;
  side: string;
  game_date: string;
  line: number;
  odds?: number;
  prop_source?: string;
  last_update_at?: string;
  player_team_abbrev?: string;
  home_team_abbrev?: string;
  away_team_abbrev?: string;
  game_season_year?: string;
  min_roll5?: number;
  pts_per_min_roll5?: number;
  reb_per_min_roll5?: number;
  ast_per_min_roll5?: number;
  min_roll10?: number;
  pts_per_min_roll10?: number;
  team_min_rank_l10?: number;
  team_usg_rank_l10?: number;
  expected_pace?: number;
  opp_def_rating_roll10?: number;
  team_spread?: number;
  game_total?: number;
}

export interface ApiMLPrediction {
  prop: string;
  game_id: string;
  player_id: number;
  prediction: number;
  predicted_at: string;
  game_date?: string;
  player_name?: string;
  model_path?: string;
}

export interface ApiGameSlate {
  game_date: string;
  games: ApiGame[];
  props: ApiPropLine[];
  predictions: ApiMLPrediction[];
}

export interface ApiPlayerGame {
  game_id: string;
  game_date: string;
  matchup?: string;
  opp_team_abbreviation?: string;
  wl?: string;
  min: number;
  pts: number;
  reb: number;
  ast: number;
  plus_minus?: number;
}

export interface ApiRollingAvg5 {
  min_roll5?: number;
  pts_roll5?: number;
  reb_roll5?: number;
  ast_roll5?: number;
}

export interface ApiRollingAvg10 {
  min_roll10?: number;
  pts_roll10?: number;
  reb_roll10?: number;
  ast_roll10?: number;
}

export interface ApiMLPredictionSummary {
  prop: string;
  game_id: string;
  prediction: number;
  predicted_at: string;
  game_date?: string;
}

export interface ApiPlayerProfile {
  player_id: number;
  player_name: string;
  normalized_name: string;
  team_abbreviation?: string;
  team_name?: string;
  career_game_count?: number;
  recent_games: ApiPlayerGame[];
  rolling_avg_5?: ApiRollingAvg5;
  rolling_avg_10?: ApiRollingAvg10;
  predictions: ApiMLPredictionSummary[];
}

export interface ApiHealth {
  status: string;
  db?: string;
}

export type ApiLeague = "nba" | "wnba";
/** All Players league filter — ``all`` merges both live-props feeds. */
export type ApiLeagueFilter = ApiLeague | "all";

export interface ApiLivePropModel {
  p_over?: number | null;
  p_under?: number | null;
  lean?: string | null;
  min_q10?: number | null;
  min_q50?: number | null;
  min_q90?: number | null;
  stat_q10?: number | null;
  stat_q50?: number | null;
  stat_q90?: number | null;
}

export interface ApiLivePropGameContext {
  opp_def_rating?: number | null;
  opp_def_rating_rank?: number | null;
  opp_pace?: number | null;
  team_def_rating?: number | null;
  team_pace?: number | null;
  game_total?: number | null;
  team_spread?: number | null;
}

export interface ApiLivePropForm {
  over_l5?: number | null;
  over_l10?: number | null;
  over_l15?: number | null;
}

export interface ApiLivePropVsOpp {
  n_games?: number | null;
  avg_stat?: number | null;
  over_rate_at_line?: number | null;
}

export interface ApiLivePropPick {
  platform: string;
  player: string;
  team_abbr?: string | null;
  opponent_abbr?: string | null;
  is_home?: boolean | null;
  market: string;
  line?: number | null;
  game_date?: string | null;
  league?: string | null;
  run_at?: string | null;
  model: ApiLivePropModel;
  game_context: ApiLivePropGameContext;
  form: ApiLivePropForm;
  vs_opp: ApiLivePropVsOpp;
}

export interface ApiLivePropsResponse {
  generated_at: string;
  league: string;
  game_date: string;
  n_picks: number;
  picks: ApiLivePropPick[];
}

/** Nested Top Legs parlays: leg count → book → FlatParlayRow[]. */
export type ApiLiveSlatesNested = Record<string, Record<string, FlatParlayRow[]>>;

export interface ApiLiveSlatesResponse {
  generated_at: string;
  league: string;
  game_date: string;
  run_at?: string | null;
  count: number;
  slates: ApiLiveSlatesNested;
}

