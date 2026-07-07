-- Opponent + schedule context per player-game.
-- Used by the backend and model training to enrich any player-game row
-- with pre-game defensive environment and fatigue signals.
-- Grain: one row per (game_id, player_id).

select
    game_id,
    player_id,
    player_name,
    game_date,
    season_year,
    team_id,
    team_abbreviation,
    opp_team_id,
    opp_team_abbreviation,
    -- home/away + fatigue
    is_home,
    is_b2b,
    days_rest,
    game_number,
    -- team rolling pace / defence (L10)
    team_pace_roll10,
    team_def_rating_roll10,
    team_poss_roll10,
    -- opponent rolling pace / defence (L10 + L5)
    opp_def_rating_roll10,
    opp_pace_roll10,
    opp_poss_roll10,
    opp_def_rating_roll5,
    opp_pace_roll5,
    opp_poss_roll5,
    -- opponent season-to-date defensive allowed averages
    opp_team_pts_allowed,
    opp_team_fta_allowed,
    opp_team_fg3a_allowed,
    opp_team_fg_pct_allowed,
    opp_team_poss_allowed,
    opp_team_ast_allowed,
    opp_team_tov_forced,
    opp_team_reb_allowed,
    -- pace matchup summary
    expected_pace,
    pace_differential,
    -- Vegas lines
    team_spread,
    game_total
from {{ ref('int_player_game_features') }}
