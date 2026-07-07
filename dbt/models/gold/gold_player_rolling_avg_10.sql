-- L10 prior-game rolling averages + MIN/PPM model inputs.
-- Covers min_features.MIN_FEATURES and ppm_features.PPM_FEATURES columns
-- used in gold.py model training.
-- Grain: one row per (game_id, player_id).

select
    game_id,
    player_id,
    player_name,
    game_date,
    season_year,
    team_id,
    team_abbreviation,
    -- MIN-model features (min_features.MIN_FEATURES)
    min_10_ewm,
    min_season_mean,
    starter_roll10_pct,
    consec_starts,
    min_rate_of_change,
    team_min_rank_l10,
    team_usg_rank_l10,
    min_p10_l10,
    min_p90_l10,
    min_std_l10,
    min_season_std,
    spd_10_ewm,
    top_player_active,
    active_stars_count,
    -- PPM-model features (ppm_features.PPM_FEATURES)
    pts_per_min_season_avg,
    pts_per_min_x_opp_pts_allowed,
    cfga_per_min_x_opp_fg_pct_allowed,
    fga_per_min_x_opp_def_rating,
    fg3a_per_min_x_opp_team_fg3a_allowed,
    fta_per_min_x_opp_fta_allowed,
    ppm_season_std,
    team_pts_per_min_rank_l10,
    ppm_p10_l10,
    ppm_p90_l10,
    -- shared L10 player rolling averages
    min_roll10,
    pts_roll10,
    reb_roll10,
    ast_roll10,
    tov_roll10,
    usg_pct_roll10,
    plus_minus_roll10,
    pts_per_min_roll10,
    reb_per_min_roll10,
    ast_per_min_roll10,
    fga_per_min_roll10,
    fg3a_per_min_roll10,
    fta_per_min_roll10,
    cfga_per_min_roll10,
    spd_roll10
from {{ ref('int_player_game_features') }}
