-- PPM (points per minute) quantile model feature table.
-- Columns mirror src.features.ppm_features.PPM_FEATURES.
-- Grain: one row per (game_id, player_id).

select
    f.*,
    -- model features
    i.pts_per_min_season_avg,
    i.pts_per_min_x_opp_pts_allowed,
    i.cfga_per_min_x_opp_fg_pct_allowed,
    i.fga_per_min_x_opp_def_rating,
    i.fg3a_per_min_x_opp_team_fg3a_allowed,
    i.fta_per_min_x_opp_fta_allowed,
    i.ppm_season_std,
    i.team_pts_per_min_rank_l10,
    i.team_usg_rank_l10,
    i.team_min_rank_l10,
    i.ppm_p10_l10,
    i.ppm_p90_l10,
    -- targets / metadata for training
    i.pts_per_min,
    i.min,
    i.matchup
from {{ ref('features') }} as f
inner join {{ ref('int_player_game_features') }} as i
    on f.game_id = i.game_id
   and f.player_id = i.player_id
