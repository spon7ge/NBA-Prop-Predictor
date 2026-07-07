-- MIN quantile model feature table.
-- Columns mirror src.features.min_features.MIN_FEATURES.
-- Grain: one row per (game_id, player_id).

select
    f.*,
    -- model features
    i.min_10_ewm,
    i.min_season_mean,
    i.starter_roll10_pct,
    i.consec_starts,
    i.min_rate_of_change,
    i.team_min_rank_l10,
    i.team_usg_rank_l10,
    i.min_p10_l10,
    i.min_p90_l10,
    i.min_std_l10,
    i.min_season_std,
    i.spd_10_ewm,
    i.top_player_active,
    i.active_stars_count,
    -- targets / metadata for training
    i.min,
    i.matchup
from {{ ref('features') }} as f
inner join {{ ref('int_player_game_features') }} as i
    on f.game_id = i.game_id
   and f.player_id = i.player_id
