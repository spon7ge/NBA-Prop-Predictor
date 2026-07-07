-- RPM (rebounds per minute) quantile model feature table.
-- Columns mirror src.features.rpm_features.RPM_FEATURES.
-- Grain: one row per (game_id, player_id).

select
    f.*,
    -- model features
    i.reb_per_min_season_avg,
    i.reb_per_min_10_ewm,
    i.oreb_dreb_ratio,
    i.position_enc,
    i.rbc_per_min_10_ewm,
    i.rpm_season_std,
    i.reb_roll10_slope,
    i.reb_per_min_p10_l10,
    i.reb_per_min_p90_l10,
    -- targets / metadata for training
    i.reb_per_min,
    i.min,
    i.matchup
from {{ ref('features') }} as f
inner join {{ ref('int_player_game_features') }} as i
    on f.game_id = i.game_id
   and f.player_id = i.player_id
