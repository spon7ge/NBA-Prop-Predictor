-- APM (assists per minute) quantile model feature table.
-- Columns mirror src.features.apm_features.APM_FEATURES.
-- Grain: one row per (game_id, player_id).

select
    f.*,
    -- model features
    i.ast_per_min_season_avg,
    i.team_ast_per_min_rank_l10,
    i.opp_team_ast_allowed,
    i.pass_per_min_5_ewm,
    i.position_encoded,
    i.ast_per_min_std_season,
    i.apm_trend,
    -- targets / metadata for training
    i.ast_per_min,
    i.min,
    i.matchup
from {{ ref('features') }} as f
inner join {{ ref('int_player_game_features') }} as i
    on f.game_id = i.game_id
   and f.player_id = i.player_id
