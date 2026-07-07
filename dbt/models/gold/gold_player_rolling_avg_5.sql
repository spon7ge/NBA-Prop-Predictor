-- L5 prior-game rolling averages per player-game.
-- "Prior" means the window excludes the current game (1 preceding),
-- so these are valid pre-game features with no data leakage.
-- Grain: one row per (game_id, player_id).

select
    game_id,
    player_id,
    player_name,
    game_date,
    season_year,
    team_id,
    team_abbreviation,
    opp_team_abbreviation,
    -- L5 rolling averages
    min_roll5,
    pts_roll5,
    reb_roll5,
    ast_roll5,
    tov_roll5,
    usg_pct_roll5,
    plus_minus_roll5,
    pts_per_min_roll5,
    reb_per_min_roll5,
    ast_per_min_roll5,
    poss_per_min_roll5,
    sast_per_min_roll5,
    tchs_per_min_roll5,
    -- convenience derived rates from rolling totals
    case
        when min_roll5 > 0 then reb_roll5 / min_roll5 else null
    end as reb_per_min_roll5_derived,
    case
        when min_roll5 > 0 then ast_roll5 / min_roll5 else null
    end as ast_per_min_roll5_derived,
    case
        when min_roll5 > 0 then pts_roll5 / min_roll5 else null
    end as pts_per_min_roll5_derived
from {{ ref('int_player_game_features') }}
