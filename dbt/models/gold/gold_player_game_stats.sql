-- Per-game box score + derived per-minute rates for every player-game.
-- Foundation for ad-hoc analysis, model training, and feature back-testing.
-- Grain: one row per (game_id, player_id).

select
    -- identifiers
    game_id,
    player_id,
    player_name,
    normalized_name,
    season_year,
    season_type,
    game_date,
    matchup,
    -- team / opponent
    team_id,
    team_abbreviation,
    team_name,
    opp_team_id,
    opp_team_abbreviation,
    opp_team_name,
    -- game flags
    wl,
    starting,
    is_home,
    is_playoff,
    pos,
    -- counting stats
    min,
    pts,
    reb,
    oreb,
    dreb,
    ast,
    tov,
    stl,
    blk,
    fgm,
    fga,
    fg3a,
    fta,
    plus_minus,
    -- advanced efficiency
    usg_pct,
    ts_pct,
    efg_pct,
    net_rating,
    pie,
    poss,
    ast_pct,
    ast_to,
    -- tracking
    spd,
    dist,
    cfga,
    sast,
    pass,
    tchs,
    -- team / game context
    team_pace,
    team_def_rating,
    team_spread,
    game_total,
    -- per-minute rates
    pts_per_min,
    reb_per_min,
    ast_per_min,
    tov_per_min,
    fga_per_min,
    fg3a_per_min,
    fta_per_min,
    cfga_per_min,
    poss_per_min,
    sast_per_min,
    pass_per_min,
    tchs_per_min
from {{ ref('int_player_game_features') }}
