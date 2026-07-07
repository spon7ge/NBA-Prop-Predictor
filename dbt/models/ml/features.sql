-- Shared ML input layer: rolling form, schedule, usage, and opponent defence.
-- One row per (game_id, player_id). Prop-specific models extend this base.
-- Grain: one row per player-game.

select
    game_id,
    player_id,
    player_name,
    normalized_name,
    season_year,
    season_type,
    game_date,
    team_id,
    team_abbreviation,
    opp_team_id,
    opp_team_abbreviation,
    starting,
    pos,
    -- schedule / home-away
    is_home,
    days_rest,
    is_b2b,
    game_number,
    -- L5 rolling averages (prior games only)
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
    pass_per_min_roll5,
    -- L10 rolling averages (prior games only)
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
    pass_per_min_roll10,
    -- opponent defensive environment
    opp_def_rating_roll10,
    opp_def_rating_roll5,
    opp_pace_roll10,
    opp_team_pts_allowed,
    opp_team_ast_allowed,
    opp_team_reb_allowed,
    -- pace matchup
    expected_pace,
    pace_differential,
    team_spread,
    game_total
from {{ ref('int_player_game_features') }}
