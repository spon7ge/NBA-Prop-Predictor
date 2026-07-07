-- Prop lines enriched with player identity, game context, and pre-game
-- rolling features. One row per unique prop quote (bookmaker × market ×
-- player × side × game_date × prop_source).
-- Joins silver_props → silver_players → silver_games → gold rolling tables.

with props as (
    select * from {{ ref('silver_props') }}
),

players as (
    select player_id, team_abbreviation
    from {{ ref('silver_players') }}
),

games as (
    select game_date, home_team_abbrev, away_team_abbrev, season_year
    from {{ ref('silver_games') }}
),

roll5 as (
    select
        player_id,
        game_date,
        min_roll5,
        pts_per_min_roll5,
        reb_per_min_roll5,
        ast_per_min_roll5
    from {{ ref('gold_player_rolling_avg_5') }}
),

roll10 as (
    select
        player_id,
        game_date,
        min_roll10,
        pts_per_min_roll10,
        team_min_rank_l10,
        team_usg_rank_l10
    from {{ ref('gold_player_rolling_avg_10') }}
),

matchup as (
    select
        player_id,
        game_date,
        expected_pace,
        opp_def_rating_roll10,
        team_spread,
        game_total
    from {{ ref('gold_matchup_features') }}
)

select
    -- prop identity
    p.bookmaker,
    p.market_category,
    p.player_id,
    p.player_name,
    p.player_name_raw,
    p.normalized_name,
    p.side,
    p.game_date,
    p.line,
    p.odds,
    p.last_update_at,
    p.data_pulled_at,
    p.fetched_at,
    p.prop_source,
    -- player + game context
    pl.team_abbreviation                                as player_team_abbrev,
    g.home_team_abbrev,
    g.away_team_abbrev,
    g.season_year                                       as game_season_year,
    -- L5 pre-game rolling context
    r5.min_roll5,
    r5.pts_per_min_roll5,
    r5.reb_per_min_roll5,
    r5.ast_per_min_roll5,
    -- L10 pre-game rolling context
    r10.min_roll10,
    r10.pts_per_min_roll10,
    r10.team_min_rank_l10,
    r10.team_usg_rank_l10,
    -- matchup / pace context
    m.expected_pace,
    m.opp_def_rating_roll10,
    m.team_spread,
    m.game_total
from props as p
left join players as pl
    on p.player_id = pl.player_id
left join games as g
    on p.game_date = g.game_date
   and pl.team_abbreviation in (g.home_team_abbrev, g.away_team_abbrev)
left join roll5 as r5
    on p.player_id = r5.player_id
   and p.game_date = r5.game_date
left join roll10 as r10
    on p.player_id = r10.player_id
   and p.game_date = r10.game_date
left join matchup as m
    on p.player_id = m.player_id
   and p.game_date = m.game_date
