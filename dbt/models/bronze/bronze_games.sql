-- Bronze games: light cleanup on two raw sources.
--   1. raw.team_base  — historical NBA games (deduped to one row per game_id)
--   2. raw.games      — Odds-API events (spreads/totals scrapers)

with gamelog_games as (
    select
        game_id,
        max(game_date) as game_date_raw,
        max(season_year) as season_year,
        max(matchup) as matchup,
        max(fetched_at) as fetched_at
    from {{ source('raw', 'team_base') }}
    where game_id is not null
    group by 1
),

gamelog_parsed as (
    select
        game_id,
        null::bigint as event_id,
        case
            when game_date_raw ~ '^\d{4}-\d{2}-\d{2}$'
                then game_date_raw::date
            when game_date_raw ~ '^\d{4}-\d{2}-\d{2}T'
                then left(game_date_raw, 10)::date
            else null::date
        end as game_date,
        null::timestamptz as commence_at,
        case
            when matchup like '% vs. %'
                then trim(split_part(matchup, ' vs. ', 2))
            when matchup like '% @ %'
                then trim(split_part(matchup, ' @ ', 2))
            else null
        end as home_team,
        case
            when matchup like '% vs. %'
                then trim(split_part(matchup, ' vs. ', 1))
            when matchup like '% @ %'
                then trim(split_part(matchup, ' @ ', 1))
            else null
        end as away_team,
        season_year,
        'gamelogs'::text as source,
        fetched_at
    from gamelog_games
),

odds_games as (
    select
        null::text as game_id,
        event_id,
        case
            when commence_time ~ '^\d{4}-\d{2}-\d{2}$'
                then commence_time::date
            when commence_time ~ '^\d{4}-\d{2}-\d{2}T'
                then left(commence_time, 10)::date
            else null::date
        end as game_date,
        case
            when commence_time ~ '^\d{4}-\d{2}-\d{2}T'
                then commence_time::timestamptz
            when commence_time ~ '^\d{4}-\d{2}-\d{2}$'
                then (commence_time || 'T00:00:00Z')::timestamptz
            else null::timestamptz
        end as commence_at,
        home_team,
        away_team,
        null::text as season_year,
        'odds_api'::text as source,
        fetched_at
    from {{ source('raw', 'games') }}
)

select * from gamelog_parsed
union all
select * from odds_games
