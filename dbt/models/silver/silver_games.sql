-- Standardized game dimension: deduped rows with team tricodes.

with bronze as (
    select * from {{ ref('bronze_games') }}
),

standardized as (
    select
        game_id,
        event_id,
        game_date,
        commence_at,
        {{ standardize_team_abbrev('home_team') }} as home_team_abbrev,
        {{ standardize_team_abbrev('away_team') }} as away_team_abbrev,
        home_team as home_team_raw,
        away_team as away_team_raw,
        season_year,
        source,
        fetched_at
    from bronze
    where game_date is not null
),

deduped as (
    select
        *,
        row_number() over (
            partition by
                coalesce(game_id, 'event-' || event_id::text),
                game_date,
                home_team_abbrev,
                away_team_abbrev
            order by fetched_at desc nulls last, commence_at desc nulls last
        ) as row_num
    from standardized
    where home_team_abbrev is not null
      and away_team_abbrev is not null
      and home_team_abbrev <> away_team_abbrev
)

select
    game_id,
    event_id,
    game_date,
    commence_at,
    home_team_abbrev,
    away_team_abbrev,
    home_team_raw,
    away_team_raw,
    season_year,
    source,
    fetched_at
from deduped
where row_num = 1
