-- Canonical player dimension from NBA gamelogs.
-- One row per player_id; normalized_name powers prop matching.

with appearances as (
    select
        player_id,
        player_name,
        team_id,
        team_abbreviation,
        team_name,
        count(*) as game_count
    from {{ source('raw', 'player_base') }}
    where player_id is not null
      and player_name is not null
      and btrim(player_name) <> ''
    group by 1, 2, 3, 4, 5
),

canonical as (
    select
        player_id,
        player_name,
        team_id,
        team_abbreviation,
        team_name,
        game_count,
        row_number() over (
            partition by player_id
            order by game_count desc, length(player_name) desc, player_name asc
        ) as name_rank
    from appearances
),

aliases as (
    select
        player_id,
        array_agg(distinct player_name order by player_name) as name_aliases
    from appearances
    group by 1
),

game_totals as (
    select
        player_id,
        sum(game_count) as career_game_count
    from appearances
    group by 1
),

latest_team as (
    select distinct on (player_id)
        player_id,
        team_id,
        team_abbreviation,
        team_name
    from {{ source('raw', 'player_base') }}
    where player_id is not null
    order by player_id, game_date desc nulls last, fetched_at desc nulls last
)

select
    c.player_id,
    c.player_name,
    {{ normalize_player_name('c.player_name') }} as normalized_name,
    lt.team_id,
    {{ standardize_team_abbrev('lt.team_abbreviation') }} as team_abbreviation,
    lt.team_name,
    a.name_aliases,
    gt.career_game_count
from canonical as c
inner join aliases as a
    on c.player_id = a.player_id
inner join game_totals as gt
    on c.player_id = gt.player_id
left join latest_team as lt
    on c.player_id = lt.player_id
where c.name_rank = 1
