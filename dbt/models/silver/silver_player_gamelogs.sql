-- Enriched player-gamelog fact table for gold feature models.
-- Sources: silver.player_gamelogs (Python-merged base + adv + tracking + team context)
--          joined to silver_players (dbt) to attach normalized_name.
-- Derives per-minute rates and flag columns used throughout gold.

with gamelogs as (
    select * from {{ source('silver', 'player_gamelogs') }}
    where game_date is not null
      and player_id is not null
),

players as (
    select player_id, normalized_name
    from {{ ref('silver_players') }}
)

select
    gl.game_id,
    gl.player_id,
    gl.player_name,
    p.normalized_name,
    gl.season_year,
    gl.season_type,
    gl.game_date,
    gl.matchup,
    gl.team_id,
    gl.team_abbreviation,
    gl.team_name,
    gl.opp_team_id,
    gl.opp_team_abbreviation,
    gl.opp_team_name,
    gl.wl,
    gl.min,
    gl.pts,
    gl.reb,
    gl.oreb,
    gl.dreb,
    gl.ast,
    gl.tov,
    gl.stl,
    gl.blk,
    gl.fgm,
    gl.fga,
    gl.fg3a,
    gl.fta,
    gl.plus_minus,
    -- advanced
    gl.usg_pct,
    gl.ts_pct,
    gl.efg_pct,
    gl.net_rating,
    gl.pie,
    gl.poss,
    gl.ast_pct,
    gl.ast_to,
    gl.oreb_pct,
    gl.dreb_pct,
    gl.reb_pct,
    -- tracking
    gl.spd,
    gl.dist,
    gl.cfga,
    gl.sast,
    gl.pass,
    gl.tchs,
    -- team context
    gl.team_pace,
    gl.team_def_rating,
    gl.team_poss,
    gl.team_ast,
    gl.team_reb,
    gl.team_fg_pct,
    gl.team_fga,
    gl.team_fg3a,
    -- opponent context
    gl.opp_pts,
    gl.opp_fgm,
    gl.opp_fga,
    gl.opp_fg3a,
    gl.opp_fta,
    gl.opp_def_rating,
    gl.opp_pace,
    gl.opp_poss,
    gl.opp_ast,
    gl.opp_tov,
    gl.opp_reb,
    -- metadata
    gl.start_position,
    gl.pos,
    gl.is_playoff,
    gl.team_spread,
    gl.game_total,
    -- derived flags
    case
        when gl.start_position is not null
          and btrim(gl.start_position) <> ''
        then 1 else 0
    end as starting,
    case when gl.matchup like '% vs.%' then 1 else 0 end as is_home,
    -- per-minute rates (null-safe)
    gl.pts  / nullif(gl.min, 0) as pts_per_min,
    gl.reb  / nullif(gl.min, 0) as reb_per_min,
    gl.ast  / nullif(gl.min, 0) as ast_per_min,
    gl.tov  / nullif(gl.min, 0) as tov_per_min,
    gl.fga  / nullif(gl.min, 0) as fga_per_min,
    gl.fg3a / nullif(gl.min, 0) as fg3a_per_min,
    gl.fta  / nullif(gl.min, 0) as fta_per_min,
    gl.cfga / nullif(gl.min, 0) as cfga_per_min,
    gl.poss / nullif(gl.min, 0) as poss_per_min,
    gl.sast / nullif(gl.min, 0) as sast_per_min,
    gl.pass / nullif(gl.min, 0) as pass_per_min,
    gl.tchs / nullif(gl.min, 0) as tchs_per_min,
    gl.oreb / nullif(gl.min, 0) as oreb_per_min,
    gl.dreb / nullif(gl.min, 0) as dreb_per_min,
    gl.orbc / nullif(gl.min, 0) as orbc_per_min,
    gl.drbc / nullif(gl.min, 0) as drbc_per_min,
    gl.rbc  / nullif(gl.min, 0) as rbc_per_min
from gamelogs as gl
left join players as p
    on gl.player_id = p.player_id
