-- Shared window-feature intermediate for all gold models.
-- Computes L5/L10 rolling averages, season-to-date means, team/opponent
-- rolling defensive ratings, and rank features — all in one pass over
-- silver_player_gamelogs so each gold model pays the scan cost only once.
-- Materialised as a table so downstream gold models are simple SELECTs.

with base as (
    select * from {{ ref('silver_player_gamelogs') }}
),

-- ── Player-level rolling windows ────────────────────────────────────────────
player_windows as (
    select
        b.*,

        -- L5 player rolls
        avg(b.min) over (
            partition by b.player_id
            order by b.game_date
            rows between 5 preceding and 1 preceding
        ) as min_roll5,
        avg(b.pts) over (
            partition by b.player_id
            order by b.game_date
            rows between 5 preceding and 1 preceding
        ) as pts_roll5,
        avg(b.reb) over (
            partition by b.player_id
            order by b.game_date
            rows between 5 preceding and 1 preceding
        ) as reb_roll5,
        avg(b.ast) over (
            partition by b.player_id
            order by b.game_date
            rows between 5 preceding and 1 preceding
        ) as ast_roll5,
        avg(b.tov) over (
            partition by b.player_id
            order by b.game_date
            rows between 5 preceding and 1 preceding
        ) as tov_roll5,
        avg(b.usg_pct) over (
            partition by b.player_id
            order by b.game_date
            rows between 5 preceding and 1 preceding
        ) as usg_pct_roll5,
        avg(b.plus_minus) over (
            partition by b.player_id
            order by b.game_date
            rows between 5 preceding and 1 preceding
        ) as plus_minus_roll5,
        avg(b.pts_per_min) over (
            partition by b.player_id
            order by b.game_date
            rows between 5 preceding and 1 preceding
        ) as pts_per_min_roll5,
        avg(b.reb_per_min) over (
            partition by b.player_id
            order by b.game_date
            rows between 5 preceding and 1 preceding
        ) as reb_per_min_roll5,
        avg(b.ast_per_min) over (
            partition by b.player_id
            order by b.game_date
            rows between 5 preceding and 1 preceding
        ) as ast_per_min_roll5,
        avg(b.poss_per_min) over (
            partition by b.player_id
            order by b.game_date
            rows between 5 preceding and 1 preceding
        ) as poss_per_min_roll5,
        avg(b.sast_per_min) over (
            partition by b.player_id
            order by b.game_date
            rows between 5 preceding and 1 preceding
        ) as sast_per_min_roll5,
        avg(b.tchs_per_min) over (
            partition by b.player_id
            order by b.game_date
            rows between 5 preceding and 1 preceding
        ) as tchs_per_min_roll5,
        avg(b.pass_per_min) over (
            partition by b.player_id
            order by b.game_date
            rows between 5 preceding and 1 preceding
        ) as pass_per_min_roll5,
        avg(b.oreb_pct) over (
            partition by b.player_id
            order by b.game_date
            rows between 5 preceding and 1 preceding
        ) as oreb_pct_roll5,
        avg(b.dreb_pct) over (
            partition by b.player_id
            order by b.game_date
            rows between 5 preceding and 1 preceding
        ) as dreb_pct_roll5,
        avg(b.rbc_per_min) over (
            partition by b.player_id
            order by b.game_date
            rows between 5 preceding and 1 preceding
        ) as rbc_per_min_roll5,

        -- L10 player rolls
        avg(b.min) over (
            partition by b.player_id
            order by b.game_date
            rows between 10 preceding and 1 preceding
        ) as min_roll10,
        avg(b.pts) over (
            partition by b.player_id
            order by b.game_date
            rows between 10 preceding and 1 preceding
        ) as pts_roll10,
        avg(b.reb) over (
            partition by b.player_id
            order by b.game_date
            rows between 10 preceding and 1 preceding
        ) as reb_roll10,
        avg(b.ast) over (
            partition by b.player_id
            order by b.game_date
            rows between 10 preceding and 1 preceding
        ) as ast_roll10,
        avg(b.tov) over (
            partition by b.player_id
            order by b.game_date
            rows between 10 preceding and 1 preceding
        ) as tov_roll10,
        avg(b.usg_pct) over (
            partition by b.player_id
            order by b.game_date
            rows between 10 preceding and 1 preceding
        ) as usg_pct_roll10,
        avg(b.plus_minus) over (
            partition by b.player_id
            order by b.game_date
            rows between 10 preceding and 1 preceding
        ) as plus_minus_roll10,
        avg(b.pts_per_min) over (
            partition by b.player_id
            order by b.game_date
            rows between 10 preceding and 1 preceding
        ) as pts_per_min_roll10,
        avg(b.reb_per_min) over (
            partition by b.player_id
            order by b.game_date
            rows between 10 preceding and 1 preceding
        ) as reb_per_min_roll10,
        avg(b.ast_per_min) over (
            partition by b.player_id
            order by b.game_date
            rows between 10 preceding and 1 preceding
        ) as ast_per_min_roll10,
        avg(b.fga_per_min) over (
            partition by b.player_id
            order by b.game_date
            rows between 10 preceding and 1 preceding
        ) as fga_per_min_roll10,
        avg(b.fg3a_per_min) over (
            partition by b.player_id
            order by b.game_date
            rows between 10 preceding and 1 preceding
        ) as fg3a_per_min_roll10,
        avg(b.fta_per_min) over (
            partition by b.player_id
            order by b.game_date
            rows between 10 preceding and 1 preceding
        ) as fta_per_min_roll10,
        avg(b.cfga_per_min) over (
            partition by b.player_id
            order by b.game_date
            rows between 10 preceding and 1 preceding
        ) as cfga_per_min_roll10,
        avg(b.spd) over (
            partition by b.player_id
            order by b.game_date
            rows between 10 preceding and 1 preceding
        ) as spd_roll10,
        avg(b.pass_per_min) over (
            partition by b.player_id
            order by b.game_date
            rows between 10 preceding and 1 preceding
        ) as pass_per_min_roll10,
        avg(b.oreb_pct) over (
            partition by b.player_id
            order by b.game_date
            rows between 10 preceding and 1 preceding
        ) as oreb_pct_roll10,
        avg(b.dreb_pct) over (
            partition by b.player_id
            order by b.game_date
            rows between 10 preceding and 1 preceding
        ) as dreb_pct_roll10,
        avg(b.rbc_per_min) over (
            partition by b.player_id
            order by b.game_date
            rows between 10 preceding and 1 preceding
        ) as rbc_per_min_roll10,

        -- Season-to-date means (prior games only)
        avg(b.min) over (
            partition by b.player_id, b.season_year
            order by b.game_date
            rows between unbounded preceding and 1 preceding
        ) as min_season_mean,
        avg(b.pts_per_min) over (
            partition by b.player_id, b.season_year
            order by b.game_date
            rows between unbounded preceding and 1 preceding
        ) as pts_per_min_season_avg,
        avg(b.reb_per_min) over (
            partition by b.player_id, b.season_year
            order by b.game_date
            rows between unbounded preceding and 1 preceding
        ) as reb_per_min_season_avg,
        avg(b.ast_per_min) over (
            partition by b.player_id, b.season_year
            order by b.game_date
            rows between unbounded preceding and 1 preceding
        ) as ast_per_min_season_avg,

        -- Dispersion (L10 window)
        stddev_samp(b.min) over (
            partition by b.player_id
            order by b.game_date
            rows between 10 preceding and 1 preceding
        ) as min_std_l10,
        min(b.min) over (
            partition by b.player_id
            order by b.game_date
            rows between 10 preceding and 1 preceding
        ) as min_p10_l10,
        max(b.min) over (
            partition by b.player_id
            order by b.game_date
            rows between 10 preceding and 1 preceding
        ) as min_p90_l10,
        min(b.pts_per_min) over (
            partition by b.player_id
            order by b.game_date
            rows between 10 preceding and 1 preceding
        ) as ppm_p10_l10,
        max(b.pts_per_min) over (
            partition by b.player_id
            order by b.game_date
            rows between 10 preceding and 1 preceding
        ) as ppm_p90_l10,
        min(b.reb_per_min) over (
            partition by b.player_id
            order by b.game_date
            rows between 10 preceding and 1 preceding
        ) as reb_per_min_p10_l10,
        max(b.reb_per_min) over (
            partition by b.player_id
            order by b.game_date
            rows between 10 preceding and 1 preceding
        ) as reb_per_min_p90_l10,

        -- Starter roll (L10)
        avg(b.starting) over (
            partition by b.player_id
            order by b.game_date
            rows between 10 preceding and 1 preceding
        ) as starter_roll10_pct,

        -- Schedule context
        b.game_date - lag(b.game_date) over (
            partition by b.player_id
            order by b.game_date
        ) as days_rest,
        row_number() over (
            partition by b.player_id, b.season_year
            order by b.game_date
        ) - 1 as game_number

    from base as b
),

-- ── Season-level dispersion and team-relative ranks ─────────────────────────
player_ranks as (
    select
        pw.*,
        stddev_samp(pw.min) over (
            partition by pw.player_id, pw.season_year
            order by pw.game_date
            rows between unbounded preceding and 1 preceding
        ) as min_season_std,
        stddev_samp(pw.pts_per_min) over (
            partition by pw.player_id, pw.season_year
            order by pw.game_date
            rows between unbounded preceding and 1 preceding
        ) as ppm_season_std,
        stddev_samp(pw.reb_per_min) over (
            partition by pw.player_id, pw.season_year
            order by pw.game_date
            rows between unbounded preceding and 1 preceding
        ) as rpm_season_std,
        stddev_samp(pw.ast_per_min) over (
            partition by pw.player_id, pw.season_year
            order by pw.game_date
            rows between unbounded preceding and 1 preceding
        ) as ast_per_min_std_season,
        dense_rank() over (
            partition by pw.team_id, pw.game_date
            order by pw.min_roll10 desc nulls last, pw.player_id asc
        ) as team_min_rank_l10,
        dense_rank() over (
            partition by pw.team_id, pw.game_date
            order by pw.usg_pct_roll10 desc nulls last, pw.player_id asc
        ) as team_usg_rank_l10,
        dense_rank() over (
            partition by pw.team_id, pw.game_date
            order by pw.pts_per_min_roll10 desc nulls last, pw.player_id asc
        ) as team_pts_per_min_rank_l10,
        dense_rank() over (
            partition by pw.team_id, pw.game_date
            order by pw.ast_per_min_roll10 desc nulls last, pw.player_id asc
        ) as team_ast_per_min_rank_l10,
        pw.min_roll10 - lag(pw.min_roll10, 4) over (
            partition by pw.player_id
            order by pw.game_date
        ) as min_rate_of_change,
        pw.reb_per_min_roll10 - lag(pw.reb_per_min_roll10, 5) over (
            partition by pw.player_id
            order by pw.game_date
        ) / 5.0 as reb_roll10_slope,
        pw.oreb_pct_roll10 / nullif(pw.dreb_pct_roll10, 0) as oreb_dreb_ratio,
        pw.ast_per_min_roll5 - pw.ast_per_min_season_avg as apm_trend,
        case upper(trim(coalesce(pw.pos, 'UNK')))
            when 'G' then 0
            when 'F' then 1
            when 'C' then 2
            when 'G-F' then 3
            when 'F-G' then 4
            when 'F-C' then 5
            when 'C-F' then 6
            else 7
        end as position_encoded,
        sum(pw.starting) over (
            partition by pw.player_id
            order by pw.game_date
            rows between unbounded preceding and current row
        ) as consec_starts_raw
    from player_windows as pw
),

-- ── Team-level rolling defensive / pace context ─────────────────────────────
team_game as (
    select distinct
        team_id,
        season_year,
        game_id,
        game_date,
        team_abbreviation,
        team_pace,
        team_def_rating,
        team_poss,
        opp_fgm,
        opp_fga,
        opp_fg3a,
        opp_fta,
        opp_pts,
        opp_poss,
        opp_ast,
        opp_tov,
        opp_reb,
        opp_def_rating,
        opp_pace
    from base
),

team_roll as (
    select
        tg.*,
        avg(tg.team_pace) over (
            partition by tg.team_abbreviation
            order by tg.game_date
            rows between 10 preceding and 1 preceding
        ) as team_pace_roll10,
        avg(tg.team_def_rating) over (
            partition by tg.team_abbreviation
            order by tg.game_date
            rows between 10 preceding and 1 preceding
        ) as team_def_rating_roll10,
        avg(tg.team_poss) over (
            partition by tg.team_abbreviation
            order by tg.game_date
            rows between 10 preceding and 1 preceding
        ) as team_poss_roll10,
        avg(tg.team_def_rating) over (
            partition by tg.team_abbreviation
            order by tg.game_date
            rows between 5 preceding and 1 preceding
        ) as team_def_rating_roll5,
        avg(tg.team_pace) over (
            partition by tg.team_abbreviation
            order by tg.game_date
            rows between 5 preceding and 1 preceding
        ) as team_pace_roll5,
        avg(tg.team_poss) over (
            partition by tg.team_abbreviation
            order by tg.game_date
            rows between 5 preceding and 1 preceding
        ) as team_poss_roll5,
        -- Season-to-date opponent-allowed averages (defensive profiling)
        avg(tg.opp_pts) over (
            partition by tg.team_id, tg.season_year
            order by tg.game_date
            rows between unbounded preceding and 1 preceding
        ) as team_pts_allowed,
        avg(tg.opp_fta) over (
            partition by tg.team_id, tg.season_year
            order by tg.game_date
            rows between unbounded preceding and 1 preceding
        ) as team_fta_allowed,
        avg(tg.opp_fg3a) over (
            partition by tg.team_id, tg.season_year
            order by tg.game_date
            rows between unbounded preceding and 1 preceding
        ) as team_fg3a_allowed,
        avg(tg.opp_poss) over (
            partition by tg.team_id, tg.season_year
            order by tg.game_date
            rows between unbounded preceding and 1 preceding
        ) as team_poss_allowed,
        avg(tg.opp_ast) over (
            partition by tg.team_id, tg.season_year
            order by tg.game_date
            rows between unbounded preceding and 1 preceding
        ) as team_ast_allowed,
        avg(tg.opp_tov) over (
            partition by tg.team_id, tg.season_year
            order by tg.game_date
            rows between unbounded preceding and 1 preceding
        ) as team_tov_forced,
        avg(tg.opp_reb) over (
            partition by tg.team_id, tg.season_year
            order by tg.game_date
            rows between unbounded preceding and 1 preceding
        ) as team_reb_allowed,
        sum(tg.opp_fgm) over w / nullif(sum(tg.opp_fga) over w, 0) as team_fg_pct_allowed
    from team_game as tg
    window w as (
        partition by tg.team_id, tg.season_year
        order by tg.game_date
        rows between unbounded preceding and 1 preceding
    )
),

team_allowed as (
    select
        team_id,
        game_id,
        team_pace_roll10,
        team_def_rating_roll10,
        team_poss_roll10,
        team_def_rating_roll5,
        team_pace_roll5,
        team_poss_roll5,
        team_pts_allowed,
        team_fta_allowed,
        team_fg3a_allowed,
        team_fg_pct_allowed,
        team_poss_allowed,
        team_ast_allowed,
        team_tov_forced,
        team_reb_allowed
    from team_roll
),

opp_team_roll as (
    select
        team_id         as opp_team_id,
        game_id,
        team_def_rating_roll10  as opp_def_rating_roll10,
        team_pace_roll10        as opp_pace_roll10,
        team_poss_roll10        as opp_poss_roll10,
        team_def_rating_roll5   as opp_def_rating_roll5,
        team_pace_roll5         as opp_pace_roll5,
        team_poss_roll5         as opp_poss_roll5
    from team_allowed
),

opp_allowed as (
    select
        team_id         as opp_team_id,
        game_id,
        team_pts_allowed        as opp_team_pts_allowed,
        team_fta_allowed        as opp_team_fta_allowed,
        team_fg3a_allowed       as opp_team_fg3a_allowed,
        team_fg_pct_allowed     as opp_team_fg_pct_allowed,
        team_poss_allowed       as opp_team_poss_allowed,
        team_ast_allowed        as opp_team_ast_allowed,
        team_tov_forced         as opp_team_tov_forced,
        team_reb_allowed        as opp_team_reb_allowed
    from team_allowed
)

-- ── Final assembly ───────────────────────────────────────────────────────────
select
    -- identifiers
    pr.game_id,
    pr.player_id,
    pr.player_name,
    pr.normalized_name,
    pr.season_year,
    pr.season_type,
    pr.game_date,
    pr.matchup,
    pr.team_id,
    pr.team_abbreviation,
    pr.team_name,
    pr.opp_team_id,
    pr.opp_team_abbreviation,
    pr.opp_team_name,
    -- game context
    pr.wl,
    pr.starting,
    pr.is_home,
    pr.is_playoff,
    pr.pos,
    -- raw box stats
    pr.min,
    pr.pts,
    pr.reb,
    pr.oreb,
    pr.dreb,
    pr.ast,
    pr.tov,
    pr.stl,
    pr.blk,
    pr.fgm,
    pr.fga,
    pr.fg3a,
    pr.fta,
    pr.plus_minus,
    -- advanced
    pr.usg_pct,
    pr.ts_pct,
    pr.efg_pct,
    pr.net_rating,
    pr.pie,
    pr.poss,
    pr.ast_pct,
    pr.ast_to,
    -- tracking
    pr.spd,
    pr.dist,
    pr.cfga,
    pr.sast,
    pr.pass,
    pr.tchs,
    -- team context (raw)
    pr.team_pace,
    pr.team_def_rating,
    pr.team_spread,
    pr.game_total,
    -- per-minute rates
    pr.pts_per_min,
    pr.reb_per_min,
    pr.ast_per_min,
    pr.tov_per_min,
    pr.fga_per_min,
    pr.fg3a_per_min,
    pr.fta_per_min,
    pr.cfga_per_min,
    pr.poss_per_min,
    pr.sast_per_min,
    pr.pass_per_min,
    pr.tchs_per_min,
    -- L5 player rolls
    pr.min_roll5,
    pr.pts_roll5,
    pr.reb_roll5,
    pr.ast_roll5,
    pr.tov_roll5,
    pr.usg_pct_roll5,
    pr.plus_minus_roll5,
    pr.pts_per_min_roll5,
    pr.reb_per_min_roll5,
    pr.ast_per_min_roll5,
    pr.poss_per_min_roll5,
    pr.sast_per_min_roll5,
    pr.tchs_per_min_roll5,
    pr.pass_per_min_roll5,
    pr.oreb_pct_roll5,
    pr.dreb_pct_roll5,
    pr.rbc_per_min_roll5,
    -- L10 player rolls
    pr.min_roll10,
    pr.pts_roll10,
    pr.reb_roll10,
    pr.ast_roll10,
    pr.tov_roll10,
    pr.usg_pct_roll10,
    pr.plus_minus_roll10,
    pr.pts_per_min_roll10,
    pr.reb_per_min_roll10,
    pr.ast_per_min_roll10,
    pr.fga_per_min_roll10,
    pr.fg3a_per_min_roll10,
    pr.fta_per_min_roll10,
    pr.cfga_per_min_roll10,
    pr.spd_roll10,
    pr.pass_per_min_roll10,
    pr.oreb_pct_roll10,
    pr.dreb_pct_roll10,
    pr.rbc_per_min_roll10,
    -- season-to-date
    pr.min_season_mean,
    pr.pts_per_min_season_avg,
    pr.reb_per_min_season_avg,
    pr.ast_per_min_season_avg,
    pr.min_std_l10,
    pr.min_p10_l10,
    pr.min_p90_l10,
    pr.ppm_p10_l10,
    pr.ppm_p90_l10,
    pr.reb_per_min_p10_l10,
    pr.reb_per_min_p90_l10,
    pr.min_season_std,
    pr.ppm_season_std,
    pr.rpm_season_std,
    pr.ast_per_min_std_season,
    pr.starter_roll10_pct,
    -- schedule
    pr.days_rest,
    case when pr.days_rest = 1 then 1 else 0 end as is_b2b,
    pr.game_number,
    -- team ranks
    pr.team_min_rank_l10,
    pr.team_usg_rank_l10,
    pr.team_pts_per_min_rank_l10,
    pr.team_ast_per_min_rank_l10,
    pr.min_rate_of_change,
    pr.reb_roll10_slope,
    pr.oreb_dreb_ratio,
    pr.apm_trend,
    pr.position_encoded,
    pr.position_encoded as position_enc,
    pr.consec_starts_raw as consec_starts,
    -- placeholder availability flags (populated downstream)
    0::smallint as top_player_active,
    0::smallint as active_stars_count,
    -- EWM-aligned aliases (min_roll10 used as proxy)
    pr.min_roll10          as min_10_ewm,
    pr.spd_roll10          as spd_10_ewm,
    pr.pts_per_min_roll10  as pts_per_min_10_ewm,
    pr.fga_per_min_roll10  as fga_per_min_10_ewm,
    pr.cfga_per_min_roll10 as cfga_per_min_10_ewm,
    pr.fg3a_per_min_roll10 as fg3a_per_min_10_ewm,
    pr.fta_per_min_roll10  as fta_per_min_10_ewm,
    pr.rbc_per_min_roll10  as rbc_per_min_10_ewm,
    pr.reb_per_min_roll10  as reb_per_min_10_ewm,
    pr.pass_per_min_roll5  as pass_per_min_5_ewm,
    -- interaction features (player rate × opponent defensive profile)
    pr.pts_per_min_roll10  * oa.opp_team_pts_allowed          as pts_per_min_x_opp_pts_allowed,
    pr.cfga_per_min_roll10 * oa.opp_team_fg_pct_allowed        as cfga_per_min_x_opp_fg_pct_allowed,
    pr.fga_per_min_roll10  * ot.opp_def_rating_roll10          as fga_per_min_x_opp_def_rating,
    pr.fg3a_per_min_roll10 * oa.opp_team_fg3a_allowed          as fg3a_per_min_x_opp_team_fg3a_allowed,
    pr.fta_per_min_roll10  * oa.opp_team_fta_allowed           as fta_per_min_x_opp_fta_allowed,
    -- team rolling context (joined)
    ta.team_pace_roll10,
    ta.team_def_rating_roll10,
    ta.team_poss_roll10,
    -- opponent rolling context (joined)
    ot.opp_def_rating_roll10,
    ot.opp_pace_roll10,
    ot.opp_poss_roll10,
    ot.opp_def_rating_roll5,
    ot.opp_pace_roll5,
    ot.opp_poss_roll5,
    -- opponent season-to-date defensive allowed averages
    oa.opp_team_pts_allowed,
    oa.opp_team_fta_allowed,
    oa.opp_team_fg3a_allowed,
    oa.opp_team_fg_pct_allowed,
    oa.opp_team_poss_allowed,
    oa.opp_team_ast_allowed,
    oa.opp_team_tov_forced,
    oa.opp_team_reb_allowed,
    -- expected pace / pace differential
    (ta.team_pace_roll10 + ot.opp_pace_roll10) / 2.0 as expected_pace,
    ta.team_pace_roll10 - ot.opp_pace_roll10          as pace_differential
from player_ranks as pr
left join team_allowed as ta
    on pr.team_id = ta.team_id
   and pr.game_id = ta.game_id
left join opp_team_roll as ot
    on pr.opp_team_id = ot.opp_team_id
   and pr.game_id = ot.game_id
left join opp_allowed as oa
    on pr.opp_team_id = oa.opp_team_id
   and pr.game_id = oa.game_id
