-- Backtesting accuracy: score ml.predictions against realized game stats.
--
-- Grain: one row per (prop, game_id, player_id, model_id).
--
-- Games whose actuals have not yet landed are excluded via the inner join
-- on silver_player_gamelogs (safe to run daily — incomplete games simply
-- produce no rows rather than errors).
--
-- Upstream lineage:
--   ml.predictions  (source)
--   silver_player_gamelogs (ref) → actual counting stats + per-min rates
--   gold_prop_history (ref)      → book lines for hit-rate calculation
--
-- Key output columns:
--   signed_error          – prediction minus actual in model-native units
--                           (per-min rates for ppm/rpm/apm, minutes for min).
--                           Positive = model over-predicted.
--   abs_error             – |signed_error|; average → MAE per model.
--   q50_below_prediction  – 1 when actual fell below the median (Q50) prediction.
--                           Long-run average should be ~0.50 for a calibrated model.
--   predicted_total       – rate × actual_min converted back to a counting-stat
--                           total so it can be compared to book lines (points,
--                           rebounds, assists). For min the prediction is already
--                           a total.
--   hit                   – 1 if the model's predicted direction (over/under the
--                           book line) matched the actual outcome; null when no
--                           book line is available for that player × date × market.

with predictions as (
    select
        prop,
        game_id,
        player_id,
        prediction,
        predicted_at,
        game_date,
        player_name,
        model_id
    from {{ source('ml', 'predictions') }}
    where game_date  is not null
      and model_id   is not null
),

actuals as (
    select
        game_id,
        player_id,
        game_date,
        min,
        pts,
        reb,
        ast,
        pts_per_min,
        reb_per_min,
        ast_per_min
    from {{ ref('silver_player_gamelogs') }}
    where min > 0   -- exclude DNP / no-play rows so errors aren't inflated
),

-- Attach actual values; games without a matching gamelog row are dropped.
scored as (
    select
        p.prop,
        p.game_id,
        p.player_id,
        p.player_name,
        p.game_date,
        p.model_id,
        p.predicted_at,
        p.prediction,

        -- actual value in the same units the model predicted (per-min or min)
        case p.prop
            when 'min' then a.min
            when 'ppm' then a.pts_per_min
            when 'rpm' then a.reb_per_min
            when 'apm' then a.ast_per_min
        end                                                 as actual_stat,

        -- counting-stat totals (pts / reb / ast / min) for book-line comparison
        case p.prop
            when 'min' then a.min
            when 'ppm' then a.pts
            when 'rpm' then a.reb
            when 'apm' then a.ast
        end                                                 as actual_total,

        -- convert rate predictions to counting-stat totals using actual minutes
        -- so they live in the same space as book lines
        case p.prop
            when 'min' then p.prediction
            else            p.prediction * nullif(a.min, 0)
        end                                                 as predicted_total,

        -- standardized market category key used to join gold_prop_history
        case p.prop
            when 'min' then 'minutes'
            when 'ppm' then 'points'
            when 'rpm' then 'rebounds'
            when 'apm' then 'assists'
        end                                                 as market_category,

        -- signed error in model-native units; positive = model over-predicted
        p.prediction - case p.prop
            when 'min' then a.min
            when 'ppm' then a.pts_per_min
            when 'rpm' then a.reb_per_min
            when 'apm' then a.ast_per_min
        end                                                 as signed_error,

        abs(p.prediction - case p.prop
            when 'min' then a.min
            when 'ppm' then a.pts_per_min
            when 'rpm' then a.reb_per_min
            when 'apm' then a.ast_per_min
        end)                                                as abs_error,

        -- Q50 calibration flag: 1 when actual fell strictly below median prediction
        case
            when (case p.prop
                    when 'min' then a.min
                    when 'ppm' then a.pts_per_min
                    when 'rpm' then a.reb_per_min
                    when 'apm' then a.ast_per_min
                  end) < p.prediction
            then 1.0 else 0.0
        end                                                 as q50_below_prediction

    from predictions as p
    inner join actuals as a
        on  p.game_id   = a.game_id
        and p.player_id = a.player_id
),

-- Average book 'over' lines per (player, game_date, market_category)
-- across all bookmakers/sources so each scored row gets a single comparable line.
book_lines as (
    select
        player_id,
        game_date,
        market_category,
        avg(line)   as book_line
    from {{ ref('gold_prop_history') }}
    where side = 'over'
      and line is not null
    group by player_id, game_date, market_category
)

select
    s.prop,
    s.game_id,
    s.player_id,
    s.player_name,
    s.game_date,
    s.model_id,
    s.predicted_at,
    s.prediction,
    s.actual_stat,
    s.actual_total,
    s.predicted_total,
    s.market_category,
    s.signed_error,
    s.abs_error,
    s.q50_below_prediction,
    bl.book_line,

    -- 1 when model direction matches actual direction vs the book line;
    -- null when no book line was recorded for this player × date × market.
    case
        when bl.book_line is null                                  then null
        when s.predicted_total >  bl.book_line
             and s.actual_total  >  bl.book_line                   then 1.0
        when s.predicted_total <= bl.book_line
             and s.actual_total  <= bl.book_line                   then 1.0
        else 0.0
    end                                                             as hit

from scored as s
left join book_lines as bl
    on  s.player_id       = bl.player_id
    and s.game_date       = bl.game_date
    and s.market_category = bl.market_category
