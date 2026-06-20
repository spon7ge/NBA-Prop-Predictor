-- NYK 2026 Playoff Analysis — DuckDB SQL
-- Companion to notebooks/NYK_Analysis.ipynb
-- Load data first:
--   CREATE TABLE pbp AS SELECT * FROM read_parquet('data/raw/pbp_stats/p26_analysis.parquet');
--   CREATE TABLE nyk_stats AS SELECT * FROM read_csv('data/raw/playoff_stats/P26.csv') WHERE TEAM_ID = 1610612752;

-- ── Views ────────────────────────────────────────────────────────────────────

CREATE OR REPLACE VIEW nyk_games AS
SELECT DISTINCT gameId FROM pbp WHERE teamId = 1610612752;

CREATE OR REPLACE VIEW pbp_nyk AS
SELECT p.* FROM pbp p INNER JOIN nyk_games g USING (gameId);

CREATE OR REPLACE VIEW game_meta AS
SELECT DISTINCT
    LPAD(CAST(GAME_ID AS VARCHAR), 10, '0') AS gameId,
    GAME_ID, GAME_DATE, MATCHUP, WL, OPP_OPP_ABBREVIATION_base
FROM nyk_stats;

CREATE OR REPLACE VIEW nyk_loc AS
SELECT gameId, MAX(location) AS nyk_loc
FROM pbp WHERE teamId = 1610612752
GROUP BY 1;

CREATE OR REPLACE VIEW games_enriched AS
WITH base AS (
    SELECT
        p.gameId, p.actionNumber, p.period, p.clock, p.teamId,
        p.playerNameI, p.actionType, p.description, p.shotValue,
        CAST(regexp_extract(p.clock, 'PT(\d+)M', 1) AS INT) * 60
            + CAST(regexp_extract(p.clock, 'M([\d.]+)S', 1) AS DOUBLE) AS secs_remaining,
        l.nyk_loc,
        LAST_VALUE(TRY_CAST(p.scoreHome AS DOUBLE) IGNORE NULLS) OVER (
            PARTITION BY p.gameId ORDER BY p.actionNumber
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS score_home_n,
        LAST_VALUE(TRY_CAST(p.scoreAway AS DOUBLE) IGNORE NULLS) OVER (
            PARTITION BY p.gameId ORDER BY p.actionNumber
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS score_away_n
    FROM pbp_nyk p
    INNER JOIN nyk_loc l USING (gameId)
)
SELECT *,
    CASE WHEN nyk_loc = 'h' THEN COALESCE(score_home_n, 0) ELSE COALESCE(score_away_n, 0) END AS nyk_live,
    CASE WHEN nyk_loc = 'h' THEN COALESCE(score_away_n, 0) ELSE COALESCE(score_home_n, 0) END AS opp_live,
    CASE WHEN nyk_loc = 'h' THEN COALESCE(score_home_n, 0) ELSE COALESCE(score_away_n, 0) END
      - CASE WHEN nyk_loc = 'h' THEN COALESCE(score_away_n, 0) ELSE COALESCE(score_home_n, 0) END AS live_margin
FROM base;

CREATE OR REPLACE VIEW clutch_plays AS
SELECT * FROM games_enriched
WHERE period >= 4 AND secs_remaining <= 300 AND ABS(live_margin) <= 5;

-- ── 1. Quarter scoring splits by series ───────────────────────────────────────

WITH period_ends AS (
    SELECT gameId, period, nyk_live, opp_live
    FROM games_enriched
    QUALIFY ROW_NUMBER() OVER (PARTITION BY gameId, period ORDER BY actionNumber DESC) = 1
),
quarter_pts AS (
    SELECT
        gameId, period,
        nyk_live - COALESCE(LAG(nyk_live) OVER (PARTITION BY gameId ORDER BY period), 0) AS nyk_score_q,
        opp_live - COALESCE(LAG(opp_live) OVER (PARTITION BY gameId ORDER BY period), 0) AS opp_score_q
    FROM period_ends
    WHERE period <= 4
)
SELECT
    'vs ' || m.OPP_OPP_ABBREVIATION_base AS series,
    'Q' || CAST(q.period AS VARCHAR) AS quarter,
    ROUND(AVG(q.nyk_score_q), 1) AS nyk_pts,
    ROUND(AVG(q.opp_score_q), 1) AS opp_pts,
    COUNT(DISTINCT q.gameId) AS games
FROM quarter_pts q
JOIN game_meta m ON LPAD(CAST(q.gameId AS VARCHAR), 10, '0') = m.gameId
GROUP BY 1, 2
ORDER BY 1, 2;

-- ── 2. Clutch scorers ─────────────────────────────────────────────────────────

SELECT
    playerNameI,
    SUM(CASE
        WHEN actionType = 'Made Shot' THEN shotValue
        WHEN actionType = 'Free Throw' AND UPPER(description) NOT LIKE 'MISS%' THEN 1
        ELSE 0
    END) AS clutch_pts
FROM clutch_plays
WHERE teamId = 1610612752
GROUP BY 1
HAVING clutch_pts > 0
ORDER BY clutch_pts DESC
LIMIT 8;

-- ── 3. Player playoff averages ──────────────────────────────────────────────

SELECT
    PLAYER_NAME,
    COUNT(DISTINCT GAME_ID) AS GP,
    ROUND(AVG(PTS), 2) AS PPG,
    ROUND(AVG(PLUS_MINUS), 2) AS avg_pm,
    RANK() OVER (ORDER BY AVG(PTS) DESC) AS pts_rank
FROM nyk_stats
GROUP BY 1
HAVING COUNT(DISTINCT GAME_ID) >= 2
ORDER BY pts_rank;

-- ── 4. Analyst question: comeback wins when trailing 10+ ──────────────────────

WITH max_def AS (
    SELECT gameId, MIN(live_margin) AS max_deficit
    FROM games_enriched
    GROUP BY 1
)
SELECT
    m.GAME_DATE,
    m.MATCHUP,
    GREATEST(-d.max_deficit, 0) AS pts_trailed,
    m.TEAM_PTS,
    m.OPP_PTS
FROM max_def d
JOIN game_meta gm ON LPAD(CAST(d.gameId AS VARCHAR), 10, '0') = gm.gameId
JOIN (
    SELECT DISTINCT GAME_DATE, MATCHUP, TEAM_PTS, OPP_PTS
    FROM nyk_stats
) m ON gm.GAME_DATE = m.GAME_DATE AND gm.MATCHUP = m.MATCHUP
WHERE gm.WL = 'W' AND GREATEST(-d.max_deficit, 0) >= 10
ORDER BY pts_trailed DESC;
