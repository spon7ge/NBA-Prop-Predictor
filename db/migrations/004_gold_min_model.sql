-- Gold layer: NBA MIN quantile model training / inference frame.
-- Source: src.utils.gold.build_min_gold_from_silver(..., league="nba")
-- Upload: src.utils.db.upsert_prop_gold("min", df, league="nba")

CREATE SCHEMA IF NOT EXISTS gold;

-- Rename legacy unprefixed table if present.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'gold' AND table_name = 'player_min_model'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'gold' AND table_name = 'nba_player_min_model'
    ) THEN
        ALTER TABLE gold.player_min_model RENAME TO nba_player_min_model;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS gold.nba_player_min_model (
    game_id                      TEXT        NOT NULL,
    player_id                    BIGINT      NOT NULL,
    built_at                     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    season                       TEXT,
    player_name                  TEXT,
    game_date                    DATE,
    min                          NUMERIC,
    starting                     SMALLINT,
    min_10_ewm                   NUMERIC,
    min_season_mean              NUMERIC,
    starter_roll10_pct           NUMERIC,
    consec_starts                NUMERIC,
    min_rate_of_change           NUMERIC,
    team_min_rank_l10            NUMERIC,
    team_usg_rank_l10            NUMERIC,
    min_p10_l10                  NUMERIC,
    min_p90_l10                  NUMERIC,
    min_std_l10                  NUMERIC,
    min_season_std               NUMERIC,
    spd_10_ewm                   NUMERIC,
    usg_pct_lag1                 NUMERIC,
    usg_pct_lag2                 NUMERIC,
    ast_pct_lag1                 NUMERIC,
    ast_pct_lag2                 NUMERIC,
    pie_lag1                     NUMERIC,
    pie_lag2                     NUMERIC,
    games_played_last_7_days     SMALLINT,
    games_played_last_14_days    SMALLINT,
    min_sum_last_7_days          NUMERIC,
    top_player_active            SMALLINT,
    active_stars_count           SMALLINT,

    PRIMARY KEY (game_id, player_id)
);

CREATE INDEX IF NOT EXISTS idx_gold_nba_min_season ON gold.nba_player_min_model (season);
CREATE INDEX IF NOT EXISTS idx_gold_nba_min_date ON gold.nba_player_min_model (game_date);
CREATE INDEX IF NOT EXISTS idx_gold_nba_min_player ON gold.nba_player_min_model (player_id);

ALTER TABLE gold.nba_player_min_model ADD COLUMN IF NOT EXISTS usg_pct_lag1 NUMERIC;
ALTER TABLE gold.nba_player_min_model ADD COLUMN IF NOT EXISTS usg_pct_lag2 NUMERIC;
ALTER TABLE gold.nba_player_min_model ADD COLUMN IF NOT EXISTS ast_pct_lag1 NUMERIC;
ALTER TABLE gold.nba_player_min_model ADD COLUMN IF NOT EXISTS ast_pct_lag2 NUMERIC;
ALTER TABLE gold.nba_player_min_model ADD COLUMN IF NOT EXISTS pie_lag1 NUMERIC;
ALTER TABLE gold.nba_player_min_model ADD COLUMN IF NOT EXISTS pie_lag2 NUMERIC;
ALTER TABLE gold.nba_player_min_model ADD COLUMN IF NOT EXISTS games_played_last_7_days SMALLINT;
ALTER TABLE gold.nba_player_min_model ADD COLUMN IF NOT EXISTS games_played_last_14_days SMALLINT;
ALTER TABLE gold.nba_player_min_model ADD COLUMN IF NOT EXISTS min_sum_last_7_days NUMERIC;
