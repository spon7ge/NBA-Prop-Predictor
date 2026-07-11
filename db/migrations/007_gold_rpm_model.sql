-- Gold layer: NBA RPM quantile model training / inference frame.
-- Source: src.utils.gold.build_rpm_gold_from_silver(..., league="nba")
-- Upload: src.utils.db.upsert_prop_gold("rpm", df, league="nba")

CREATE SCHEMA IF NOT EXISTS gold;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'gold' AND table_name = 'player_rpm_model'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'gold' AND table_name = 'nba_player_rpm_model'
    ) THEN
        ALTER TABLE gold.player_rpm_model RENAME TO nba_player_rpm_model;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS gold.nba_player_rpm_model (
    game_id                      TEXT        NOT NULL,
    player_id                    BIGINT      NOT NULL,
    built_at                     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    season                       TEXT,
    player_name                  TEXT,
    game_date                    DATE,
    reb_per_min                  NUMERIC,
    min                          NUMERIC,
    starting                     SMALLINT,
    reb_per_min_season_avg       NUMERIC,
    reb_per_min_10_ewm           NUMERIC,
    oreb_dreb_ratio              NUMERIC,
    position_enc                 SMALLINT,
    rbc_per_min_10_ewm           NUMERIC,
    rpm_season_std               NUMERIC,
    reb_roll10_slope             NUMERIC,
    reb_per_min_p10_l10          NUMERIC,
    reb_per_min_p90_l10          NUMERIC,

    PRIMARY KEY (game_id, player_id)
);

CREATE INDEX IF NOT EXISTS idx_gold_nba_rpm_season ON gold.nba_player_rpm_model (season);
CREATE INDEX IF NOT EXISTS idx_gold_nba_rpm_date ON gold.nba_player_rpm_model (game_date);
CREATE INDEX IF NOT EXISTS idx_gold_nba_rpm_player ON gold.nba_player_rpm_model (player_id);
