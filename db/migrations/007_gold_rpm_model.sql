-- Gold layer: NBA RPM model training / inference frame.
-- Source: src.pipeline.gold.build_rpm_gold_from_silver(..., league="nba")
-- Upload: src.utils.db.upsert_prop_gold("rpm", df, league="nba")
-- Feature columns match src.pipeline.features.rpm_features.RPM_FEATURES.

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
    game_id                                 TEXT        NOT NULL,
    player_id                               BIGINT      NOT NULL,
    built_at                                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    season                                  TEXT,
    player_name                             TEXT,
    game_date                               DATE,
    reb_per_min                             NUMERIC,
    min                                     NUMERIC,
    starting                                SMALLINT,
    -- Recent rebounding rate (5)
    player_reb_per_min_ewma                 NUMERIC,
    player_reb_per_min_expanding_mean       NUMERIC,
    player_reb_per_min_season_mean          NUMERIC,
    player_reb_per_min_roll5_mean           NUMERIC,
    player_reb_per_min_roll10_mean          NUMERIC,
    -- Rebound share / split (5)
    player_reb_pct_season_mean              NUMERIC,
    player_reb_pct_roll10_mean              NUMERIC,
    player_oreb_pct_roll10_mean             NUMERIC,
    player_dreb_pct_season_mean             NUMERIC,
    player_dreb_pct_roll10_mean             NUMERIC,
    -- Rebound chances (3)
    player_orbc_roll10_mean                 NUMERIC,
    player_rbc_roll10_mean                  NUMERIC,
    player_drbc_roll10_mean                 NUMERIC,
    -- Minutes (3)
    player_min_roll10_mean                  NUMERIC,
    player_min_roll5_mean                   NUMERIC,
    player_min_season_mean                  NUMERIC,
    -- Rebound-chance trend (2)
    rbc_trend_5v20                          NUMERIC,
    drbc_trend_5v20                         NUMERIC,
    -- Physical / role proxy (2)
    position_encoded                        SMALLINT,
    starting_rate_last10                    NUMERIC,
    -- Matchup / opportunity (1)
    opp_team_team_reb_pct_roll10_mean       NUMERIC,
    -- Schedule (1)
    days_rest                               NUMERIC,

    PRIMARY KEY (game_id, player_id)
);

CREATE INDEX IF NOT EXISTS idx_gold_nba_rpm_season ON gold.nba_player_rpm_model (season);
CREATE INDEX IF NOT EXISTS idx_gold_nba_rpm_date ON gold.nba_player_rpm_model (game_date);
CREATE INDEX IF NOT EXISTS idx_gold_nba_rpm_player ON gold.nba_player_rpm_model (player_id);

-- Ensure selected features exist on legacy wide tables.
ALTER TABLE gold.nba_player_rpm_model ADD COLUMN IF NOT EXISTS player_reb_per_min_ewma NUMERIC;
ALTER TABLE gold.nba_player_rpm_model ADD COLUMN IF NOT EXISTS player_reb_per_min_expanding_mean NUMERIC;
ALTER TABLE gold.nba_player_rpm_model ADD COLUMN IF NOT EXISTS player_reb_per_min_season_mean NUMERIC;
ALTER TABLE gold.nba_player_rpm_model ADD COLUMN IF NOT EXISTS player_reb_per_min_roll5_mean NUMERIC;
ALTER TABLE gold.nba_player_rpm_model ADD COLUMN IF NOT EXISTS player_reb_per_min_roll10_mean NUMERIC;
ALTER TABLE gold.nba_player_rpm_model ADD COLUMN IF NOT EXISTS player_reb_pct_season_mean NUMERIC;
ALTER TABLE gold.nba_player_rpm_model ADD COLUMN IF NOT EXISTS player_reb_pct_roll10_mean NUMERIC;
ALTER TABLE gold.nba_player_rpm_model ADD COLUMN IF NOT EXISTS player_oreb_pct_roll10_mean NUMERIC;
ALTER TABLE gold.nba_player_rpm_model ADD COLUMN IF NOT EXISTS player_dreb_pct_season_mean NUMERIC;
ALTER TABLE gold.nba_player_rpm_model ADD COLUMN IF NOT EXISTS player_dreb_pct_roll10_mean NUMERIC;
ALTER TABLE gold.nba_player_rpm_model ADD COLUMN IF NOT EXISTS player_orbc_roll10_mean NUMERIC;
ALTER TABLE gold.nba_player_rpm_model ADD COLUMN IF NOT EXISTS player_rbc_roll10_mean NUMERIC;
ALTER TABLE gold.nba_player_rpm_model ADD COLUMN IF NOT EXISTS player_drbc_roll10_mean NUMERIC;
ALTER TABLE gold.nba_player_rpm_model ADD COLUMN IF NOT EXISTS player_min_roll10_mean NUMERIC;
ALTER TABLE gold.nba_player_rpm_model ADD COLUMN IF NOT EXISTS player_min_roll5_mean NUMERIC;
ALTER TABLE gold.nba_player_rpm_model ADD COLUMN IF NOT EXISTS player_min_season_mean NUMERIC;
ALTER TABLE gold.nba_player_rpm_model ADD COLUMN IF NOT EXISTS rbc_trend_5v20 NUMERIC;
ALTER TABLE gold.nba_player_rpm_model ADD COLUMN IF NOT EXISTS drbc_trend_5v20 NUMERIC;
ALTER TABLE gold.nba_player_rpm_model ADD COLUMN IF NOT EXISTS position_encoded SMALLINT;
ALTER TABLE gold.nba_player_rpm_model ADD COLUMN IF NOT EXISTS starting_rate_last10 NUMERIC;
ALTER TABLE gold.nba_player_rpm_model ADD COLUMN IF NOT EXISTS opp_team_team_reb_pct_roll10_mean NUMERIC;
ALTER TABLE gold.nba_player_rpm_model ADD COLUMN IF NOT EXISTS days_rest NUMERIC;
