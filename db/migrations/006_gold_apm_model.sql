-- Gold layer: NBA APM model training / inference frame.
-- Source: src.pipeline.gold.build_apm_gold_from_silver(..., league="nba")
-- Upload: src.utils.db.upsert_prop_gold("apm", df, league="nba")
-- Feature columns match src.pipeline.features.apm_features.APM_FEATURES.

CREATE SCHEMA IF NOT EXISTS gold;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'gold' AND table_name = 'player_apm_model'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'gold' AND table_name = 'nba_player_apm_model'
    ) THEN
        ALTER TABLE gold.player_apm_model RENAME TO nba_player_apm_model;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS gold.nba_player_apm_model (
    game_id                                 TEXT        NOT NULL,
    player_id                               BIGINT      NOT NULL,
    built_at                                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    season                                  TEXT,
    player_name                             TEXT,
    game_date                               DATE,
    ast_per_min                             NUMERIC,
    min                                     NUMERIC,
    starting                                SMALLINT,
    -- Recent assist rate (1)
    player_ast_per_min_ewma                 NUMERIC,
    -- Ball-handling volume (2)
    player_pass_roll10_mean                 NUMERIC,
    player_pass_season_mean                 NUMERIC,
    -- Secondary playmaking / role (2)
    player_sast_roll10_mean                 NUMERIC,
    player_ast_ratio_roll10_mean            NUMERIC,
    -- Minutes / role (2)
    player_min_roll5_mean                   NUMERIC,
    min_trend_5v20                          NUMERIC,
    -- Role / physical proxy (1)
    position_encoded                        SMALLINT,
    -- Team / matchup context (3)
    opp_team_team_def_rating_roll10_mean    NUMERIC,
    opp_team_team_pace_roll10_mean          NUMERIC,
    own_team_team_ast_pct_roll10_mean       NUMERIC,

    PRIMARY KEY (game_id, player_id)
);

CREATE INDEX IF NOT EXISTS idx_gold_nba_apm_season ON gold.nba_player_apm_model (season);
CREATE INDEX IF NOT EXISTS idx_gold_nba_apm_date ON gold.nba_player_apm_model (game_date);
CREATE INDEX IF NOT EXISTS idx_gold_nba_apm_player ON gold.nba_player_apm_model (player_id);

-- Ensure selected features exist on legacy wide tables.
ALTER TABLE gold.nba_player_apm_model ADD COLUMN IF NOT EXISTS player_ast_per_min_ewma NUMERIC;
ALTER TABLE gold.nba_player_apm_model ADD COLUMN IF NOT EXISTS player_pass_roll10_mean NUMERIC;
ALTER TABLE gold.nba_player_apm_model ADD COLUMN IF NOT EXISTS player_pass_season_mean NUMERIC;
ALTER TABLE gold.nba_player_apm_model ADD COLUMN IF NOT EXISTS player_sast_roll10_mean NUMERIC;
ALTER TABLE gold.nba_player_apm_model ADD COLUMN IF NOT EXISTS player_ast_ratio_roll10_mean NUMERIC;
ALTER TABLE gold.nba_player_apm_model ADD COLUMN IF NOT EXISTS player_min_roll5_mean NUMERIC;
ALTER TABLE gold.nba_player_apm_model ADD COLUMN IF NOT EXISTS min_trend_5v20 NUMERIC;
ALTER TABLE gold.nba_player_apm_model ADD COLUMN IF NOT EXISTS position_encoded SMALLINT;
ALTER TABLE gold.nba_player_apm_model ADD COLUMN IF NOT EXISTS opp_team_team_def_rating_roll10_mean NUMERIC;
ALTER TABLE gold.nba_player_apm_model ADD COLUMN IF NOT EXISTS opp_team_team_pace_roll10_mean NUMERIC;
ALTER TABLE gold.nba_player_apm_model ADD COLUMN IF NOT EXISTS own_team_team_ast_pct_roll10_mean NUMERIC;
