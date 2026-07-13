-- Gold layer: NBA PPM model training / inference frame.
-- Source: src.pipeline.gold.build_ppm_gold_from_silver(..., league="nba")
-- Upload: src.utils.db.upsert_prop_gold("ppm", df, league="nba")
-- Feature columns match src.pipeline.features.ppm_features.PPM_FEATURES.

CREATE SCHEMA IF NOT EXISTS gold;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'gold' AND table_name = 'player_ppm_model'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'gold' AND table_name = 'nba_player_ppm_model'
    ) THEN
        ALTER TABLE gold.player_ppm_model RENAME TO nba_player_ppm_model;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS gold.nba_player_ppm_model (
    game_id                                 TEXT        NOT NULL,
    player_id                               BIGINT      NOT NULL,
    built_at                                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    season                                  TEXT,
    player_name                             TEXT,
    game_date                               DATE,
    pts_per_min                             NUMERIC,
    min                                     NUMERIC,
    starting                                SMALLINT,
    -- Recent scoring rate (3)
    player_pts_per_min_expanding_mean       NUMERIC,
    player_pts_per_min_roll5_mean           NUMERIC,
    player_pts_per_min_roll10_mean          NUMERIC,
    -- Usage (1)
    player_usg_pct_season_mean              NUMERIC,
    -- Shot volume (2)
    player_fga_season_mean                  NUMERIC,
    player_fga_roll5_mean                   NUMERIC,
    -- Free throws (1)
    player_ft_pct_season_mean               NUMERIC,
    -- Efficiency (2)
    player_fg3_pct_roll10_mean              NUMERIC,
    player_efg_pct_roll10_mean              NUMERIC,
    -- Minutes (2)
    player_min_season_mean                  NUMERIC,
    player_min_roll10_mean                  NUMERIC,
    -- Role trend (2)
    min_trend_5v20                          NUMERIC,
    ts_trend_5v20                           NUMERIC,
    -- Context (3)
    opp_team_team_def_rating_roll10_mean    NUMERIC,
    days_rest                               NUMERIC,
    starting_rate_last10                    NUMERIC,

    PRIMARY KEY (game_id, player_id)
);

CREATE INDEX IF NOT EXISTS idx_gold_nba_ppm_season ON gold.nba_player_ppm_model (season);
CREATE INDEX IF NOT EXISTS idx_gold_nba_ppm_date ON gold.nba_player_ppm_model (game_date);
CREATE INDEX IF NOT EXISTS idx_gold_nba_ppm_player ON gold.nba_player_ppm_model (player_id);

-- Ensure selected features exist on legacy wide tables.
ALTER TABLE gold.nba_player_ppm_model ADD COLUMN IF NOT EXISTS player_pts_per_min_expanding_mean NUMERIC;
ALTER TABLE gold.nba_player_ppm_model ADD COLUMN IF NOT EXISTS player_pts_per_min_roll5_mean NUMERIC;
ALTER TABLE gold.nba_player_ppm_model ADD COLUMN IF NOT EXISTS player_pts_per_min_roll10_mean NUMERIC;
ALTER TABLE gold.nba_player_ppm_model ADD COLUMN IF NOT EXISTS player_usg_pct_season_mean NUMERIC;
ALTER TABLE gold.nba_player_ppm_model ADD COLUMN IF NOT EXISTS player_fga_season_mean NUMERIC;
ALTER TABLE gold.nba_player_ppm_model ADD COLUMN IF NOT EXISTS player_fga_roll5_mean NUMERIC;
ALTER TABLE gold.nba_player_ppm_model ADD COLUMN IF NOT EXISTS player_ft_pct_season_mean NUMERIC;
ALTER TABLE gold.nba_player_ppm_model ADD COLUMN IF NOT EXISTS player_fg3_pct_roll10_mean NUMERIC;
ALTER TABLE gold.nba_player_ppm_model ADD COLUMN IF NOT EXISTS player_efg_pct_roll10_mean NUMERIC;
ALTER TABLE gold.nba_player_ppm_model ADD COLUMN IF NOT EXISTS player_min_season_mean NUMERIC;
ALTER TABLE gold.nba_player_ppm_model ADD COLUMN IF NOT EXISTS player_min_roll10_mean NUMERIC;
ALTER TABLE gold.nba_player_ppm_model ADD COLUMN IF NOT EXISTS min_trend_5v20 NUMERIC;
ALTER TABLE gold.nba_player_ppm_model ADD COLUMN IF NOT EXISTS ts_trend_5v20 NUMERIC;
ALTER TABLE gold.nba_player_ppm_model ADD COLUMN IF NOT EXISTS opp_team_team_def_rating_roll10_mean NUMERIC;
ALTER TABLE gold.nba_player_ppm_model ADD COLUMN IF NOT EXISTS days_rest NUMERIC;
ALTER TABLE gold.nba_player_ppm_model ADD COLUMN IF NOT EXISTS starting_rate_last10 NUMERIC;
