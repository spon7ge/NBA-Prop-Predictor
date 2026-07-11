-- Gold layer: NBA PPM quantile model training / inference frame.
-- Source: src.utils.gold.build_ppm_gold_from_silver(..., league="nba")
-- Upload: src.utils.db.upsert_prop_gold("ppm", df, league="nba")

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
    game_id                                  TEXT        NOT NULL,
    player_id                                BIGINT      NOT NULL,
    built_at                                 TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    season                                   TEXT,
    player_name                              TEXT,
    game_date                                DATE,
    pts_per_min                              NUMERIC,
    min                                      NUMERIC,
    starting                                 SMALLINT,
    pts_per_min_season_avg                   NUMERIC,
    pts_per_min_x_opp_pts_allowed            NUMERIC,
    cfga_per_min_x_opp_fg_pct_allowed        NUMERIC,
    fga_per_min_x_opp_def_rating             NUMERIC,
    "3pa_per_min_x_opp_team_fg3a_allowed"    NUMERIC,
    fta_per_min_x_opp_fta_allowed            NUMERIC,
    ppm_season_std                           NUMERIC,
    team_pts_per_min_rank_l10                NUMERIC,
    team_usg_rank_l10                        NUMERIC,
    team_min_rank_l10                        NUMERIC,
    ppm_p10_l10                              NUMERIC,
    ppm_p90_l10                              NUMERIC,

    PRIMARY KEY (game_id, player_id)
);

CREATE INDEX IF NOT EXISTS idx_gold_nba_ppm_season ON gold.nba_player_ppm_model (season);
CREATE INDEX IF NOT EXISTS idx_gold_nba_ppm_date ON gold.nba_player_ppm_model (game_date);
CREATE INDEX IF NOT EXISTS idx_gold_nba_ppm_player ON gold.nba_player_ppm_model (player_id);
