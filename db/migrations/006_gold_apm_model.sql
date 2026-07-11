-- Gold layer: NBA APM quantile model training / inference frame.
-- Source: src.utils.gold.build_apm_gold_from_silver(..., league="nba")
-- Upload: src.utils.db.upsert_prop_gold("apm", df, league="nba")

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
    game_id                      TEXT        NOT NULL,
    player_id                    BIGINT      NOT NULL,
    built_at                     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    season                       TEXT,
    player_name                  TEXT,
    game_date                    DATE,
    ast_per_min                  NUMERIC,
    min                          NUMERIC,
    starting                     SMALLINT,
    ast_per_min_season_avg       NUMERIC,
    team_ast_per_min_rank_l10    NUMERIC,
    opp_team_ast_allowed         NUMERIC,
    pass_per_min_5_ewm           NUMERIC,
    position_encoded             SMALLINT,
    ast_per_min_std_season       NUMERIC,
    apm_trend                    NUMERIC,

    PRIMARY KEY (game_id, player_id)
);

CREATE INDEX IF NOT EXISTS idx_gold_nba_apm_season ON gold.nba_player_apm_model (season);
CREATE INDEX IF NOT EXISTS idx_gold_nba_apm_date ON gold.nba_player_apm_model (game_date);
CREATE INDEX IF NOT EXISTS idx_gold_nba_apm_player ON gold.nba_player_apm_model (player_id);
