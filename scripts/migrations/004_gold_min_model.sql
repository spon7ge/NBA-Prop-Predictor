-- Gold layer: MIN quantile model training / inference frame.
-- Source: src.utils.gold.build_min_gold_dataset
-- Upload: src.utils.db.upsert_gold

CREATE SCHEMA IF NOT EXISTS gold;

CREATE TABLE IF NOT EXISTS gold.player_min_model (
    game_id              TEXT        NOT NULL,
    player_id            BIGINT      NOT NULL,
    built_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    season               TEXT,
    player_name          TEXT,
    game_date            DATE,
    min                  NUMERIC,
    starting             SMALLINT,
    min_10_ewm           NUMERIC,
    min_season_mean      NUMERIC,
    starter_roll10_pct   NUMERIC,
    consec_starts        NUMERIC,
    min_rate_of_change   NUMERIC,
    team_min_rank_l10    NUMERIC,
    team_usg_rank_l10    NUMERIC,
    min_p10_l10          NUMERIC,
    min_p90_l10          NUMERIC,
    min_std_l10          NUMERIC,
    min_season_std       NUMERIC,
    spd_10_ewm           NUMERIC,
    top_player_active    SMALLINT,
    active_stars_count   SMALLINT,

    PRIMARY KEY (game_id, player_id)
);

CREATE INDEX IF NOT EXISTS idx_gold_min_season ON gold.player_min_model (season);
CREATE INDEX IF NOT EXISTS idx_gold_min_date ON gold.player_min_model (game_date);
CREATE INDEX IF NOT EXISTS idx_gold_min_player ON gold.player_min_model (player_id);
