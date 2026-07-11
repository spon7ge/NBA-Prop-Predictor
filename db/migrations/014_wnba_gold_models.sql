-- Gold layer: WNBA model frames (parallel to NBA gold tables).
-- Source: src.pipeline.gold.build_*_gold_from_silver(..., league="wnba")
-- Upload: src.utils.db.upsert_prop_gold(prop, df, league="wnba")
--
-- MIN columns follow src.pipeline.features.wnba_min_features.WNBA_MIN_FEATURES
-- (not a copy of the NBA min schema).

CREATE SCHEMA IF NOT EXISTS gold;

-- ── MIN ───────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS gold.wnba_player_min_model (
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
    usg_pct_lag1                 NUMERIC,
    usg_pct_lag2                 NUMERIC,
    ast_pct_lag1                 NUMERIC,
    pie_lag1                     NUMERIC,
    games_played_last_7_days     SMALLINT,
    games_played_last_14_days    SMALLINT,
    min_sum_last_7_days          NUMERIC,
    days_rest                    NUMERIC,

    PRIMARY KEY (game_id, player_id)
);

CREATE INDEX IF NOT EXISTS idx_gold_wnba_min_season ON gold.wnba_player_min_model (season);
CREATE INDEX IF NOT EXISTS idx_gold_wnba_min_date ON gold.wnba_player_min_model (game_date);
CREATE INDEX IF NOT EXISTS idx_gold_wnba_min_player ON gold.wnba_player_min_model (player_id);

-- ── PPM ───────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS gold.wnba_player_ppm_model (
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

CREATE INDEX IF NOT EXISTS idx_gold_wnba_ppm_season ON gold.wnba_player_ppm_model (season);
CREATE INDEX IF NOT EXISTS idx_gold_wnba_ppm_date ON gold.wnba_player_ppm_model (game_date);
CREATE INDEX IF NOT EXISTS idx_gold_wnba_ppm_player ON gold.wnba_player_ppm_model (player_id);

-- ── APM ───────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS gold.wnba_player_apm_model (
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

CREATE INDEX IF NOT EXISTS idx_gold_wnba_apm_season ON gold.wnba_player_apm_model (season);
CREATE INDEX IF NOT EXISTS idx_gold_wnba_apm_date ON gold.wnba_player_apm_model (game_date);
CREATE INDEX IF NOT EXISTS idx_gold_wnba_apm_player ON gold.wnba_player_apm_model (player_id);

-- ── RPM ───────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS gold.wnba_player_rpm_model (
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

CREATE INDEX IF NOT EXISTS idx_gold_wnba_rpm_season ON gold.wnba_player_rpm_model (season);
CREATE INDEX IF NOT EXISTS idx_gold_wnba_rpm_date ON gold.wnba_player_rpm_model (game_date);
CREATE INDEX IF NOT EXISTS idx_gold_wnba_rpm_player ON gold.wnba_player_rpm_model (player_id);
