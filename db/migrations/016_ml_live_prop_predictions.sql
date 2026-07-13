-- Migration 016: ml.nba_live_prop_predictions + ml.wnba_live_prop_predictions
-- Separate tables per league, populated by scripts/run_live_props.py.
-- Served by GET /api/live-props?league=nba|wnba.

-- ── NBA ───────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ml.nba_live_prop_predictions (
    id                   BIGSERIAL    PRIMARY KEY,
    run_at               TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    game_date            DATE         NOT NULL,

    -- player / game identity
    player_name          TEXT         NOT NULL,
    team_abbr            TEXT,
    opponent_abbr        TEXT,
    is_home              BOOLEAN,
    market               TEXT         NOT NULL,   -- PTS | AST | REB
    bookmaker            TEXT         NOT NULL,   -- PrizePicks | Underdog | Betr | etc.
    line                 REAL,

    -- model outputs
    stat_q10             REAL,
    stat_q50             REAL,
    stat_q90             REAL,
    min_q10              REAL,
    min_q50              REAL,
    min_q90              REAL,
    p_over               REAL,
    p_under              REAL,

    -- game context
    opp_def_rating       REAL,
    opp_def_rating_rank  INT,
    opp_pace             REAL,
    team_def_rating      REAL,
    team_pace            REAL,
    game_total           REAL,
    team_spread          REAL,

    -- recent form vs the line
    over_l5              REAL,
    over_l10             REAL,
    over_l15             REAL,

    -- vs tonight's opponent (matchup history)
    vs_opp_n_games       INT,
    vs_opp_avg_stat      REAL,
    vs_opp_over_rate     REAL,

    UNIQUE (run_at, game_date, player_name, market, bookmaker)
);

CREATE INDEX IF NOT EXISTS idx_nba_live_prop_date
    ON ml.nba_live_prop_predictions (game_date);
CREATE INDEX IF NOT EXISTS idx_nba_live_prop_run_at
    ON ml.nba_live_prop_predictions (run_at DESC);

-- ── WNBA ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ml.wnba_live_prop_predictions (
    id                   BIGSERIAL    PRIMARY KEY,
    run_at               TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    game_date            DATE         NOT NULL,

    -- player / game identity
    player_name          TEXT         NOT NULL,
    team_abbr            TEXT,
    opponent_abbr        TEXT,
    is_home              BOOLEAN,
    market               TEXT         NOT NULL,
    bookmaker            TEXT         NOT NULL,
    line                 REAL,

    -- model outputs
    stat_q10             REAL,
    stat_q50             REAL,
    stat_q90             REAL,
    min_q10              REAL,
    min_q50              REAL,
    min_q90              REAL,
    p_over               REAL,
    p_under              REAL,

    -- game context
    opp_def_rating       REAL,
    opp_def_rating_rank  INT,
    opp_pace             REAL,
    team_def_rating      REAL,
    team_pace            REAL,
    game_total           REAL,
    team_spread          REAL,

    -- recent form vs the line
    over_l5              REAL,
    over_l10             REAL,
    over_l15             REAL,

    -- vs tonight's opponent (matchup history)
    vs_opp_n_games       INT,
    vs_opp_avg_stat      REAL,
    vs_opp_over_rate     REAL,

    UNIQUE (run_at, game_date, player_name, market, bookmaker)
);

CREATE INDEX IF NOT EXISTS idx_wnba_live_prop_date
    ON ml.wnba_live_prop_predictions (game_date);
CREATE INDEX IF NOT EXISTS idx_wnba_live_prop_run_at
    ON ml.wnba_live_prop_predictions (run_at DESC);
