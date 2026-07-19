-- Migration 018: ml.nba_live_prop_grades + ml.wnba_live_prop_grades
-- Graded live props: latest prediction run per slate joined to silver box scores.
-- Populated by scripts/grade_live_props.py (Airflow after silver).
-- Served by GET /api/performance?league=nba|wnba.

-- ── NBA ───────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ml.nba_live_prop_grades (
    id              BIGSERIAL    PRIMARY KEY,
    graded_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    run_at          TIMESTAMPTZ  NOT NULL,
    game_date       DATE         NOT NULL,

    player_name     TEXT         NOT NULL,
    team_abbr       TEXT,
    opponent_abbr   TEXT,
    market          TEXT         NOT NULL,   -- PTS | AST | REB
    bookmaker       TEXT         NOT NULL,
    line            REAL,
    side            TEXT         NOT NULL,   -- over | under

    -- snapshot of model lean at grade time
    stat_q10        REAL,
    stat_q50        REAL,
    p_over          REAL,
    p_under         REAL,

    -- outcomes
    actual_stat     REAL,
    actual_min      REAL,
    hit             BOOLEAN      NOT NULL,
    miss_reason     TEXT         NOT NULL,   -- dnp | blowout | … | clean_hit | squeaker
    abs_error       REAL,                    -- |actual − line|; null for DNP

    UNIQUE (run_at, game_date, player_name, market, bookmaker)
);

CREATE INDEX IF NOT EXISTS idx_nba_live_prop_grades_date
    ON ml.nba_live_prop_grades (game_date);
CREATE INDEX IF NOT EXISTS idx_nba_live_prop_grades_run_at
    ON ml.nba_live_prop_grades (run_at DESC);
CREATE INDEX IF NOT EXISTS idx_nba_live_prop_grades_miss
    ON ml.nba_live_prop_grades (miss_reason);

-- ── WNBA ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ml.wnba_live_prop_grades (
    id              BIGSERIAL    PRIMARY KEY,
    graded_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    run_at          TIMESTAMPTZ  NOT NULL,
    game_date       DATE         NOT NULL,

    player_name     TEXT         NOT NULL,
    team_abbr       TEXT,
    opponent_abbr   TEXT,
    market          TEXT         NOT NULL,
    bookmaker       TEXT         NOT NULL,
    line            REAL,
    side            TEXT         NOT NULL,

    stat_q10        REAL,
    stat_q50        REAL,
    p_over          REAL,
    p_under         REAL,

    actual_stat     REAL,
    actual_min      REAL,
    hit             BOOLEAN      NOT NULL,
    miss_reason     TEXT         NOT NULL,
    abs_error       REAL,

    UNIQUE (run_at, game_date, player_name, market, bookmaker)
);

CREATE INDEX IF NOT EXISTS idx_wnba_live_prop_grades_date
    ON ml.wnba_live_prop_grades (game_date);
CREATE INDEX IF NOT EXISTS idx_wnba_live_prop_grades_run_at
    ON ml.wnba_live_prop_grades (run_at DESC);
CREATE INDEX IF NOT EXISTS idx_wnba_live_prop_grades_miss
    ON ml.wnba_live_prop_grades (miss_reason);
