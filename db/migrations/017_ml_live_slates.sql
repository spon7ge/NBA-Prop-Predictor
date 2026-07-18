-- Migration 017: ml.nba_live_slates + ml.wnba_live_slates
-- Populated by scripts/run_live_slates.py (enrich_dfs_picks → greedy parlays).
-- Served by GET /api/live-slates?league=nba|wnba.

-- ── NBA ───────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ml.nba_live_slates (
    id         BIGSERIAL    PRIMARY KEY,
    run_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    game_date  DATE         NOT NULL,
    bookmaker  TEXT         NOT NULL,  -- prizepicks | underdog | draftkings | betr
    n_legs     INT          NOT NULL,  -- 2 | 3 | 5 | 6
    parlays    JSONB        NOT NULL,  -- array of FlatParlayRow (LEGS, EV, …)

    UNIQUE (run_at, game_date, bookmaker, n_legs)
);

CREATE INDEX IF NOT EXISTS idx_nba_live_slates_date
    ON ml.nba_live_slates (game_date);
CREATE INDEX IF NOT EXISTS idx_nba_live_slates_run_at
    ON ml.nba_live_slates (run_at DESC);

-- ── WNBA ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ml.wnba_live_slates (
    id         BIGSERIAL    PRIMARY KEY,
    run_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    game_date  DATE         NOT NULL,
    bookmaker  TEXT         NOT NULL,
    n_legs     INT          NOT NULL,
    parlays    JSONB        NOT NULL,

    UNIQUE (run_at, game_date, bookmaker, n_legs)
);

CREATE INDEX IF NOT EXISTS idx_wnba_live_slates_date
    ON ml.wnba_live_slates (game_date);
CREATE INDEX IF NOT EXISTS idx_wnba_live_slates_run_at
    ON ml.wnba_live_slates (run_at DESC);
