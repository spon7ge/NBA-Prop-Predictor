-- 002_raw_props.sql
-- Creates the two raw prop-line tables populated by NBAPropFinder.
-- Run once: Supabase Dashboard → SQL Editor → paste and run.
--
--   raw.props_dfs  ← NBAPropFinder(region='us_dfs')   — DFS books (PrizePicks, Underdog, …)
--   raw.props_us   ← NBAPropFinder(region='us' / 'us,eu') — US sportsbooks (DK, FD, MGM, …)
--
-- Natural key: a single player prop from a single bookmaker for a single game.
-- On conflict the line/odds/last_update columns are updated so you always have
-- the most recent market price — safe to run the scraper multiple times per day.

CREATE SCHEMA IF NOT EXISTS raw;

-- ── raw.props_dfs ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS raw.props_dfs (
    -- composite primary key
    bookmaker       TEXT        NOT NULL,
    category        TEXT        NOT NULL,  -- market key: player_points, player_rebounds, …
    name            TEXT        NOT NULL,  -- player name as returned by the API
    over_under      TEXT        NOT NULL,  -- 'Over' or 'Under'
    commence_time   TEXT        NOT NULL,  -- game date (YYYY-MM-DD)
    -- mutable market data — updated on every upsert
    line            NUMERIC,
    odds            NUMERIC,
    last_update     TEXT,
    data_pulled_at  TEXT,
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (bookmaker, category, name, over_under, commence_time)
);

-- ── raw.props_us ──────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS raw.props_us (
    bookmaker       TEXT        NOT NULL,
    category        TEXT        NOT NULL,
    name            TEXT        NOT NULL,
    over_under      TEXT        NOT NULL,
    commence_time   TEXT        NOT NULL,
    line            NUMERIC,
    odds            NUMERIC,
    last_update     TEXT,
    data_pulled_at  TEXT,
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (bookmaker, category, name, over_under, commence_time)
);
