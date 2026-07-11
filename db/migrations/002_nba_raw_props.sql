-- 013_wnba_raw_props.sql
-- WNBA prop-line tables populated by WNBAPropFinder.
--
--   raw.wnba_props_dfs  ← WNBAPropFinder(region='us_dfs')
--   raw.wnba_props_us   ← WNBAPropFinder(region='us' / 'us,eu')
--
-- Column names are lowercase snake_case to match upsert_df() normalization.
-- PK includes data_pulled_at (one row per scrape) and line (books often
-- publish multiple alternate lines for the same player/market).
--
-- If an earlier draft was applied with a different PK, drop first:
--   DROP TABLE IF EXISTS raw.wnba_props_dfs;
--   DROP TABLE IF EXISTS raw.wnba_props_us;

CREATE SCHEMA IF NOT EXISTS raw;

CREATE TABLE IF NOT EXISTS raw.nba_props_dfs (
    bookmaker       TEXT        NOT NULL,
    category        TEXT        NOT NULL,
    name            TEXT        NOT NULL,
    over_under      TEXT        NOT NULL,
    line            NUMERIC     NOT NULL,
    odds            NUMERIC,
    commence_time   TEXT        NOT NULL,
    last_update     TEXT,
    data_pulled_at  TEXT        NOT NULL,
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (bookmaker, category, name, over_under, commence_time, data_pulled_at, line)
);

CREATE TABLE IF NOT EXISTS raw.nba_props_us (
    bookmaker       TEXT        NOT NULL,
    category        TEXT        NOT NULL,
    name            TEXT        NOT NULL,
    over_under      TEXT        NOT NULL,
    line            NUMERIC     NOT NULL,
    odds            NUMERIC,
    commence_time   TEXT        NOT NULL,
    last_update     TEXT,
    data_pulled_at  TEXT        NOT NULL,
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (bookmaker, category, name, over_under, commence_time, data_pulled_at, line)
);
