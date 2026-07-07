-- 006_bronze_schema.sql
-- Bronze layer: dbt-managed views/tables for light cleanup on raw.* sources.
-- Run once in Supabase SQL Editor before first `dbt run`.

CREATE SCHEMA IF NOT EXISTS bronze;

-- Optional landing table for Odds-API game events (team-line scrapers).
-- Populated by upload scripts; bronze_games also reads NBA game logs from raw.team_base.
CREATE TABLE IF NOT EXISTS raw.games (
    event_id       BIGINT      NOT NULL,
    home_team      TEXT,
    away_team      TEXT,
    commence_time  TEXT,
    fetched_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (event_id)
);
