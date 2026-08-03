-- 023_odds_wnba_betrivers.sql
-- ParlayAPI BetRivers prop snapshot table (schema odds).
-- Same shape as odds.wnba_betmgm / other Parlay sportsbook tables (migration 021).

CREATE SCHEMA IF NOT EXISTS odds;

CREATE TABLE IF NOT EXISTS odds.wnba_betrivers (
    league           TEXT        NOT NULL,  -- nba | wnba
    player_name      TEXT        NOT NULL,
    market_type      TEXT        NOT NULL,
    stat_category    TEXT,
    side             TEXT        NOT NULL,  -- over | under
    line_score       NUMERIC     NOT NULL,
    american_price   INTEGER     NOT NULL,
    scraped_at       TIMESTAMPTZ NOT NULL,
    fetched_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (league, player_name, market_type, side, line_score, scraped_at)
);

CREATE INDEX IF NOT EXISTS odds_wnba_betrivers_league_scraped_at_idx
    ON odds.wnba_betrivers (league, scraped_at DESC);
