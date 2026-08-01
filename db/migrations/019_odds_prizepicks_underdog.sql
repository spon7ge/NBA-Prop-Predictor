-- 019_odds_prizepicks_underdog.sql
-- PrizePicks and Underdog scraper snapshot tables (schema odds).
-- Tables: odds.wnba_prizepicks, odds.wnba_underdogs
-- Full snapshot per scrape; scraped_at in PK for line-move analysis.

CREATE SCHEMA IF NOT EXISTS odds;

CREATE TABLE IF NOT EXISTS odds.wnba_prizepicks (
    league           TEXT        NOT NULL,  -- nba | wnba
    player_name      TEXT        NOT NULL,
    stat_type        TEXT        NOT NULL,
    line_score       NUMERIC     NOT NULL,
    odds_type        TEXT        NOT NULL DEFAULT 'standard',
    line_updated_at  TIMESTAMPTZ,
    scraped_at       TIMESTAMPTZ NOT NULL,
    fetched_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (league, player_name, stat_type, odds_type, line_score, scraped_at)
);

CREATE INDEX IF NOT EXISTS odds_wnba_prizepicks_league_scraped_at_idx
    ON odds.wnba_prizepicks (league, scraped_at DESC);

CREATE TABLE IF NOT EXISTS odds.wnba_underdogs (
    league              TEXT        NOT NULL,  -- nba | wnba
    player_name         TEXT        NOT NULL,
    stat_name           TEXT        NOT NULL,
    line_score          NUMERIC     NOT NULL,
    side                TEXT        NOT NULL,  -- over | under
    american_price      INTEGER,
    payout_multiplier   NUMERIC,
    line_updated_at     TIMESTAMPTZ,
    scraped_at          TIMESTAMPTZ NOT NULL,
    fetched_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (league, player_name, stat_name, side, line_score, scraped_at)
);

CREATE INDEX IF NOT EXISTS odds_wnba_underdogs_league_scraped_at_idx
    ON odds.wnba_underdogs (league, scraped_at DESC);
