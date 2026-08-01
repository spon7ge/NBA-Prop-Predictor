-- 021_odds_wnba_parlay_books.sql
-- ParlayAPI Caesars / BetMGM / Pinnacle / bet365 prop snapshot tables (schema odds).
-- Same shape as odds.wnba_fanduel / odds.wnba_draftkings (migration 020).

CREATE SCHEMA IF NOT EXISTS odds;

CREATE TABLE IF NOT EXISTS odds.wnba_caesars (
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

CREATE INDEX IF NOT EXISTS odds_wnba_caesars_league_scraped_at_idx
    ON odds.wnba_caesars (league, scraped_at DESC);

CREATE TABLE IF NOT EXISTS odds.wnba_betmgm (
    league           TEXT        NOT NULL,
    player_name      TEXT        NOT NULL,
    market_type      TEXT        NOT NULL,
    stat_category    TEXT,
    side             TEXT        NOT NULL,
    line_score       NUMERIC     NOT NULL,
    american_price   INTEGER     NOT NULL,
    scraped_at       TIMESTAMPTZ NOT NULL,
    fetched_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (league, player_name, market_type, side, line_score, scraped_at)
);

CREATE INDEX IF NOT EXISTS odds_wnba_betmgm_league_scraped_at_idx
    ON odds.wnba_betmgm (league, scraped_at DESC);

CREATE TABLE IF NOT EXISTS odds.wnba_pinnacle (
    league           TEXT        NOT NULL,
    player_name      TEXT        NOT NULL,
    market_type      TEXT        NOT NULL,
    stat_category    TEXT,
    side             TEXT        NOT NULL,
    line_score       NUMERIC     NOT NULL,
    american_price   INTEGER     NOT NULL,
    scraped_at       TIMESTAMPTZ NOT NULL,
    fetched_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (league, player_name, market_type, side, line_score, scraped_at)
);

CREATE INDEX IF NOT EXISTS odds_wnba_pinnacle_league_scraped_at_idx
    ON odds.wnba_pinnacle (league, scraped_at DESC);

CREATE TABLE IF NOT EXISTS odds.wnba_bet365 (
    league           TEXT        NOT NULL,
    player_name      TEXT        NOT NULL,
    market_type      TEXT        NOT NULL,
    stat_category    TEXT,
    side             TEXT        NOT NULL,
    line_score       NUMERIC     NOT NULL,
    american_price   INTEGER     NOT NULL,
    scraped_at       TIMESTAMPTZ NOT NULL,
    fetched_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (league, player_name, market_type, side, line_score, scraped_at)
);

CREATE INDEX IF NOT EXISTS odds_wnba_bet365_league_scraped_at_idx
    ON odds.wnba_bet365 (league, scraped_at DESC);
