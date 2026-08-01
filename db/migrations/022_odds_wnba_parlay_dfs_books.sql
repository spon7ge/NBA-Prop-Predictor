-- 022_odds_wnba_parlay_dfs_books.sql
-- ParlayAPI PrizePicks / Underdog / Betr / Novig / Sleeper / Pick6 prop snapshot
-- tables (schema odds). Same shape as odds.wnba_fanduel (migration 020).
--
-- Note: odds.wnba_prizepicks and odds.wnba_underdogs (migration 019) remain the
-- scraper-shaped tables. Parlay snapshots use *_parlay names for those two books
-- to avoid colliding with the different column layouts.

CREATE SCHEMA IF NOT EXISTS odds;

CREATE TABLE IF NOT EXISTS odds.wnba_prizepicks_parlay (
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

CREATE INDEX IF NOT EXISTS odds_wnba_prizepicks_parlay_league_scraped_at_idx
    ON odds.wnba_prizepicks_parlay (league, scraped_at DESC);

CREATE TABLE IF NOT EXISTS odds.wnba_underdog_parlay (
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

CREATE INDEX IF NOT EXISTS odds_wnba_underdog_parlay_league_scraped_at_idx
    ON odds.wnba_underdog_parlay (league, scraped_at DESC);

CREATE TABLE IF NOT EXISTS odds.wnba_betr (
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

CREATE INDEX IF NOT EXISTS odds_wnba_betr_league_scraped_at_idx
    ON odds.wnba_betr (league, scraped_at DESC);

CREATE TABLE IF NOT EXISTS odds.wnba_novig (
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

CREATE INDEX IF NOT EXISTS odds_wnba_novig_league_scraped_at_idx
    ON odds.wnba_novig (league, scraped_at DESC);

CREATE TABLE IF NOT EXISTS odds.wnba_sleeper (
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

CREATE INDEX IF NOT EXISTS odds_wnba_sleeper_league_scraped_at_idx
    ON odds.wnba_sleeper (league, scraped_at DESC);

CREATE TABLE IF NOT EXISTS odds.wnba_pick6 (
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

CREATE INDEX IF NOT EXISTS odds_wnba_pick6_league_scraped_at_idx
    ON odds.wnba_pick6 (league, scraped_at DESC);
