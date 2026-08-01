# Odds schema + WNBA Prop Picks (Sharp FD/DK + Supabase PP/UD)

Date: 2026-07-31  
Status: Approved for planning

## Goal

Store PrizePicks and Underdog scraper boards in Supabase under schema `odds` as full scrape snapshots (for line-move analysis). Serve `/wnba/prop_picks` with **FanDuel + DraftKings from Sharp** and **PrizePicks + Underdog from the latest Supabase snapshot**.

## Decisions

| Topic | Choice |
| --- | --- |
| Store | Supabase schema `odds` |
| Tables | `odds.wnba_prizepicks`, `odds.wnba_underdogs` |
| Retention | Full snapshot every scrape (`scraped_at` in PK) — unchanged lines still get new rows |
| League | Shared tables; `league` column (`nba` / `wnba`) |
| Sharp books on prop picks | FanDuel + DraftKings only |
| Scraper books on prop picks | PrizePicks + Underdog from latest `scraped_at` for `league=wnba` |
| Dropped from prop picks endpoint | Sharp Underdog, BetMGM, BetRivers (for this page) |
| Scrape → DB | Scrapers (or post-scrape loader) insert batch after writing JSON |
| API scrape | Never run Playwright/HTTP scrapers inside the request path |
| Line-move analysis | Out of scope for v1 UI; schema supports later `odds.line_moves` / queries |

## Architecture

```
prizepicks_scraper / underdog_scraper
  → JSON under data/props/...
  → upsert/insert snapshot rows → odds.wnba_prizepicks / odds.wnba_underdogs
                                      (shared scraped_at per run)

GET /api/wnba/props/today
  ├── Sharp (SHARP_API_KEY): fanduel + draftkings main props
  ├── Supabase: latest odds.wnba_prizepicks WHERE league=wnba
  ├── Supabase: latest odds.wnba_underdogs  WHERE league=wnba
  └── Merge → rows keyed by (player, stat, side)
        columns: fanduel?, draftkings?, prizepicks?, underdog?
```

## Schema

### Migration

Create `db/migrations/019_odds_prizepicks_underdog.sql` (next number if 019 taken).

```sql
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
```

### Snapshot semantics

- One scrape run → one `scraped_at` timestamp applied to every row in that batch.
- Same player/stat/line on the next scrape → **new rows** (different `scraped_at`).
- Latest board: `WHERE scraped_at = (SELECT MAX(scraped_at) FROM odds.wnba_prizepicks WHERE league = $1)`.
- Line-move analysis later: compare consecutive `scraped_at` partitions for the same natural key excluding `scraped_at` / `line_score` as needed.

### Mapping from scraper JSON

**PrizePicks** (`projections[]`):

| JSON | Column |
| --- | --- |
| `league` | `league` (lowercased) |
| `player` | `player_name` |
| `stat_type` | `stat_type` |
| `line_score` | `line_score` |
| `odds_type` | `odds_type` |
| `updated_at` | `line_updated_at` |
| export `fetched_at` / run time | `scraped_at` |

**Underdog** (`picks[]`):

| JSON | Column |
| --- | --- |
| sport/league from scrape | `league` (lowercased) |
| `full_name` | `player_name` |
| `stat_name` | `stat_name` |
| `stat_value` | `line_score` |
| `choice` | `side` |
| `american_price` | `american_price` (parsed int) |
| `payout_multiplier` | `payout_multiplier` |
| `updated_at` | `line_updated_at` |
| export `fetched_at` / run time | `scraped_at` |

## Loader / scraper wiring

- Prefer: after successful scrape save, insert the batch via existing `upsert_df` / Postgres path (`SUPABASE_DB_URL`), schema `odds`.
- Idempotent within a scrape: PK prevents duplicate identical rows for the same `scraped_at`.
- Failure to write Supabase should log loudly but **not** delete the local JSON file.
- Optional CLI flag or env `PRIZEPICKS_SKIP_DB=1` / `UNDERDOG_SKIP_DB=1` to skip DB insert during local debugging.

## Prop picks API / UI

### Backend

- Narrow Sharp fetch to `fanduel`, `draftkings` only in `sharp_props.py` (or prop-picks caller).
- New small readers: `get_latest_prizepicks_props(league)`, `get_latest_underdog_props(league)` using Supabase/Postgres.
- Extend `WnbaPropLine` schema:
  - Keep `fanduel`, `draftkings`
  - Add `prizepicks: { line, odds_american? } | null`
  - Add `underdog: { line, odds_american? } | null` sourced from **Supabase**, not Sharp
  - Remove `betmgm`, `betrivers` from this response (or leave null unused — prefer remove to avoid confusion)
- Merge grain remains: one row per player + normalized stat + side (`over`/`under`).
  - PrizePicks often has no explicit over/under (DFS more/less); map as both sides sharing the same line **or** emit a single conventional side — **v1: treat as `over` and `under` rows both carrying the same `line_score` when building the table grain to match FD/DK**, unless product prefers one side only. Prefer: **duplicate line onto both over and under cells for PrizePicks** so filters still work; odds_american null for PP.
  - Underdog already has `side` + `american_price`.
- Stat label normalization: map PP `stat_type` / UD `stat_name` into the same human labels used for Sharp where possible; unmatched stats still show with raw label.
- Cache: keep ~45s in-process for the merged response; include scraper `scraped_at` in `as_of` or a small `sources` metadata field if easy.

### Frontend

- `BOOK_COLUMNS`: FanDuel, DraftKings, PrizePicks, Underdog.
- Caption: `Odds by FanDuel, DraftKings, PrizePicks & Underdog`.
- Update types/tests accordingly.
- Filters unchanged (stat / side / team).

## Out of scope (v1)

- Line-move UI or `odds.line_moves` table
- Live scrape inside API
- BetMGM / BetRivers on prop picks
- Changing matchup odds Sharp path
- NBA prop picks page (schema supports NBA; UI remains WNBA)

## Testing

- SQL migration reviewed / applied in Supabase
- Unit tests: JSON → row mapping; american price parse; latest-snapshot query helper (mocked DB or fixtures)
- Backend tests: merge Sharp FD/DK + mock PP/UD snapshots; schema omits betmgm/betrivers
- Frontend tests: book columns + API types

## Success criteria

- `odds.wnba_prizepicks` and `odds.wnba_underdogs` exist with PKs/indexes above
- Successful WNBA scrapes land a new `scraped_at` batch in Supabase
- `/wnba/prop_picks` shows FD/DK (Sharp) and PP/UD (latest Supabase) without calling scrapers on request
- Historical snapshots remain queryable for future line-move analysis
