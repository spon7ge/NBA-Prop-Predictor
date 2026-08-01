# Sharp FanDuel / DraftKings odds snapshots

Date: 2026-07-31  
Status: Approved for planning

## Goal

Persist FanDuel and DraftKings WNBA player-prop main lines into Supabase every time Sharp is successfully fetched for prop picks, throttled to at most one snapshot every **30 minutes**. Prop picks UI continues to serve **live Sharp**; tables are for history / future line-move analysis.

## Decisions

| Topic | Choice |
| --- | --- |
| Tables | `odds.wnba_fanduel`, `odds.wnba_draftkings` |
| Trigger | After successful Sharp fetch in `get_today_props` |
| Throttle | Skip write if latest `scraped_at` for that book is &lt; 30 minutes old |
| Interval config | `SHARP_PROPS_SNAPSHOT_MINUTES` (default `30`) |
| Prop picks UI source | Unchanged — live Sharp for FD/DK |
| DB failure | Log; never fail the API response |
| Missing DB URL | Skip write quietly / with warn |

## Schema

Migration: `db/migrations/020_odds_wnba_fanduel_draftkings.sql`

Both tables share the same shape:

```sql
CREATE TABLE IF NOT EXISTS odds.wnba_fanduel (
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

CREATE INDEX IF NOT EXISTS odds_wnba_fanduel_league_scraped_at_idx
    ON odds.wnba_fanduel (league, scraped_at DESC);

-- odds.wnba_draftkings: identical columns / PK / index naming
```

### Snapshot semantics

- One Sharp success that passes the throttle → one shared `scraped_at` for the FanDuel batch and the DraftKings batch.
- Main lines only (`is_main_line`, `player_*` markets, side in `{over, under}`), same filters as prop picks normalization.
- Rows split by `sportsbook` into the matching table.
- Next allowed write → new `scraped_at` (full snapshot again, even if lines unchanged).

### Throttle

For each book table (or jointly — prefer **joint** gate): if `MAX(scraped_at)` for `league=wnba` on **either** table is within the last N minutes, skip both writes. Prefer checking one table or `GREATEST` of both maxes so FD and DK stay paired.

Default N = 30. Env `SHARP_PROPS_SNAPSHOT_MINUTES` overrides.

## Implementation

```
get_today_props()
  ├── fetch Sharp (existing cache ~45s)
  ├── maybe_persist_sharp_props(rows)   # new, best-effort
  │     ├── if no SUPABASE_DB_URL → return
  │     ├── if last scraped_at within N min → return
  │     ├── map rows → FD / DK DataFrames
  │     └── insert batches (same scraped_at)
  └── merge + return response (unchanged)
```

- Reuse existing Postgres / `upsert_df` patterns from PrizePicks/Underdog loaders where practical.
- Keep mapping/throttle helpers unit-testable without live Sharp or DB.

## Out of scope

- Serving prop picks FD/DK from Supabase instead of Sharp
- Line-move UI
- NBA prop picks page (schema allows `league=nba` later)
- Changing PP/UD scrape → DB path

## Testing

- Row mapping from Sharp fixture → FD/DK row dicts
- Throttle: recent `scraped_at` → no write; stale/missing → write attempted
- Persist failure swallowed (API path still returns props)
- Migration SQL reviewed

## Success criteria

- Tables exist with PKs/indexes above
- Successful prop-picks Sharp fetches write at most once per 30 minutes
- Historical snapshots queryable by `scraped_at`
- Prop picks response unchanged when DB is down or skipped
