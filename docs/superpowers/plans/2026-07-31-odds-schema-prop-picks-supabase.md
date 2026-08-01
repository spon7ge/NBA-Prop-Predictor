# Odds Schema + Prop Picks Supabase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `odds.wnba_prizepicks` / `odds.wnba_underdogs` snapshot tables, load scraper boards into Supabase, and serve `/wnba/prop_picks` with Sharp FD/DK plus latest Supabase PrizePicks/Underdog.

**Architecture:** Scrapers write JSON then insert a full `scraped_at` batch via `upsert_df`. Prop picks API fetches Sharp for `fanduel`/`draftkings` only, reads latest Supabase snapshots for PP/UD, merges into one row grain. Frontend book columns become FD, DK, PrizePicks, Underdog.

**Tech Stack:** Postgres/Supabase migrations, pandas + `src.utils.db.upsert_df`, FastAPI/Pydantic, React/Vitest

## Global Constraints

- Full snapshot every scrape (`scraped_at` in PK); unchanged lines still insert new rows
- Shared tables with `league` in (`nba`, `wnba`)
- Never run scrapers inside the API request path
- Sharp on prop picks: FanDuel + DraftKings only
- Scraper books: PrizePicks + Underdog from latest `scraped_at` where `league=wnba`
- Remove BetMGM / BetRivers from prop picks schema/UI
- PrizePicks has no O/U: v1 mirrors the same line onto both `over` and `under` rows; `odds_american` null for PP
- Do not commit unless the user explicitly asks
- Spec: `docs/superpowers/specs/2026-07-31-odds-schema-prop-picks-supabase-design.md`

---

## File Structure

| File | Responsibility |
| --- | --- |
| `db/migrations/019_odds_prizepicks_underdog.sql` | Create `odds` schema + tables + indexes |
| `src/odds/snapshot_rows.py` | Pure mappers: scraper JSON → DataFrame rows; american price parse |
| `src/odds/load_snapshots.py` | Insert snapshot batches via `upsert_df` |
| `src/scrapers/prizepicks_scraper.py` | After save, call loader (unless `PRIZEPICKS_SKIP_DB`) |
| `src/scrapers/underdog_scraper.py` | After save, call loader (unless `UNDERDOG_SKIP_DB`) |
| `backend/app/schemas/wnba_props.py` | Books: fanduel, draftkings, prizepicks, underdog |
| `backend/app/services/odds_snapshots.py` | Read latest PP/UD boards from Supabase/Postgres |
| `backend/app/services/sharp_props.py` | Sharp FD/DK only; merge Supabase PP/UD |
| `frontend/src/lib/api.ts` + PropPicksTable + tests | Types + columns |

---

### Task 1: Migration

**Files:**
- Create: `db/migrations/019_odds_prizepicks_underdog.sql`

**Interfaces:**
- Produces: `odds.wnba_prizepicks`, `odds.wnba_underdogs` with PKs/indexes from the spec

- [ ] **Step 1: Write migration SQL** exactly as in the spec (schema, two tables, two indexes)

- [ ] **Step 2: Apply in Supabase SQL editor** (or `psql "$SUPABASE_DB_URL" -f db/migrations/019_odds_prizepicks_underdog.sql`)

Expected: tables exist; `\d odds.wnba_prizepicks` shows PK including `scraped_at`

---

### Task 2: Snapshot row mappers (TDD)

**Files:**
- Create: `src/odds/__init__.py` (empty or package docstring)
- Create: `src/odds/snapshot_rows.py`
- Create: `tests/odds/test_snapshot_rows.py`

**Interfaces:**
- Produces:
  - `parse_american_price(raw: str | int | None) -> int | None`
  - `prizepicks_projections_to_rows(projections: list[dict], *, league: str, scraped_at: datetime) -> list[dict]`
  - `underdog_picks_to_rows(picks: list[dict], *, league: str, scraped_at: datetime) -> list[dict]`
- Row keys must match table columns (snake_case)

- [ ] **Step 1: Write failing tests**

```python
from datetime import datetime, timezone
from src.odds.snapshot_rows import (
    parse_american_price,
    prizepicks_projections_to_rows,
    underdog_picks_to_rows,
)

def test_parse_american_price():
    assert parse_american_price("+477") == 477
    assert parse_american_price("-130") == -130
    assert parse_american_price(None) is None

def test_prizepicks_mapper():
    scraped = datetime(2026, 8, 1, tzinfo=timezone.utc)
    rows = prizepicks_projections_to_rows(
        [{"player": "A'ja Wilson", "stat_type": "Points", "line_score": 22.5,
          "odds_type": "standard", "updated_at": "2026-07-31T12:00:00-04:00", "league": "WNBA"}],
        league="wnba",
        scraped_at=scraped,
    )
    assert rows[0]["player_name"] == "A'ja Wilson"
    assert rows[0]["league"] == "wnba"
    assert rows[0]["line_score"] == 22.5
    assert rows[0]["scraped_at"] == scraped

def test_underdog_mapper():
    scraped = datetime(2026, 8, 1, tzinfo=timezone.utc)
    rows = underdog_picks_to_rows(
        [{"full_name": "Caitlin Clark", "stat_name": "points", "stat_value": "19.5",
          "choice": "over", "american_price": "-130", "payout_multiplier": "0.94",
          "updated_at": "2026-07-31T23:57:11Z"}],
        league="wnba",
        scraped_at=scraped,
    )
    assert rows[0]["side"] == "over"
    assert rows[0]["american_price"] == -130
    assert float(rows[0]["line_score"]) == 19.5
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor
source nba_model/bin/activate
python -m pytest tests/odds/test_snapshot_rows.py -v
```

- [ ] **Step 3: Implement `src/odds/snapshot_rows.py`**

Skip rows missing required fields (`player_name`, `stat_*`, `line_score`; Underdog also needs `side`). Lowercase `league`. Parse `line_updated_at` with `datetime.fromisoformat` when possible else leave `None`.

- [ ] **Step 4: Run tests — expect PASS**

---

### Task 3: Load snapshots into Supabase

**Files:**
- Create: `src/odds/load_snapshots.py`
- Create: `tests/odds/test_load_snapshots.py` (unit-test with mocked `upsert_df`)

**Interfaces:**
- Produces:
  - `load_prizepicks_snapshot(projections, *, league: str, scraped_at: datetime | None = None) -> int`
  - `load_underdog_snapshot(picks, *, league: str, scraped_at: datetime | None = None) -> int`
- Returns row count inserted; no-op `0` on empty
- Uses `upsert_df(table, df, schema="odds", conflict_cols=[...pk cols...], lineage_col="fetched_at")`

PrizePicks conflict cols:
`["league", "player_name", "stat_type", "odds_type", "line_score", "scraped_at"]`

Underdog conflict cols:
`["league", "player_name", "stat_name", "side", "line_score", "scraped_at"]`

- [ ] **Step 1: Failing test** that `load_prizepicks_snapshot` builds a DataFrame and calls `upsert_df` with `schema="odds"` (monkeypatch)

- [ ] **Step 2: Implement loaders**

```python
def load_prizepicks_snapshot(projections, *, league: str, scraped_at=None) -> int:
    if os.environ.get("PRIZEPICKS_SKIP_DB", "").strip() in {"1", "true", "yes"}:
        return 0
    scraped_at = scraped_at or datetime.now(timezone.utc)
    rows = prizepicks_projections_to_rows(projections, league=league, scraped_at=scraped_at)
    if not rows:
        return 0
    import pandas as pd
    from src.utils.db import upsert_df
    upsert_df(
        "prizepicks",
        pd.DataFrame(rows),
        schema="odds",
        conflict_cols=[
            "league", "player_name", "stat_type", "odds_type", "line_score", "scraped_at",
        ],
        lineage_col="fetched_at",
    )
    return len(rows)
```

Mirror for Underdog with `UNDERDOG_SKIP_DB`.

- [ ] **Step 3: Tests PASS**

---

### Task 4: Wire scrapers

**Files:**
- Modify: `src/scrapers/prizepicks_scraper.py` (`_save_by_league` / after `save_projections`)
- Modify: `src/scrapers/underdog_scraper.py` (after successful per-sport save)

**Interfaces:**
- Consumes: `load_prizepicks_snapshot`, `load_underdog_snapshot`
- After writing JSON for a league, convert that league’s projections/picks and load with `league=slug`

- [ ] **Step 1: PrizePicks** — after each league file save, call:

```python
try:
    from src.odds.load_snapshots import load_prizepicks_snapshot
    n = load_prizepicks_snapshot(
        [p.to_dict() for p in rows],
        league=league_slug(league_name),
        scraped_at=datetime.now(timezone.utc),  # same stamp for the run preferred
    )
    logger.info(f"Supabase odds.wnba_prizepicks upserted {n} rows ({league_name})")
except Exception as e:
    logger.error(f"Supabase prizepicks load failed (JSON kept): {e}")
```

Use one shared `scraped_at` for the whole `run()` so NBA+WNBA in the same invocation share a batch timestamp **or** per-league stamp — prefer **one stamp per `run()`** stored on the scraper instance.

- [ ] **Step 2: Underdog** — same pattern after saving each sport file; map sport → `nba`/`wnba`.

- [ ] **Step 3: Manual smoke** (optional, needs DB): scrape WNBA with DB enabled; `select count(*), max(scraped_at) from odds.wnba_prizepicks where league='wnba';`

---

### Task 5: Backend schema + odds readers

**Files:**
- Modify: `backend/app/schemas/wnba_props.py`
- Create: `backend/app/services/odds_snapshots.py`
- Create/Modify: `backend/tests/test_odds_snapshots.py`
- Modify: `backend/tests/test_sharp_props.py` (book list expectations)

**Interfaces:**
- `PROP_SPORTSBOOKS = ("fanduel", "draftkings", "prizepicks", "underdog")`
- `WnbaPropLine`: drop `betmgm`/`betrivers`; add `prizepicks: WnbaPropBookQuote | None` (`odds_american` may be unused — keep field; PP uses line only, odds can be omitted via making `odds_american` optional **or** use `0` sentinel — prefer make `odds_american: int | None = None` on `WnbaPropBookQuote`)
- Produces:
  - `fetch_latest_prizepicks(league: str = "wnba") -> list[dict]`  
    columns at least: player_name, stat_type, line_score, odds_type
  - `fetch_latest_underdog(league: str = "wnba") -> list[dict]`  
    columns: player_name, stat_name, line_score, side, american_price

Latest query pattern:

```sql
SELECT * FROM odds.wnba_prizepicks
WHERE league = :league
  AND scraped_at = (
    SELECT MAX(scraped_at) FROM odds.wnba_prizepicks WHERE league = :league
  )
```

Use SQLAlchemy engine from existing backend DB helpers if present; otherwise `src.utils.db.get_engine()` / raw SQL. If DB unavailable, return `[]` and log warning (do not crash props endpoint).

- [ ] **Step 1: Update schema + failing tests for book list**
- [ ] **Step 2: Implement `odds_snapshots.py` with mocked engine tests**
- [ ] **Step 3: Tests PASS**

---

### Task 6: Merge into `get_today_props`

**Files:**
- Modify: `backend/app/services/sharp_props.py`
- Modify: `backend/tests/test_sharp_props.py`
- Fixtures as needed

**Interfaces:**
- Sharp loop only over `("fanduel", "draftkings")` (or filter `PROP_SPORTSBOOKS` to Sharp-only constant `SHARP_PROP_SPORTSBOOKS = ("fanduel", "draftkings")`)
- After building Sharp buckets, call `fetch_latest_prizepicks` / `fetch_latest_underdog`
- Stat label helpers:
  - PP: use `stat_type` as display `stat`; `market_type` like `prizepicks:{stat_type}`
  - UD: title-case `stat_name`
- Merge key: `(norm_player_name, normalized_stat_key, side)` aligned with existing Sharp keying as much as possible
- PrizePicks: for each projection, attach quote to **both** `over` and `under` buckets (create buckets if missing), `WnbaPropBookQuote(line=line_score, odds_american=None)`
- Underdog: attach to matching `side` with `odds_american=american_price`
- Include a row if any of the four books is present
- `sportsbooks` in response = `list(PROP_SPORTSBOOKS)` new tuple
- Roster enrichment for team/logo continues for new players when possible

- [ ] **Step 1: Failing tests** — mock Sharp FD/DK + mock snapshot lists; assert PP/UD filled; betmgm absent
- [ ] **Step 2: Implement merge**
- [ ] **Step 3: `pytest backend/tests/test_sharp_props.py backend/tests/test_odds_snapshots.py -v` PASS**

---

### Task 7: Frontend

**Files:**
- Modify: `frontend/src/lib/api.ts` (`ApiWnbaPropLine` books)
- Modify: `frontend/src/components/league/PropPicksTable.tsx` (`BOOK_COLUMNS`)
- Modify: `frontend/src/components/league/PropPicksTable.test.tsx`
- Modify: `frontend/src/lib/api.test.ts` / `AppRouter.test.tsx` as needed
- Caption text under table if present

**Interfaces:**
- Books: fanduel, draftkings, prizepicks, underdog
- Handle null `odds_american` in pill render (show line only for PrizePicks)

- [ ] **Step 1: Update types + failing UI tests**
- [ ] **Step 2: Implement column/render changes**
- [ ] **Step 3: `cd frontend && npm test -- --run PropPicksTable api AppRouter` PASS**

---

### Task 8: End-to-end checklist (manual)

- [ ] Migration applied on Supabase
- [ ] Run Underdog + PrizePicks scrapers with DB creds; confirm new `scraped_at` batches
- [ ] Hit `GET /api/wnba/props/today`; confirm four books, no betmgm/betrivers
- [ ] Open `/wnba/prop_picks` and verify columns

---

## Spec coverage

| Spec item | Task |
| --- | --- |
| `odds` schema + tables + indexes | 1 |
| Snapshot PK / line-move friendly history | 1–3 |
| JSON → row mapping | 2 |
| Scraper → Supabase insert | 3–4 |
| Skip-DB env flags | 3–4 |
| Sharp FD/DK only | 5–6 |
| Latest PP/UD read | 5 |
| Merge + PP both sides | 6 |
| Frontend books | 7 |
| Manual verification | 8 |
