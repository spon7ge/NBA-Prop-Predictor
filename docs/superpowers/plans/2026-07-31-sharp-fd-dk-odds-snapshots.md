# Sharp FD/DK Odds Snapshots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist FanDuel and DraftKings Sharp main-line props into `odds.wnba_fanduel` / `odds.wnba_draftkings` at most once every 30 minutes after a successful Sharp fetch in `get_today_props`.

**Architecture:** Migration for tables; map Sharp rows → book-specific row dicts; throttle via `MAX(scraped_at)`; best-effort insert from `get_today_props` without failing the API. Prop picks UI still serves live Sharp.

**Tech Stack:** Postgres/Supabase, pandas/`upsert_df`, FastAPI `sharp_props`, pytest

## Global Constraints

- Throttle default 30 minutes; env `SHARP_PROPS_SNAPSHOT_MINUTES`.
- Joint gate: skip both books if latest scrape on either table is within N minutes.
- Main lines only; side `over`/`under`; books `fanduel` / `draftkings`.
- DB failures log; never fail API. Missing DB URL → skip.
- Prop picks response still from live Sharp for FD/DK.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `db/migrations/020_odds_wnba_fanduel_draftkings.sql` | Create tables + indexes |
| `src/odds/snapshot_rows.py` | `sharp_props_to_book_rows` |
| `src/odds/load_snapshots.py` | Throttle + load FD/DK batches |
| `backend/app/services/sharp_props.py` | Call persist after Sharp fetch |
| `tests/odds/test_snapshot_rows.py` | Mapping tests |
| `tests/odds/test_load_snapshots.py` | Load/throttle tests |
| `backend/tests/test_sharp_props.py` | Persist hook does not break API |

---

### Task 1: Migration

**Files:**
- Create: `db/migrations/020_odds_wnba_fanduel_draftkings.sql`

- [ ] **Step 1: Write migration** matching the spec (both tables, PKs, indexes).
- [ ] **Step 2: Commit** (if user requested commits; otherwise leave unstaged).

### Task 2: Sharp → row mapping (TDD)

**Files:**
- Modify: `src/odds/snapshot_rows.py`
- Modify: `tests/odds/test_snapshot_rows.py`

- [ ] **Step 1: Failing tests** for `sharp_props_to_book_rows(rows, *, sportsbook, league, scraped_at)`.
- [ ] **Step 2: Implement** filter main `player_*` over/under for that book; columns per spec.
- [ ] **Step 3: Tests pass.**

### Task 3: Throttle + load + wire API (TDD)

**Files:**
- Modify: `src/odds/load_snapshots.py`
- Modify: `tests/odds/test_load_snapshots.py`
- Modify: `backend/app/services/sharp_props.py`
- Modify: `backend/tests/test_sharp_props.py` as needed

- [ ] **Step 1: Tests** for throttle skip / write, `maybe_persist_sharp_props`.
- [ ] **Step 2: Implement** `should_persist_sharp_props`, `load_fanduel_snapshot`, `load_draftkings_snapshot`, `maybe_persist_sharp_props`.
- [ ] **Step 3: Call** from `get_today_props` after successful Sharp fetch (try/except).
- [ ] **Step 4: Verify** backend + odds tests pass.
