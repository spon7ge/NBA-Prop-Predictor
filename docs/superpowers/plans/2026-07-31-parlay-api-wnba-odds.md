# ParlayAPI WNBA Odds Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Sharp with ParlayAPI for WNBA prop picks (6 books) and matchup odds (Pinnacle → DK), and snapshot all 6 books every 30 minutes.

**Architecture:** Shared `parlay_client`; `parlay_props` / `parlay_odds` services; extend per-book snapshot tables; drop PP/UD from prop picks response/UI.

**Tech Stack:** FastAPI, httpx, ParlayAPI v1, Postgres/Supabase, React/Vite, pytest, Vitest

## Global Constraints

- Base URL `https://parlay-api.com/v1`; auth via `PARLAY_API_KEY` (query `apiKey` and/or header per Parlay docs).
- Prop books: `fanduel`, `draftkings`, `caesars`, `betmgm`, `pinnacle`, `bet365`.
- Matchup: prefer Pinnacle, else DraftKings. Keep `WnbaOddsGame` shape.
- Main line = closest to −110/−110 per `(player, market, book)`.
- Snapshot throttle: `PARLAY_PROPS_SNAPSHOT_MINUTES` (default 30); joint gate across 6 tables.
- DB failures never fail API. Missing key → empty + error string.
- No Sharp calls on WNBA props/odds paths after cutover.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `db/migrations/021_odds_wnba_parlay_books.sql` | caesars, betmgm, pinnacle, bet365 tables |
| `backend/app/core/config.py` | `PARLAY_API_KEY` |
| `backend/app/schemas/wnba_props.py` | 6 book fields; drop PP/UD |
| `backend/app/services/parlay_client.py` | Shared GET helper |
| `backend/app/services/parlay_props.py` | Fetch/normalize/cache props |
| `backend/app/services/parlay_odds.py` | Fetch/normalize/cache matchup odds |
| `backend/app/api/routes/wnba_props.py` | Wire `get_today_props` |
| `backend/app/api/routes/wnba_odds.py` | Wire `get_today_odds` |
| `src/odds/snapshot_rows.py` | `parlay_props_to_book_rows` |
| `src/odds/load_snapshots.py` | 6-book throttle + persist |
| `frontend/src/components/league/PropPicksTable.tsx` | 6 columns |
| Tests + OpenAPI regen | Backend/frontend/fixtures |

---

### Task 1: Migration + schema + config

**Files:**
- Create: `db/migrations/021_odds_wnba_parlay_books.sql`
- Modify: `backend/app/core/config.py`
- Modify: `backend/app/schemas/wnba_props.py`

- [ ] **Step 1:** Create four tables matching migration 020 shape for caesars, betmgm, pinnacle, bet365.
- [ ] **Step 2:** Add `PARLAY_API_KEY` (strip quotes like Sharp).
- [ ] **Step 3:** Update `PROP_SPORTSBOOKS` and `WnbaPropLine` to six books; remove prizepicks/underdog fields.

### Task 2: Parlay props normalize + client (TDD)

**Files:**
- Create: `backend/app/services/parlay_client.py`
- Create: `backend/app/services/parlay_props.py`
- Create: `backend/tests/fixtures/parlay_wnba_props.json`
- Create: `backend/tests/test_parlay_props.py`
- Modify: `backend/app/api/routes/wnba_props.py`

- [ ] **Step 1:** Failing tests for main-line picker, Over/Under expand, six-book buckets, missing-key path.
- [ ] **Step 2:** Implement client + `normalize_parlay_props` + `fetch_parlay_prop_rows` + `get_today_props` (no PP/UD merge; ESPN roster enrichment adapted for Parlay team names).
- [ ] **Step 3:** Point route at `parlay_props.get_today_props`. Tests pass.

### Task 3: Parlay matchup odds (TDD)

**Files:**
- Create: `backend/app/services/parlay_odds.py`
- Create: `backend/tests/fixtures/parlay_wnba_odds.json`
- Create: `backend/tests/test_parlay_odds.py`
- Modify: `backend/app/api/routes/wnba_odds.py`

- [ ] **Step 1:** Failing tests for Pinnacle preferred / DK fallback / team abbrev mapping.
- [ ] **Step 2:** Implement TOA-shape normalize + fetch + `get_today_odds`.
- [ ] **Step 3:** Wire route. Tests pass.

### Task 4: Snapshots for six books (TDD)

**Files:**
- Modify: `src/odds/snapshot_rows.py`
- Modify: `src/odds/load_snapshots.py`
- Modify: `tests/odds/test_snapshot_rows.py`
- Modify: `tests/odds/test_load_snapshots.py`
- Wire persist from `parlay_props.get_today_props`

- [ ] **Step 1:** `parlay_props_to_book_rows` maps Parlay main lines (after same main-line selection) to over/under snapshot rows.
- [ ] **Step 2:** `maybe_persist_parlay_props` with joint 6-table throttle; env `PARLAY_PROPS_SNAPSHOT_MINUTES` (fallback read `SHARP_PROPS_SNAPSHOT_MINUTES` for compat).
- [ ] **Step 3:** Call after successful Parlay fetch; swallow errors. Tests pass.

### Task 5: Frontend + OpenAPI

**Files:**
- Modify: `frontend/src/components/league/PropPicksTable.tsx` (+ tests)
- Modify: related frontend fixtures/tests (`api.test.ts`, `AppRouter.test.tsx`, `filterPropLines` fixtures as needed)
- Regenerate: `frontend/openapi.json` + `src/lib/api.schema.d.ts`

- [ ] **Step 1:** Six book columns; update caption; drop PP/UD.
- [ ] **Step 2:** Export OpenAPI from FastAPI and run `npm run generate:api`.
- [ ] **Step 3:** Frontend tests pass.

### Task 6: Cleanup verification

- [ ] **Step 1:** Ensure WNBA routes no longer import Sharp services.
- [ ] **Step 2:** Run backend props/odds/snapshot tests + frontend PropPicks tests.
- [ ] **Step 3:** Leave Sharp modules in tree if unused elsewhere (out of scope to delete).
