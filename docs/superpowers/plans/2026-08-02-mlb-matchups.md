# MLB Matchups Hub Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Live `/mlb/matchups` with dated Stats API scoreboard, Sharp MLB odds (DK/FD), and MatchupsPanel parity with WNBA.

**Architecture:** Parallel MLB APIs (`/api/mlb/scoreboard?date=`, `/api/mlb/odds/today`) + `MlbMatchupsPage` mirroring `WnbaMatchupsPage`. Parameterize Sharp fetch with `league=mlb` and markets `run_line,total_runs`. Cards use `gameDetailHref`.

**Tech Stack:** FastAPI · Sharp API · MLB Stats API · React · TanStack Query · Vitest · pytest

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-02-mlb-matchups-design.md`
- Sharp: `league=mlb`, markets `run_line,total_runs` (adjust if probe shows otherwise)
- Odds window: today + 2 ET days
- Prefer DraftKings over FanDuel
- Card links: MLB → `/mlb/games/{pk}`; WNBA unchanged
- 502 + no-store on hard scoreboard miss
- Verify: `PYTHONPATH=backend python3 -m pytest backend/tests/test_mlb_* -v` and focused frontend vitest + `npm run check:api`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `backend/app/services/mlb_scoreboard.py` | `get_scoreboard_for_date` |
| `backend/app/api/routes/mlb_scoreboard.py` | Dated GET |
| `backend/app/schemas/mlb_odds.py` | Odds models (same fields as WNBA) |
| `backend/app/services/mlb_odds.py` | Sharp MLB fetch/normalize/cache |
| `backend/app/services/sharp_odds.py` | Extract shared pagination helper with `league` + `market` params (optional refactor) |
| `backend/app/api/routes/mlb_odds.py` | `GET /mlb/odds/today` |
| `backend/app/main.py` | Mount |
| `frontend` hooks/api/page/card | Dated scoreboard, odds, MlbMatchupsPage, gameDetailHref on cards |

### Task 1: Dated MLB scoreboard
### Task 2: Sharp MLB odds service + route
### Task 3: OpenAPI + frontend API client
### Task 4: Hooks + MlbMatchupsPage + card links
### Task 5: System design doc + verify

Each task: TDD → implement → commit (see spec for interfaces).
