# WNBA Matchup DraftKings Odds Pill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show DraftKings spread/total odds on WNBA matchup cards via a Sharp API backend proxy.

**Architecture:** Separate `GET /api/wnba/odds/today` fetches Sharp odds (`point_spread` + `total_points`, DraftKings, main lines), normalizes to home/away abbrevs + favorite spread + total, caches ~45s. Frontend `useWnbaOdds` polls ~60s and merges onto scoreboard games by abbrev for `MatchupGameCard` pills.

**Tech Stack:** FastAPI, httpx, Pydantic, React, TanStack Query, Vitest, pytest

## Global Constraints

- Sportsbook: DraftKings only
- Markets: full-game main `point_spread` and `total_points` only (not 1st half / alternates)
- Show pill when either half present; partial OK; hide when both missing
- Odds failure → empty games; scoreboard still works
- `SHARP_API_KEY` server-side only
- Scope: WNBA matchup cards only
- Caption: `Odds by DraftKings`
- Pill format: `Spread: ATL -12.5 · Total: 178.5` (omit missing half)

## File structure

| File | Responsibility |
| --- | --- |
| `backend/app/core/config.py` | `SHARP_API_KEY` |
| `backend/app/schemas/wnba_odds.py` | Response models |
| `backend/app/services/sharp_odds.py` | Fetch, normalize, cache |
| `backend/app/api/routes/wnba_odds.py` | Route |
| `backend/app/main.py` | Register router + description |
| `backend/tests/fixtures/sharp_wnba_odds.json` | Fixture |
| `backend/tests/test_sharp_odds.py` | Service + route tests |
| `frontend/src/lib/api.ts` | Types + `fetchWnbaOdds` |
| `frontend/src/hooks/useWnbaOdds.ts` | Query hook |
| `frontend/src/components/league/types.ts` | Odds fields on `MatchupGame` |
| `frontend/src/components/league/mergeMatchupOdds.ts` | Merge helper |
| `frontend/src/pages/LeagueMatchupsPage.tsx` | Wire hook + merge |
| `frontend/src/components/league/MatchupGameCard.tsx` | Pill UI |
| Tests for api, merge, card | Coverage |

---

### Task 1: Backend Sharp normalize + route

**Files:**
- Create: `backend/app/schemas/wnba_odds.py`
- Create: `backend/app/services/sharp_odds.py`
- Create: `backend/app/api/routes/wnba_odds.py`
- Create: `backend/tests/fixtures/sharp_wnba_odds.json`
- Create: `backend/tests/test_sharp_odds.py`
- Modify: `backend/app/core/config.py`
- Modify: `backend/app/main.py`

**Interfaces:**
- Produces: `GET /api/wnba/odds/today` → `{ asOf, sportsbook, games: [{ homeAbbrev, awayAbbrev, spreadTeamAbbrev, spreadLine, total }], error? }`
- Produces: `normalize_sharp_odds(rows) -> list[WnbaOddsGame]`
- Produces: `get_today_odds() -> WnbaOddsResponse` (cache, never raises for empty)

- [ ] **Step 1: Write fixture + failing tests**

Fixture (`sharp_wnba_odds.json`) — trimmed Sharp-shaped rows for ATL@SEA (main spread ±12.5, total 179.5) and WAS@DAL (home object null, favorite DAL -3.5, total 167.5), plus one alternate (`is_main_line: false`) and one `1st_half_point_spread` to prove filters.

Tests in `test_sharp_odds.py`:

```python
def test_normalize_picks_favorite_spread_and_total():
    rows = json.loads(FIXTURE.read_text())["data"]
    games = normalize_sharp_odds(rows)
    atl = next(g for g in games if g.home_abbrev == "ATL")
    assert atl.away_abbrev == "SEA"
    assert atl.spread_team_abbrev == "ATL"
    assert atl.spread_line == -12.5
    assert atl.total == 179.5

def test_normalize_handles_missing_home_object_and_favorite_away():
    ...
    assert was.home_abbrev == "WAS"
    assert was.spread_team_abbrev == "DAL"
    assert was.spread_line == -3.5

def test_normalize_ignores_halves_and_alternates():
    assert all(g.home_abbrev in {"ATL", "WAS"} for g in games)

def test_odds_route_returns_games_when_fetch_ok(...):
def test_odds_route_empty_when_no_key(...):
def test_odds_route_stale_cache_on_error(...):
```

- [ ] **Step 2: Run tests — expect FAIL**

`cd backend && python -m pytest tests/test_sharp_odds.py -v`

- [ ] **Step 3: Implement**

`config.py`: `SHARP_API_KEY: str | None = os.environ.get("SHARP_API_KEY")`

`schemas/wnba_odds.py`: Pydantic models with camelCase aliases matching the design (`homeAbbrev`, etc.) via `serialization_alias` / `populate_by_name`.

`sharp_odds.py`:
- URL `https://api.sharpapi.io/api/v1/odds` with params `league=wnba`, `sportsbook=draftkings`, `market=point_spread,total_points`, `is_main_line=true`, `limit=200`
- Header `X-API-Key`
- Keep only `is_main_line` + market_type in `{point_spread, total_points}`
- Resolve abbrev from `home.abbreviation` / `away.abbreviation`, else first token of `home_team`/`away_team` if it looks like a tricode, else name map; run through `canonical_abbrev` from `wnba_scoreboard`
- Per `event_id`: pick spread row with `line < 0` (else any); total from over/under line
- Cache TTL 45s; on error return last good; no key → empty games + error string
- Pagination: if `meta.pagination.has_more`, follow `next_offset` up to a small cap (e.g. 3 pages)

`wnba_odds.py` route: always 200 + `Cache-Control: no-store`

`main.py`: include router; mention `/api/wnba/odds/today` in description

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add backend/app/core/config.py backend/app/schemas/wnba_odds.py \
  backend/app/services/sharp_odds.py backend/app/api/routes/wnba_odds.py \
  backend/app/main.py backend/tests/fixtures/sharp_wnba_odds.json \
  backend/tests/test_sharp_odds.py
git commit -m "feat: add WNBA DraftKings odds proxy via Sharp API"
```

---

### Task 2: Frontend fetch, merge, pill UI

**Files:**
- Modify: `frontend/src/lib/api.ts`
- Create: `frontend/src/hooks/useWnbaOdds.ts`
- Modify: `frontend/src/components/league/types.ts`
- Create: `frontend/src/components/league/mergeMatchupOdds.ts`
- Create: `frontend/src/components/league/mergeMatchupOdds.test.ts`
- Modify: `frontend/src/pages/LeagueMatchupsPage.tsx`
- Modify: `frontend/src/components/league/MatchupGameCard.tsx`
- Modify: `frontend/src/components/league/MatchupGameCard.test.tsx`
- Create: `frontend/src/lib/api.odds.test.ts` (or extend `api.test.ts`)

**Interfaces:**
- Consumes: `/api/wnba/odds/today` shape from Task 1
- Produces: `MatchupGame.odds?: { spreadTeamAbbrev, spreadLine, total } | null`
- Produces: pill + `Odds by DraftKings` caption

- [ ] **Step 1: Failing tests**

Merge + card tests for full pill, partial, absent, merge by abbrev (use `canonical` forms if needed — compare with scoreboard abbrevs as returned).

- [ ] **Step 2: Run — expect FAIL**

`cd frontend && npx vitest run src/components/league/mergeMatchupOdds.test.ts src/components/league/MatchupGameCard.test.tsx`

- [ ] **Step 3: Implement**

```ts
// types
odds?: {
  spreadTeamAbbrev: string | null;
  spreadLine: number | null;
  total: number | null;
} | null;
```

`formatOddsPill(odds)` helper in card or small util:
- build parts; join with ` · `

`useWnbaOdds`: `refetchInterval: 60_000`, queryKey `["wnba","odds","today"]`

`WnbaMatchupsPage`: merge `mapToMatchupGames(games)` with odds games via `homeAbbrev`+`awayAbbrev` (also try `canonical`-style equality by uppercasing)

Card: below team rows, pill + caption when odds has any non-null half

- [ ] **Step 4: Run frontend tests + backend tests**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: show DraftKings odds pill on WNBA matchup cards"
```

---

## Spec coverage check

| Spec requirement | Task |
| --- | --- |
| Separate odds endpoint | 1 |
| Sharp DraftKings spread/total | 1 |
| Cache / empty on failure | 1 |
| Pill + Odds by DraftKings | 2 |
| Partial lines | 1+2 |
| Merge by abbrev | 2 |
| WNBA matchups only | 2 |
| Key server-side | 1 |
