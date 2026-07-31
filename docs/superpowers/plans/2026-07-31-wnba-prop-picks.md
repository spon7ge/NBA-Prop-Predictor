# WNBA Prop Picks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Display today’s WNBA player prop main lines (FanDuel + DraftKings) on `/wnba/prop_picks` as a flat table with empty model/EV columns.

**Architecture:** New `GET /api/wnba/props/today` proxies Sharp (`market=props`, `sportsbook=draftkings,fanduel`, `is_main_line=true`), normalizes to one row per player+market+side with per-book quotes, caches ~45s. Frontend page + hairline table polls ~60s; Prop Picks subnav links to the route.

**Tech Stack:** FastAPI, httpx, Pydantic, React, TanStack Query, Vitest, pytest

## Global Constraints

- Route: `/wnba/prop_picks`
- Sportsbooks: FanDuel + DraftKings only
- Markets: all Sharp `player_*` via `market=props`; main lines only
- Row grain: player + market + side; show both Over and Under when provided
- Model / O/U% / EV always null/blank in v1
- Visual: hairline rows + odds pills; caption `Odds by FanDuel & DraftKings`
- `SHARP_API_KEY` server-side only; failure → empty props, page still loads
- Dedicated `sharp_props.py` (do not change matchup odds behavior)

## File structure

| File | Responsibility |
| --- | --- |
| `backend/app/schemas/wnba_props.py` | Response models |
| `backend/app/services/sharp_props.py` | Fetch, normalize, cache |
| `backend/app/api/routes/wnba_props.py` | Route |
| `backend/app/main.py` | Register router + description |
| `backend/tests/fixtures/sharp_wnba_props.json` | Fixture |
| `backend/tests/test_sharp_props.py` | Service + route tests |
| `frontend/src/lib/api.ts` | Types + `fetchWnbaProps` |
| `frontend/src/hooks/useWnbaProps.ts` | Query hook |
| `frontend/src/components/league/PropPicksTable.tsx` | Table UI |
| `frontend/src/components/league/PropPicksTable.test.tsx` | Table tests |
| `frontend/src/pages/LeaguePropPicksPage.tsx` | Page shell |
| `frontend/src/components/league/LeagueSubnav.tsx` | Enable Prop Picks link |
| `frontend/src/AppRouter.tsx` | Route registration |
| Subnav / router / api tests | Coverage |

---

### Task 1: Backend Sharp props normalize + route

**Files:**
- Create: `backend/app/schemas/wnba_props.py`
- Create: `backend/app/services/sharp_props.py`
- Create: `backend/app/api/routes/wnba_props.py`
- Create: `backend/tests/fixtures/sharp_wnba_props.json`
- Create: `backend/tests/test_sharp_props.py`
- Modify: `backend/app/main.py`

**Interfaces:**
- Produces: `normalize_sharp_props(rows) -> list[WnbaPropLine]`
- Produces: `get_today_props() -> WnbaPropsResponse`
- Produces: `GET /api/wnba/props/today` → `{ as_of, sportsbooks, props, error? }`

- [ ] **Step 1: Write fixture + failing tests**

Fixture: Rhyne Howard assists over/under on both books; Allisha Gray assists over FanDuel only; one non-main alternate; one `point_spread` row to ignore.

```python
def test_normalize_merges_books_and_both_sides():
    rows = json.loads(FIXTURE.read_text())["data"]
    props = normalize_sharp_props(rows)
    over = next(p for p in props if p.player_name == "Rhyne Howard" and p.side == "over")
    under = next(p for p in props if p.player_name == "Rhyne Howard" and p.side == "under")
    assert over.stat == "Assists"
    assert over.fanduel.line == 3.5 and over.fanduel.odds_american == -114
    assert over.draftkings.line == 3.5 and over.draftkings.odds_american == -120
    assert under.fanduel and under.draftkings

def test_normalize_keeps_row_when_one_book_missing():
    gray = next(p for p in props if p.player_name == "Allisha Gray" and p.side == "over")
    assert gray.fanduel is not None
    assert gray.draftkings is None

def test_normalize_ignores_non_props_and_alternates():
    assert all(p.market_type.startswith("player_") for p in props)

def test_props_route_ok / empty_when_no_key / stale_cache_on_error:  # mirror odds tests
```

- [ ] **Step 2: Run — expect FAIL**

`cd backend && python -m pytest tests/test_sharp_props.py -v`

- [ ] **Step 3: Implement schemas, service, route, main registration**

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add WNBA Sharp props proxy for FanDuel and DraftKings"
```

---

### Task 2: Frontend API, table, page, nav

**Files:**
- Modify: `frontend/src/lib/api.ts`
- Create: `frontend/src/hooks/useWnbaProps.ts`
- Create: `frontend/src/components/league/PropPicksTable.tsx`
- Create: `frontend/src/components/league/PropPicksTable.test.tsx`
- Create: `frontend/src/pages/LeaguePropPicksPage.tsx`
- Modify: `frontend/src/components/league/LeagueSubnav.tsx`
- Modify: `frontend/src/components/league/LeagueSubnav.test.tsx`
- Modify: `frontend/src/AppRouter.tsx`
- Modify: `frontend/src/AppRouter.test.tsx`
- Modify: `frontend/src/lib/api.test.ts` (if present pattern)

**Interfaces:**
- Consumes: `/api/wnba/props/today`
- Produces: Prop Picks table + active subnav link

- [ ] **Step 1: Failing tests** — table columns, both sides, blank model/EV, empty state; subnav link; router renders page

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement fetch, hook, table (hairline + pills), page, nav, router**

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add WNBA Prop Picks page with FanDuel and DraftKings lines"
```

---

## Spec coverage

| Spec item | Task |
| --- | --- |
| Sharp props proxy + cache | 1 |
| Both sides / both books / main lines | 1 |
| Empty model/O/U%/EV | 1–2 |
| Flat hairline table + pills | 2 |
| `/wnba/prop_picks` + subnav | 2 |
| Tests | 1–2 |
