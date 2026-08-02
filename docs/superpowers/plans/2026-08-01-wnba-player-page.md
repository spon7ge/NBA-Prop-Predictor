# WNBA Player Profile Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `/wnba/player/:playerId` with identity + season averages and an expandable recent-games log, linked from WNBA Leaders names, backed by `GET /api/wnba/player/{player_id}` from stats.wnba.com.

**Architecture:** Backend proxies `leaguedashplayerstats` (identity + avgs), `commonplayerinfo` (position), and `playergamelog` (season games), normalizes into one response, caches 10 minutes per player. Frontend page under `HomeChromeLayout` renders header + recent games; Leaders name cells link to the player route. “See more” is client-only expand of the full `games` array.

**Tech Stack:** FastAPI · Pydantic · httpx · pytest · React 19 · TypeScript · Vite · TanStack Query · React Router · Vitest · Testing Library · Tailwind 4 · openapi-typescript

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-01-wnba-player-page-design.md`
- Coding standards: `CLAUDE.md` (small focused modules, strong typing, tests with code)
- Route: `/wnba/player/:playerId` only; no `/nba/player/...`
- Player ID: same string as leaders `player_id` (stats.wnba.com `PLAYER_ID`)
- Entry point v1: Leaders name links only
- Attribution copy: exact string `Data: stats.wnba.com`
- HTTP responses: `Cache-Control: no-store`; freshness via in-process ~10-minute TTL
- Recent games UI: default 5 rows; expand in place; hide control when ≤5 games
- Game columns: Date | Matchup | MIN | PTS | FG | 3PT | FT | REB | AST | TO | STL | BLK
- Averages: PTS, REB, AST, FG%, 3P% as one-decimal display strings
- Verify backend: `cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor && PYTHONPATH=backend python3 -m pytest backend/tests/test_wnba_player.py -v`
- Verify frontend: `cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor/frontend && npm run test -- --run src/components/league/PlayerHeader.test.tsx src/components/league/PlayerRecentGames.test.tsx src/components/league/LeaderCategoryCard.test.tsx src/pages/LeaguePlayerPage.test.tsx src/AppRouter.test.tsx && npm run build`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `backend/app/schemas/wnba_player.py` | Response models |
| `backend/app/services/wnba_player.py` | Fetch, normalize, cache, headshot URL |
| `backend/app/api/routes/wnba_player.py` | `GET /api/wnba/player/{player_id}` |
| `backend/app/main.py` | Register router + description mention |
| `backend/app/openapi_export.py` | Add required path |
| `backend/tests/fixtures/stats_wnba_player_dash.json` | Trimmed leaguedash row(s) |
| `backend/tests/fixtures/stats_wnba_player_info.json` | commonplayerinfo fixture |
| `backend/tests/fixtures/stats_wnba_player_gamelog.json` | playergamelog fixture (≥6 games) |
| `backend/tests/test_wnba_player.py` | Normalize + route + 404/502/cache |
| `frontend/openapi.json` + `src/lib/api.schema.d.ts` | Regenerated OpenAPI types |
| `frontend/src/lib/api.ts` | Types + `fetchWnbaPlayer` |
| `frontend/src/hooks/useWnbaPlayer.ts` | React Query hook |
| `frontend/src/components/league/PlayerHeader.tsx` | Bio left, avgs right |
| `frontend/src/components/league/PlayerHeader.test.tsx` | Header + headshot fallback |
| `frontend/src/components/league/PlayerRecentGames.tsx` | Table + see more / show less |
| `frontend/src/components/league/PlayerRecentGames.test.tsx` | Expand / hide control |
| `frontend/src/pages/LeaguePlayerPage.tsx` | Compose page |
| `frontend/src/pages/LeaguePlayerPage.test.tsx` | Page smoke + states |
| `frontend/src/AppRouter.tsx` | Register route |
| `frontend/src/AppRouter.test.tsx` | Route smoke |
| `frontend/src/components/league/LeaderCategoryCard.tsx` | Name → Link |
| `frontend/src/components/league/LeaderCategoryCard.test.tsx` | Link href |

---

### Task 1: Schema + normalize (pure)

**Files:**
- Create: `backend/app/schemas/wnba_player.py`
- Create: `backend/app/services/wnba_player.py` (normalize helpers only in this task — no HTTP yet beyond stubs if needed)
- Create: `backend/tests/fixtures/stats_wnba_player_dash.json`
- Create: `backend/tests/fixtures/stats_wnba_player_info.json`
- Create: `backend/tests/fixtures/stats_wnba_player_gamelog.json`
- Create: `backend/tests/test_wnba_player.py`

**Interfaces:**
- Consumes: stats.wnba.com `resultSets` payloads (`headers` + `rowSet`)
- Produces:
  - Models: `WnbaPlayerAverages`, `WnbaPlayerGame`, `WnbaPlayerResponse`
  - `rows_as_dicts(payload: dict) -> list[dict]` (or reuse private helper)
  - `format_avg(raw: Any) -> str | None` — one decimal
  - `format_pct(raw: Any) -> str | None` — if `0 < raw <= 1`, multiply by 100 then one decimal; if already `> 1`, one decimal as-is
  - `made_attempt(made: Any, attempted: Any) -> str` — `"m-a"`
  - `headshot_url_for(player_id: str) -> str` — CDN template (see Step 3)
  - `normalize_wnba_player(*, player_id: str, season: int, dash: dict, info: dict, gamelog: dict) -> WnbaPlayerResponse | None` — `None` if no matching dash row

- [ ] **Step 1: Write fixtures**

`stats_wnba_player_dash.json` — minimal `leaguedashplayerstats` with one player row including `PLAYER_ID`, `PLAYER_NAME`, `TEAM_ABBREVIATION`, `TEAM_NAME`, `PTS`, `REB`, `AST`, `FG_PCT`, `FG3_PCT`.

`stats_wnba_player_info.json` — minimal `commonplayerinfo` with `POSITION` (and optionally `TEAM_NAME`).

`stats_wnba_player_gamelog.json` — `playergamelog` with **at least 6** games, newest-first friendly, fields: `Game_ID`/`GAME_ID`, `GAME_DATE`, `MATCHUP`, `MIN`, `PTS`, `FGM`, `FGA`, `FG3M`, `FG3A`, `FTM`, `FTA`, `REB`, `AST`, `TOV`, `STL`, `BLK`.

Use realistic values for player_id `1628932` / name `A'ja Wilson` so tests read clearly.

- [ ] **Step 2: Write failing normalize tests**

```python
from __future__ import annotations

import json
from pathlib import Path

from app.services import wnba_player as svc

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_format_pct_handles_fraction_and_percent():
    assert svc.format_pct(0.482) == "48.2"
    assert svc.format_pct(48.2) == "48.2"
    assert svc.format_pct(None) is None


def test_made_attempt():
    assert svc.made_attempt(11, 20) == "11-20"


def test_normalize_player_happy_path():
    result = svc.normalize_wnba_player(
        player_id="1628932",
        season=2026,
        dash=_load("stats_wnba_player_dash.json"),
        info=_load("stats_wnba_player_info.json"),
        gamelog=_load("stats_wnba_player_gamelog.json"),
    )
    assert result is not None
    assert result.player_id == "1628932"
    assert result.name == "A'ja Wilson"
    assert result.position  # from info
    assert result.team_abbrev == "LVA"
    assert result.averages.pts  # one-decimal string
    assert result.averages.fg_pct
    assert result.averages.fg3_pct
    assert len(result.games) >= 6
    g0 = result.games[0]
    assert g0.fg  # "m-a"
    assert g0.three_pt
    assert g0.ft
    assert result.source_label == "stats.wnba.com"
    assert result.headshot_url  # non-empty CDN URL containing player_id


def test_normalize_unknown_player_returns_none():
    result = svc.normalize_wnba_player(
        player_id="99999999",
        season=2026,
        dash=_load("stats_wnba_player_dash.json"),
        info=_load("stats_wnba_player_info.json"),
        gamelog=_load("stats_wnba_player_gamelog.json"),
    )
    assert result is None
```

- [ ] **Step 3: Run tests — expect FAIL**

Run: `cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor && PYTHONPATH=backend python3 -m pytest backend/tests/test_wnba_player.py -v`

Expected: FAIL (module / functions missing)

- [ ] **Step 4: Implement schemas + normalize**

`backend/app/schemas/wnba_player.py`:

```python
from __future__ import annotations

from pydantic import BaseModel


class WnbaPlayerAverages(BaseModel):
    pts: str
    reb: str
    ast: str
    fg_pct: str
    fg3_pct: str


class WnbaPlayerGame(BaseModel):
    game_id: str
    game_date: str
    matchup: str
    min: str
    pts: str
    fg: str
    three_pt: str
    ft: str
    reb: str
    ast: str
    to: str
    stl: str
    blk: str


class WnbaPlayerResponse(BaseModel):
    player_id: str
    name: str
    position: str | None
    team_name: str
    team_abbrev: str
    headshot_url: str | None
    season: int
    averages: WnbaPlayerAverages
    games: list[WnbaPlayerGame]
    source_label: str = "stats.wnba.com"
```

In `wnba_player.py`:
- Reuse the same `resultSets` → dict-rows pattern as `wnba_leaders._rows_as_dicts`.
- Find dash row where `str(PLAYER_ID) == player_id`.
- Position from info’s first CommonPlayerInfo row (`POSITION` or `POSITION_ABBREVIATION`).
- `team_abbrev` from dash `TEAM_ABBREVIATION` (upper).
- `team_name` from dash `TEAM_NAME` if present, else info team name, else abbrev.
- Sort games by `GAME_DATE` descending if upstream order is uncertain.
- Headshot: `https://cdn.wnba.com/headshots/wnba/latest/1040x760/{player_id}.png` (if live 404s during manual check, switch to the ak-static CMS path used by WNBA sites — keep one pattern; UI has onError fallback).

- [ ] **Step 5: Run tests — expect PASS**

Run: `PYTHONPATH=backend python3 -m pytest backend/tests/test_wnba_player.py -v`  
Expected: PASS for normalize tests

- [ ] **Step 6: Commit**

```bash
git add backend/app/schemas/wnba_player.py backend/app/services/wnba_player.py \
  backend/tests/fixtures/stats_wnba_player_*.json backend/tests/test_wnba_player.py
git commit -m "$(cat <<'EOF'
feat: add WNBA player normalize and response schemas

EOF
)"
```

---

### Task 2: Fetch, cache, route

**Files:**
- Modify: `backend/app/services/wnba_player.py`
- Create: `backend/app/api/routes/wnba_player.py`
- Modify: `backend/app/main.py`
- Modify: `backend/tests/test_wnba_player.py`

**Interfaces:**
- Consumes: `normalize_wnba_player`, schemas from Task 1
- Produces:
  - `async def fetch_leaguedashplayerstats(season: int) -> dict`
  - `async def fetch_commonplayerinfo(player_id: str) -> dict`
  - `async def fetch_playergamelog(player_id: str, season: int) -> dict`
  - `async def get_wnba_player(player_id: str) -> WnbaPlayerResponse` — raises `HTTPException(404)` when normalize returns `None`; on upstream failure serves stale cache for that player_id if present, else re-raises
  - Route: `GET /api/wnba/player/{player_id}` → `WnbaPlayerResponse`

Upstream URLs (LeagueID=10, SeasonType Regular Season, same headers as leaders):
- `https://stats.wnba.com/stats/leaguedashplayerstats` — same params as `wnba_leaders.fetch_leaguedashplayerstats`
- `https://stats.wnba.com/stats/commonplayerinfo?PlayerID={id}&LeagueID=10`
- `https://stats.wnba.com/stats/playergamelog?PlayerID={id}&Season={year}&SeasonType=Regular%20Season&LeagueID=10`

Cache: dict keyed by `player_id` → `{response, expires_at, season}`; TTL `10 * 60`; clear season mismatch. Prefer fetching dash + info + gamelog concurrently with `asyncio.gather`. Reuse `current_wnba_season_year` from `wnba_leaders` (import) to avoid duplicating season logic.

- [ ] **Step 1: Add failing route tests**

```python
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(autouse=True)
def clear_player_cache():
    svc._cache.clear()
    yield
    svc._cache.clear()


def test_player_route_200_no_store():
    async def fake_get(player_id: str):
        return svc.normalize_wnba_player(
            player_id="1628932",
            season=2026,
            dash=_load("stats_wnba_player_dash.json"),
            info=_load("stats_wnba_player_info.json"),
            gamelog=_load("stats_wnba_player_gamelog.json"),
        )

    with patch.object(svc, "get_wnba_player", side_effect=fake_get):
        client = TestClient(app)
        res = client.get("/api/wnba/player/1628932")
    assert res.status_code == 200
    assert res.headers.get("cache-control") == "no-store"
    assert res.json()["name"] == "A'ja Wilson"


def test_player_route_404():
    from fastapi import HTTPException

    async def missing(player_id: str):
        raise HTTPException(status_code=404, detail="Player not found")

    with patch.object(svc, "get_wnba_player", side_effect=missing):
        client = TestClient(app)
        res = client.get("/api/wnba/player/999")
    assert res.status_code == 404


def test_player_route_502_cold():
    async def boom(*_a, **_k):
        raise RuntimeError("upstream down")

    with patch.object(svc, "fetch_leaguedashplayerstats", side_effect=boom), \
         patch.object(svc, "fetch_commonplayerinfo", side_effect=boom), \
         patch.object(svc, "fetch_playergamelog", side_effect=boom):
        # If get_wnba_player is real: patch fetches; expect 502 from route
        client = TestClient(app)
        res = client.get("/api/wnba/player/1628932")
    assert res.status_code == 502
    assert res.headers.get("cache-control") == "no-store"
```

Prefer testing `get_wnba_player` with patched fetches for cache/stale behavior (mirror `test_wnba_leaders_route.py`), and keep route tests thin.

- [ ] **Step 2: Run — expect FAIL** (route missing / 404)

- [ ] **Step 3: Implement fetch + cache + route + register**

`backend/app/api/routes/wnba_player.py` — mirror `wnba_leaders.py` (try/except → 502, `Cache-Control: no-store`). Map missing player to 404 inside `get_wnba_player` or the route.

Register in `main.py`: `app.include_router(wnba_player.router, prefix="/api")` and mention the path in the app description string alongside other WNBA exceptions if that list exists.

- [ ] **Step 4: Run — expect PASS**

Run: `PYTHONPATH=backend python3 -m pytest backend/tests/test_wnba_player.py -v`

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/wnba_player.py backend/app/api/routes/wnba_player.py \
  backend/app/main.py backend/tests/test_wnba_player.py
git commit -m "$(cat <<'EOF'
feat: add GET /api/wnba/player/{id} with fetch and cache

EOF
)"
```

---

### Task 3: OpenAPI + frontend fetch + hook

**Files:**
- Modify: `backend/app/openapi_export.py` — add `"/api/wnba/player/{player_id}"` to `REQUIRED_WNBA_PATHS`
- Regenerate: `frontend/openapi.json`, `frontend/src/lib/api.schema.d.ts`
- Modify: `frontend/src/lib/api.ts`
- Create: `frontend/src/hooks/useWnbaPlayer.ts`
- Create: `frontend/src/hooks/useWnbaPlayer.test.tsx` (optional but preferred; mirror leaders hook test)

**Interfaces:**
- Produces:
  - `ApiWnbaPlayerResponse`, `ApiWnbaPlayerAverages`, `ApiWnbaPlayerGame` from schemas
  - `fetchWnbaPlayer(playerId: string): Promise<ApiWnbaPlayerResponse>`
  - `useWnbaPlayer(playerId: string)` — queryKey `["wnba", "player", playerId]`; `enabled: Boolean(playerId)`; expose `hasNeverLoaded` like leaders

- [ ] **Step 1: Export OpenAPI + generate types**

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor
# ensure openapi_export REQUIRED path includes /api/wnba/player/{player_id}
python3 -c "from app.openapi_export import export_openapi; export_openapi()" 
# or: PYTHONPATH=backend python3 -m app.openapi_export / scripts/export_openapi.py if that exists
cd frontend && npm run generate:api
```

Confirm `WnbaPlayerResponse` appears in `api.schema.d.ts`.

- [ ] **Step 2: Add api.ts helpers**

```typescript
export type ApiWnbaPlayerAverages = Schemas["WnbaPlayerAverages"];
export type ApiWnbaPlayerGame = Schemas["WnbaPlayerGame"];
export type ApiWnbaPlayerResponse = Schemas["WnbaPlayerResponse"];

export async function fetchWnbaPlayer(
  playerId: string,
): Promise<ApiWnbaPlayerResponse> {
  const res = await fetch(`${API_BASE}/api/wnba/player/${playerId}`, {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`Player request failed: ${res.status}`);
  }
  return res.json();
}
```

- [ ] **Step 3: Hook**

```typescript
import { useQuery } from "@tanstack/react-query";
import { fetchWnbaPlayer } from "@/lib/api";

export function useWnbaPlayer(playerId: string) {
  const query = useQuery({
    queryKey: ["wnba", "player", playerId],
    queryFn: () => fetchWnbaPlayer(playerId),
    enabled: Boolean(playerId),
  });
  return {
    ...query,
    hasNeverLoaded: query.isError && query.data === undefined,
  };
}
```

- [ ] **Step 4: Commit**

```bash
git add backend/app/openapi_export.py frontend/openapi.json \
  frontend/src/lib/api.schema.d.ts frontend/src/lib/api.ts \
  frontend/src/hooks/useWnbaPlayer.ts frontend/src/hooks/useWnbaPlayer.test.tsx
git commit -m "$(cat <<'EOF'
chore: sync OpenAPI and add WNBA player fetch helper

EOF
)"
```

---

### Task 4: PlayerHeader + PlayerRecentGames

**Files:**
- Create: `frontend/src/components/league/PlayerHeader.tsx`
- Create: `frontend/src/components/league/PlayerHeader.test.tsx`
- Create: `frontend/src/components/league/PlayerRecentGames.tsx`
- Create: `frontend/src/components/league/PlayerRecentGames.test.tsx`

**Interfaces:**
- Consumes: `ApiWnbaPlayerResponse` fields
- Produces:
  - `PlayerHeader({ player }: { player: ApiWnbaPlayerResponse })`
  - `PlayerRecentGames({ games }: { games: ApiWnbaPlayerGame[] })` — internal `expanded` state; default show 5

- [ ] **Step 1: Failing component tests**

`PlayerHeader.test.tsx`: renders name, position, team_name, avg labels PTS/REB/AST/FG%/3P%, values from fixture object. Simulate img `error` → placeholder still present (`alt` or role).

`PlayerRecentGames.test.tsx`: with 6 games, only 5 rows visible; click “See more” → 6 rows + “Show less”; with 3 games, no “See more” button.

- [ ] **Step 2: Run — expect FAIL**

`cd frontend && npm run test -- --run src/components/league/PlayerHeader.test.tsx src/components/league/PlayerRecentGames.test.tsx`

- [ ] **Step 3: Implement UI**

Layout (match leaders charcoal style):
- Header: flex row on md+; left stack headshot (rounded, fixed size) + name/position/team; right grid of five avg tiles (label muted, value bold). Show `%` suffix in FG%/3P% labels; values already numeric strings from API.
- Games: overflow-x-auto table; section heading “Recent games”; button text exact **See more** / **Show less**.
- Empty games: “No games yet”.

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/league/PlayerHeader.tsx \
  frontend/src/components/league/PlayerHeader.test.tsx \
  frontend/src/components/league/PlayerRecentGames.tsx \
  frontend/src/components/league/PlayerRecentGames.test.tsx
git commit -m "$(cat <<'EOF'
feat: add WNBA player header and recent games components

EOF
)"
```

---

### Task 5: LeaguePlayerPage + route

**Files:**
- Create: `frontend/src/pages/LeaguePlayerPage.tsx`
- Create: `frontend/src/pages/LeaguePlayerPage.test.tsx`
- Modify: `frontend/src/AppRouter.tsx`
- Modify: `frontend/src/AppRouter.test.tsx`

**Interfaces:**
- Consumes: `useParams().playerId`, `useWnbaPlayer`, `PlayerHeader`, `PlayerRecentGames`, `LeagueSubnav`
- Produces: page at `/wnba/player/:playerId`

- [ ] **Step 1: Failing page + router tests**

Page test: mock `fetch` for `/api/wnba/player/1628932` returning a minimal `ApiWnbaPlayerResponse`; render with `MemoryRouter` initial entry `/wnba/player/1628932` **or** render page with a wrapper that provides router params. Assert name, “Recent games”, attribution `Data: stats.wnba.com`.

Router test (extend `AppRouter.test.tsx`): navigate to `/wnba/player/1628932` with fetch mock → finds player name.

404/error: when fetch returns 404, page shows “Player not found”.

- [ ] **Step 2: Implement page**

```tsx
// Sketch — match LeagueLeadersPage chrome
export function LeaguePlayerPage() {
  const { playerId = "" } = useParams();
  const { data, isLoading, hasNeverLoaded, error } = useWnbaPlayer(playerId);
  // loading skeleton when isLoading && !data
  // if hasNeverLoaded && status 404-ish → "Player not found"
  // else error message
  // else: LeagueSubnav + PlayerHeader + PlayerRecentGames + attribution
}
```

Wire `AppRouter.tsx`:

```tsx
<Route path="/wnba/player/:playerId" element={<LeaguePlayerPage />} />
```

- [ ] **Step 3: Run tests + build**

`npm run test -- --run src/pages/LeaguePlayerPage.test.tsx src/AppRouter.test.tsx`  
`npm run build`

- [ ] **Step 4: Commit**

```bash
git add frontend/src/pages/LeaguePlayerPage.tsx \
  frontend/src/pages/LeaguePlayerPage.test.tsx \
  frontend/src/AppRouter.tsx frontend/src/AppRouter.test.tsx
git commit -m "$(cat <<'EOF'
feat: add /wnba/player/:playerId page route

EOF
)"
```

---

### Task 6: Leaders name → player link

**Files:**
- Modify: `frontend/src/components/league/LeaderCategoryCard.tsx`
- Create: `frontend/src/components/league/LeaderCategoryCard.test.tsx`

**Interfaces:**
- Consumes: `row.player_id`, React Router `Link`
- Produces: name cell links to `/wnba/player/${row.player_id}`

- [ ] **Step 1: Failing test**

Render card with one leader; assert `getByRole("link", { name: "A'ja Wilson" })` has `href` ending `/wnba/player/1628932` (use `MemoryRouter`).

- [ ] **Step 2: Implement**

Replace name `<td>` text with:

```tsx
<Link
  to={`/wnba/player/${row.player_id}`}
  className="text-white hover:underline focus-visible:underline"
>
  {row.name}
</Link>
```

Do not wrap the whole row.

- [ ] **Step 3: Run related tests**

`npm run test -- --run src/components/league/LeaderCategoryCard.test.tsx src/pages/LeagueLeadersPage.test.tsx`

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/league/LeaderCategoryCard.tsx \
  frontend/src/components/league/LeaderCategoryCard.test.tsx
git commit -m "$(cat <<'EOF'
feat: link WNBA leaders names to player profiles

EOF
)"
```

---

## Spec coverage checklist

| Spec requirement | Task |
| --- | --- |
| `/wnba/player/:playerId` | 5 |
| `GET /api/wnba/player/{id}` | 2 |
| stats.wnba dash + info + gamelog | 1–2 |
| Header bio + PTS/REB/AST/FG%/3P% | 4 |
| Last 5 + expand in place | 4 |
| Box-score game columns | 1, 4 |
| Leaders name links only | 6 |
| Attribution `Data: stats.wnba.com` | 5 |
| Cache ~10 min / 404 / 502 | 2 |
| OpenAPI + typed client | 3 |
| No NBA player route / no other entry points | Global + out of scope |

## Plan self-review

- No TBD placeholders; CDN headshot has explicit fallback path if URL pattern fails.
- Types consistent: `WnbaPlayerResponse` / `ApiWnbaPlayerResponse` / `useWnbaPlayer(playerId)`.
- “See more” is UI-only; API always returns full season games.
