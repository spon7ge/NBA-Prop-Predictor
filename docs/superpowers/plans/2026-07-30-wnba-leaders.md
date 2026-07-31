# WNBA League Leaders Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `/wnba/leaders` with a boxseats-style top-10 grid (PTS/REB/AST/STL/BLK/3PM) fed by `GET /api/wnba/leaders` from stats.wnba.com, and enable Leaders navigation in the WNBA subnav.

**Architecture:** Backend proxies `leaguedashplayerstats` (PerGame, LeagueID=10), normalizes into six ranked categories, caches 10 minutes. Frontend page under `HomeChromeLayout` renders subnav + season label + card grid + attribution; team abbrev colors come from a hardcoded map.

**Tech Stack:** FastAPI · Pydantic · httpx · pytest · React 19 · TypeScript · Vite · TanStack Query · React Router · Vitest · Testing Library · Tailwind 4

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-30-wnba-leaders-design.md`
- Coding standards: `CLAUDE.md` (small focused modules, strong typing, tests with code)
- Brand: **HoopVista**; subnav label remains “HoopVista Picks”
- Route: `/wnba/leaders` only; no `/nba/leaders`
- Data: stats.wnba.com only (no ESPN merge)
- Categories (fixed order): Points, Rebounds, Assists, Steals, Blocks, 3-Pointers — top 10, per-game
- Page chrome: **no** `LeagueHero`
- Attribution copy: exact string `Data: stats.wnba.com`
- HTTP responses: `Cache-Control: no-store`; freshness via in-process 10-minute TTL
- Verify backend: `cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor && PYTHONPATH=backend python3 -m pytest backend/tests/test_wnba_leaders_normalize.py backend/tests/test_wnba_leaders_route.py -v`
- Verify frontend: `cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor/frontend && npm run test -- --run src/components/league src/pages/LeagueLeadersPage.test.tsx src/AppRouter.test.tsx && npm run build`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `backend/app/schemas/wnba_leaders.py` | Response models |
| `backend/app/services/wnba_leaders.py` | Fetch, normalize, cache |
| `backend/app/api/routes/wnba_leaders.py` | `GET /api/wnba/leaders` |
| `backend/app/main.py` | Register router + description |
| `backend/tests/fixtures/stats_wnba_leaguedashplayerstats.json` | Upstream fixture |
| `backend/tests/test_wnba_leaders_normalize.py` | Normalize unit tests |
| `backend/tests/test_wnba_leaders_route.py` | Route + cache + 502 |
| `frontend/src/lib/api.ts` | Types + `fetchWnbaLeaders` |
| `frontend/src/hooks/useWnbaLeaders.ts` | React Query hook |
| `frontend/src/components/league/wnbaTeamColors.ts` | Abbrev → hex map |
| `frontend/src/components/league/LeaderCategoryCard.tsx` | One category table card |
| `frontend/src/components/league/LeadersGrid.tsx` | Season label + grid + attribution |
| `frontend/src/components/league/LeadersGrid.test.tsx` | Grid / loading / error |
| `frontend/src/pages/LeagueLeadersPage.tsx` | Compose page |
| `frontend/src/pages/LeagueLeadersPage.test.tsx` | Page smoke |
| `frontend/src/AppRouter.tsx` | Register `/wnba/leaders` |
| `frontend/src/AppRouter.test.tsx` | Route smoke |
| `frontend/src/components/league/LeagueSubnav.tsx` | Matchups + Leaders links |
| `frontend/src/components/league/LeagueSubnav.test.tsx` | Active + disabled rules |

---

### Task 1: Backend schemas + normalize

**Files:**
- Create: `backend/app/schemas/wnba_leaders.py`
- Create: `backend/app/services/wnba_leaders.py` (normalize + helpers only in this task)
- Create: `backend/tests/fixtures/stats_wnba_leaguedashplayerstats.json`
- Create: `backend/tests/test_wnba_leaders_normalize.py`

**Interfaces:**
- Consumes: stats.wnba.com `resultSets[0]` with `headers` + `rowSet`
- Produces:
  - `normalize_leaguedashplayerstats(payload: dict, *, season: int) -> WnbaLeadersResponse`
  - Models: `WnbaLeaderRow`, `WnbaLeaderCategory`, `WnbaLeadersResponse`

- [ ] **Step 1: Write the fixture**

Create `backend/tests/fixtures/stats_wnba_leaguedashplayerstats.json` with a minimal `resultSets` payload. Include at least 11 players so top-10 truncation is testable, with varied PTS/REB/AST/STL/BLK/FG3M:

```json
{
  "resource": "leaguedashplayerstats",
  "parameters": {
    "LeagueID": "10",
    "PerMode": "PerGame",
    "Season": "2026",
    "SeasonType": "Regular Season"
  },
  "resultSets": [
    {
      "name": "LeagueDashPlayerStats",
      "headers": [
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "GP",
        "PTS",
        "REB",
        "AST",
        "STL",
        "BLK",
        "FG3M"
      ],
      "rowSet": [
        [1001, "A'ja Wilson", 1611661319, "LVA", 25, 26.2, 9.1, 2.4, 1.2, 2.0, 0.4],
        [1002, "Kelsey Mitchell", 1611661313, "IND", 28, 23.5, 2.1, 3.0, 1.0, 0.1, 2.8],
        [1003, "Caitlin Clark", 1611661313, "IND", 24, 21.5, 5.0, 8.0, 1.5, 0.5, 3.0],
        [1004, "Napheesa Collier", 1611661324, "MIN", 26, 22.0, 8.5, 3.2, 1.8, 1.5, 1.2],
        [1005, "Breanna Stewart", 1611661313, "NYL", 22, 20.1, 7.2, 3.5, 1.1, 1.0, 1.5],
        [1006, "Sabrina Ionescu", 1611661313, "NYL", 27, 19.8, 4.0, 6.1, 1.3, 0.3, 3.1],
        [1007, "Alyssa Thomas", 1611661317, "PHO", 25, 14.0, 8.0, 8.3, 1.6, 0.4, 0.1],
        [1008, "Jessica Shepard", 1611661321, "DAL", 24, 12.0, 12.1, 3.0, 0.8, 0.5, 0.2],
        [1009, "Rhyne Howard", 1611661330, "ATL", 26, 16.0, 4.5, 3.8, 2.4, 0.6, 2.5],
        [1010, "Marina Mabrey", 1611661325, "TOR", 25, 15.5, 3.5, 3.2, 1.0, 0.2, 3.3],
        [1011, "Dearica Hamby", 1611661320, "LAS", 27, 17.0, 7.0, 2.5, 1.2, 0.3, 0.8],
        [1012, "Incomplete Row", 1611661319, "LVA", null, null, null, null, null, null, null]
      ]
    }
  ]
}
```

(Adjust TEAM_IDs freely; only abbreviations and stats matter.)

- [ ] **Step 2: Write failing normalize tests**

Create `backend/tests/test_wnba_leaders_normalize.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from app.services.wnba_leaders import normalize_leaguedashplayerstats

FIXTURES = Path(__file__).parent / "fixtures"


def _payload():
    return json.loads(
        (FIXTURES / "stats_wnba_leaguedashplayerstats.json").read_text()
    )


def test_normalize_six_categories_top_ten_order():
    result = normalize_leaguedashplayerstats(_payload(), season=2026)
    assert result.season == 2026
    assert result.pace == "per_game"
    keys = [c.key for c in result.categories]
    assert keys == [
        "points",
        "rebounds",
        "assists",
        "steals",
        "blocks",
        "three_pointers",
    ]
    assert [c.stat for c in result.categories] == [
        "PTS",
        "REB",
        "AST",
        "STL",
        "BLK",
        "3PM",
    ]
    for cat in result.categories:
        assert 1 <= len(cat.leaders) <= 10
        assert [r.rank for r in cat.leaders] == list(
            range(1, len(cat.leaders) + 1)
        )


def test_normalize_points_leader_and_truncation():
    result = normalize_leaguedashplayerstats(_payload(), season=2026)
    points = result.categories[0]
    assert points.leaders[0].name == "A'ja Wilson"
    assert points.leaders[0].team_abbrev == "LVA"
    assert points.leaders[0].gp == 25
    assert points.leaders[0].value == "26.2"
    assert points.leaders[0].player_id == "1001"
    assert len(points.leaders) == 10


def test_normalize_skips_incomplete_rows():
    result = normalize_leaguedashplayerstats(_payload(), season=2026)
    names = {
        row.name
        for cat in result.categories
        for row in cat.leaders
    }
    assert "Incomplete Row" not in names


def test_normalize_empty_result_set():
    empty = {
        "resultSets": [
            {"name": "LeagueDashPlayerStats", "headers": [], "rowSet": []}
        ]
    }
    result = normalize_leaguedashplayerstats(empty, season=2026)
    assert len(result.categories) == 6
    for cat in result.categories:
        assert cat.leaders == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor && PYTHONPATH=backend python3 -m pytest backend/tests/test_wnba_leaders_normalize.py -v
```

Expected: FAIL (module / function missing).

- [ ] **Step 4: Implement schemas + normalize**

Create `backend/app/schemas/wnba_leaders.py`:

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

LeaderCategoryKey = Literal[
    "points",
    "rebounds",
    "assists",
    "steals",
    "blocks",
    "three_pointers",
]


class WnbaLeaderRow(BaseModel):
    rank: int
    player_id: str
    name: str
    team_abbrev: str
    gp: int
    value: str


class WnbaLeaderCategory(BaseModel):
    key: LeaderCategoryKey
    label: str
    stat: str
    leaders: list[WnbaLeaderRow]


class WnbaLeadersResponse(BaseModel):
    season: int
    pace: Literal["per_game"] = "per_game"
    categories: list[WnbaLeaderCategory]
```

Create `backend/app/services/wnba_leaders.py` with normalize (fetch/cache come in Task 2):

```python
from __future__ import annotations

import logging
from typing import Any

from app.schemas.wnba_leaders import (
    WnbaLeaderCategory,
    WnbaLeaderRow,
    WnbaLeadersResponse,
)

logger = logging.getLogger(__name__)

_CATEGORY_SPECS: list[tuple[str, str, str, str]] = [
    # key, label, display_stat, upstream_header
    ("points", "Points", "PTS", "PTS"),
    ("rebounds", "Rebounds", "REB", "REB"),
    ("assists", "Assists", "AST", "AST"),
    ("steals", "Steals", "STL", "STL"),
    ("blocks", "Blocks", "BLK", "BLK"),
    ("three_pointers", "3-Pointers", "3PM", "FG3M"),
]

TOP_N = 10


def _rows_as_dicts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    sets = payload.get("resultSets") or []
    if not sets:
        return []
    block = sets[0] or {}
    headers = [str(h) for h in (block.get("headers") or [])]
    if not headers:
        return []
    out: list[dict[str, Any]] = []
    for raw in block.get("rowSet") or []:
        if not isinstance(raw, (list, tuple)):
            continue
        out.append({headers[i]: raw[i] for i in range(min(len(headers), len(raw)))})
    return out


def _format_value(raw: Any) -> str | None:
    try:
        num = float(raw)
    except (TypeError, ValueError):
        return None
    return f"{num:.1f}"


def _leader_row(rank: int, player: dict[str, Any], value: str) -> WnbaLeaderRow | None:
    player_id = player.get("PLAYER_ID")
    name = str(player.get("PLAYER_NAME") or "").strip()
    abbrev = str(player.get("TEAM_ABBREVIATION") or "").strip().upper()
    gp_raw = player.get("GP")
    try:
        gp = int(gp_raw)
    except (TypeError, ValueError):
        return None
    if player_id is None or not name or not abbrev:
        return None
    return WnbaLeaderRow(
        rank=rank,
        player_id=str(player_id),
        name=name,
        team_abbrev=abbrev,
        gp=gp,
        value=value,
    )


def normalize_leaguedashplayerstats(
    payload: dict[str, Any], *, season: int
) -> WnbaLeadersResponse:
    players = _rows_as_dicts(payload)
    categories: list[WnbaLeaderCategory] = []
    for key, label, stat, header in _CATEGORY_SPECS:
        scored: list[tuple[float, dict[str, Any], str]] = []
        for player in players:
            formatted = _format_value(player.get(header))
            if formatted is None:
                continue
            scored.append((float(formatted), player, formatted))
        scored.sort(key=lambda item: item[0], reverse=True)
        leaders: list[WnbaLeaderRow] = []
        for idx, (_num, player, formatted) in enumerate(scored[:TOP_N], start=1):
            row = _leader_row(idx, player, formatted)
            if row is not None:
                leaders.append(row)
        categories.append(
            WnbaLeaderCategory(
                key=key,  # type: ignore[arg-type]
                label=label,
                stat=stat,
                leaders=leaders,
            )
        )
    return WnbaLeadersResponse(season=season, pace="per_game", categories=categories)
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor && PYTHONPATH=backend python3 -m pytest backend/tests/test_wnba_leaders_normalize.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/schemas/wnba_leaders.py backend/app/services/wnba_leaders.py backend/tests/fixtures/stats_wnba_leaguedashplayerstats.json backend/tests/test_wnba_leaders_normalize.py
git commit -m "$(cat <<'EOF'
feat: normalize WNBA league leaders from stats.wnba.com

EOF
)"
```

---

### Task 2: Backend fetch, cache, and route

**Files:**
- Modify: `backend/app/services/wnba_leaders.py`
- Create: `backend/app/api/routes/wnba_leaders.py`
- Modify: `backend/app/main.py`
- Create: `backend/tests/test_wnba_leaders_route.py`

**Interfaces:**
- Consumes: `normalize_leaguedashplayerstats`
- Produces:
  - `async def get_wnba_leaders() -> WnbaLeadersResponse`
  - `GET /api/wnba/leaders`
  - Module cache `_cache` with keys `response`, `expires_at`, `season`

- [ ] **Step 1: Write failing route tests**

Create `backend/tests/test_wnba_leaders_route.py`:

```python
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import wnba_leaders as svc

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def clear_cache():
    svc._cache.clear()
    yield
    svc._cache.clear()


def test_leaders_returns_no_store_and_categories():
    payload = json.loads(
        (FIXTURES / "stats_wnba_leaguedashplayerstats.json").read_text()
    )

    async def fake_fetch(season: int):
        return payload

    with patch.object(svc, "fetch_leaguedashplayerstats", side_effect=fake_fetch):
        client = TestClient(app)
        res = client.get("/api/wnba/leaders")
    assert res.status_code == 200
    assert res.headers.get("cache-control") == "no-store"
    body = res.json()
    assert body["pace"] == "per_game"
    assert len(body["categories"]) == 6
    assert body["categories"][0]["key"] == "points"
    assert body["categories"][0]["leaders"][0]["name"] == "A'ja Wilson"


def test_leaders_uses_cache_within_ttl():
    payload = json.loads(
        (FIXTURES / "stats_wnba_leaguedashplayerstats.json").read_text()
    )
    calls = {"n": 0}

    async def fake_fetch(season: int):
        calls["n"] += 1
        return payload

    with patch.object(svc, "fetch_leaguedashplayerstats", side_effect=fake_fetch):
        client = TestClient(app)
        assert client.get("/api/wnba/leaders").status_code == 200
        assert client.get("/api/wnba/leaders").status_code == 200
    assert calls["n"] == 1


def test_leaders_stale_while_error():
    payload = json.loads(
        (FIXTURES / "stats_wnba_leaguedashplayerstats.json").read_text()
    )

    async def ok(season: int):
        return payload

    async def boom(season: int):
        raise RuntimeError("upstream down")

    with patch.object(svc, "fetch_leaguedashplayerstats", side_effect=ok):
        client = TestClient(app)
        assert client.get("/api/wnba/leaders").status_code == 200

    svc._cache["expires_at"] = 0

    with patch.object(svc, "fetch_leaguedashplayerstats", side_effect=boom):
        res = client.get("/api/wnba/leaders")
    assert res.status_code == 200
    assert res.json()["categories"][0]["leaders"][0]["name"] == "A'ja Wilson"


def test_leaders_502_no_store_when_cold():
    async def boom(season: int):
        raise RuntimeError("upstream down")

    with patch.object(svc, "fetch_leaguedashplayerstats", side_effect=boom):
        client = TestClient(app)
        res = client.get("/api/wnba/leaders")
    assert res.status_code == 502
    assert res.headers.get("cache-control") == "no-store"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor && PYTHONPATH=backend python3 -m pytest backend/tests/test_wnba_leaders_route.py -v
```

Expected: FAIL (route / fetch missing).

- [ ] **Step 3: Implement fetch, cache, route, register**

Append to `backend/app/services/wnba_leaders.py`:

```python
import asyncio
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

ET = ZoneInfo("America/New_York")
STATS_URL = "https://stats.wnba.com/stats/leaguedashplayerstats"
STATS_TIMEOUT_SECONDS = 10.0
CACHE_TTL_SECONDS = 10 * 60

_cache: dict = {}  # response, expires_at, season
_refresh_lock: asyncio.Lock | None = None
_refresh_lock_loop: asyncio.AbstractEventLoop | None = None


def current_wnba_season_year() -> int:
    return datetime.now(ET).year


def _get_refresh_lock() -> asyncio.Lock:
    global _refresh_lock, _refresh_lock_loop
    loop = asyncio.get_running_loop()
    if _refresh_lock is None or _refresh_lock_loop is not loop:
        _refresh_lock = asyncio.Lock()
        _refresh_lock_loop = loop
    return _refresh_lock


async def fetch_leaguedashplayerstats(season: int) -> dict:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.wnba.com/",
        "Accept": "application/json",
    }
    params = {
        "LastNGames": "0",
        "LeagueID": "10",
        "MeasureType": "Base",
        "Month": "0",
        "OpponentTeamID": "0",
        "PaceAdjust": "N",
        "PerMode": "PerGame",
        "Period": "0",
        "PlusMinus": "N",
        "Rank": "N",
        "Season": str(season),
        "SeasonType": "Regular Season",
        "TeamID": "0",
    }
    async with httpx.AsyncClient(
        timeout=STATS_TIMEOUT_SECONDS, headers=headers
    ) as client:
        res = await client.get(STATS_URL, params=params)
        res.raise_for_status()
        return res.json()


def _fresh_cached() -> WnbaLeadersResponse | None:
    cached = _cache.get("response")
    if cached is None:
        return None
    if _cache.get("season") != current_wnba_season_year():
        return None
    if time.time() >= float(_cache.get("expires_at") or 0):
        return None
    return cached


async def get_wnba_leaders() -> WnbaLeadersResponse:
    fresh = _fresh_cached()
    if fresh is not None:
        return fresh

    lock = _get_refresh_lock()
    async with lock:
        fresh = _fresh_cached()
        if fresh is not None:
            return fresh
        season = current_wnba_season_year()
        try:
            payload = await fetch_leaguedashplayerstats(season)
            response = normalize_leaguedashplayerstats(payload, season=season)
        except Exception:
            stale = _cache.get("response")
            if stale is not None and _cache.get("season") == season:
                logger.warning("WNBA leaders refresh failed; serving stale cache")
                return stale
            raise
        _cache["response"] = response
        _cache["expires_at"] = time.time() + CACHE_TTL_SECONDS
        _cache["season"] = season
        return response
```

Create `backend/app/api/routes/wnba_leaders.py`:

```python
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Response

from app.schemas.wnba_leaders import WnbaLeadersResponse
from app.services.wnba_leaders import get_wnba_leaders

logger = logging.getLogger(__name__)

router = APIRouter(tags=["wnba"])
_NO_STORE = {"Cache-Control": "no-store"}


@router.get("/wnba/leaders", response_model=WnbaLeadersResponse)
async def wnba_leaders(response: Response) -> WnbaLeadersResponse:
    response.headers["Cache-Control"] = "no-store"
    try:
        return await get_wnba_leaders()
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("WNBA leaders unavailable: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="WNBA leaders are temporarily unavailable",
            headers=_NO_STORE,
        ) from exc
```

In `backend/app/main.py`:

- Import `wnba_leaders` next to `wnba_scoreboard`.
- `app.include_router(wnba_leaders.router, prefix="/api")` beside the other WNBA routers.
- Update the FastAPI `description` string to mention `/api/wnba/leaders` as a direct stats.wnba.com call.

- [ ] **Step 4: Run route tests**

Run:

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor && PYTHONPATH=backend python3 -m pytest backend/tests/test_wnba_leaders_route.py backend/tests/test_wnba_leaders_normalize.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/wnba_leaders.py backend/app/api/routes/wnba_leaders.py backend/app/main.py backend/tests/test_wnba_leaders_route.py
git commit -m "$(cat <<'EOF'
feat: add GET /api/wnba/leaders proxy with cache

EOF
)"
```

---

### Task 3: Frontend API client + hook

**Files:**
- Modify: `frontend/src/lib/api.ts`
- Create: `frontend/src/hooks/useWnbaLeaders.ts`
- Create: `frontend/src/hooks/useWnbaLeaders.test.tsx`

**Interfaces:**
- Produces:
  - `ApiWnbaLeadersResponse` (+ category/row types)
  - `fetchWnbaLeaders(): Promise<ApiWnbaLeadersResponse>`
  - `useWnbaLeaders()` → query result with `hasNeverLoaded`

- [ ] **Step 1: Write failing hook test**

Create `frontend/src/hooks/useWnbaLeaders.test.tsx`:

```tsx
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { useWnbaLeaders } from "./useWnbaLeaders";

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
}

describe("useWnbaLeaders", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("loads leaders payload", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        season: 2026,
        pace: "per_game",
        categories: [
          {
            key: "points",
            label: "Points",
            stat: "PTS",
            leaders: [
              {
                rank: 1,
                player_id: "1",
                name: "A'ja Wilson",
                team_abbrev: "LVA",
                gp: 25,
                value: "26.2",
              },
            ],
          },
        ],
      }),
    });

    const { result } = renderHook(() => useWnbaLeaders(), { wrapper });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.categories[0].leaders[0].name).toBe(
      "A'ja Wilson",
    );
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/api/wnba/leaders"),
      expect.any(Object),
    );
  });

  it("sets hasNeverLoaded on cold error", async () => {
    fetchMock.mockResolvedValue({ ok: false, status: 502 });
    const { result } = renderHook(() => useWnbaLeaders(), { wrapper });
    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.hasNeverLoaded).toBe(true);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor/frontend && npm run test -- --run src/hooks/useWnbaLeaders.test.tsx
```

Expected: FAIL (missing module / export).

- [ ] **Step 3: Add API types, fetch, and hook**

Append to `frontend/src/lib/api.ts`:

```ts
export type ApiWnbaLeaderRow = {
  rank: number;
  player_id: string;
  name: string;
  team_abbrev: string;
  gp: number;
  value: string;
};

export type ApiWnbaLeaderCategory = {
  key:
    | "points"
    | "rebounds"
    | "assists"
    | "steals"
    | "blocks"
    | "three_pointers";
  label: string;
  stat: string;
  leaders: ApiWnbaLeaderRow[];
};

export type ApiWnbaLeadersResponse = {
  season: number;
  pace: "per_game";
  categories: ApiWnbaLeaderCategory[];
};

export async function fetchWnbaLeaders(): Promise<ApiWnbaLeadersResponse> {
  const res = await fetch(`${API_BASE}/api/wnba/leaders`, {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`Leaders request failed: ${res.status}`);
  }
  return res.json();
}
```

Create `frontend/src/hooks/useWnbaLeaders.ts`:

```ts
import { useQuery } from "@tanstack/react-query";
import { fetchWnbaLeaders } from "@/lib/api";

export function useWnbaLeaders() {
  const query = useQuery({
    queryKey: ["wnba", "leaders"],
    queryFn: fetchWnbaLeaders,
  });

  return {
    ...query,
    hasNeverLoaded: query.isError && query.data === undefined,
  };
}
```

- [ ] **Step 4: Run hook test**

Run:

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor/frontend && npm run test -- --run src/hooks/useWnbaLeaders.test.tsx
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/api.ts frontend/src/hooks/useWnbaLeaders.ts frontend/src/hooks/useWnbaLeaders.test.tsx
git commit -m "$(cat <<'EOF'
feat: add WNBA leaders API client and hook

EOF
)"
```

---

### Task 4: Team colors + category card + leaders grid

**Files:**
- Create: `frontend/src/components/league/wnbaTeamColors.ts`
- Create: `frontend/src/components/league/LeaderCategoryCard.tsx`
- Create: `frontend/src/components/league/LeadersGrid.tsx`
- Create: `frontend/src/components/league/LeadersGrid.test.tsx`

**Interfaces:**
- Consumes: `ApiWnbaLeadersResponse` / category + row shapes
- Produces:
  - `teamColor(abbrev: string): string`
  - `LeaderCategoryCard({ category })`
  - `LeadersGrid({ season, categories, isLoading?, isError? })`

- [ ] **Step 1: Write failing grid tests**

Create `frontend/src/components/league/LeadersGrid.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { LeadersGrid } from "./LeadersGrid";
import type { ApiWnbaLeaderCategory } from "@/lib/api";

const categories: ApiWnbaLeaderCategory[] = [
  {
    key: "points",
    label: "Points",
    stat: "PTS",
    leaders: [
      {
        rank: 1,
        player_id: "1",
        name: "A'ja Wilson",
        team_abbrev: "LVA",
        gp: 25,
        value: "26.2",
      },
    ],
  },
  {
    key: "rebounds",
    label: "Rebounds",
    stat: "REB",
    leaders: [],
  },
];

describe("LeadersGrid", () => {
  it("renders season label, cards, colors, and attribution", () => {
    render(
      <LeadersGrid season={2026} categories={categories} />,
    );
    expect(screen.getByText("2026 season · per game")).toBeInTheDocument();
    expect(screen.getByText("Points")).toBeInTheDocument();
    expect(screen.getByText("A'ja Wilson")).toBeInTheDocument();
    expect(screen.getByText("LVA")).toBeInTheDocument();
    expect(screen.getByText("26.2")).toBeInTheDocument();
    expect(screen.getByText("No data")).toBeInTheDocument();
    expect(screen.getByText("Data: stats.wnba.com")).toBeInTheDocument();
  });

  it("shows loading skeletons", () => {
    render(<LeadersGrid season={2026} categories={[]} isLoading />);
    expect(screen.getByLabelText(/loading leaders/i)).toBeInTheDocument();
  });

  it("shows error copy when never loaded", () => {
    render(
      <LeadersGrid season={2026} categories={[]} isError />,
    );
    expect(screen.getByText(/leaders unavailable/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor/frontend && npm run test -- --run src/components/league/LeadersGrid.test.tsx
```

Expected: FAIL (missing module).

- [ ] **Step 3: Implement colors, card, and grid**

Create `frontend/src/components/league/wnbaTeamColors.ts`:

```ts
/** Brand colors for stats.wnba.com tricodes on dark UI. */
const WNBA_TEAM_COLORS: Record<string, string> = {
  ATL: "#C8102E",
  CHI: "#4E8FD0",
  CON: "#FC4C02",
  DAL: "#C4D600",
  GSV: "#FFC72C",
  IND: "#FFCD00",
  LAS: "#552583",
  LVA: "#C8102E",
  MIN: "#236192",
  NYL: "#6ECEB2",
  PHO: "#E56020",
  SEA: "#2C5234",
  TOR: "#B4975A",
  WAS: "#E31837",
};

const FALLBACK = "rgba(255,255,255,0.5)";

export function teamColor(abbrev: string): string {
  const key = abbrev.trim().toUpperCase();
  return WNBA_TEAM_COLORS[key] ?? FALLBACK;
}
```

Create `frontend/src/components/league/LeaderCategoryCard.tsx`:

```tsx
import type { ApiWnbaLeaderCategory } from "@/lib/api";
import { teamColor } from "./wnbaTeamColors";

type LeaderCategoryCardProps = {
  category: ApiWnbaLeaderCategory;
};

export function LeaderCategoryCard({ category }: LeaderCategoryCardProps) {
  return (
    <section className="rounded-xl border border-white/10 bg-[#141414] p-4">
      <h3 className="mb-3 text-base font-semibold text-white">
        {category.label}
      </h3>
      <table className="w-full text-left text-sm">
        <thead>
          <tr className="text-[11px] tracking-wide text-white/40 uppercase">
            <th className="pb-2 font-medium">#</th>
            <th className="pb-2 font-medium">Player</th>
            <th className="pb-2 font-medium">Team</th>
            <th className="pb-2 text-right font-medium">GP</th>
            <th className="pb-2 text-right font-medium">{category.stat}</th>
          </tr>
        </thead>
        <tbody>
          {category.leaders.length === 0 ? (
            <tr>
              <td colSpan={5} className="py-3 text-white/40">
                No data
              </td>
            </tr>
          ) : (
            category.leaders.map((row) => (
              <tr key={`${category.key}-${row.rank}-${row.player_id}`}>
                <td className="py-1.5 text-white/50">{row.rank}</td>
                <td className="py-1.5 text-white">{row.name}</td>
                <td
                  className="py-1.5 font-semibold"
                  style={{ color: teamColor(row.team_abbrev) }}
                >
                  {row.team_abbrev}
                </td>
                <td className="py-1.5 text-right text-white/50">{row.gp}</td>
                <td className="py-1.5 text-right font-semibold text-white">
                  {row.value}
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </section>
  );
}
```

Create `frontend/src/components/league/LeadersGrid.tsx`:

```tsx
import type { ApiWnbaLeaderCategory } from "@/lib/api";
import { LeaderCategoryCard } from "./LeaderCategoryCard";

type LeadersGridProps = {
  season: number;
  categories: ApiWnbaLeaderCategory[];
  isLoading?: boolean;
  isError?: boolean;
};

function Skeletons() {
  return (
    <div
      className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3"
      aria-label="Loading leaders"
    >
      {Array.from({ length: 6 }, (_, i) => (
        <div
          key={i}
          className="h-72 animate-pulse rounded-xl border border-white/10 bg-white/5"
        />
      ))}
    </div>
  );
}

export function LeadersGrid({
  season,
  categories,
  isLoading = false,
  isError = false,
}: LeadersGridProps) {
  return (
    <section className="mx-auto max-w-6xl space-y-4 px-4 sm:px-6">
      <p className="text-sm text-white/45">
        {season} season · per game
      </p>
      {isLoading ? (
        <Skeletons />
      ) : isError ? (
        <p className="text-sm text-white/50">Leaders unavailable</p>
      ) : (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {categories.map((category) => (
            <LeaderCategoryCard key={category.key} category={category} />
          ))}
        </div>
      )}
      <p className="text-xs text-white/35">Data: stats.wnba.com</p>
    </section>
  );
}
```

- [ ] **Step 4: Run grid tests**

Run:

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor/frontend && npm run test -- --run src/components/league/LeadersGrid.test.tsx
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/league/wnbaTeamColors.ts frontend/src/components/league/LeaderCategoryCard.tsx frontend/src/components/league/LeadersGrid.tsx frontend/src/components/league/LeadersGrid.test.tsx
git commit -m "$(cat <<'EOF'
feat: add WNBA leaders grid UI

EOF
)"
```

---

### Task 5: Page, router, and subnav

**Files:**
- Create: `frontend/src/pages/LeagueLeadersPage.tsx`
- Create: `frontend/src/pages/LeagueLeadersPage.test.tsx`
- Modify: `frontend/src/AppRouter.tsx`
- Modify: `frontend/src/AppRouter.test.tsx`
- Modify: `frontend/src/components/league/LeagueSubnav.tsx`
- Modify: `frontend/src/components/league/LeagueSubnav.test.tsx`

**Interfaces:**
- Consumes: `useWnbaLeaders`, `LeadersGrid`, `LeagueSubnav`
- Produces: `/wnba/leaders` route; subnav Links for Matchups + Leaders (WNBA)

- [ ] **Step 1: Write failing page / router / subnav tests**

Create `frontend/src/pages/LeagueLeadersPage.test.tsx`:

```tsx
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { LeagueLeadersPage } from "./LeagueLeadersPage";

function renderPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={["/wnba/leaders"]}>
        <LeagueLeadersPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("LeagueLeadersPage", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders leaders from API", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        season: 2026,
        pace: "per_game",
        categories: [
          {
            key: "points",
            label: "Points",
            stat: "PTS",
            leaders: [
              {
                rank: 1,
                player_id: "1",
                name: "A'ja Wilson",
                team_abbrev: "LVA",
                gp: 25,
                value: "26.2",
              },
            ],
          },
        ],
      }),
    });

    renderPage();
    expect(await screen.findByText("Points")).toBeInTheDocument();
    expect(screen.getByText("A'ja Wilson")).toBeInTheDocument();
    expect(screen.getByText("Data: stats.wnba.com")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Leaders" })).toHaveAttribute(
      "aria-current",
      "page",
    );
  });
});
```

Add to `frontend/src/AppRouter.test.tsx` (keep existing scoreboard mock; extend fetch mock to also answer `/api/wnba/leaders` when URL includes `leaders`):

```tsx
it("renders WNBA leaders at /wnba/leaders", async () => {
  fetchMock.mockImplementation(async (input: RequestInfo) => {
    const url = String(input);
    if (url.includes("/api/wnba/leaders")) {
      return {
        ok: true,
        json: async () => ({
          season: 2026,
          pace: "per_game",
          categories: [],
        }),
      };
    }
    return {
      ok: true,
      json: async () => ({ date: "2026-07-29", fetched_at: "", games: [] }),
    };
  });
  renderWithProviders(["/wnba/leaders"]);
  expect(await screen.findByText(/2026 season · per game/i)).toBeInTheDocument();
  expect(screen.getByText("Data: stats.wnba.com")).toBeInTheDocument();
});
```

Replace `LeagueSubnav.test.tsx` contents with:

```tsx
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { LeagueSubnav } from "./LeagueSubnav";

describe("LeagueSubnav", () => {
  it("links Matchups and Leaders on WNBA; disables others", () => {
    render(
      <MemoryRouter initialEntries={["/wnba/leaders"]}>
        <LeagueSubnav league="wnba" />
      </MemoryRouter>,
    );
    const leaders = screen.getByRole("link", { name: "Leaders" });
    expect(leaders).toHaveAttribute("href", "/wnba/leaders");
    expect(leaders).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("link", { name: "Matchups" })).toHaveAttribute(
      "href",
      "/wnba/matchups",
    );
    expect(
      screen.getByRole("button", { name: "HoopVista Picks" }),
    ).toBeDisabled();
  });

  it("keeps Leaders disabled on NBA", () => {
    render(
      <MemoryRouter initialEntries={["/nba/matchups"]}>
        <LeagueSubnav league="nba" />
      </MemoryRouter>,
    );
    expect(screen.getByRole("button", { name: "Leaders" })).toBeDisabled();
    expect(screen.getByRole("link", { name: "Matchups" })).toHaveAttribute(
      "href",
      "/nba/matchups",
    );
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor/frontend && npm run test -- --run src/pages/LeagueLeadersPage.test.tsx src/components/league/LeagueSubnav.test.tsx src/AppRouter.test.tsx
```

Expected: FAIL on new assertions / missing page.

- [ ] **Step 3: Implement page, route, and subnav**

Create `frontend/src/pages/LeagueLeadersPage.tsx`:

```tsx
import { LeagueSubnav } from "@/components/league/LeagueSubnav";
import { LeadersGrid } from "@/components/league/LeadersGrid";
import { useWnbaLeaders } from "@/hooks/useWnbaLeaders";

export function LeagueLeadersPage() {
  const { data, isLoading, hasNeverLoaded } = useWnbaLeaders();
  const season = data?.season ?? new Date().getFullYear();

  return (
    <div className="space-y-6 py-6">
      <LeagueSubnav league="wnba" />
      <LeadersGrid
        season={season}
        categories={data?.categories ?? []}
        isLoading={isLoading && !data}
        isError={hasNeverLoaded}
      />
    </div>
  );
}
```

In `frontend/src/AppRouter.tsx`, import `LeagueLeadersPage` and add:

```tsx
<Route path="/wnba/leaders" element={<LeagueLeadersPage />} />
```

(under `HomeChromeLayout`, next to matchups routes).

Replace `frontend/src/components/league/LeagueSubnav.tsx` with:

```tsx
import { Link, useLocation } from "react-router-dom";
import type { LeagueSlug } from "./types";

type LeagueSubnavProps = {
  league: LeagueSlug;
};

const exploreItems = [
  "Matchups",
  "HoopVista Picks",
  "Leaders",
  "Standings",
  "Playoff race",
  "Clutch",
] as const;

const learnItems = ["How it works", "Glossary"] as const;

export function LeagueSubnav({ league }: LeagueSubnavProps) {
  const { pathname } = useLocation();
  const activeClassName =
    league === "wnba"
      ? "bg-violet-600 text-white"
      : "bg-sky-600 text-white";

  function itemPath(item: string): string | null {
    if (item === "Matchups") return `/${league}/matchups`;
    if (item === "Leaders" && league === "wnba") return "/wnba/leaders";
    return null;
  }

  function isActive(item: string): boolean {
    if (item === "Matchups") return pathname.endsWith("/matchups");
    if (item === "Leaders") return pathname.endsWith("/leaders");
    return false;
  }

  function renderItem(item: string) {
    const href = itemPath(item);
    const active = isActive(item);
    const className = active
      ? `rounded-full px-4 py-2 text-sm font-semibold ${activeClassName}`
      : href
        ? "rounded-full border border-white/10 bg-white/[0.03] px-4 py-2 text-sm font-medium text-white/70 hover:text-white"
        : "cursor-not-allowed rounded-full border border-white/10 bg-white/[0.03] px-4 py-2 text-sm font-medium text-white/35";

    if (href) {
      return (
        <Link
          key={item}
          to={href}
          aria-current={active ? "page" : undefined}
          className={className}
        >
          {item}
        </Link>
      );
    }

    return (
      <button
        key={item}
        type="button"
        disabled
        className={className}
      >
        {item}
      </button>
    );
  }

  return (
    <nav
      aria-label={`${league.toUpperCase()} sections`}
      className="mx-auto max-w-6xl px-4 py-5 sm:px-6"
    >
      <div className="flex gap-6 overflow-x-auto rounded-2xl border border-white/10 bg-[#121212] px-4 py-3">
        <div className="shrink-0">
          <p className="mb-2 px-1 text-[10px] font-semibold tracking-[0.18em] text-white/35 uppercase">
            Explore
          </p>
          <div className="flex gap-2">{exploreItems.map(renderItem)}</div>
        </div>
        <div className="shrink-0">
          <p className="mb-2 px-1 text-[10px] font-semibold tracking-[0.18em] text-white/35 uppercase">
            Learn
          </p>
          <div className="flex gap-2">{learnItems.map(renderItem)}</div>
        </div>
      </div>
    </nav>
  );
}
```

- [ ] **Step 4: Run frontend tests + build**

Run:

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor/frontend && npm run test -- --run src/pages/LeagueLeadersPage.test.tsx src/components/league/LeagueSubnav.test.tsx src/components/league/LeadersGrid.test.tsx src/hooks/useWnbaLeaders.test.tsx src/AppRouter.test.tsx && npm run build
```

Expected: PASS / build succeeds.

Also re-run backend:

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor && PYTHONPATH=backend python3 -m pytest backend/tests/test_wnba_leaders_normalize.py backend/tests/test_wnba_leaders_route.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/LeagueLeadersPage.tsx frontend/src/pages/LeagueLeadersPage.test.tsx frontend/src/AppRouter.tsx frontend/src/AppRouter.test.tsx frontend/src/components/league/LeagueSubnav.tsx frontend/src/components/league/LeagueSubnav.test.tsx
git commit -m "$(cat <<'EOF'
feat: ship /wnba/leaders page and subnav links

EOF
)"
```

---

## Spec coverage checklist

| Spec requirement | Task |
| --- | --- |
| `/wnba/leaders` under HomeChromeLayout | 5 |
| No LeagueHero | 5 |
| Subnav Matchups + Leaders links; others disabled | 5 |
| NBA Leaders stays disabled | 5 |
| Six categories top 10 per-game | 1 |
| `GET /api/wnba/leaders` from stats.wnba.com | 2 |
| 10-minute cache + stale-while-error + 502 cold | 2 |
| Season label + attribution | 4 |
| Hardcoded team colors | 4 |
| Loading / error / empty category | 4 |
| React Query hook | 3 |
| No ESPN merge / no NBA route | out of scope (enforced) |

## Plan self-review notes

- No TBD/placeholder steps; concrete code and commands included.
- Types consistent: `WnbaLeadersResponse` ↔ `ApiWnbaLeadersResponse`; category keys match.
- Fixture TEAM_IDs are illustrative; normalize ignores them.
- If live `leaguedashplayerstats` uses a different three-pointer header than `FG3M`, adjust `_CATEGORY_SPECS` only after confirming against a live payload (keep fixture + tests in sync).
