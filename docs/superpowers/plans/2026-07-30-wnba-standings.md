# WNBA Standings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `/wnba/standings` with boxseats-style Eastern/Western Conference tables (full columns) fed by `GET /api/wnba/standings` from ESPN, and enable Standings navigation in the WNBA subnav.

**Architecture:** Backend proxies ESPN `site.api` v2 WNBA standings, normalizes into East/West rows with display strings, caches 10 minutes. Frontend page under `HomeChromeLayout` renders subnav + season label + two conference cards + attribution; abbrev colors from existing `wnbaTeamColors`; logos from ESPN URLs.

**Tech Stack:** FastAPI · Pydantic · httpx · pytest · React 19 · TypeScript · Vite · TanStack Query · React Router · Vitest · Testing Library · Tailwind 4

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-30-wnba-standings-design.md`
- Coding standards: `CLAUDE.md` (small focused modules, strong typing, defensive mapping, tests with code, focused commits)
- Brand: **HoopVista**; subnav label remains “HoopVista Picks”
- Route: `/wnba/standings` only; no `/nba/standings`
- Data: ESPN standings only (`site.api.espn.com/apis/v2/sports/basketball/wnba/standings`)
- Columns (fixed): `#` · `TEAM` · `W-L` · `PCT` · `GB` · `Home` · `Away` · `L10` · `Diff` · `Strk`
- Layout: 2-col desktop / stacked mobile (East above West)
- Page chrome: **no** `LeagueHero`
- Attribution copy: exact string `Data: ESPN`
- HTTP responses: `Cache-Control: no-store`; freshness via in-process 10-minute TTL
- Verify backend: `cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor && PYTHONPATH=backend python3 -m pytest backend/tests/test_wnba_standings_normalize.py backend/tests/test_wnba_standings_route.py -v`
- Verify frontend: `cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor/frontend && npm run test -- --run src/components/league src/pages/LeagueStandingsPage.test.tsx src/AppRouter.test.tsx && npm run build`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `backend/app/schemas/wnba_standings.py` | Response models |
| `backend/app/services/wnba_standings.py` | Fetch, normalize, cache |
| `backend/app/api/routes/wnba_standings.py` | `GET /api/wnba/standings` |
| `backend/app/main.py` | Register router + description |
| `backend/tests/fixtures/espn_wnba_standings.json` | Upstream fixture |
| `backend/tests/test_wnba_standings_normalize.py` | Normalize unit tests |
| `backend/tests/test_wnba_standings_route.py` | Route + cache + 502 |
| `frontend/src/lib/api.ts` | Types + `fetchWnbaStandings` |
| `frontend/src/hooks/useWnbaStandings.ts` | React Query hook |
| `frontend/src/components/league/StandingsConferenceCard.tsx` | One conference table card |
| `frontend/src/components/league/StandingsGrid.tsx` | Season label + grid + attribution |
| `frontend/src/components/league/StandingsGrid.test.tsx` | Grid / loading / error |
| `frontend/src/pages/LeagueStandingsPage.tsx` | Compose page |
| `frontend/src/pages/LeagueStandingsPage.test.tsx` | Page smoke |
| `frontend/src/AppRouter.tsx` | Register `/wnba/standings` |
| `frontend/src/AppRouter.test.tsx` | Route smoke |
| `frontend/src/components/league/LeagueSubnav.tsx` | + Standings link |
| `frontend/src/components/league/LeagueSubnav.test.tsx` | Active + disabled rules |

---

### Task 1: Backend schemas + normalize

**Files:**
- Create: `backend/app/schemas/wnba_standings.py`
- Create: `backend/app/services/wnba_standings.py` (normalize + helpers only in this task)
- Create: `backend/tests/fixtures/espn_wnba_standings.json`
- Create: `backend/tests/test_wnba_standings_normalize.py`

**Interfaces:**
- Consumes: ESPN standings payload with `season.year` and `children[]` conferences (`standings.entries[]` of `team` + `stats`)
- Produces:
  - `normalize_espn_standings(payload: dict) -> WnbaStandingsResponse`
  - Models: `WnbaStandingsRow`, `WnbaStandingsConference`, `WnbaStandingsResponse`

- [ ] **Step 1: Write the fixture**

Create `backend/tests/fixtures/espn_wnba_standings.json` with two conferences and enough entries to assert ordering + skip-incomplete. Include full `stats` arrays using ESPN `name` / `displayValue` fields:

```json
{
  "season": {
    "year": 2026,
    "displayName": "2026"
  },
  "children": [
    {
      "id": "1",
      "name": "Eastern Conference",
      "abbreviation": "E",
      "standings": {
        "entries": [
          {
            "team": {
              "id": "5",
              "abbreviation": "IND",
              "displayName": "Indiana Fever",
              "logos": [
                {
                  "href": "https://a.espncdn.com/i/teamlogos/wnba/500/ind.png"
                }
              ]
            },
            "stats": [
              {"name": "playoffSeed", "displayValue": "1"},
              {"name": "wins", "displayValue": "18", "value": 18},
              {"name": "losses", "displayValue": "10", "value": 10},
              {"name": "overall", "displayValue": "18-10"},
              {"name": "winPercent", "displayValue": ".643"},
              {"name": "gamesBehind", "displayValue": "-"},
              {"name": "Home", "displayValue": "11-5"},
              {"name": "Road", "displayValue": "7-5"},
              {"name": "Last Ten Games", "displayValue": "8-2"},
              {"name": "pointDifferential", "displayValue": "+169"},
              {"name": "differential", "displayValue": "+6.0"},
              {"name": "streak", "displayValue": "W4"}
            ]
          },
          {
            "team": {
              "id": "20",
              "abbreviation": "ATL",
              "displayName": "Atlanta Dream",
              "logos": [
                {
                  "href": "https://a.espncdn.com/i/teamlogos/wnba/500/atl.png"
                }
              ]
            },
            "stats": [
              {"name": "playoffSeed", "displayValue": "2"},
              {"name": "wins", "displayValue": "17", "value": 17},
              {"name": "losses", "displayValue": "10", "value": 10},
              {"name": "overall", "displayValue": "17-10"},
              {"name": "winPercent", "displayValue": ".630"},
              {"name": "gamesBehind", "displayValue": "0.5"},
              {"name": "Home", "displayValue": "8-5"},
              {"name": "Road", "displayValue": "9-5"},
              {"name": "Last Ten Games", "displayValue": "6-4"},
              {"name": "pointDifferential", "displayValue": "+91"},
              {"name": "differential", "displayValue": "+3.4"},
              {"name": "streak", "displayValue": "L1"}
            ]
          },
          {
            "team": {
              "id": "999",
              "abbreviation": "",
              "displayName": "Broken Entry"
            },
            "stats": []
          }
        ]
      }
    },
    {
      "id": "2",
      "name": "Western Conference",
      "abbreviation": "W",
      "standings": {
        "entries": [
          {
            "team": {
              "id": "16",
              "abbreviation": "MIN",
              "displayName": "Minnesota Lynx",
              "logos": [
                {
                  "href": "https://a.espncdn.com/i/teamlogos/wnba/500/min.png"
                }
              ]
            },
            "stats": [
              {"name": "playoffSeed", "displayValue": "1"},
              {"name": "wins", "displayValue": "24", "value": 24},
              {"name": "losses", "displayValue": "6", "value": 6},
              {"name": "overall", "displayValue": "24-6"},
              {"name": "winPercent", "displayValue": ".800"},
              {"name": "gamesBehind", "displayValue": "-"},
              {"name": "Home", "displayValue": "13-2"},
              {"name": "Road", "displayValue": "11-4"},
              {"name": "Last Ten Games", "displayValue": "8-2"},
              {"name": "pointDifferential", "displayValue": "+312"},
              {"name": "differential", "displayValue": "+10.4"},
              {"name": "streak", "displayValue": "W3"}
            ]
          },
          {
            "team": {
              "id": "17",
              "abbreviation": "LV",
              "displayName": "Las Vegas Aces",
              "logos": []
            },
            "stats": [
              {"name": "playoffSeed", "displayValue": "2"},
              {"name": "wins", "displayValue": "19", "value": 19},
              {"name": "losses", "displayValue": "8", "value": 8},
              {"name": "overall", "displayValue": "19-8"},
              {"name": "winPercent", "displayValue": ".704"},
              {"name": "gamesBehind", "displayValue": "3.5"},
              {"name": "Home", "displayValue": "10-3"},
              {"name": "Road", "displayValue": "9-5"},
              {"name": "Last Ten Games", "displayValue": "7-3"},
              {"name": "pointDifferential", "displayValue": "+140"},
              {"name": "differential", "displayValue": "+5.2"},
              {"name": "streak", "displayValue": "W1"}
            ]
          }
        ]
      }
    }
  ]
}
```

- [ ] **Step 2: Write failing normalize tests**

Create `backend/tests/test_wnba_standings_normalize.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from app.services.wnba_standings import normalize_espn_standings

FIXTURES = Path(__file__).parent / "fixtures"


def _payload():
    return json.loads((FIXTURES / "espn_wnba_standings.json").read_text())


def test_normalize_east_then_west_order_and_season():
    result = normalize_espn_standings(_payload())
    assert result.season == 2026
    assert [c.key for c in result.conferences] == ["east", "west"]
    assert [c.label for c in result.conferences] == [
        "Eastern Conference",
        "Western Conference",
    ]
    assert len(result.conferences[0].teams) == 2
    assert len(result.conferences[1].teams) == 2


def test_normalize_maps_row_fields():
    east = normalize_espn_standings(_payload()).conferences[0]
    ind = east.teams[0]
    assert ind.rank == 1
    assert ind.team_id == "5"
    assert ind.abbrev == "IND"
    assert ind.name == "Indiana Fever"
    assert ind.logo_url == "https://a.espncdn.com/i/teamlogos/wnba/500/ind.png"
    assert ind.wins == 18
    assert ind.losses == 10
    assert ind.wl == "18-10"
    assert ind.pct == ".643"
    assert ind.gb == "-"
    assert ind.home == "11-5"
    assert ind.away == "7-5"
    assert ind.l10 == "8-2"
    assert ind.diff == "+169"
    assert ind.streak == "W4"


def test_normalize_skips_incomplete_and_null_logo():
    result = normalize_espn_standings(_payload())
    east_names = [t.name for t in result.conferences[0].teams]
    assert "Broken Entry" not in east_names
    lv = result.conferences[1].teams[1]
    assert lv.abbrev == "LV"
    assert lv.logo_url is None


def test_normalize_empty_children():
    result = normalize_espn_standings({"season": {"year": 2026}, "children": []})
    assert result.season == 2026
    assert result.conferences == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor && PYTHONPATH=backend python3 -m pytest backend/tests/test_wnba_standings_normalize.py -v
```

Expected: FAIL (import error / module not found for `app.services.wnba_standings`)

- [ ] **Step 4: Implement schemas**

Create `backend/app/schemas/wnba_standings.py`:

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

ConferenceKey = Literal["east", "west"]


class WnbaStandingsRow(BaseModel):
    rank: int
    team_id: str
    abbrev: str
    name: str
    logo_url: str | None
    wins: int
    losses: int
    wl: str
    pct: str
    gb: str
    home: str
    away: str
    l10: str
    diff: str
    streak: str


class WnbaStandingsConference(BaseModel):
    key: ConferenceKey
    label: str
    teams: list[WnbaStandingsRow]


class WnbaStandingsResponse(BaseModel):
    season: int
    conferences: list[WnbaStandingsConference]
```

- [ ] **Step 5: Implement normalize helpers + `normalize_espn_standings`**

Create `backend/app/services/wnba_standings.py` with normalize only (fetch/cache come in Task 2):

```python
from __future__ import annotations

import logging
from typing import Any

from app.schemas.wnba_standings import (
    ConferenceKey,
    WnbaStandingsConference,
    WnbaStandingsResponse,
    WnbaStandingsRow,
)

logger = logging.getLogger(__name__)

_CONF_BY_ABBREV: dict[str, tuple[ConferenceKey, str]] = {
    "E": ("east", "Eastern Conference"),
    "W": ("west", "Western Conference"),
}


def _stat_map(stats: list[Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for raw in stats or []:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        if name:
            out[name] = raw
    return out


def _display(stats: dict[str, dict[str, Any]], *names: str) -> str | None:
    for name in names:
        block = stats.get(name)
        if not block:
            continue
        value = block.get("displayValue")
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _int_stat(stats: dict[str, dict[str, Any]], name: str) -> int | None:
    block = stats.get(name)
    if not block:
        return None
    raw = block.get("value", block.get("displayValue"))
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def _logo_url(team: dict[str, Any]) -> str | None:
    logos = team.get("logos") or []
    if not isinstance(logos, list):
        return None
    for logo in logos:
        if not isinstance(logo, dict):
            continue
        href = str(logo.get("href") or "").strip()
        if href:
            return href
    return None


def _row_from_entry(entry: dict[str, Any]) -> WnbaStandingsRow | None:
    team = entry.get("team") or {}
    if not isinstance(team, dict):
        return None
    team_id = str(team.get("id") or "").strip()
    abbrev = str(team.get("abbreviation") or "").strip().upper()
    name = str(team.get("displayName") or "").strip()
    if not team_id or not abbrev or not name:
        return None
    stats = _stat_map(entry.get("stats") or [])
    rank = _int_stat(stats, "playoffSeed")
    wins = _int_stat(stats, "wins")
    losses = _int_stat(stats, "losses")
    wl = _display(stats, "overall")
    pct = _display(stats, "winPercent")
    gb = _display(stats, "gamesBehind")
    home = _display(stats, "Home")
    away = _display(stats, "Road")
    l10 = _display(stats, "Last Ten Games")
    diff = _display(stats, "pointDifferential", "differential")
    streak = _display(stats, "streak")
    if (
        rank is None
        or wins is None
        or losses is None
        or not wl
        or not pct
        or not gb
        or not home
        or not away
        or not l10
        or not diff
        or not streak
    ):
        return None
    return WnbaStandingsRow(
        rank=rank,
        team_id=team_id,
        abbrev=abbrev,
        name=name,
        logo_url=_logo_url(team),
        wins=wins,
        losses=losses,
        wl=wl,
        pct=pct,
        gb=gb,
        home=home,
        away=away,
        l10=l10,
        diff=diff,
        streak=streak,
    )


def _season_year(payload: dict[str, Any]) -> int | None:
    season = payload.get("season")
    if isinstance(season, dict):
        year = season.get("year")
        try:
            return int(year)
        except (TypeError, ValueError):
            return None
    if isinstance(season, int):
        return season
    return None


def normalize_espn_standings(payload: dict[str, Any]) -> WnbaStandingsResponse:
    season = _season_year(payload)
    if season is None:
        raise ValueError("ESPN standings payload missing season.year")

    by_key: dict[ConferenceKey, WnbaStandingsConference] = {}
    for child in payload.get("children") or []:
        if not isinstance(child, dict):
            continue
        abbrev = str(child.get("abbreviation") or "").strip().upper()
        mapped = _CONF_BY_ABBREV.get(abbrev)
        if mapped is None:
            name = str(child.get("name") or "").lower()
            if "eastern" in name:
                mapped = ("east", "Eastern Conference")
            elif "western" in name:
                mapped = ("west", "Western Conference")
            else:
                continue
        key, default_label = mapped
        label = str(child.get("name") or "").strip() or default_label
        standings = child.get("standings") or {}
        entries = standings.get("entries") if isinstance(standings, dict) else []
        teams: list[WnbaStandingsRow] = []
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            row = _row_from_entry(entry)
            if row is not None:
                teams.append(row)
        by_key[key] = WnbaStandingsConference(key=key, label=label, teams=teams)

    conferences: list[WnbaStandingsConference] = []
    for key in ("east", "west"):
        if key in by_key:
            conferences.append(by_key[key])
    return WnbaStandingsResponse(season=season, conferences=conferences)
```

- [ ] **Step 6: Run tests to verify they pass**

Run:

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor && PYTHONPATH=backend python3 -m pytest backend/tests/test_wnba_standings_normalize.py -v
```

Expected: PASS (all four tests)

- [ ] **Step 7: Commit**

```bash
git add backend/app/schemas/wnba_standings.py \
  backend/app/services/wnba_standings.py \
  backend/tests/fixtures/espn_wnba_standings.json \
  backend/tests/test_wnba_standings_normalize.py
git commit -m "$(cat <<'EOF'
feat: normalize ESPN WNBA standings into East/West rows

EOF
)"
```

---

### Task 2: Backend fetch, cache, and route

**Files:**
- Modify: `backend/app/services/wnba_standings.py` (add fetch + cache + `get_wnba_standings`)
- Create: `backend/app/api/routes/wnba_standings.py`
- Modify: `backend/app/main.py`
- Create: `backend/tests/test_wnba_standings_route.py`

**Interfaces:**
- Consumes: `normalize_espn_standings`
- Produces:
  - `async def fetch_espn_standings() -> dict`
  - `async def get_wnba_standings() -> WnbaStandingsResponse`
  - Route `GET /api/wnba/standings`

- [ ] **Step 1: Write failing route tests**

Create `backend/tests/test_wnba_standings_route.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import wnba_standings as svc

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def clear_cache():
    svc._cache.clear()
    yield
    svc._cache.clear()


def test_standings_returns_no_store_and_conferences():
    payload = json.loads((FIXTURES / "espn_wnba_standings.json").read_text())

    async def fake_fetch():
        return payload

    with patch.object(svc, "fetch_espn_standings", side_effect=fake_fetch):
        client = TestClient(app)
        res = client.get("/api/wnba/standings")
    assert res.status_code == 200
    assert res.headers.get("cache-control") == "no-store"
    body = res.json()
    assert body["season"] == 2026
    assert body["conferences"][0]["key"] == "east"
    assert body["conferences"][0]["teams"][0]["abbrev"] == "IND"
    assert body["conferences"][1]["key"] == "west"


def test_standings_uses_cache_within_ttl():
    payload = json.loads((FIXTURES / "espn_wnba_standings.json").read_text())
    calls = {"n": 0}

    async def fake_fetch():
        calls["n"] += 1
        return payload

    with patch.object(svc, "fetch_espn_standings", side_effect=fake_fetch):
        client = TestClient(app)
        assert client.get("/api/wnba/standings").status_code == 200
        assert client.get("/api/wnba/standings").status_code == 200
    assert calls["n"] == 1


def test_standings_stale_while_error():
    payload = json.loads((FIXTURES / "espn_wnba_standings.json").read_text())

    async def ok():
        return payload

    async def boom():
        raise RuntimeError("upstream down")

    with patch.object(svc, "fetch_espn_standings", side_effect=ok):
        client = TestClient(app)
        assert client.get("/api/wnba/standings").status_code == 200

    svc._cache["expires_at"] = 0

    with patch.object(svc, "fetch_espn_standings", side_effect=boom):
        res = client.get("/api/wnba/standings")
    assert res.status_code == 200
    assert res.json()["conferences"][0]["teams"][0]["abbrev"] == "IND"


def test_standings_502_no_store_when_cold():
    async def boom():
        raise RuntimeError("upstream down")

    with patch.object(svc, "fetch_espn_standings", side_effect=boom):
        client = TestClient(app)
        res = client.get("/api/wnba/standings")
    assert res.status_code == 502
    assert res.headers.get("cache-control") == "no-store"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor && PYTHONPATH=backend python3 -m pytest backend/tests/test_wnba_standings_route.py -v
```

Expected: FAIL (import / route not registered)

- [ ] **Step 3: Add fetch + cache to the service**

Append to `backend/app/services/wnba_standings.py` (keep existing normalize code):

```python
import asyncio
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

ET = ZoneInfo("America/New_York")
ESPN_URL = "https://site.api.espn.com/apis/v2/sports/basketball/wnba/standings"
ESPN_TIMEOUT_SECONDS = 10.0
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


async def fetch_espn_standings() -> dict:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(
        timeout=ESPN_TIMEOUT_SECONDS, headers=headers
    ) as client:
        res = await client.get(ESPN_URL)
        res.raise_for_status()
        return res.json()


def _fresh_cached() -> WnbaStandingsResponse | None:
    cached = _cache.get("response")
    if cached is None:
        return None
    if _cache.get("season") != current_wnba_season_year():
        return None
    if time.time() >= float(_cache.get("expires_at") or 0):
        return None
    return cached


async def get_wnba_standings() -> WnbaStandingsResponse:
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
            payload = await fetch_espn_standings()
            response = normalize_espn_standings(payload)
        except Exception:
            stale = _cache.get("response")
            if stale is not None and _cache.get("season") == season:
                logger.warning("WNBA standings refresh failed; serving stale cache")
                return stale
            raise
        # Prefer ESPN season year; fall back already handled in normalize
        _cache["response"] = response
        _cache["expires_at"] = time.time() + CACHE_TTL_SECONDS
        _cache["season"] = response.season
        return response
```

Note: place imports at top of file (merge with existing imports; do not duplicate `logging` / schema imports).

- [ ] **Step 4: Add route**

Create `backend/app/api/routes/wnba_standings.py`:

```python
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Response

from app.schemas.wnba_standings import WnbaStandingsResponse
from app.services.wnba_standings import get_wnba_standings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["wnba"])
_NO_STORE = {"Cache-Control": "no-store"}


@router.get("/wnba/standings", response_model=WnbaStandingsResponse)
async def wnba_standings(response: Response) -> WnbaStandingsResponse:
    response.headers["Cache-Control"] = "no-store"
    try:
        return await get_wnba_standings()
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("WNBA standings unavailable: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="WNBA standings are temporarily unavailable",
            headers=_NO_STORE,
        ) from exc
```

- [ ] **Step 5: Register router in `main.py`**

Modify `backend/app/main.py`:

1. Add `wnba_standings` to the `app.api.routes` import list.
2. Update description to mention `/api/wnba/standings` alongside other live WNBA exceptions.
3. Add `app.include_router(wnba_standings.router, prefix="/api")` next to the other WNBA routers.

- [ ] **Step 6: Run tests to verify they pass**

Run:

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor && PYTHONPATH=backend python3 -m pytest backend/tests/test_wnba_standings_normalize.py backend/tests/test_wnba_standings_route.py -v
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add backend/app/services/wnba_standings.py \
  backend/app/api/routes/wnba_standings.py \
  backend/app/main.py \
  backend/tests/test_wnba_standings_route.py
git commit -m "$(cat <<'EOF'
feat: add GET /api/wnba/standings ESPN proxy with cache

EOF
)"
```

---

### Task 3: Frontend API client + hook

**Files:**
- Modify: `frontend/src/lib/api.ts`
- Create: `frontend/src/hooks/useWnbaStandings.ts`
- Create: `frontend/src/hooks/useWnbaStandings.test.tsx`
- Modify: `frontend/src/lib/api.test.ts` (add fetch test if file already covers leaders)

**Interfaces:**
- Consumes: `GET /api/wnba/standings`
- Produces:
  - `ApiWnbaStandingsResponse` types
  - `fetchWnbaStandings(): Promise<ApiWnbaStandingsResponse>`
  - `useWnbaStandings()` with `hasNeverLoaded`

- [ ] **Step 1: Write failing hook test**

Create `frontend/src/hooks/useWnbaStandings.test.tsx`:

```tsx
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { useWnbaStandings } from "./useWnbaStandings";

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return (
    <QueryClientProvider client={client}>{children}</QueryClientProvider>
  );
}

describe("useWnbaStandings", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("loads standings from API", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        season: 2026,
        conferences: [
          { key: "east", label: "Eastern Conference", teams: [] },
          { key: "west", label: "Western Conference", teams: [] },
        ],
      }),
    });

    const { result } = renderHook(() => useWnbaStandings(), { wrapper });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.season).toBe(2026);
    expect(result.current.hasNeverLoaded).toBe(false);
    expect(String(fetchMock.mock.calls[0][0])).toContain("/api/wnba/standings");
  });

  it("sets hasNeverLoaded on cold error", async () => {
    fetchMock.mockResolvedValue({ ok: false, status: 502 });
    const { result } = renderHook(() => useWnbaStandings(), { wrapper });
    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.hasNeverLoaded).toBe(true);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor/frontend && npm run test -- --run src/hooks/useWnbaStandings.test.tsx
```

Expected: FAIL (module not found)

- [ ] **Step 3: Add API types + fetch**

Append to `frontend/src/lib/api.ts`:

```ts
export type ApiWnbaStandingsRow = {
  rank: number;
  team_id: string;
  abbrev: string;
  name: string;
  logo_url: string | null;
  wins: number;
  losses: number;
  wl: string;
  pct: string;
  gb: string;
  home: string;
  away: string;
  l10: string;
  diff: string;
  streak: string;
};

export type ApiWnbaStandingsConference = {
  key: "east" | "west";
  label: string;
  teams: ApiWnbaStandingsRow[];
};

export type ApiWnbaStandingsResponse = {
  season: number;
  conferences: ApiWnbaStandingsConference[];
};

export async function fetchWnbaStandings(): Promise<ApiWnbaStandingsResponse> {
  const res = await fetch(`${API_BASE}/api/wnba/standings`, {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`Standings request failed: ${res.status}`);
  }
  return res.json();
}
```

- [ ] **Step 4: Add hook**

Create `frontend/src/hooks/useWnbaStandings.ts`:

```ts
import { useQuery } from "@tanstack/react-query";
import { fetchWnbaStandings } from "@/lib/api";

export function useWnbaStandings() {
  const query = useQuery({
    queryKey: ["wnba", "standings"],
    queryFn: fetchWnbaStandings,
  });

  return {
    ...query,
    hasNeverLoaded: query.isError && query.data === undefined,
  };
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor/frontend && npm run test -- --run src/hooks/useWnbaStandings.test.tsx
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add frontend/src/lib/api.ts \
  frontend/src/hooks/useWnbaStandings.ts \
  frontend/src/hooks/useWnbaStandings.test.tsx
git commit -m "$(cat <<'EOF'
feat: add WNBA standings API client and hook

EOF
)"
```

---

### Task 4: Conference card + standings grid

**Files:**
- Create: `frontend/src/components/league/StandingsConferenceCard.tsx`
- Create: `frontend/src/components/league/StandingsGrid.tsx`
- Create: `frontend/src/components/league/StandingsGrid.test.tsx`

**Interfaces:**
- Consumes: `ApiWnbaStandingsConference`, `teamColor`, `TeamAbbrevAvatar`
- Produces: `StandingsGrid({ season, conferences, isLoading?, isError? })`

- [ ] **Step 1: Write failing grid tests**

Create `frontend/src/components/league/StandingsGrid.test.tsx`:

```tsx
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { StandingsGrid } from "./StandingsGrid";
import type { ApiWnbaStandingsConference } from "@/lib/api";

const sample: ApiWnbaStandingsConference[] = [
  {
    key: "east",
    label: "Eastern Conference",
    teams: [
      {
        rank: 1,
        team_id: "5",
        abbrev: "IND",
        name: "Indiana Fever",
        logo_url: "https://a.espncdn.com/i/teamlogos/wnba/500/ind.png",
        wins: 18,
        losses: 10,
        wl: "18-10",
        pct: ".643",
        gb: "-",
        home: "11-5",
        away: "7-5",
        l10: "8-2",
        diff: "+169",
        streak: "W4",
      },
    ],
  },
  {
    key: "west",
    label: "Western Conference",
    teams: [
      {
        rank: 1,
        team_id: "16",
        abbrev: "MIN",
        name: "Minnesota Lynx",
        logo_url: null,
        wins: 24,
        losses: 6,
        wl: "24-6",
        pct: ".800",
        gb: "-",
        home: "13-2",
        away: "11-4",
        l10: "8-2",
        diff: "-12",
        streak: "L2",
      },
    ],
  },
];

describe("StandingsGrid", () => {
  it("renders season, conferences, rows, and attribution", () => {
    render(
      <StandingsGrid season={2026} conferences={sample} />,
    );
    expect(screen.getByText("2026 regular season")).toBeInTheDocument();
    expect(screen.getByText("Eastern Conference")).toBeInTheDocument();
    expect(screen.getByText("Western Conference")).toBeInTheDocument();
    expect(screen.getByText("Indiana Fever")).toBeInTheDocument();
    expect(screen.getByText("IND")).toBeInTheDocument();
    expect(screen.getByText("18-10")).toBeInTheDocument();
    expect(screen.getByText("Data: ESPN")).toBeInTheDocument();
  });

  it("shows loading skeletons", () => {
    render(
      <StandingsGrid season={2026} conferences={[]} isLoading />,
    );
    expect(screen.getByLabelText("Loading standings")).toBeInTheDocument();
  });

  it("shows error copy when never loaded", () => {
    render(
      <StandingsGrid season={2026} conferences={[]} isError />,
    );
    expect(screen.getByText("Standings unavailable")).toBeInTheDocument();
  });

  it("shows No data for empty conference", () => {
    render(
      <StandingsGrid
        season={2026}
        conferences={[
          { key: "east", label: "Eastern Conference", teams: [] },
        ]}
      />,
    );
    expect(screen.getByText("No data")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor/frontend && npm run test -- --run src/components/league/StandingsGrid.test.tsx
```

Expected: FAIL (module not found)

- [ ] **Step 3: Implement `StandingsConferenceCard`**

Create `frontend/src/components/league/StandingsConferenceCard.tsx`:

```tsx
import type { ApiWnbaStandingsConference } from "@/lib/api";
import { TeamAbbrevAvatar } from "@/components/TeamAbbrevAvatar";
import { teamColor } from "./wnbaTeamColors";

type StandingsConferenceCardProps = {
  conference: ApiWnbaStandingsConference;
};

function diffClass(diff: string): string {
  if (diff.startsWith("+")) return "text-emerald-400";
  if (diff.startsWith("-") && diff !== "-") return "text-rose-400";
  return "text-white/50";
}

function streakClass(streak: string): string {
  if (streak.startsWith("W")) return "text-emerald-400";
  if (streak.startsWith("L")) return "text-rose-400";
  return "text-white/50";
}

export function StandingsConferenceCard({
  conference,
}: StandingsConferenceCardProps) {
  return (
    <section className="rounded-xl border border-white/10 bg-[#141414] p-4">
      <h3 className="mb-3 text-base font-semibold text-white">
        {conference.label}
      </h3>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[720px] text-left text-sm">
          <thead>
            <tr className="text-[11px] tracking-wide text-white/40 uppercase">
              <th className="pb-2 pr-2 font-medium">#</th>
              <th className="pb-2 pr-2 font-medium">Team</th>
              <th className="pb-2 pr-2 font-medium">W-L</th>
              <th className="pb-2 pr-2 font-medium">PCT</th>
              <th className="pb-2 pr-2 font-medium">GB</th>
              <th className="pb-2 pr-2 font-medium">Home</th>
              <th className="pb-2 pr-2 font-medium">Away</th>
              <th className="pb-2 pr-2 font-medium">L10</th>
              <th className="pb-2 pr-2 font-medium">Diff</th>
              <th className="pb-2 font-medium">Strk</th>
            </tr>
          </thead>
          <tbody>
            {conference.teams.length === 0 ? (
              <tr>
                <td colSpan={10} className="py-3 text-white/40">
                  No data
                </td>
              </tr>
            ) : (
              conference.teams.map((row) => (
                <tr key={`${conference.key}-${row.team_id}`}>
                  <td className="py-1.5 pr-2 text-white/50">{row.rank}</td>
                  <td className="py-1.5 pr-2">
                    <div className="flex items-center gap-2">
                      <TeamAbbrevAvatar
                        abbrev={row.abbrev}
                        logoUrl={row.logo_url}
                        sizeClassName="size-5"
                      />
                      <span
                        className="font-semibold"
                        style={{ color: teamColor(row.abbrev) }}
                      >
                        {row.abbrev}
                      </span>
                      <span className="text-white/80">{row.name}</span>
                    </div>
                  </td>
                  <td className="py-1.5 pr-2 text-white">{row.wl}</td>
                  <td className="py-1.5 pr-2 text-white/70">{row.pct}</td>
                  <td className="py-1.5 pr-2 text-white/70">{row.gb}</td>
                  <td className="py-1.5 pr-2 text-white/70">{row.home}</td>
                  <td className="py-1.5 pr-2 text-white/70">{row.away}</td>
                  <td className="py-1.5 pr-2 text-white/70">{row.l10}</td>
                  <td className={`py-1.5 pr-2 font-medium ${diffClass(row.diff)}`}>
                    {row.diff}
                  </td>
                  <td className={`py-1.5 font-medium ${streakClass(row.streak)}`}>
                    {row.streak}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}
```

- [ ] **Step 4: Implement `StandingsGrid`**

Create `frontend/src/components/league/StandingsGrid.tsx`:

```tsx
import type { ApiWnbaStandingsConference } from "@/lib/api";
import { StandingsConferenceCard } from "./StandingsConferenceCard";

type StandingsGridProps = {
  season: number;
  conferences: ApiWnbaStandingsConference[];
  isLoading?: boolean;
  isError?: boolean;
};

function Skeletons() {
  return (
    <div
      className="grid grid-cols-1 gap-4 lg:grid-cols-2"
      aria-label="Loading standings"
    >
      {Array.from({ length: 2 }, (_, i) => (
        <div
          key={i}
          className="h-72 animate-pulse rounded-xl border border-white/10 bg-white/5"
        />
      ))}
    </div>
  );
}

export function StandingsGrid({
  season,
  conferences,
  isLoading = false,
  isError = false,
}: StandingsGridProps) {
  return (
    <section className="mx-auto max-w-6xl space-y-4 px-4 sm:px-6">
      <p className="text-sm text-white/45">{season} regular season</p>
      {isLoading ? (
        <Skeletons />
      ) : isError ? (
        <p className="text-sm text-white/50">Standings unavailable</p>
      ) : (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {conferences.map((conference) => (
            <StandingsConferenceCard
              key={conference.key}
              conference={conference}
            />
          ))}
        </div>
      )}
      <p className="text-xs text-white/35">Data: ESPN</p>
    </section>
  );
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor/frontend && npm run test -- --run src/components/league/StandingsGrid.test.tsx
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/league/StandingsConferenceCard.tsx \
  frontend/src/components/league/StandingsGrid.tsx \
  frontend/src/components/league/StandingsGrid.test.tsx
git commit -m "$(cat <<'EOF'
feat: add WNBA standings conference grid UI

EOF
)"
```

---

### Task 5: Page, router, and subnav

**Files:**
- Create: `frontend/src/pages/LeagueStandingsPage.tsx`
- Create: `frontend/src/pages/LeagueStandingsPage.test.tsx`
- Modify: `frontend/src/AppRouter.tsx`
- Modify: `frontend/src/AppRouter.test.tsx`
- Modify: `frontend/src/components/league/LeagueSubnav.tsx`
- Modify: `frontend/src/components/league/LeagueSubnav.test.tsx`

**Interfaces:**
- Consumes: `useWnbaStandings`, `StandingsGrid`, `LeagueSubnav`
- Produces: route `/wnba/standings`; WNBA subnav Standings link

- [ ] **Step 1: Write failing page + subnav + router tests**

Create `frontend/src/pages/LeagueStandingsPage.test.tsx`:

```tsx
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { LeagueStandingsPage } from "./LeagueStandingsPage";

function renderPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={["/wnba/standings"]}>
        <LeagueStandingsPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("LeagueStandingsPage", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders standings from API", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        season: 2026,
        conferences: [
          {
            key: "east",
            label: "Eastern Conference",
            teams: [
              {
                rank: 1,
                team_id: "5",
                abbrev: "IND",
                name: "Indiana Fever",
                logo_url: null,
                wins: 18,
                losses: 10,
                wl: "18-10",
                pct: ".643",
                gb: "-",
                home: "11-5",
                away: "7-5",
                l10: "8-2",
                diff: "+169",
                streak: "W4",
              },
            ],
          },
        ],
      }),
    });

    renderPage();
    expect(await screen.findByText("Eastern Conference")).toBeInTheDocument();
    expect(screen.getByText("Indiana Fever")).toBeInTheDocument();
    expect(screen.getByText("Data: ESPN")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Standings" })).toHaveAttribute(
      "aria-current",
      "page",
    );
  });
});
```

Update `LeagueSubnav.test.tsx` so the WNBA case asserts Standings is a link to `/wnba/standings`, and the NBA case asserts Standings stays a disabled button:

```tsx
it("links Matchups, Leaders, and Standings on WNBA; disables others", () => {
  render(
    <MemoryRouter initialEntries={["/wnba/standings"]}>
      <LeagueSubnav league="wnba" />
    </MemoryRouter>,
  );
  const standings = screen.getByRole("link", { name: "Standings" });
  expect(standings).toHaveAttribute("href", "/wnba/standings");
  expect(standings).toHaveAttribute("aria-current", "page");
  expect(screen.getByRole("link", { name: "Leaders" })).toHaveAttribute(
    "href",
    "/wnba/leaders",
  );
  expect(screen.getByRole("link", { name: "Matchups" })).toHaveAttribute(
    "href",
    "/wnba/matchups",
  );
  expect(
    screen.getByRole("button", { name: "HoopVista Picks" }),
  ).toBeDisabled();
});

it("keeps Leaders and Standings disabled on NBA", () => {
  render(
    <MemoryRouter initialEntries={["/nba/matchups"]}>
      <LeagueSubnav league="nba" />
    </MemoryRouter>,
  );
  expect(screen.getByRole("button", { name: "Leaders" })).toBeDisabled();
  expect(screen.getByRole("button", { name: "Standings" })).toBeDisabled();
  expect(screen.getByRole("link", { name: "Matchups" })).toHaveAttribute(
    "href",
    "/nba/matchups",
  );
});
```

Add to `AppRouter.test.tsx`:

```tsx
it("renders WNBA standings at /wnba/standings", async () => {
  fetchMock.mockImplementation(async (input: RequestInfo) => {
    const url = String(input);
    if (url.includes("/api/wnba/standings")) {
      return {
        ok: true,
        json: async () => ({
          season: 2026,
          conferences: [],
        }),
      };
    }
    return {
      ok: true,
      json: async () => ({ date: "2026-07-29", fetched_at: "", games: [] }),
    };
  });
  renderWithProviders(["/wnba/standings"]);
  expect(
    await screen.findByText(/2026 regular season/i),
  ).toBeInTheDocument();
  expect(screen.getByText("Data: ESPN")).toBeInTheDocument();
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor/frontend && npm run test -- --run src/pages/LeagueStandingsPage.test.tsx src/components/league/LeagueSubnav.test.tsx src/AppRouter.test.tsx
```

Expected: FAIL on missing page / Standings still disabled

- [ ] **Step 3: Implement page**

Create `frontend/src/pages/LeagueStandingsPage.tsx`:

```tsx
import { LeagueSubnav } from "@/components/league/LeagueSubnav";
import { StandingsGrid } from "@/components/league/StandingsGrid";
import { useWnbaStandings } from "@/hooks/useWnbaStandings";

export function LeagueStandingsPage() {
  const { data, isLoading, hasNeverLoaded } = useWnbaStandings();
  const season = data?.season ?? new Date().getFullYear();

  return (
    <div className="space-y-6 py-6">
      <LeagueSubnav league="wnba" />
      <StandingsGrid
        season={season}
        conferences={data?.conferences ?? []}
        isLoading={isLoading && !data}
        isError={hasNeverLoaded}
      />
    </div>
  );
}
```

- [ ] **Step 4: Wire router**

In `frontend/src/AppRouter.tsx`:

1. Import `LeagueStandingsPage`.
2. Add `<Route path="/wnba/standings" element={<LeagueStandingsPage />} />` next to the leaders route.

- [ ] **Step 5: Enable Standings in subnav**

Update `itemPath` / `isActive` in `LeagueSubnav.tsx`:

```tsx
function itemPath(item: string): string | null {
  if (item === "Matchups") return `/${league}/matchups`;
  if (item === "Leaders" && league === "wnba") return "/wnba/leaders";
  if (item === "Standings" && league === "wnba") return "/wnba/standings";
  return null;
}

function isActive(item: string): boolean {
  if (item === "Matchups") return pathname.endsWith("/matchups");
  if (item === "Leaders") return pathname.endsWith("/leaders");
  if (item === "Standings") return pathname.endsWith("/standings");
  return false;
}
```

- [ ] **Step 6: Run frontend verification**

Run:

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor/frontend && npm run test -- --run src/components/league src/pages/LeagueStandingsPage.test.tsx src/hooks/useWnbaStandings.test.tsx src/AppRouter.test.tsx && npm run build
```

Expected: PASS + successful build

- [ ] **Step 7: Run backend verification**

Run:

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor && PYTHONPATH=backend python3 -m pytest backend/tests/test_wnba_standings_normalize.py backend/tests/test_wnba_standings_route.py -v
```

Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add frontend/src/pages/LeagueStandingsPage.tsx \
  frontend/src/pages/LeagueStandingsPage.test.tsx \
  frontend/src/AppRouter.tsx \
  frontend/src/AppRouter.test.tsx \
  frontend/src/components/league/LeagueSubnav.tsx \
  frontend/src/components/league/LeagueSubnav.test.tsx
git commit -m "$(cat <<'EOF'
feat: ship /wnba/standings page and subnav link

EOF
)"
```

---

## Spec coverage checklist

| Spec requirement | Task |
| --- | --- |
| `/wnba/standings` under HomeChromeLayout | Task 5 |
| No LeagueHero | Task 5 |
| Season label `{season} regular season` | Task 4 |
| East/West conference cards | Task 4 |
| Full column set | Task 4 |
| Desktop 2-col / mobile stack | Task 4 (`lg:grid-cols-2`) |
| Diff/Strk color coding | Task 4 |
| Team logos + colored abbrevs | Task 4 |
| `Data: ESPN` | Task 4 |
| ESPN backend proxy + normalize | Tasks 1–2 |
| 10-minute cache, stale-on-error, 502 cold | Task 2 |
| Subnav Standings enabled (WNBA only) | Task 5 |
| Loading / error / empty states | Task 4 |
| No `/nba/standings` | Task 5 (route not registered) |
| CLAUDE.md: typed models, defensive skip, tests+code, focused commits | All tasks |
