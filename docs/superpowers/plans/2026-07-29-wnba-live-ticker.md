# WNBA Live Ticker & LIVE NOW Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `GET /api/wnba/scoreboard/today` (ESPN + stats.wnba.com → merged Pydantic slate) and wire TanStack Query polling into `LiveTicker` + `LiveNowSection`.

**Architecture:** Backend service fetches both upstreams in parallel, normalizes each to `WnbaGame`, field-merges conflicts, caches 30–60s by slate state, and always returns `Cache-Control: no-store`. Frontend adds TanStack Query with a shared `useWnbaScoreboard` hook (18s interval while any game is not Final) consumed by home chrome and LIVE NOW.

**Tech Stack:** FastAPI · Pydantic · httpx · asyncio · React 19 · TypeScript · Vite 6 · TanStack Query · Vitest · Testing Library

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-29-wnba-live-ticker-design.md`
- Coding standards: `CLAUDE.md` (small focused modules, strong typing, tests with code, explain why in comments)
- Endpoint: `GET /api/wnba/scoreboard/today` only (no NBA / no generic league param)
- "Today" = calendar date in `America/New_York`
- Per-field merge preferring more complete values
- Response always `Cache-Control: no-store`
- Poll interval: `18_000` ms while any `status !== "final"`; stop when all Final or empty
- Full day's slate on ticker + LIVE NOW
- No team logos; letter-circle avatars
- Upstream URLs:
  - ESPN: `https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard?dates=YYYYMMDD`
  - stats.wnba.com: `https://stats.wnba.com/stats/scoreboardv3?GameDate=MM/DD/YYYY&LeagueID=10`
- Verify backend with `python3 -m pytest backend/tests/ -v`
- Verify frontend with `cd frontend && npm run test && npm run build`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `backend/requirements.txt` | Add `httpx` |
| `backend/app/schemas/wnba_scoreboard.py` | Pydantic models |
| `backend/app/services/wnba_scoreboard.py` | Fetch, normalize, merge, TTL cache |
| `backend/app/api/routes/wnba_scoreboard.py` | Thin route + headers |
| `backend/app/main.py` | Register router; update description |
| `backend/tests/fixtures/espn_wnba_scoreboard.json` | Minimal ESPN fixture |
| `backend/tests/fixtures/stats_wnba_scoreboard.json` | Minimal stats.wnba fixture |
| `backend/tests/test_wnba_scoreboard_normalize.py` | Normalize + merge + TTL tests |
| `backend/tests/test_wnba_scoreboard_route.py` | Route / cache-control / failover |
| `frontend/package.json` | Add `@tanstack/react-query` |
| `frontend/vite.config.ts` | Dev proxy `/api` → backend |
| `frontend/src/main.tsx` | `QueryClientProvider` |
| `frontend/src/lib/api.ts` | `fetchWnbaScoreboard()` |
| `frontend/src/hooks/useWnbaScoreboard.ts` | Query + interval guard |
| `frontend/src/hooks/useWnbaScoreboard.test.tsx` | Polling guard tests |
| `frontend/src/components/home/types.ts` | Add `status` on live/ticker types |
| `frontend/src/components/home/mapScoreboard.ts` | API → UI type mappers |
| `frontend/src/components/home/LiveTicker.tsx` | Unchanged API (receives games) |
| `frontend/src/components/home/LiveNowSection.tsx` | Loading / empty / violet / pulse |
| `frontend/src/components/home/LiveNowSection.test.tsx` | Updated empty/loading/card tests |
| `frontend/src/layouts/HomeChromeLayout.tsx` | Hook → ticker |
| `frontend/src/pages/HomePage.tsx` | Hook → LIVE NOW |

---

### Task 1: Pydantic schema + ESPN normalizer

**Files:**
- Create: `backend/app/schemas/wnba_scoreboard.py`
- Create: `backend/app/services/wnba_scoreboard.py` (normalize helpers only for this task)
- Create: `backend/tests/fixtures/espn_wnba_scoreboard.json`
- Create: `backend/tests/test_wnba_scoreboard_normalize.py`
- Modify: `backend/requirements.txt` (add `httpx>=0.27.0` for later tasks; unused yet)

**Interfaces:**
- Produces:
  - `GameStatus = Literal["scheduled", "live", "halftime", "final"]`
  - `class WnbaTeam(BaseModel): abbrev: str; name: str; score: int | None`
  - `class WnbaGame(BaseModel): id: str; league: Literal["wnba"]; status: GameStatus; status_label: str; away: WnbaTeam; home: WnbaTeam; start_time_et: str`
  - `class WnbaScoreboardResponse(BaseModel): date: str; games: list[WnbaGame]; fetched_at: str`
  - `def normalize_espn_scoreboard(payload: dict, *, date_et: str) -> list[WnbaGame]`

- [ ] **Step 1: Write fixture + failing test**

Create `backend/tests/fixtures/espn_wnba_scoreboard.json`:

```json
{
  "events": [
    {
      "id": "401749001",
      "date": "2026-07-29T23:00Z",
      "competitions": [
        {
          "competitors": [
            {
              "homeAway": "home",
              "score": "44",
              "team": {
                "abbreviation": "DAL",
                "displayName": "Dallas Wings"
              }
            },
            {
              "homeAway": "away",
              "score": "36",
              "team": {
                "abbreviation": "ATL",
                "displayName": "Atlanta Dream"
              }
            }
          ]
        }
      ],
      "status": {
        "type": {
          "state": "in",
          "completed": false,
          "name": "STATUS_IN_PROGRESS",
          "shortDetail": "7:13 - 3rd",
          "detail": "7:13 - 3rd Quarter"
        },
        "period": 3,
        "displayClock": "7:13"
      }
    }
  ]
}
```

Create `backend/tests/test_wnba_scoreboard_normalize.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from app.services.wnba_scoreboard import normalize_espn_scoreboard

FIXTURES = Path(__file__).parent / "fixtures"


def test_normalize_espn_live_game():
    payload = json.loads((FIXTURES / "espn_wnba_scoreboard.json").read_text())
    games = normalize_espn_scoreboard(payload, date_et="2026-07-29")
    assert len(games) == 1
    g = games[0]
    assert g.id == "espn-401749001"
    assert g.league == "wnba"
    assert g.status == "live"
    assert g.status_label == "Q3 7:13"
    assert g.away.abbrev == "ATL"
    assert g.away.name == "Atlanta Dream"
    assert g.away.score == 36
    assert g.home.abbrev == "DAL"
    assert g.home.score == 44
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor && PYTHONPATH=backend python3 -m pytest backend/tests/test_wnba_scoreboard_normalize.py::test_normalize_espn_live_game -v`

Expected: FAIL (module not found)

- [ ] **Step 3: Implement schema + ESPN normalizer**

`backend/app/schemas/wnba_scoreboard.py`:

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

GameStatus = Literal["scheduled", "live", "halftime", "final"]


class WnbaTeam(BaseModel):
    abbrev: str
    name: str
    score: int | None = None


class WnbaGame(BaseModel):
    id: str
    league: Literal["wnba"] = "wnba"
    status: GameStatus
    status_label: str
    away: WnbaTeam
    home: WnbaTeam
    start_time_et: str


class WnbaScoreboardResponse(BaseModel):
    date: str = Field(description="YYYY-MM-DD in America/New_York")
    games: list[WnbaGame]
    fetched_at: str
```

In `backend/app/services/wnba_scoreboard.py`, implement:

```python
from __future__ import annotations

from app.schemas.wnba_scoreboard import GameStatus, WnbaGame, WnbaTeam


def _espn_status(status_block: dict) -> tuple[GameStatus, str]:
    typ = status_block.get("type") or {}
    name = str(typ.get("name") or "")
    state = str(typ.get("state") or "")
    short = str(typ.get("shortDetail") or typ.get("detail") or "")
    period = status_block.get("period")
    clock = str(status_block.get("displayClock") or "").strip()

    if typ.get("completed") or name == "STATUS_FINAL" or state == "post":
        return "final", "Final"
    if "HALFTIME" in name.upper() or short.lower() == "halftime":
        return "halftime", "Halftime"
    if state == "in" or name == "STATUS_IN_PROGRESS":
        # Prefer compact Qn clock when period + clock present
        if isinstance(period, int) and period > 0 and clock:
            return "live", f"Q{period} {clock}"
        return "live", short or "Live"
    # scheduled
    label = short or "Scheduled"
    return "scheduled", label


def normalize_espn_scoreboard(payload: dict, *, date_et: str) -> list[WnbaGame]:
    games: list[WnbaGame] = []
    for event in payload.get("events") or []:
        comps = (event.get("competitions") or [{}])[0]
        teams = {c.get("homeAway"): c for c in (comps.get("competitors") or [])}
        away_c, home_c = teams.get("away") or {}, teams.get("home") or {}
        status, label = _espn_status(event.get("status") or {})
        start = str(event.get("date") or "")
        def team(c: dict) -> WnbaTeam:
            t = c.get("team") or {}
            raw = c.get("score")
            score = int(raw) if raw not in (None, "") else None
            return WnbaTeam(
                abbrev=str(t.get("abbreviation") or ""),
                name=str(t.get("displayName") or ""),
                score=score if status != "scheduled" else None,
            )
        games.append(
            WnbaGame(
                id=f"espn-{event.get('id')}",
                status=status,
                status_label=label,
                away=team(away_c),
                home=team(home_c),
                start_time_et=start,
            )
        )
    return games
```

Keep `date_et` in the signature for later filtering (ESPN `?dates=` already scopes; no-op filter OK for now).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=backend python3 -m pytest backend/tests/test_wnba_scoreboard_normalize.py::test_normalize_espn_live_game -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/requirements.txt backend/app/schemas/wnba_scoreboard.py backend/app/services/wnba_scoreboard.py backend/tests/fixtures/espn_wnba_scoreboard.json backend/tests/test_wnba_scoreboard_normalize.py
git commit -m "$(cat <<'EOF'
Add WNBA scoreboard schema and ESPN normalizer.

EOF
)"
```

---

### Task 2: stats.wnba.com normalizer + field merge + TTL

**Files:**
- Modify: `backend/app/services/wnba_scoreboard.py`
- Create: `backend/tests/fixtures/stats_wnba_scoreboard.json`
- Modify: `backend/tests/test_wnba_scoreboard_normalize.py`

**Interfaces:**
- Consumes: `WnbaGame`, `normalize_espn_scoreboard`
- Produces:
  - `def normalize_stats_scoreboard(payload: dict, *, date_et: str) -> list[WnbaGame]`
  - `def merge_games(espn: list[WnbaGame], stats: list[WnbaGame]) -> list[WnbaGame]`
  - `def cache_ttl_seconds(games: list[WnbaGame]) -> int`  # 30 if any live/halftime else 60
  - `def prefer_complete(a, b)` field helpers used by merge

- [ ] **Step 1: Write fixtures + failing tests**

`backend/tests/fixtures/stats_wnba_scoreboard.json`:

```json
{
  "scoreboard": {
    "gameDate": "2026-07-29",
    "games": [
      {
        "gameId": "1022600123",
        "gameStatus": 2,
        "gameStatusText": "Q3 7:10",
        "period": 3,
        "gameClock": "PT7M10.00S",
        "gameTimeUTC": "2026-07-29T23:00:00Z",
        "homeTeam": {
          "teamTricode": "DAL",
          "teamName": "Wings",
          "teamCity": "Dallas",
          "score": 45
        },
        "awayTeam": {
          "teamTricode": "ATL",
          "teamName": "Dream",
          "teamCity": "Atlanta",
          "score": 36
        }
      }
    ]
  }
}
```

Append to `backend/tests/test_wnba_scoreboard_normalize.py`:

```python
from app.schemas.wnba_scoreboard import WnbaGame, WnbaTeam
from app.services.wnba_scoreboard import (
    cache_ttl_seconds,
    merge_games,
    normalize_stats_scoreboard,
)


def test_normalize_stats_live_game():
    payload = json.loads((FIXTURES / "stats_wnba_scoreboard.json").read_text())
    games = normalize_stats_scoreboard(payload, date_et="2026-07-29")
    assert len(games) == 1
    g = games[0]
    assert g.id == "1022600123"
    assert g.status == "live"
    assert g.away.abbrev == "ATL"
    assert g.home.score == 45


def test_merge_prefers_non_null_and_richer_fields():
    espn = [
        WnbaGame(
            id="espn-1",
            status="live",
            status_label="Q3 7:13",
            away=WnbaTeam(abbrev="ATL", name="Atlanta Dream", score=36),
            home=WnbaTeam(abbrev="DAL", name="Dallas Wings", score=44),
            start_time_et="2026-07-29T23:00:00Z",
        )
    ]
    stats = [
        WnbaGame(
            id="1022600123",
            status="live",
            status_label="Q3 7:10",
            away=WnbaTeam(abbrev="ATL", name="Atlanta Dream", score=36),
            home=WnbaTeam(abbrev="DAL", name="Dallas Wings", score=45),
            start_time_et="2026-07-29T23:00:00Z",
        )
    ]
    merged = merge_games(espn, stats)
    assert len(merged) == 1
    assert merged[0].id == "1022600123"  # prefer stats id
    assert merged[0].home.score == 45  # prefer non-stale higher completeness: non-null from stats


def test_cache_ttl_live_vs_final():
    live = [
        WnbaGame(
            id="1",
            status="live",
            status_label="Q1 10:00",
            away=WnbaTeam(abbrev="ATL", name="A", score=0),
            home=WnbaTeam(abbrev="DAL", name="D", score=0),
            start_time_et="2026-07-29T23:00:00Z",
        )
    ]
    final = [
        WnbaGame(
            id="1",
            status="final",
            status_label="Final",
            away=WnbaTeam(abbrev="ATL", name="A", score=80),
            home=WnbaTeam(abbrev="DAL", name="D", score=75),
            start_time_et="2026-07-29T23:00:00Z",
        )
    ]
    assert cache_ttl_seconds(live) == 30
    assert cache_ttl_seconds(final) == 60
    assert cache_ttl_seconds([]) == 60
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=backend python3 -m pytest backend/tests/test_wnba_scoreboard_normalize.py -v`

Expected: FAIL on missing `normalize_stats_scoreboard` / `merge_games` / `cache_ttl_seconds`

- [ ] **Step 3: Implement stats normalizer, merge, TTL**

Add to `backend/app/services/wnba_scoreboard.py`:

```python
import re
from datetime import datetime, timezone

_STATUS_MAP = {1: "scheduled", 2: "live", 3: "final"}


def _parse_iso_clock(game_clock: str | None) -> str | None:
    if not game_clock:
        return None
    # PT7M10.00S → 7:10
    m = re.match(r"PT(?:(\d+)M)?(?:(\d+)(?:\.\d+)?S)?", game_clock)
    if not m:
        return None
    mins = int(m.group(1) or 0)
    secs = int(float(m.group(2) or 0))
    return f"{mins}:{secs:02d}"


def _stats_status(game: dict) -> tuple[GameStatus, str]:
    code = int(game.get("gameStatus") or 1)
    text = str(game.get("gameStatusText") or "").strip()
    if code == 3 or text.lower() == "final":
        return "final", "Final"
    if "half" in text.lower():
        return "halftime", "Halftime"
    if code == 2:
        period = game.get("period")
        clock = _parse_iso_clock(game.get("gameClock"))
        if isinstance(period, int) and period > 0 and clock:
            return "live", f"Q{period} {clock}"
        return "live", text or "Live"
    return "scheduled", text or "Scheduled"


def normalize_stats_scoreboard(payload: dict, *, date_et: str) -> list[WnbaGame]:
    board = payload.get("scoreboard") or payload
    games: list[WnbaGame] = []
    for g in board.get("games") or []:
        status, label = _stats_status(g)
        away, home = g.get("awayTeam") or {}, g.get("homeTeam") or {}

        def team(t: dict) -> WnbaTeam:
            city = str(t.get("teamCity") or "").strip()
            name = str(t.get("teamName") or "").strip()
            full = f"{city} {name}".strip()
            raw = t.get("score")
            score = int(raw) if raw is not None and status != "scheduled" else None
            return WnbaTeam(
                abbrev=str(t.get("teamTricode") or ""),
                name=full,
                score=score,
            )

        games.append(
            WnbaGame(
                id=str(g.get("gameId")),
                status=status,
                status_label=label,
                away=team(away),
                home=team(home),
                start_time_et=str(g.get("gameTimeUTC") or ""),
            )
        )
    return games


def _match_key(game: WnbaGame) -> tuple[str, str]:
    return (game.away.abbrev.upper(), game.home.abbrev.upper())


def _score_richness(label: str) -> int:
    return len(label or "")


def merge_games(espn: list[WnbaGame], stats: list[WnbaGame]) -> list[WnbaGame]:
    by_key: dict[tuple[str, str], WnbaGame] = {}
    for g in espn:
        by_key[_match_key(g)] = g
    for g in stats:
        key = _match_key(g)
        if key not in by_key:
            by_key[key] = g
            continue
        a = by_key[key]
        # Prefer stats id when present and not espn-prefixed
        game_id = g.id if not g.id.startswith("espn-") else a.id
        if a.id.startswith("espn-") and not g.id.startswith("espn-"):
            game_id = g.id
        status = g.status if g.status != "scheduled" or a.status == "scheduled" else a.status
        # Prefer richer status_label
        status_label = (
            g.status_label
            if _score_richness(g.status_label) >= _score_richness(a.status_label)
            else a.status_label
        )
        def pick_score(x: int | None, y: int | None) -> int | None:
            if x is None:
                return y
            if y is None:
                return x
            return y  # prefer stats when both present

        by_key[key] = WnbaGame(
            id=game_id,
            status=status if status in ("scheduled", "live", "halftime", "final") else a.status,
            status_label=status_label,
            away=WnbaTeam(
                abbrev=g.away.abbrev or a.away.abbrev,
                name=g.away.name if len(g.away.name) >= len(a.away.name) else a.away.name,
                score=pick_score(a.away.score, g.away.score),
            ),
            home=WnbaTeam(
                abbrev=g.home.abbrev or a.home.abbrev,
                name=g.home.name if len(g.home.name) >= len(a.home.name) else a.home.name,
                score=pick_score(a.home.score, g.home.score),
            ),
            start_time_et=g.start_time_et or a.start_time_et,
        )
    return sorted(by_key.values(), key=lambda g: g.start_time_et or g.id)


def cache_ttl_seconds(games: list[WnbaGame]) -> int:
    if any(g.status in ("live", "halftime") for g in games):
        return 30
    return 60
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=backend python3 -m pytest backend/tests/test_wnba_scoreboard_normalize.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/wnba_scoreboard.py backend/tests/fixtures/stats_wnba_scoreboard.json backend/tests/test_wnba_scoreboard_normalize.py
git commit -m "$(cat <<'EOF'
Add stats.wnba normalizer, field merge, and cache TTL.

EOF
)"
```

---

### Task 3: Scoreboard service + FastAPI route

**Files:**
- Modify: `backend/app/services/wnba_scoreboard.py`
- Create: `backend/app/api/routes/wnba_scoreboard.py`
- Modify: `backend/app/main.py`
- Create: `backend/tests/test_wnba_scoreboard_route.py`
- Ensure: `httpx` in `backend/requirements.txt`

**Interfaces:**
- Consumes: `normalize_espn_scoreboard`, `normalize_stats_scoreboard`, `merge_games`, `cache_ttl_seconds`, `WnbaScoreboardResponse`
- Produces:
  - `async def get_today_scoreboard() -> WnbaScoreboardResponse` (uses module-level cache)
  - Route `GET /wnba/scoreboard/today` mounted at `/api`

- [ ] **Step 1: Write failing route tests**

`backend/tests/test_wnba_scoreboard_route.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import wnba_scoreboard as svc

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def clear_cache():
    svc._cache.clear()
    yield
    svc._cache.clear()


def test_scoreboard_today_returns_no_store_and_games():
    espn = json.loads((FIXTURES / "espn_wnba_scoreboard.json").read_text())
    stats = json.loads((FIXTURES / "stats_wnba_scoreboard.json").read_text())

    async def fake_fetch_espn(date_et: str):
        return espn

    async def fake_fetch_stats(date_et: str):
        return stats

    with (
        patch.object(svc, "fetch_espn_scoreboard", side_effect=fake_fetch_espn),
        patch.object(svc, "fetch_stats_scoreboard", side_effect=fake_fetch_stats),
    ):
        client = TestClient(app)
        res = client.get("/api/wnba/scoreboard/today")
    assert res.status_code == 200
    assert res.headers.get("cache-control") == "no-store"
    body = res.json()
    assert body["date"]
    assert len(body["games"]) == 1
    assert body["games"][0]["league"] == "wnba"


def test_scoreboard_stale_while_error_when_both_fail_after_success():
    espn = json.loads((FIXTURES / "espn_wnba_scoreboard.json").read_text())

    async def ok_espn(date_et: str):
        return espn

    async def ok_stats(date_et: str):
        return {"scoreboard": {"games": []}}

    async def boom(date_et: str):
        raise RuntimeError("upstream down")

    with (
        patch.object(svc, "fetch_espn_scoreboard", side_effect=ok_espn),
        patch.object(svc, "fetch_stats_scoreboard", side_effect=ok_stats),
    ):
        client = TestClient(app)
        assert client.get("/api/wnba/scoreboard/today").status_code == 200

    svc._cache["expires_at"] = 0  # force TTL expiry

    with (
        patch.object(svc, "fetch_espn_scoreboard", side_effect=boom),
        patch.object(svc, "fetch_stats_scoreboard", side_effect=boom),
    ):
        res = client.get("/api/wnba/scoreboard/today")
    assert res.status_code == 200
    assert len(res.json()["games"]) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=backend python3 -m pytest backend/tests/test_wnba_scoreboard_route.py -v`

Expected: FAIL (route / symbols missing). Install deps if needed: `pip install httpx pytest`

- [ ] **Step 3: Implement fetchers, cache, route, register**

Extend `backend/app/services/wnba_scoreboard.py` with:

```python
import asyncio
import logging
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

from app.schemas.wnba_scoreboard import WnbaScoreboardResponse

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard"
STATS_URL = "https://stats.wnba.com/stats/scoreboardv3"

_cache: dict = {}  # keys: response, expires_at


def today_et_date() -> str:
    return datetime.now(ET).date().isoformat()


async def fetch_espn_scoreboard(date_et: str) -> dict:
    dates = date_et.replace("-", "")
    async with httpx.AsyncClient(timeout=8.0) as client:
        r = await client.get(ESPN_URL, params={"dates": dates})
        r.raise_for_status()
        return r.json()


async def fetch_stats_scoreboard(date_et: str) -> dict:
    # MM/DD/YYYY for stats.wnba.com
    y, m, d = date_et.split("-")
    game_date = f"{m}/{d}/{y}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.wnba.com/",
        "Origin": "https://www.wnba.com",
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(timeout=8.0, headers=headers) as client:
        r = await client.get(
            STATS_URL,
            params={"GameDate": game_date, "LeagueID": "10"},
        )
        r.raise_for_status()
        return r.json()


async def get_today_scoreboard() -> WnbaScoreboardResponse:
    now = time.time()
    cached = _cache.get("response")
    if cached is not None and now < float(_cache.get("expires_at") or 0):
        return cached

    date_et = today_et_date()
    espn_payload, stats_payload = None, None
    results = await asyncio.gather(
        fetch_espn_scoreboard(date_et),
        fetch_stats_scoreboard(date_et),
        return_exceptions=True,
    )
    if isinstance(results[0], Exception):
        logger.warning("ESPN scoreboard fetch failed: %s", results[0])
    else:
        espn_payload = results[0]
    if isinstance(results[1], Exception):
        logger.warning("stats.wnba scoreboard fetch failed: %s", results[1])
    else:
        stats_payload = results[1]

    if espn_payload is None and stats_payload is None:
        if cached is not None:
            return cached  # stale-while-error
        raise RuntimeError("Both WNBA scoreboard upstreams failed")

    espn_games = (
        normalize_espn_scoreboard(espn_payload, date_et=date_et)
        if espn_payload is not None
        else []
    )
    stats_games = (
        normalize_stats_scoreboard(stats_payload, date_et=date_et)
        if stats_payload is not None
        else []
    )
    games = merge_games(espn_games, stats_games)
    response = WnbaScoreboardResponse(
        date=date_et,
        games=games,
        fetched_at=datetime.now(tz=ET).isoformat(),
    )
    _cache["response"] = response
    _cache["expires_at"] = now + cache_ttl_seconds(games)
    return response
```

`backend/app/api/routes/wnba_scoreboard.py`:

```python
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response

from app.schemas.wnba_scoreboard import WnbaScoreboardResponse
from app.services.wnba_scoreboard import get_today_scoreboard

router = APIRouter(tags=["wnba"])


@router.get("/wnba/scoreboard/today", response_model=WnbaScoreboardResponse)
async def wnba_scoreboard_today(response: Response) -> WnbaScoreboardResponse:
    response.headers["Cache-Control"] = "no-store"
    try:
        return await get_today_scoreboard()
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
```

In `backend/app/main.py`:
- Import `wnba_scoreboard` from `app.api.routes`
- `app.include_router(wnba_scoreboard.router, prefix="/api")`
- Update `description` to note this route calls ESPN + stats.wnba.com (other routes remain DB-only)

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=backend python3 -m pytest backend/tests/test_wnba_scoreboard_route.py backend/tests/test_wnba_scoreboard_normalize.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/wnba_scoreboard.py backend/app/api/routes/wnba_scoreboard.py backend/app/main.py backend/tests/test_wnba_scoreboard_route.py backend/requirements.txt
git commit -m "$(cat <<'EOF'
Add WNBA scoreboard today endpoint with dual-source cache.

EOF
)"
```

---

### Task 4: TanStack Query provider + API client + hook

**Files:**
- Modify: `frontend/package.json` (via npm install)
- Modify: `frontend/vite.config.ts`
- Modify: `frontend/src/main.tsx`
- Create: `frontend/src/lib/api.ts`
- Create: `frontend/src/hooks/useWnbaScoreboard.ts`
- Create: `frontend/src/hooks/useWnbaScoreboard.test.tsx`
- Create: `frontend/src/components/home/mapScoreboard.ts`
- Modify: `frontend/src/components/home/types.ts`

**Interfaces:**
- Produces:
  - `type WnbaScoreboardResponse` (FE mirror of API)
  - `fetchWnbaScoreboard(): Promise<WnbaScoreboardResponse>`
  - `useWnbaScoreboard()` → `{ data, isLoading, isError, games, tickerGames, liveGames, shouldPoll }`
  - `mapToTickerGames` / `mapToLiveGames`
  - Extend `TickerGame` / `LiveGame` with `status: "scheduled" | "live" | "halftime" | "final"`

- [ ] **Step 1: Install dependency + write failing hook test**

```bash
cd frontend && npm install @tanstack/react-query
```

Extend types in `frontend/src/components/home/types.ts`:

```ts
export type GameStatus = "scheduled" | "live" | "halftime" | "final";

export type TickerGame = {
  id: string;
  league: HomeLeague;
  awayAbbrev: string;
  homeAbbrev: string;
  statusLabel: string;
  status: GameStatus;
};

export type LiveGame = {
  id: string;
  league: HomeLeague;
  statusLabel: string;
  status: GameStatus;
  away: LiveGameTeam;
  home: LiveGameTeam;
};
```

`frontend/src/hooks/useWnbaScoreboard.test.tsx`:

```tsx
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { useWnbaScoreboard } from "./useWnbaScoreboard";

const fetchMock = vi.fn();

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

describe("useWnbaScoreboard", () => {
  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("disables refetch interval when all games are final", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        date: "2026-07-29",
        fetched_at: "2026-07-29T12:00:00-04:00",
        games: [
          {
            id: "1",
            league: "wnba",
            status: "final",
            status_label: "Final",
            start_time_et: "2026-07-29T23:00:00Z",
            away: { abbrev: "ATL", name: "Atlanta Dream", score: 80 },
            home: { abbrev: "DAL", name: "Dallas Wings", score: 75 },
          },
        ],
      }),
    });

    const { result } = renderHook(() => useWnbaScoreboard(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.shouldPoll).toBe(false);
  });

  it("enables polling when any game is live", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        date: "2026-07-29",
        fetched_at: "2026-07-29T12:00:00-04:00",
        games: [
          {
            id: "1",
            league: "wnba",
            status: "live",
            status_label: "Q3 7:13",
            start_time_et: "2026-07-29T23:00:00Z",
            away: { abbrev: "ATL", name: "Atlanta Dream", score: 36 },
            home: { abbrev: "DAL", name: "Dallas Wings", score: 44 },
          },
        ],
      }),
    });

    const { result } = renderHook(() => useWnbaScoreboard(), { wrapper });
    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.shouldPoll).toBe(true);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm run test -- src/hooks/useWnbaScoreboard.test.tsx`

Expected: FAIL (module not found)

- [ ] **Step 3: Implement api, mappers, hook, provider, proxy**

`frontend/src/lib/api.ts`:

```ts
export type ApiGameStatus = "scheduled" | "live" | "halftime" | "final";

export type ApiWnbaTeam = {
  abbrev: string;
  name: string;
  score: number | null;
};

export type ApiWnbaGame = {
  id: string;
  league: "wnba";
  status: ApiGameStatus;
  status_label: string;
  away: ApiWnbaTeam;
  home: ApiWnbaTeam;
  start_time_et: string;
};

export type WnbaScoreboardResponse = {
  date: string;
  games: ApiWnbaGame[];
  fetched_at: string;
};

export async function fetchWnbaScoreboard(): Promise<WnbaScoreboardResponse> {
  const res = await fetch("/api/wnba/scoreboard/today", {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`Scoreboard request failed: ${res.status}`);
  }
  return res.json();
}
```

`frontend/src/components/home/mapScoreboard.ts`:

```ts
import type { ApiWnbaGame } from "@/lib/api";
import type { LiveGame, TickerGame } from "./types";

export function mapToTickerGames(games: ApiWnbaGame[]): TickerGame[] {
  return games.map((g) => ({
    id: g.id,
    league: g.league,
    awayAbbrev: g.away.abbrev,
    homeAbbrev: g.home.abbrev,
    statusLabel: g.status_label,
    status: g.status,
  }));
}

export function mapToLiveGames(games: ApiWnbaGame[]): LiveGame[] {
  return games.map((g) => ({
    id: g.id,
    league: g.league,
    statusLabel: g.status_label,
    status: g.status,
    away: {
      abbrev: g.away.abbrev,
      name: g.away.name,
      score: g.away.score,
    },
    home: {
      abbrev: g.home.abbrev,
      name: g.home.name,
      score: g.home.score,
    },
  }));
}

export function shouldPollScoreboard(games: ApiWnbaGame[] | undefined): boolean {
  if (!games || games.length === 0) return false;
  return games.some((g) => g.status !== "final");
}
```

`frontend/src/hooks/useWnbaScoreboard.ts`:

```ts
import { useQuery } from "@tanstack/react-query";
import { fetchWnbaScoreboard } from "@/lib/api";
import {
  mapToLiveGames,
  mapToTickerGames,
  shouldPollScoreboard,
} from "@/components/home/mapScoreboard";

const REFETCH_MS = 18_000;

export function useWnbaScoreboard() {
  const query = useQuery({
    queryKey: ["wnba", "scoreboard", "today"],
    queryFn: fetchWnbaScoreboard,
    refetchInterval: (q) =>
      shouldPollScoreboard(q.state.data?.games) ? REFETCH_MS : false,
  });

  const games = query.data?.games ?? [];
  return {
    ...query,
    games,
    tickerGames: mapToTickerGames(games),
    liveGames: mapToLiveGames(games),
    shouldPoll: shouldPollScoreboard(query.data?.games),
  };
}
```

Update `frontend/src/main.tsx`:

```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AppRouter } from "./AppRouter";
import "./index.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 10_000,
      refetchOnWindowFocus: true,
    },
  },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AppRouter />
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);
```

In `frontend/vite.config.ts`, add:

```ts
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
```

- [ ] **Step 4: Run hook tests**

Run: `cd frontend && npm run test -- src/hooks/useWnbaScoreboard.test.tsx`

Expected: PASS. Also update any existing `LiveGame` fixtures in tests to include `status` if TypeScript fails.

- [ ] **Step 5: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/vite.config.ts frontend/src/main.tsx frontend/src/lib/api.ts frontend/src/hooks/useWnbaScoreboard.ts frontend/src/hooks/useWnbaScoreboard.test.tsx frontend/src/components/home/mapScoreboard.ts frontend/src/components/home/types.ts
git commit -m "$(cat <<'EOF'
Add TanStack Query WNBA scoreboard hook and API client.

EOF
)"
```

---

### Task 5: LIVE NOW UI (loading / empty / violet / pulse)

**Files:**
- Modify: `frontend/src/components/home/LiveNowSection.tsx`
- Modify: `frontend/src/components/home/LiveNowSection.test.tsx`
- Modify: `frontend/src/components/home/format.ts` (optional helper for in-progress count)

**Interfaces:**
- Consumes: `LiveGame` with `status`
- Produces: `LiveNowSection({ games?, isLoading? })`
  - Loading (`isLoading && no games`): skeletons
  - Loaded empty: `0 games in progress`, **no** skeletons
  - Loaded with games: cards for full slate; subtitle counts only `live` + `halftime`
  - Violet WNBA pill; pulse + red status only when live/halftime

- [ ] **Step 1: Rewrite failing/updated tests**

Replace `frontend/src/components/home/LiveNowSection.test.tsx` with:

```tsx
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { LiveNowSection } from "./LiveNowSection";
import type { LiveGame } from "./types";

const liveGame: LiveGame = {
  id: "g1",
  league: "wnba",
  status: "live",
  statusLabel: "Q3 7:13",
  away: { abbrev: "ATL", name: "Atlanta Dream", score: 36 },
  home: { abbrev: "DAL", name: "Dallas Wings", score: 44 },
};

const finalGame: LiveGame = {
  id: "g2",
  league: "wnba",
  status: "final",
  statusLabel: "Final",
  away: { abbrev: "NYL", name: "New York Liberty", score: 90 },
  home: { abbrev: "LAS", name: "Los Angeles Sparks", score: 80 },
};

describe("LiveNowSection", () => {
  it("shows skeletons while loading with no games", () => {
    const { container } = render(<LiveNowSection isLoading games={[]} />);
    expect(screen.getByText("0 games in progress")).toBeInTheDocument();
    expect(container.querySelectorAll("article[aria-hidden]")).toHaveLength(3);
  });

  it("shows empty state without skeletons when loaded with zero games", () => {
    const { container } = render(<LiveNowSection isLoading={false} games={[]} />);
    expect(screen.getByText("0 games in progress")).toBeInTheDocument();
    expect(container.querySelectorAll("article[aria-hidden]")).toHaveLength(0);
  });

  it("counts only in-progress games in the subtitle", () => {
    render(<LiveNowSection games={[liveGame, finalGame]} />);
    expect(screen.getByText("1 game in progress")).toBeInTheDocument();
    expect(screen.getByText("ATL")).toBeInTheDocument();
    expect(screen.getByText("NYL")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npm run test -- src/components/home/LiveNowSection.test.tsx`

Expected: FAIL (old empty behavior still shows skeletons; missing `isLoading` / status)

- [ ] **Step 3: Update LiveNowSection**

```tsx
type LiveNowSectionProps = {
  games?: LiveGame[];
  isLoading?: boolean;
};

const leaguePill: Record<HomeLeague, string> = {
  nba: "bg-sky-600/90 text-white",
  wnba: "bg-violet-600/90 text-white",
};

function LiveGameCard({ game }: { game: LiveGame }) {
  const inProgress = game.status === "live" || game.status === "halftime";
  return (
    <article className="rounded-xl border border-white/10 bg-[#141414] p-4">
      <div className="mb-4 flex items-center justify-between">
        <span
          className={`rounded-full px-2 py-0.5 text-[10px] font-semibold tracking-wide uppercase ${leaguePill[game.league]}`}
        >
          {game.league}
        </span>
        <span
          className={`flex items-center gap-2 text-xs ${
            inProgress ? "text-red-400" : "text-white/55"
          }`}
        >
          {inProgress ? (
            <span className="size-1.5 animate-pulse rounded-full bg-red-500" />
          ) : null}
          {game.statusLabel}
        </span>
      </div>
      {/* team rows unchanged */}
    </article>
  );
}

export function LiveNowSection({
  games,
  isLoading = false,
}: LiveNowSectionProps) {
  const list = normalizeLiveGames(games);
  const inProgressCount = list.filter(
    (g) => g.status === "live" || g.status === "halftime",
  ).length;
  const showSkeletons = isLoading && list.length === 0;

  return (
    <section id="live-now" className="mx-auto max-w-6xl px-4 pb-10 sm:px-6">
      <SectionHeading
        title="Live Now"
        subtitle={formatGamesInProgress(inProgressCount)}
      />
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {showSkeletons
          ? Array.from({ length: LIVE_NOW_SKELETON_COUNT }, (_, i) => (
              <SkeletonGameCard key={i} />
            ))
          : list.map((game) => <LiveGameCard key={game.id} game={game} />)}
      </div>
    </section>
  );
}
```

Keep existing team-row markup from the current file.

- [ ] **Step 4: Run tests**

Run: `cd frontend && npm run test -- src/components/home/LiveNowSection.test.tsx`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/home/LiveNowSection.tsx frontend/src/components/home/LiveNowSection.test.tsx
git commit -m "$(cat <<'EOF'
Update LIVE NOW for loading, empty, and live status styling.

EOF
)"
```

---

### Task 6: Wire hook into chrome + home page

**Files:**
- Modify: `frontend/src/layouts/HomeChromeLayout.tsx`
- Modify: `frontend/src/pages/HomePage.tsx`
- Create or modify: `frontend/src/layouts/HomeChromeLayout.test.tsx` (optional light test)
- Fix any broken home tests that construct `LiveGame` without `status`

**Interfaces:**
- Consumes: `useWnbaScoreboard()`
- Produces: Live ticker + LIVE NOW fed from shared query cache

- [ ] **Step 1: Write a small layout wiring test (optional but preferred)**

If adding `HomeChromeLayout.test.tsx`, mock the hook:

```tsx
import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { HomeChromeLayout } from "./HomeChromeLayout";

vi.mock("@/hooks/useWnbaScoreboard", () => ({
  useWnbaScoreboard: () => ({
    isLoading: false,
    tickerGames: [
      {
        id: "1",
        league: "wnba",
        awayAbbrev: "ATL",
        homeAbbrev: "DAL",
        statusLabel: "Q3 7:13",
        status: "live",
      },
    ],
    liveGames: [],
  }),
}));

describe("HomeChromeLayout", () => {
  it("renders ticker games from scoreboard hook", () => {
    render(
      <MemoryRouter initialEntries={["/"]}>
        <Routes>
          <Route element={<HomeChromeLayout />}>
            <Route path="/" element={<div>home</div>} />
          </Route>
        </Routes>
      </MemoryRouter>,
    );
    expect(screen.getByText("ATL")).toBeInTheDocument();
    expect(screen.getByText("DAL")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails (before wiring)**

Run: `cd frontend && npm run test -- src/layouts/HomeChromeLayout.test.tsx`

Expected: FAIL until layout uses hook (or pass mock but ticker empty — assert fails)

- [ ] **Step 3: Wire components**

`frontend/src/layouts/HomeChromeLayout.tsx`:

```tsx
import { Outlet } from "react-router-dom";
import { HomeNav } from "@/components/home/HomeNav";
import { LiveTicker } from "@/components/home/LiveTicker";
import { useWnbaScoreboard } from "@/hooks/useWnbaScoreboard";

export function HomeChromeLayout() {
  const { tickerGames } = useWnbaScoreboard();
  return (
    <div className="min-h-screen bg-black text-white">
      <HomeNav />
      <LiveTicker games={tickerGames} />
      <main>
        <Outlet />
      </main>
    </div>
  );
}
```

`frontend/src/pages/HomePage.tsx`:

```tsx
import { TicketHero } from "@/components/home/TicketHero";
import { LiveNowSection } from "@/components/home/LiveNowSection";
import { StoriesSection } from "@/components/home/StoriesSection";
import { ExploreSection } from "@/components/home/ExploreSection";
import { LearnTheGameSection } from "@/components/home/LearnTheGameSection";
import { useWnbaScoreboard } from "@/hooks/useWnbaScoreboard";

export function HomePage() {
  const { liveGames, isLoading } = useWnbaScoreboard();
  return (
    <>
      <TicketHero />
      <LiveNowSection games={liveGames} isLoading={isLoading} />
      <StoriesSection />
      <ExploreSection />
      <LearnTheGameSection />
    </>
  );
}
```

Both hooks share `queryKey` so only one network poll runs.

- [ ] **Step 4: Full verification**

```bash
PYTHONPATH=backend python3 -m pytest backend/tests/test_wnba_scoreboard_normalize.py backend/tests/test_wnba_scoreboard_route.py -v
cd frontend && npm run test && npm run build
```

Expected: all PASS / build succeeds.

Manual smoke (optional): start backend (`uvicorn app.main:app --app-dir backend --reload`) + `npm run dev`, open `/`, confirm ticker + LIVE NOW populate for today's WNBA slate.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/layouts/HomeChromeLayout.tsx frontend/src/pages/HomePage.tsx frontend/src/layouts/HomeChromeLayout.test.tsx
git commit -m "$(cat <<'EOF'
Wire WNBA scoreboard query into live ticker and LIVE NOW.

EOF
)"
```

---

## Spec coverage self-check

| Spec requirement | Task |
| --- | --- |
| `GET /api/wnba/scoreboard/today` | 3 |
| Parallel ESPN + stats.wnba.com | 3 |
| Normalize + Pydantic schema | 1–2 |
| Field-level merge | 2 |
| Cache 30–60s by game state | 2–3 |
| `Cache-Control: no-store` | 3 |
| Stale-while-error on dual failure | 3 |
| Today = America/New_York | 3 (`today_et_date`) |
| TanStack Query + 18s poll | 4 |
| Stop poll when all Final / empty | 4 |
| Ticker + LIVE NOW full slate | 5–6 |
| Violet pill, pulse only in-progress | 5 |
| Loading skeletons / empty no skeletons | 5 |
| No logos / no NBA endpoint | honored (out of scope) |

## Placeholder / consistency notes

- API field names use snake_case (`status_label`, `start_time_et`) on the wire; FE maps to camelCase UI types.
- `LiveGame.status` required after Task 4 — update any leftover fixtures in the same commits that touch those tests.
- stats.wnba.com may require browser-like headers; fixtures cover offline tests regardless of live upstream availability.
