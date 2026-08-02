# MLB Home Live Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add MLB to home nav, live ticker, and Live Now via `GET /api/mlb/scoreboard/today` (MLB Stats API), plus `/mlb/matchups` and `/mlb/games/:gamePk` coming-soon stubs.

**Architecture:** Parallel league scoreboard (same pattern as WNBA). Backend normalizes `statsapi.mlb.com` schedule into a WNBA-shaped payload with `mlb_game_pk`. Frontend adds `useMlbScoreboard`, merges with WNBA for ticker/Live Now, and links MLB rows to stub game routes.

**Tech Stack:** FastAPI · Pydantic · httpx · React 19 · TypeScript · Vite 6 · TanStack Query · React Router · Vitest · pytest · OpenAPI codegen

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-02-mlb-home-live-design.md`
- Coding standards: `CLAUDE.md` (small focused modules, strong typing, tests with code)
- Upstream: MLB Stats API only — `https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={YYYY-MM-DD}&hydrate=team,linescore`
- Nav logo CDN: `https://a.espncdn.com/i/teamlogos/leagues/500/mlb.png`
- Team logos: `https://www.mlbstatic.com/team-logos/{teamId}.svg`
- Hard upstream failure → **502** + `Cache-Control: no-store` (mirror WNBA)
- Client merge: MLB failure must not wipe WNBA (and vice versa)
- No Statcast, props, real matchups panel, About MLB rows, or NBA live scoreboard
- Slate date: reuse the same ET rollover idea as WNBA (`slate_et_date` at 3:00 AM ET) — implement a local copy in `mlb_scoreboard.py` (do not import from WNBA service) unless a tiny shared util already exists
- Verify backend: `python3 -m pytest backend/tests/test_mlb_scoreboard_normalize.py backend/tests/test_mlb_scoreboard_route.py -v`
- Verify frontend: `cd frontend && npm run test && npm run check:api && npm run build`
- OpenAPI: `python3 scripts/export_openapi.py` then `cd frontend && npm run generate:api`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `backend/app/schemas/mlb_scoreboard.py` | `MlbTeam`, `MlbGame`, `MlbScoreboardResponse` |
| `backend/app/services/mlb_scoreboard.py` | Normalize schedule, fetch, cache, `get_today_scoreboard` |
| `backend/app/api/routes/mlb_scoreboard.py` | `GET /mlb/scoreboard/today` |
| `backend/app/main.py` | Mount MLB router; mention in description |
| `backend/app/openapi_export.py` | Require `/api/mlb/scoreboard/today` |
| `backend/tests/fixtures/mlb_statsapi_schedule.json` | Minimal schedule fixture |
| `backend/tests/test_mlb_scoreboard_normalize.py` | Status / score / logo / venue mapping |
| `backend/tests/test_mlb_scoreboard_route.py` | Route 200 + 502 smoke |
| `frontend/openapi.json` + `src/lib/api.schema.d.ts` | Regenerated types |
| `frontend/src/lib/api.ts` | `fetchMlbScoreboard`, `ApiMlbGame` types |
| `frontend/src/components/home/types.ts` | `HomeLeague` includes `"mlb"`; optional `mlbGamePk` |
| `frontend/src/components/home/gameDetailHref.ts` | League-aware detail path |
| `frontend/src/components/home/mapScoreboard.ts` | Map MLB + WNBA; pass through `mlbGamePk` |
| `frontend/src/components/home/mergeLeagueScoreboards.ts` | Merge ticker/live lists + error flags |
| `frontend/src/hooks/useMlbScoreboard.ts` | TanStack Query wrapper |
| `frontend/src/components/home/HomeNav.tsx` | MLB nav entry |
| `frontend/src/components/league/LeagueHero.tsx` | MLB hero copy + CDN logo |
| `frontend/src/pages/LeagueMatchupsPage.tsx` | MLB coming-soon branch |
| `frontend/src/pages/MlbGameStubPage.tsx` | Game detail stub |
| `frontend/src/AppRouter.tsx` | `/mlb/matchups`, `/mlb/games/:gamePk` |
| `frontend/src/layouts/HomeChromeLayout.tsx` | Dual hooks + merge for ticker |
| `frontend/src/pages/HomePage.tsx` | Dual hooks + merge for Live Now |
| `frontend/src/components/home/LiveTicker.tsx` | Link via `gameDetailHref` |
| `frontend/src/components/home/LiveNowSection.tsx` | Link via `gameDetailHref` |
| `docs/superpowers/specs/2026-08-02-website-api-system-design.md` | Page ↔ API table update |

---

### Task 1: MLB scoreboard schema + normalize (backend)

**Files:**
- Create: `backend/app/schemas/mlb_scoreboard.py`
- Create: `backend/app/services/mlb_scoreboard.py` (normalize helpers only first; fetch/cache in Task 2)
- Create: `backend/tests/fixtures/mlb_statsapi_schedule.json`
- Create: `backend/tests/test_mlb_scoreboard_normalize.py`

**Interfaces:**
- Produces:
  - `normalize_mlb_schedule(payload: dict, *, date_et: str) -> list[MlbGame]`
  - `_mlb_status(status: dict, linescore: dict | None, game_date: str) -> tuple[GameStatus, str]`
  - `_team_logo_url(team_id: int | None) -> str | None`
  - `MlbGame.mlb_game_pk: str`, `league: Literal["mlb"]`
- Consumes: Stats API schedule JSON shape (`dates[].games[]`)

- [ ] **Step 1: Write the fixture**

Create `backend/tests/fixtures/mlb_statsapi_schedule.json` with three games (Preview, Live, Final). Minimal shape:

```json
{
  "dates": [
    {
      "date": "2026-08-02",
      "games": [
        {
          "gamePk": 824900,
          "gameDate": "2026-08-02T23:10:00Z",
          "status": {
            "abstractGameState": "Preview",
            "detailedState": "Scheduled",
            "codedGameState": "S"
          },
          "teams": {
            "away": {
              "score": 0,
              "leagueRecord": { "wins": 50, "losses": 50 },
              "team": {
                "id": 147,
                "abbreviation": "NYY",
                "name": "New York Yankees"
              }
            },
            "home": {
              "score": 0,
              "leagueRecord": { "wins": 55, "losses": 45 },
              "team": {
                "id": 121,
                "abbreviation": "NYM",
                "name": "New York Mets"
              }
            }
          },
          "venue": { "name": "Citi Field" }
        },
        {
          "gamePk": 824971,
          "gameDate": "2026-08-02T17:10:00Z",
          "status": {
            "abstractGameState": "Live",
            "detailedState": "In Progress",
            "codedGameState": "I"
          },
          "linescore": {
            "currentInning": 8,
            "currentInningOrdinal": "8th",
            "inningState": "Top",
            "isTopInning": true
          },
          "teams": {
            "away": {
              "score": 9,
              "leagueRecord": { "wins": 40, "losses": 60 },
              "team": {
                "id": 139,
                "abbreviation": "TB",
                "name": "Tampa Bay Rays"
              }
            },
            "home": {
              "score": 0,
              "leagueRecord": { "wins": 48, "losses": 52 },
              "team": {
                "id": 141,
                "abbreviation": "TOR",
                "name": "Toronto Blue Jays"
              }
            }
          },
          "venue": { "name": "Rogers Centre" }
        },
        {
          "gamePk": 824807,
          "gameDate": "2026-08-02T17:35:00Z",
          "status": {
            "abstractGameState": "Final",
            "detailedState": "Final",
            "codedGameState": "F"
          },
          "teams": {
            "away": {
              "score": 8,
              "leagueRecord": { "wins": 59, "losses": 53 },
              "team": {
                "id": 143,
                "abbreviation": "PHI",
                "name": "Philadelphia Phillies"
              }
            },
            "home": {
              "score": 2,
              "leagueRecord": { "wins": 45, "losses": 65 },
              "team": {
                "id": 110,
                "abbreviation": "BAL",
                "name": "Baltimore Orioles"
              }
            }
          },
          "venue": { "name": "Oriole Park at Camden Yards" }
        }
      ]
    }
  ]
}
```

Also cover postponed in a unit-test-built payload (not required in fixture file): `abstractGameState: "Preview"` or `"Final"` with `detailedState: "Postponed"` — map to `status="scheduled"`, `status_label="Postponed"`.

- [ ] **Step 2: Write failing normalize tests**

```python
# backend/tests/test_mlb_scoreboard_normalize.py
from __future__ import annotations

import json
from pathlib import Path

from app.services.mlb_scoreboard import normalize_mlb_schedule

FIXTURES = Path(__file__).parent / "fixtures"


def test_normalize_mlb_schedule_maps_preview_live_final():
    payload = json.loads((FIXTURES / "mlb_statsapi_schedule.json").read_text())
    games = normalize_mlb_schedule(payload, date_et="2026-08-02")
    by_pk = {g.mlb_game_pk: g for g in games}

    preview = by_pk["824900"]
    assert preview.league == "mlb"
    assert preview.id == "mlb-824900"
    assert preview.status == "scheduled"
    assert preview.away.abbrev == "NYY"
    assert preview.home.abbrev == "NYM"
    assert preview.away.score is None  # hide scores for scheduled
    assert preview.away.logo_url == "https://www.mlbstatic.com/team-logos/147.svg"
    assert preview.venue == "Citi Field"

    live = by_pk["824971"]
    assert live.status == "live"
    assert live.status_label == "Top 8th"
    assert live.away.score == 9
    assert live.home.score == 0
    assert live.away.record == "40-60"

    final = by_pk["824807"]
    assert final.status == "final"
    assert final.status_label == "Final"
    assert final.away.score == 8


def test_normalize_postponed_is_scheduled_with_label():
    payload = {
        "dates": [
            {
                "date": "2026-08-02",
                "games": [
                    {
                        "gamePk": 1,
                        "gameDate": "2026-08-02T23:00:00Z",
                        "status": {
                            "abstractGameState": "Final",
                            "detailedState": "Postponed",
                            "codedGameState": "D",
                        },
                        "teams": {
                            "away": {
                                "team": {
                                    "id": 1,
                                    "abbreviation": "AAA",
                                    "name": "Away",
                                }
                            },
                            "home": {
                                "team": {
                                    "id": 2,
                                    "abbreviation": "HHH",
                                    "name": "Home",
                                }
                            },
                        },
                    }
                ],
            }
        ]
    }
    games = normalize_mlb_schedule(payload, date_et="2026-08-02")
    assert len(games) == 1
    assert games[0].status == "scheduled"
    assert games[0].status_label == "Postponed"
```

- [ ] **Step 3: Run tests — expect FAIL**

Run: `python3 -m pytest backend/tests/test_mlb_scoreboard_normalize.py -v`  
Expected: FAIL (module / function missing)

- [ ] **Step 4: Implement schema + normalize**

`backend/app/schemas/mlb_scoreboard.py` — mirror WNBA models:

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

GameStatus = Literal["scheduled", "live", "halftime", "final"]

_RESPONSE_CONFIG = ConfigDict(json_schema_serialization_defaults_required=True)


class MlbTeam(BaseModel):
    model_config = _RESPONSE_CONFIG

    abbrev: str
    name: str
    score: int | None = None
    record: str | None = None
    logo_url: str | None = None


class MlbGame(BaseModel):
    model_config = _RESPONSE_CONFIG

    id: str
    mlb_game_pk: str
    league: Literal["mlb"] = "mlb"
    status: GameStatus
    status_label: str
    away: MlbTeam
    home: MlbTeam
    start_time_et: str
    venue: str | None = None
    venue_city: str | None = None


class MlbScoreboardResponse(BaseModel):
    model_config = _RESPONSE_CONFIG

    date: str = Field(description="YYYY-MM-DD in America/New_York")
    games: list[MlbGame]
    fetched_at: str
```

In `mlb_scoreboard.py`, implement:

- `TEAM_LOGO = "https://www.mlbstatic.com/team-logos/{id}.svg"`
- Status mapping:
  - If `detailedState` contains `Postponed` / `Cancelled` / `Suspended` (case-insensitive) → `("scheduled", detailedState)`
  - `abstractGameState == "Final"` → `("final", "Final")` (or detailedState if useful)
  - `abstractGameState == "Live"` → `("live", f"{inningState} {currentInningOrdinal}")` e.g. `Top 8th`; fallback `"Live"`
  - else → `("scheduled", format_tip_label(gameDate) or detailedState or "Scheduled")`
- Scores: `None` when status is `scheduled`; otherwise int from team side
- Record: `f"{wins}-{losses}"` from `leagueRecord` when present
- `venue` from `game["venue"]["name"]`; `venue_city` `None` unless address city exists (optional)
- Sort games by `start_time_et` then `id`

- [ ] **Step 5: Run tests — expect PASS**

Run: `python3 -m pytest backend/tests/test_mlb_scoreboard_normalize.py -v`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/schemas/mlb_scoreboard.py \
  backend/app/services/mlb_scoreboard.py \
  backend/tests/fixtures/mlb_statsapi_schedule.json \
  backend/tests/test_mlb_scoreboard_normalize.py
git commit -m "$(cat <<'EOF'
feat(mlb): normalize Stats API schedule into scoreboard games

EOF
)"
```

---

### Task 2: Fetch, cache, and route

**Files:**
- Modify: `backend/app/services/mlb_scoreboard.py`
- Create: `backend/app/api/routes/mlb_scoreboard.py`
- Modify: `backend/app/main.py`
- Create: `backend/tests/test_mlb_scoreboard_route.py`

**Interfaces:**
- Consumes: `normalize_mlb_schedule`
- Produces: `async def get_today_scoreboard() -> MlbScoreboardResponse`
- Route: `GET /api/mlb/scoreboard/today` → 200 or 502

- [ ] **Step 1: Write failing route test**

```python
# backend/tests/test_mlb_scoreboard_route.py
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.mlb_scoreboard import MlbGame, MlbScoreboardResponse, MlbTeam


@pytest.fixture
def client():
    return TestClient(app)


def _sample_response() -> MlbScoreboardResponse:
    team = MlbTeam(abbrev="NYY", name="New York Yankees", score=1)
    home = MlbTeam(abbrev="BOS", name="Boston Red Sox", score=0)
    game = MlbGame(
        id="mlb-1",
        mlb_game_pk="1",
        status="live",
        status_label="Top 1st",
        away=team,
        home=home,
        start_time_et="2026-08-02T17:00:00Z",
    )
    return MlbScoreboardResponse(
        date="2026-08-02",
        games=[game],
        fetched_at="2026-08-02T12:00:00+00:00",
    )


def test_mlb_scoreboard_today_ok(client):
    with patch(
        "app.api.routes.mlb_scoreboard.get_today_scoreboard",
        new=AsyncMock(return_value=_sample_response()),
    ):
        res = client.get("/api/mlb/scoreboard/today")
    assert res.status_code == 200
    assert res.headers.get("cache-control") == "no-store"
    body = res.json()
    assert body["games"][0]["league"] == "mlb"
    assert body["games"][0]["mlb_game_pk"] == "1"


def test_mlb_scoreboard_today_upstream_failure_is_502(client):
    with patch(
        "app.api.routes.mlb_scoreboard.get_today_scoreboard",
        new=AsyncMock(side_effect=RuntimeError("upstream down")),
    ):
        res = client.get("/api/mlb/scoreboard/today")
    assert res.status_code == 502
    assert res.headers.get("cache-control") == "no-store"
```

- [ ] **Step 2: Run route test — expect FAIL**

Run: `python3 -m pytest backend/tests/test_mlb_scoreboard_route.py -v`  
Expected: FAIL (router missing / 404)

- [ ] **Step 3: Implement fetch + cache + route**

In `mlb_scoreboard.py` add (pattern from `wnba_scoreboard.py`, simplified — no ESPN merge, no overnight carryover required for v1 unless easy):

```python
SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
TIMEOUT_SECONDS = 12.0

async def fetch_mlb_schedule(date_et: str) -> dict:
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        r = await client.get(
            SCHEDULE_URL,
            params={
                "sportId": 1,
                "date": date_et,
                "hydrate": "team,linescore",
            },
        )
        r.raise_for_status()
        return r.json()

async def get_today_scoreboard() -> MlbScoreboardResponse:
    # cache TTL: 30s if any live else 60s; key by slate_et_date()
    # on miss: fetch → normalize → MlbScoreboardResponse(date, games, fetched_at=utcnow iso)
    ...
```

Implement `slate_et_date()` locally (copy WNBA 3 AM ET rollover logic).

Route file:

```python
# backend/app/api/routes/mlb_scoreboard.py
router = APIRouter(tags=["mlb"])

@router.get("/mlb/scoreboard/today", response_model=MlbScoreboardResponse)
async def mlb_scoreboard_today(response: Response) -> MlbScoreboardResponse:
    response.headers["Cache-Control"] = "no-store"
    try:
        return await get_today_scoreboard()
    except Exception as exc:
        logger.warning("MLB scoreboard unavailable: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="MLB scoreboard is temporarily unavailable",
            headers={"Cache-Control": "no-store"},
        ) from exc
```

Mount in `main.py`:

```python
from app.api.routes import mlb_scoreboard
# ...
app.include_router(mlb_scoreboard.router, prefix="/api")
```

Update `app` description string to mention `/api/mlb/scoreboard/today`.

- [ ] **Step 4: Run tests — expect PASS**

Run: `python3 -m pytest backend/tests/test_mlb_scoreboard_normalize.py backend/tests/test_mlb_scoreboard_route.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/mlb_scoreboard.py \
  backend/app/api/routes/mlb_scoreboard.py \
  backend/app/main.py \
  backend/tests/test_mlb_scoreboard_route.py
git commit -m "$(cat <<'EOF'
feat(mlb): expose /api/mlb/scoreboard/today from Stats API

EOF
)"
```

---

### Task 3: OpenAPI export + frontend API client

**Files:**
- Modify: `backend/app/openapi_export.py`
- Modify: `backend/tests/test_export_openapi.py`
- Modify: `frontend/openapi.json` (via export script)
- Modify: `frontend/src/lib/api.schema.d.ts` (via generate)
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/lib/api.test.ts` (add MLB fetch case if file tests fetch helpers)

**Interfaces:**
- Produces: `fetchMlbScoreboard(): Promise<MlbScoreboardResponse>`
- Types: `ApiMlbGame`, `ApiMlbTeam`, `MlbScoreboardResponse` from OpenAPI schemas

- [ ] **Step 1: Require MLB path in export test**

In `openapi_export.py`:

```python
REQUIRED_MLB_PATHS = (
    "/api/mlb/scoreboard/today",
)

REQUIRED_FRONTEND_PATHS = REQUIRED_WNBA_PATHS + REQUIRED_MLB_PATHS
```

Update `test_export_openapi.py` to assert `/api/mlb/scoreboard/today` is present (rename helper if needed).

- [ ] **Step 2: Export + generate**

```bash
python3 scripts/export_openapi.py
cd frontend && npm run generate:api
```

Expected: `openapi.json` includes `MlbScoreboardResponse` / `MlbGame` with `mlb_game_pk`.

- [ ] **Step 3: Add client helper**

In `api.ts`:

```typescript
export type ApiMlbTeam = Schemas["MlbTeam"];
export type ApiMlbGame = Schemas["MlbGame"];
export type MlbScoreboardResponse = Schemas["MlbScoreboardResponse"];

export async function fetchMlbScoreboard(): Promise<MlbScoreboardResponse> {
  const res = await fetch(`${API_BASE}/api/mlb/scoreboard/today`, {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`MLB scoreboard request failed: ${res.status}`);
  }
  return res.json();
}
```

- [ ] **Step 4: Verify**

Run: `python3 -m pytest backend/tests/test_export_openapi.py -v`  
Run: `cd frontend && npm run check:api`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/openapi_export.py backend/tests/test_export_openapi.py \
  frontend/openapi.json frontend/src/lib/api.schema.d.ts frontend/src/lib/api.ts \
  frontend/src/lib/api.test.ts
git commit -m "$(cat <<'EOF'
feat(mlb): wire OpenAPI types and fetchMlbScoreboard client

EOF
)"
```

---

### Task 4: Frontend types, mappers, and detail href

**Files:**
- Modify: `frontend/src/components/home/types.ts`
- Create: `frontend/src/components/home/gameDetailHref.ts`
- Create: `frontend/src/components/home/gameDetailHref.test.ts`
- Modify: `frontend/src/components/home/mapScoreboard.ts`
- Modify: `frontend/src/components/home/mapScoreboard.test.ts`
- Modify: `frontend/src/components/league/types.ts` (via `HomeLeague` import — ensure `MatchupGame` gets `mlbGamePk?`)

**Interfaces:**
- Produces:
  - `HomeLeague = "nba" | "wnba" | "mlb"`
  - `TickerGame.mlbGamePk?: string | null`
  - `LiveGame.mlbGamePk?: string | null`
  - `gameDetailHref(game: { league; espnEventId?; mlbGamePk? }): string | null`
  - Mappers accept `ApiWnbaGame | ApiMlbGame` (discriminate on `league` / presence of ids)

- [ ] **Step 1: Failing tests for href + mapper**

```typescript
// gameDetailHref.test.ts
import { describe, expect, it } from "vitest";
import { gameDetailHref } from "./gameDetailHref";

describe("gameDetailHref", () => {
  it("returns WNBA espn path", () => {
    expect(
      gameDetailHref({
        league: "wnba",
        espnEventId: "401",
        mlbGamePk: null,
      }),
    ).toBe("/games/401");
  });

  it("returns MLB stub path", () => {
    expect(
      gameDetailHref({
        league: "mlb",
        espnEventId: null,
        mlbGamePk: "824971",
      }),
    ).toBe("/mlb/games/824971");
  });

  it("returns null for NBA / missing ids", () => {
    expect(
      gameDetailHref({ league: "nba", espnEventId: null, mlbGamePk: null }),
    ).toBeNull();
  });
});
```

Extend `mapScoreboard.test.ts` with an MLB game object (`league: "mlb", mlb_game_pk: "9", espn_event_id` absent) asserting `mlbGamePk === "9"` and `espnEventId` undefined/null.

- [ ] **Step 2: Run — expect FAIL**

Run: `cd frontend && npx vitest run src/components/home/gameDetailHref.test.ts src/components/home/mapScoreboard.test.ts`  
Expected: FAIL

- [ ] **Step 3: Implement**

```typescript
// types.ts
export type HomeLeague = "nba" | "wnba" | "mlb";
// add mlbGamePk?: string | null to TickerGame and LiveGame

// gameDetailHref.ts
export function gameDetailHref(game: {
  league: HomeLeague;
  espnEventId?: string | null;
  mlbGamePk?: string | null;
}): string | null {
  if (game.league === "mlb" && game.mlbGamePk) {
    return `/mlb/games/${game.mlbGamePk}`;
  }
  if (game.espnEventId) {
    return `/games/${game.espnEventId}`;
  }
  return null;
}
```

Update mappers to a shared input type:

```typescript
type ScoreboardGame = {
  id: string;
  league: HomeLeague;
  status: GameStatus;
  status_label: string;
  espn_event_id?: string | null;
  mlb_game_pk?: string | null;
  away: { abbrev: string; name: string; score: number | null; logo_url: string | null; record?: string | null };
  home: { abbrev: string; name: string; score: number | null; logo_url: string | null; record?: string | null };
  venue?: string | null;
  venue_city?: string | null;
};
```

Map `mlbGamePk: g.mlb_game_pk ?? null` and `espnEventId: g.espn_event_id ?? null`.

- [ ] **Step 4: Run — expect PASS**

Run: `cd frontend && npx vitest run src/components/home/gameDetailHref.test.ts src/components/home/mapScoreboard.test.ts`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/home/types.ts \
  frontend/src/components/home/gameDetailHref.ts \
  frontend/src/components/home/gameDetailHref.test.ts \
  frontend/src/components/home/mapScoreboard.ts \
  frontend/src/components/home/mapScoreboard.test.ts \
  frontend/src/components/league/types.ts
git commit -m "$(cat <<'EOF'
feat(mlb): extend home types and scoreboard mappers for MLB ids

EOF
)"
```

---

### Task 5: Nav, league shell, and stub routes

**Files:**
- Modify: `frontend/src/components/home/HomeNav.tsx`
- Modify: `frontend/src/components/home/HomeNav.test.tsx` (create if missing; else extend)
- Modify: `frontend/src/components/league/LeagueHero.tsx`
- Modify: `frontend/src/components/league/LeagueHero.test.tsx`
- Modify: `frontend/src/pages/LeagueMatchupsPage.tsx`
- Create: `frontend/src/pages/MlbGameStubPage.tsx`
- Modify: `frontend/src/AppRouter.tsx`
- Modify: `frontend/src/AppRouter.test.tsx`

**Interfaces:**
- Routes: `/mlb/matchups`, `/mlb/games/:gamePk`
- Nav: MLB → `/mlb/matchups`

- [ ] **Step 1: Update AppRouter tests (failing first)**

Replace the “unknown league `/mlb/matchups` → 404” case with:

```typescript
it("renders MLB coming-soon hub at /mlb/matchups", async () => {
  renderWithProviders(["/mlb/matchups"]);
  expect(
    await screen.findByRole("heading", { name: /major league baseball/i }),
  ).toBeInTheDocument();
  expect(screen.getByText(/coming soon/i)).toBeInTheDocument();
});

it("renders MLB game stub at /mlb/games/:gamePk", async () => {
  renderWithProviders(["/mlb/games/824971"]);
  expect(await screen.findByText(/coming soon/i)).toBeInTheDocument();
});
```

Add HomeNav assertion: link name MLB → `/mlb/matchups`.

- [ ] **Step 2: Run — expect FAIL**

Run: `cd frontend && npx vitest run src/AppRouter.test.tsx`  
Expected: FAIL (still 404 / missing heading)

- [ ] **Step 3: Implement UI + routes**

`HomeNav.tsx`:

```typescript
const MLB_LOGO =
  "https://a.espncdn.com/i/teamlogos/leagues/500/mlb.png";

const leagues = [
  { id: "nba", label: "NBA", icon: nbaLogo },
  { id: "wnba", label: "WNBA", icon: wnbaLogo },
  { id: "mlb", label: "MLB", icon: MLB_LOGO },
] as const;
```

`LeagueHero` — add `mlb` entry (CDN string for `image`):

```typescript
mlb: {
  label: "MLB",
  title: "Major League Baseball",
  blurb:
    "Tonight's matchups and live scores—standings, leaders, and props coming soon.",
  image: "https://a.espncdn.com/i/teamlogos/leagues/500/mlb.png",
},
```

`LeagueMatchupsPage`:

```tsx
if (league === "mlb") {
  return (
    <div className="space-y-0 pb-8">
      <LeagueHero league="mlb" />
      <LeagueSubnav league="mlb" />
      <p className="mx-auto max-w-6xl px-4 text-sm text-white/40 sm:px-6">
        MLB matchups coming soon.
      </p>
    </div>
  );
}
```

Refactor NBA branch if needed so `nba` and `mlb` both use coming-soon (avoid a fall-through that always says NBA).

`MlbGameStubPage.tsx`:

```tsx
export function MlbGameStubPage() {
  return (
    <div className="mx-auto max-w-6xl px-4 py-16 sm:px-6">
      <h1 className="text-2xl font-semibold text-white">Game detail</h1>
      <p className="mt-3 text-sm text-white/40">
        MLB game detail coming soon.
      </p>
    </div>
  );
}
```

`AppRouter.tsx`:

```tsx
<Route path="/mlb/matchups" element={<LeagueMatchupsPage league="mlb" />} />
<Route path="/mlb/games/:gamePk" element={<MlbGameStubPage />} />
```

- [ ] **Step 4: Run — expect PASS**

Run: `cd frontend && npx vitest run src/AppRouter.test.tsx src/components/home/HomeNav.test.tsx src/components/league/LeagueHero.test.tsx`  
Expected: PASS (adjust test file paths if HomeNav tests live elsewhere)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/home/HomeNav.tsx \
  frontend/src/components/league/LeagueHero.tsx \
  frontend/src/components/league/LeagueHero.test.tsx \
  frontend/src/pages/LeagueMatchupsPage.tsx \
  frontend/src/pages/MlbGameStubPage.tsx \
  frontend/src/AppRouter.tsx \
  frontend/src/AppRouter.test.tsx \
  frontend/src/components/home/HomeNav.test.tsx
git commit -m "$(cat <<'EOF'
feat(mlb): add nav entry and coming-soon matchups/game stubs

EOF
)"
```

---

### Task 6: Hooks, merge, ticker + Live Now wiring

**Files:**
- Create: `frontend/src/hooks/useMlbScoreboard.ts`
- Create: `frontend/src/hooks/useMlbScoreboard.test.tsx`
- Create: `frontend/src/components/home/mergeLeagueScoreboards.ts`
- Create: `frontend/src/components/home/mergeLeagueScoreboards.test.ts`
- Modify: `frontend/src/layouts/HomeChromeLayout.tsx`
- Modify: `frontend/src/layouts/HomeChromeLayout.test.tsx`
- Modify: `frontend/src/pages/HomePage.tsx`
- Modify: `frontend/src/components/home/LiveTicker.tsx`
- Modify: `frontend/src/components/home/LiveNowSection.tsx`
- Modify: `frontend/src/components/home/LiveTicker.test.tsx`
- Modify: `frontend/src/components/home/LiveNowSection.test.tsx`

**Interfaces:**
- Produces:

```typescript
export function mergeLeagueScoreboards(parts: Array<{
  tickerGames: TickerGame[];
  liveGames: LiveGame[];
  isLoading: boolean;
  hasNeverLoaded: boolean;
  shouldPoll?: boolean;
}>): {
  tickerGames: TickerGame[];
  liveGames: LiveGame[];
  isLoading: boolean;
  hasNeverLoaded: boolean;
  shouldPoll: boolean;
}
```

Rules:
- Concatenate ticker/live arrays (WNBA then MLB is fine; or sort live-first already handled inside ticker)
- `hasNeverLoaded` = every part has `hasNeverLoaded === true` (all never loaded)
- `isLoading` = any part loading **and** merged live list still empty (for skeletons)
- `shouldPoll` = any part wants poll

- [ ] **Step 1: Failing merge + link tests**

```typescript
// mergeLeagueScoreboards.test.ts
it("keeps WNBA games when MLB never loaded", () => {
  const merged = mergeLeagueScoreboards([
    {
      tickerGames: [wnbaTicker],
      liveGames: [wnbaLive],
      isLoading: false,
      hasNeverLoaded: false,
    },
    {
      tickerGames: [],
      liveGames: [],
      isLoading: false,
      hasNeverLoaded: true,
    },
  ]);
  expect(merged.tickerGames).toHaveLength(1);
  expect(merged.hasNeverLoaded).toBe(false);
});
```

LiveTicker / LiveNow: MLB game with `mlbGamePk` links to `/mlb/games/...`.

- [ ] **Step 2: Run — expect FAIL**

Run: `cd frontend && npx vitest run src/components/home/mergeLeagueScoreboards.test.ts src/components/home/LiveTicker.test.tsx src/components/home/LiveNowSection.test.tsx`  
Expected: FAIL

- [ ] **Step 3: Implement hook + merge + wire**

```typescript
// useMlbScoreboard.ts — mirror useWnbaScoreboard today-only
export function useMlbScoreboard() {
  const query = useQuery({
    queryKey: ["mlb", "scoreboard", "today"],
    queryFn: fetchMlbScoreboard,
    refetchInterval: (q) =>
      shouldPollScoreboard(q.state.data?.games) ? 18_000 : false,
  });
  const games = query.data?.games ?? [];
  return {
    ...query,
    games,
    tickerGames: mapToTickerGames(games),
    liveGames: mapToLiveGames(games),
    shouldPoll: shouldPollScoreboard(query.data?.games),
    hasNeverLoaded: query.isError && query.data === undefined,
  };
}
```

`mapToTickerGames` / `mapToLiveGames` / `shouldPollScoreboard` must accept the shared `ScoreboardGame[]` (or `ApiMlbGame[] | ApiWnbaGame[]`) from Task 4 so these calls type-check without casts.

`useMlbScoreboard.test.tsx`: mock `fetchMlbScoreboard`, assert query key `["mlb","scoreboard","today"]` and that `tickerGames` / `liveGames` map from the fixture game.

`HomeChromeLayout`:

```tsx
const wnba = useWnbaScoreboard();
const mlb = useMlbScoreboard();
const { tickerGames, hasNeverLoaded } = mergeLeagueScoreboards([wnba, mlb]);
```

`HomePage`: same for `liveGames` / `isLoading` / `hasNeverLoaded`.

`LiveTicker` / `LiveNowSection`: replace `espnEventId` link checks with:

```tsx
const href = gameDetailHref(game);
if (href) { return <Link to={href}>...</Link>; }
```

- [ ] **Step 4: Run frontend suite subset — expect PASS**

Run: `cd frontend && npx vitest run src/components/home src/layouts/HomeChromeLayout.test.tsx src/pages/HomePage.tsx src/hooks/useMlbScoreboard.test.tsx`  
Then: `cd frontend && npm run test`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/hooks/useMlbScoreboard.ts \
  frontend/src/components/home/mergeLeagueScoreboards.ts \
  frontend/src/components/home/mergeLeagueScoreboards.test.ts \
  frontend/src/layouts/HomeChromeLayout.tsx \
  frontend/src/layouts/HomeChromeLayout.test.tsx \
  frontend/src/pages/HomePage.tsx \
  frontend/src/components/home/LiveTicker.tsx \
  frontend/src/components/home/LiveNowSection.tsx \
  frontend/src/components/home/LiveTicker.test.tsx \
  frontend/src/components/home/LiveNowSection.test.tsx \
  frontend/src/hooks/useMlbScoreboard.test.tsx
git commit -m "$(cat <<'EOF'
feat(mlb): merge MLB scoreboard into home ticker and Live Now

EOF
)"
```

---

### Task 7: System design doc + final verification

**Files:**
- Modify: `docs/superpowers/specs/2026-08-02-website-api-system-design.md`

- [ ] **Step 1: Update page ↔ API map**

Add rows for:

| Route / surface | API | Notes |
| --- | --- | --- |
| Chrome ticker / Live Now | `GET /api/mlb/scoreboard/today` (+ WNBA) | Client merge |
| `/mlb/matchups` | none | Coming soon |
| `/mlb/games/:gamePk` | none | Coming soon |

Update overview text that product surface is “primarily WNBA” to mention MLB live on home + stubs.

- [ ] **Step 2: Full verification**

```bash
python3 -m pytest backend/tests/test_mlb_scoreboard_normalize.py \
  backend/tests/test_mlb_scoreboard_route.py \
  backend/tests/test_export_openapi.py -v
cd frontend && npm run check:api && npm run test && npm run build
```

Expected: all PASS

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-08-02-website-api-system-design.md
git commit -m "$(cat <<'EOF'
docs: note MLB scoreboard and home stubs in website API map

EOF
)"
```

---

## Spec coverage checklist

| Spec requirement | Task |
| --- | --- |
| `/api/mlb/scoreboard/today` + Stats API | 1–2 |
| Status / postponed / Top 8th labels | 1 |
| 502 + no-store | 2 |
| OpenAPI + `fetchMlbScoreboard` | 3 |
| `HomeLeague` + `mlbGamePk` + links | 4, 6 |
| Nav CDN logo → `/mlb/matchups` | 5 |
| Matchups + game stubs | 5 |
| Merge ticker/Live Now; isolate failures | 6 |
| Update website-api system design | 7 |
| No Statcast / props / About MLB | (explicit non-goals — no tasks) |
