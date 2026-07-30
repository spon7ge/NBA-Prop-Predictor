# WNBA Game Detail Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Open `/games/:espnEventId` from LIVE NOW and the ticker, backed by `GET /api/wnba/games/{espnEventId}` that normalizes ESPN’s WNBA summary into header, shot chart, and play-by-play UI.

**Architecture:** Extend today’s scoreboard games with `espn_event_id`. New backend service fetches ESPN `summary?event=…`, normalizes header/plays/shots, caches briefly with stale-while-error, and returns `Cache-Control: no-store`. Frontend adds `useGameDetail`, a `GameDetailPage` under `HomeChromeLayout`, and links from LIVE NOW + ticker when an ESPN id is present.

**Tech Stack:** FastAPI · Pydantic · httpx · React 19 · TypeScript · Vite 6 · TanStack Query · Vitest · Testing Library · SVG shot chart

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-29-wnba-game-detail-design.md`
- Coding standards: `CLAUDE.md` (small focused modules, strong typing, tests with code, explain why in comments)
- Route: `GET /api/wnba/games/{espn_event_id}` (WNBA only; shapes leave room for NBA later)
- Upstream: `https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/summary?event={id}`
- Response always `Cache-Control: no-store`
- Detail poll: `18_000` ms while `status` is `live` or `halftime`; stop for `scheduled` / `final`
- Back always navigates to `/`
- FG counts = made / total from normalized `shots` list
- No team logo images; letter avatars fine if needed
- Verify backend with `python3 -m pytest backend/tests/ -v`
- Verify frontend with `cd frontend && npm run test && npm run build`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `backend/app/schemas/wnba_scoreboard.py` | Add `espn_event_id` |
| `backend/app/services/wnba_scoreboard.py` | Set/preserve `espn_event_id` on normalize + merge |
| `backend/tests/test_wnba_scoreboard_normalize.py` | Assert `espn_event_id` preserved |
| `backend/app/schemas/wnba_game_detail.py` | Detail Pydantic models |
| `backend/app/services/wnba_game_detail.py` | Fetch, normalize, TTL cache |
| `backend/app/api/routes/wnba_game_detail.py` | Thin route + headers |
| `backend/app/main.py` | Register router; update description |
| `backend/tests/fixtures/espn_wnba_summary.json` | Minimal ESPN summary fixture |
| `backend/tests/test_wnba_game_detail_normalize.py` | Normalize tests |
| `backend/tests/test_wnba_game_detail_route.py` | Route / cache / errors |
| `frontend/src/lib/api.ts` | Types + `fetchGameDetail`; scoreboard `espn_event_id` |
| `frontend/src/lib/api.test.ts` | fetchGameDetail tests |
| `frontend/src/components/home/types.ts` | Optional `espnEventId` |
| `frontend/src/components/home/mapScoreboard.ts` | Map `espn_event_id` → `espnEventId` |
| `frontend/src/components/home/mapScoreboard.test.ts` | Mapper coverage |
| `frontend/src/hooks/useGameDetail.ts` | Query + live poll guard |
| `frontend/src/hooks/useGameDetail.test.tsx` | Polling / never-loaded |
| `frontend/src/components/game/types.ts` | UI types for detail |
| `frontend/src/components/game/GameHeader.tsx` | Back row + scoreboard card |
| `frontend/src/components/game/ShotChart.tsx` | Filters + SVG court + shots |
| `frontend/src/components/game/PlayByPlay.tsx` | Period filter + feed |
| `frontend/src/components/game/*.test.tsx` | Component tests |
| `frontend/src/pages/GameDetailPage.tsx` | Compose detail UI |
| `frontend/src/AppRouter.tsx` | `/games/:espnEventId` |
| `frontend/src/AppRouter.test.tsx` | Route smoke |
| `frontend/src/components/home/LiveNowSection.tsx` | Card → Link |
| `frontend/src/components/home/LiveTicker.tsx` | Chip → Link |
| `frontend/src/components/home/LiveNowSection.test.tsx` | Link href |
| `frontend/src/components/home/LiveTicker.test.tsx` | Link href |

---

### Task 1: Preserve `espn_event_id` on scoreboard

**Files:**
- Modify: `backend/app/schemas/wnba_scoreboard.py`
- Modify: `backend/app/services/wnba_scoreboard.py`
- Modify: `backend/tests/test_wnba_scoreboard_normalize.py`

**Interfaces:**
- Produces: `WnbaGame.espn_event_id: str | None`
- Consumes: existing `normalize_espn_scoreboard`, `merge_games`

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_wnba_scoreboard_normalize.py`:

```python
def test_normalize_espn_sets_espn_event_id():
    payload = json.loads((FIXTURES / "espn_wnba_scoreboard.json").read_text())
    g = normalize_espn_scoreboard(payload, date_et="2026-07-29")[0]
    assert g.espn_event_id == "401749001"


def test_merge_preserves_espn_event_id_when_stats_id_wins():
    espn = [
        WnbaGame(
            id="espn-401749001",
            espn_event_id="401749001",
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
            espn_event_id=None,
            status="live",
            status_label="Q3 7:13",
            away=WnbaTeam(abbrev="ATL", name="Atlanta Dream", score=36),
            home=WnbaTeam(abbrev="DAL", name="Dallas Wings", score=44),
            start_time_et="2026-07-29T23:00:00Z",
        )
    ]
    merged = merge_games(espn, stats)
    assert len(merged) == 1
    assert merged[0].id == "1022600123"
    assert merged[0].espn_event_id == "401749001"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest backend/tests/test_wnba_scoreboard_normalize.py::test_normalize_espn_sets_espn_event_id backend/tests/test_wnba_scoreboard_normalize.py::test_merge_preserves_espn_event_id_when_stats_id_wins -v`

Expected: FAIL (field missing / AssertionError)

- [ ] **Step 3: Minimal implementation**

In `backend/app/schemas/wnba_scoreboard.py`, add to `WnbaGame`:

```python
espn_event_id: str | None = None
```

In `normalize_espn_scoreboard`, when building each game:

```python
raw_id = str(event.get("id") or "").strip()
games.append(
    WnbaGame(
        id=f"espn-{raw_id}" if raw_id else "espn-unknown",
        espn_event_id=raw_id or None,
        # ...existing fields...
    )
)
```

In `normalize_stats_scoreboard`, leave `espn_event_id=None` (default).

In `merge_games`, when constructing the merged `WnbaGame`, set:

```python
espn_event_id=a.espn_event_id or g.espn_event_id,
```

(Use the ESPN-side game `a` first so the event id survives when `game_id` becomes the stats id.)

Also set `espn_event_id` on any path that constructs `WnbaGame` from ESPN-only rows (already set) or stats-only rows (`None`).

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest backend/tests/test_wnba_scoreboard_normalize.py -v`

Expected: PASS (including new tests)

- [ ] **Step 5: Commit**

```bash
git add backend/app/schemas/wnba_scoreboard.py backend/app/services/wnba_scoreboard.py backend/tests/test_wnba_scoreboard_normalize.py
git commit -m "Preserve ESPN event ids on WNBA scoreboard games."
```

---

### Task 2: Game detail schema + ESPN summary normalizer

**Files:**
- Create: `backend/app/schemas/wnba_game_detail.py`
- Create: `backend/app/services/wnba_game_detail.py` (normalize helpers only)
- Create: `backend/tests/fixtures/espn_wnba_summary.json`
- Create: `backend/tests/test_wnba_game_detail_normalize.py`

**Interfaces:**
- Produces:
  - `class GameDetailTeam(BaseModel): id: str; abbrev: str; name: str; score: int | None; color: str`
  - `class GameDetailShot(BaseModel): id: str; team_id: str; player_name: str; made: bool; x: float; y: float; period: int; clock: str`
  - `class GameDetailPlay(BaseModel): id: str; team_id: str | None; period: int; clock: str; text: str; scoring: bool; away_score: int; home_score: int; shooting: bool`
  - `class GameDetailLatestPlay(BaseModel): id: str; clock: str; period: int; text: str; team_id: str | None`
  - `class WnbaGameDetail(BaseModel): espn_event_id: str; league: Literal["wnba"]; status: GameStatus; status_label: str; venue: str | None; away: GameDetailTeam; home: GameDetailTeam; fg_made: int; fg_attempted: int; latest_play: GameDetailLatestPlay | None; shots: list[GameDetailShot]; plays: list[GameDetailPlay]; fetched_at: str`
  - `def normalize_espn_summary(payload: dict, *, espn_event_id: str, fetched_at: str) -> WnbaGameDetail`
  - Reuse `GameStatus` from `app.schemas.wnba_scoreboard` (or re-export)

- [ ] **Step 1: Write fixture + failing test**

Create `backend/tests/fixtures/espn_wnba_summary.json` (trimmed real shape):

```json
{
  "header": {
    "id": "401857098",
    "competitions": [
      {
        "status": {
          "period": 1,
          "displayClock": "4:13",
          "type": {
            "state": "in",
            "completed": false,
            "name": "STATUS_IN_PROGRESS",
            "shortDetail": "4:13 - 1st",
            "detail": "4:13 - 1st Quarter"
          }
        },
        "competitors": [
          {
            "homeAway": "home",
            "score": "9",
            "team": {
              "id": "21",
              "abbreviation": "PHX",
              "displayName": "Phoenix Mercury",
              "color": "e56020",
              "alternateColor": "1c105e"
            }
          },
          {
            "homeAway": "away",
            "score": "10",
            "team": {
              "id": "129153",
              "abbreviation": "GS",
              "displayName": "Golden State Valkyries",
              "color": "553987",
              "alternateColor": "ffc72c"
            }
          }
        ]
      }
    ]
  },
  "gameInfo": {
    "venue": { "fullName": "Mortgage Matchup Center" }
  },
  "plays": [
    {
      "id": "40185709810",
      "text": "Laeticia Amihere makes two point shot",
      "awayScore": 10,
      "homeScore": 8,
      "scoringPlay": true,
      "shootingPlay": true,
      "scoreValue": 2,
      "period": { "number": 1 },
      "clock": { "displayValue": "4:29" },
      "coordinate": { "x": 25, "y": 5 },
      "team": { "id": "129153" },
      "participants": [{ "athlete": { "id": "1" } }]
    },
    {
      "id": "40185709811",
      "text": "Veronica Burton shooting foul",
      "awayScore": 10,
      "homeScore": 9,
      "scoringPlay": false,
      "shootingPlay": false,
      "scoreValue": 0,
      "period": { "number": 1 },
      "clock": { "displayValue": "4:13" },
      "coordinate": { "x": 0, "y": 0 },
      "team": { "id": "129153" },
      "participants": []
    },
    {
      "id": "40185709809",
      "text": "Alyssa Thomas misses 12-foot jumper",
      "awayScore": 8,
      "homeScore": 8,
      "scoringPlay": false,
      "shootingPlay": true,
      "scoreValue": 0,
      "period": { "number": 1 },
      "clock": { "displayValue": "4:40" },
      "coordinate": { "x": 37, "y": 12 },
      "team": { "id": "21" },
      "participants": [{ "athlete": { "id": "2" } }]
    }
  ]
}
```

Create `backend/tests/test_wnba_game_detail_normalize.py`:

```python
import json
from pathlib import Path

from app.services.wnba_game_detail import normalize_espn_summary

FIXTURES = Path(__file__).parent / "fixtures"


def test_normalize_espn_summary_header_shots_plays():
    payload = json.loads((FIXTURES / "espn_wnba_summary.json").read_text())
    detail = normalize_espn_summary(
        payload, espn_event_id="401857098", fetched_at="2026-07-29T19:00:00-04:00"
    )
    assert detail.espn_event_id == "401857098"
    assert detail.league == "wnba"
    assert detail.status == "live"
    assert detail.status_label == "4:13 - 1st"
    assert detail.venue == "Mortgage Matchup Center"
    assert detail.away.abbrev == "GS"
    assert detail.away.score == 10
    assert detail.away.color.startswith("#")
    assert detail.home.abbrev == "PHX"
    assert detail.home.score == 9
    assert detail.fg_attempted == 2
    assert detail.fg_made == 1
    assert len(detail.shots) == 2
    made = next(s for s in detail.shots if s.made)
    assert made.player_name == "Laeticia Amihere"
    assert made.x == 25
    assert made.y == 5
    assert detail.latest_play is not None
    assert "Burton" in detail.latest_play.text
    assert len(detail.plays) == 3
    scoring = [p for p in detail.plays if p.scoring]
    assert len(scoring) == 1
    assert scoring[0].away_score == 10
    assert scoring[0].home_score == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest backend/tests/test_wnba_game_detail_normalize.py::test_normalize_espn_summary_header_shots_plays -v`

Expected: FAIL (import / function missing)

- [ ] **Step 3: Write schema + normalizer**

`backend/app/schemas/wnba_game_detail.py` — models as listed in Interfaces.

`backend/app/services/wnba_game_detail.py` — implement:

```python
FALLBACK_AWAY_COLOR = "#7C3AED"
FALLBACK_HOME_COLOR = "#EA580C"


def _hex_color(raw: str | None, fallback: str) -> str:
    s = str(raw or "").strip().lstrip("#")
    if len(s) == 6 and all(c in "0123456789abcdefABCDEF" for c in s):
        return f"#{s.upper()}"
    return fallback


def _player_name_from_text(text: str) -> str:
    # ESPN texts start with the athlete name before the verb.
    for verb in (" makes ", " misses ", " shooting ", " defensive ", " offensive "):
        if verb in text:
            return text.split(verb, 1)[0].strip()
    return text.split(" ", 2)[0] if text else ""


def normalize_espn_summary(payload: dict, *, espn_event_id: str, fetched_at: str) -> WnbaGameDetail:
    header = payload.get("header") or {}
    comp = (header.get("competitions") or [{}])[0]
    status_block = comp.get("status") or {}
    teams = {c.get("homeAway"): c for c in (comp.get("competitors") or [])}
    away_c, home_c = teams.get("away") or {}, teams.get("home") or {}
    venue = ((payload.get("gameInfo") or {}).get("venue") or {}).get("fullName")
    status, status_label = _detail_status(status_block)  # prefer shortDetail for label

    def team(c: dict, fallback_color: str) -> GameDetailTeam:
        t = c.get("team") or {}
        raw = c.get("score")
        score = int(raw) if raw not in (None, "") else None
        return GameDetailTeam(
            id=str(t.get("id") or ""),
            abbrev=str(t.get("abbreviation") or ""),
            name=str(t.get("displayName") or ""),
            score=score if status != "scheduled" else None,
            color=_hex_color(t.get("color"), fallback_color),
        )

    raw_plays = payload.get("plays") or []
    plays: list[GameDetailPlay] = []
    shots: list[GameDetailShot] = []
    for p in raw_plays:
        period = int((p.get("period") or {}).get("number") or 0)
        clock = str((p.get("clock") or {}).get("displayValue") or "")
        team_id = str((p.get("team") or {}).get("id") or "") or None
        text = str(p.get("text") or "")
        shooting = bool(p.get("shootingPlay"))
        scoring = bool(p.get("scoringPlay"))
        play = GameDetailPlay(
            id=str(p.get("id") or ""),
            team_id=team_id,
            period=period,
            clock=clock,
            text=text,
            scoring=scoring,
            away_score=int(p.get("awayScore") or 0),
            home_score=int(p.get("homeScore") or 0),
            shooting=shooting,
        )
        plays.append(play)
        if shooting:
            coord = p.get("coordinate") or {}
            shots.append(
                GameDetailShot(
                    id=play.id,
                    team_id=team_id or "",
                    player_name=_player_name_from_text(text),
                    made=scoring,
                    x=float(coord.get("x") or 0),
                    y=float(coord.get("y") or 0),
                    period=period,
                    clock=clock,
                )
            )

    latest_src = raw_plays[-1] if raw_plays else None
    latest = None
    if latest_src is not None:
        latest = GameDetailLatestPlay(
            id=str(latest_src.get("id") or ""),
            clock=str((latest_src.get("clock") or {}).get("displayValue") or ""),
            period=int((latest_src.get("period") or {}).get("number") or 0),
            text=str(latest_src.get("text") or ""),
            team_id=str((latest_src.get("team") or {}).get("id") or "") or None,
        )

    return WnbaGameDetail(
        espn_event_id=espn_event_id,
        status=status,
        status_label=status_label,
        venue=str(venue) if venue else None,
        away=team(away_c, FALLBACK_AWAY_COLOR),
        home=team(home_c, FALLBACK_HOME_COLOR),
        fg_made=sum(1 for s in shots if s.made),
        fg_attempted=len(shots),
        latest_play=latest,
        shots=shots,
        plays=list(reversed(plays)),  # newest-first for UI
        fetched_at=fetched_at,
    )
```

Status label: prefer `status.type.shortDetail` (e.g. `"4:13 - 1st"`) to match the mockup; map machine `status` with the same rules as scoreboard (`in` → live, `post`+completed → final, halftime, else scheduled). Implement `_detail_status` accordingly (can mirror scoreboard `_espn_status` but keep shortDetail as the live label).

Colors: `#` + team `color` (uppercase), else fallbacks above.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest backend/tests/test_wnba_game_detail_normalize.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/schemas/wnba_game_detail.py backend/app/services/wnba_game_detail.py backend/tests/fixtures/espn_wnba_summary.json backend/tests/test_wnba_game_detail_normalize.py
git commit -m "Normalize ESPN WNBA game summary into detail schema."
```

---

### Task 3: Game detail fetch, cache, and route

**Files:**
- Modify: `backend/app/services/wnba_game_detail.py` (fetch + cache)
- Create: `backend/app/api/routes/wnba_game_detail.py`
- Modify: `backend/app/main.py`
- Create: `backend/tests/test_wnba_game_detail_route.py`

**Interfaces:**
- Produces:
  - `async def get_game_detail(espn_event_id: str) -> WnbaGameDetail`
  - Route `GET /api/wnba/games/{espn_event_id}` → `WnbaGameDetail`, header `Cache-Control: no-store`
  - Raises / maps: `404` when ESPN returns empty/error payload for unknown event; `502` when upstream fails with empty cache

- [ ] **Step 1: Write failing route tests**

```python
import json
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.services import wnba_game_detail as svc

FIXTURES = Path(__file__).parent / "fixtures"
client = TestClient(app)


def test_game_detail_200_no_store():
    payload = json.loads((FIXTURES / "espn_wnba_summary.json").read_text())

    async def fake_fetch(espn_event_id: str):
        return payload

    with patch.object(svc, "fetch_espn_summary", side_effect=fake_fetch):
        svc.clear_game_detail_cache()
        res = client.get("/api/wnba/games/401857098")
    assert res.status_code == 200
    assert res.headers["Cache-Control"] == "no-store"
    body = res.json()
    assert body["espn_event_id"] == "401857098"
    assert body["away"]["abbrev"] == "GS"


def test_game_detail_404_when_espn_says_not_found():
    async def fake_fetch(espn_event_id: str):
        return {"code": 404, "message": "Not found"}

    with patch.object(svc, "fetch_espn_summary", side_effect=fake_fetch):
        svc.clear_game_detail_cache()
        res = client.get("/api/wnba/games/999")
    assert res.status_code == 404
    assert res.headers.get("Cache-Control") == "no-store"


def test_game_detail_stale_while_error():
    payload = json.loads((FIXTURES / "espn_wnba_summary.json").read_text())

    async def ok(espn_event_id: str):
        return payload

    with patch.object(svc, "fetch_espn_summary", side_effect=ok):
        svc.clear_game_detail_cache()
        assert client.get("/api/wnba/games/401857098").status_code == 200

    async def boom(espn_event_id: str):
        raise RuntimeError("down")

    with patch.object(svc, "fetch_espn_summary", side_effect=boom):
        res = client.get("/api/wnba/games/401857098")
    assert res.status_code == 200
    assert res.json()["espn_event_id"] == "401857098"


def test_game_detail_502_when_never_cached():
    async def boom(espn_event_id: str):
        raise RuntimeError("down")

    with patch.object(svc, "fetch_espn_summary", side_effect=boom):
        svc.clear_game_detail_cache()
        res = client.get("/api/wnba/games/401857098")
    assert res.status_code == 502
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest backend/tests/test_wnba_game_detail_route.py -v`

Expected: FAIL (route missing)

- [ ] **Step 3: Implement fetch, cache, route**

In `wnba_game_detail.py`:

```python
ESPN_SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/summary"
ESPN_TIMEOUT_SECONDS = 8.0

# Per-event cache: { espn_event_id: { response, expires_at } }
_cache: dict[str, dict] = {}


def clear_game_detail_cache() -> None:
    _cache.clear()


def cache_ttl_seconds(detail: WnbaGameDetail) -> int:
    if detail.status in ("live", "halftime"):
        return 15
    return 60


async def fetch_espn_summary(espn_event_id: str) -> dict:
    async with httpx.AsyncClient(timeout=ESPN_TIMEOUT_SECONDS) as client:
        res = await client.get(ESPN_SUMMARY_URL, params={"event": espn_event_id})
        res.raise_for_status()
        return res.json()


def _is_not_found_payload(payload: dict) -> bool:
    return "header" not in payload and payload.get("code") in (404, 400)


async def get_game_detail(espn_event_id: str) -> WnbaGameDetail:
    now = time.time()
    cached = _cache.get(espn_event_id)
    if cached and cached["expires_at"] > now:
        return cached["response"]

    try:
        payload = await fetch_espn_summary(espn_event_id)
    except Exception:
        if cached:
            return cached["response"]
        raise

    if _is_not_found_payload(payload):
        raise LookupError(espn_event_id)

    try:
        detail = normalize_espn_summary(
            payload,
            espn_event_id=espn_event_id,
            fetched_at=datetime.now(ET).isoformat(),
        )
    except Exception:
        if cached:
            return cached["response"]
        raise

    _cache[espn_event_id] = {
        "response": detail,
        "expires_at": now + cache_ttl_seconds(detail),
    }
    return detail
```

Import `time`, `datetime`, and `ET` (`ZoneInfo("America/New_York")`) at module top.

Route:

```python
@router.get("/wnba/games/{espn_event_id}", response_model=WnbaGameDetail)
async def wnba_game_detail(espn_event_id: str, response: Response) -> WnbaGameDetail:
    response.headers["Cache-Control"] = "no-store"
    try:
        return await get_game_detail(espn_event_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Game not found", headers=_NO_STORE) from exc
    except Exception as exc:
        logger.warning("WNBA game detail unavailable: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="WNBA game detail is temporarily unavailable",
            headers=_NO_STORE,
        ) from exc
```

Register in `main.py` and mention the new upstream call in the app description.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest backend/tests/test_wnba_game_detail_route.py backend/tests/test_wnba_game_detail_normalize.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/wnba_game_detail.py backend/app/api/routes/wnba_game_detail.py backend/app/main.py backend/tests/test_wnba_game_detail_route.py
git commit -m "Add WNBA game detail API backed by ESPN summary."
```

---

### Task 4: Frontend API client + scoreboard `espnEventId` mapping

**Files:**
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/lib/api.test.ts`
- Modify: `frontend/src/components/home/types.ts`
- Modify: `frontend/src/components/home/mapScoreboard.ts`
- Modify: `frontend/src/components/home/mapScoreboard.test.ts`

**Interfaces:**
- Produces:
  - `ApiWnbaGame.espn_event_id: string | null`
  - `fetchGameDetail(espnEventId: string): Promise<ApiWnbaGameDetail>`
  - `LiveGame.espnEventId?: string | null` and `TickerGame.espnEventId?: string | null`
  - Mappers copy `espn_event_id` → `espnEventId`

- [ ] **Step 1: Write failing tests**

In `api.test.ts`:

```ts
it("fetchGameDetail hits /api/wnba/games/:id", async () => {
  fetchMock.mockResolvedValue({
    ok: true,
    json: async () => ({ espn_event_id: "401857098", league: "wnba" }),
  });
  const { fetchGameDetail } = await import("./api");
  await fetchGameDetail("401857098");
  expect(fetchMock).toHaveBeenCalledWith(
    "/api/wnba/games/401857098",
    expect.objectContaining({ cache: "no-store" }),
  );
});
```

In `mapScoreboard.test.ts`, extend an existing game fixture with `espn_event_id: "401857098"` and assert:

```ts
expect(mapToLiveGames([game])[0].espnEventId).toBe("401857098");
expect(mapToTickerGames([game])[0].espnEventId).toBe("401857098");
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npm run test -- src/lib/api.test.ts src/components/home/mapScoreboard.test.ts`

Expected: FAIL

- [ ] **Step 3: Implement types + fetch + mappers**

Add to `api.ts` (mirror backend schema; snake_case from JSON):

```ts
export type ApiGameDetailTeam = {
  id: string;
  abbrev: string;
  name: string;
  score: number | null;
  color: string;
};

export type ApiGameDetailShot = {
  id: string;
  team_id: string;
  player_name: string;
  made: boolean;
  x: number;
  y: number;
  period: number;
  clock: string;
};

export type ApiGameDetailPlay = {
  id: string;
  team_id: string | null;
  period: number;
  clock: string;
  text: string;
  scoring: boolean;
  away_score: number;
  home_score: number;
  shooting: boolean;
};

export type ApiGameDetailLatestPlay = {
  id: string;
  clock: string;
  period: number;
  text: string;
  team_id: string | null;
};

export type ApiWnbaGameDetail = {
  espn_event_id: string;
  league: "wnba";
  status: ApiGameStatus;
  status_label: string;
  venue: string | null;
  away: ApiGameDetailTeam;
  home: ApiGameDetailTeam;
  fg_made: number;
  fg_attempted: number;
  latest_play: ApiGameDetailLatestPlay | null;
  shots: ApiGameDetailShot[];
  plays: ApiGameDetailPlay[];
  fetched_at: string;
};

export async function fetchGameDetail(
  espnEventId: string,
): Promise<ApiWnbaGameDetail> {
  const res = await fetch(`${API_BASE}/api/wnba/games/${espnEventId}`, {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`Game detail request failed: ${res.status}`);
  }
  return res.json();
}
```

Add `espn_event_id: string | null` to `ApiWnbaGame`. Add `espnEventId?: string | null` to home types and map it in both mappers.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npm run test -- src/lib/api.test.ts src/components/home/mapScoreboard.test.ts`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/api.ts frontend/src/lib/api.test.ts frontend/src/components/home/types.ts frontend/src/components/home/mapScoreboard.ts frontend/src/components/home/mapScoreboard.test.ts
git commit -m "Add game detail API client and espnEventId mapping."
```

---

### Task 5: `useGameDetail` hook

**Files:**
- Create: `frontend/src/hooks/useGameDetail.ts`
- Create: `frontend/src/hooks/useGameDetail.test.tsx`

**Interfaces:**
- Produces: `useGameDetail(espnEventId: string | undefined)` → query result + `hasNeverLoaded: boolean` + `shouldPoll: boolean`
- Poll when `data.status` is `live` or `halftime`; interval `18_000`

- [ ] **Step 1: Write failing tests**

```tsx
it("enables polling for live games", async () => {
  fetchMock.mockResolvedValue({
    ok: true,
    json: async () => ({
      espn_event_id: "1",
      league: "wnba",
      status: "live",
      status_label: "Q1 4:13",
      venue: null,
      away: { id: "a", abbrev: "GS", name: "GS", score: 10, color: "#553987" },
      home: { id: "b", abbrev: "PHX", name: "PHX", score: 9, color: "#E56020" },
      fg_made: 0,
      fg_attempted: 0,
      latest_play: null,
      shots: [],
      plays: [],
      fetched_at: "",
    }),
  });
  const { result } = renderHook(() => useGameDetail("1"), { wrapper });
  await waitFor(() => expect(result.current.isSuccess).toBe(true));
  expect(result.current.shouldPoll).toBe(true);
});

it("disables polling for final games", async () => {
  fetchMock.mockResolvedValue({
    ok: true,
    json: async () => ({
      espn_event_id: "1",
      league: "wnba",
      status: "final",
      status_label: "Final",
      venue: null,
      away: { id: "a", abbrev: "GS", name: "GS", score: 80, color: "#553987" },
      home: { id: "b", abbrev: "PHX", name: "PHX", score: 75, color: "#E56020" },
      fg_made: 0,
      fg_attempted: 0,
      latest_play: null,
      shots: [],
      plays: [],
      fetched_at: "",
    }),
  });
  const { result } = renderHook(() => useGameDetail("1"), { wrapper });
  await waitFor(() => expect(result.current.isSuccess).toBe(true));
  expect(result.current.shouldPoll).toBe(false);
});

it("flags hasNeverLoaded on first failure", async () => {
  fetchMock.mockResolvedValue({ ok: false, status: 502 });
  const { result } = renderHook(() => useGameDetail("1"), { wrapper });
  await waitFor(() => expect(result.current.isError).toBe(true));
  expect(result.current.hasNeverLoaded).toBe(true);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm run test -- src/hooks/useGameDetail.test.tsx`

Expected: FAIL

- [ ] **Step 3: Implement hook**

```ts
export function useGameDetail(espnEventId: string | undefined) {
  const query = useQuery({
    queryKey: ["wnba", "game", espnEventId],
    queryFn: () => fetchGameDetail(espnEventId!),
    enabled: Boolean(espnEventId),
    refetchInterval: (q) => {
      const status = q.state.data?.status;
      return status === "live" || status === "halftime" ? 18_000 : false;
    },
  });
  const status = query.data?.status;
  const shouldPoll = status === "live" || status === "halftime";
  return {
    ...query,
    shouldPoll,
    hasNeverLoaded: query.isError && query.data === undefined,
  };
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npm run test -- src/hooks/useGameDetail.test.tsx`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/hooks/useGameDetail.ts frontend/src/hooks/useGameDetail.test.tsx
git commit -m "Add useGameDetail hook with live polling."
```

---

### Task 6: Game UI components (header, shot chart, PBP)

**Files:**
- Create: `frontend/src/components/game/types.ts`
- Create: `frontend/src/components/game/GameHeader.tsx`
- Create: `frontend/src/components/game/ShotChart.tsx`
- Create: `frontend/src/components/game/PlayByPlay.tsx`
- Create: `frontend/src/components/game/GameHeader.test.tsx`
- Create: `frontend/src/components/game/ShotChart.test.tsx`
- Create: `frontend/src/components/game/PlayByPlay.test.tsx`

**Interfaces:**
- Produces presentational components that take already-mapped camelCase props (map from API in the page, or accept `ApiWnbaGameDetail` and map locally — prefer a thin `mapGameDetail` in `components/game/mapGameDetail.ts` if types diverge).
- UI types use camelCase: `espnEventId`, `statusLabel`, `fgMade`, `fgAttempted`, `latestPlay`, `teamId`, `playerName`, `awayScore`, `homeScore`.

Shot chart coordinates: ESPN half-court scale is approximately **x ∈ [0, 50]** (sideline→sideline) and **y ∈ [0, 47]** (basket→half). SVG viewBox `0 0 500 470`; map `cx = x * 10`, `cy = y * 10`. Made = filled circle; missed = stroke-only. Filter by `both | away.id | home.id`.

- [ ] **Step 1: Write failing component tests**

Use a shared fixture object `detail` matching mapped UI types.

`GameHeader.test.tsx`: renders team names, scores, venue, status, Back link to `/`.

`ShotChart.test.tsx`: renders “Shot chart”; clicking team filter hides other team’s shots; shows `1/2 FG` and “Data: ESPN”; shows latest play text.

`PlayByPlay.test.tsx`: renders plays; period pill filters; scoring play shows running score.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npm run test -- src/components/game`

Expected: FAIL

- [ ] **Step 3: Implement components**

Dark styling consistent with home (`bg-[#141414]`, borders `border-white/10`, amber scores `text-amber-300`).

`GameHeader`: Link/button “Back” → `to="/"`. Status on the right. Card with status|venue line, colored team names, score boxes.

`ShotChart`: local state `filter: "both" | string` (team id). Half-court SVG (paint, hoop, 3pt arc — simple paths are fine). Legend Made/Missed.

`PlayByPlay`: derive periods from plays; default selected period = max period present (or current from latest play). List newest-first within period. Highlight first item. Scoring rows: `bg-white/5` + `{awayScore}-{homeScore}` on the right. Colored dots from team color map.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npm run test -- src/components/game`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/game
git commit -m "Add game detail header, shot chart, and play-by-play UI."
```

---

### Task 7: `GameDetailPage` + router

**Files:**
- Create: `frontend/src/pages/GameDetailPage.tsx`
- Modify: `frontend/src/AppRouter.tsx`
- Modify: `frontend/src/AppRouter.test.tsx`

**Interfaces:**
- Route param `espnEventId` from `useParams`
- Page calls `useGameDetail(espnEventId)` and renders loading / error / content

- [ ] **Step 1: Write failing router test**

```tsx
it("renders game detail at /games/:espnEventId", async () => {
  fetchMock.mockImplementation(async (url: string) => {
    if (String(url).includes("/api/wnba/games/")) {
      return {
        ok: true,
        json: async () => ({
          espn_event_id: "401857098",
          league: "wnba",
          status: "live",
          status_label: "4:13 - 1st",
          venue: "Mortgage Matchup Center",
          away: {
            id: "129153",
            abbrev: "GS",
            name: "Golden State Valkyries",
            score: 10,
            color: "#553987",
          },
          home: {
            id: "21",
            abbrev: "PHX",
            name: "Phoenix Mercury",
            score: 9,
            color: "#E56020",
          },
          fg_made: 1,
          fg_attempted: 2,
          latest_play: null,
          shots: [],
          plays: [],
          fetched_at: "",
        }),
      };
    }
    return {
      ok: true,
      json: async () => ({ date: "2026-07-29", fetched_at: "", games: [] }),
    };
  });
  renderWithProviders(["/games/401857098"]);
  expect(await screen.findByText(/Golden State Valkyries/i)).toBeInTheDocument();
  expect(screen.getByText("No live games")).toBeInTheDocument(); // chrome ticker still present
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm run test -- src/AppRouter.test.tsx`

Expected: FAIL (route missing / page missing)

- [ ] **Step 3: Implement page + route**

```tsx
// AppRouter — under HomeChromeLayout:
<Route path="/games/:espnEventId" element={<GameDetailPage />} />
```

`GameDetailPage`:
- `const { espnEventId } = useParams()`
- Loading skeletons when `isLoading && !data`
- `hasNeverLoaded` → “Unable to load game” + Back link
- Else compose `GameHeader` + grid `ShotChart` | `PlayByPlay`
- Scheduled empty: if `plays.length === 0` && status scheduled, panels show “Tip-off pending”

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npm run test -- src/AppRouter.test.tsx`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/GameDetailPage.tsx frontend/src/AppRouter.tsx frontend/src/AppRouter.test.tsx
git commit -m "Route /games/:espnEventId to the game detail page."
```

---

### Task 8: Wire LIVE NOW + ticker links

**Files:**
- Modify: `frontend/src/components/home/LiveNowSection.tsx`
- Modify: `frontend/src/components/home/LiveTicker.tsx`
- Modify: `frontend/src/components/home/LiveNowSection.test.tsx`
- Modify: `frontend/src/components/home/LiveTicker.test.tsx`

**Interfaces:**
- When `espnEventId` is truthy, wrap card/chip in `<Link to={`/games/${espnEventId}`}>`
- When missing, render non-link (same visual, not clickable)

- [ ] **Step 1: Write failing tests**

```tsx
// LiveNowSection — game with espnEventId
expect(screen.getByRole("link", { name: /Atlanta Dream/i })).toHaveAttribute(
  "href",
  "/games/401857098",
);

// LiveTicker — same
expect(screen.getByRole("link", { name: /ATL/i })).toHaveAttribute(
  "href",
  "/games/401857098",
);
```

Update fixtures in those tests to include `espnEventId: "401857098"`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npm run test -- src/components/home/LiveNowSection.test.tsx src/components/home/LiveTicker.test.tsx`

Expected: FAIL

- [ ] **Step 3: Implement links**

`LiveGameCard`: if `game.espnEventId`, wrap `<article>` content in `Link` (or make the article a `Link` with the same classes + `hover:border-white/20`). Ensure accessible name includes team names.

`TickerItem`: wrap content in `Link` when `espnEventId` present; keep mono styling.

- [ ] **Step 4: Run full frontend verification**

Run: `cd frontend && npm run test && npm run build`

Expected: all PASS / build succeeds

Also run: `python3 -m pytest backend/tests/test_wnba_game_detail_normalize.py backend/tests/test_wnba_game_detail_route.py backend/tests/test_wnba_scoreboard_normalize.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/home/LiveNowSection.tsx frontend/src/components/home/LiveTicker.tsx frontend/src/components/home/LiveNowSection.test.tsx frontend/src/components/home/LiveTicker.test.tsx
git commit -m "Link LIVE NOW and ticker games to the detail page."
```

---

## Spec coverage checklist

| Spec requirement | Task |
| --- | --- |
| `/games/:espnEventId` under home chrome | 7 |
| LIVE NOW + ticker entry points | 8 |
| Header + shot chart + PBP | 6, 7 |
| Backend ESPN summary proxy | 2, 3 |
| `espn_event_id` on scoreboard | 1, 4 |
| All statuses + tip-off pending | 6, 7 |
| Poll live/halftime only | 5 |
| `Cache-Control: no-store` + stale-while-error | 3 |
| FG from shots list | 2 |
| Back → `/` | 6 |
| Tests for normalize/route/UI/links | 1–8 |
