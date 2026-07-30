# WNBA Scheduled Matchup Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For scheduled WNBA games, show ESPN matchup prediction, projected starters (from each team's last game), season leaders, and injuries on `/games/:espnEventId`; keep the existing live/final shot chart + play-by-play + win probability layout unchanged.

**Architecture:** Extend the existing `GET /api/wnba/games/{espnEventId}` ESPN summary proxy. Sync-normalize `predictor`, `leaders`, and `injuries` from the primary summary. When status is `scheduled`, fan out from `get_game_detail` to prior-game summaries via `lastFiveGames`, then pass those payloads into `normalize_espn_summary` to build projected starters. Frontend `GameDetailPage` branches on `status`.

**Tech Stack:** FastAPI, Pydantic, httpx, pytest, React, TypeScript, Vitest, Testing Library, Tailwind

## Global Constraints

- Scheduled → preview UI only; live / halftime / final → existing header + ShotChart + PlayByPlay + WinProbabilityPanel only.
- Extend the existing game-detail endpoint; do not add a second preview route.
- All preview data comes from ESPN via the backend; frontend never calls ESPN.
- Omit a preview section when its payload field is `null`.
- Projected starters are all-or-nothing: if either team's prior-game resolve fails, `projected_starters` is `null`.
- Empty injuries on both sides → `injuries: null`.
- Follow TDD: failing test before production code for each task.
- Brand as HoopVista dark charcoal cards; use team `color` on prediction bar and abbrevs.

---

## File Structure

- Modify `backend/app/schemas/wnba_game_detail.py` — preview Pydantic models + fields on `WnbaGameDetail`
- Modify `backend/app/services/wnba_game_detail.py` — normalize helpers + scheduled prior-game fan-out in `get_game_detail`
- Modify `backend/tests/test_wnba_game_detail_normalize.py` — prediction / leaders / injuries / starters tests
- Create `backend/tests/fixtures/espn_wnba_summary_scheduled_preview.json` — scheduled summary with predictor/leaders/injuries/lastFiveGames
- Create `backend/tests/fixtures/espn_wnba_summary_prior_away.json` — prior boxscore with starters for away team side
- Create `backend/tests/fixtures/espn_wnba_summary_prior_home.json` — prior boxscore with starters for home team side
- Modify `backend/tests/test_wnba_game_detail_route.py` — assert new fields serialize
- Modify `frontend/src/lib/api.ts` — API TypeScript types
- Modify `frontend/src/components/game/types.ts` — UI camelCase types
- Modify `frontend/src/components/game/mapGameDetail.ts` (+ test) — map new fields
- Modify `frontend/src/components/game/testFixtures.ts` — scheduled fixture helpers
- Create `frontend/src/components/game/MatchupPrediction.tsx` (+ test)
- Create `frontend/src/components/game/ProjectedStarters.tsx` (+ test)
- Create `frontend/src/components/game/SeasonLeaders.tsx` (+ test)
- Create `frontend/src/components/game/InjuryReport.tsx` (+ test)
- Modify `frontend/src/pages/GameDetailPage.tsx` (+ page/router tests as needed)

---

### Task 1: Backend schema + sync preview fields (prediction, leaders, injuries)

**Files:**
- Modify: `backend/app/schemas/wnba_game_detail.py`
- Modify: `backend/app/services/wnba_game_detail.py`
- Modify: `backend/tests/test_wnba_game_detail_normalize.py`
- Create: `backend/tests/fixtures/espn_wnba_summary_scheduled_preview.json`

**Interfaces:**
- Consumes: `normalize_espn_summary(payload, *, espn_event_id, fetched_at) -> WnbaGameDetail`
- Produces:
  - `GameDetailMatchupPrediction(away_win_pct: int, home_win_pct: int, source_label: str)`
  - `GameDetailSeasonLeader(stat: Literal["points","assists","rebounds"], label: str, name: str, value: str)`
  - `GameDetailSeasonLeaders(away: list[...], home: list[...])`
  - `GameDetailInjury(name: str, position: str | None, status: str, detail: str | None)`
  - `GameDetailInjuries(away: list[...], home: list[...])`
  - `WnbaGameDetail.matchup_prediction | None`, `.season_leaders | None`, `.injuries | None`, `.projected_starters | None` (null in this task)

- [ ] **Step 1: Add a minimal scheduled preview fixture**

Write `backend/tests/fixtures/espn_wnba_summary_scheduled_preview.json` by copying the existing summary fixture header shape and adding:

```json
{
  "header": { "...same shape as espn_wnba_summary.json with STATUS_SCHEDULED..." },
  "predictor": {
    "header": "Matchup Predictor",
    "homeTeam": { "id": "home1", "gameProjection": "32.8" },
    "awayTeam": { "id": "away1", "gameProjection": "67.2" }
  },
  "leaders": [
    {
      "team": { "id": "away1", "abbreviation": "MIN" },
      "leaders": [
        {
          "name": "pointsPerGame",
          "displayName": "Points",
          "leaders": [{ "displayValue": "19.5", "athlete": { "displayName": "Olivia Miles" } }]
        },
        {
          "name": "assistsPerGame",
          "displayName": "Assists",
          "leaders": [{ "displayValue": "6.0", "athlete": { "displayName": "Olivia Miles" } }]
        },
        {
          "name": "reboundsPerGame",
          "displayName": "Rebounds",
          "leaders": [{ "displayValue": "7.9", "athlete": { "displayName": "Natasha Howard" } }]
        }
      ]
    },
    {
      "team": { "id": "home1", "abbreviation": "TOR" },
      "leaders": [
        {
          "name": "pointsPerGame",
          "displayName": "Points",
          "leaders": [{ "displayValue": "21.1", "athlete": { "displayName": "Marina Mabrey" } }]
        },
        {
          "name": "assistsPerGame",
          "displayName": "Assists",
          "leaders": [{ "displayValue": "5.5", "athlete": { "displayName": "Julie Allemand" } }]
        },
        {
          "name": "reboundsPerGame",
          "displayName": "Rebounds",
          "leaders": [{ "displayValue": "4.8", "athlete": { "displayName": "Maria Conde" } }]
        }
      ]
    }
  ],
  "injuries": [
    {
      "team": { "id": "home1" },
      "injuries": [
        {
          "status": "Out",
          "athlete": {
            "displayName": "Nyara Sabally",
            "position": { "abbreviation": "F" }
          },
          "details": { "type": "Ribs" }
        }
      ]
    },
    { "team": { "id": "away1" }, "injuries": [] }
  ],
  "plays": [],
  "boxscore": { "teams": [] }
}
```

Ensure `header.competitions[0].competitors` use team ids `away1` / `home1` matching the blocks above, and status type name `STATUS_SCHEDULED`.

- [ ] **Step 2: Write failing normalization tests**

```python
def test_normalize_includes_matchup_prediction_leaders_injuries():
    payload = load_fixture("espn_wnba_summary_scheduled_preview.json")
    detail = normalize_espn_summary(
        payload,
        espn_event_id="401857099",
        fetched_at="2026-07-30T00:00:00-04:00",
    )
    assert detail.status == "scheduled"
    assert detail.matchup_prediction is not None
    assert detail.matchup_prediction.away_win_pct == 67
    assert detail.matchup_prediction.home_win_pct == 33
    assert detail.matchup_prediction.source_label == "ESPN game projection"
    assert detail.season_leaders is not None
    assert [r.stat for r in detail.season_leaders.away] == [
        "points",
        "assists",
        "rebounds",
    ]
    assert detail.season_leaders.away[0].name == "Olivia Miles"
    assert detail.season_leaders.away[0].value == "19.5"
    assert detail.season_leaders.home[0].name == "Marina Mabrey"
    assert detail.injuries is not None
    assert detail.injuries.away == []
    assert detail.injuries.home[0].name == "Nyara Sabally"
    assert detail.injuries.home[0].status == "Out"
    assert detail.injuries.home[0].detail == "Ribs"
    assert detail.injuries.home[0].position == "F"
    assert detail.projected_starters is None


def test_normalize_injuries_null_when_both_sides_empty():
    payload = load_fixture("espn_wnba_summary_scheduled_preview.json")
    payload["injuries"] = [
        {"team": {"id": "away1"}, "injuries": []},
        {"team": {"id": "home1"}, "injuries": []},
    ]
    detail = normalize_espn_summary(
        payload,
        espn_event_id="401857099",
        fetched_at="2026-07-30T00:00:00-04:00",
    )
    assert detail.injuries is None


def test_normalize_preview_fields_null_when_missing():
    payload = load_fixture("espn_wnba_summary.json")
    detail = normalize_espn_summary(
        payload,
        espn_event_id="401749001",
        fetched_at="2026-07-30T00:00:00-04:00",
    )
    assert detail.matchup_prediction is None
    assert detail.season_leaders is None
    assert detail.injuries is None
    assert detail.projected_starters is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `PYTHONPATH=backend pytest backend/tests/test_wnba_game_detail_normalize.py::test_normalize_includes_matchup_prediction_leaders_injuries backend/tests/test_wnba_game_detail_normalize.py::test_normalize_injuries_null_when_both_sides_empty backend/tests/test_wnba_game_detail_normalize.py::test_normalize_preview_fields_null_when_missing -v`

Expected: FAIL (missing fields / AttributeError).

- [ ] **Step 4: Implement schema + normalizers**

Add to `backend/app/schemas/wnba_game_detail.py`:

```python
class GameDetailMatchupPrediction(BaseModel):
    away_win_pct: int
    home_win_pct: int
    source_label: str


class GameDetailStarter(BaseModel):
    jersey: str | None
    name: str
    position: str | None


class GameDetailProjectedStarters(BaseModel):
    note: str
    away: list[GameDetailStarter]
    home: list[GameDetailStarter]


class GameDetailSeasonLeader(BaseModel):
    stat: Literal["points", "assists", "rebounds"]
    label: str
    name: str
    value: str


class GameDetailSeasonLeaders(BaseModel):
    away: list[GameDetailSeasonLeader]
    home: list[GameDetailSeasonLeader]


class GameDetailInjury(BaseModel):
    name: str
    position: str | None
    status: str
    detail: str | None


class GameDetailInjuries(BaseModel):
    away: list[GameDetailInjury]
    home: list[GameDetailInjury]
```

Add to `WnbaGameDetail`:

```python
matchup_prediction: GameDetailMatchupPrediction | None
projected_starters: GameDetailProjectedStarters | None
season_leaders: GameDetailSeasonLeaders | None
injuries: GameDetailInjuries | None
```

Export new models from `__all__`.

In `wnba_game_detail.py`, add helpers:

```python
_LEADER_STAT_MAP = {
    "pointsPerGame": ("points", "Points"),
    "assistsPerGame": ("assists", "Assists"),
    "reboundsPerGame": ("rebounds", "Rebounds"),
}


def _normalize_matchup_prediction(payload: dict) -> GameDetailMatchupPrediction | None:
    predictor = payload.get("predictor")
    if not isinstance(predictor, dict):
        return None
    try:
        away = float((predictor.get("awayTeam") or {}).get("gameProjection"))
        home = float((predictor.get("homeTeam") or {}).get("gameProjection"))
    except (TypeError, ValueError):
        return None
    return GameDetailMatchupPrediction(
        away_win_pct=round(away),
        home_win_pct=round(home),
        source_label="ESPN game projection",
    )


def _leaders_for_team(blocks: list, team_id: str) -> list[GameDetailSeasonLeader]:
    rows: list[GameDetailSeasonLeader] = []
    for block in blocks:
        if str((block.get("team") or {}).get("id") or "") != team_id:
            continue
        for cat in block.get("leaders") or []:
            mapped = _LEADER_STAT_MAP.get(str(cat.get("name") or ""))
            if not mapped:
                continue
            stat, label = mapped
            entry = (cat.get("leaders") or [None])[0] or {}
            athlete = entry.get("athlete") or {}
            name = str(athlete.get("displayName") or "").strip()
            value = str(entry.get("displayValue") or "").strip()
            if not name or not value:
                continue
            rows.append(
                GameDetailSeasonLeader(stat=stat, label=label, name=name, value=value)
            )
    return rows


def _normalize_season_leaders(
    payload: dict, *, away_id: str, home_id: str
) -> GameDetailSeasonLeaders | None:
    blocks = payload.get("leaders")
    if not isinstance(blocks, list) or not blocks:
        return None
    away = _leaders_for_team(blocks, away_id)
    home = _leaders_for_team(blocks, home_id)
    if not away and not home:
        return None
    return GameDetailSeasonLeaders(away=away, home=home)


def _injuries_for_team(blocks: list, team_id: str) -> list[GameDetailInjury]:
    rows: list[GameDetailInjury] = []
    for block in blocks:
        if str((block.get("team") or {}).get("id") or "") != team_id:
            continue
        for item in block.get("injuries") or []:
            athlete = item.get("athlete") or {}
            name = str(athlete.get("displayName") or "").strip()
            if not name:
                continue
            pos = athlete.get("position") or {}
            position = str(pos.get("abbreviation") or "").strip() or None
            details = item.get("details") or {}
            detail = str(details.get("type") or "").strip() or None
            status = str(item.get("status") or "").strip() or "Unknown"
            rows.append(
                GameDetailInjury(
                    name=name, position=position, status=status, detail=detail
                )
            )
    return rows


def _normalize_injuries(
    payload: dict, *, away_id: str, home_id: str
) -> GameDetailInjuries | None:
    blocks = payload.get("injuries")
    if not isinstance(blocks, list):
        return None
    away = _injuries_for_team(blocks, away_id)
    home = _injuries_for_team(blocks, home_id)
    if not away and not home:
        return None
    return GameDetailInjuries(away=away, home=home)
```

Wire into `normalize_espn_summary` return (after team ids known):

```python
away_id = str((away_c.get("team") or {}).get("id") or "")
home_id = str((home_c.get("team") or {}).get("id") or "")
matchup_prediction = _normalize_matchup_prediction(payload)
season_leaders = _normalize_season_leaders(
    payload, away_id=away_id, home_id=home_id
)
injuries = _normalize_injuries(payload, away_id=away_id, home_id=home_id)
# projected_starters = None for now
```

Pass the four fields into `WnbaGameDetail(...)`. Update any existing tests that construct `WnbaGameDetail` directly if they break.

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=backend pytest backend/tests/test_wnba_game_detail_normalize.py -v`

Expected: PASS (including existing win-probability tests).

- [ ] **Step 6: Commit**

```bash
git add backend/app/schemas/wnba_game_detail.py backend/app/services/wnba_game_detail.py backend/tests/test_wnba_game_detail_normalize.py backend/tests/fixtures/espn_wnba_summary_scheduled_preview.json
git commit -m "Add ESPN matchup prediction, leaders, and injuries to game detail."
```

---

### Task 2: Projected starters fan-out for scheduled games

**Files:**
- Modify: `backend/app/services/wnba_game_detail.py`
- Modify: `backend/tests/test_wnba_game_detail_normalize.py`
- Create: `backend/tests/fixtures/espn_wnba_summary_prior_away.json`
- Create: `backend/tests/fixtures/espn_wnba_summary_prior_home.json`
- Modify: `backend/tests/test_wnba_game_detail_route.py` (optional get_game_detail unit if route already covers)

**Interfaces:**
- Consumes: primary summary `lastFiveGames`, `fetch_espn_summary(event_id)`
- Produces: `normalize_espn_summary(..., prior_game_summaries: dict[str, dict] | None = None)` where keys are ESPN team ids; `GameDetailProjectedStarters`
- `get_game_detail` fetches prior summaries only when status is scheduled

- [ ] **Step 1: Extend scheduled fixture with lastFiveGames; add prior fixtures**

Append to scheduled preview fixture:

```json
"lastFiveGames": [
  {
    "team": { "id": "home1", "abbreviation": "TOR" },
    "events": [{ "id": "401857060" }]
  },
  {
    "team": { "id": "away1", "abbreviation": "MIN" },
    "events": [{ "id": "401857069" }]
  }
]
```

Create prior fixtures with `boxscore.players` for the matching team only (minimal):

```json
{
  "boxscore": {
    "players": [
      {
        "team": { "id": "away1", "abbreviation": "MIN" },
        "statistics": [
          {
            "athletes": [
              {
                "starter": true,
                "athlete": {
                  "displayName": "Natasha Howard",
                  "jersey": "1",
                  "position": { "abbreviation": "F" }
                }
              },
              {
                "starter": true,
                "athlete": {
                  "displayName": "Napheesa Collier",
                  "jersey": "24",
                  "position": { "abbreviation": "F" }
                }
              },
              {
                "starter": true,
                "athlete": {
                  "displayName": "Kayla McBride",
                  "jersey": "21",
                  "position": { "abbreviation": "G" }
                }
              },
              {
                "starter": true,
                "athlete": {
                  "displayName": "Courtney Williams",
                  "jersey": "10",
                  "position": { "abbreviation": "G" }
                }
              },
              {
                "starter": true,
                "athlete": {
                  "displayName": "Olivia Miles",
                  "jersey": "5",
                  "position": { "abbreviation": "G" }
                }
              },
              {
                "starter": false,
                "athlete": {
                  "displayName": "Bench Player",
                  "jersey": "99",
                  "position": { "abbreviation": "G" }
                }
              }
            ]
          }
        ]
      }
    ]
  }
}
```

Mirror for home (`home1`) with five TOR starters (e.g. Conde, Juskaite, Fagbenle, Mabrey, Allemand).

- [ ] **Step 2: Write failing starter tests**

```python
def test_normalize_projected_starters_from_prior_summaries():
    payload = load_fixture("espn_wnba_summary_scheduled_preview.json")
    priors = {
        "away1": load_fixture("espn_wnba_summary_prior_away.json"),
        "home1": load_fixture("espn_wnba_summary_prior_home.json"),
    }
    detail = normalize_espn_summary(
        payload,
        espn_event_id="401857099",
        fetched_at="2026-07-30T00:00:00-04:00",
        prior_game_summaries=priors,
    )
    assert detail.projected_starters is not None
    assert detail.projected_starters.note == "from each team's last game"
    assert len(detail.projected_starters.away) == 5
    assert detail.projected_starters.away[0].name == "Natasha Howard"
    assert detail.projected_starters.away[0].jersey == "1"
    assert detail.projected_starters.away[0].position == "F"
    assert len(detail.projected_starters.home) == 5


def test_normalize_projected_starters_null_if_either_side_missing():
    payload = load_fixture("espn_wnba_summary_scheduled_preview.json")
    priors = {"away1": load_fixture("espn_wnba_summary_prior_away.json")}
    detail = normalize_espn_summary(
        payload,
        espn_event_id="401857099",
        fetched_at="2026-07-30T00:00:00-04:00",
        prior_game_summaries=priors,
    )
    assert detail.projected_starters is None


def test_normalize_ignores_priors_when_not_scheduled():
    payload = load_fixture("espn_wnba_summary.json")
    priors = {
        "away1": load_fixture("espn_wnba_summary_prior_away.json"),
        "home1": load_fixture("espn_wnba_summary_prior_home.json"),
    }
    detail = normalize_espn_summary(
        payload,
        espn_event_id="401749001",
        fetched_at="2026-07-30T00:00:00-04:00",
        prior_game_summaries=priors,
    )
    assert detail.status != "scheduled"
    assert detail.projected_starters is None
```

Add an async `get_game_detail` test (new or in route file) that mocks `fetch_espn_summary` to return scheduled payload first, then prior payloads for the lastFiveGames event ids, and asserts `projected_starters` is populated. Example:

```python
@pytest.mark.asyncio
async def test_get_game_detail_fetches_prior_games_for_starters(monkeypatch):
    from app.services import wnba_game_detail as svc

    svc.clear_game_detail_cache()
    scheduled = load_fixture("espn_wnba_summary_scheduled_preview.json")
    prior_away = load_fixture("espn_wnba_summary_prior_away.json")
    prior_home = load_fixture("espn_wnba_summary_prior_home.json")

    async def fake_fetch(event_id: str) -> dict:
        return {
            "401857099": scheduled,
            "401857069": prior_away,
            "401857060": prior_home,
        }[event_id]

    monkeypatch.setattr(svc, "fetch_espn_summary", fake_fetch)
    detail = await svc.get_game_detail("401857099")
    assert detail.projected_starters is not None
    assert len(detail.projected_starters.away) == 5
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `PYTHONPATH=backend pytest backend/tests/test_wnba_game_detail_normalize.py::test_normalize_projected_starters_from_prior_summaries backend/tests/test_wnba_game_detail_normalize.py::test_normalize_projected_starters_null_if_either_side_missing backend/tests/test_wnba_game_detail_normalize.py::test_normalize_ignores_priors_when_not_scheduled -v`

Expected: FAIL (`prior_game_summaries` unexpected / starters still None).

- [ ] **Step 4: Implement starter extraction + fan-out**

```python
def _starters_from_summary(summary: dict, *, team_id: str) -> list[GameDetailStarter] | None:
    players = (summary.get("boxscore") or {}).get("players")
    if not isinstance(players, list):
        return None
    for block in players:
        if str((block.get("team") or {}).get("id") or "") != team_id:
            continue
        stats = block.get("statistics") or []
        if not stats:
            return None
        athletes = stats[0].get("athletes") or []
        starters: list[GameDetailStarter] = []
        for row in athletes:
            if not row.get("starter"):
                continue
            athlete = row.get("athlete") or {}
            name = str(athlete.get("displayName") or "").strip()
            if not name:
                continue
            jersey = athlete.get("jersey")
            jersey_s = str(jersey).strip() if jersey is not None and str(jersey).strip() else None
            pos = athlete.get("position") or {}
            position = str(pos.get("abbreviation") or "").strip() or None
            starters.append(
                GameDetailStarter(jersey=jersey_s, name=name, position=position)
            )
            if len(starters) == 5:
                break
        return starters if len(starters) == 5 else None
    return None


def _normalize_projected_starters(
    *,
    status: GameStatus,
    away_id: str,
    home_id: str,
    prior_game_summaries: dict[str, dict] | None,
) -> GameDetailProjectedStarters | None:
    if status != "scheduled" or not prior_game_summaries:
        return None
    away_summary = prior_game_summaries.get(away_id)
    home_summary = prior_game_summaries.get(home_id)
    if not away_summary or not home_summary:
        return None
    away = _starters_from_summary(away_summary, team_id=away_id)
    home = _starters_from_summary(home_summary, team_id=home_id)
    if away is None or home is None:
        return None
    return GameDetailProjectedStarters(
        note="from each team's last game",
        away=away,
        home=home,
    )


def _prior_event_ids_by_team(payload: dict) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for block in payload.get("lastFiveGames") or []:
        team_id = str((block.get("team") or {}).get("id") or "")
        events = block.get("events") or []
        if not team_id or not events:
            continue
        event_id = str(events[0].get("id") or "").strip()
        if event_id:
            mapping[team_id] = event_id
    return mapping
```

Update `normalize_espn_summary` signature:

```python
def normalize_espn_summary(
    payload: dict,
    *,
    espn_event_id: str,
    fetched_at: str,
    prior_game_summaries: dict[str, dict] | None = None,
) -> WnbaGameDetail:
```

Set `projected_starters = _normalize_projected_starters(...)`.

Update `get_game_detail` after a successful primary fetch and before/around normalize:

```python
# Peek status cheaply from payload header, or normalize twice — prefer peek:
status_block = (
    ((payload.get("header") or {}).get("competitions") or [{}])[0].get("status")
    or {}
)
status, _ = _detail_status(status_block)
prior_game_summaries: dict[str, dict] | None = None
if status == "scheduled":
    prior_ids = _prior_event_ids_by_team(payload)
    # Resolve away/home team ids from competitors the same way normalize does
    # then fetch both prior event ids with asyncio.gather; on any failure → None
    ...
detail = normalize_espn_summary(
    payload,
    espn_event_id=espn_event_id,
    fetched_at=...,
    prior_game_summaries=prior_game_summaries,
)
```

Concrete fan-out body:

```python
import asyncio

async def _fetch_prior_game_summaries(
    payload: dict, *, away_id: str, home_id: str
) -> dict[str, dict] | None:
    prior_ids = _prior_event_ids_by_team(payload)
    away_event = prior_ids.get(away_id)
    home_event = prior_ids.get(home_id)
    if not away_event or not home_event:
        return None
    try:
        away_summary, home_summary = await asyncio.gather(
            fetch_espn_summary(away_event),
            fetch_espn_summary(home_event),
        )
    except Exception:
        return None
    return {away_id: away_summary, home_id: home_summary}
```

Extract competitor team ids with a small helper shared with normalize (or duplicate the competitors lookup once in `get_game_detail`).

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=backend pytest backend/tests/test_wnba_game_detail_normalize.py backend/tests/test_wnba_game_detail_route.py -v`

Expected: PASS. Also run the new asyncio get_game_detail test if placed in a separate file.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/wnba_game_detail.py backend/tests/fixtures/espn_wnba_summary_scheduled_preview.json backend/tests/fixtures/espn_wnba_summary_prior_away.json backend/tests/fixtures/espn_wnba_summary_prior_home.json backend/tests/test_wnba_game_detail_normalize.py backend/tests/test_wnba_game_detail_route.py
git commit -m "Resolve projected starters from prior ESPN game boxscores."
```

---

### Task 3: Frontend API types + mapGameDetail

**Files:**
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/components/game/types.ts`
- Modify: `frontend/src/components/game/mapGameDetail.ts`
- Modify: `frontend/src/components/game/mapGameDetail.test.ts`
- Modify: `frontend/src/components/game/testFixtures.ts`

**Interfaces:**
- Consumes: snake_case API fields from Task 1–2
- Produces: camelCase `GameDetail` fields:
  - `matchupPrediction`, `projectedStarters`, `seasonLeaders`, `injuries`

- [ ] **Step 1: Write failing mapper test**

```ts
it("maps matchup preview fields", () => {
  const mapped = mapGameDetail({
    ...apiDetailBase,
    matchup_prediction: {
      away_win_pct: 67,
      home_win_pct: 33,
      source_label: "ESPN game projection",
    },
    projected_starters: {
      note: "from each team's last game",
      away: [{ jersey: "1", name: "Natasha Howard", position: "F" }],
      home: [{ jersey: "10", name: "Maria Conde", position: "F" }],
    },
    season_leaders: {
      away: [
        {
          stat: "points",
          label: "Points",
          name: "Olivia Miles",
          value: "19.5",
        },
      ],
      home: [],
    },
    injuries: {
      away: [],
      home: [
        {
          name: "Nyara Sabally",
          position: "F",
          status: "Out",
          detail: "Ribs",
        },
      ],
    },
  });
  expect(mapped.matchupPrediction?.awayWinPct).toBe(67);
  expect(mapped.projectedStarters?.away[0].name).toBe("Natasha Howard");
  expect(mapped.seasonLeaders?.away[0].stat).toBe("points");
  expect(mapped.injuries?.home[0].detail).toBe("Ribs");
});

it("maps null preview fields", () => {
  const mapped = mapGameDetail({
    ...apiDetailBase,
    matchup_prediction: null,
    projected_starters: null,
    season_leaders: null,
    injuries: null,
  });
  expect(mapped.matchupPrediction).toBeNull();
  expect(mapped.projectedStarters).toBeNull();
  expect(mapped.seasonLeaders).toBeNull();
  expect(mapped.injuries).toBeNull();
});
```

Reuse/extend whatever `apiDetailBase` pattern already exists in `mapGameDetail.test.ts`.

- [ ] **Step 2: Run test to verify it fails**

Run: `npm --prefix frontend run test -- src/components/game/mapGameDetail.test.ts`

Expected: FAIL (types / missing properties).

- [ ] **Step 3: Implement types + mapper**

In `api.ts`, add API types and fields on `ApiWnbaGameDetail`.

In `types.ts`:

```ts
export type GameDetailMatchupPrediction = {
  awayWinPct: number;
  homeWinPct: number;
  sourceLabel: string;
};

export type GameDetailStarter = {
  jersey: string | null;
  name: string;
  position: string | null;
};

export type GameDetailProjectedStarters = {
  note: string;
  away: GameDetailStarter[];
  home: GameDetailStarter[];
};

export type GameDetailSeasonLeader = {
  stat: "points" | "assists" | "rebounds";
  label: string;
  name: string;
  value: string;
};

export type GameDetailSeasonLeaders = {
  away: GameDetailSeasonLeader[];
  home: GameDetailSeasonLeader[];
};

export type GameDetailInjury = {
  name: string;
  position: string | null;
  status: string;
  detail: string | null;
};

export type GameDetailInjuries = {
  away: GameDetailInjury[];
  home: GameDetailInjury[];
};
```

Add to `GameDetail`:

```ts
matchupPrediction: GameDetailMatchupPrediction | null;
projectedStarters: GameDetailProjectedStarters | null;
seasonLeaders: GameDetailSeasonLeaders | null;
injuries: GameDetailInjuries | null;
```

Map in `mapGameDetail.ts`. Update `testFixtures.ts` `detail` / `buildGameDetailFixture` with `null` preview fields (and a scheduled helper if useful).

- [ ] **Step 4: Run tests to verify they pass**

Run: `npm --prefix frontend run test -- src/components/game/mapGameDetail.test.ts`

Expected: PASS. Fix any fixture type errors in other game component tests.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/api.ts frontend/src/components/game/types.ts frontend/src/components/game/mapGameDetail.ts frontend/src/components/game/mapGameDetail.test.ts frontend/src/components/game/testFixtures.ts
git commit -m "Map ESPN matchup preview fields into game detail types."
```

---

### Task 4: MatchupPrediction + ProjectedStarters UI

**Files:**
- Create: `frontend/src/components/game/MatchupPrediction.tsx`
- Create: `frontend/src/components/game/MatchupPrediction.test.tsx`
- Create: `frontend/src/components/game/ProjectedStarters.tsx`
- Create: `frontend/src/components/game/ProjectedStarters.test.tsx`

**Interfaces:**
- Consumes: `MatchupPrediction({ detail }: { detail: GameDetail })`, `ProjectedStarters({ detail }: { detail: GameDetail })`
- Produces: section UI or `null` when field missing

- [ ] **Step 1: Write failing component tests**

```tsx
it("renders prediction bar and source", () => {
  render(
    <MatchupPrediction
      detail={buildScheduledDetail({
        matchupPrediction: {
          awayWinPct: 67,
          homeWinPct: 33,
          sourceLabel: "ESPN game projection",
        },
      })}
    />,
  );
  expect(screen.getByText(/Matchup prediction/i)).toBeInTheDocument();
  expect(screen.getByText(/MIN/)).toBeInTheDocument();
  expect(screen.getByText("67%")).toBeInTheDocument();
  expect(screen.getByText("33%")).toBeInTheDocument();
  expect(screen.getByText("ESPN game projection")).toBeInTheDocument();
});

it("renders nothing without prediction", () => {
  const { container } = render(
    <MatchupPrediction
      detail={buildScheduledDetail({ matchupPrediction: null })}
    />,
  );
  expect(container).toBeEmptyDOMElement();
});

it("renders starters for both teams", () => {
  render(
    <ProjectedStarters
      detail={buildScheduledDetail({
        projectedStarters: {
          note: "from each team's last game",
          away: [
            { jersey: "1", name: "Natasha Howard", position: "F" },
          ],
          home: [
            { jersey: "10", name: "Maria Conde", position: "F" },
          ],
        },
      })}
    />,
  );
  expect(screen.getByText(/Projected starters/i)).toBeInTheDocument();
  expect(screen.getByText(/from each team's last game/i)).toBeInTheDocument();
  expect(screen.getByText("Natasha Howard")).toBeInTheDocument();
  expect(screen.getByText("Maria Conde")).toBeInTheDocument();
});
```

Add `buildScheduledDetail` in `testFixtures.ts` if not already present.

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm --prefix frontend run test -- src/components/game/MatchupPrediction.test.tsx src/components/game/ProjectedStarters.test.tsx`

Expected: FAIL (modules missing).

- [ ] **Step 3: Implement components**

`MatchupPrediction.tsx` — charcoal card; title; flex bar with away width `awayWinPct%` and home width `homeWinPct%` using `detail.away.color` / `detail.home.color`; labels `AWAY_ABBREV xx%` / `xx% HOME_ABBREV`; muted source line.

`ProjectedStarters.tsx` — charcoal card; title + note; two columns (`md:grid-cols-2`); rows `#jersey name position` (omit `#` when jersey null).

Return `null` when the corresponding detail field is null.

- [ ] **Step 4: Run tests to verify they pass**

Run: `npm --prefix frontend run test -- src/components/game/MatchupPrediction.test.tsx src/components/game/ProjectedStarters.test.tsx`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/game/MatchupPrediction.tsx frontend/src/components/game/MatchupPrediction.test.tsx frontend/src/components/game/ProjectedStarters.tsx frontend/src/components/game/ProjectedStarters.test.tsx frontend/src/components/game/testFixtures.ts
git commit -m "Add matchup prediction and projected starters panels."
```

---

### Task 5: SeasonLeaders + InjuryReport UI

**Files:**
- Create: `frontend/src/components/game/SeasonLeaders.tsx`
- Create: `frontend/src/components/game/SeasonLeaders.test.tsx`
- Create: `frontend/src/components/game/InjuryReport.tsx`
- Create: `frontend/src/components/game/InjuryReport.test.tsx`

**Interfaces:**
- Consumes: `SeasonLeaders({ detail })`, `InjuryReport({ detail })`
- Produces: section UI or `null`

- [ ] **Step 1: Write failing tests**

```tsx
it("renders points assists rebounds leaders", () => {
  render(
    <SeasonLeaders
      detail={buildScheduledDetail({
        seasonLeaders: {
          away: [
            {
              stat: "points",
              label: "Points",
              name: "Olivia Miles",
              value: "19.5",
            },
          ],
          home: [
            {
              stat: "points",
              label: "Points",
              name: "Marina Mabrey",
              value: "21.1",
            },
          ],
        },
      })}
    />,
  );
  expect(screen.getByText(/Season leaders/i)).toBeInTheDocument();
  expect(screen.getByText("Olivia Miles")).toBeInTheDocument();
  expect(screen.getByText("19.5")).toBeInTheDocument();
  expect(screen.getByText("Marina Mabrey")).toBeInTheDocument();
});

it("renders injury rows", () => {
  render(
    <InjuryReport
      detail={buildScheduledDetail({
        injuries: {
          away: [],
          home: [
            {
              name: "Nyara Sabally",
              position: "F",
              status: "Out",
              detail: "Ribs",
            },
          ],
        },
      })}
    />,
  );
  expect(screen.getByText(/Injury report/i)).toBeInTheDocument();
  expect(screen.getByText("Nyara Sabally")).toBeInTheDocument();
  expect(screen.getByText(/Out/)).toBeInTheDocument();
  expect(screen.getByText(/Ribs/)).toBeInTheDocument();
});
```

Also assert each returns empty DOM when field is null.

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm --prefix frontend run test -- src/components/game/SeasonLeaders.test.tsx src/components/game/InjuryReport.test.tsx`

Expected: FAIL.

- [ ] **Step 3: Implement components**

Two-column layouts matching starters; season leaders show `label`, `name`, `value`; injuries show name, position, status, detail. Return `null` when field is null. Empty side list may show muted “None listed” for injuries when the other side has rows.

- [ ] **Step 4: Run tests to verify they pass**

Run: `npm --prefix frontend run test -- src/components/game/SeasonLeaders.test.tsx src/components/game/InjuryReport.test.tsx`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/game/SeasonLeaders.tsx frontend/src/components/game/SeasonLeaders.test.tsx frontend/src/components/game/InjuryReport.tsx frontend/src/components/game/InjuryReport.test.tsx
git commit -m "Add season leaders and injury report panels."
```

---

### Task 6: GameDetailPage status branch + verification

**Files:**
- Modify: `frontend/src/pages/GameDetailPage.tsx`
- Modify: `frontend/src/AppRouter.test.tsx` and/or create `frontend/src/pages/GameDetailPage.test.tsx`
- Verify: `frontend/src/hooks/useGameDetail.ts` (already skips poll unless live/halftime — no change unless tests missing)

**Interfaces:**
- Consumes: mapped `GameDetail.status`
- Produces: scheduled stack vs live stack

- [ ] **Step 1: Write failing page tests**

```tsx
it("shows matchup preview sections for scheduled games", async () => {
  // mock fetchGameDetail / useGameDetail data with status scheduled + preview fields
  render(/* GameDetailPage or router /games/401857099 */);
  expect(await screen.findByText(/Matchup prediction/i)).toBeInTheDocument();
  expect(screen.getByText(/Projected starters/i)).toBeInTheDocument();
  expect(screen.getByText(/Season leaders/i)).toBeInTheDocument();
  expect(screen.queryByText(/Shot chart/i)).not.toBeInTheDocument();
  expect(screen.queryByText(/Play-by-play/i)).not.toBeInTheDocument();
});

it("shows live panels for live games", async () => {
  // mock status live without requiring preview fields
  render(/* ... */);
  expect(await screen.findByText(/Shot chart/i)).toBeInTheDocument();
  expect(screen.getByText(/Play-by-play/i)).toBeInTheDocument();
  expect(screen.queryByText(/Matchup prediction/i)).not.toBeInTheDocument();
});
```

Follow existing `AppRouter.test.tsx` fetch-mock patterns.

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm --prefix frontend run test -- src/AppRouter.test.tsx src/pages/GameDetailPage.test.tsx`

Expected: FAIL (scheduled still shows shot chart).

- [ ] **Step 3: Implement page branch**

```tsx
const isScheduled = detail.status === "scheduled";

return (
  <div className="mx-auto max-w-6xl space-y-4 px-4 py-6 sm:px-6">
    <GameHeader detail={detail} />
    {isScheduled ? (
      <>
        <MatchupPrediction detail={detail} />
        <ProjectedStarters detail={detail} />
        <SeasonLeaders detail={detail} />
        <InjuryReport detail={detail} />
      </>
    ) : (
      <>
        <div className="grid gap-4 lg:grid-cols-2">
          <ShotChart detail={detail} />
          <PlayByPlay detail={detail} />
        </div>
        <WinProbabilityPanel detail={detail} />
      </>
    )}
  </div>
);
```

- [ ] **Step 4: Run full verification**

Run:

```bash
PYTHONPATH=backend pytest backend/tests/test_wnba_game_detail_normalize.py backend/tests/test_wnba_game_detail_route.py -v
npm --prefix frontend run test -- src/components/game src/pages/GameDetailPage.test.tsx src/AppRouter.test.tsx src/hooks/useGameDetail.test.tsx
npm --prefix frontend run build
```

Expected: all PASS; build succeeds.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/GameDetailPage.tsx frontend/src/pages/GameDetailPage.test.tsx frontend/src/AppRouter.test.tsx
git commit -m "Branch game detail page for scheduled matchup preview."
```

---

## Spec coverage checklist

| Spec requirement | Task |
| --- | --- |
| Matchup prediction from ESPN `predictor` | Task 1, 4 |
| Season leaders PTS/AST/REB | Task 1, 5 |
| Injuries | Task 1, 5 |
| Projected starters from last game boxscore | Task 2, 4 |
| Starters fan-out only when scheduled | Task 2 |
| All-or-nothing starters null | Task 2 |
| Empty injuries → null | Task 1 |
| Scheduled UI branch | Task 6 |
| Live/final unchanged | Task 6 |
| No poll for scheduled | existing `useGameDetail` (verify in Task 6) |
| Omit null sections | Tasks 4–5 (`return null`) |
| Single existing API endpoint | Tasks 1–2 |

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-30-wnba-scheduled-matchup-preview.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks  
**2. Inline Execution** — execute tasks in this session with executing-plans checkpoints  

Which approach?
