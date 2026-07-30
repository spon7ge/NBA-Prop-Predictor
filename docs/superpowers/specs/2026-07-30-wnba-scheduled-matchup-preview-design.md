# WNBA scheduled matchup preview (ESPN)

Date: 2026-07-30  
Status: Approved for planning

## Goal

When a user opens a **scheduled** WNBA game from Matchups (or any link to `/games/:espnEventId`), show a boxseats-style matchup preview: both teams, matchup prediction, projected starting five, season leaders, and injury report — all from ESPN. **Live / halftime / final** games keep the existing header + shot chart + play-by-play + win probability layout and do not show the preview sections.

## Decisions

| Topic | Choice |
| --- | --- |
| Status branching | Scheduled → preview; live / halftime / final → existing live layout |
| Live/final extra sections | None (no prediction / starters / leaders / injuries on those statuses) |
| Injury report | Included in v1 for scheduled games |
| Architecture | Extend existing `GET /api/wnba/games/{espnEventId}` (Approach 1) |
| Route | Unchanged `/games/:espnEventId` under `HomeChromeLayout` |
| Branding | HoopVista dark chrome; team colors on abbrevs and prediction bar |
| Starters source | Prior-game boxscore starters via `lastFiveGames` (not on scheduled summary) |

## Architecture

```
Matchups / LIVE NOW / ticker
        │
        ▼
  /games/:espnEventId
        │
        ▼
  useGameDetail(id) → GET /api/wnba/games/{espnEventId}
        │
        ▼
  ESPN summary?event=…
        ├── always: header, shots, plays, win_probability (as today)
        ├── always when present: predictor, leaders, injuries → normalize
        └── if status == scheduled:
              lastFiveGames → prior event summaries → starter lineups
        │
        ▼
  GameDetailPage
    ├── scheduled: GameHeader + MatchupPrediction + ProjectedStarters
    │              + SeasonLeaders + InjuryReport
    └── live/halftime/final: GameHeader + ShotChart + PlayByPlay
                             + WinProbabilityPanel
```

## Data schema

### Detail response additions

`GET /api/wnba/games/{espnEventId}` gains optional fields (null when absent or not applicable):

```json
{
  "matchup_prediction": {
    "away_win_pct": 67,
    "home_win_pct": 33,
    "source_label": "ESPN game projection"
  },
  "projected_starters": {
    "note": "from each team's last game",
    "away": [
      { "jersey": "1", "name": "Natasha Howard", "position": "F" }
    ],
    "home": []
  },
  "season_leaders": {
    "away": [
      {
        "stat": "points",
        "label": "Points",
        "name": "Olivia Miles",
        "value": "19.5"
      }
    ],
    "home": []
  },
  "injuries": {
    "away": [
      {
        "name": "Player Name",
        "position": "F",
        "status": "Out",
        "detail": "Ribs"
      }
    ],
    "home": []
  }
}
```

### Nested types

| Type | Fields |
| --- | --- |
| Matchup prediction | `away_win_pct`, `home_win_pct` (ints 0–100), `source_label` |
| Starter | `jersey` (`string \| null`), `name`, `position` (`string \| null`) |
| Projected starters | `note`, `away[]`, `home[]` |
| Season leader | `stat` (`points` \| `assists` \| `rebounds`), `label`, `name`, `value` (display string) |
| Season leaders | `away[]`, `home[]` |
| Injury | `name`, `position` (`string \| null`), `status`, `detail` (`string \| null`) |
| Injuries | `away[]`, `home[]` |

### ESPN source mapping

Endpoint (existing): `https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/summary?event={espnEventId}`

| UI need | ESPN source |
| --- | --- |
| Matchup prediction | `predictor.awayTeam.gameProjection` / `predictor.homeTeam.gameProjection` (round to int %) |
| Season leaders | `leaders[]` — categories `pointsPerGame`, `assistsPerGame`, `reboundsPerGame` only |
| Injuries | `injuries[]` — athlete name/position, `status`, `details.type` as detail |
| Projected starters | For each team in `lastFiveGames`, take the most recent prior `events[0].id`, fetch that event’s summary, read `boxscore.players[].statistics[0].athletes` where `starter === true` (jersey, displayName, position abbreviation) |

### Backend rules

- Normalize prediction / leaders / injuries whenever present on the primary summary (cheap; no extra network).
- Perform prior-game starter fan-out **only** when game `status` is `scheduled` and prior event ids are available.
- Cap starters at 5 per team (first five with `starter: true`).
- If prior-game fetch or starter parse fails for either team, set `projected_starters` to `null` (all-or-nothing; keeps the UI simple).
- If a whole block is missing from ESPN, return `null` for that field; do not fail the detail response.
- Empty injury arrays on both sides → `injuries: null` so the frontend omits the card.
- Keep existing cache / stale-while-error / `Cache-Control: no-store` behavior for the primary summary. Prior-game fetches for starters may use a short in-memory cache keyed by event id to avoid duplicate ESPN hits when many scheduled games share recent opponents.

## UI

### Scheduled page regions (top → bottom)

| Region | Behavior |
| --- | --- |
| Back + status | Existing back row; tip/status label |
| Game header | Existing `GameHeader` (teams in team colors, venue/time, score boxes as `–` / null pre-tip) |
| Matchup prediction | Title; horizontal bar split by team colors; `AWAY xx%` / `xx% HOME`; source line under bar |
| Projected starters | Title + note; two columns (away \| home); jersey · name · position |
| Season leaders | Title; two columns; Points / Assists / Rebounds with name + value |
| Injury report | Title; two columns; name, position, status, detail when present |

Omit any section whose corresponding payload field is `null`.

### Live / final

Unchanged: `GameHeader` + `ShotChart` + `PlayByPlay` + `WinProbabilityPanel`. Preview components are not mounted.

### Visual system

- Dark page (`#0B0B0B`), charcoal cards (`#141414`), white/gray text — match existing game detail.
- Team accent colors from detail `away.color` / `home.color` on abbrevs and prediction bar.
- Geist Sans / Mono already in app; no new font stack.
- Mobile: single column stack; starters / leaders / injuries become stacked team blocks under `md` two-column where helpful.

### Polling

| Status | Polling |
| --- | --- |
| `scheduled` | No interval refetch (single fetch) |
| `live` / `halftime` | Existing ~15–20s |
| `final` | None |

## File layout

```
frontend/src/pages/GameDetailPage.tsx          # status branch
frontend/src/components/game/
  MatchupPrediction.tsx
  ProjectedStarters.tsx
  SeasonLeaders.tsx
  InjuryReport.tsx
  types.ts                                     # + preview types
  mapGameDetail.ts                             # map new fields
frontend/src/hooks/useGameDetail.ts            # scheduled: no poll

backend/app/schemas/wnba_game_detail.py        # + preview models
backend/app/services/wnba_game_detail.py       # normalize + starter fan-out
backend/tests/fixtures/                        # predictor/leaders/injuries + prior boxscore
backend/tests/test_wnba_game_detail_*.py
```

## Testing

### Backend

- Fixture with `predictor` → rounded away/home win % + source label.
- Fixture with `leaders` → PTS/AST/REB per team; ignore other categories.
- Fixture with `injuries` → mapped rows; both-empty → `null`.
- Scheduled normalize with `lastFiveGames` + mocked prior summaries → five starters per side with jersey/name/position.
- Live status does not call prior-game fan-out.
- Missing blocks → null fields; overall detail still `200`.

### Frontend

- `GameDetailPage`: scheduled mounts preview sections; live mounts shot/PBP/win %.
- Each preview component renders from fixtures; null field → section not rendered.
- `useGameDetail` does not poll when status is scheduled.
- Existing live/final tests continue to pass.

## Out of scope

- NBA matchup preview
- Odds, ATS, last-five records, season series UI
- Changing route to `/wnba/game/:id`
- Changing Back target from `/` to matchups
- Showing preview sections on live/final
- Frontend calling ESPN directly

## Success criteria

- From `/wnba/matchups`, opening a scheduled game shows teams, ESPN matchup prediction, projected starters (last game), season leaders (PTS/AST/REB), and injuries when ESPN provides them.
- Live and final games continue to show the existing shot chart, play-by-play, and win probability UI without preview sections.
- Partial ESPN gaps hide only the affected section.
- `npm run build` (frontend) and backend game-detail tests pass.
