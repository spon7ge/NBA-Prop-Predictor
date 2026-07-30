# WNBA game detail win probability panel

Date: 2026-07-30  
Status: Approved for planning

## Goal

Extend the existing WNBA game detail page with a single shared panel beneath `Shot chart` and `Play-by-play` that shows ESPN-driven win probability and team stats for the current game. The panel should match the dark visual style already used on the page and feel interactive on hover/focus.

## Decisions

| Topic | Choice |
| --- | --- |
| Placement | One shared full-width panel below the two existing game-detail panels |
| Data source | Extend the existing backend ESPN summary normalization |
| Chart behavior | Interactive line/area chart with hover and keyboard-focus detail |
| Score snapshot | Show the relevant score and both teams' win percentages for the hovered point |
| Team stats | Render below the chart as horizontal comparison rows |
| Missing ESPN data | Show a compact unavailable state instead of failing the whole page |

## Architecture

Existing game detail flow stays intact; only the payload and page composition expand:

```text
/games/:espnEventId
       |
       v
useGameDetail(id) -> GET /api/wnba/games/{espnEventId}
       |
       v
ESPN summary payload -> normalize header + shots + plays + win probability + team stats
       |
       v
GameDetailPage
  - GameHeader
  - ShotChart
  - PlayByPlay
  - WinProbabilityPanel
```

This remains a single game-detail endpoint rather than introducing a second request just for predictor data.

## Data schema

### Detail response additions

`GET /api/wnba/games/{espnEventId}` gains:

```json
{
  "win_probability": {
    "summary": "Above the midline favors PHX",
    "timeline": [
      {
        "id": "40185709810",
        "period": 1,
        "clock": "4:29",
        "away_score": 10,
        "home_score": 8,
        "away_win_pct": 46,
        "home_win_pct": 54,
        "team_id": "129153"
      }
    ],
    "team_stats": [
      {
        "key": "field_goal_pct",
        "label": "Field goal %",
        "away_value": 41,
        "home_value": 49
      }
    ]
  }
}
```

### New nested types

**Win probability point:** `id`, `period`, `clock`, `away_score`, `home_score`, `away_win_pct`, `home_win_pct`, `team_id`.

**Team stat row:** `key`, `label`, `away_value`, `home_value`.

### Mapping rules

- Parse win-probability points from the ESPN summary source used for game detail.
- Preserve score, period, and clock context on each point so the frontend tooltip can describe the exact state.
- Normalize percentages as whole numbers from `0` to `100`.
- Normalize stat rows only for the small set needed by the UI: `Field goal %`, `Three point %`, `Free throw %`, `Rebounds`, `Offensive rebounds`, `Assists`.
- If either the timeline or the stat block is absent, the panel still renders with the available subsection.
- If neither exists, return `win_probability: null` and let the frontend show an unavailable state.

## UI

### Page layout

`GameDetailPage` changes from a two-panel grid to:

1. `GameHeader`
2. Two-column grid: `ShotChart` | `PlayByPlay`
3. Full-width `WinProbabilityPanel`

### Panel contents

| Region | Behavior |
| --- | --- |
| Title row | `Win probability` heading with muted helper text based on the current leader / summary |
| Chart | Team-colored line/area chart over time; midline at 50%; hover/focus reveals score state and percentages |
| Tooltip / active point | Shows away score-home score plus away/home win percentages at the selected point |
| Team stats block | Horizontal comparison rows under the chart with away left, home right, label centered or aligned consistently |
| Legend | Team abbreviations and colors visible near the chart or stats rows |

### Interaction details

- Default active point should be the latest point when live/final data exists.
- Hovering the chart updates the active point.
- Keyboard users can move across points via focusable markers or a simple accessible fallback state.
- The chart should remain readable on smaller screens; stacking labels is acceptable if horizontal space gets tight.

### Empty / fallback states

| State | UI |
| --- | --- |
| No win probability yet | Muted panel copy such as `Win probability unavailable for this game yet.` |
| Team stats only | Render stat rows without the chart |
| Timeline only | Render chart without stats |
| Scheduled pre-tip | Same unavailable copy unless ESPN exposes pregame predictor data |

## File layout

```text
frontend/src/pages/GameDetailPage.tsx
frontend/src/components/game/
  WinProbabilityPanel.tsx
  WinProbabilityPanel.test.tsx
  types.ts
  mapGameDetail.ts
frontend/src/lib/api.ts

backend/app/schemas/wnba_game_detail.py
backend/app/services/wnba_game_detail.py
backend/tests/test_wnba_game_detail_normalize.py
backend/tests/test_wnba_game_detail_route.py
```

## Testing

### Backend

- Fixture normalization populates win-probability timeline points when ESPN provides them.
- Stat-row normalization maps the expected labels and values.
- Route returns the new `win_probability` shape without changing existing fields.
- Missing predictor/stat blocks serialize as `null` or partial sections as designed.

### Frontend

- Mapper converts API `win_probability` into camelCase game-detail types.
- `WinProbabilityPanel` renders latest point by default.
- Hover/focus updates the displayed score snapshot and percentages.
- Fallback states render when chart or stats are missing.
- `GameDetailPage` places the panel beneath the two existing sections.

## Out of scope

- Adding box-score tables or deeper team/player stats beyond the six comparison rows
- Creating a separate predictor endpoint
- Supporting NBA game-detail win probability in this change
- Reworking the existing shot-chart or play-by-play layouts beyond making room for the new panel

## Success criteria

- A WNBA game detail page can render a shared win-probability panel beneath `Shot chart` and `Play-by-play`.
- Hovering or focusing the chart reveals the score state and both teams' ESPN win percentages for that moment.
- Team stats render beneath the chart in a clear away-vs-home comparison.
- Games without predictor data degrade gracefully without breaking the rest of the page.
