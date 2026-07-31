# WNBA Prop Picks — FanDuel & DraftKings lines

Date: 2026-07-31  
Status: Approved for planning

## Goal

Ship `/wnba/prop_picks` as a flat table of today’s WNBA player prop main lines from FanDuel and DraftKings. Show both Over and Under when available. Leave model prediction, over/under %, and EV empty for now. Keep the existing dark WNBA league aesthetic (hairline rows + odds pills).

## Decisions

| Topic | Choice |
| --- | --- |
| Route | `/wnba/prop_picks` |
| Data source | Sharp API via backend proxy (`SHARP_API_KEY`); never expose key to browser |
| Endpoint | New `GET /api/wnba/props/today` (do not fold into matchup odds) |
| Sportsbooks | FanDuel + DraftKings (`sportsbook=draftkings,fanduel`) |
| Markets | All Sharp `player_*` markets (`market=props`) |
| Lines | Main lines only (`is_main_line=true`) |
| Row grain | One row per player + market + side (`over` / `under`) |
| Both sides | Showcase Over and Under when either book provides that side |
| Empty columns | Model prediction, O/U %, EV always blank/null in v1 |
| Layout | Flat table (not grouped by game/player) |
| Visual style | Option B: hairline rows + mono odds pills; empty cells blank |
| Cache | ~45s successful Sharp props responses in-process; on error serve last good cache if present |
| Frontend poll | ~60s fixed interval |
| Nav | Enable Prop Picks link in `LeagueSubnav` for WNBA |

## Architecture

```
LeaguePropPicksPage (/wnba/prop_picks)
  ├── LeagueSubnav (Prop Picks active)
  └── PropPicksTable
        └── useWnbaProps() → GET /api/wnba/props/today
                 │
                 ▼
        Backend Sharp client (SHARP_API_KEY)
          GET https://api.sharpapi.io/api/v1/odds
            ?league=wnba
            &sportsbook=draftkings,fanduel
            &market=props
            &is_main_line=true
                 │
                 ▼
        Normalize → props[{ player, stat, side, fanduel?, draftkings? }]
        Cache ~45s in-process
```

Reuse patterns from `sharp_odds.py` (pagination, TTL cache, graceful empty on missing key / Sharp errors). Implement props in a dedicated `sharp_props.py` so matchup spread/total fetch stays unchanged; share only small helpers if duplication becomes painful.

## UI

Page chrome matches other WNBA explore pages (`LeagueSubnav`; no hero required for v1).

Table columns (left → right):

| Column | Content |
| --- | --- |
| Player | `player_name` |
| Stat | Human label (e.g. Assists from `stat_category` / `market_type`) |
| O/U | `Over` or `Under` — the side whose odds are shown in the book columns |
| Model | Blank |
| O/U % | Blank |
| EV | Blank |
| FanDuel | Odds pill `line odds` for that side, or blank if missing |
| DraftKings | Same |

Rules:

- Flat list, horizontally scrollable on narrow viewports if needed; container `max-w-6xl`.
- Book cells use small rounded pills (same visual language as matchup odds pills), mono text, American odds with proper minus sign formatting.
- If books disagree on line for the same player/market/side, each column shows that book’s own line + odds.
- Include a row when at least one of FanDuel or DraftKings has that side; leave the other book blank.
- Caption under table: `Odds by FanDuel & DraftKings`.
- Loading: skeleton hairline rows. Error / empty Sharp: short muted message (e.g. `Prop lines unavailable`); do not crash the page.

## API

### `GET /api/wnba/props/today`

```json
{
  "as_of": "2026-07-31T22:00:00Z",
  "sportsbooks": ["fanduel", "draftkings"],
  "props": [
    {
      "player_name": "Rhyne Howard",
      "stat": "Assists",
      "market_type": "player_assists",
      "side": "over",
      "model_prediction": null,
      "over_under_pct": null,
      "ev": null,
      "fanduel": { "line": 3.5, "odds_american": -114 },
      "draftkings": { "line": 3.5, "odds_american": -120 }
    },
    {
      "player_name": "Rhyne Howard",
      "stat": "Assists",
      "market_type": "player_assists",
      "side": "under",
      "model_prediction": null,
      "over_under_pct": null,
      "ev": null,
      "fanduel": { "line": 3.5, "odds_american": -114 },
      "draftkings": { "line": 3.5, "odds_american": -110 }
    }
  ],
  "error": null
}
```

Normalization:

- Keep Sharp rows where `is_player_prop` / `market_type` starts with `player_` and `is_main_line` is true.
- Side from `selection_type` in `{over, under}` (drop unrecognized sides).
- Player from `player_name` (fallback to `selection` if needed).
- Stat label: title-case `stat_category` when present; else derive from `market_type` (`player_assists` → Assists).
- Group key: `(player_name, market_type, side)` → attach `fanduel` / `draftkings` book quotes.
- Sort: player name, then market/stat, then over before under.
- Paginate Sharp fetches (limit/offset or cursor) until exhausted or a safe page cap, same spirit as matchup odds.
- Missing `SHARP_API_KEY` or Sharp failure → `200` with `props: []` and optional `error`. Prefer last good cache when available.
- Update FastAPI app description if it lists live external-data exceptions.

## File layout

```
backend/app/services/sharp_props.py       # dedicated Sharp props fetch/normalize/cache
backend/app/schemas/wnba_props.py
backend/app/api/routes/wnba_props.py
backend/app/main.py                       # router registration
backend/tests/fixtures/sharp_wnba_props.json
backend/tests/test_sharp_props.py

frontend/src/lib/api.ts                   # types + fetchWnbaProps
frontend/src/hooks/useWnbaProps.ts
frontend/src/pages/LeaguePropPicksPage.tsx
frontend/src/components/league/PropPicksTable.tsx (+ test)
frontend/src/components/league/LeagueSubnav.tsx  # enable Prop Picks link
frontend/src/AppRouter.tsx
frontend/src/AppRouter.test.tsx / LeagueSubnav.test.tsx
```

## Testing

### Backend

- Fixture with FD + DK over/under for one player → two rows, both books filled.
- One book missing a side → row still present; missing book null.
- Non-main / non-player markets ignored.
- Missing key / Sharp failure → `200` + empty `props` (stale cache when present).

### Frontend

- Table renders player, stat, O/U, blank model/O/U%/EV, and FD/DK pills.
- Both Over and Under rows appear when API returns both.
- Subnav Prop Picks is a link to `/wnba/prop_picks` and marks active.
- Empty / error states render muted copy.

## Out of scope

- Model predictions, O/U hit rates, EV calculations
- Alternate lines
- NBA Prop Picks
- Grouping, filters, sort controls (beyond default sort)
- Deep links into FanDuel / DraftKings
- Persisting props into Postgres / Airflow
- Changing matchup spread/total odds behavior

## Success criteria

- `/wnba/prop_picks` shows today’s WNBA main-line player props as a flat table.
- FanDuel and DraftKings columns show line + American odds for the row’s O/U side.
- Over and Under both appear when Sharp provides them.
- Model / O/U% / EV columns are present but empty.
- Sharp key stays server-side; page stays usable if Sharp is down or the key is missing.
- Prop Picks subnav navigates to the page.
- Relevant backend and frontend tests pass.
