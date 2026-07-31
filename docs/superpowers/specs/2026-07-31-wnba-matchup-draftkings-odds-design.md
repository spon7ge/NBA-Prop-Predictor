# WNBA matchup DraftKings odds pill

Date: 2026-07-31  
Status: Approved for planning

## Goal

On WNBA matchup cards (`/wnba/matchups`), show a compact DraftKings odds pill with spread and/or total (e.g. `Spread: ATL -12.5 · Total: 178.5`), with an `Odds by DraftKings` caption under the pill. Data comes from Sharp API via a backend proxy; the API key never reaches the browser.

## Decisions

| Topic | Choice |
| --- | --- |
| When to show | Whenever DraftKings has at least one of spread or total; hide pill + caption when both missing |
| Partial lines | Allow one-sided pills (`Spread: …` or `Total: …` alone) |
| Fetch path | Backend proxy using `SHARP_API_KEY` from env |
| Endpoint shape | Separate `GET /api/wnba/odds/today` (do not fold into scoreboard) |
| Sportsbook | DraftKings only (`sportsbook=draftkings`) |
| Markets | Spread + total only (no moneyline / props) |
| Merge | Frontend joins odds onto scoreboard games by home + away abbrev |
| Scope | WNBA `MatchupGameCard` only |
| Cache | ~45s successful Sharp responses in-process; on error serve last good cache if present |
| Frontend poll | ~60s fixed interval |
| Failure mode | Odds empty; scoreboard cards still render |

## Architecture

```
MatchupsPanel (WNBA)
  ├── useWnbaScoreboard()     → ESPN games (existing)
  └── useWnbaOdds()           → GET /api/wnba/odds/today
           │
           ▼
  Backend Sharp client (SHARP_API_KEY)
    GET https://api.sharpapi.io/api/v1/odds
      ?league=wnba&sportsbook=draftkings&market=spread,total
           │
           ▼
  Normalize → games[{ homeAbbrev, awayAbbrev, spreadTeamAbbrev?, spreadLine?, total? }]
  Cache ~45s in-process
           │
           ▼
  Frontend merges onto MatchupGame by home+away abbrev
  MatchupGameCard shows pill + caption when any line exists
```

## UI

Place the odds block **below the two team rows**, left-aligned inside the card (chevron unchanged).

When any line exists:

- Pill: `Spread: ATL -12.5 · Total: 178.5` (omit missing half; join with ` · ` when both present)
- Caption under pill: `Odds by DraftKings`

Rules:

- Spread: `Spread: {abbrev} {line}` using the favorite’s line when available (negative preferred; otherwise the available team + line).
- Total: `Total: {line}` (no Over/Under label).
- No pill and no caption when both halves are missing.
- Styling: muted charcoal/white to match existing cards; do not over-brand with violet beyond the existing card accent.
- Odds are display-only; the whole card remains one link to game detail when `espnEventId` is set.

## API

### `GET /api/wnba/odds/today`

```json
{
  "asOf": "2026-07-31T21:00:00Z",
  "sportsbook": "draftkings",
  "games": [
    {
      "homeAbbrev": "ATL",
      "awayAbbrev": "IND",
      "spreadTeamAbbrev": "ATL",
      "spreadLine": -12.5,
      "total": 178.5
    }
  ]
}
```

- Missing half → `null` for that field.
- Include a game in `games` only if at least one of spread or total is non-null.
- Abbrev mapping: normalize Sharp team names → ESPN-style WNBA abbrevs used by the scoreboard. Unmatched rows are dropped and logged.
- Missing `SHARP_API_KEY` or Sharp error → `200` with `games: []` (optional `error` string for debug). Prefer last good cache when available.
- Update FastAPI app description to list this route among live external-data exceptions.

## File layout

```
backend/app/services/sharp_odds.py          # Sharp fetch + normalize + cache
backend/app/schemas/wnba_odds.py
backend/app/api/routes/wnba_odds.py
backend/app/core/config.py                  # SHARP_API_KEY
backend/app/main.py                         # router + description
backend/tests/…                             # fixtures + route/service tests

frontend/src/lib/api.ts                     # types + fetch helper
frontend/src/hooks/useWnbaOdds.ts           # or colocated with scoreboard hook pattern
frontend/src/components/league/types.ts     # optional odds fields on MatchupGame
frontend/src/components/league/MatchupsPanel.tsx
frontend/src/components/league/MatchupGameCard.tsx
frontend/src/components/league/MatchupGameCard.test.tsx
```

## Testing

### Backend

- Fixture Sharp payload → favorite spread abbrev + line + total normalized.
- Partial markets → only one half set; game still included.
- Missing key / Sharp failure → `200` + empty `games` (stale cache when present).
- Team name → abbrev mapping for common WNBA nicknames.

### Frontend

- Card renders pill + `Odds by DraftKings` when odds present.
- Partial pill text; no pill when odds absent.
- Merge by home/away abbrev; unmatched games unchanged.

## Out of scope

- NBA odds or NBA matchup cards
- Home ticker / LIVE NOW cards
- Moneyline, player props, other sportsbooks
- Deep links to DraftKings
- Persisting odds into Postgres / Airflow

## Success criteria

- WNBA matchup cards show DraftKings spread and/or total when Sharp returns them.
- Caption `Odds by DraftKings` appears under the pill.
- Sharp key stays server-side.
- Scoreboard remains usable if Sharp is down or the key is missing.
- Relevant backend and frontend tests pass.
