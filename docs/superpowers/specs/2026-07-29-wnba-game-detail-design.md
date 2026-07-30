# WNBA game detail page (ESPN summary)

Date: 2026-07-29  
Status: Approved for planning

## Goal

When a user clicks a game in LIVE NOW or the live ticker, navigate to a dedicated game detail page that matches the provided boxseats-style mockup: header/scoreboard, shot chart with team filters, and play-by-play with period filter. Data comes from ESPN via a backend summary proxy. Ship WNBA now; keep types/routes shaped so NBA can plug in later.

## Decisions

| Topic | Choice |
| --- | --- |
| Navigation | Dedicated page `/games/:espnEventId` under `HomeChromeLayout` |
| Entry points | LIVE NOW cards + ticker chips |
| v1 content | Full mockup: header + shot chart + play-by-play |
| League scope | WNBA live; league-agnostic shapes for NBA later |
| Game statuses | All (scheduled, live/halftime, final) |
| Data path | Backend ESPN summary proxy (Approach 1) |
| Scoreboard ids | Keep existing `id`; add `espn_event_id` for detail links |
| Branding | HoopVista chrome (nav + ticker); game UI follows mockup structure |

## Architecture

```
LIVE NOW / ticker click
        │
        ▼
  /games/:espnEventId     (HomeChromeLayout — nav + ticker stay)
        │
        ▼
  useGameDetail(id)  →  GET /api/wnba/games/{espnEventId}
        │
        ▼
  ESPN …/basketball/wnba/summary?event=… → normalize → header + shots + plays
        │
        ▼
  GameDetailPage: Back | header card | Shot chart | Play-by-play
```

### Routing

| Path | Element |
| --- | --- |
| `/games/:espnEventId` | `GameDetailPage` (nested under `HomeChromeLayout`) |
| Back control | Always navigate to `/` |

Param is the raw ESPN event id (e.g. `401749001`), not the scoreboard’s stats-preferred `id`.

### Scoreboard link fix

Today’s merge prefers stats.wnba.com ids, which cannot call ESPN summary. Extend each scoreboard game with:

| Field | Type | Notes |
| --- | --- | --- |
| `espn_event_id` | `string \| null` | Raw ESPN event id whenever ESPN contributed to the merge |

- LIVE NOW and ticker link to `/games/{espn_event_id}` only when present.
- Keep existing `id` for React keys and scoreboard identity.
- Frontend `LiveGame` / `TickerGame` gain optional `espnEventId`.

### Polling & cache

- Detail query refetches every ~15–20s while status is `live` or `halftime`; stop for `scheduled` / `final`.
- API responds with `Cache-Control: no-store`.
- Short TTL cache + stale-while-error (same pattern as scoreboard): serve last good payload on ESPN failure after a prior success; `502` if never cached; `404` for unknown/unavailable event.

## Data schema

### Detail response

`GET /api/wnba/games/{espnEventId}`

```json
{
  "espn_event_id": "401749001",
  "league": "wnba",
  "status": "live",
  "status_label": "4:13 - 1st",
  "venue": "Mortgage Matchup Center",
  "away": {
    "id": "…",
    "abbrev": "GS",
    "name": "Golden State Valkyries",
    "score": 10,
    "color": "#5B2C6F"
  },
  "home": {
    "id": "…",
    "abbrev": "PHX",
    "name": "Phoenix Mercury",
    "score": 9,
    "color": "#E56020"
  },
  "fg_made": 6,
  "fg_attempted": 16,
  "latest_play": {
    "id": "…",
    "clock": "4:29",
    "period": 1,
    "text": "Laeticia Amihere makes two point shot",
    "team_id": "…"
  },
  "shots": [],
  "plays": [],
  "fetched_at": "ISO-8601"
}
```

`latest_play`, `venue`, and team `score` values may be `null` when unavailable (e.g. pre-tip).

### Nested types

**Team:** `id`, `abbrev`, `name`, `score` (`null` before tip), `color` (hex; ESPN `color` / `alternateColor`, else fixed away/home fallback accents).

**Shot:** `id`, `team_id`, `player_name`, `made`, `x`, `y`, `period`, `clock`.

**Play:** `id`, `team_id`, `period`, `clock`, `text`, `scoring`, `away_score`, `home_score`, `shooting`.

**Latest play:** subset used by the shot-chart action bar; may be `null` when no plays yet.

### ESPN source mapping

Endpoint: `https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/summary?event={espnEventId}`

| UI need | ESPN source |
| --- | --- |
| Header (teams, scores, clock, venue) | `header` / competitions / status / venue |
| Team colors | competitor team `color` / `alternateColor` |
| Play-by-play | `plays` array |
| Shot chart | shooting plays with coordinates |
| FG made/attempted | Count from shooting plays in the normalized `shots` list (made / total) |

## UI

### Page regions

| Region | Behavior |
| --- | --- |
| Back row | “← Back” left; `status_label` right |
| Game header card | Venue + status with league accent; away/home names in team colors; large amber scores in dark boxes |
| Shot chart (left) | Title + Both / away / home filters; current-action bar from `latest_play`; half-court SVG; filled = made, hollow = missed; legend + `fg_made/fg_attempted FG` + “Data: ESPN” |
| Play-by-play (right) | Title + period pill (default = current or last period with plays); newest-first; top play highlighted; scoring plays get subtle bg + running score; team color dots |

Layout: single column stacked on small screens; two columns (chart \| PBP) from `md` up. Dark theme consistent with home (`#0B0B0B` page, charcoal cards).

### Empty / edge states

| State | UI |
| --- | --- |
| Loading | Skeleton header + two panels |
| Scheduled, no plays | Header with tip/status; muted “Tip-off pending” in chart and PBP |
| Error / never loaded | “Unable to load game” + Back home |
| 404 | Same unavailable copy + Back |
| Refetch error after success | Keep last good payload |

### Entry points

- `LiveGameCard`: whole card is a `Link` when `espnEventId` is set; hover/focus styles.
- Ticker chips: same `Link` when `espnEventId` is set. Marquee pause-on-hover is optional and not required for v1.

## File layout

```
frontend/src/pages/GameDetailPage.tsx
frontend/src/components/game/
  GameHeader.tsx
  ShotChart.tsx
  PlayByPlay.tsx
  types.ts
frontend/src/hooks/useGameDetail.ts
frontend/src/lib/api.ts                    # + fetchGameDetail
frontend/src/AppRouter.tsx                 # + /games/:espnEventId
frontend/src/components/home/LiveNowSection.tsx   # Link
frontend/src/components/home/LiveTicker.tsx       # Link
frontend/src/components/home/types.ts             # espnEventId
frontend/src/components/home/mapScoreboard.ts     # map espn_event_id

backend/app/schemas/wnba_game_detail.py
backend/app/services/wnba_game_detail.py
backend/app/api/routes/wnba_game_detail.py
backend/app/schemas/wnba_scoreboard.py     # + espn_event_id
backend/app/services/wnba_scoreboard.py    # preserve espn_event_id on merge
backend/app/main.py                        # register router
```

## Testing

### Backend

- Normalize fixture from ESPN summary → header, shots, scoring flags, FG counts.
- Route: `200` + `no-store`; `404` bad event; stale-while-error; `502` with empty cache.
- Scoreboard merge preserves `espn_event_id` when stats id wins.

### Frontend

- Hook polls only for live/halftime.
- Components render from fixtures (filters, made/missed, period filter, scoring highlight).
- LIVE NOW + ticker link to `/games/{espnEventId}` when present.
- Router mounts detail under home chrome.

## Out of scope

- NBA summary fetch implementation (types/routes only prepared)
- Settings modal, About changes, team logo images (letter avatars fine)
- Box score tables, win probability, odds
- Frontend calling ESPN directly
- Changing scoreboard primary `id` semantics beyond adding `espn_event_id`

## Success criteria

- Clicking a LIVE NOW card or ticker chip with an ESPN id opens `/games/:espnEventId`.
- Page shows header, shot chart, and play-by-play matching the mockup structure with live ESPN data.
- Scheduled games show header + tip-off pending empty states; final games show full chart and PBP.
- Live games poll and update without full-page remount flicker replacing good data with errors.
- `npm run build` (frontend) and backend game-detail tests pass.
