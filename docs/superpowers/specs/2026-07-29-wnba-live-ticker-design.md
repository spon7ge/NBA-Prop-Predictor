# WNBA live ticker & LIVE NOW scoreboard

Date: 2026-07-29  
Status: Approved for planning

## Goal

Wire real WNBA scoreboard data into the home chrome live ticker and the home LIVE NOW cards. Backend fetches ESPN and stats.wnba.com in parallel, normalizes to one schema, caches briefly by game state, and returns with `Cache-Control: no-store`. Frontend polls with TanStack Query while any game is not Final.

## Decisions

| Topic | Choice |
| --- | --- |
| Approach | Dedicated `GET /api/wnba/scoreboard/today` + shared home query hook |
| UI surfaces | Live ticker + LIVE NOW cards |
| Slate content | Today's full slate (scheduled + live + final) on both surfaces |
| Source conflicts | Per-field merge: prefer the more complete value |
| "Today" | Calendar date in `America/New_York` |
| Browser cache | Always `Cache-Control: no-store` |
| Team logos | Out of scope (letter-circle avatars) |
| NBA endpoint | Out of scope |

## Architecture

```
ESPN scoreboard ──┐
                  ├──(parallel)──→ field-merge → Pydantic → TTL cache → GET /api/wnba/scoreboard/today
stats.wnba.com ───┘                                      Cache-Control: no-store
                                                              │
                              TanStack Query (18s while any non-Final)
                                                              │
                                    ┌─────────────────────────┴─────────────────────────┐
                                    ▼                                                   ▼
                              LiveTicker                                      LiveNowSection
```

## Normalized schema

Response envelope:

```json
{
  "date": "YYYY-MM-DD",
  "games": [/* WnbaGame */],
  "fetched_at": "ISO-8601"
}
```

Per game:

| Field | Type / values | Purpose |
| --- | --- | --- |
| `id` | string | Stable id (prefer stats.wnba.com; else ESPN) |
| `league` | `"wnba"` | League tag for UI pills |
| `status` | `scheduled` \| `live` \| `halftime` \| `final` | Machine status for polling/UI |
| `status_label` | string | Display: `"7:00 PM ET"`, `"Q2 4:12"`, `"Halftime"`, `"Final"` |
| `away` / `home` | `{ abbrev, name, score }` | `score` is `null` until tip |
| `start_time_et` | ISO string | Sort scheduled games |

Frontend maps 1:1 onto existing `TickerGame` and `LiveGame` types (extend if `status` is needed for pulse/polling guards).

## Backend

### Modules

| Path | Role |
| --- | --- |
| `backend/app/schemas/wnba_scoreboard.py` | Pydantic models |
| `backend/app/services/wnba_scoreboard.py` | Fetch, normalize, merge, TTL cache |
| `backend/app/api/routes/wnba_scoreboard.py` | Thin route + response headers |
| `backend/app/main.py` | Register router; note this route calls external APIs |

### Fetch & merge

- Parallel fetch via `asyncio.gather` with short timeouts (~5–8s).
- One source fails: log warning, continue with the other.
- Both fail: return last successful response even if TTL expired (stale-while-error); else `502` if never cached.
- Match games across sources by team abbrevs + tip-time window on the same ET day.
- Per-field merge: non-null score beats null; richer clock/status label beats bare period; prefer stats id when both match.

### Server TTL (by slate state)

| Slate state | TTL |
| --- | --- |
| Any `live` or `halftime` | ~30s |
| Only `scheduled` | ~60s |
| All `final` or empty day | ~60s |

Response always includes `Cache-Control: no-store`.

## Frontend

### Dependencies & wiring

- Add `@tanstack/react-query`; wrap the app in `QueryClientProvider`.
- Hook `useWnbaScoreboard()`:
  - `queryKey: ["wnba", "scoreboard", "today"]`
  - `queryFn` → `GET /api/wnba/scoreboard/today`
  - `refetchInterval: 18_000` while any game has `status !== "final"`
  - Disable interval when all games are Final or the slate is empty
- `HomeChromeLayout` passes mapped games into `LiveTicker`.
- `HomePage` / `LiveNowSection` consume the same query data for cards.

### UI (aligned to boxseats LIVE NOW reference)

- Cards: violet WNBA league pill, red status text, abbrev + full name, amber score boxes.
- Pulse indicator only for `live` / `halftime` (not Scheduled or Final).
- LIVE NOW subtitle counts in-progress games (`live` + `halftime`), e.g. `"2 games in progress"`.
- Loading: keep existing skeleton cards.
- Empty day (loaded, zero games): count `0`, no skeleton cards, ticker muted empty message.
- No team logos in v1 (letter-circle avatars).

## Error handling

| Case | Behavior |
| --- | --- |
| One upstream fails | `200` with games from the healthy source |
| Both fail + prior success | `200` from stale cache (ignore TTL) |
| Both fail + never cached | `502`; FE keeps last successful query data; muted error only if never loaded |
| Timeout / bad JSON | Treat as source failure; route must not crash |

## Testing

**Backend**

- Normalize ESPN fixture → schema
- Normalize stats.wnba.com fixture → schema
- Field-merge conflict cases
- TTL selection by slate state
- Response includes `Cache-Control: no-store`

**Frontend**

- Polling interval disabled when all games are Final
- LIVE NOW renders cards from fixture data
- Loading → skeletons; empty day → 0 + no cards

## Out of scope

- NBA scoreboard endpoint or shared `/api/scoreboard/{league}` abstraction
- Team logos
- WebSockets / SSE
- Filtering LIVE NOW by nav league click (nav still scrolls to `#live-now`)
- Changing About / Stories / Explore content

## Success criteria

- Visiting home shows today's WNBA slate in the ticker and LIVE NOW when games exist.
- Scores/status update while any game is in progress (≈18s poll).
- Polling stops once every game is Final.
- Backend freshness is server TTL only; browsers do not cache the response.
