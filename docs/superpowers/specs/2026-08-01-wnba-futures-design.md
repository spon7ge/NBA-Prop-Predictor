# WNBA Futures (replace Clutch)

Date: 2026-08-01  
Status: Approved for planning

## Goal

Replace the disabled Explore **Clutch** pill with **Futures**, linking to `/wnba/futures`. Show live championship (Finals Winner) odds from ESPN’s futures feed — same board ESPN surfaces at [espn.com/wnba/futures](https://www.espn.com/wnba/futures) — in HoopVista’s quiet league-hub chrome.

## Decisions

| Topic | Choice |
| --- | --- |
| Nav | Rename **Clutch** → **Futures**; WNBA-only link to `/wnba/futures` |
| NBA | Futures remains disabled (no `/nba/futures` in v1) |
| Data source | ESPN core API (not HTML scrape, not Sharp) |
| Endpoint upstream | `GET https://sports.core.api.espn.com/v2/sports/basketball/leagues/wnba/seasons/{season}/futures` |
| Season | Current WNBA season year (e.g. `2026`); no season picker |
| Markets | All markets returned by ESPN for that season (today: `WNBA - Winner`) |
| Display name | Map `WNBA - Winner` → **Finals Winner**; otherwise use ESPN `name` |
| Provider | Prefer active provider named like ESPN BET; else first active / first listed |
| Sort | Shortest American odds first (favorites at top) |
| UI shell | Same as standings: `LeagueSubnav` + content (no `LeagueHero` date strip required) |
| Cache | ~5 minutes in-process |

## Architecture

```
LeagueSubnav "Futures"
        │
        ▼
/wnba/futures  →  LeagueFuturesPage
        │
        ▼
GET /api/wnba/futures
        │
        ▼
wnba_futures service
  fetch season futures index
  for each market → pick provider → books[]
  resolve team $ref → abbrev, name, logo_url
  normalize + sort entries
```

## Backend

### Route

`GET /api/wnba/futures`

Response shape:

```json
{
  "season": 2026,
  "as_of": "2026-08-02T00:00:00Z",
  "markets": [
    {
      "id": "8146",
      "name": "WNBA - Winner",
      "display_name": "Finals Winner",
      "provider": "ESPN BET",
      "entries": [
        {
          "team_id": "8",
          "abbrev": "NYL",
          "name": "New York Liberty",
          "logo_url": "https://...",
          "odds_american": "+250"
        }
      ]
    }
  ],
  "error": null
}
```

### Normalize

- Parse `items[]` markets; each has `futures[]` provider blobs with `books[]`.
- Select one provider per market (ESPN BET preferred).
- For each book row, resolve `team.$ref` (or embedded team) to `team_id`, `abbrev`, `name`, `logo_url` via ESPN team resource; cache team lookups.
- Skip rows that cannot resolve to a usable team + odds string.
- Sort entries by American odds ascending (more negative / shorter plus = favorite). Prefer numeric parse of American odds for sort; keep display string from ESPN (`+250`).
- `Cache-Control: no-store` on the HTTP response (match other live WNBA routes); service TTL still ~5 min.

### Season selection

Derive current season as the calendar year in America/New_York unless an existing league helper already defines WNBA season — reuse if present.

## Frontend

### Subnav

- `exploreItems`: replace `"Clutch"` with `"Futures"`.
- `itemPath`: `"Futures"` + `league === "wnba"` → `/wnba/futures`.
- `isActive`: pathname ends with `/futures`.

### Route

- `AppRouter`: `/wnba/futures` → `LeagueFuturesPage`.

### Page

- `LeagueSubnav league="wnba"`.
- Section heading / market title from `display_name`.
- Quiet list or table: `TeamAbbrevAvatar` + name + mono American odds.
- Caption: `Odds by {provider}` (text is fine; no new logo asset required for ESPN BET).
- Loading skeletons; error with no data; empty markets copy.
- If multiple markets: one block per market (title + rows).

### OpenAPI

Export `frontend/openapi.json` + regenerate `api.schema.d.ts` for the new path.

## Error handling

| Case | Behavior |
| --- | --- |
| ESPN down, cache hit | Return cached payload; optional `error` string |
| ESPN down, no cache | `502` / UI “Unable to load futures” |
| Empty `markets` | 200 with `markets: []`; empty UI message |
| Team `$ref` fails | Omit that entry; continue |
| NBA Futures click | Still disabled |

## Testing

### Backend

- Normalize fixture: Finals Winner entries sorted; provider selected; display_name mapping.
- Route 200 with mocked fetch.
- Upstream failure with cache → stale; without cache → 502 or empty+error per existing live-route pattern.

### Frontend

- Subnav shows Futures; Clutch gone; link to `/wnba/futures` for WNBA; disabled on NBA.
- Page renders market title, team rows, odds, provider caption.
- Loading / error / empty states.
- Router includes `/wnba/futures`.

## Out of scope

- NBA futures page
- Season picker / historical seasons
- Sharp or multi-book comparison
- Playoff race pill changes
- Scraping espn.com HTML
- Betting CTAs / deep links into sportsbooks

## Success criteria

- Explore shows **Futures** instead of Clutch; WNBA navigates to `/wnba/futures`.
- Page lists Finals Winner (or whatever markets ESPN returns) with team + American odds from ESPN’s feed.
- Quiet HoopVista styling; failures degrade gracefully.
