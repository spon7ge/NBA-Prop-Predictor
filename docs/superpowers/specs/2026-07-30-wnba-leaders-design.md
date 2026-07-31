# WNBA league leaders page

Date: 2026-07-30  
Status: Approved for planning

## Goal

Ship `/wnba/leaders` that matches the boxseats-style leaders mockup: Explore subnav with Leaders active, season/per-game label, and a 3-column grid of top-10 boards for Points, Rebounds, Assists, Steals, Blocks, and 3-Pointers. Numbers come from stats.wnba.com via a backend proxy. Brand as HoopVista.

## Decisions

| Topic | Choice |
| --- | --- |
| Scope | WNBA only; no `/nba/leaders` in v1 |
| Data source | stats.wnba.com league dash player stats (Approach 1 — backend proxy) |
| Categories | Six mockup boards, top 10 each, per-game averages |
| Page chrome | No `LeagueHero`; subnav → season label → grid → attribution |
| Team colors | Hardcoded frontend WNBA abbrev→color map |
| Subnav | Matchups + Leaders navigable; other Explore/Learn items stay disabled |
| Attribution | `Data: stats.wnba.com` (honest to upstream; no ESPN call) |
| Architecture | Backend ranks into categories; thin React page |

## Architecture

```
HomeNav WNBA → /wnba/matchups (existing)
LeagueSubnav Leaders → /wnba/leaders
        │
        ▼
  LeagueLeadersPage (HomeChromeLayout — nav + ticker stay)
        │
        ├── LeagueSubnav (Leaders active)
        ├── "{season} season · per game"
        ├── LeadersGrid (6× LeaderCategoryCard)
        └── "Data: stats.wnba.com"
        │
        ▼
  useWnbaLeaders() → GET /api/wnba/leaders
        │
        ▼
  stats.wnba.com leaguedashplayerstats (PerGame, LeagueID=10)
        → normalize → six top-10 categories → cache
```

### Routing

| Path | Element |
| --- | --- |
| `/wnba/leaders` | `LeagueLeadersPage` under `HomeChromeLayout` |
| `/nba/leaders` | Not registered (404); NBA subnav Leaders stays disabled |

- `LeagueSubnav` uses React Router `Link` for **Matchups** (`/:league/matchups`) and, when `league === "wnba"`, **Leaders** (`/wnba/leaders`).
- Active pill: Matchups when pathname ends with `/matchups`; Leaders when pathname ends with `/leaders`.
- Other items remain disabled / non-navigating.
- HomeNav WNBA active pill continues to cover any pathname starting with `/wnba`.

## Page structure

Vertical stack under nav + ticker (no league hero):

1. **LeagueSubnav** — Explore + Learn; Matchups and Leaders enabled for WNBA.
2. **Season label** — muted text: `{season} season · per game` from API `season` field.
3. **LeadersGrid** — responsive grid (`1` col mobile → `2` md → `3` lg) of category cards in fixed order:
   - Points (PTS)
   - Rebounds (REB)
   - Assists (AST)
   - Steals (STL)
   - Blocks (BLK)
   - 3-Pointers (3PM)
4. **Attribution** — `Data: stats.wnba.com` under the grid.

### Category card

- Charcoal card (`~#141414`), rounded, padded.
- Title (category label) top-left.
- Table columns: `#` | Player | Team | GP | `{STAT}`
- Ten rows: rank, full name, team abbrev (brand color from map), games played, per-game value (one decimal).
- No row borders; muted header labels; bold stat values.
- Unknown team abbrev → muted/white fallback color.

### Visual system

- Dark HoopVista landing: black page, charcoal cards, white/gray text.
- Existing Geist Sans + Geist Mono (no new fonts).
- WNBA accent violet for active Leaders pill.
- No new motion required in v1.

### States

| State | Behavior |
| --- | --- |
| Loading | Skeleton cards in the same grid |
| Error (never loaded) | Muted “Leaders unavailable” copy; do not wipe a prior successful payload |
| Empty category | Still render the card with a short “No data” row |
| Refetch | No fast polling; React Query default stale time is fine; backend cache is the primary freshness control |

## Data & API

### Endpoint

`GET /api/wnba/leaders` → `WnbaLeadersResponse`

### Upstream

- Host: `stats.wnba.com`
- Endpoint: league dash player stats (`/stats/leaguedashplayerstats`)
- Params (intent): `LeagueID=10`, `SeasonType=Regular Season`, `PerMode=PerGame`, base measure type
- Season: pass `Season={YYYY}` where `YYYY` is the current calendar year in America/New_York (WNBA seasons are labeled by calendar year)

### Response model

```
WnbaLeadersResponse
  season: int
  pace: "per_game"
  categories: list[WnbaLeaderCategory]

WnbaLeaderCategory
  key: "points" | "rebounds" | "assists" | "steals" | "blocks" | "three_pointers"
  label: str          # "Points", …
  stat: str           # "PTS", "REB", "AST", "STL", "BLK", "3PM"
  leaders: list[WnbaLeaderRow]  # length ≤ 10

WnbaLeaderRow
  rank: int
  player_id: str
  name: str
  team_abbrev: str
  gp: int
  value: str          # formatted per-game average, e.g. "26.2"
```

### Mapping rules

- One upstream fetch → sort players by each category’s field descending → take top 10 → assign `rank` 1..n.
- Stat fields map from stats.wnba.com columns (PTS, REB, AST, STL, BLK, FG3M or equivalent three-pointers made per game).
- Team abbrev: use the stats tricode as returned (already canonical in scoreboard merge).
- `value` formatted to one decimal place as a string for stable UI.
- Skip / ignore players with missing required fields for a category rather than crashing the board.

### Cache

- In-process cache similar to scoreboard: store response + `expires_at`.
- TTL: **10 minutes** (season averages change slowly; no live clock).
- On upstream failure: return stale cache if present; otherwise `502` with `Cache-Control: no-store`.
- Successful responses may use a short public/private cache header or `no-store` consistent with other WNBA live routes — prefer **`no-store` on the HTTP response** and rely on the in-process TTL so clients always revalidate through our proxy.

### Frontend fetch

- `fetchWnbaLeaders()` in `lib/api.ts`
- `useWnbaLeaders()` React Query hook (`queryKey: ["wnba", "leaders"]`)
- Map API snake_case → UI types in a small mapper if needed (or consume snake_case consistently with other API types)

### Team colors

Hardcoded map in `frontend/src/components/league/wnbaTeamColors.ts`:

- Keys: stats/canonical tricodes (e.g. `LV`, `IND`, `NY`, `DAL`, `PHX`, `ATL`, `TOR`, …)
- Values: hex brand colors suitable on dark backgrounds
- `teamColor(abbrev): string` returns mapped color or a muted fallback (`text-white/50` equivalent hex)

No ESPN color fetch in v1.

## File layout

```
backend/app/schemas/wnba_leaders.py
backend/app/services/wnba_leaders.py
backend/app/api/routes/wnba_leaders.py
backend/app/main.py                              # include router; mention in description
backend/tests/test_wnba_leaders_normalize.py
backend/tests/test_wnba_leaders_route.py
backend/tests/fixtures/stats_wnba_leaguedashplayerstats.json

frontend/src/pages/LeagueLeadersPage.tsx
frontend/src/pages/LeagueLeadersPage.test.tsx
frontend/src/components/league/LeadersGrid.tsx
frontend/src/components/league/LeadersGrid.test.tsx
frontend/src/components/league/LeaderCategoryCard.tsx
frontend/src/components/league/wnbaTeamColors.ts
frontend/src/hooks/useWnbaLeaders.ts
frontend/src/lib/api.ts                          # types + fetchWnbaLeaders
frontend/src/AppRouter.tsx                       # /wnba/leaders
frontend/src/components/league/LeagueSubnav.tsx   # Matchups + Leaders links
frontend/src/components/league/LeagueSubnav.test.tsx
```

## Testing

### Backend

- Fixture → normalize produces six categories in order, ≤10 rows each, ranks 1..n, formatted values.
- Missing/partial player rows do not break other categories.
- Route: happy path `200`; cold upstream failure → `502` + `no-store`.
- Cache: second call within TTL does not re-hit upstream (unit-level or mocked).

### Frontend

- Router mounts Leaders page at `/wnba/leaders`.
- Subnav: Matchups and Leaders are links; active state follows route; other items disabled; on NBA league subnav, Leaders remains disabled.
- Grid renders season label, six titles, sample rows, colored abbrevs, attribution.
- Loading skeletons and never-loaded error copy.
- `npm run build` and relevant Vitest / pytest suites pass.

## Out of scope

- NBA leaders page or coming-soon stub
- ESPN merge for leaders (logos, colors from ESPN)
- Totals vs per-game toggle; additional categories (FG%, MIN, etc.)
- Player profile / game log deep links
- Minimum GP qualifying filter UI (use upstream defaults / raw ranks)
- `LeagueHero` on the leaders page
- Date or season switcher

## Success criteria

- Clicking **Leaders** on WNBA subnav opens `/wnba/leaders` matching the mockup structure (subnav, season label, six top-10 cards).
- Boards show live per-game leaders from stats.wnba.com via `GET /api/wnba/leaders`.
- Team abbrevs use brand colors from the frontend map.
- Attribution reads `Data: stats.wnba.com`.
- Matchups and Leaders both navigate; other subnav items stay disabled.
- NBA hub is unchanged (Leaders still disabled; no NBA leaders route).
