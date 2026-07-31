# WNBA standings page

Date: 2026-07-30  
Status: Approved for planning

## Goal

Ship `/wnba/standings` that matches the boxseats-style standings mockup: Explore subnav with Standings active, season label, and Eastern / Western Conference tables with full columns (`W-L`, `PCT`, `GB`, `Home`, `Away`, `L10`, `Diff`, `Strk`). Numbers come from ESPN via a backend proxy. Brand as HoopVista.

## Decisions

| Topic | Choice |
| --- | --- |
| Scope | WNBA only; no `/nba/standings` in v1 |
| Data source | ESPN WNBA standings API (Approach 1 — backend proxy) |
| Layout | Full conference tables with all mockup columns |
| Responsive | Side-by-side on desktop; stacked East-above-West on mobile |
| Page chrome | No `LeagueHero`; subnav → season label → grid → attribution |
| Team colors | Existing frontend `wnbaTeamColors` map for abbrevs |
| Team logos | ESPN logo URLs from standings payload |
| Subnav | Matchups + Leaders + Standings navigable for WNBA; other Explore/Learn items stay disabled |
| Attribution | `Data: ESPN` |
| Architecture | Backend normalizes ESPN → conferences; thin React page |

## Architecture

```
HomeNav WNBA → /wnba/matchups (existing)
LeagueSubnav Standings → /wnba/standings
        │
        ▼
  LeagueStandingsPage (HomeChromeLayout — nav + ticker stay)
        │
        ├── LeagueSubnav (Standings active)
        ├── "{season} regular season"
        ├── StandingsGrid (East | West conference cards)
        └── "Data: ESPN"
        │
        ▼
  useWnbaStandings() → GET /api/wnba/standings
        │
        ▼
  ESPN site.api v2 WNBA standings
        → normalize East/West rows → cache
```

### Routing

| Path | Element |
| --- | --- |
| `/wnba/standings` | `LeagueStandingsPage` under `HomeChromeLayout` |
| `/nba/standings` | Not registered (404); NBA subnav Standings stays disabled |

- `LeagueSubnav` uses React Router `Link` for **Matchups** (`/:league/matchups`), and when `league === "wnba"`: **Leaders** (`/wnba/leaders`) and **Standings** (`/wnba/standings`).
- Active pill: Matchups when pathname ends with `/matchups`; Leaders with `/leaders`; Standings with `/standings`.
- Other items remain disabled / non-navigating.
- HomeNav WNBA active pill continues to cover any pathname starting with `/wnba`.

## Page structure

Vertical stack under nav + ticker (no league hero):

1. **LeagueSubnav** — Explore + Learn; Matchups, Leaders, and Standings enabled for WNBA.
2. **Season label** — muted text: `{season} regular season` from API `season` field.
3. **StandingsGrid** — responsive grid (`1` col mobile → `2` cols `lg+`) of conference cards:
   - Eastern Conference
   - Western Conference
4. **Attribution** — `Data: ESPN` under the grid.

### Conference card

- Charcoal card (`~#141414`), rounded, padded.
- Title (conference label) top-left.
- Table columns: `#` | `TEAM` | `W-L` | `PCT` | `GB` | `Home` | `Away` | `L10` | `Diff` | `Strk`
- Each row: rank/seed, logo + colored abbrev + full team name, record strings, win pct, games behind, home/away/L10 records, point differential, streak.
- Wide tables may scroll horizontally inside the card; the conference grid stacks on mobile and sits side-by-side on desktop.
- No row borders; muted header labels.
- Unknown team abbrev → muted/white fallback color.
- `Diff`: green when positive (or `+`), red when negative; neutral/muted for zero/`-` if needed.
- `Strk`: green when starts with `W`, red when starts with `L`.

### Visual system

- Dark HoopVista landing: black page, charcoal cards, white/gray text.
- Existing Geist Sans + Geist Mono (no new fonts).
- WNBA accent violet for active Standings pill.
- No new motion required in v1.

### States

| State | Behavior |
| --- | --- |
| Loading | Skeleton conference cards in the same grid |
| Error (never loaded) | Muted “Standings unavailable” copy; do not wipe a prior successful payload |
| Empty conference | Still render the card with a short “No data” row |
| Refetch | No fast polling; React Query default stale time is fine; backend cache is the primary freshness control |

## Data & API

### Endpoint

`GET /api/wnba/standings` → `WnbaStandingsResponse`

### Upstream

- Host: `site.api.espn.com`
- Endpoint: `/apis/v2/sports/basketball/wnba/standings`
- Season: use ESPN payload `season` / `season.year` (or equivalent) when present; otherwise current calendar year in America/New_York

### Response model

```
WnbaStandingsResponse
  season: int
  conferences: list[WnbaStandingsConference]

WnbaStandingsConference
  key: "east" | "west"
  label: str          # "Eastern Conference", "Western Conference"
  teams: list[WnbaStandingsRow]

WnbaStandingsRow
  rank: int
  team_id: str
  abbrev: str
  name: str           # displayName, e.g. "Indiana Fever"
  logo_url: str | null
  wins: int
  losses: int
  wl: str             # "18-10"
  pct: str            # ".643"
  gb: str             # "-" or "1.5"
  home: str           # "11-5"
  away: str           # "7-5"
  l10: str            # "8-2"
  diff: str           # "+169" or "+6.0" (prefer pointDifferential display)
  streak: str         # "W4" / "L2"
```

### Mapping rules

- Walk ESPN `children` conferences; map by name/abbreviation to `east` / `west` (Eastern / `E`, Western / `W`).
- Preserve ESPN entry order within each conference (already seed-ordered).
- For each entry, read `team` + `stats[]` by `name` / `abbreviation` / display label:
  - `playoffSeed` → `rank`
  - `wins` / `losses` / `overall` → `wins`, `losses`, `wl`
  - `winPercent` → `pct`
  - `gamesBehind` → `gb`
  - `Home` → `home`
  - `Road` → `away`
  - `Last Ten Games` → `l10`
  - Prefer `pointDifferential` display for `diff`; fall back to `differential`
  - `streak` → `streak`
- Logo: first `team.logos[].href` when present; else `null`.
- Abbrev: uppercase ESPN team abbreviation; name: `displayName`.
- Skip malformed entries (missing team id/abbrev) rather than failing the whole conference.
- Conference order in the response: East then West.

### Cache

- In-process cache similar to leaders/scoreboard: store response + `expires_at`.
- TTL: **10 minutes** (standings change slowly between games).
- On upstream failure: return stale cache if present; otherwise `502` with `Cache-Control: no-store`.
- Prefer **`no-store` on the HTTP response** and rely on the in-process TTL so clients always revalidate through our proxy.

### Frontend fetch

- `fetchWnbaStandings()` in `lib/api.ts`
- `useWnbaStandings()` React Query hook (`queryKey: ["wnba", "standings"]`)
- Map API snake_case → UI types in a small mapper if needed (or consume snake_case consistently with other API types)

### Team colors

Reuse `frontend/src/components/league/wnbaTeamColors.ts` (no new color fetch). Logos come from ESPN URLs in the standings payload.

## File layout

```
backend/app/schemas/wnba_standings.py
backend/app/services/wnba_standings.py
backend/app/api/routes/wnba_standings.py
backend/app/main.py                              # include router; mention in description
backend/tests/test_wnba_standings_normalize.py
backend/tests/test_wnba_standings_route.py
backend/tests/fixtures/espn_wnba_standings.json

frontend/src/pages/LeagueStandingsPage.tsx
frontend/src/pages/LeagueStandingsPage.test.tsx
frontend/src/components/league/StandingsGrid.tsx
frontend/src/components/league/StandingsGrid.test.tsx
frontend/src/components/league/StandingsConferenceCard.tsx
frontend/src/hooks/useWnbaStandings.ts
frontend/src/lib/api.ts                          # types + fetchWnbaStandings
frontend/src/AppRouter.tsx                       # /wnba/standings
frontend/src/components/league/LeagueSubnav.tsx   # + Standings link
frontend/src/components/league/LeagueSubnav.test.tsx
```

## Testing

### Backend

- Fixture → normalize produces East then West, expected team counts, mapped columns, ranks from seeds.
- Missing/partial entries do not break the other conference.
- Route: happy path `200`; cold upstream failure → `502` + `no-store`.
- Cache: second call within TTL does not re-hit upstream (unit-level or mocked).

### Frontend

- Router mounts Standings page at `/wnba/standings`.
- Subnav: Matchups, Leaders, and Standings are links for WNBA; active state follows route; on NBA league subnav, Standings remains disabled.
- Grid renders season label, two conference titles, sample rows (logo/abbrev/name + columns), colored Diff/Strk, attribution.
- Loading skeletons and never-loaded error copy.
- `npm run build` and relevant Vitest / pytest suites pass.

## Out of scope

- NBA standings page or coming-soon stub
- Playoff race tab / clinch markers
- Team profile deep links
- Season switcher or historical seasons UI
- `LeagueHero` on the standings page
- Trade-deadline or promo banners from the mockup

## Success criteria

- Clicking **Standings** on WNBA subnav opens `/wnba/standings` matching the mockup structure (subnav, season label, East/West tables).
- Tables show current standings from ESPN via `GET /api/wnba/standings` with the full column set.
- Desktop shows conferences side-by-side; mobile stacks East above West.
- Team abbrevs use brand colors from the frontend map; logos use ESPN URLs when available.
- Attribution reads `Data: ESPN`.
- Matchups, Leaders, and Standings all navigate; other subnav items stay disabled.
- NBA hub is unchanged (Standings still disabled; no NBA standings route).
