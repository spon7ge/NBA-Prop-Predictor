# MLB home live slice (nav, ticker, Live Now)

Date: 2026-08-02  
Status: Approved for planning  
Scope: First MLB product slice — home chrome + stubs; full MLB league surfaces later  
Audience: Implementers of `/api/mlb/scoreboard` and home merge

## Goal

Ship Major League Baseball onto the HoopVista home surface: nav entry with logo, live ticker, and Live Now cards, fed by the MLB Stats API. Provide coming-soon stubs for `/mlb/matchups` and `/mlb/games/:gamePk`. This is slice 1 of a larger MLB product (same eventual shape as WNBA); Statcast, props, standings, and real game centers are out of scope here.

## Decisions

| Topic | Choice |
| --- | --- |
| Architecture | Parallel league scoreboard (Approach 1): `/api/mlb/scoreboard/today` + `useMlbScoreboard`; merge with WNBA on the client |
| Upstream | MLB Stats API only (`statsapi.mlb.com` schedule) |
| Nav click | `/mlb/matchups` coming-soon stub (same pattern as NBA) |
| Game click | `/mlb/games/:gamePk` coming-soon stub |
| Nav logo | Public CDN URL (not a local asset yet): ESPN MLB mark `https://a.espncdn.com/i/teamlogos/leagues/500/mlb.png` |
| NBA live | Unchanged (still no NBA scoreboard) |
| About / brand | No MLB About sources or HoopVista rename in this slice |

## Non-goals

- Real MLB matchups slate, odds, prop picks, leaders, standings, futures
- Baseball Savant / Statcast
- ESPN MLB scoreboard merge
- About page MLB data-source rows
- Unified multi-league backend endpoint
- Real MLB game detail (box score, play-by-play)

---

## 1. Architecture

```text
Browser
  → HomeChromeLayout / HomePage
       useWnbaScoreboard + useMlbScoreboard
       merge tickerGames / liveGames
  → LiveTicker + LiveNowSection

Nav MLB → /mlb/matchups (coming soon)
MLB card → /mlb/games/:gamePk (coming soon)

GET /api/mlb/scoreboard/today
  → mlb_scoreboard service
  → statsapi.mlb.com /api/v1/schedule (sportId=1, ET today, hydrate team+linescore)
```

Keep WNBA paths untouched. Do not refactor into a single aggregated scoreboard API in this slice.

---

## 2. Backend API & mapping

### Endpoint

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/api/mlb/scoreboard/today` | Primary home feed |

Optional later (not required for home): dated `?date=` — skip unless needed for stubs.

### Upstream

- Base: `https://statsapi.mlb.com/api/v1/schedule`
- Params: `sportId=1`, `date={YYYY-MM-DD}` in America/New_York, `hydrate=team,linescore` (or equivalent fields needed for scores, inning state, logos, venue).

### Response (Pydantic; parallel to WNBA)

| Field | Type | Notes |
| --- | --- | --- |
| `date` | string | `YYYY-MM-DD` ET |
| `games` | list | Mapped games |
| `fetched_at` | string | ISO timestamp |

### Game model

| Field | Mapping |
| --- | --- |
| `id` | `mlb-{gamePk}` |
| `mlb_game_pk` | Stats API `gamePk` as a **string** (parity with `espn_event_id`) |
| `league` | `"mlb"` |
| `status` | `Preview` → `scheduled`; `Live` → `live`; `Final` → `final`. No baseball `halftime`; mid-inning stays `live`. Postponed / cancelled / suspended → `scheduled` with `status_label` from `detailedState` (they appear in the “Today” ticker, never in Live Now) |
| `status_label` | Tip/start time ET, linescore inning (`Top 5th` / `Bot 3rd`), or `Final` / `detailedState` |
| `away` / `home` | `abbrev`, `name`, `score`, `record`, `logo_url` (MLB team CDN) |
| `start_time_et` | From `gameDate` |
| `venue` / `venue_city` | When present |

Reuse the shared status literal `"scheduled" | "live" | "halftime" | "final"` so existing `isInProgressStatus` works; MLB simply never emits `halftime`.

### Caching & freshness

- Short in-process cache (same order of magnitude as WNBA scoreboard).
- Today endpoint responses that must stay fresh: `Cache-Control: no-store`.
- Frontend: refetch ~18s while any merged game is in progress (WNBA or MLB).

### Errors

- Hard upstream / parse failure with no usable cache → **502** + `Cache-Control: no-store` (same as WNBA scoreboard today).
- Isolation: client merge keeps WNBA games if the MLB request fails (and vice versa).

---

## 3. Frontend

### Types

- Extend `HomeLeague` (and `LeagueSlug`) with `"mlb"`.
- Ticker / LiveGame / MatchupGame: keep `espnEventId` for basketball; add optional `mlbGamePk`.
- Link helper: WNBA → `/games/{espnEventId}`; MLB → `/mlb/games/{mlbGamePk}`.

### Nav

- `HomeNav`: third league entry — MLB label + ESPN CDN logo (`…/leagues/500/mlb.png`).
- `to="/mlb/matchups"`; `aria-current` when `pathname.startsWith("/mlb")`.
- Accent: distinct from NBA sky / WNBA violet (e.g. red-tinted active styles when needed); coming-soon body can stay muted like NBA.

### Home merge

- `HomeChromeLayout` and `HomePage` use both hooks.
- Merge `tickerGames` and `liveGames` from both leagues.
- **Partial success:** if one league has loaded and the other failed, keep the successful league’s games.
- **`isError` / hasNeverLoaded for ticker:** true only when *neither* league has ever loaded successfully.
- **Loading:** show Live Now skeletons until the first successful payload from either league; prefer rendering partial results over waiting for both.
- Live Now cards already show a league badge; MLB cards link to the stub game route.

### Routes

| Path | Element |
| --- | --- |
| `/mlb/matchups` | `LeagueMatchupsPage` with `league="mlb"` — hero + subnav + “MLB matchups coming soon” |
| `/mlb/games/:gamePk` | Stub under `HomeChromeLayout`: “MLB game detail coming soon” |
| `/games/:espnEventId` | Unchanged (WNBA) |

### League shell

- Extend `LeagueHero` for MLB: label `MLB`, title like `Major League Baseball`, short blurb, CDN (or hero) logo.
- Extend `LeagueSubnav` to accept `mlb` (Matchups active; other items disabled as today for NBA).
- `LeagueMatchupsPage`: branch for `mlb` like `nba` (coming-soon body, no scoreboard panel yet).

### OpenAPI

- Export backend schema → `frontend/openapi.json` → regenerate `api.schema.d.ts`.
- Add `fetchMlbScoreboard` (or equivalent) in `lib/api.ts`.
- Verify with `npm run check:api`.

---

## 4. Testing

| Area | Coverage |
| --- | --- |
| Backend | Normalize fixture schedule JSON → statuses, scores, `mlb_game_pk`, logos, venue; route smoke test |
| Frontend | Nav MLB link + active state; `/mlb/matchups` and `/mlb/games/:id` stubs; merge WNBA+MLB into ticker/Live Now; link targets |
| Contract | OpenAPI export + `check:api` |

---

## 5. File layout (expected)

```text
backend/app/
  api/routes/mlb_scoreboard.py
  services/mlb_scoreboard.py
  schemas/mlb_scoreboard.py
backend/tests/ (normalize + route)

frontend/src/
  hooks/useMlbScoreboard.ts
  pages/MlbGameStubPage.tsx   # or equivalent thin stub
  components/home/HomeNav.tsx          # +mlb
  components/home/types.ts             # HomeLeague
  components/home/LiveTicker.tsx       # mlb link
  components/home/LiveNowSection.tsx   # mlb link if needed
  components/league/LeagueHero.tsx     # +mlb
  components/league/LeagueSubnav.tsx   # +mlb
  pages/LeagueMatchupsPage.tsx         # +mlb branch
  layouts/HomeChromeLayout.tsx         # merge hooks
  pages/HomePage.tsx                   # merge live games
  AppRouter.tsx                        # routes
  lib/api.ts + openapi.json
```

Exact filenames may follow existing WNBA naming conventions.

---

## 6. Success criteria

- Home nav shows MLB with logo and navigates to `/mlb/matchups` coming soon.
- When MLB Stats API returns live (or today’s) games, they appear in the ticker and Live Now alongside WNBA.
- Clicking an MLB game opens `/mlb/games/:gamePk` stub; WNBA game links unchanged.
- MLB API outage does not clear a successful WNBA ticker/Live Now.
- Backend tests cover mapping; frontend tests cover nav, stubs, and merge; OpenAPI types stay in sync.

## Maintenance

After implementation, update `docs/superpowers/specs/2026-08-02-website-api-system-design.md` page ↔ API table to include MLB scoreboard + stub routes.
