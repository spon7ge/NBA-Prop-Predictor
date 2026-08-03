# HoopVista website + API system design

Date: 2026-08-02  
Status: Current-state reference (onboarding)  
Scope: React website (`frontend/`) + FastAPI read path (`backend/`)  
Audience: Contributors and future-you navigating page ↔ API wiring

## Goal

Document how the live HoopVista website is structured today: routes, shared chrome, data-fetching hooks, FastAPI endpoints, and upstream sources — so a contributor can answer “what powers this page?” without reverse-engineering the tree.

## Non-goals

- Airflow DAGs, dbt medallion transforms, model training, or scraper internals (see root `README.md`)
- Full catalog of unused DB-backed dashboard routes beyond noting they exist
- Deployment / hosting runbooks
- MLB Statcast, props, or About-page MLB content (not in current slice)

---

## 1. Overview & boundaries

HoopVista’s public site is a React + Vite app talking to a FastAPI service over `/api`. The product surface today is primarily **WNBA** (home, league hubs, game detail, prop picks, player pages). **MLB** adds live scoreboard on the home chrome plus a live matchups hub (dated slate + Sharp odds); game detail remains a stub. NBA is scaffolded (`/nba/matchups` placeholder).

### Read-path model

```text
Browser
  → React (TanStack Query hooks → lib/api.ts)
  → FastAPI (/api/wnba/*, /api/mlb/scoreboard/today for the live site)
  → Upstream adapters
      (ESPN, stats.wnba.com, MLB Stats API, ParlayAPI, Supabase odds snapshots, RotoWire)
```

Two backend families exist. The **live website mainly uses the WNBA and MLB upstream routes**, not the older Postgres research-dashboard routes.

| Family | Examples | Used by current React pages? |
|--------|----------|------------------------------|
| WNBA upstream | `/api/wnba/scoreboard/*`, `props/today`, `games/{id}`, … | Yes |
| MLB upstream | `/api/mlb/scoreboard/*`, `/api/mlb/odds/today` | Yes (home + matchups) |
| DB-backed (silver / gold / ml) | `/api/live-props`, `/api/predictions`, `/api/games/{date}/slate`, … | No |

Shared chrome (`HomeChromeLayout`) wraps most routes with nav, live ticker, and footer. `/about` is static (no API).

---

## 2. Frontend shell & route map

### Stack

| Layer | Choice |
|-------|--------|
| UI runtime | React 19 + TypeScript |
| Bundler | Vite 6 |
| Styling | Tailwind CSS v4, Geist, lucide-react |
| Routing | React Router (`BrowserRouter` + `AppRouter`) |
| Data fetching | TanStack Query |
| API types | OpenAPI → `src/lib/api.schema.d.ts` (`npm run generate:api`) |

**Dev:** Vite proxies `/api` → `http://127.0.0.1:8000`.  
**Static hosts:** set `VITE_API_BASE_URL` to the API origin (no proxy).

### App tree

```text
main.tsx
  QueryClientProvider → BrowserRouter → AppRouter
    HomeChromeLayout (HomeNav + LiveTicker + SiteFooter)
      /                    HomePage
      /about               AboutPage (static)
      /games/:espnEventId  GameDetailPage
      /wnba/matchups       LeagueMatchupsPage (league="wnba")
      /wnba/prop_picks     LeaguePropPicksPage
      /wnba/leaders        LeagueLeadersPage
      /wnba/standings      LeagueStandingsPage
      /wnba/futures        LeagueFuturesPage
      /wnba/player/:id     LeaguePlayerPage
      /nba/matchups        LeagueMatchupsPage (placeholder)
      /mlb/matchups        LeagueMatchupsPage (league="mlb", live slate)
      /mlb/games/:gamePk   MlbGameStubPage (coming soon)
    * → NotFoundPage
```

### Conventions

| Layer | Role |
|-------|------|
| `pages/` | Route composition |
| `components/{home,league,game,about}/` | UI by domain |
| `hooks/useWnba*.ts`, `useMlbScoreboard.ts`, `useGameDetail.ts` | TanStack Query wrappers |
| `lib/api.ts` | Typed `fetch` helpers + `VITE_API_BASE_URL` |

**Chrome data:** Layout and home merge WNBA + MLB scoreboards via `mergeLeagueScoreboards`. Each league has its own query key; failures in one league do not clear the other. While any game is live in either league, the relevant scoreboard refetches about every 18 seconds.

---

## 3. Page ↔ API map

| Route | Page job | Hook(s) | API | Upstream / notes |
|-------|----------|---------|-----|------------------|
| `/` | Brand, LIVE NOW, explainer, league CTAs | `useWnbaScoreboard`, `useMlbScoreboard` | WNBA + MLB scoreboard today | Client merge via `mergeLeagueScoreboards` |
| `/about` | Product / tech / data story | — | none | Static components under `components/about/` |
| Chrome ticker | All chrome routes | same scoreboard queries | WNBA + MLB today | Client merge; isolated per-league error handling |
| `/wnba/matchups?date=` | Daily slate; odds when date is in odds window | `useWnbaScoreboard(date)`, `useWnbaOdds` | scoreboard (`/today` or `?date=`), `GET /api/wnba/odds/today` | Odds via ParlayAPI; client merges onto cards (`mergeMatchupOdds`) |
| `/wnba/prop_picks` | Filterable DFS + US book prop table | `useWnbaProps`, scoreboard | `GET /api/wnba/props/today` | DFS-first board from Supabase PrizePicks/Underdog snapshots + Parlay US books; hide past tip-offs via scoreboard |
| `/wnba/leaders` | Season leaderboards | `useWnbaLeaders` | `GET /api/wnba/leaders` | stats.wnba.com |
| `/wnba/standings` | East / West standings | `useWnbaStandings` | `GET /api/wnba/standings` | ESPN |
| `/wnba/futures` | Championship / award futures | `useWnbaFutures` | `GET /api/wnba/futures` | ESPN core futures API |
| `/wnba/player/:id` | Bio, averages, recent games | `useWnbaPlayer` | `GET /api/wnba/player/{id}` | stats.wnba.com (info + dash + gamelog) |
| `/games/:espnEventId` | Full game center | `useGameDetail` | `GET /api/wnba/games/{id}` | ESPN summary; RotoWire / ESPN roster for scheduled starters |
| `/nba/matchups` | Placeholder | — | none | “NBA matchups coming soon” |
| `/mlb/matchups?date=` | Daily slate; odds when date is in odds window | `useMlbScoreboard(date)`, `useMlbOdds` | scoreboard (`/today` or `?date=`), `GET /api/mlb/odds/today` | Stats API schedule; Sharp MLB run line/total (DK prefer FD); cards → `/mlb/games/:gamePk` |
| `/mlb/games/:gamePk` | Placeholder | — | none | “MLB game detail coming soon” |

### Cross-cutting API behavior

- Most WNBA handlers use short in-process caches; responses that must stay fresh often send `Cache-Control: no-store`.
- MLB scoreboard today always returns `Cache-Control: no-store`; upstream failures surface as HTTP 502 (also no-store).
- OpenAPI is the contract. Backend export → `frontend/openapi.json` → regenerate `api.schema.d.ts`. Verify with `npm run check:api`.
- Prop picks: server builds DFS-anchored rows (`parlay_props` + `dfs_attach` + `odds_snapshots`); the UI filters by book / stat / team / side client-side (`filterPropLines`).

### Prop picks data flow (detail)

```text
LeaguePropPicksPage
  → GET /api/wnba/props/today
  → parlay_props.get_today_props()
       ├─ ParlayAPI player props (US sportsbooks; main lines)
       ├─ odds_snapshots (latest Supabase odds.wnba_prizepicks / odds.wnba_underdogs)
       └─ attach_dfs_snapshots()  # DFS-first rows; match US books to DFS lines
  → client excludePastGameProps(scoreboard) + PropPicksFilters
```

---

## 4. Backend layout, errors, testing

### Backend shape

```text
backend/app/
  main.py           # CORS + router mount (API v0.3.0)
  api/routes/       # thin HTTP handlers (esp. wnba_*, mlb_scoreboard)
  services/         # fetch, merge, cache, map to schemas
  schemas/          # Pydantic response models
  core/             # config; DB helpers for older DB-backed routes
```

Site-facing WNBA and MLB routes are adapters over external APIs and Supabase odds tables. Older `/api/live-*`, `/api/predictions`, and related routes remain mounted for the research dashboard but are **not** wired into the current React pages.

### Errors & empty states

- Hooks distinguish “never loaded” from “refetch failed after a successful load” — keep last good scoreboard / props when possible.
- `mergeLeagueScoreboards` only surfaces an error when **every** league scoreboard has never loaded; one league failing leaves the other visible.
- Pages show loading skeletons, empty copy, or “Unable to load…”; player page maps HTTP 404 to “Player not found”.
- Upstream failures are handled in services (partial payloads and optional `error` fields where schemas allow, e.g. props).

### Testing

| Area | Tooling |
|------|---------|
| Frontend | Vitest + Testing Library (pages, hooks, mappers) |
| Backend | pytest on services and routes |
| Contract | `npm run check:api` after OpenAPI export |

### Maintenance

When you add or change a page:

1. Update the page ↔ API table in this doc.
2. Keep `lib/api.ts` + OpenAPI types in sync.
3. Prefer shared hooks / query keys for scoreboard-like shared data.

Feature-level history lives under `docs/superpowers/specs/` and `docs/superpowers/plans/`. Full platform (ETL / ML) overview remains in the root `README.md`.

---

## 5. Quick reference — endpoints used by the site

| Method | Path | Primary service module |
|--------|------|------------------------|
| GET | `/api/wnba/scoreboard/today` | `wnba_scoreboard` |
| GET | `/api/wnba/scoreboard?date=` | `wnba_scoreboard` |
| GET | `/api/wnba/odds/today` | `parlay_odds` |
| GET | `/api/wnba/props/today` | `parlay_props` (+ `dfs_attach`, `odds_snapshots`) |
| GET | `/api/wnba/leaders` | `wnba_leaders` |
| GET | `/api/wnba/standings` | `wnba_standings` |
| GET | `/api/wnba/futures` | `wnba_futures` |
| GET | `/api/wnba/player/{player_id}` | `wnba_player` |
| GET | `/api/wnba/games/{espn_event_id}` | `wnba_game_detail` |
| GET | `/api/mlb/scoreboard/today` | `mlb_scoreboard` (MLB Stats API) |
| GET | `/api/mlb/scoreboard?date=` | `mlb_scoreboard` |
| GET | `/api/mlb/odds/today` | `mlb_odds` (Sharp `league=mlb`) |

Health (ops, not UI): `GET /api/health`.
