# MLB matchups hub (scoreboard + Sharp odds)

Date: 2026-08-02  
Status: Approved for planning  
Scope: Live `/mlb/matchups` parity with WNBA matchups (date nav, slate panel, DK/FD odds)  
Audience: Implementers of MLB dated scoreboard, odds route, and matchups page

## Goal

Ship a real MLB matchups hub at `/mlb/matchups` matching WNBA: ET date navigation, LIVE NOW / REST OF THE SLATE cards from the MLB Stats API scoreboard, and DraftKings/FanDuel main-line odds from Sharp (`league=mlb`) when the selected date is in the odds window. Game cards link to `/mlb/games/:gamePk` (existing stub).

## Decisions

| Topic | Choice |
| --- | --- |
| Architecture | Parallel MLB stack (Approach 1) — not a generalized multi-league page |
| Scoreboard | Stats API schedule; add dated `GET /api/mlb/scoreboard?date=` |
| Odds | Sharp API (`league=mlb`), DraftKings + FanDuel, prefer DK (same merge as WNBA) |
| Odds window | Today + next 2 ET days (`isOddsWindowDate`) |
| Card links | `gameDetailHref` → `/mlb/games/{mlbGamePk}` |
| Game detail | Stub only (unchanged) |
| NBA matchups | Unchanged coming-soon |

## Non-goals

- Real MLB game detail / box score
- MLB props, standings, leaders, futures
- Parlay / The Odds API for MLB
- Refactoring WNBA matchups into a shared parameterized page
- Overnight live carryover on dated MLB boards (not required for dated route)

---

## 1. Architecture

```text
/mlb/matchups?date=
  LeagueHero + LeagueSubnav
  useMlbScoreboard(selectedDate)
       → GET /api/mlb/scoreboard/today   (selected === today)
       → GET /api/mlb/scoreboard?date=   (otherwise)
  useMlbOdds()
       → GET /api/mlb/odds/today
  mergeMatchupOdds(mapToMatchupGames(games), odds?, selectedDate)
  MatchupsPanel → MatchupGameCard (gameDetailHref)
```

Reuse existing UI: `MatchupsPanel`, `mergeMatchupOdds`, `isOddsWindowDate`, slate date helpers. Do not change WNBA routes or Sharp WNBA fetch behavior except shared helpers extracted carefully if needed.

---

## 2. Backend

### Scoreboard

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/api/mlb/scoreboard/today` | Existing |
| GET | `/api/mlb/scoreboard?date=YYYY-MM-DD` | New; `date` required, ET calendar day |

- Upstream: `statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}&hydrate=team,linescore`
- Reuse `normalize_mlb_schedule` from the home-live slice
- Dated route: short per-date cache (same order as WNBA dated TTL); **no** overnight carryover
- Hard failure → **502** + `Cache-Control: no-store`

### Odds

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/api/mlb/odds/today` | New |

- Upstream: Sharp `https://api.sharpapi.io/api/v1/odds` with `league=mlb`, `sportsbook` in `{draftkings,fanduel}`, `is_main_line=true`, paginated like WNBA
- Markets: request `run_line,total_runs` for MLB. If a live Sharp probe shows different keys for MLB mains, switch the request string to match — still normalize into `spread_team_abbrev`, `spread_line`, `total` for the shared card UI
- Response shape: parallel to WNBA — either reuse `WnbaOddsGame` / `WnbaOddsResponse` renamed in OpenAPI via shared models, **or** add `MlbOddsGame` / `MlbOddsResponse` with identical fields. Prefer **shared generic models** (`OddsGame` / `OddsResponse`) only if the rename is cheap; otherwise duplicate MLB schemas with the same fields to avoid a wide WNBA refactor
- Merge: prefer DraftKings over FanDuel per game key (same as `merge_odds_prefer_primary`)
- Team matching: Sharp abbreviations when present; MLB name→abbrev map for fallbacks (all 30 clubs)
- Cache ~45s; last-good on error; optional `error` string; missing `SHARP_API_KEY` → empty games + error (mirror WNBA)
- Response `Cache-Control: no-store` on the route

### OpenAPI

- Export paths: `/api/mlb/scoreboard` (dated), `/api/mlb/odds/today`
- Keep `/api/mlb/scoreboard/today` required

---

## 3. Frontend

### Client

- `fetchMlbScoreboardByDate(dateEt)`
- `fetchMlbOdds()` → `/api/mlb/odds/today`
- Types from OpenAPI (`MlbOddsResponse` or shared odds schemas)

### Hooks

- Extend `useMlbScoreboard(dateEt?: string)` like `useWnbaScoreboard`: today key `["mlb","scoreboard","today"]`, dated key `["mlb","scoreboard", date]`; poll only for today
- Add `useMlbOdds()` with key `["mlb","odds","today"]`

### Page

- Replace MLB coming-soon branch in `LeagueMatchupsPage` with `MlbMatchupsPage` mirroring `WnbaMatchupsPage` (search params, invalid date wipe, odds window, panel props)
- Hero date from scoreboard `data.date` when present

### Cards

- `MatchupGameCard`: link via `gameDetailHref(game)` instead of `espnEventId`-only check
- Update card tests for MLB `mlbGamePk` → `/mlb/games/...`

### Errors / loading

- Identical semantics to WNBA matchups: never wipe good data; skeletons when loading with empty list; odds errors do not clear games

---

## 4. Testing

| Area | Coverage |
| --- | --- |
| Backend scoreboard | Dated route 200/502; normalize already covered |
| Backend odds | Normalize Sharp MLB fixture rows → spread/total; merge prefer DK; route smoke; missing key |
| Frontend | `useMlbScoreboard` dated path; `useMlbOdds`; AppRouter/page shows MatchupsPanel not “coming soon”; card MLB href; OpenAPI `check:api` |

---

## 5. File layout (expected)

```text
backend/app/
  api/routes/mlb_scoreboard.py     # + dated handler
  api/routes/mlb_odds.py           # new
  services/mlb_scoreboard.py       # + get_scoreboard_for_date
  services/mlb_odds.py             # or sharp_odds parameterized by league
  schemas/mlb_odds.py              # if not sharing WNBA odds models
  main.py                          # mount mlb_odds

frontend/src/
  hooks/useMlbScoreboard.ts        # dated
  hooks/useMlbOdds.ts
  lib/api.ts                       # fetch helpers
  pages/LeagueMatchupsPage.tsx     # MlbMatchupsPage
  components/league/MatchupGameCard.tsx
```

Exact factoring of Sharp fetch (shared `fetch_sharp_odds_rows(league=...)` vs copy) is an implementation choice; prefer a small parameterized helper over duplicating pagination.

---

## 6. Success criteria

- `/mlb/matchups` shows today’s MLB slate with live/rest sections when Stats API has games
- Prev/next date changes `?date=` and loads that slate
- Within odds window, cards show DK (or FD fallback) run line / total when Sharp returns data
- MLB cards navigate to `/mlb/games/:gamePk`; WNBA cards still use `/games/:espnEventId`
- WNBA matchups behavior unchanged
- OpenAPI types in sync; focused backend + frontend tests pass

## Maintenance

Update `docs/superpowers/specs/2026-08-02-website-api-system-design.md` page ↔ API rows for `/mlb/matchups` and the new endpoints.
