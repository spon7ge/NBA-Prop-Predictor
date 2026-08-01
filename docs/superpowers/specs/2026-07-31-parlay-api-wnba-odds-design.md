# ParlayAPI WNBA odds (replace Sharp)

Date: 2026-07-31  
Status: Approved for planning

## Goal

Cut Sharp out of WNBA live odds. Prop picks and matchup odds come from [ParlayAPI](https://parlay-api.com/docs). Prop picks show six sportsbooks; matchup cards prefer Pinnacle with DraftKings fallback; all six books snapshot to Supabase on a 30-minute throttle.

## Decisions

| Topic | Choice |
| --- | --- |
| Provider | ParlayAPI (`https://parlay-api.com/v1`), key `PARLAY_API_KEY` |
| Scope | Replace Sharp for WNBA props + matchup odds |
| Prop books | `fanduel`, `draftkings`, `caesars`, `betmgm`, `pinnacle`, `bet365` |
| Prop picks UI | Those six only — drop PrizePicks / Underdog from this table |
| Matchup odds | One book per game: Pinnacle preferred, else DraftKings |
| Architecture | Shared `parlay_client` + `parlay_props` / `parlay_odds` services |
| Snapshots | Per-book tables (keep FD/DK; add four more), joint 30-min throttle |
| Throttle env | `PARLAY_PROPS_SNAPSHOT_MINUTES` (default `30`) |
| Main lines | Closest-to-balanced over/under (~−110/−110) per player/market/book |
| Cache | ~45s in-process; on error serve last good if present |
| Routes | Unchanged: `GET /api/wnba/props/today`, `GET /api/wnba/odds/today` |

## Architecture

```
Prop picks                          Matchup odds
─────────                          ────────────
GET /api/wnba/props/today           GET /api/wnba/odds/today
        │                                   │
        ▼                                   ▼
  parlay_props service               parlay_odds service
        │                                   │
        └──────────┬────────────────────────┘
                   ▼
            parlay_client
   base: https://parlay-api.com/v1
   key:  PARLAY_API_KEY
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
 /sports/basketball_wnba/props   /sports/basketball_wnba/odds
 bookmakers=six books            regions=us
                                 markets=spreads,totals
                                 bookmakers=pinnacle,draftkings
```

- Same frontend routes/pages; only fetch + normalize layer changes.
- PrizePicks/Underdog scrapers and DB tables remain for other uses; they are not merged into prop picks.
- `SHARP_API_KEY` unused for WNBA after this cutover (leave in `.env` for now; cleanup later).

## Prop picks

### Parlay fetch

```
GET /v1/sports/basketball_wnba/props
  ?apiKey=...
  &bookmakers=draftkings,fanduel,caesars,betmgm,pinnacle,bet365
  &markets=player_points,player_rebounds,player_assists,player_threes,
           player_steals,player_blocks,player_turnovers,player_pra,
           player_pts_rebs,player_pts_asts,player_rebs_asts,
           player_double_double,player_triple_double
```

(Exact market list may be trimmed to what Parlay returns live for WNBA; discover via `/props/markets` if needed.)

### Normalize

Parlay rows look like:

```json
{
  "bookmaker": "draftkings",
  "player": "Rhyne Howard",
  "market_key": "player_assists",
  "market": "Assists",
  "line": 3.5,
  "over_price": -114,
  "under_price": -110,
  "home_team": "...",
  "away_team": "..."
}
```

Rules:

1. Filter to the six bookmaker keys above.
2. **Main line:** Parlay has no `is_main_line`. For each `(player, market_key, bookmaker)`, keep the single line whose over/under American prices are closest to balanced (minimize distance from −110/−110, or equivalently closest to 50/50 implied). Drop alternate lines.
3. Expand each kept row into Over and Under sides (`odds_american` = `over_price` / `under_price`).
4. Bucket by `(player, market_type, side)` with one quote slot per book.
5. Include a side-row if **any** of the six books has that side; leave missing books `null`.
6. Team/logo: keep ESPN roster index (Parlay gives full team names only).
7. `stat` from Parlay `market` label when present, else derive from `market_key`.

### API response

```json
{
  "as_of": "2026-07-31T22:00:00Z",
  "sportsbooks": [
    "fanduel", "draftkings", "caesars", "betmgm", "pinnacle", "bet365"
  ],
  "props": [
    {
      "player_name": "Rhyne Howard",
      "team_abbrev": "ATL",
      "logo_url": "...",
      "stat": "Assists",
      "market_type": "player_assists",
      "side": "over",
      "model_prediction": null,
      "over_under_pct": null,
      "ev": null,
      "fanduel": { "line": 3.5, "odds_american": -114 },
      "draftkings": { "line": 3.5, "odds_american": -120 },
      "caesars": null,
      "betmgm": { "line": 3.5, "odds_american": -115 },
      "pinnacle": { "line": 3.5, "odds_american": -108 },
      "bet365": null
    }
  ]
}
```

Remove `prizepicks` / `underdog` fields from `WnbaPropLine` and `PROP_SPORTSBOOKS`.

### UI

Column order: FanDuel → DraftKings → Caesars → BetMGM → Pinnacle → bet365.

Same hairline rows + mono odds pills. Caption: `Odds by FanDuel, DraftKings, Caesars, BetMGM, Pinnacle & bet365`. Horizontally scrollable on narrow viewports. Model / EV stay blank in this change.

## Matchup odds

```
GET /v1/sports/basketball_wnba/odds
  ?regions=us
  &markets=spreads,totals
  &bookmakers=pinnacle,draftkings
```

- Per game: prefer **Pinnacle** for favorite spread + total; if that book is missing either market, fall back to **DraftKings** for the missing piece(s) (prefer consistent book when both markets exist on Pinnacle).
- Keep existing `WnbaOddsGame` / `WnbaOddsResponse` shape; set `sportsbook` to `"pinnacle"` or `"draftkings"` for the book that supplied the displayed lines.
- Cache ~45s; last-good on error; missing key → empty + error string.

## Snapshots

### Schema

Migration after `020`: create four tables matching `odds.wnba_fanduel` / `odds.wnba_draftkings`:

- `odds.wnba_caesars`
- `odds.wnba_betmgm`
- `odds.wnba_pinnacle`
- `odds.wnba_bet365`

Same columns, PK, and `(league, scraped_at DESC)` index naming pattern as migration 020.

### Semantics

```
get_today_props()
  ├── fetch Parlay props (existing cache ~45s)
  ├── maybe_persist_parlay_props(rows)   # best-effort
  │     ├── if no SUPABASE_DB_URL → return
  │     ├── if last scraped_at within N min (any of six tables) → return
  │     ├── map main-line rows → per-book DataFrames
  │     └── insert batches (shared scraped_at)
  └── return response
```

- Trigger after successful Parlay props fetch/normalize.
- Joint throttle across all six tables for `league=wnba`.
- Default N = 30 via `PARLAY_PROPS_SNAPSHOT_MINUTES`.
- DB failure: log; never fail the API response.
- Replace Sharp-oriented snapshot helpers (`sharp_props_to_book_rows`, `maybe_persist_sharp_props`) with Parlay equivalents covering all six books.

## Config & errors

- `PARLAY_API_KEY` required for live WNBA odds/props.
- Missing key or Parlay HTTP errors → empty props/odds + `error` string; serve last good cache when present.
- Snapshot path independent: live response succeeds even if DB write fails.

## Implementation sketch

| Area | Change |
| --- | --- |
| `backend/app/services/parlay_client.py` | Shared HTTP helper (base URL, key, timeout) |
| `backend/app/services/parlay_props.py` | Replace `sharp_props` for `/props/today` |
| `backend/app/services/parlay_odds.py` | Replace `sharp_odds` for `/odds/today` |
| Routes | Point at new services; delete or stop importing Sharp WNBA callers |
| `backend/app/schemas/wnba_props.py` | Six books; drop PP/UD |
| Frontend `PropPicksTable` + types/tests | Six columns; update caption |
| `db/migrations/021_*.sql` | Four new book tables |
| `src/odds/snapshot_rows.py` + `load_snapshots.py` | Parlay six-book persist + throttle |
| Config | Read `PARLAY_API_KEY`; snapshot minutes env |

## Out of scope

- NBA prop picks / NBA odds migration
- Serving prop picks from Supabase instead of live Parlay
- Line-move UI / CLV charts
- Removing PrizePicks/Underdog scrapers or their tables
- Parlay WebSocket streaming
- Deleting `SHARP_API_KEY` from env / removing Sharp modules used only if nothing else remains

## Testing

- Normalize Parlay props fixture → six-book Over/Under rows; main-line picker drops alts
- Normalize Parlay odds fixture → Pinnacle preferred; DraftKings when Pinnacle missing
- Snapshot mapping for all six books; joint throttle skips when any table is fresh
- Persist failure swallowed; props API still returns
- Frontend: six columns; no PP/UD; caption updated
- Missing `PARLAY_API_KEY` degrades gracefully

## Success criteria

- `/wnba/prop_picks` shows live lines from the six Parlay books only
- `/wnba/odds` (matchup) uses Pinnacle when present, else DraftKings — no Sharp calls
- Six snapshot tables writable at most once per 30 minutes after a successful props fetch
- Missing key or Parlay errors degrade gracefully (empty + message / last cache)
