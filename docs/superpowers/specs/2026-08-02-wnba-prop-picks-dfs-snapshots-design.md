# WNBA Prop Picks — DFS snapshots + Parlay sportsbooks

Date: 2026-08-02  
Status: Approved for planning

## Goal

On `/wnba/prop_picks`, showcase the **latest PrizePicks and Underdog main lines** from Supabase scraper snapshots, then attach matching **US sportsbook** quotes from the Parlay API. The board is **DFS-first**: only rows with PrizePicks, Underdog, or both appear; sportsbook-only props are dropped.

## Decisions

| Topic | Choice |
| --- | --- |
| Architecture | Extend `parlay_props` (approach 1) — same `GET /api/wnba/props/today` |
| DFS source | Supabase latest snapshots only (`odds.wnba_prizepicks`, `odds.wnba_underdogs`) |
| Parlay DFS quotes | Strip / ignore Parlay’s own `prizepicks` / `underdog` bookmaker rows on the response |
| Board anchor | Build rows from PP ∪ UD; match US books *to* DFS lines |
| PrizePicks main | `odds_type == "standard"` only (drop demon / goblin) |
| Underdog main | One line per player + stat + side from latest snapshot |
| Sportsbook lines | Parlay main lines only (`select_parlay_main_lines` + market allowlist) |
| Row keep rule | Keep if `prizepicks` and/or `underdog` quote is present after merge |
| Line match | Prefer exact DFS line; else closest sportsbook line for that player+stat+side |
| Frontend API | Unchanged client shape (`WnbaPropLine`) |
| UI polish | Default Book filter to PrizePicks + Underdog (optional in same change) |
| Out of scope | Demon/goblin UI, new composite service, Sharp props path, NBA props |

## Architecture

```
LeaguePropPicksPage  (/wnba/prop_picks)
        │
        ▼
GET /api/wnba/props/today
        │
        ▼
parlay_props.get_today_props()
  ├─ fetch_parlay_prop_rows()          # US books + market allowlist
  ├─ select_parlay_main_lines()
  ├─ normalize sportsbook quotes only  # clear Parlay PP/UD
  ├─ fetch_latest_prizepicks()
  ├─ fetch_latest_underdog()
  └─ attach_dfs_snapshots()            # DFS rows + match US books
```

## Data flow

1. **DFS snapshots** — Read latest `scraped_at` for WNBA PrizePicks and Underdog via existing `odds_snapshots` helpers.
2. **Filter PP** — Keep rows where `odds_type` equals `standard` (case-insensitive). Drop demon/goblin.
3. **Index UD** — One quote per `(normalized player, stat_key, side)` from the snapshot.
4. **Seed board** — Create `WnbaPropLine` buckets keyed by `(norm_player, stat_key, side)`:
   - PrizePicks standard rows seed **both** `over` and `under` with the same line (`odds_american` null).
   - Underdog rows seed the snapshot’s side only (with American price when present).
5. **Parlay sportsbooks** — Fetch Parlay props, drop non-allowlisted / milestone / `*_alt` markets, select main lines, normalize into per-book quotes. Do **not** copy Parlay `prizepicks` / `underdog` onto the line.
6. **Attach US books** — For each DFS bucket, find Parlay quotes with the same normalized player + stat key + side. **Target line:** PrizePicks line if present on the bucket, else Underdog line. Prefer `line == target`; otherwise pick minimal `|line - target|`. If PP and UD lines differ on the same bucket, still prefer an exact match to **either** DFS line before falling back to closest-to-PP. Leave the book cell null if no Parlay candidate exists for that player+stat+side.
7. **Emit** — Return only buckets that have `prizepicks` and/or `underdog` set. Sort as today (player, market, side).

### Example

| Source | Player | Stat | Line |
| --- | --- | --- | --- |
| PrizePicks (standard) | Caitlin Clark | Points | 19.5 |
| Underdog | Caitlin Clark | Points over | 19.5 (−108) |
| FanDuel (Parlay) | Caitlin Clark | Points over | 19.5 / 20.5 |

Result row: PP 19.5, UD 19.5 (−108), FanDuel **19.5**; a FanDuel-only 22.5 points prop with no DFS never appears.

## Backend

### Ownership

| Piece | Responsibility |
| --- | --- |
| `app/services/odds_snapshots.py` | Unchanged latest-snapshot reads |
| `app/services/parlay_props.py` | Pipeline orchestration; strip Parlay DFS; call attach |
| Helper (in `parlay_props` or small sibling module) | `attach_dfs_snapshots(parlay_props, pp_rows, ud_rows) -> list[WnbaPropLine]` |
| Stat key helpers | Reuse/adapt Sharp’s `_stat_key_from_pp_stat_type`, `_stat_key_from_ud_stat_name`, and Parlay `market_key` → stat key mapping |
| Player names | `norm_player_name` from roster helpers |
| Route | `wnba_props.py` stays pointed at `parlay_props.get_today_props` |

### Market allowlist (already in progress)

Keep the uncommitted allowlist that drops milestone / alt Parlay markets before main-line selection. That remains part of this feature’s sportsbook input quality.

### Response shape

Unchanged `WnbaPropsResponse` / `WnbaPropLine`. DFS quotes:

- PrizePicks: `line` set, `odds_american` usually `null`
- Underdog: `line` + `american_price` when present

### Errors & empty states

| Condition | Behavior |
| --- | --- |
| Missing `PARLAY_API_KEY` | Existing error payload |
| Supabase empty / unavailable | Log warning; return `props: []` (no invented DFS from Parlay) |
| Parlay fails, snapshots OK | Return DFS rows with blank US book cells |
| Both fail | Existing error / empty handling |
| Cache | Keep ~45s TTL after successful compose |

## Frontend

- No required schema changes.
- Optional: default `PropPicksFilters` book selection to PrizePicks + Underdog so those columns lead.
- Footer may note odds from Parlay + DFS snapshots if copy is updated; not required for v1.
- Continue excluding past games via existing scoreboard filter.

## Testing

Unit tests (backend):

1. PrizePicks non-standard (`demon` / `goblin`) rows are ignored.
2. Row with PP only is kept; UD only is kept; both is kept.
3. Sportsbook-only Parlay prop is dropped.
4. Exact DFS line wins over a farther sportsbook alt.
5. Parlay-native prizepicks/underdog quotes do not appear on output when snapshots are empty for that player (and are never preferred over snapshots).
6. Stat key mapping covers core markets used on the board (points, rebounds, assists, threes, PRA combos as supported by scrapers + allowlist).

No new E2E required for v1; existing prop picks page tests remain valid if defaults change.

## Non-goals

- Replacing scrapers with Parlay DFS persistence tables (`*_parlay`)
- Showing demon/goblin as separate columns
- NBA `/prop_picks` parity in this change
- Reviving Sharp as the sportsbook source
