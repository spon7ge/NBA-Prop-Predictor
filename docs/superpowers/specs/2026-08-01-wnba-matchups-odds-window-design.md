# WNBA matchups odds window + DK/FD fallback

Date: 2026-08-01  
Status: Approved for planning

## Goal

Show Sharp team odds on `/wnba/matchups` cards for **today, tomorrow, and the day after** (ET) whenever lines exist. Prefer DraftKings; fall back to FanDuel **per game** when DK has no usable line for that matchup.

## Decisions

| Topic | Choice |
| --- | --- |
| Odds window | Today + tomorrow + day after (ET); hide past and day+3+ |
| Provider | Sharp (unchanged route) |
| Primary book | DraftKings |
| Fallback | FanDuel, per game (whole game from one book) |
| Route | Keep `GET /api/wnba/odds/today` |
| `game_date` | Parse `YYYY-MM-DD` from Sharp `event_id` when present |
| Merge key | `(away_abbrev, home_abbrev, game_date?)` |
| Card UI | Existing odds pill; no sportsbook badge |
| Usable line | At least one of spread or total after normalize |

Supersedes the date-nav decision “odds only on today’s slate” for the three-day window only. Past dates and day+3+ still hide odds.

## Architecture

```
Sharp draftkings ──┐
                   ├─► sharp_odds: normalize each book
Sharp fanduel ─────┘         │
                             ▼
                    per-game merge (DK > FD)
                             │
                             ▼
                   GET /api/wnba/odds/today
                   games[{ home, away, spread, total,
                           game_date?, sportsbook? }]
                             │
                             ▼
/wnba/matchups[?date=YYYY-MM-DD]
  selectedDate ∈ {today, today+1, today+2}?
    yes → mergeMatchupOdds (prefer matching game_date)
    no  → no odds on cards
```

## Backend

### Fetch

- Parallel Sharp calls with `sportsbook=draftkings` and `sportsbook=fanduel`.
- Same markets as today: `point_spread,total_points`, `is_main_line=true`, league `wnba`.
- If one book fails, still normalize and return the other.
- Cache the **merged** response ~45s (same TTL as today).

### Normalize

- Existing collapse to favorite spread + total per event.
- Add optional `game_date` when `event_id` contains `YYYY-MM-DD` (e.g. `wnba_dream_storm_2026-07-31_b3` → `2026-07-31`).
- Add optional per-game `sportsbook` (`draftkings` | `fanduel`).
- Top-level response `sportsbook` remains `"draftkings"` as the default label; per-game field is source of truth for fallbacks.

### Merge (DK preferred, per game)

1. Build maps keyed by `(away_abbrev, home_abbrev, game_date or "")`.
2. Start from all DK games with a usable line.
3. For each FD game with a usable line whose key is not already covered by DK, append the FD game.
4. Do **not** mix DK spread with FD total on the same card — entire game from one book.

### Schema

`WnbaOddsGame` gains optional:

- `game_date: str | None` — ISO date when known
- `sportsbook: str | None` — book that supplied the lines

## Frontend

### Window gate (`WnbaMatchupsPage`)

Replace `isToday ? odds : undefined` with:

- Merge when `selectedDate` is in `{today, today+1, today+2}` (ET via existing slate helpers).
- Otherwise pass no odds into `mergeMatchupOdds`.

### Merge matching (`mergeMatchupOdds`)

- When an odds row has `game_date`, prefer matching that date to the scoreboard slate date (plus abbrevs).
- When `game_date` is missing, keep abbrev-only matching (backward compatible).
- Pill format unchanged (`formatOddsPill`).

### Tests to update

- Date-nav / matchups tests that assert odds only on today → allow +1/+2; still absent on past and day+3.

## Error handling

| Case | Behavior |
| --- | --- |
| DK fails, FD ok | Serve FD games |
| FD fails, DK ok | Serve DK games |
| Both fail, cache hit | Last good merged payload + `error` |
| Both fail, no cache | Empty `games` + `error` |
| No line for a matchup | Card has no odds pill |
| Outside 3-day window | No merge even if payload has lines |
| Unparseable `event_id` date | Omit `game_date`; abbrev-only match |

## Testing

### Backend

- Parse `game_date` from `event_id`.
- Same key: DK preferred over FD.
- FD fills a game missing from DK.
- One book throws → other still returned.
- Existing normalize / route tests still pass (optional fields).

### Frontend

- Odds merge on today, today+1, today+2.
- Odds absent on past dates and day+3.
- Date-keyed merge when `game_date` present.
- Abbrev-only fallback when `game_date` absent.

## Out of scope

- Past-date odds
- Mixing books within a single game’s markets
- Sportsbook label on matchup cards
- ParlayAPI cutover
- NBA matchups odds
- New dated odds route (`?date=`)

## Success criteria

- Tomorrow / day-after matchup cards show spread/total when Sharp has lines.
- Cards without DK lines still show odds when FanDuel has them for that game.
- Past and far-future slates do not show odds.
- DK remains the default when both books have the same matchup.
- Existing today odds behavior remains correct.
