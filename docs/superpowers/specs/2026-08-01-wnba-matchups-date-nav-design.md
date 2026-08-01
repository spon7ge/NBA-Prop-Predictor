# WNBA matchups date navigation

Date: 2026-08-01  
Status: Approved for planning

## Goal

Enable the existing Previous / Next day controls on `/wnba/matchups` so users can browse any ET slate forward or backward. Today remains the default; other dates load real scoreboard data.

## Decisions

| Topic | Choice |
| --- | --- |
| Range | Unlimited — any past/future day upstream returns (empty slate OK) |
| Odds | Show odds only on today’s slate; hide off today |
| URL state | `?date=YYYY-MM-DD`; omit param when viewing today |
| API | `GET /api/wnba/scoreboard?date=YYYY-MM-DD`; keep `/scoreboard/today` |
| Overnight carryover | Only on `/today`; dated endpoint returns that calendar day only |
| Center label | “Today” on slate date; short date (e.g. `Jul 28`) otherwise; click returns to today |
| Polling | Live refetch only when viewing today with in-progress games |

## Architecture

```
/wnba/matchups[?date=YYYY-MM-DD]
        │
        ├── parse date (invalid → fall back to today, clear bad param)
        │
        ├── is today?
        │     yes → useWnbaScoreboard() + useWnbaOdds() → merge odds
        │     no  → fetchWnbaScoreboard(date) → no odds merge
        │
        └── MatchupsPanel
              prev/next update ?date (±1 ET day)
              center label: Today | short date
```

### Backend

**New route:** `GET /api/wnba/scoreboard?date=YYYY-MM-DD`

- Reuses `_merged_games_for_date` (ESPN + stats merge).
- Does **not** apply overnight live carryover (that stays on `/today` only).
- Invalid `date` → `422`. `date` is required on this route.
- Response shape unchanged: `{ date, games, fetched_at }`.
- Light per-date cache, separate from the today’s live cache, so arrowing around does not hammer upstream.
- On total upstream failure with no usable cache: `502` with `Cache-Control: no-store` (same spirit as today).

**Unchanged:** `GET /api/wnba/scoreboard/today` for ticker, prop pages, and the default matchups slate.

### Frontend

**URL**

| View | Path |
| --- | --- |
| Today | `/wnba/matchups` (no query) |
| Other day | `/wnba/matchups?date=2026-07-28` |

- Prev / Next enabled; step ±1 calendar day in America/New_York.
- Navigating to today clears `?date`.
- Invalid `?date` falls back to today and replaces/clears the bad param.
- Clicking the center label jumps to today and clears `?date`.

**Page wiring (`WnbaMatchupsPage`)**

- Selected date drives scoreboard fetch and `LeagueHero` `dateEt`.
- Odds merge only when selected date equals today’s slate date.
- Empty copy: always “No games on this slate” (today and other dates).
- Loading: existing skeletons while switching dates.
- Error with no data: “Unable to load matchups” status (unchanged).

**Components**

- Page owns URL search params and passes `selectedDate`, `isToday`, `onPrevDay`, `onNextDay`, and `onGoToday` into `MatchupsPanel`.
- Enable prev/next buttons; use normal (non-disabled) styling when enabled.
- Center control is always a button: label “Today” or short date; activates `onGoToday` (no-op when already today).

## Error handling

| Case | Behavior |
| --- | --- |
| Empty slate | 200 with `games: []`; UI empty message |
| Bad date format | Backend 422; frontend never sends invalid (local fallback) |
| Upstream down, no cache | 502; UI error status |
| Stale dated cache | Fixed TTL (~5 min) for non-today dates; no live polling |

## Testing

### Backend

- Valid `date` → 200 and `response.date` matches request.
- Invalid `date` → 422.
- Empty day → 200 with `games: []`.
- `/today` behavior unchanged (existing tests still pass).

### Frontend

- Prev / Next update `?date` by ±1 day.
- On slate date, label is “Today” and URL has no `date` param.
- Off today, odds pills are absent.
- Empty copy renders for a date with no games.
- Clicking center label from a past/future date returns to today.

## Out of scope

- Dated odds or props APIs
- NBA matchups date navigation
- Season window clamp or calendar date picker
- Changing live ticker date (always today’s slate)

## Success criteria

- On `/wnba/matchups`, chevrons move the slate forward and backward without range limits.
- URL reflects the selected date; sharing `?date=` opens that slate.
- Today shows odds; other dates do not.
- Empty and error states remain clear.
- Live ticker and `/scoreboard/today` continue to work as today.
