# Live ticker TODAY fallback

Date: 2026-07-31  
Status: Approved for planning

## Goal

When no games are live, the home chrome ticker switches from an empty “No live games” state to a **TODAY** rail that marquees today’s remaining slate (upcoming scheduled games, then finals). Live mode is unchanged when any game is in progress.

## Decisions

| Topic | Choice |
| --- | --- |
| Approach | Fallback logic inside `LiveTicker` (full day slate already passed in) |
| Live mode | Any `live` / `halftime` → **LIVE** rail + in-progress games only |
| Idle mode | Else, any `scheduled` or `final` → **TODAY** rail + scheduled first, then finals |
| Empty / error | Empty `games` keeps **LIVE** chrome; copy is “No live games” or “Scoreboard unavailable” when `isError` |
| Scheduled item format | `ATL @ DAL 7:00 PM ET` (no scores) |
| Live / final item format | `ATL 36 — DAL 44 Q3 7:13` (omit null scores) |
| TODAY rail style | Same size/tracking as LIVE; muted `text-white/50`; static muted dot (no red pulse) |
| Data / API | Unchanged — `HomeChromeLayout` still passes `tickerGames` from `useWnbaScoreboard` |
| Sort order | Preserve scoreboard API order within each status group (scheduled block, then final block) |

## Mode selection

```
games (full day slate)
        │
        ▼
  any live/halftime? ──yes──► mode=live → filter in-progress
        │
        no
        ▼
  any scheduled/final? ──yes──► mode=today → scheduled[], then final[]
        │
        no
        ▼
  empty copy (No live games / Scoreboard unavailable)
```

When live games exist, scheduled and final games stay out of the strip.

## Visual structure

1. **Rail (fixed)** — Mode-dependent label left of the viewport.
   - **LIVE:** pulsing red dot + red uppercase `LIVE` (existing).
   - **TODAY:** static muted dot + muted uppercase `TODAY`.
2. **Marquee** — Existing CSS duplicate-track marquee, hover pause, `prefers-reduced-motion` behavior unchanged.
3. **Items** — Divider-separated monospace chips; links to `/games/:espnEventId` when present; duplicate track non-interactive + `aria-hidden`.

## Files touched

| File | Change |
| --- | --- |
| `frontend/src/components/home/LiveTicker.tsx` | Mode selection, TODAY rail, scheduled `@` rendering |
| `frontend/src/components/home/LiveTicker.test.tsx` | LIVE vs TODAY cases; scheduled/final fallback; empty/error unchanged |

No changes to backend, `mapScoreboard`, hooks, or `HomeChromeLayout`.

## Testing

- Empty `games` → “No live games”; `isError` → “Scoreboard unavailable”.
- Only scheduled → **TODAY** rail + `@` format + tip time; no “No live games”.
- Only finals → **TODAY** rail + score / em-dash / status format.
- Mixed scheduled + final (no live) → scheduled items before finals; **TODAY** rail.
- Any live (+ optional scheduled) → **LIVE** rail; scheduled hidden; live items with scores.
- Marquee duplicate track still `aria-hidden` and without focusable links.

## Out of scope

- Re-sorting by tip-off time beyond API order
- NBA ticker wiring
- Hiding the ticker strip on empty days
- Backend or scoreboard schema changes
- Per-team brand colors

## Success criteria

- With live games, behavior matches the current LIVE strip.
- With no live games but today’s slate present, users see **TODAY** and upcoming (then final) matchups instead of empty copy.
- Scheduled chips use `@`; live/final chips use scores and `—`.
- Unit tests for the cases above pass.
