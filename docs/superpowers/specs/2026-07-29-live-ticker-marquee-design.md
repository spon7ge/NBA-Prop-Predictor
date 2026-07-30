# Live Ticker Marquee Design

Date: 2026-07-29  
Status: Approved for planning

## Goal

Upgrade `LiveTicker` to match the boxseats-style score strip: fixed `● LIVE` label, divider-separated game items with score/matchup formatting, and a continuous CSS marquee that pauses on hover and respects reduced motion.

## Decisions

| Topic | Choice |
| --- | --- |
| Item format | Live/halftime/final: `ATL 36 — DAL 44 Q3 7:13`; scheduled: `ATL @ DAL 7:00 PM ET` |
| Motion | CSS infinite marquee (duplicate track, `translateX(-50%)`) |
| Hover | Pause via `animation-play-state: paused` |
| Accessibility | Honor `prefers-reduced-motion` (static strip); do not announce duplicated track twice |
| Team colors | Simple two-tone (away one accent, home another) — no per-team palette |
| Layout | Full-bleed ticker under nav; LIVE fixed left; track scrolls beside it |
| Backend | Unchanged — scores already on scoreboard API; extend frontend `TickerGame` only |

## Visual structure

1. **LIVE rail (fixed)** — Pulsing red dot + `LIVE` in red uppercase. Does not scroll.
2. **Marquee viewport** — `overflow: hidden` track to the right of LIVE.
3. **Items** — Flex row with thin vertical dividers and horizontal padding between games.
4. **Typography** — Small monospace-leaning type for abbreviations, scores, and status.
5. **Empty / error** — Static muted copy next to LIVE (`No live games` / `Scoreboard unavailable`); no marquee.

## Data model

Extend `TickerGame`:

```ts
awayScore: number | null;
homeScore: number | null;
```

`mapToTickerGames` maps `g.away.score` / `g.home.score`.

Rendering rules:

- **scheduled** (or both scores null): `awayAbbrev @ homeAbbrev statusLabel`
- **otherwise**: `awayAbbrev [score] — homeAbbrev [score] statusLabel` (omit a score token when null)

## Motion

- Render the game list twice in one animated flex track.
- CSS keyframes move the track by exactly 50% of its width for a seamless loop.
- Hover on the ticker pauses the animation.
- Under `prefers-reduced-motion: reduce`, disable the animation and show a single static (optionally horizontally scrollable) row.
- Duplicate copy is `aria-hidden` so assistive tech hears each game once.

## Files touched

| File | Change |
| --- | --- |
| `frontend/src/components/home/types.ts` | Add score fields on `TickerGame` |
| `frontend/src/components/home/mapScoreboard.ts` | Map scores into ticker games |
| `frontend/src/components/home/mapScoreboard.test.ts` | Assert scores on mapper output |
| `frontend/src/components/home/LiveTicker.tsx` | Layout, item format, CSS marquee |
| `frontend/src/components/home/LiveTicker.test.tsx` | Format + empty/error cases |

Optional: small CSS module or Tailwind `@keyframes` in existing global CSS if keyframes cannot live cleanly in the component.

`HomeChromeLayout` / hook wiring stays the same. Any test fixtures that construct `TickerGame` literals (e.g. layout tests) must include `awayScore` / `homeScore`.

## Testing

- Empty and never-loaded error copy unchanged.
- Live game shows abbrevs, scores, em dash, and status.
- Scheduled game shows `@` and status; no score numbers.
- Mapper includes `awayScore` / `homeScore`.
- Few games still duplicate the track (seamless loop).

## Out of scope

- Per-team brand color maps
- League name chips inside ticker items
- JS / `requestAnimationFrame` scroll engines
- Backend or API schema changes
- Nav bar redesign

## Success criteria

- With games present, the ticker continuously scrolls left and pauses on hover.
- Item formatting matches the reference pattern (scores for in-progress/final; `@` for scheduled).
- Reduced-motion users see a non-animated strip.
- Existing empty/error behavior is preserved.
- Unit tests for mapper + ticker format pass.
