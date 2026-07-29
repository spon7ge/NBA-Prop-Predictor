# HoopVista Home Page Design

Date: 2026-07-29  
Status: Approved for planning

## Goal

Replace `/` with a dark, ticket-themed marketing/home landing that follows the structure of the provided boxseats mockup, branded as **HoopVista**. Move the existing slate app to `/slate`. Live ticker and LIVE NOW use empty/skeleton states so real score data can be wired later.

## Decisions

| Topic | Choice |
| --- | --- |
| Routing | `/` = new home; existing slate → `/slate` |
| Branding | HoopVista (not boxseats) |
| Live data (v1) | Empty / skeleton UI; typed optional props for future data |
| League nav | NBA + WNBA only (same visual pattern as mockup) |
| Approach | Full page rebuild at `/`; slate behavior unchanged aside from route |

## Routing

| Path | Element |
| --- | --- |
| `/` | New `HomePage` |
| `/slate` | Existing `App` (Top Legs / All Players / Results) |
| `/lab` | Unchanged |
| `*` | Unchanged `NotFoundPage` |

Landing chrome is separate from the slate Header/Footer for v1. No shared shell extraction yet.

## Page structure

Four vertical regions:

1. **HomeNav** — HoopVista mark (bar-chart icon + wordmark) left; center links NBA / WNBA with colored dots; right About (placeholder href) + settings gear (non-functional icon for v1).
2. **LiveTicker** — LIVE indicator + horizontal strip. Empty: muted “No live games” text (no skeleton chips). Accepts optional `games?: TickerGame[]`.
3. **TicketHero** — Large ticket card with dot-grid background.
   - Left: `SEASON PASS` label; **hoopvista** + BETA pill; tagline “your seat to every game.”; short props-focused description; primary CTA “See what’s inside” (smooth scroll to LIVE NOW); secondary “Learn the games” → `/slate`.
   - Right: ticket icon + decorative ring of sport icons (static).
   - Stub footer: gate/section/row/seat monospace details + `ADMIT ONE`.
4. **LiveNowSection** — “LIVE NOW” + count (“0 games in progress” when empty). Grid of three skeleton game cards (league pill, status, two team rows, score boxes) so layout matches the mockup without fake scores.

## File layout

```
frontend/src/pages/HomePage.tsx
frontend/src/components/home/HomeNav.tsx
frontend/src/components/home/LiveTicker.tsx
frontend/src/components/home/TicketHero.tsx
frontend/src/components/home/LiveNowSection.tsx
frontend/src/components/home/types.ts   # TickerGame, LiveGameCard, etc.
frontend/src/AppRouter.tsx              # wire / and /slate
```

## Visual system

- Dark landing only: black page background, charcoal (`~#121212`) cards, white/gray text.
- Existing Geist Sans + Geist Mono (no new Inter font).
- Rounded cards (~12–16px), pill CTAs matching mockup structure.
- Light motion: pulsing LIVE dot; optional subtle hero border glow. Sport-icon ring static in v1.
- League accent colors on nav dots and skeleton league pills (NBA / WNBA only).

## Data & empty states

- No API calls on the home page in v1.
- Components accept optional game arrays; when absent or empty:
  - Ticker: muted “No live games” message.
  - LIVE NOW: count `0`; three skeleton cards for layout fidelity.
- Types define the future shape (`id`, league, away/home abbrev + name, score, status/clock) so a later fetch can drop in without restructuring the UI.

## CTAs & navigation behavior

- “See what’s inside” → scroll to `#live-now`.
- “Learn the games” → `Link` to `/slate`.
- NBA / WNBA nav links: styled `<a href="#live-now">` anchors that scroll to LIVE NOW; they do not set `/slate` league filters in v1.
- About: placeholder (`#` or disabled-looking link).
- Settings: decorative button with `aria-label`, no modal.

## Out of scope

- Real live scores or scoreboard APIs.
- Settings modal / About page content.
- Multi-sport leagues beyond NBA / WNBA.
- Refactoring slate Header/Footer to the new dark chrome.
- Renaming the product globally outside this landing.

## Success criteria

- Visiting `/` shows the four-section layout matching the mockup structure with HoopVista branding.
- Visiting `/slate` shows the existing slate app working as before.
- LIVE NOW and ticker render coherent empty/skeleton states with no console errors.
- Primary and secondary hero CTAs behave as specified.
