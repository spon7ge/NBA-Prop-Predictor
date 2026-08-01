# HoopVista homepage — Apple-style rebuild

Date: 2026-08-01  
Status: Approved

## Goal

Rebuild the marketing homepage to feel calm, content-first, and Apple-like (quiet chrome, generous space, restrained motion) while keeping live games as a first-class section. Dark-only for this pass.

## Decisions

| Topic | Choice |
| --- | --- |
| Layout | Brand-first stack |
| Color mode | Dark only (no light mode in this pass) |
| Ticker | Keep `LiveTicker`; quieter type/opacity; marquee still respects `prefers-reduced-motion` |
| Hero | Split: copy left, floating NBA↔WNBA logo crossfade right; no border/card/frame |
| Hero CTA | Anchor/scroll to Live now (`#live-now`) |
| Live now | Keep section; quieter cards; **retain team icons** (`TeamAbbrevAvatar` / existing logo treatment) |
| Stories | Keep; place **after Live**; max 2–3 items; typography-led |
| Feature strip | New “Built for clarity” — Props / Edges / Explain |
| League CTAs | “Enter a league” — NBA → `/nba/matchups`, WNBA → `/wnba/matchups` |
| Removed | Ticket-style `TicketHero` (icon ring / ticket chrome); Explore; Learn the Game |
| Nav | Restyle quieter; keep NBA / WNBA / About; reduce loud colored active pills |
| Tokens | Prefer `@theme` / CSS variables for color, radius, muted text; avoid new hardcoded hex in components |
| Font | Geist only (existing); hierarchy via weight/size/tracking |

## Page structure (top → bottom)

1. `HomeNav`
2. `LiveTicker` (chrome)
3. **BrandHero** — Season Pass · Beta, HoopVista, tagline, supporting line, “See what’s live”; right: logo slideshow
4. **Live now** — live scoreboard cards with team icons
5. **Stories** — 2–3 editorial cards
6. **FeatureStrip** — Props / Edges / Explain
7. **LeagueCtaSection** — NBA / WNBA entry

## Components

| Piece | Action |
| --- | --- |
| `BrandHero` | New — replaces `TicketHero` on `HomePage` |
| `LeagueLogoSlideshow` | New — NBA/WNBA assets (`basketball.png`, `wnba_basketball.png`); opacity/scale crossfade; pause optional; static first logo under `prefers-reduced-motion` |
| `LiveNowSection` | Restyle; keep data wiring + team icons |
| `StoriesSection` | Restyle; cap density |
| `FeatureStrip` | New |
| `LeagueCtaSection` | New |
| `HomeNav` / `LiveTicker` | Restyle only |
| `ExploreSection` / `LearnTheGameSection` / `TicketHero` | Remove from `HomePage` (files may remain unused until cleanup) |

## Visual rules

- Near-black background, near-white text, muted secondary at low opacity
- One primary CTA (white pill) in hero; secondary actions via text/links
- Hairline borders or background separation for cards — soft shadows rare/absent
- Section padding scales mobile → desktop; shared `max-w-*` container
- lucide-react only where icons remain; consistent stroke; no decorative icon rings
- Focus rings visible; semantic headings per section

## Motion

- Logo slideshow: slow opacity (+ slight scale) crossfade (~5–7s cycle)
- No marquee/flashy hero motion beyond that
- `prefers-reduced-motion`: static logo; ticker already disables marquee

## Data

- No new APIs. Continue `useWnbaScoreboard` for ticker + Live now.
- Stories remain static defaults (existing `DEFAULT_STORIES`) unless already prop-driven.

## Out of scope

- Light mode / dual theme
- League hub pages (matchups, props, standings, leaders) visual overhaul
- About page redesign
- Reintroducing Explore / Learn content
- Scoreboard or props API changes
- Official NBA/WNBA trademark logo assets beyond existing app icons (use current basketball assets)

## Test plan

- Unit: `BrandHero` / `LeagueLogoSlideshow` render; reduced-motion shows a single static logo; CTA targets `#live-now`
- Unit: `HomePage` includes Live, Stories, Feature strip, League CTAs; excludes Explore / Learn / ticket hero
- Unit: `LiveNowSection` still renders team icons for games with abbrevs
- Update nav/ticker tests if class contracts change
- Manual: homepage at `/` — hero crossfade, live cards with icons, stories after live, league links work; check reduced-motion in OS settings
