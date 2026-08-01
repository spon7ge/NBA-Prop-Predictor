# Homepage prop explainer — annotated demo card

Date: 2026-08-01  
Status: Approved

## Goal

Add a PrizePicks-style teaching section on the marketing homepage so first-time visitors understand how HoopVista reads a player prop: market line, model projection, EV, and Over/Under side. Static LeBron example; interactive side toggle; Apple-quiet dark aesthetic.

## Decisions

| Topic | Choice |
| --- | --- |
| Placement | New section **after** `FeatureStrip`, **before** `LeagueCtaSection` |
| Approach | Annotated demo card (not mini table, not multi-prop board) |
| Card hierarchy | **Line-first** — hero number is FanDuel line `22.5`; Model / EV / FanDuel as metrics row |
| Interaction | Full demo — toggle Over/Under flips selection, EV sign/color, and callout emphasis |
| Annotations | **Hybrid** — dotted leaders on desktop; stacked accent-bar callouts on mobile |
| CTA | White pill “See live props” → `/wnba/prop_picks` (live props route today; no `/nba/props` yet) |
| Data | Static demo only — no API |
| Color mode | Dark only; muted accents (soft green / soft red for EV), not PrizePicks purple/neon |
| Font | Geist (existing); hierarchy via weight/size/tracking |

## Page structure (updated)

1. `HomeNav` / `LiveTicker` (chrome)
2. `BrandHero`
3. Live now
4. Stories
5. FeatureStrip
6. **`PropExplainerSection`** ← new
7. LeagueCtaSection

Amends [homepage Apple-style rebuild](./2026-08-01-homepage-apple-style-rebuild-design.md) by inserting this section; FeatureStrip stays as Follow / Understand / Decide copy only.

## Components

| Piece | Responsibility |
| --- | --- |
| `PropExplainerSection` | Section shell: eyebrow, headline, supporting line, summary blurb, CTA link; **owns** `selectedSide` state and passes it + setters to card/callouts |
| `PropExplainerCard` | Line-first card UI; Over/Under buttons; metrics row; player header + matchup |
| `PropExplainerCallouts` | Desktop side callouts + leader hints; mobile stacked list; opacity/emphasis from `selectedSide` |

Files may be colocated under `frontend/src/components/home/`; split only if a single file becomes hard to navigate.

Reuse `TeamAbbrevAvatar` with `abbrev="LAL"` and `logoUrl={null}` (letter fallback).

## Demo content (static)

| Field | Value |
| --- | --- |
| Player | LeBron James |
| Team / pos | LAL · F |
| Matchup | DEN vs LAL |
| Tip | Tue 7:00pm |
| Stat | Points |
| Line (FanDuel) | 22.5 |
| Odds | −110 |
| Model | 24.7 |
| EV Over | +4% |
| EV Under | −4% |
| Default side | Over |

Copy tone matches homepage: short, plain English. Four callout slots (same titles in both layouts):

| Id | Title | Role |
| --- | --- | --- |
| `line` | The number to beat | Explains FanDuel 22.5 −110 |
| `side` | Pick the side | Explains Over vs Under |
| `edge` | Model edge | Explains 24.7 vs 22.5 → EV on Over |
| `flip` | Why EV flipped | Explains negative EV on Under |

**Over emphasis:** `line` + `side` full; `edge` secondary; `flip` hidden or heavily dimmed.  
**Under emphasis:** `edge` + `flip` full; `line` + `side` dimmed.

Exact body strings may be tuned in implementation; titles and emphasis rules stay fixed.

## Layout

### Desktop

- Centered section within existing `max-w-*` homepage container.
- Three-column teach row: left callouts | card | right callouts.
- Quiet dotted leaders (muted green when positive story, muted red when Under / negative EV) — decorative, not required for understanding.
- Primary CTA below the teach row.

### Mobile

- Single column: headline → card → stacked callouts → CTA.
- No side leaders; left accent bar on active callout items.
- Over/Under remain full-width twin buttons on the card.

## Interaction

1. Mount with **Over** selected; EV shows **+4%** (positive accent); apply Over emphasis rules above.
2. Selecting **Under**: button chrome swaps; EV shows **−4%** (negative accent); apply Under emphasis rules.
3. Selecting **Over** again restores Over emphasis.
4. Keyboard: buttons are real `<button>`s; selected side reflected with `aria-pressed`.
5. `prefers-reduced-motion`: no motion required beyond instant state change (no mandatory animation).

## Visual rules

- Near-black section background consistent with homepage; card: hairline border `white/10`, soft fill — no heavy shadow required.
- Selected side: white fill / dark text (matches Apple-style primary control), not purple.
- EV chip: soft green border/text for positive; soft red for negative.
- No jersey illustration; no flame/popularity chrome from the PrizePicks reference.
- lucide-react only if arrows on Over/Under need icons; prefer simple unicode/CSS arrows consistent with mockups.

## Data / API

None. Constants colocated with the section (e.g. `DEMO_PROP` object). Do not fetch odds or model for this section.

## Out of scope

- Creating `/nba/props` or NBA prop picks page
- Live/randomizing demo player
- Multi-player carousel or Players/Teams/Culture tabs
- Wiring to `PropPicksTable` row styles
- Light mode

## Test plan

- Unit: `PropExplainerCard` / section render LeBron, 22.5, Model 24.7, FanDuel −110
- Unit: default Over + EV +4%; click Under → EV −4% and `aria-pressed` updates
- Unit: CTA `href` is `/wnba/prop_picks`
- Unit: `HomePage` includes explainer between FeatureStrip and LeagueCtaSection
- Manual: desktop callouts visible; narrow viewport shows stacked callouts; reduced-motion still usable

## Success criteria

A new visitor can explain, after one toggle, what the line, model, and EV mean — without leaving the homepage.
