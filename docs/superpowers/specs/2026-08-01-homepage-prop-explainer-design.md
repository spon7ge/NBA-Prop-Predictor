# Homepage prop explainer — annotated demo card

Date: 2026-08-01  
Status: Approved

## Goal

Add a PrizePicks-style teaching section on the marketing homepage so first-time visitors understand how HoopVista reads a player prop: market line, odds, model projection, and EV. Static LeBron Over example; Apple-quiet dark aesthetic.

## Decisions

| Topic | Choice |
| --- | --- |
| Placement | New section **after** `FeatureStrip`, **before** `LeagueCtaSection` |
| Approach | Annotated demo card (not mini table, not multi-prop board) |
| Card hierarchy | **Line-first** — hero number is FanDuel line `22.5`; Model / EV / FanDuel as metrics row |
| Interaction | Static Over example — no Under toggle; four always-on teaching callouts |
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
| `PropExplainerSection` | Section shell: eyebrow, headline, supporting line, summary blurb, CTA link |
| `PropExplainerCard` | Line-first card UI; static Over badge; metrics row; player header + matchup |
| `PropExplainerCallouts` | Desktop side callouts + leader hints; mobile stacked list; all callouts emphasized |

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
| EV | +4% |
| Side | Over (static badge) |

Copy tone matches homepage: short, plain English. Four callout slots (same titles in both layouts):

| Id | Title | Role |
| --- | --- | --- |
| `line` | The number to beat | Explains 22.5 points line and what Over means |
| `odds` | What −110 means | Minus = favorite; −110 means bet $110 to profit $100 |
| `edge` | Our model’s guess | Explains 24.7 vs 22.5 |
| `ev` | What EV means | Explains +4% EV in plain English |

All four callouts stay fully emphasized (static Over example — no Under toggle).

## Layout

### Desktop

- Centered section within existing `max-w-*` homepage container.
- Three-column teach row: left callouts (`line`, `edge`) | card | right callouts (`odds`, `ev`).
- Quiet dotted leaders (muted green) — decorative, not required for understanding.
- Primary CTA below the teach row.

### Mobile

- Single column: headline → card → stacked callouts → CTA.
- No side leaders; left accent bar on callout items.
- Card shows a static Over badge (not twin Over/Under buttons).

## Interaction

Static demo only. No Over/Under toggle. Card always shows Over + EV +4%.

## Visual rules

- Near-black section background consistent with homepage; card: hairline border `white/10`, soft fill — no heavy shadow required.
- Over badge: white fill / dark text (matches Apple-style primary control), not purple.
- EV chip: soft green border/text for positive.
- No jersey illustration; no flame/popularity chrome from the PrizePicks reference.

## Data / API

None. Constants colocated with the section (e.g. `DEMO_PROP` object). Do not fetch odds or model for this section.

## Out of scope

- Creating `/nba/props` or NBA prop picks page
- Live/randomizing demo player
- Multi-player carousel or Players/Teams/Culture tabs
- Wiring to `PropPicksTable` row styles
- Light mode
- Interactive Over/Under toggle

## Test plan

- Unit: `PropExplainerCard` / section render LeBron, 22.5, Model 24.7, FanDuel −110, Over badge
- Unit: EV +4%; no Under control present
- Unit: callouts teach line, odds (−110), model, and EV
- Unit: CTA `href` is `/wnba/prop_picks`
- Unit: `HomePage` includes explainer between FeatureStrip and LeagueCtaSection
- Manual: desktop callouts visible; narrow viewport shows stacked callouts

## Success criteria

A new visitor can explain what the line, −110 odds, model, and EV mean — without leaving the homepage.
