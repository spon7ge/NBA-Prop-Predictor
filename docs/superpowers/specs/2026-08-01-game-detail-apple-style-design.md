# Game detail page — Apple-style quiet surfaces

Date: 2026-08-01  
Status: Approved

## Goal

Bring `/games/:espnEventId` in line with the homepage Apple-style system so scheduled, live/halftime, and final states feel like the same calm, content-first page as Live now — not a louder, older theme when the game tips off.

## Decisions

| Topic | Choice |
| --- | --- |
| Scope | Whole page — header + scheduled panels + live/final panels |
| Approach | Quiet surfaces (match Live now cards) |
| Color mode | Dark only (existing page chrome) |
| Section shell | Shared quiet surface: `rounded-xl border border-white/10 bg-white/[0.03] p-4` |
| Scores | White mono — drop amber score boxes / amber scoring-play numbers |
| Live accent | Red pulse + `text-red-400` only — drop `violet-500` |
| Final / scheduled status | Muted `text-white/55`; no status accent pulse (live red only when in progress) |
| Hardcoded hex | Remove `#141414` and amber/violet accents from this page; prefer theme/opacity utilities |
| Data / routes | Unchanged — restyle only |
| Font | Geist only; hierarchy via weight/size/tracking |

## Visual language

Align with homepage Live now (`LiveNowSection` card treatment):

- Near-black page background, near-white primary text, muted secondary via white opacity
- Hairline borders + translucent fill for section surfaces — no soft shadows
- One live accent (red) used only for in-progress status
- Team name colors stay (data-driven identity); scores stay neutral white
- Filters remain quiet pills (`bg-white/15` active)
- Latest-play strip: hairline border only (no nested heavy card)

## Page structure (unchanged)

1. Back + status top bar
2. Scoreboard header card
3. **Scheduled:** Matchup prediction → Projected starters → Season leaders → Injury report  
   **Live / half / final:** Shot chart + Play-by-play (2-col) → Win probability → Box score

## Components

| Piece | Action |
| --- | --- |
| `GameHeader` | Quiet surface; white scores; red live only; single status·venue line in card |
| Optional `GameSection` | Small shared shell primitive if it avoids repeating the surface class |
| `MatchupPrediction` / `ProjectedStarters` / `SeasonLeaders` / `InjuryReport` | Swap `#141414` shell → quiet surface |
| `ShotChart` / `PlayByPlay` / `WinProbabilityPanel` / `BoxScore` | Wrap in quiet shell; drop amber accents; unify title scale |
| `GameDetailPage` skeleton | Match quiet surface |
| Hooks / mappers / API | No changes |

### Header specifics

- Keep Back → `/` and status label in the top bar
- Scoreboard card: `statusLabel · venue` once (remove redundant status chrome)
- Team rows: logo + name (team color) + white mono score — same hierarchy as Live now

### Live panel specifics

- Shot chart and Play-by-play each get their own shell inside the existing `lg:grid-cols-2` grid
- Win probability and Box score each get their own shell below
- Scoring play scores: white mono (not amber)

## Out of scope

- Light mode
- API, polling, or mapper changes
- New game-detail features or layout reorders
- League hub / matchups / props visual overhaul
- Full design-token type-scale migration beyond this page’s surfaces/accents

## Test plan

- Unit: `GameHeader` — live uses red (not violet); final/scheduled muted; scores render without amber chrome; logos still optional
- Unit: live path sections render inside the quiet shell (class or role)
- Unit: scheduled panels still render content after surface swap
- Existing GameDetail / ShotChart / PlayByPlay / WinProb / BoxScore behavior tests stay green
- Manual: open a live (or final) `/games/:id` and a scheduled one — same quiet chrome as Live now; no violet/amber/`#141414`
