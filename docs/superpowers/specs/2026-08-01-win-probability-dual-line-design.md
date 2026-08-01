# Win probability — dual-line scrub chart

Date: 2026-08-01  
Status: Approved

## Goal

Restyle the game-detail win probability chart to match the dual-team reference: two team-colored lines, a scrubber with on-chart abbrev/% labels, and a muted “rest of game” after the scrub point — while staying on the quiet Apple-style `GameSection` surface and existing timeline data.

## Decisions

| Topic | Choice |
| --- | --- |
| Scope | Frontend `WinProbabilityPanel` (+ tests) only |
| Approach | Rebuild SVG dual series (no chart library) |
| Series | Away + home polylines from `awayWinPct` / `homeWinPct` |
| Colors | Team colors from `detail.away.color` / `detail.home.color` |
| Scrub default | Latest / last timeline point |
| Future mute | Path after `activeIndex` drawn muted (low opacity / neutral stroke) |
| Active labels | On-chart: colored dots + `ABBR` + `%`; game clock `Q{n} {clock}` near scrub |
| Removed | Home-only area fill; floating score tooltip; “Above the midline favors …” copy |
| Kept | `GameSection` wrapper; axis grid/labels; mouse scrub + a11y range; team stats below |
| Data | No API / mapper / shape changes |

## Chart behavior

1. Plot both win-% series over the existing timeline (index on X, 0–100% on Y).
2. User scrubs via mouse move on the SVG or the existing accessible range input.
3. At `activeIndex`:
   - Vertical scrub line
   - Dot on each series
   - Labels for away and home (`abbrev` + percent)
   - Clock label from `period` + `clock` (not wall-clock time)
4. Segments with index `> activeIndex` render muted so the scrubbed “present” reads first.
5. Empty / missing `winProbability` keeps the current unavailable message inside `GameSection`.

## Components

| Piece | Action |
| --- | --- |
| `WinProbabilityPanel.tsx` | Dual paths, scrub labels, mute-after-scrub; drop area fill + old tooltip + midline copy |
| `WinProbabilityPanel.test.tsx` | Cover dual labels, mute, empty state |
| Hooks / mappers / API | Unchanged |

## Out of scope

- Wall-clock timestamps
- Chart libraries (Recharts, etc.)
- Redesigning the team-stats bars
- Backend / ESPN win-probability payload changes
- Light mode

## Test plan

- Unit: both team abbrevs + % appear for the active (default last) point
- Unit: scrubbing / setting index updates labels
- Unit: muted future segment present when not at last index (opacity/class or dual path assertion)
- Unit: empty `winProbability` still shows unavailable copy
- Manual: live or final `/games/:id` — dual lines, scrub feels like the reference, team stats still below
