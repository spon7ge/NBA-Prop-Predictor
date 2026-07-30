# WNBA win probability UI refresh

Date: 2026-07-30  
Status: Approved for planning

## Goal

Make the existing WNBA game-detail win probability section read as a real chart-first visualization instead of a small numeric summary. Keep the current backend data flow and interaction model, but update the frontend presentation so it more closely matches the mockup.

## Decisions

| Topic | Choice |
| --- | --- |
| Scope | Frontend-only UI refresh |
| Data path | Reuse the existing `win_probability` payload unchanged |
| Primary focus | Taller, more obvious chart with stronger visual hierarchy |
| Interaction | Keep the current scrub/slider interaction; make the active state clearer |
| Stats block | Keep team stats underneath, styled as comparison rows |

## Architecture

This is a presentation pass on top of the existing game-detail data flow:

```text
GameDetailPage
  -> WinProbabilityPanel
      -> chart-first header + legend
      -> larger SVG chart
      -> active score / percentage state
      -> comparison stat rows
```

No API or backend changes are needed for this work.

## UI

### Panel changes

- Increase the chart height so it visually dominates the panel.
- Strengthen the chart styling:
  - clearer line stroke
  - more visible area fill
  - readable 50% midline
  - stronger active-point indicator
- Add clearer away/home identification near the chart so it reads like a two-team comparison.
- Keep the active score and percentages, but style them as chart context rather than the main event.

### Team stats block

- Keep the existing comparison rows under the chart.
- Style the away/home values and labels to resemble the mockup more closely.
- Preserve the current stat set: `Field goal %`, `Three point %`, `Free throw %`, `Rebounds`, `Offensive rebounds`, `Assists`.

### Interaction

- Preserve mouse scrubbing across the chart.
- Preserve the single slider control for keyboard accessibility.
- Make the currently selected point visually clearer than it is today.

### Empty / fallback behavior

- Keep the existing unavailable copy when there is no win probability data.
- Keep timeline-only and stats-only partial rendering behavior.

## File layout

```text
frontend/src/components/game/WinProbabilityPanel.tsx
frontend/src/components/game/WinProbabilityPanel.test.tsx
```

## Testing

- Update component tests to reflect the larger chart-first layout.
- Preserve coverage for:
  - unavailable state
  - timeline-only state
  - stats-only state
  - interaction changing the active point
- Verify the frontend build still passes.

## Out of scope

- Backend or API changes
- Reworking game-detail page layout outside `WinProbabilityPanel`
- Adding a third-party charting library
- New stat categories

## Success criteria

- The win probability section reads visually as a chart, not just a number.
- Away/home context is immediately legible.
- The active chart point is more obvious.
- Team stats still render clearly beneath the chart.
