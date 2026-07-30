# Win Probability UI Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refresh the WNBA game-detail win probability section so it reads as a chart-first visualization that more closely matches the mockup.

**Architecture:** Keep the existing `win_probability` data flow untouched and focus only on `WinProbabilityPanel` presentation. The work splits into one component-restyle task and one verification/integration task so the chart UI can be reviewed independently before final page-level confirmation.

**Tech Stack:** React, TypeScript, Vitest, Testing Library, Tailwind utility classes

## Global Constraints

- Frontend-only UI refresh.
- Reuse the existing `win_probability` payload unchanged.
- Primary focus is a taller, more obvious chart with stronger visual hierarchy.
- Keep the current scrub/slider interaction; make the active state clearer.
- Keep team stats underneath, styled as comparison rows.
- Keep the existing unavailable copy when there is no win probability data.
- Keep timeline-only and stats-only partial rendering behavior.
- No backend or API changes.
- No third-party charting library.

---

## File Structure

- Modify `frontend/src/components/game/WinProbabilityPanel.tsx` to restyle the panel into a chart-first module while preserving existing data and accessibility behavior.
- Modify `frontend/src/components/game/WinProbabilityPanel.test.tsx` to lock the refreshed layout and interaction expectations.
- Optionally modify `frontend/src/components/game/testFixtures.ts` only if the panel tests need richer comparison-row or legend data.

### Task 1: Refresh `WinProbabilityPanel`

**Files:**
- Modify: `frontend/src/components/game/WinProbabilityPanel.tsx`
- Modify: `frontend/src/components/game/WinProbabilityPanel.test.tsx`
- Modify: `frontend/src/components/game/testFixtures.ts` (only if test data needs a richer stat block)

**Interfaces:**
- Consumes: `WinProbabilityPanel({ detail }: { detail: GameDetail })`
- Produces: the same public component interface with updated visual hierarchy and the same unavailable / partial-data behavior

- [ ] **Step 1: Write the failing UI test for chart-first presentation**

```typescript
it("renders a larger chart-first win probability module", () => {
  render(<WinProbabilityPanel detail={buildGameDetailFixture()} />);

  expect(screen.getByText("Win probability")).toBeInTheDocument();
  expect(screen.getByLabelText("Win probability chart")).toBeInTheDocument();
  expect(screen.getByText("GS")).toBeInTheDocument();
  expect(screen.getByText("PHX")).toBeInTheDocument();
  expect(screen.getByText("Field goal %")).toBeInTheDocument();
});
```

- [ ] **Step 2: Run the panel test to verify it fails**

Run: `npm --prefix "/Users/alexgonzalez/Documents/NBA-Prop-Predictor/frontend" run test -- src/components/game/WinProbabilityPanel.test.tsx`

Expected: FAIL because the current panel does not expose the larger chart-first layout or chart label expected by the new test.

- [ ] **Step 3: Update the panel implementation with the minimal visual refresh**

```typescript
<section className="rounded-xl border border-white/10 bg-[#141414] p-4 md:p-5">
  <div className="flex items-start justify-between gap-4">
    <div>
      <h2 className="text-sm font-semibold text-white">Win probability</h2>
      <p className="mt-1 text-xs text-white/50">
        {data.summary ?? `${detail.away.abbrev} vs ${detail.home.abbrev}`}
      </p>
    </div>
    <div className="flex items-center gap-3 text-xs">
      <span className="flex items-center gap-1 text-white/60">
        <span className="size-2 rounded-full" style={{ backgroundColor: detail.away.color }} />
        {detail.away.abbrev}
      </span>
      <span className="flex items-center gap-1 text-white/60">
        <span className="size-2 rounded-full" style={{ backgroundColor: detail.home.color }} />
        {detail.home.abbrev}
      </span>
    </div>
  </div>

  <svg
    aria-label="Win probability chart"
    viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
    className="mt-4 w-full"
  >
```

- [ ] **Step 4: Increase the chart hierarchy and active-state styling**

```typescript
const CHART_WIDTH = 320;
const CHART_HEIGHT = 180;

<line
  x1={0}
  x2={CHART_WIDTH}
  y1={CHART_HEIGHT / 2}
  y2={CHART_HEIGHT / 2}
  stroke="rgba(255,255,255,0.2)"
  strokeDasharray="4 4"
/>
<path d={path.area} fill={`${detail.home.color}26`} />
<path d={path.line} fill="none" stroke={detail.home.color} strokeWidth={2.5} />
{activePoint ? (
  <>
    <circle
      cx={xForIndex(activeIndex, points.length, CHART_WIDTH)}
      cy={yForPct(activePoint.homeWinPct, CHART_HEIGHT)}
      r={9}
      fill={`${detail.home.color}33`}
      pointerEvents="none"
    />
    <circle
      cx={xForIndex(activeIndex, points.length, CHART_WIDTH)}
      cy={yForPct(activePoint.homeWinPct, CHART_HEIGHT)}
      r={4.5}
      fill={detail.home.color}
      pointerEvents="none"
    />
  </>
) : null}
```

- [ ] **Step 5: Restyle the score context and stat rows**

```typescript
<div className="mt-4 flex items-end justify-between gap-4">
  <div>
    <p className="text-xs uppercase tracking-wide text-white/45">Active state</p>
    <div className="mt-1 flex items-baseline gap-2 text-white">
      <span className="font-mono text-lg font-semibold">
        {activePoint ? `${activePoint.awayScore}-${activePoint.homeScore}` : "—"}
      </span>
      <span className="text-sm text-white/65">
        Q{activePoint?.period} {activePoint?.clock}
      </span>
    </div>
  </div>
</div>

<div className="mt-6 space-y-3">
  {data.teamStats.map((stat) => (
    <div key={stat.key} className="space-y-1.5">
      <div className="grid grid-cols-[40px_1fr_40px] items-center gap-3 text-sm">
        <span className="text-white/80">{stat.awayValue}</span>
        <span className="text-center text-white/55">{stat.label}</span>
        <span className="text-right text-white/80">{stat.homeValue}</span>
      </div>
      <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-2">
        <div className="h-1.5 rounded-full" style={{ backgroundColor: `${detail.away.color}66` }} />
        <span className="text-[11px] text-white/35">vs</span>
        <div className="h-1.5 rounded-full" style={{ backgroundColor: `${detail.home.color}66` }} />
      </div>
    </div>
  ))}
</div>
```

- [ ] **Step 6: Keep fallback and partial rendering tests green**

```typescript
it("keeps timeline-only and stats-only states renderable", () => {
  const { rerender } = render(
    <WinProbabilityPanel
      detail={buildGameDetailFixture({
        winProbability: {
          ...buildGameDetailFixture().winProbability!,
          teamStats: [],
        },
      })}
    />,
  );

  expect(screen.getByLabelText("Win probability chart")).toBeInTheDocument();

  rerender(
    <WinProbabilityPanel
      detail={buildGameDetailFixture({
        winProbability: {
          ...buildGameDetailFixture().winProbability!,
          timeline: [],
        },
      })}
    />,
  );

  expect(screen.getByText("Field goal %")).toBeInTheDocument();
});
```

- [ ] **Step 7: Run the panel test file to verify it passes**

Run: `npm --prefix "/Users/alexgonzalez/Documents/NBA-Prop-Predictor/frontend" run test -- src/components/game/WinProbabilityPanel.test.tsx`

Expected: PASS

### Task 2: Verify the refreshed panel in the existing frontend surface

**Files:**
- Modify: `frontend/src/components/game/WinProbabilityPanel.test.tsx` (if Task 1 revealed missing assertions)
- Test: `frontend/src/components/game/WinProbabilityPanel.test.tsx`
- Test: `frontend/src/AppRouter.test.tsx`

**Interfaces:**
- Consumes: refreshed `WinProbabilityPanel`
- Produces: confidence that the updated panel still works in the routed game-detail page and still builds cleanly

- [ ] **Step 1: Run the focused routed/component suite**

Run: `npm --prefix "/Users/alexgonzalez/Documents/NBA-Prop-Predictor/frontend" run test -- src/components/game/WinProbabilityPanel.test.tsx src/AppRouter.test.tsx src/components/game/mapGameDetail.test.ts src/components/game/GameHeader.test.tsx src/components/game/PlayByPlay.test.tsx src/components/game/ShotChart.test.tsx`

Expected: PASS with the updated chart-first UI and no regressions in the surrounding game-detail tests.

- [ ] **Step 2: Run the frontend production build**

Run: `npm --prefix "/Users/alexgonzalez/Documents/NBA-Prop-Predictor/frontend" run build`

Expected: `tsc -b && vite build` exits 0.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/game/WinProbabilityPanel.tsx frontend/src/components/game/WinProbabilityPanel.test.tsx frontend/src/components/game/testFixtures.ts
git commit -m "Refresh win probability panel UI"
```
