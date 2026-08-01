# Dual-Line Win Probability Scrub Chart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the game-detail win probability chart as dual team-colored lines with scrub labels and a muted “rest of game” after the scrub point.

**Architecture:** Keep scrub state and team-stats UI in `WinProbabilityPanel`. Replace the home-only area/line + floating score tooltip with pure path helpers that emit vivid/muted SVG `d` strings for away and home series, plus on-chart clock and `ABBR %` labels at `activeIndex`. No API or mapper changes.

**Tech Stack:** React 19 · TypeScript · Vite · Tailwind CSS v4 · Vitest · Testing Library

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-01-win-probability-dual-line-design.md`
- Dual polylines from existing `awayWinPct` / `homeWinPct` only
- Team colors from `detail.away.color` / `detail.home.color`
- Mute path segments with index `> activeIndex`
- Active labels: abbrev + `%`; clock `Q{n} {clock}` (not wall-clock)
- Remove: home-only area fill; floating score tooltip; “Above the midline favors …” copy
- Keep: `GameSection`, axis labels/grid, mouse scrub + a11y range, team stats
- No API / mapper / chart-library changes

---

## File Structure

| File | Responsibility |
| --- | --- |
| `frontend/src/components/game/winProbabilityPaths.ts` | Pure helpers: coords, path `d` strings split at scrub |
| `frontend/src/components/game/winProbabilityPaths.test.ts` | Unit tests for path splits |
| `frontend/src/components/game/WinProbabilityPanel.tsx` | Dual-line chart UI + labels |
| `frontend/src/components/game/WinProbabilityPanel.test.tsx` | Panel behavior tests (update obsolete tooltip assertions) |

---

### Task 1: Dual-series path helpers

**Files:**
- Create: `frontend/src/components/game/winProbabilityPaths.ts`
- Create: `frontend/src/components/game/winProbabilityPaths.test.ts`

**Interfaces:**
- Consumes: chart geometry constants (re-export or duplicate the same numbers used by the panel today: width 640, height 140, pads L36/R8/T8/B6)
- Produces:
  - `export function xForIndex(index: number, count: number): number`
  - `export function yForPct(pct: number): number`
  - `export function nearestIndexForClientX(clientX: number, rect: DOMRect, count: number): number`
  - `export type SeriesKey = "away" | "home"`
  - `export function buildSeriesPathD(points: { awayWinPct: number; homeWinPct: number }[], series: SeriesKey, fromIndex: number, toIndex: number): string`
  - `export function buildSplitSeriesPaths(points: { awayWinPct: number; homeWinPct: number }[], activeIndex: number): { awayVivid: string; awayMuted: string; homeVivid: string; homeMuted: string }`

Semantics for `buildSeriesPathD`:
- Inclusive range `[fromIndex, toIndex]` on the timeline
- Empty string if `points.length === 0` or `fromIndex > toIndex` or out of bounds
- Single point → `M x y` only
- Y from `awayWinPct` or `homeWinPct` per `series`

Semantics for `buildSplitSeriesPaths`:
- Clamp `activeIndex` to `[0, points.length - 1]`
- Vivid: `[0, activeIndex]`
- Muted: `[activeIndex, points.length - 1]` (shares the scrub point so the join is continuous)
- If `activeIndex === last`, muted strings are empty (or `M` of last only — prefer **empty** muted when no future)

- [ ] **Step 1: Write the failing tests**

Create `frontend/src/components/game/winProbabilityPaths.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import {
  buildSeriesPathD,
  buildSplitSeriesPaths,
  xForIndex,
  yForPct,
} from "./winProbabilityPaths";

const points = [
  { awayWinPct: 56, homeWinPct: 44 },
  { awayWinPct: 46, homeWinPct: 54 },
  { awayWinPct: 40, homeWinPct: 60 },
];

describe("winProbabilityPaths", () => {
  it("maps index 0 to the left of the plot and 100% near the top", () => {
    expect(xForIndex(0, 3)).toBeLessThan(xForIndex(2, 3));
    expect(yForPct(100)).toBeLessThan(yForPct(0));
  });

  it("builds a polyline for a home series range", () => {
    const d = buildSeriesPathD(points, "home", 0, 2);
    expect(d.startsWith("M")).toBe(true);
    expect(d.includes(" L")).toBe(true);
  });

  it("returns empty string for an inverted range", () => {
    expect(buildSeriesPathD(points, "away", 2, 1)).toBe("");
  });

  it("splits vivid through scrub and muted after, sharing the scrub point", () => {
    const split = buildSplitSeriesPaths(points, 1);
    expect(split.homeVivid).toBe(buildSeriesPathD(points, "home", 0, 1));
    expect(split.homeMuted).toBe(buildSeriesPathD(points, "home", 1, 2));
    expect(split.awayVivid).toBe(buildSeriesPathD(points, "away", 0, 1));
    expect(split.awayMuted).toBe(buildSeriesPathD(points, "away", 1, 2));
  });

  it("omits muted paths when scrub is at the last point", () => {
    const split = buildSplitSeriesPaths(points, 2);
    expect(split.homeMuted).toBe("");
    expect(split.awayMuted).toBe("");
    expect(split.homeVivid).toBe(buildSeriesPathD(points, "home", 0, 2));
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npm test -- src/components/game/winProbabilityPaths.test.ts`

Expected: FAIL — cannot resolve `./winProbabilityPaths`

- [ ] **Step 3: Implement helpers**

Create `frontend/src/components/game/winProbabilityPaths.ts`:

```ts
const CHART_WIDTH = 640;
const CHART_HEIGHT = 140;
const CHART_PAD_LEFT = 36;
const CHART_PAD_RIGHT = 8;
const CHART_PAD_TOP = 8;
const CHART_PAD_BOTTOM = 6;
export const PLOT_WIDTH = CHART_WIDTH - CHART_PAD_LEFT - CHART_PAD_RIGHT;
export const PLOT_HEIGHT = CHART_HEIGHT - CHART_PAD_TOP - CHART_PAD_BOTTOM;

export const CHART_GEOMETRY = {
  width: CHART_WIDTH,
  height: CHART_HEIGHT,
  padLeft: CHART_PAD_LEFT,
  padRight: CHART_PAD_RIGHT,
  padTop: CHART_PAD_TOP,
  padBottom: CHART_PAD_BOTTOM,
  plotWidth: PLOT_WIDTH,
  plotHeight: PLOT_HEIGHT,
  yLabelX: CHART_PAD_LEFT - 8,
} as const;

export type SeriesKey = "away" | "home";

type PctPoint = { awayWinPct: number; homeWinPct: number };

export function xForIndex(index: number, count: number): number {
  if (count <= 1) return CHART_PAD_LEFT + PLOT_WIDTH / 2;
  return CHART_PAD_LEFT + (index / (count - 1)) * PLOT_WIDTH;
}

export function yForPct(pct: number): number {
  return CHART_PAD_TOP + PLOT_HEIGHT - (pct / 100) * PLOT_HEIGHT;
}

export function nearestIndexForClientX(
  clientX: number,
  rect: DOMRect,
  count: number,
): number {
  if (count <= 1) return 0;
  const plotLeft = rect.left + (CHART_PAD_LEFT / CHART_WIDTH) * rect.width;
  const plotWidth = (PLOT_WIDTH / CHART_WIDTH) * rect.width;
  const ratio = Math.min(Math.max((clientX - plotLeft) / plotWidth, 0), 1);
  return Math.round(ratio * (count - 1));
}

export function buildSeriesPathD(
  points: PctPoint[],
  series: SeriesKey,
  fromIndex: number,
  toIndex: number,
): string {
  if (points.length === 0) return "";
  if (fromIndex > toIndex) return "";
  if (fromIndex < 0 || toIndex >= points.length) return "";

  const coords = [];
  for (let i = fromIndex; i <= toIndex; i++) {
    const pct =
      series === "home" ? points[i].homeWinPct : points[i].awayWinPct;
    coords.push({ x: xForIndex(i, points.length), y: yForPct(pct) });
  }

  return coords
    .map((c, i) => `${i === 0 ? "M" : "L"}${c.x} ${c.y}`)
    .join(" ");
}

export function buildSplitSeriesPaths(
  points: PctPoint[],
  activeIndex: number,
): {
  awayVivid: string;
  awayMuted: string;
  homeVivid: string;
  homeMuted: string;
} {
  if (points.length === 0) {
    return { awayVivid: "", awayMuted: "", homeVivid: "", homeMuted: "" };
  }
  const scrub = Math.min(Math.max(activeIndex, 0), points.length - 1);
  const last = points.length - 1;
  return {
    awayVivid: buildSeriesPathD(points, "away", 0, scrub),
    homeVivid: buildSeriesPathD(points, "home", 0, scrub),
    awayMuted:
      scrub < last ? buildSeriesPathD(points, "away", scrub, last) : "",
    homeMuted:
      scrub < last ? buildSeriesPathD(points, "home", scrub, last) : "",
  };
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npm test -- src/components/game/winProbabilityPaths.test.ts`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/game/winProbabilityPaths.ts \
  frontend/src/components/game/winProbabilityPaths.test.ts
git commit -m "$(cat <<'EOF'
feat: add dual win-probability series path helpers

EOF
)"
```

---

### Task 2: Dual-line chart UI + test rewrite

**Files:**
- Modify: `frontend/src/components/game/WinProbabilityPanel.tsx`
- Modify: `frontend/src/components/game/WinProbabilityPanel.test.tsx`

**Interfaces:**
- Consumes: helpers from `./winProbabilityPaths` (`CHART_GEOMETRY`, `xForIndex`, `yForPct`, `nearestIndexForClientX`, `buildSplitSeriesPaths`)
- Produces: unchanged `WinProbabilityPanel({ detail }: { detail: GameDetail })`

- [ ] **Step 1: Rewrite failing / obsolete panel tests first**

In `WinProbabilityPanel.test.tsx`, replace the tests that assert the floating score tooltip and midline copy with dual-line scrub assertions. Keep GameSection / unavailable / team-stats / dense-slider tests.

Replace these three tests’ bodies (keep names updated to match behavior):

1. Rename `"renders the latest win probability point in a floating tooltip"` → `"renders dual on-chart labels for the latest point"`:

```ts
  it("renders dual on-chart labels for the latest point", () => {
    render(<WinProbabilityPanel detail={buildGameDetailFixture()} />);

    expect(screen.queryByText(/Above the midline favors/)).not.toBeInTheDocument();
    expect(screen.queryByText("GS 10–8 PHX")).not.toBeInTheDocument();
    expect(screen.getByText("PHX")).toBeInTheDocument();
    expect(screen.getByText("54%")).toBeInTheDocument();
    expect(screen.getByText("GS")).toBeInTheDocument();
    expect(screen.getByText("46%")).toBeInTheDocument();
    expect(screen.getByText("Q1 4:29")).toBeInTheDocument();
    expect(screen.getByText("Field goal %")).toBeInTheDocument();
  });
```

Note: fixture last point is period 1 clock `4:29`, home 54 / away 46 (see `testFixtures.ts`). If abbrev+label nodes are split, assert via `getByText` on each token as above. If multiple `GS`/`PHX` exist (team stats legend), scope to the chart container:

```ts
    const chart = screen.getByLabelText("Win probability chart");
    expect(chart).toHaveTextContent("PHX");
    expect(chart).toHaveTextContent("54%");
    expect(chart).toHaveTextContent("GS");
    expect(chart).toHaveTextContent("46%");
    expect(chart).toHaveTextContent("Q1 4:29");
```

Prefer chart-scoped assertions to avoid legend collisions.

2. Update pointer scrub test — expect label % change, not score string:

```ts
  it("updates on-chart labels when pointer moves near an earlier timeline point", () => {
    const { container } = render(
      <WinProbabilityPanel detail={buildGameDetailFixture()} />,
    );
    const chart = container.querySelector("svg");
    expect(chart).not.toBeNull();
    chart!.getBoundingClientRect = () =>
      ({
        left: 0,
        top: 0,
        width: 640,
        height: 140,
        right: 640,
        bottom: 140,
        x: 0,
        y: 0,
        toJSON: () => ({}),
      }) as DOMRect;

    fireEvent.mouseMove(chart!, { clientX: 40, clientY: 110 });

    expect(chart).toHaveTextContent("44%");
    expect(chart).toHaveTextContent("56%");
    expect(chart).toHaveTextContent("Q1 8:00");
    expect(chart).not.toHaveTextContent("54%");
  });
```

3. Update keyboard slider test similarly (value `"0"` → early point labels; no `GS 2–0 PHX`).

4. Add mute assertion:

```ts
  it("renders muted future path segments when scrub is not at the end", () => {
    const { container } = render(
      <WinProbabilityPanel detail={buildGameDetailFixture()} />,
    );
    const slider = screen.getByRole("slider", {
      name: /win probability timeline/i,
    });
    fireEvent.change(slider, { target: { value: "0" } });

    const muted = container.querySelectorAll("[data-wp-segment='muted']");
    expect(muted.length).toBeGreaterThanOrEqual(2);
    muted.forEach((el) => {
      expect(el.getAttribute("opacity")).toBe("0.35");
    });
  });
```

5. Update `"renders timeline-only data without team stats"` to drop score tooltip assertions; keep chart + slider + no Field goal %.

- [ ] **Step 2: Run panel tests — expect failures**

Run: `cd frontend && npm test -- src/components/game/WinProbabilityPanel.test.tsx`

Expected: FAIL on missing dual labels / still showing midline or score tooltip / missing muted segments

- [ ] **Step 3: Implement panel rewrite**

In `WinProbabilityPanel.tsx`:

1. Delete local `xForIndex` / `yForPct` / `nearestIndexForClientX` / `buildWinProbabilityPath` / `SummaryText` / `ActiveTooltip`.
2. Import from `./winProbabilityPaths`.
3. Compute:

```ts
  const scrub = Math.min(Math.max(activeIndex, 0), Math.max(points.length - 1, 0));
  const paths = buildSplitSeriesPaths(points, scrub);
  const activePoint =
    points.length > 0 ? (points[scrub] ?? points[points.length - 1]) : null;
```

4. Render four `<path>` elements (skip empty `d`):

```tsx
  const vividProps = {
    fill: "none",
    strokeWidth: 2.25,
    strokeLinejoin: "round" as const,
    strokeLinecap: "round" as const,
    "data-wp-segment": "vivid",
  };
  const mutedProps = {
    fill: "none",
    strokeWidth: 2.25,
    strokeLinejoin: "round" as const,
    strokeLinecap: "round" as const,
    stroke: "rgba(255,255,255,0.28)",
    opacity: 0.35,
    "data-wp-segment": "muted",
  };
```

- Away vivid: `stroke={detail.away.color}` + `d={paths.awayVivid}`
- Home vivid: `stroke={detail.home.color}` + `d={paths.homeVivid}`
- Away/home muted: `mutedProps` + respective `d` when non-empty

5. Scrub UI (when `activePoint`):
- Vertical dashed line at `xForIndex(scrub, points.length)` — `stroke="rgba(255,255,255,0.45)"` `strokeDasharray="3 3"`
- Circle on away + home at scrub (fill team color, white stroke optional)
- SVG `<text>` labels near each dot: `{abbrev}` and `{pct}%` in team color (or white with team-colored abbrev). Keep layout simple: abbrev+pct as one string per team to the right of the scrub (flip left if scrub past 75% of plot), e.g. `PHX 54%` / `GS 46%` as separate text nodes OR two lines.
- Clock text above plot or near top of scrub: `Q${activePoint.period} ${activePoint.clock}` with `data-wp-clock` and fill `rgba(255,255,255,0.7)`

For tests that use `toHaveTextContent("54%")`, putting abbrev and pct in SVG `<text>` nodes is enough.

Example label block:

```tsx
  const scrubX = xForIndex(scrub, points.length);
  const placeLeft = scrubX > CHART_GEOMETRY.padLeft + CHART_GEOMETRY.plotWidth * 0.75;
  const labelX = placeLeft ? scrubX - 8 : scrubX + 8;
  const labelAnchor = placeLeft ? "end" : "start";
```

```tsx
  <text x={labelX} y={yForPct(activePoint.homeWinPct)} fill={detail.home.color} textAnchor={labelAnchor} dominantBaseline="middle" style={{ fontSize: "11px", fontWeight: 600 }}>
    {detail.home.abbrev} {activePoint.homeWinPct}%
  </text>
  <text x={labelX} y={yForPct(activePoint.awayWinPct)} fill={detail.away.color} textAnchor={labelAnchor} dominantBaseline="middle" style={{ fontSize: "11px", fontWeight: 600 }}>
    {detail.away.abbrev} {activePoint.awayWinPct}%
  </text>
  <text x={scrubX} y={CHART_GEOMETRY.padTop - 2} fill="rgba(255,255,255,0.7)" textAnchor="middle" style={{ fontSize: "10px" }}>
    {`Q${activePoint.period} ${activePoint.clock}`}
  </text>
```

If clock sits outside viewBox (padTop - 2), bump `CHART_PAD_TOP` usage by increasing viewBox top padding **or** place clock at `padTop + 10` inside the plot. Prefer placing clock inside the plot at `y={CHART_GEOMETRY.padTop + 10}` so it stays visible without geometry changes.

6. Remove area fill path and `ActiveTooltip` portal div.
7. Keep axis, midline dashed 50%, grid, range input, team stats unchanged.
8. Use `CHART_GEOMETRY.width/height` for `viewBox`.

- [ ] **Step 4: Run panel + path tests**

Run:

```bash
cd frontend && npm test -- src/components/game/WinProbabilityPanel.test.tsx src/components/game/winProbabilityPaths.test.ts
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/game/WinProbabilityPanel.tsx \
  frontend/src/components/game/WinProbabilityPanel.test.tsx
git commit -m "$(cat <<'EOF'
feat: dual-line win probability chart with muted future scrub

EOF
)"
```

---

## Spec coverage (self-review)

| Spec requirement | Task |
| --- | --- |
| Dual away/home polylines | Task 1 + 2 |
| Mute after scrub | Task 1 split + Task 2 muted paths |
| On-chart abbrev/% + Qn clock | Task 2 |
| Remove area fill, score tooltip, midline copy | Task 2 |
| Keep GameSection, axes, scrub a11y, team stats | Task 2 |
| No API/mapper/library | Both (frontend UI only) |

## Placeholder scan

No TBD steps; concrete helpers, tests, and SVG props included.

## Type consistency

- `buildSplitSeriesPaths` / geometry exports from Task 1 consumed by Task 2
- Panel public props unchanged
