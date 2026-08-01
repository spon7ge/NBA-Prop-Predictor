# Game Detail Apple-Style Quiet Surfaces Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restyle `/games/:espnEventId` so scheduled, live/halftime, and final states share the same quiet Apple-style surfaces as homepage Live now.

**Architecture:** Frontend-only restyle. Introduce a small shared `GameSection` shell (Live-now quiet surface class), apply it to header + all scheduled/live panels, replace amber/violet/`#141414` accents with white mono scores and red-only live status. No API, hook, or mapper changes.

**Tech Stack:** React 19 · TypeScript · Vite · Tailwind CSS v4 · Vitest · Testing Library · React Router

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-01-game-detail-apple-style-design.md`
- Section shell class (verbatim): `rounded-xl border border-white/10 bg-white/[0.03] p-4`
- Scores: white mono — drop amber score boxes / amber scoring-play numbers
- Live/halftime: red pulse + `text-red-400` only — drop `violet-500`
- Final/scheduled: muted `text-white/55`; no red/violet status pulse
- Remove `#141414` from this page
- Prefer theme/opacity utilities; no new hardcoded hex on this page
- No API / polling / mapper / route changes
- Dark only; Geist already in app

---

## File Structure

| File | Responsibility |
| --- | --- |
| `frontend/src/components/game/GameSection.tsx` | Shared quiet section shell + exported surface class constant |
| `frontend/src/components/game/GameSection.test.tsx` | Shell class contract |
| `frontend/src/components/game/GameHeader.tsx` | Quiet scoreboard; white scores; red live only |
| `frontend/src/components/game/GameHeader.test.tsx` | Header accent / score assertions |
| `frontend/src/components/game/MatchupPrediction.tsx` | Use `GameSection` |
| `frontend/src/components/game/ProjectedStarters.tsx` | Use `GameSection` |
| `frontend/src/components/game/SeasonLeaders.tsx` | Use `GameSection` |
| `frontend/src/components/game/InjuryReport.tsx` | Use `GameSection` |
| `frontend/src/components/game/ShotChart.tsx` | Wrap in `GameSection`; keep filters |
| `frontend/src/components/game/PlayByPlay.tsx` | Wrap in `GameSection`; white scoring scores |
| `frontend/src/components/game/WinProbabilityPanel.tsx` | Wrap in `GameSection` |
| `frontend/src/components/game/BoxScore.tsx` | Wrap in `GameSection` |
| `frontend/src/pages/GameDetailPage.tsx` | Skeleton quiet surface |
| Existing panel `*.test.tsx` | Update only if class/DOM contracts break |

---

### Task 1: `GameSection` shell primitive

**Files:**
- Create: `frontend/src/components/game/GameSection.tsx`
- Create: `frontend/src/components/game/GameSection.test.tsx`

**Interfaces:**
- Consumes: `React.ComponentProps<"section">`
- Produces:
  - `export const GAME_SECTION_SURFACE = "rounded-xl border border-white/10 bg-white/[0.03] p-4"`
  - `export function GameSection({ children, className, ...props }: React.ComponentProps<"section">): JSX.Element`

- [ ] **Step 1: Write the failing test**

Create `frontend/src/components/game/GameSection.test.tsx`:

```tsx
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { GAME_SECTION_SURFACE, GameSection } from "./GameSection";

describe("GameSection", () => {
  it("applies the quiet Live-now surface classes", () => {
    render(
      <GameSection>
        <h2>Shot chart</h2>
      </GameSection>,
    );
    const section = screen.getByRole("heading", { name: "Shot chart" }).closest(
      "section",
    );
    expect(section).toHaveClass(
      "rounded-xl",
      "border",
      "border-white/10",
      "bg-white/[0.03]",
      "p-4",
    );
  });

  it("exports GAME_SECTION_SURFACE matching the Live-now card treatment", () => {
    expect(GAME_SECTION_SURFACE).toBe(
      "rounded-xl border border-white/10 bg-white/[0.03] p-4",
    );
  });

  it("merges an optional className", () => {
    render(
      <GameSection className="space-y-3">
        <span>inner</span>
      </GameSection>,
    );
    expect(screen.getByText("inner").closest("section")).toHaveClass(
      "space-y-3",
      "rounded-xl",
    );
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test -- src/components/game/GameSection.test.tsx`

Expected: FAIL — cannot resolve `./GameSection`

- [ ] **Step 3: Write minimal implementation**

Create `frontend/src/components/game/GameSection.tsx`:

```tsx
import type { ComponentProps } from "react";

/** Quiet surface matching homepage Live now cards. */
export const GAME_SECTION_SURFACE =
  "rounded-xl border border-white/10 bg-white/[0.03] p-4";

export function GameSection({
  children,
  className = "",
  ...props
}: ComponentProps<"section">) {
  return (
    <section
      className={`${GAME_SECTION_SURFACE} ${className}`.trim()}
      {...props}
    >
      {children}
    </section>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test -- src/components/game/GameSection.test.tsx`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/game/GameSection.tsx frontend/src/components/game/GameSection.test.tsx
git commit -m "$(cat <<'EOF'
feat: add quiet GameSection shell for game detail

EOF
)"
```

---

### Task 2: Quiet `GameHeader`

**Files:**
- Modify: `frontend/src/components/game/GameHeader.tsx`
- Modify: `frontend/src/components/game/GameHeader.test.tsx`

**Interfaces:**
- Consumes: `GameDetail` from `./types`; `GAME_SECTION_SURFACE` from `./GameSection` (or inline same class string on the scoreboard card)
- Produces: unchanged `GameHeader({ detail }: { detail: GameDetail })`

- [ ] **Step 1: Write the failing tests**

Append to `GameHeader.test.tsx` (keep existing tests; add):

```tsx
  it("uses red live accents, not violet", () => {
    const { container } = renderHeader({ status: "live", statusLabel: "4:13 - 1st" });
    expect(container.querySelector(".bg-violet-500")).toBeNull();
    expect(container.querySelector(".bg-red-500")).not.toBeNull();
    expect(container.querySelector(".text-red-400")).not.toBeNull();
  });

  it("does not pulse a live accent when final", () => {
    const { container } = renderHeader({
      status: "final",
      statusLabel: "Final",
    });
    expect(container.querySelector(".bg-red-500")).toBeNull();
    expect(container.querySelector(".bg-violet-500")).toBeNull();
    expect(container.querySelector(".animate-pulse")).toBeNull();
  });

  it("renders scores in white mono without amber chrome", () => {
    const { container } = renderHeader();
    expect(container.querySelector(".text-amber-300")).toBeNull();
    const score = screen.getByText("10");
    expect(score.className).toMatch(/text-white/);
    expect(score.className).toMatch(/font-mono/);
  });

  it("uses the quiet surface on the scoreboard card", () => {
    const { container } = renderHeader();
    expect(container.querySelector(".bg-\\[\\#141414\\]")).toBeNull();
    // Card wraps team names
    const card = screen.getByText("Golden State Valkyries").closest("div.rounded-xl");
    expect(card).toHaveClass("border-white/10", "bg-white/[0.03]", "p-4");
  });
```

Note: if `closest("div.rounded-xl")` is fragile, assert via `container.innerHTML` / classList on the scoreboard wrapper you identify after render. Prefer querying the element that contains both team names.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npm test -- src/components/game/GameHeader.test.tsx`

Expected: FAIL on violet/amber/`#141414` assertions

- [ ] **Step 3: Implement quiet header**

Update `GameHeader.tsx`:

1. Import `GAME_SECTION_SURFACE` from `./GameSection`.
2. Replace `ScoreBox` with Live-now-style score (no black amber box):

```tsx
function ScoreValue({ score }: { score: number | null }) {
  return (
    <span className="shrink-0 font-mono text-xl font-semibold tracking-tight text-white">
      {score ?? "–"}
    </span>
  );
}
```

3. Top bar status: keep `statusAccent` map (`live`/`halftime` → `text-red-400`; `scheduled`/`final` → `text-white/55`). Keep red `animate-pulse` dot only when `inProgress`.
4. Scoreboard card: `className={GAME_SECTION_SURFACE}`.
5. Meta line inside card: always use muted decorative dot `bg-white/25` (no violet/red in the card — live red lives only in the top bar). Keep `statusLabel · venue` once.
6. `TeamRow` uses `ScoreValue` instead of `ScoreBox`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npm test -- src/components/game/GameHeader.test.tsx`

Expected: PASS (including existing logo/back/venue tests)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/game/GameHeader.tsx frontend/src/components/game/GameHeader.test.tsx
git commit -m "$(cat <<'EOF'
feat: quiet GameHeader scores and live accents

EOF
)"
```

---

### Task 3: Scheduled panels → `GameSection`

**Files:**
- Modify: `frontend/src/components/game/MatchupPrediction.tsx`
- Modify: `frontend/src/components/game/ProjectedStarters.tsx`
- Modify: `frontend/src/components/game/SeasonLeaders.tsx`
- Modify: `frontend/src/components/game/InjuryReport.tsx`
- Modify tests only if they assert `#141414` or outer class strings (check with grep)

**Interfaces:**
- Consumes: `GameSection` from `./GameSection`
- Produces: same public component APIs as today

- [ ] **Step 1: Grep for brittle class assertions**

Run:

```bash
cd frontend && rg '#141414|bg-\[#141414\]' src/components/game src/pages/GameDetailPage.test.tsx
```

Expected: hits in the four panel components (and possibly skeleton). Fix any tests that assert `#141414` before/with the swap.

- [ ] **Step 2: Write a failing scheduled-surface smoke (if no existing class test)**

If `MatchupPrediction` has a test file, append; otherwise add a focused assertion in `GameDetailPage.test.tsx` for scheduled path — prefer extending an existing scheduled render test:

```tsx
  it("scheduled panels use quiet surfaces instead of #141414", async () => {
    // existing fetch mock returns scheduled detail with preview fields
    // after render:
    expect(document.querySelector(".bg-\\[\\#141414\\]")).toBeNull();
    expect(screen.getByText("Matchup prediction").closest("section")).toHaveClass(
      "bg-white/[0.03]",
      "border-white/10",
    );
  });
```

Use the page’s existing scheduled fixture/mock pattern in `GameDetailPage.test.tsx` (see `baseGameDetail` + scheduled status).

- [ ] **Step 3: Run to verify failure (still on `#141414`)**

Run: `cd frontend && npm test -- src/pages/GameDetailPage.test.tsx`

Expected: FAIL until panels swap

- [ ] **Step 4: Swap shells**

In each of the four files, replace:

```tsx
<section className="rounded-xl border border-white/10 bg-[#141414] p-4">
```

with:

```tsx
import { GameSection } from "./GameSection";
// ...
<GameSection>
```

and close with `</GameSection>`. Keep inner content unchanged.

- [ ] **Step 5: Run tests**

Run:

```bash
cd frontend && npm test -- src/pages/GameDetailPage.test.tsx src/components/game/MatchupPrediction.test.tsx src/components/game/ProjectedStarters.test.tsx src/components/game/SeasonLeaders.test.tsx src/components/game/InjuryReport.test.tsx
```

Expected: PASS (skip missing test files if any)

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/game/MatchupPrediction.tsx \
  frontend/src/components/game/ProjectedStarters.tsx \
  frontend/src/components/game/SeasonLeaders.tsx \
  frontend/src/components/game/InjuryReport.tsx \
  frontend/src/pages/GameDetailPage.test.tsx
git commit -m "$(cat <<'EOF'
feat: apply quiet GameSection to scheduled game panels

EOF
)"
```

---

### Task 4: Live panels — wrap + drop amber

**Files:**
- Modify: `frontend/src/components/game/ShotChart.tsx`
- Modify: `frontend/src/components/game/PlayByPlay.tsx`
- Modify: `frontend/src/components/game/WinProbabilityPanel.tsx`
- Modify: `frontend/src/components/game/BoxScore.tsx`
- Modify: `frontend/src/components/game/PlayByPlay.test.tsx` (amber → white if asserted)
- Optionally: `frontend/src/components/game/ShotChart.test.tsx`, `WinProbabilityPanel.test.tsx`, `BoxScore.test.tsx`

**Interfaces:**
- Consumes: `GameSection` from `./GameSection`
- Produces: same public APIs; root element becomes `<GameSection>` (or `<GameSection>` wrapping current root)

- [ ] **Step 1: Write failing tests**

Append to `PlayByPlay.test.tsx`:

```tsx
  it("wraps content in the quiet GameSection surface", () => {
    render(<PlayByPlay detail={detail} />);
    const heading = screen.getByRole("heading", { name: /play-by-play/i });
    expect(heading.closest("section")).toHaveClass(
      "rounded-xl",
      "border-white/10",
      "bg-white/[0.03]",
      "p-4",
    );
  });

  it("renders scoring scores in white mono without amber", () => {
    const { container } = render(<PlayByPlay detail={detail} />);
    expect(container.querySelector(".text-amber-300")).toBeNull();
    const score = screen.getByText("2-3");
    expect(score.className).toMatch(/text-white/);
    expect(score.className).toMatch(/font-mono/);
  });
```

Add similar “quiet section wraps heading” tests for `ShotChart` (`/shot chart/i`), `WinProbabilityPanel` (`/win probability/i`), and `BoxScore` (use team abbrev heading / first team name — BoxScore has no h2; assert the outer `section` from `GameSection` via `screen.getByText(detail.away.abbrev).closest("section")`).

For BoxScore, wrap the whole multi-team block in one `GameSection` (spec: each panel gets a shell — one shell for the box score block is enough).

- [ ] **Step 2: Run to verify failure**

Run:

```bash
cd frontend && npm test -- src/components/game/PlayByPlay.test.tsx src/components/game/ShotChart.test.tsx src/components/game/WinProbabilityPanel.test.tsx src/components/game/BoxScore.test.tsx
```

Expected: FAIL on missing section shell / amber still present

- [ ] **Step 3: Implement**

**PlayByPlay.tsx**
- Outer root: `<GameSection>` instead of `<div>`
- Change scoring score class from `text-amber-300` to `text-white` (keep `font-mono font-semibold`)
- Unify title to `text-sm font-semibold text-white` (match other panels)

**ShotChart.tsx**
- Outer root: `<GameSection>` instead of `<div>`
- Leave SVG / filters as-is

**WinProbabilityPanel.tsx**
- Wrap both the empty and populated return trees in `<GameSection>` (or single outer `GameSection` for each return)

**BoxScore.tsx**
- When rendering content, wrap the existing `<section className="space-y-10 pt-2">` as:

```tsx
<GameSection className="space-y-10">
  ...
</GameSection>
```

(drop redundant `pt-2` if padding comes from the shell)

- [ ] **Step 4: Run tests**

Run:

```bash
cd frontend && npm test -- src/components/game/PlayByPlay.test.tsx src/components/game/ShotChart.test.tsx src/components/game/WinProbabilityPanel.test.tsx src/components/game/BoxScore.test.tsx
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/game/ShotChart.tsx \
  frontend/src/components/game/PlayByPlay.tsx \
  frontend/src/components/game/WinProbabilityPanel.tsx \
  frontend/src/components/game/BoxScore.tsx \
  frontend/src/components/game/*.test.tsx
git commit -m "$(cat <<'EOF'
feat: quiet-wrap live game panels and drop amber scores

EOF
)"
```

---

### Task 5: Skeleton quiet surface + regression sweep

**Files:**
- Modify: `frontend/src/pages/GameDetailPage.tsx`
- Modify: `frontend/src/pages/GameDetailPage.test.tsx` if needed

**Interfaces:**
- Consumes: `GAME_SECTION_SURFACE` from `@/components/game/GameSection`
- Produces: unchanged `GameDetailPage`

- [ ] **Step 1: Update skeleton**

In `GameDetailSkeleton`, replace the header placeholder card:

```tsx
<div className="rounded-xl border border-white/10 bg-[#141414] p-4">
```

with:

```tsx
import { GAME_SECTION_SURFACE } from "@/components/game/GameSection";
// ...
<div className={GAME_SECTION_SURFACE}>
```

- [ ] **Step 2: Assert no `#141414` / amber / violet on live page render**

Append to `GameDetailPage.test.tsx` (use existing live mock if present; otherwise mirror scheduled test with `status: "live"` and minimal shots/plays):

```tsx
  it("live detail page has no legacy #141414, amber, or violet chrome", async () => {
    // mock fetch → live detail (existing live test pattern in this file)
    renderGameDetail("401857098");
    await screen.findByText(/* a known live label or team name from mock */);
    expect(document.body.innerHTML).not.toMatch(/#141414/);
    expect(document.querySelector(".text-amber-300")).toBeNull();
    expect(document.querySelector(".bg-violet-500")).toBeNull();
  });
```

Fill the `findByText` target from the existing live mock in `GameDetailPage.test.tsx` / `AppRouter.test.tsx` fixtures already used for live.

- [ ] **Step 3: Run full game-detail related suite**

Run:

```bash
cd frontend && npm test -- src/components/game src/pages/GameDetailPage.test.tsx src/hooks/useGameDetail.test.tsx
```

Expected: PASS

- [ ] **Step 4: Grep leftover legacy tokens on game detail UI**

Run:

```bash
cd frontend && rg 'bg-\[#141414\]|text-amber-300|bg-violet-500' src/components/game src/pages/GameDetailPage.tsx
```

Expected: no matches

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/GameDetailPage.tsx frontend/src/pages/GameDetailPage.test.tsx
git commit -m "$(cat <<'EOF'
feat: quiet game detail skeleton and legacy accent sweep

EOF
)"
```

---

## Spec coverage (self-review)

| Spec requirement | Task |
| --- | --- |
| Shared quiet shell matching Live now | Task 1 |
| GameHeader quiet surface, white scores, red live only, single status·venue | Task 2 |
| Scheduled panels surface swap | Task 3 |
| Live panels wrapped; amber scoring scores dropped; title scale unified | Task 4 |
| Skeleton quiet surface; no `#141414`/amber/violet on page | Task 5 |
| No API/hooks/mapper changes | All tasks (frontend UI only) |

## Placeholder scan

No TBD/TODO steps; concrete classes, file paths, commands, and commit messages included.

## Type consistency

- `GAME_SECTION_SURFACE` and `GameSection` defined in Task 1; consumed by Tasks 2–5
- Public panel/`GameHeader` prop types unchanged
