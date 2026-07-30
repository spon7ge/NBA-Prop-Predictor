# Live Ticker Marquee Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `LiveTicker` to a boxseats-style scrolling score strip with scores, dividers, CSS marquee, hover pause, and reduced-motion fallback.

**Architecture:** Extend `TickerGame` with scores; map them in `mapToTickerGames`. Reformat ticker items (live with em dash + scores; scheduled with `@`). CSS infinite marquee via duplicated track + `@keyframes` in `index.css`; fixed LIVE rail; pause on hover; static under `prefers-reduced-motion`.

**Tech Stack:** React, TypeScript, Tailwind v4, Vitest, Testing Library

## Global Constraints

- Item format: live/halftime/final → `ATL 36 — DAL 44 Q3 7:13`; scheduled → `ATL @ DAL 7:00 PM ET`
- Motion: CSS infinite marquee (duplicate track, `translateX(-50%)`)
- Hover: `animation-play-state: paused`
- Accessibility: honor `prefers-reduced-motion`; duplicate track `aria-hidden`
- Team colors: two-tone only (away sky, home rose)
- Full-bleed ticker; no backend changes
- Out of scope: per-team colors, league chips, JS scroll engines, nav redesign

---

### Task 1: Extend `TickerGame` + mapper scores

**Files:**
- Modify: `frontend/src/components/home/types.ts`
- Modify: `frontend/src/components/home/mapScoreboard.ts`
- Modify: `frontend/src/components/home/mapScoreboard.test.ts`
- Modify: `frontend/src/layouts/HomeChromeLayout.test.tsx` (fixture scores)
- Modify: `frontend/src/components/home/LiveTicker.test.tsx` (fixture scores)

**Interfaces:**
- Produces: `TickerGame` with `awayScore: number | null` and `homeScore: number | null`
- Produces: `mapToTickerGames` includes those fields from API scores

- [ ] **Step 1: Write the failing mapper assertion**

In `mapScoreboard.test.ts`, update the expected ticker object:

```ts
expect(mapToTickerGames([apiGame()])).toEqual([
  {
    id: "g1",
    league: "wnba",
    awayAbbrev: "ATL",
    homeAbbrev: "DAL",
    statusLabel: "Q3 7:13",
    status: "live",
    awayScore: 36,
    homeScore: 44,
  },
]);
```

Add a scheduled case asserting null scores:

```ts
it("maps null scores for scheduled ticker games", () => {
  const scheduled = apiGame({
    status: "scheduled",
    status_label: "7:00 PM ET",
    away: { abbrev: "NYL", name: "New York Liberty", score: null },
    home: { abbrev: "LVA", name: "Las Vegas Aces", score: null },
  });
  expect(mapToTickerGames([scheduled])[0]).toMatchObject({
    awayScore: null,
    homeScore: null,
    status: "scheduled",
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/components/home/mapScoreboard.test.ts`
Expected: FAIL — expected object missing `awayScore` / `homeScore`

- [ ] **Step 3: Minimal implementation**

`types.ts` — add to `TickerGame`:

```ts
awayScore: number | null;
homeScore: number | null;
```

`mapScoreboard.ts`:

```ts
export function mapToTickerGames(games: ApiWnbaGame[]): TickerGame[] {
  return games.map((g) => ({
    id: g.id,
    league: g.league,
    awayAbbrev: g.away.abbrev,
    homeAbbrev: g.home.abbrev,
    statusLabel: g.status_label,
    status: g.status,
    awayScore: g.away.score,
    homeScore: g.home.score,
  }));
}
```

Update all `TickerGame` fixtures in tests with `awayScore` / `homeScore`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/components/home/mapScoreboard.test.ts src/components/home/LiveTicker.test.tsx src/layouts/HomeChromeLayout.test.tsx`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/home/types.ts frontend/src/components/home/mapScoreboard.ts frontend/src/components/home/mapScoreboard.test.ts frontend/src/components/home/LiveTicker.test.tsx frontend/src/layouts/HomeChromeLayout.test.tsx
git commit -m "Add scores to ticker game mapping."
```

---

### Task 2: Ticker item format (live scores vs scheduled @)

**Files:**
- Modify: `frontend/src/components/home/LiveTicker.tsx`
- Modify: `frontend/src/components/home/LiveTicker.test.tsx`

**Interfaces:**
- Consumes: `TickerGame` with scores from Task 1
- Produces: visible format per status as in Global Constraints

- [ ] **Step 1: Write failing format tests**

```ts
const liveGame: TickerGame = {
  id: "g1",
  league: "wnba",
  awayAbbrev: "ATL",
  homeAbbrev: "DAL",
  statusLabel: "Q3 7:13",
  status: "live",
  awayScore: 36,
  homeScore: 44,
};

const scheduledGame: TickerGame = {
  id: "g2",
  league: "wnba",
  awayAbbrev: "NYL",
  homeAbbrev: "LVA",
  statusLabel: "7:00 PM ET",
  status: "scheduled",
  awayScore: null,
  homeScore: null,
};

it("formats live games with scores and an em dash", () => {
  render(<LiveTicker games={[liveGame]} />);
  expect(screen.getByText("36")).toBeInTheDocument();
  expect(screen.getByText("44")).toBeInTheDocument();
  expect(screen.getByText("—")).toBeInTheDocument();
  expect(screen.queryByText("@")).not.toBeInTheDocument();
});

it("formats scheduled games with @ and no scores", () => {
  render(<LiveTicker games={[scheduledGame]} />);
  expect(screen.getByText("@")).toBeInTheDocument();
  expect(screen.queryByText("—")).not.toBeInTheDocument();
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/components/home/LiveTicker.test.tsx`
Expected: FAIL — live still shows `@`, no score text

- [ ] **Step 3: Minimal item formatting**

In `LiveTicker.tsx`, for each game:

- If `status === "scheduled"` OR both scores null: render `away @ home statusLabel`
- Else: render `away [score] — home [score] statusLabel` (omit null score tokens)

Keep empty/error copy unchanged. Defer marquee polish to Task 3 if needed; formatting can ship in current list structure first.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/components/home/LiveTicker.test.tsx`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/home/LiveTicker.tsx frontend/src/components/home/LiveTicker.test.tsx
git commit -m "Format ticker items with scores or @ matchup."
```

---

### Task 3: CSS marquee layout + motion

**Files:**
- Modify: `frontend/src/index.css` (keyframes + reduced-motion)
- Modify: `frontend/src/components/home/LiveTicker.tsx`
- Modify: `frontend/src/components/home/LiveTicker.test.tsx`

**Interfaces:**
- Consumes: formatted items from Task 2
- Produces: full-bleed strip; fixed LIVE; animated duplicated track; hover pause; reduced-motion static

- [ ] **Step 1: Write failing structure tests**

```ts
it("duplicates the game list for the marquee track", () => {
  render(<LiveTicker games={[liveGame]} />);
  expect(screen.getAllByText("ATL")).toHaveLength(2);
});

it("marks the duplicate track as aria-hidden", () => {
  const { container } = render(<LiveTicker games={[liveGame]} />);
  expect(container.querySelector("[aria-hidden='true']")).toBeTruthy();
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/components/home/LiveTicker.test.tsx`
Expected: FAIL — only one ATL

- [ ] **Step 3: Implement marquee UI**

`index.css`:

```css
@keyframes ticker-marquee {
  from {
    transform: translateX(0);
  }
  to {
    transform: translateX(-50%);
  }
}

.ticker-marquee-track {
  animation: ticker-marquee 40s linear infinite;
}

.ticker-marquee:hover .ticker-marquee-track {
  animation-play-state: paused;
}

@media (prefers-reduced-motion: reduce) {
  .ticker-marquee-track {
    animation: none;
  }
}
```

`LiveTicker.tsx` structure:

- Outer: `border-b border-white/10 bg-[#0a0a0a]`
- Inner row: flex, full width, `px-4 py-2`, fixed LIVE rail + content
- When games exist: viewport `overflow-hidden` with class `ticker-marquee`; inner flex track `ticker-marquee-track` containing two copies of the item list (second `aria-hidden`)
- Items: `font-mono text-xs`, vertical dividers (`border-l border-white/10`), padding, two-tone abbrevs
- Empty/error: static muted text (no marquee classes)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/components/home/LiveTicker.test.tsx src/layouts/HomeChromeLayout.test.tsx`
Expected: PASS (layout test may see duplicate ATL — use `getAllByText` or keep asserting presence)

Update `HomeChromeLayout.test.tsx` if needed:

```ts
expect(screen.getAllByText("ATL").length).toBeGreaterThanOrEqual(1);
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/index.css frontend/src/components/home/LiveTicker.tsx frontend/src/components/home/LiveTicker.test.tsx frontend/src/layouts/HomeChromeLayout.test.tsx
git commit -m "Add CSS marquee motion to the live ticker."
```

---

## Spec coverage

| Spec requirement | Task |
| --- | --- |
| Score / `@` item format | 2 |
| `TickerGame` scores + mapper | 1 |
| CSS marquee duplicate track | 3 |
| Hover pause + reduced motion | 3 |
| Fixed LIVE + full-bleed + dividers | 3 |
| Empty/error unchanged | 2 (preserve) |
| Fixture updates | 1, 3 |
| Out of scope respected | — |

## Self-review

- No TBD placeholders
- Types consistent: `awayScore` / `homeScore` throughout
- Commits after each task per plan
