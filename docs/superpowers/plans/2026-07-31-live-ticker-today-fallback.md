# Live Ticker TODAY Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When no games are live, switch the ticker rail to **TODAY** and marquee today’s scheduled (then final) games instead of showing “No live games.”

**Architecture:** Keep full-day `tickerGames` flowing through `HomeChromeLayout`. All mode logic lives in `LiveTicker`: prefer in-progress for **LIVE**; otherwise build a scheduled-then-final list for **TODAY**; empty slate keeps existing empty/error copy. Restore scheduled `@` chip format inside `TickerItem`.

**Tech Stack:** React, TypeScript, Tailwind v4, Vitest, Testing Library

## Global Constraints

- Live mode: any `live` / `halftime` → **LIVE** rail + in-progress only
- Idle mode: else any `scheduled` / `final` → **TODAY** rail + scheduled first, then finals
- Empty `games`: keep **LIVE** chrome; “No live games” / “Scoreboard unavailable” when `isError`
- Scheduled format: `ATL @ DAL 7:00 PM ET` (no scores)
- Live/final format: `ATL 36 — DAL 44 Q3 7:13` (omit null scores)
- TODAY rail: muted `text-white/50`, static muted dot, no red pulse
- No backend, mapper, hook, or layout changes
- Preserve API order within each status group
- Out of scope: tip-off re-sort, NBA ticker, hiding strip, per-team colors

## File structure

| File | Responsibility |
| --- | --- |
| `frontend/src/components/home/LiveTicker.tsx` | Mode selection, rail label, item format, marquee |
| `frontend/src/components/home/LiveTicker.test.tsx` | LIVE vs TODAY, fallback order, empty/error, `@` format |

---

### Task 1: Mode selection + TODAY rail

**Files:**
- Modify: `frontend/src/components/home/LiveTicker.tsx`
- Modify: `frontend/src/components/home/LiveTicker.test.tsx`

**Interfaces:**
- Consumes: `TickerGame` (`status`, scores, abbrevs, `statusLabel`); `isInProgressStatus`
- Produces: `LiveTicker` chooses `displayGames` + `mode: "live" | "today" | "empty"`

- [ ] **Step 1: Write the failing tests**

In `LiveTicker.test.tsx`, replace the test that expects empty copy for scheduled-only with TODAY behavior, and add finals / mixed / rail cases. Keep existing empty/error and live-format tests.

Add this fixture next to `scheduledGame`:

```ts
const finalGame: TickerGame = {
  id: "g3",
  league: "wnba",
  awayAbbrev: "CHI",
  homeAbbrev: "MIN",
  statusLabel: "Final",
  status: "final",
  awayScore: 78,
  homeScore: 82,
};
```

Replace:

```ts
it("shows empty copy when only non-live games are provided", () => {
  render(<LiveTicker games={[scheduledGame]} />);
  expect(screen.getByText("No live games")).toBeInTheDocument();
});
```

with:

```ts
it("shows TODAY rail and scheduled games when none are live", () => {
  render(<LiveTicker games={[scheduledGame]} />);
  expect(screen.getByText("Today")).toBeInTheDocument();
  expect(screen.queryByText("No live games")).not.toBeInTheDocument();
  expect(screen.getAllByText("NYL").length).toBeGreaterThanOrEqual(1);
  expect(screen.getAllByText("7:00 PM ET").length).toBeGreaterThanOrEqual(1);
});

it("shows TODAY rail and finals when the slate is finished", () => {
  render(<LiveTicker games={[finalGame]} />);
  expect(screen.getByText("Today")).toBeInTheDocument();
  expect(screen.getAllByText("CHI").length).toBeGreaterThanOrEqual(1);
  expect(screen.getAllByText("78").length).toBeGreaterThanOrEqual(1);
  expect(screen.getAllByText("Final").length).toBeGreaterThanOrEqual(1);
});

it("orders scheduled before finals in TODAY mode", () => {
  const { container } = render(
    <LiveTicker games={[finalGame, scheduledGame]} />,
  );
  const track = container.querySelector(".ticker-marquee-track");
  const text = track?.textContent ?? "";
  expect(text.indexOf("NYL")).toBeGreaterThanOrEqual(0);
  expect(text.indexOf("NYL")).toBeLessThan(text.indexOf("CHI"));
});

it("keeps LIVE rail and hides scheduled when any game is live", () => {
  render(<LiveTicker games={[scheduledGame, liveGame]} />);
  expect(screen.getByText("Live")).toBeInTheDocument();
  expect(screen.queryByText("Today")).not.toBeInTheDocument();
  expect(screen.queryByText("NYL")).not.toBeInTheDocument();
  expect(screen.getAllByText("ATL").length).toBeGreaterThanOrEqual(1);
});
```

Leave `it("hides scheduled and final games", …)` as-is (still valid for LIVE mode with `scheduledGame + liveGame`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npx vitest run src/components/home/LiveTicker.test.tsx`

Expected: FAIL — scheduled-only still shows “No live games”; no “Today” text

- [ ] **Step 3: Implement mode selection + rail**

In `LiveTicker.tsx`, replace the body of `LiveTicker` (keep `TickerItem` / `TickerGameList` for now):

```tsx
export function LiveTicker({ games = [], isError = false }: LiveTickerProps) {
  const liveGames = games.filter((g) => isInProgressStatus(g.status));
  const scheduledGames = games.filter((g) => g.status === "scheduled");
  const finalGames = games.filter((g) => g.status === "final");

  const mode: "live" | "today" | "empty" =
    liveGames.length > 0
      ? "live"
      : scheduledGames.length > 0 || finalGames.length > 0
        ? "today"
        : "empty";

  const displayGames =
    mode === "live"
      ? liveGames
      : mode === "today"
        ? [...scheduledGames, ...finalGames]
        : [];

  const isToday = mode === "today";

  return (
    <div className="ticker-marquee border-b border-white/10 bg-[#0a0a0a]">
      <div className="mx-auto flex max-w-6xl items-center gap-4 overflow-hidden px-4 py-2 sm:px-6">
        <div className="flex shrink-0 items-center gap-2">
          <span
            className={
              isToday
                ? "size-1.5 rounded-full bg-white/40"
                : "size-1.5 animate-pulse rounded-full bg-red-500"
            }
            aria-hidden
          />
          <span
            className={
              isToday
                ? "text-[10px] font-semibold tracking-widest text-white/50 uppercase"
                : "text-[10px] font-semibold tracking-widest text-red-400 uppercase"
            }
          >
            {isToday ? "Today" : "Live"}
          </span>
        </div>

        {mode === "empty" ? (
          <p className="truncate text-xs text-white/40">
            {isError ? "Scoreboard unavailable" : "No live games"}
          </p>
        ) : (
          <div className="ticker-marquee-viewport min-w-0 flex-1 overflow-hidden">
            <div className="ticker-marquee-track flex w-max items-center">
              <TickerGameList games={displayGames} keyPrefix="a" />
              <div className="ticker-marquee-duplicate" aria-hidden="true">
                <TickerGameList
                  games={displayGames}
                  keyPrefix="b"
                  interactive={false}
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/components/home/LiveTicker.test.tsx`

Expected: PASS for mode/rail tests. Scheduled chips may still show `—` instead of `@` until Task 2.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/home/LiveTicker.tsx frontend/src/components/home/LiveTicker.test.tsx
git commit -m "feat: fall back live ticker to TODAY slate when idle"
```

---

### Task 2: Scheduled `@` item format

**Files:**
- Modify: `frontend/src/components/home/LiveTicker.tsx`
- Modify: `frontend/src/components/home/LiveTicker.test.tsx`

**Interfaces:**
- Consumes: `TickerGame.status === "scheduled"`
- Produces: scheduled `TickerItem` content uses `@` and omits scores

- [ ] **Step 1: Write the failing format test**

Add to `LiveTicker.test.tsx`:

```ts
it("formats scheduled games with @ and no scores", () => {
  render(<LiveTicker games={[scheduledGame]} />);
  expect(screen.getAllByText("@").length).toBeGreaterThanOrEqual(1);
  expect(screen.queryByText("—")).not.toBeInTheDocument();
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/components/home/LiveTicker.test.tsx`

Expected: FAIL — scheduled items still render `—`

- [ ] **Step 3: Implement scheduled branch in `TickerItem`**

Replace the `content` block in `TickerItem` with:

```tsx
const isScheduled = game.status === "scheduled";

const content = isScheduled ? (
  <>
    <span className="font-medium text-sky-400">{game.awayAbbrev}</span>
    <span className="text-white/30">@</span>
    <span className="font-medium text-rose-400">{game.homeAbbrev}</span>
    <span className="text-white/40">{game.statusLabel}</span>
  </>
) : (
  <>
    <span className="font-medium text-sky-400">{game.awayAbbrev}</span>
    {game.awayScore !== null ? (
      <span className="text-white/80">{game.awayScore}</span>
    ) : null}
    <span className="text-white/30">—</span>
    <span className="font-medium text-rose-400">{game.homeAbbrev}</span>
    {game.homeScore !== null ? (
      <span className="text-white/80">{game.homeScore}</span>
    ) : null}
    <span className="text-white/40">{game.statusLabel}</span>
  </>
);
```

- [ ] **Step 4: Run full ticker suite**

Run: `cd frontend && npx vitest run src/components/home/LiveTicker.test.tsx`

Expected: PASS — all mode, format, marquee, and link tests green

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/home/LiveTicker.tsx frontend/src/components/home/LiveTicker.test.tsx
git commit -m "feat: format scheduled ticker chips with @"
```

---

## Spec coverage check

| Spec requirement | Task |
| --- | --- |
| LIVE mode in-progress only | Task 1 |
| TODAY mode scheduled then finals | Task 1 |
| Empty/error copy + LIVE chrome | Task 1 |
| TODAY muted rail / no pulse | Task 1 |
| Scheduled `@` format | Task 2 |
| Live/final score `—` format | Task 1 (unchanged) + Task 2 (non-scheduled branch) |
| Marquee / a11y duplicate track | Unchanged behavior covered by existing tests |
| No backend/hook/layout changes | Both tasks |
