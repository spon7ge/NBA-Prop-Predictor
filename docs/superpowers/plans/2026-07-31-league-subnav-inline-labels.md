# League Subnav Inline Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put Explore and Learn labels on the same line as their pills, with a thin vertical divider between the two groups in `LeagueSubnav`.

**Architecture:** Layout-only change in `LeagueSubnav.tsx`: each section becomes a horizontal `flex items-center` group (label + pills). Learn group gets a left border divider. Existing link/disabled/active behavior stays unchanged. Extend `LeagueSubnav.test.tsx` to assert labels and divider presence.

**Tech Stack:** React, TypeScript, Tailwind v4, Vitest, Testing Library, React Router

## Global Constraints

- Flex row groups: label + pills, `items-center`
- Thin vertical divider between Explore and Learn
- `LeagueSubnav` layout only — no route / enablement changes
- Shared component — both WNBA and NBA
- Keep muted uppercase label styling; remove stacked `mb-2`
- Out of scope: new destinations, pill copy/order, LeagueHero

## File structure

| File | Responsibility |
| --- | --- |
| `frontend/src/components/league/LeagueSubnav.tsx` | Inline label + divider layout |
| `frontend/src/components/league/LeagueSubnav.test.tsx` | Labels, divider, existing nav asserts |

---

### Task 1: Inline Explore / Learn layout

**Files:**
- Modify: `frontend/src/components/league/LeagueSubnav.tsx`
- Modify: `frontend/src/components/league/LeagueSubnav.test.tsx`

**Interfaces:**
- Consumes: `LeagueSubnavProps` (`league: LeagueSlug`) — unchanged
- Produces: same navigation behavior; Explore/Learn labels inline with pills

- [ ] **Step 1: Write the failing tests**

Add to `LeagueSubnav.test.tsx`:

```ts
it("places Explore and Learn labels inline with a divider before Learn", () => {
  const { container } = render(
    <MemoryRouter initialEntries={["/wnba/matchups"]}>
      <LeagueSubnav league="wnba" />
    </MemoryRouter>,
  );
  expect(screen.getByText("Explore")).toBeInTheDocument();
  expect(screen.getByText("Learn")).toBeInTheDocument();
  const learnGroup = screen.getByText("Learn").closest("div");
  expect(learnGroup?.className).toMatch(/border-l/);
  // Labels are siblings of the pill row inside the same flex group
  expect(learnGroup?.className).toMatch(/items-center/);
  const exploreGroup = screen.getByText("Explore").closest("div");
  expect(exploreGroup?.className).toMatch(/items-center/);
  // Smoke: nav still works
  expect(screen.getByRole("link", { name: "Matchups" })).toHaveAttribute(
    "href",
    "/wnba/matchups",
  );
});
```

Keep the existing two tests as-is.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npx vitest run src/components/league/LeagueSubnav.test.tsx`

Expected: FAIL — Learn group lacks `border-l` / `items-center` (stacked layout still uses `mb-2` parent without those classes on the label wrapper).

- [ ] **Step 3: Implement inline layout**

Replace the return JSX of `LeagueSubnav` (keep `renderItem` and helpers unchanged):

```tsx
return (
  <nav
    aria-label={`${league.toUpperCase()} sections`}
    className="mx-auto max-w-6xl px-4 py-5 sm:px-6"
  >
    <div className="flex gap-6 overflow-x-auto rounded-2xl border border-white/10 bg-[#121212] px-4 py-3">
      <div className="flex shrink-0 items-center gap-2">
        <p className="px-1 text-[10px] font-semibold tracking-[0.18em] text-white/35 uppercase">
          Explore
        </p>
        <div className="flex gap-2">{exploreItems.map(renderItem)}</div>
      </div>
      <div className="flex shrink-0 items-center gap-2 border-l border-white/10 pl-6">
        <p className="px-1 text-[10px] font-semibold tracking-[0.18em] text-white/35 uppercase">
          Learn
        </p>
        <div className="flex gap-2">{learnItems.map(renderItem)}</div>
      </div>
    </div>
  </nav>
);
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/components/league/LeagueSubnav.test.tsx`

Expected: PASS (all three tests)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/league/LeagueSubnav.tsx frontend/src/components/league/LeagueSubnav.test.tsx
git commit -m "fix: put Explore and Learn labels inline with subnav pills"
```

---

## Spec coverage check

| Spec requirement | Task |
| --- | --- |
| Explore/Learn on same line as pills | Task 1 |
| Thin vertical divider | Task 1 |
| Navigation unchanged | Task 1 (existing tests + smoke assert) |
| Shared WNBA + NBA | Task 1 (shared component) |
