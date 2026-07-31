# WNBA Standings Compact Columns Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compact the WNBA standings conference table so Away / L10 / Diff / Strk stay visible in the East|West side-by-side layout without a forced wide min-width scroll.

**Architecture:** Frontend-only tweak in `StandingsConferenceCard`: drop `min-w-[720px]`, shrink type to `text-xs`, tighten TEAM→stats padding, add `tabular-nums` on record/stat cells. Keep full team names and Diff/Strk color helpers. Extend `StandingsGrid` tests to assert the extra columns and absence of the old min-width class.

**Tech Stack:** React, TypeScript, Tailwind v4, Vitest, Testing Library

## Global Constraints

- Keep `lg:grid-cols-2` East | West; do not stack solely to fit columns
- Keep logo + colored abbrev + full name (no truncate)
- Remove forced `min-w-[720px]`; retain `overflow-x-auto` as last-resort safety
- Frontend card styling only — no API / schema changes
- Diff green (`+`) / red (`-`); Strk green (`W`) / red (`L`) unchanged
- Out of scope: stacking conferences, truncating names, widening `max-w-6xl`, backend, NBA

## File structure

| File | Responsibility |
| --- | --- |
| `frontend/src/components/league/StandingsConferenceCard.tsx` | Compact table layout |
| `frontend/src/components/league/StandingsGrid.test.tsx` | Assert columns + compact table class |

---

### Task 1: Compact standings table

**Files:**
- Modify: `frontend/src/components/league/StandingsConferenceCard.tsx`
- Modify: `frontend/src/components/league/StandingsGrid.test.tsx`

**Interfaces:**
- Consumes: `ApiWnbaStandingsConference` (unchanged)
- Produces: same visual columns; table class `w-full text-left text-xs` (no `min-w-[720px]`)

- [ ] **Step 1: Write the failing tests**

In `StandingsGrid.test.tsx`, extend the happy-path test and add a compact-layout assertion:

```ts
it("renders season, conferences, rows, and attribution", () => {
  render(
    <StandingsGrid season={2026} conferences={sample} />,
  );
  expect(screen.getByText("2026 regular season")).toBeInTheDocument();
  expect(screen.getByText("Eastern Conference")).toBeInTheDocument();
  expect(screen.getByText("Western Conference")).toBeInTheDocument();
  expect(screen.getByText("Indiana Fever")).toBeInTheDocument();
  expect(screen.getByText("IND")).toBeInTheDocument();
  expect(screen.getByText("18-10")).toBeInTheDocument();
  expect(screen.getByText("7-5")).toBeInTheDocument(); // away
  expect(screen.getByText("8-2")).toBeInTheDocument(); // l10 (appears in both confs)
  expect(screen.getByText("+169")).toBeInTheDocument();
  expect(screen.getByText("W4")).toBeInTheDocument();
  expect(screen.getByText("-12")).toBeInTheDocument();
  expect(screen.getByText("L2")).toBeInTheDocument();
  expect(screen.getByText("Data: ESPN")).toBeInTheDocument();
});

it("uses a compact table without a forced min width", () => {
  const { container } = render(
    <StandingsGrid season={2026} conferences={sample} />,
  );
  const table = container.querySelector("table");
  expect(table).not.toBeNull();
  expect(table?.className).toContain("text-xs");
  expect(table?.className).not.toContain("min-w-[720px]");
});
```

Note: East away is `7-5` and West away is `11-4`; East/West both have L10 `8-2`. Prefer asserting unique values (`7-5`, `+169`, `W4`, `-12`, `L2`) plus headers:

```ts
expect(screen.getByRole("columnheader", { name: "Away" })).toBeInTheDocument();
expect(screen.getByRole("columnheader", { name: "L10" })).toBeInTheDocument();
expect(screen.getByRole("columnheader", { name: "Diff" })).toBeInTheDocument();
expect(screen.getByRole("columnheader", { name: "Strk" })).toBeInTheDocument();
```

Because two conference tables render, use `getAllByRole` if needed:

```ts
expect(screen.getAllByRole("columnheader", { name: "Away" }).length).toBe(2);
expect(screen.getAllByRole("columnheader", { name: "L10" }).length).toBe(2);
expect(screen.getAllByRole("columnheader", { name: "Diff" }).length).toBe(2);
expect(screen.getAllByRole("columnheader", { name: "Strk" }).length).toBe(2);
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npx vitest run src/components/league/StandingsGrid.test.tsx`

Expected: FAIL on the new compact-table assertion (`text-xs` missing and/or `min-w-[720px]` still present). Column value assertions may already pass.

- [ ] **Step 3: Implement compact styles**

Replace the table markup in `StandingsConferenceCard.tsx` with:

```tsx
<div className="overflow-x-auto">
  <table className="w-full text-left text-xs">
    <thead>
      <tr className="text-[10px] tracking-wide text-white/40 uppercase">
        <th className="pb-2 pr-1.5 font-medium">#</th>
        <th className="pb-2 pr-1 font-medium">Team</th>
        <th className="pb-2 pr-1.5 font-medium">W-L</th>
        <th className="pb-2 pr-1.5 font-medium">PCT</th>
        <th className="pb-2 pr-1.5 font-medium">GB</th>
        <th className="pb-2 pr-1.5 font-medium">Home</th>
        <th className="pb-2 pr-1.5 font-medium">Away</th>
        <th className="pb-2 pr-1.5 font-medium">L10</th>
        <th className="pb-2 pr-1.5 font-medium">Diff</th>
        <th className="pb-2 font-medium">Strk</th>
      </tr>
    </thead>
    <tbody>
      {conference.teams.length === 0 ? (
        <tr>
          <td colSpan={10} className="py-3 text-white/40">
            No data
          </td>
        </tr>
      ) : (
        conference.teams.map((row) => (
          <tr key={`${conference.key}-${row.team_id}`}>
            <td className="py-1 pr-1.5 text-white/50">{row.rank}</td>
            <td className="py-1 pr-1">
              <div className="flex items-center gap-1.5">
                <TeamAbbrevAvatar
                  abbrev={row.abbrev}
                  logoUrl={row.logo_url}
                  sizeClassName="size-5"
                />
                <span
                  className="font-semibold"
                  style={{ color: teamColor(row.abbrev) }}
                >
                  {row.abbrev}
                </span>
                <span className="text-white/80">{row.name}</span>
              </div>
            </td>
            <td className="py-1 pr-1.5 tabular-nums text-white">{row.wl}</td>
            <td className="py-1 pr-1.5 tabular-nums text-white/70">{row.pct}</td>
            <td className="py-1 pr-1.5 tabular-nums text-white/70">{row.gb}</td>
            <td className="py-1 pr-1.5 tabular-nums text-white/70">{row.home}</td>
            <td className="py-1 pr-1.5 tabular-nums text-white/70">{row.away}</td>
            <td className="py-1 pr-1.5 tabular-nums text-white/70">{row.l10}</td>
            <td
              className={`py-1 pr-1.5 font-medium tabular-nums ${diffClass(row.diff)}`}
            >
              {row.diff}
            </td>
            <td className={`py-1 font-medium tabular-nums ${streakClass(row.streak)}`}>
              {row.streak}
            </td>
          </tr>
        ))
      )}
    </tbody>
  </table>
</div>
```

Leave `diffClass` / `streakClass` and the outer card chrome unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/components/league/StandingsGrid.test.tsx`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/league/StandingsConferenceCard.tsx frontend/src/components/league/StandingsGrid.test.tsx
git commit -m "fix: compact WNBA standings so all columns stay visible"
```

---

## Spec coverage check

| Spec requirement | Task |
| --- | --- |
| Drop `min-w-[720px]` | Task 1 |
| `text-xs` + tighter TEAM padding | Task 1 |
| Keep full team name | Task 1 |
| Keep overflow-x-auto safety | Task 1 |
| Assert Away/L10/Diff/Strk | Task 1 |
| No API changes | Task 1 |
