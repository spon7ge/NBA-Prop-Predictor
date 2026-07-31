# Prop Picks Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the Explore subnav pill from “HoopVista Picks” to “Prop Picks”.

**Architecture:** One-string change in `LeagueSubnav`’s `exploreItems` array plus matching test assertion. No routing or enablement changes.

**Tech Stack:** React, TypeScript, Vitest, Testing Library

## Global Constraints

- New copy: `Prop Picks`
- Still disabled / non-navigating
- Scope: `LeagueSubnav.tsx` + `LeagueSubnav.test.tsx` only
- Do not rewrite historical plans/specs

## File structure

| File | Responsibility |
| --- | --- |
| `frontend/src/components/league/LeagueSubnav.tsx` | Pill label string |
| `frontend/src/components/league/LeagueSubnav.test.tsx` | Disabled-button name |

---

### Task 1: Rename pill to Prop Picks

**Files:**
- Modify: `frontend/src/components/league/LeagueSubnav.tsx`
- Modify: `frontend/src/components/league/LeagueSubnav.test.tsx`

**Interfaces:**
- Produces: disabled button accessible name `Prop Picks`

- [ ] **Step 1: Write the failing test**

In `LeagueSubnav.test.tsx`, change the disabled-button assertion:

```ts
expect(
  screen.getByRole("button", { name: "Prop Picks" }),
).toBeDisabled();
```

Remove any assertion that looks up `"HoopVista Picks"`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/components/league/LeagueSubnav.test.tsx`

Expected: FAIL — unable to find button named “Prop Picks”

- [ ] **Step 3: Update the label**

In `LeagueSubnav.tsx`, in `exploreItems`:

```ts
const exploreItems = [
  "Matchups",
  "Prop Picks",
  "Leaders",
  "Standings",
  "Playoff race",
  "Clutch",
] as const;
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/components/league/LeagueSubnav.test.tsx`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/league/LeagueSubnav.tsx frontend/src/components/league/LeagueSubnav.test.tsx
git commit -m "fix: rename HoopVista Picks subnav pill to Prop Picks"
```

---

## Spec coverage check

| Spec requirement | Task |
| --- | --- |
| Copy `Prop Picks` | Task 1 |
| Still disabled | Task 1 |
| Subnav + test only | Task 1 |
