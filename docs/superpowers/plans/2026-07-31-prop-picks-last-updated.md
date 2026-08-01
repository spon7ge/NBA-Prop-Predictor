# Prop Picks Last Updated Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show “Last updated {date} {time}” next to the Prop Picks title using React Query `dataUpdatedAt`.

**Architecture:** `LeaguePropPicksPage` reads `dataUpdatedAt` from `useWnbaProps()` and passes it as `lastUpdatedAt` into `PropPicksTable`, which formats and renders muted meta beside the existing `h2`.

**Tech Stack:** React, React Query, Vitest, Testing Library, Tailwind (existing league UI tokens).

## Global Constraints

- Timestamp source: React Query `dataUpdatedAt` only (not API `as_of`)
- Copy: `Last updated {short local datetime}` e.g. `Jul 31, 11:54 PM`
- Style: `text-sm text-white/40`, inline next to title
- Hide when no successful fetch (`!lastUpdatedAt`)

---

## File map

| File | Responsibility |
| --- | --- |
| `frontend/src/components/league/PropPicksTable.tsx` | Accept `lastUpdatedAt?: number`; render header meta |
| `frontend/src/components/league/PropPicksTable.test.tsx` | Cover show/hide of last-updated text |
| `frontend/src/pages/LeaguePropPicksPage.tsx` | Pass `dataUpdatedAt` from `useWnbaProps()` |

---

### Task 1: Table header last-updated

**Files:**
- Modify: `frontend/src/components/league/PropPicksTable.test.tsx`
- Modify: `frontend/src/components/league/PropPicksTable.tsx`
- Modify: `frontend/src/pages/LeaguePropPicksPage.tsx`

- [x] **Step 1: Write failing tests**

Add tests that:
1. With `lastUpdatedAt={Date.parse("2026-07-31T23:54:00")}` (or fixed ms), assert text matching `/Last updated/i` is present and includes a formatted month/day/time.
2. Without `lastUpdatedAt`, assert no “Last updated” text.

- [x] **Step 2: Run tests — expect fail**

```bash
cd frontend && npx vitest run src/components/league/PropPicksTable.test.tsx
```

- [x] **Step 3: Implement minimal UI**

In `PropPicksTable`:
- Add optional `lastUpdatedAt?: number`
- Export a small `formatPropPicksUpdatedAt(ms: number): string` (or keep private) using `Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })`
- Replace lone `h2` with:

```tsx
<div className="flex flex-wrap items-baseline gap-3">
  <h2 className="text-lg font-semibold text-white">Prop Picks</h2>
  {lastUpdatedAt ? (
    <p className="text-sm text-white/40">
      Last updated {formatPropPicksUpdatedAt(lastUpdatedAt)}
    </p>
  ) : null}
</div>
```

In `LeaguePropPicksPage`:
- Destructure `dataUpdatedAt` from `useWnbaProps()`
- Pass `lastUpdatedAt={dataUpdatedAt || undefined}` to `PropPicksTable`

- [x] **Step 4: Run tests — expect pass**

```bash
cd frontend && npx vitest run src/components/league/PropPicksTable.test.tsx
```

- [ ] **Step 5: Commit** (only if user asks)

---
