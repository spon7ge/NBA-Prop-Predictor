# WNBA Player Season Stats Box Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restyle player header averages into one charcoal “{season} REGULAR SEASON STATS” box with PTS · REB · AST · FG% · 3P% columns.

**Architecture:** Frontend-only change inside `PlayerHeader`: replace the five separate tiles with a single bordered box (header bar + equal columns). No API changes.

**Tech Stack:** React 19 · TypeScript · Tailwind 4 · Vitest · Testing Library

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-01-wnba-player-season-stats-box-design.md`
- Stats: PTS, REB, AST, FG%, 3P% only
- Header copy: `{season} REGULAR SEASON STATS` (uppercase)
- Charcoal hub colors (not bright blue / white card)
- Verify: `cd frontend && npm run test -- --run src/components/league/PlayerHeader.test.tsx && npm run build`

---

### Task 1: PlayerHeader season stats box

**Files:**
- Modify: `frontend/src/components/league/PlayerHeader.tsx`
- Modify: `frontend/src/components/league/PlayerHeader.test.tsx`

- [ ] **Step 1: Extend failing test**

Add assertion to the averages test (or new case):

```tsx
expect(
  screen.getByText("2026 REGULAR SEASON STATS"),
).toBeInTheDocument();
```

Keep existing PTS/REB/AST/FG%/3P% and value asserts.

- [ ] **Step 2: Run — expect FAIL**

`npm run test -- --run src/components/league/PlayerHeader.test.tsx`

- [ ] **Step 3: Implement box markup**

Replace the `grid grid-cols-5` tile map with:

```tsx
<div className="w-full shrink-0 overflow-hidden rounded-xl border border-white/10 md:max-w-md md:min-w-[18rem]">
  <div className="bg-white/10 px-3 py-2 text-center text-[11px] font-semibold tracking-wide text-white uppercase">
    {player.season} REGULAR SEASON STATS
  </div>
  <div className="grid grid-cols-5 gap-1 px-2 py-4 sm:gap-2 sm:px-3">
    {AVG_TILES.map(({ key, label }) => (
      <div key={key} className="text-center">
        <div className="text-[10px] font-medium tracking-wide text-white/40 uppercase">
          {label}
        </div>
        <div className="mt-1 text-lg font-semibold tabular-nums text-white sm:text-xl">
          {player.averages[key]}
        </div>
      </div>
    ))}
  </div>
</div>
```

- [ ] **Step 4: Run tests + build — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/league/PlayerHeader.tsx \
  frontend/src/components/league/PlayerHeader.test.tsx
git commit -m "$(cat <<'EOF'
feat: box WNBA player averages as regular season stats card

EOF
)"
```
