# Frontend Home-Only Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the frontend to the HoopVista home landing (`/`) plus NotFound, deleting slate UI, Lab, and all unused libs/deps.

**Architecture:** Surgical delete. First retarget home + router so nothing imports deleted modules, then remove files, then uninstall unused packages and verify `npm run build`.

**Tech Stack:** React 19 · TypeScript · Vite 6 · Tailwind CSS v4 · React Router · lucide-react · Geist

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-29-frontend-home-only-cleanup-design.md`
- Live routes: `/` (HomePage) and `*` (NotFoundPage) only
- Delete slate product + Lab + slate/API `lib` + `types` (do not archive)
- Remove hero “Learn the games”; remove HomeNav Slate/Lab links
- Uninstall `@tanstack/react-query`, `recharts`, `d3`, `@types/d3` when unused
- Verify with `cd frontend && npm run build` (no unit test runner)
- Do not change backend or data pipelines

---

## File Structure (end state)

| Path | Responsibility |
|------|----------------|
| `frontend/src/pages/HomePage.tsx` | Landing |
| `frontend/src/components/home/*` | Home sections |
| `frontend/src/pages/NotFoundPage.tsx` | 404 |
| `frontend/src/AppRouter.tsx` | `/` + `*` |
| `frontend/src/main.tsx` | Mount without Query / slate.css |
| `frontend/src/index.css` | Theme |
| `frontend/src/lib/utils.ts` | Keep only if still imported; else delete with shadcn |

---

### Task 1: Retarget home + slim router/main

**Files:**
- Modify: `frontend/src/components/home/HomeNav.tsx`
- Modify: `frontend/src/components/home/TicketHero.tsx`
- Modify: `frontend/src/AppRouter.tsx`
- Modify: `frontend/src/main.tsx`
- Modify: `frontend/src/pages/NotFoundPage.tsx` (confirm “Back to home”)

**Interfaces:**
- Consumes: `HomePage`, `NotFoundPage`
- Produces: app that no longer routes to `/slate` or `/lab`

- [ ] **Step 1: Update `HomeNav` — remove Slate/Lab links**

Right side should only keep the settings gear (and optional About only if still desired; spec removes Slate/Lab — current About was already replaced by Slate/Lab). Final right cluster:

```tsx
<button
  type="button"
  aria-label="Settings"
  className="rounded-md p-1.5 text-white/70 transition-colors hover:bg-white/5 hover:text-white"
>
  <Settings className="size-4" />
</button>
```

Keep logo `Link to="/"` and NBA/WNBA `#live-now` links.

- [ ] **Step 2: Update `TicketHero` — remove secondary CTA**

Delete the `Link` to `/slate` (“Learn the games”). Keep only:

```tsx
<a
  href="#live-now"
  className="inline-flex items-center gap-2 rounded-full bg-white px-5 py-2.5 text-sm font-semibold text-black no-underline transition-opacity hover:opacity-90"
>
  See what&apos;s inside
  <ArrowDown className="size-4" aria-hidden />
</a>
```

Remove unused `Link` import from `react-router-dom` if no longer needed.

- [ ] **Step 3: Replace `AppRouter.tsx`**

```tsx
import { Routes, Route } from "react-router-dom";
import { HomePage } from "@/pages/HomePage";
import { NotFoundPage } from "@/pages/NotFoundPage";

export function AppRouter() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="*" element={<NotFoundPage />} />
    </Routes>
  );
}
```

- [ ] **Step 4: Replace `main.tsx`**

```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { AppRouter } from "./AppRouter";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <AppRouter />
    </BrowserRouter>
  </StrictMode>,
);
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/home/HomeNav.tsx frontend/src/components/home/TicketHero.tsx frontend/src/AppRouter.tsx frontend/src/main.tsx frontend/src/pages/NotFoundPage.tsx
git commit -m "Point frontend shell at home-only routes."
```

---

### Task 2: Delete slate, Lab, and unused modules

**Files:**
- Delete all paths listed below
- Modify: `frontend/README.md` to describe home-only app

**Interfaces:**
- Consumes: Task 1 router/main (no imports into deleted files)
- Produces: tree with no slate/Lab sources

- [ ] **Step 1: Delete app / Lab / chart files**

```bash
cd frontend
rm -f src/App.tsx
rm -f src/pages/LabPage.tsx
rm -f src/charts/SampleD3.tsx src/charts/SampleRecharts.tsx
rm -f src/lib/labDemoQuery.ts src/lib/queryClient.ts
rmdir src/charts 2>/dev/null || true
```

- [ ] **Step 2: Delete slate UI components + CSS**

```bash
rm -f src/components/Header.tsx src/components/Footer.tsx
rm -f src/components/TopLegsView.tsx src/components/AllPlayersView.tsx src/components/ResultsView.tsx
rm -f src/components/PlayerBlock.tsx src/components/LegCard.tsx src/components/ParlayCard.tsx
rm -f src/components/TicketStrip.tsx src/components/Dropdown.tsx
rm -f src/components/LoadingSkeleton.tsx src/components/PlayerStatsPanel.tsx
rm -f src/components/HitRateByBookChart.tsx
rm -f src/styles/slate.css
rm -f public/slate.css
rmdir src/styles 2>/dev/null || true
```

- [ ] **Step 3: Delete slate/API libs + types**

```bash
rm -f src/lib/api.ts src/lib/backend.ts src/lib/queries.ts src/lib/slate.ts
rm -f src/lib/mapSlate.ts src/lib/mapLiveProps.ts src/lib/enrichSlateFromLiveProps.ts
rm -f src/lib/players.ts src/lib/legsRoi.ts src/lib/format.ts src/lib/constants.ts
rm -f src/types/api.ts src/types/slate.ts
rmdir src/types 2>/dev/null || true
```

- [ ] **Step 4: Delete unused shadcn if nothing imports them**

```bash
# From frontend/
rg -l 'components/ui/(button|card)|@/lib/utils' src || true
```

If only `button.tsx` / `card.tsx` / `utils.ts` reference each other (or nothing):

```bash
rm -f src/components/ui/button.tsx src/components/ui/card.tsx
rmdir src/components/ui 2>/dev/null || true
# Keep src/lib/utils.ts only if still imported; otherwise:
rm -f src/lib/utils.ts
```

- [ ] **Step 5: Update `frontend/README.md`**

Replace stack/dashboard/`/lab` wording with home-only: run `npm run dev`, open `http://localhost:5173/`, note Tailwind + Router + Geist, no slate dashboard.

- [ ] **Step 6: Commit**

```bash
git add -A frontend
git commit -m "Remove slate UI, Lab, and unused frontend modules."
```

Do not stage unrelated `data/` or root files.

---

### Task 3: Uninstall deps and verify build

**Files:**
- Modify: `frontend/package.json`, `frontend/package-lock.json`
- Optionally remove `frontend/components.json` if shadcn fully gone

**Interfaces:**
- Consumes: cleaned source tree
- Produces: lean dependencies + green build

- [ ] **Step 1: Uninstall unused packages**

```bash
cd frontend
npm uninstall @tanstack/react-query recharts d3
npm uninstall -D @types/d3
```

If Button/Card/`cn` were removed and nothing uses cva/clsx/tailwind-merge:

```bash
npm uninstall class-variance-authority clsx tailwind-merge
```

Keep `react-router-dom`, `lucide-react`, `geist`, Tailwind packages.

- [ ] **Step 2: Run build**

```bash
cd frontend && npm run build
```

Expected: exit 0. Fix any leftover imports if TypeScript errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/components.json
git commit -m "Drop unused frontend dependencies after home-only cleanup."
```

---

## Self-review

1. Spec coverage: routes, deletes, home CTA/nav, Query/charts uninstall, build success — Tasks 1–3.
2. No placeholders.
3. Delete commands match spec file list; README + public/slate.css included.
