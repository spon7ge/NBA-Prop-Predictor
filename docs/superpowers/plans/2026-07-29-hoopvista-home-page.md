# HoopVista Home Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a dark ticket-style HoopVista landing at `/` matching the approved mockup structure, and move the existing slate dashboard to `/slate`.

**Architecture:** New presentational home components under `frontend/src/components/home/` composed by `HomePage`. Router swaps `/` → home and `/slate` → existing `App`. No live APIs in v1; ticker empty copy + three LIVE NOW skeleton cards. Types in `types.ts` define future game shapes.

**Tech Stack:** React 19 · TypeScript · Vite 6 · Tailwind CSS v4 · React Router · lucide-react · Geist

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-29-hoopvista-home-page-design.md`
- Brand: HoopVista / `hoopvista` (not boxseats)
- Leagues in nav: NBA + WNBA only
- Live data: empty ticker (“No live games”); LIVE NOW shows `0 games in progress` + 3 skeleton cards
- CTAs: “See what’s inside” → `#live-now`; “Learn the games” → `/slate`
- Fonts: existing Geist Sans + Geist Mono only
- No new npm deps; no API calls on home
- Do not restyle slate Header/Footer beyond route move
- Verify with `npm run build` in `frontend/` (no unit test runner)

---

## File Structure

| File | Responsibility |
|------|----------------|
| `frontend/src/components/home/types.ts` | `TickerGame`, `LiveGame` types |
| `frontend/src/components/home/HomeNav.tsx` | Top nav |
| `frontend/src/components/home/LiveTicker.tsx` | LIVE strip + empty state |
| `frontend/src/components/home/TicketHero.tsx` | Ticket hero card |
| `frontend/src/components/home/LiveNowSection.tsx` | LIVE NOW + skeleton cards |
| `frontend/src/pages/HomePage.tsx` | Compose landing |
| `frontend/src/AppRouter.tsx` | `/` → HomePage, `/slate` → App |
| `frontend/src/pages/LabPage.tsx` | Point “Open dashboard” to `/slate` |

---

### Task 1: Types + home components + HomePage

**Files:**
- Create: `frontend/src/components/home/types.ts`
- Create: `frontend/src/components/home/HomeNav.tsx`
- Create: `frontend/src/components/home/LiveTicker.tsx`
- Create: `frontend/src/components/home/TicketHero.tsx`
- Create: `frontend/src/components/home/LiveNowSection.tsx`
- Create: `frontend/src/pages/HomePage.tsx`

**Interfaces:**
- Produces: `TickerGame`, `LiveGame`; exported components `HomeNav`, `LiveTicker`, `TicketHero`, `LiveNowSection`, `HomePage`

- [ ] **Step 1: Add `types.ts`**

```ts
export type HomeLeague = "nba" | "wnba";

export type TickerGame = {
  id: string;
  league: HomeLeague;
  awayAbbrev: string;
  homeAbbrev: string;
  statusLabel: string;
};

export type LiveGameTeam = {
  abbrev: string;
  name: string;
  score: number | null;
};

export type LiveGame = {
  id: string;
  league: HomeLeague;
  statusLabel: string;
  away: LiveGameTeam;
  home: LiveGameTeam;
};
```

- [ ] **Step 2: Implement `HomeNav`, `LiveTicker`, `TicketHero`, `LiveNowSection` per spec**

HomeNav: logo left; NBA/WNBA `#live-now` links with colored dots; About `#`; settings gear button (`aria-label="Settings"`).

LiveTicker: pulsing LIVE dot; if `games` empty/undefined → “No live games”.

TicketHero: SEASON PASS, hoopvista + BETA, tagline, blurb, primary scroll CTA, secondary `Link` to `/slate`, decorative icon ring, stub footer.

LiveNowSection: `id="live-now"`; header LIVE NOW + count; if no games, render exactly three skeleton cards.

- [ ] **Step 3: Compose `HomePage`**

Black full-page shell stacking Nav → Ticker → Hero → LiveNow (no games props).

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/home frontend/src/pages/HomePage.tsx
git commit -m "Add HoopVista ticket-style home page components."
```

---

### Task 2: Wire routes + Lab link + verify build

**Files:**
- Modify: `frontend/src/AppRouter.tsx`
- Modify: `frontend/src/pages/LabPage.tsx`
- Verify: `frontend/` build

**Interfaces:**
- Consumes: `HomePage` from Task 1; existing `App`
- Produces: `/` → home, `/slate` → slate

- [ ] **Step 1: Update `AppRouter`**

```tsx
import { Routes, Route } from "react-router-dom";
import App from "@/App";
import { HomePage } from "@/pages/HomePage";
import { LabPage } from "@/pages/LabPage";
import { NotFoundPage } from "@/pages/NotFoundPage";

export function AppRouter() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/slate" element={<App />} />
      <Route path="/lab" element={<LabPage />} />
      <Route path="*" element={<NotFoundPage />} />
    </Routes>
  );
}
```

- [ ] **Step 2: Change LabPage dashboard `Link` `to="/slate"`**

- [ ] **Step 3: Run build**

```bash
cd frontend && npm run build
```

Expected: exit 0.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/AppRouter.tsx frontend/src/pages/LabPage.tsx
git commit -m "Route home landing to / and move slate to /slate."
```

---

## Self-review

1. Spec coverage: routing, branding, nav leagues, empty ticker, skeleton LIVE NOW, CTAs, fonts, out-of-scope respected — covered in Tasks 1–2.
2. No placeholders.
3. Types consistent across components.
