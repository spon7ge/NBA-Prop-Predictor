# Frontend tech stack foundation

**Date:** 2026-07-29  
**Status:** Approved for planning  
**App:** `frontend/` (HoopVista)  
**Approach:** Layered foundation (Approach A) — upgrade in place; keep existing dashboard working

## Goal

Stand up the agreed frontend stack inside the existing Vite app so future product work can use Tailwind v4, shadcn/ui, React Router, chart libraries, and Geist/lucide without another bootstrap pass. Existing dashboard behavior on `/` must keep working.

## Decisions (locked)

| Choice | Value |
|--------|-------|
| Location | `frontend/` at repo root |
| Scope | Foundation: tooling, theme, layout shell, example route + charts + Query demo |
| Migration | Upgrade in place — keep existing screens working |
| CSS strategy | Tailwind-first; leave `slate.css` for legacy dashboard; retire later (not this pass) |
| Scaffold approach | Layered foundation — packages + `/lab` showcase |
| Existing stack kept | React 19, TypeScript, Vite 6, TanStack Query, `@/` alias, `/api` proxy |

## Stack

| Layer | Choice | Notes |
|-------|--------|-------|
| UI runtime | React 19 + TypeScript | Already present |
| Bundler | Vite 6 + `@vitejs/plugin-react` | Already present; add `@tailwindcss/vite` |
| Styling | Tailwind CSS v4 | Via Vite plugin; tokens in `src/index.css` |
| Components | shadcn/ui | `new-york` style, `neutral` base; primitives this pass: Button + Card |
| Routing | React Router (`react-router-dom`) | BrowserRouter |
| Data fetching | TanStack Query | Existing `queryClient` retained |
| Charts | Recharts + D3.js | Demo components on `/lab` only this pass |
| Typography | Geist (sans + mono) via `geist` npm package | Wired as default font in CSS theme |
| Icons | lucide-react | Used on Lab + available app-wide |

Helpers for shadcn: `class-variance-authority`, `clsx`, `tailwind-merge`, `src/lib/utils.ts` (`cn()`).

## Architecture

```text
main.tsx
  └─ QueryClientProvider (existing)
       └─ BrowserRouter
            └─ AppRouter
                 ├─ /      → existing App (dashboard; slate.css)
                 ├─ /lab   → LabPage (Tailwind + shadcn + charts + Query demo)
                 └─ *      → NotFound
```

- Legacy dashboard code stays in `src/components/*`, `src/lib/*`, `src/styles/slate.css`.
- New stack surface: `src/components/ui/`, `src/pages/`, `src/charts/`, `src/index.css`.
- Vite `resolve.alias["@"]`, `server.proxy["/api"]`, and `server.fs.allow` for repo-root `data/` remain unchanged.

## Target structure (additions)

```text
frontend/
  components.json
  src/
    index.css                 # Tailwind + theme tokens + Geist
    lib/utils.ts              # cn()
    AppRouter.tsx             # route table
    pages/
      LabPage.tsx
      NotFoundPage.tsx
    components/ui/
      button.tsx              # shadcn
      card.tsx                # shadcn — Lab section shells
    charts/
      SampleRecharts.tsx
      SampleD3.tsx
```

Existing `App.tsx` becomes the element rendered at `/` (no product rewrite).

## Lab showcase (`/lab`)

Single page that proves the stack is ready:

1. Geist typography + lucide icon
2. shadcn `Button` + `Card` section shells
3. TanStack Query demo (`useQuery` with a trivial static/local fetcher)
4. Recharts sample chart
5. Small D3 SVG sample
6. Link back to `/`

Lab must not break or restyle the dashboard.

## Data & error handling

| Concern | Behavior |
|---------|----------|
| Dashboard data | Existing queries (`queries.ts`, backend/data JSON) unchanged |
| Lab Query | Isolated demo query key; no dependency on slate/props APIs |
| Unknown routes | Simple NotFound page |
| Chart failures | Soft empty state inside Lab samples only |

## Out of scope

- Restyling Top Legs / All Players / TicketStrip to Tailwind
- Migrating or deleting `slate.css`
- Dark-mode toggle
- Auth, additional routes, or product features
- Installing every shadcn component (add more as product needs them)
- Changing backend or data pipelines

## Verification

1. `npm install` in `frontend/` succeeds with new deps
2. `npm run build` (`tsc -b && vite build`) succeeds
3. `npm run dev`: `/` shows existing dashboard; `/lab` shows showcase
4. No regressions to Vite proxy or `@/` imports

## Success criteria

- Listed packages installed and importable
- Tailwind + Geist theme live for new UI
- shadcn Button usable via `@/components/ui/button`
- Router wired with `/`, `/lab`, and 404
- Recharts + D3 demo components present for copy-paste into future features
- Existing dashboard continues to work on `/`

## Implementation notes (for planning)

1. Prefer official shadcn + Tailwind v4 init paths (`npx shadcn@latest init`, then `add button`).
2. Ensure `@tailwindcss/vite` is registered in `vite.config.ts` ahead of or alongside React plugin per current Tailwind v4 docs.
3. Import `src/index.css` from `main.tsx`; keep `slate.css` imported only where the dashboard needs it (today: App or existing entry path).
4. Update `frontend/README.md` in the same change set: stack list, `/lab` route, and note that `slate.css` remains for the dashboard.
