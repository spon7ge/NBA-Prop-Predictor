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

## Visual language (locked)

New UI (`/lab` and all future Tailwind/shadcn surfaces) follows these patterns. Theme tokens in `src/index.css` and Lab layout should express them; the legacy dashboard is unchanged until a later migration.

| Pattern | Usage |
|---------|--------|
| Borders | Subtle dark borders separate sections without being heavy |
| Spacing | Generous vertical spacing creates breathing room |
| Typography | Bold headings, regular body text, muted secondary text |
| Color accents | League badges provide the only color (besides white/gray) |
| Cards / containers | Subtle boxes with rounded corners group related information |
| External links / CTAs | Consistent arrow icons (lucide) for CTAs |

Implications for this foundation pass:

- **Palette:** Neutral gray/white base via shadcn `neutral`; no purple gradients, no decorative glow. Chart series on Lab use muted grays (and at most one demo “badge” accent color to show the league-badge pattern).
- **Borders:** Prefer `border` + soft neutral border color over heavy dividers or shadows.
- **Spacing:** Lab sections use generous vertical rhythm (e.g. `space-y-8` / `gap-8` scale), not tight dashboard packing.
- **Type:** Headings `font-semibold`/`bold`; body default; secondary labels `text-muted-foreground`.
- **Cards:** shadcn `Card` with rounded corners and light border; avoid multi-layer shadows.
- **CTAs:** Primary text links/buttons that navigate or act pair with a small lucide arrow (`ArrowUpRight` / `ArrowRight`) for consistency.

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

Single page that proves the stack is ready **and** demonstrates the visual language:

1. Bold page heading + muted subtitle (Geist)
2. shadcn `Card` sections with subtle borders and rounded corners; generous vertical spacing between sections
3. Demo “league badge” accent (small colored chip) — only chromatic element aside from neutrals
4. shadcn `Button` + CTA with lucide arrow icon
5. TanStack Query demo (`useQuery` with a trivial static/local fetcher)
6. Recharts sample chart (muted palette)
7. Small D3 SVG sample
8. Link back to `/` with arrow icon

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
- Visual language patterns encoded in theme + Lab (borders, spacing, type hierarchy, badge accent, cards, arrow CTAs)
- shadcn Button usable via `@/components/ui/button`
- Router wired with `/`, `/lab`, and 404
- Recharts + D3 demo components present for copy-paste into future features
- Existing dashboard continues to work on `/`

## Implementation notes (for planning)

1. Prefer official shadcn + Tailwind v4 init paths (`npx shadcn@latest init`, then `add button` + `card`).
2. Ensure `@tailwindcss/vite` is registered in `vite.config.ts` ahead of or alongside React plugin per current Tailwind v4 docs.
3. Import `src/index.css` from `main.tsx`; keep `slate.css` imported only where the dashboard needs it (today: App or existing entry path).
4. Map visual-language tokens in `src/index.css` (border, muted foreground, radius, spacing defaults) so Lab and future pages share one look.
5. Update `frontend/README.md` in the same change set: stack list, `/lab` route, visual-language summary, and note that `slate.css` remains for the dashboard.
