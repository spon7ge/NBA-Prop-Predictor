# Frontend home-only cleanup design

Date: 2026-07-29  
Status: Approved for planning

## Goal

Strip the frontend down to the HoopVista home landing only. Remove the legacy slate product UI, Lab showcase, and all supporting code/deps that existed solely for those surfaces. Keep the Tailwind / Router / Geist foundation needed to run home.

## Decisions

| Topic | Choice |
| --- | --- |
| Live routes | Home (`/`) + NotFound (`*`) only |
| Slate product | Delete entirely (views, shell, CSS, mapping libs) |
| Lab | Delete entirely |
| Slate/API `lib` + `types` | Delete (not retained for later) |
| Approach | Surgical delete; no full `src/` re-scaffold |
| Hero secondary CTA | Remove “Learn the games” (no destination) |
| Query / Recharts / D3 | Uninstall if unused after delete |

## Keep

| Path | Role |
| --- | --- |
| `frontend/src/pages/HomePage.tsx` | Landing page |
| `frontend/src/components/home/*` | Nav, ticker, hero, LIVE NOW |
| `frontend/src/pages/NotFoundPage.tsx` | 404 → link home |
| `frontend/src/AppRouter.tsx` | `/` + `*` only |
| `frontend/src/main.tsx` | Minimal mount (no Query, no slate.css) |
| `frontend/src/index.css` | Tailwind + Geist tokens |
| `frontend/src/lib/utils.ts` | `cn()` if still needed by remaining UI |
| Vite / TS / Tailwind config | Unchanged except dep cleanup |

## Delete (representative)

**App / pages**

- `App.tsx`
- `pages/LabPage.tsx`
- `charts/SampleD3.tsx`, `charts/SampleRecharts.tsx`
- `lib/labDemoQuery.ts`, `lib/queryClient.ts`

**Slate UI**

- `components/Header.tsx`, `Footer.tsx`
- `components/TopLegsView.tsx`, `AllPlayersView.tsx`, `ResultsView.tsx`
- `components/PlayerBlock.tsx`, `LegCard.tsx`, `ParlayCard.tsx`, `TicketStrip.tsx`
- `components/Dropdown.tsx`, `LoadingSkeleton.tsx`, `PlayerStatsPanel.tsx`
- `components/HitRateByBookChart.tsx`
- `styles/slate.css` and `public/slate.css` if present

**Slate / API libs & types**

- `lib/api.ts`, `backend.ts`, `queries.ts`, `slate.ts`, `mapSlate.ts`, `mapLiveProps.ts`
- `lib/enrichSlateFromLiveProps.ts`, `players.ts`, `legsRoi.ts`, `format.ts`, `constants.ts`
- `types/api.ts`, `types/slate.ts`

**Unused shadcn (if nothing imports them after cleanup)**

- `components/ui/button.tsx`, `components/ui/card.tsx`

Exact delete list is verified during implementation by import graph + `npm run build`.

## Home adjustments

- **HomeNav:** remove Slate and Lab links; keep logo, NBA/WNBA → `#live-now`, settings gear.
- **TicketHero:** keep primary “See what’s inside” → `#live-now`; remove secondary “Learn the games” link to `/slate`.
- **NotFoundPage:** copy stays “Back to home” → `/`.

## Router

```tsx
<Routes>
  <Route path="/" element={<HomePage />} />
  <Route path="*" element={<NotFoundPage />} />
</Routes>
```

## Dependencies

Uninstall when unused:

- `@tanstack/react-query`
- `recharts`
- `d3`
- `@types/d3`

Keep: `react`, `react-dom`, `react-router-dom`, `lucide-react`, `geist`, Tailwind stack, and any shadcn helpers still imported (`clsx`, `tailwind-merge`, `class-variance-authority` only if Button/Card or `cn` remain).

## Out of scope

- Building replacement product pages for legs/players/results
- Backend / data pipeline changes
- Redesigning the home visual system beyond link/CTA cleanup
- Renaming the repo or product outside the frontend tree

## Success criteria

- `npm run build` in `frontend/` succeeds
- `/` renders the home landing
- `/slate` and `/lab` hit NotFound
- No remaining imports of deleted modules
- Home nav has no Slate/Lab destinations
