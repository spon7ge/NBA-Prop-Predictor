# HoopVista Frontend

React + TypeScript dashboard for HoopVista, built with [Vite](https://vite.dev/).

## Stack

- React 19 · TypeScript · Vite
- Tailwind CSS v4 · shadcn/ui · Geist · lucide-react
- React Router · TanStack Query
- Recharts · D3.js

Legacy dashboard screens still use `src/styles/slate.css`. New UI (e.g. `/lab`) uses Tailwind + shadcn.

## Visual language (new UI)

| Pattern | Usage |
|---------|--------|
| Borders | Subtle dark borders separate sections |
| Spacing | Generous vertical spacing |
| Typography | Bold headings, regular body, muted secondary |
| Color accents | League badges are the only color besides white/gray |
| Cards | Subtle rounded boxes group related information |
| CTAs | lucide arrow icons on links/actions |

## Setup

```bash
cd frontend
npm install
```

## Development

From the `frontend/` directory:

```bash
npm run dev
```

Open http://localhost:5173 (dashboard) or http://localhost:5173/lab (stack showcase).

In dev mode, the app can read JSON from the repo-root `data/` folder (`../data/props/...`) and proxies `/api` to `http://127.0.0.1:8000`.

## Production build

```bash
npm run build
npm run preview
```

Build output goes to `frontend/dist/`. Static blog pages in `public/` are copied into the build automatically.

## Project structure

```
frontend/
  src/
    components/      # Dashboard UI + components/ui (shadcn)
    charts/          # Recharts / D3 samples
    pages/           # Lab, NotFound
    lib/             # Data fetching, utils, business logic
    styles/          # slate.css (legacy dashboard)
    types/           # TypeScript types
    index.css        # Tailwind + theme tokens
  public/            # Static assets + blog/contact/faq pages
```

## Data files

The dashboard expects the same JSON exports as before:

- `data/props/ev_analysis/*.json` — parlay slates (Top Legs)
- `data/props/enriched/dfs_enriched_latest.json` — player table (All Players)
