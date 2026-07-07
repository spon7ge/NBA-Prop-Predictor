# HoopVista Frontend

React + TypeScript dashboard for HoopVista, built with [Vite](https://vite.dev/).

This replaces the legacy static dashboard in `docs/index.html` + `docs/slate.js` with a component-based React app while keeping the same UI and data sources.

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

Open http://localhost:5173. In dev mode, the app reads JSON from the repo-root `data/` folder (`../data/props/...`).

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
    components/   # React UI (Top Legs, All Players, etc.)
    lib/          # Data fetching, formatting, business logic
    styles/       # slate.css (ported from docs/)
    types/        # TypeScript types
  public/         # Static assets + blog/contact/faq pages
```

## Data files

The dashboard expects the same JSON exports as before:

- `data/props/ev_analysis/*.json` — parlay slates (Top Legs)
- `data/props/enriched/dfs_enriched_latest.json` — player table (All Players)
