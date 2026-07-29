# HoopVista Frontend

React + TypeScript home landing for HoopVista, built with [Vite](https://vite.dev/).

## Stack

- React 19 · TypeScript · Vite
- Tailwind CSS v4 · Geist · lucide-react
- React Router

## Visual language

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

Open http://localhost:5173/ for the home landing.

## Production build

```bash
npm run build
npm run preview
```

Build output goes to `frontend/dist/`.

## Project structure

```
frontend/
  src/
    components/home/  # Nav, ticker, hero, LIVE NOW
    pages/            # Home, NotFound
    index.css         # Tailwind + theme tokens
  public/             # Static assets (favicon, etc.)
```
