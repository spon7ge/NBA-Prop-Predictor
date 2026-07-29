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

## Tests

```bash
npm test
```

Vitest + Testing Library cover routing and LIVE NOW empty/filled states.

## Production build

```bash
npm run build
npm run preview
```

Build output goes to `frontend/dist/` (gitignored; regenerate as needed).

## Project structure

```
frontend/
  src/
    components/home/  # Nav, ticker, hero, LIVE NOW (+ format helpers/tests)
    pages/            # Home, NotFound
    AppRouter.tsx     # `/` + `*`
    main.tsx
    index.css         # Tailwind + theme tokens used by the landing
  public/             # Favicon and static assets
```

Coding standards for this app follow repo `claude.md` (typing, small focused modules, input guards, tests with changes).
