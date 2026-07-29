# Frontend Tech Stack Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `frontend/` in place with Tailwind v4, shadcn/ui, React Router, Geist, lucide-react, Recharts, and D3 — keep the existing dashboard on `/` and ship a `/lab` page that proves the stack and visual language.

**Architecture:** Wrap the app in `QueryClientProvider` + `BrowserRouter`. Route `/` to the existing `App` (still using `slate.css`). Route `/lab` to a new Tailwind/shadcn Lab page with demo Query, Recharts, and D3 samples. Theme tokens and visual-language patterns live in `src/index.css`.

**Tech Stack:** React 19 · TypeScript · Vite 6 · Tailwind CSS v4 · shadcn/ui · React Router · TanStack Query · D3.js · Recharts · Geist · lucide-react

## Global Constraints

- Location: `frontend/` only — do not create a second app root.
- Upgrade in place: existing dashboard behavior on `/` must keep working.
- CSS: Tailwind-first for new UI; leave `src/styles/slate.css` for the legacy dashboard (do not migrate or delete it).
- Visual language (locked): subtle dark borders; generous vertical spacing; bold headings / regular body / muted secondary; league badges are the only color besides white/gray; subtle rounded cards; lucide arrow icons on CTAs.
- shadcn: `new-york` style, `neutral` baseColor; this pass installs Button + Card only.
- Keep Vite `@/` alias, `/api` proxy to `http://127.0.0.1:8000`, and `server.fs.allow` for repo root.
- Out of scope: restyling Top Legs / All Players, dark-mode toggle, auth, extra shadcn primitives, backend changes.
- Spec: `docs/superpowers/specs/2026-07-29-frontend-tech-stack-design.md`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `frontend/package.json` | New dependencies |
| `frontend/vite.config.ts` | Add `@tailwindcss/vite` plugin |
| `frontend/components.json` | shadcn CLI config |
| `frontend/src/index.css` | Tailwind import, CSS variables, Geist, visual-language tokens |
| `frontend/src/lib/utils.ts` | `cn()` helper for shadcn |
| `frontend/src/components/ui/button.tsx` | shadcn Button |
| `frontend/src/components/ui/card.tsx` | shadcn Card |
| `frontend/src/AppRouter.tsx` | Route table (`/`, `/lab`, `*`) |
| `frontend/src/pages/LabPage.tsx` | Stack + visual-language showcase |
| `frontend/src/pages/NotFoundPage.tsx` | 404 |
| `frontend/src/charts/SampleRecharts.tsx` | Recharts demo (muted palette) |
| `frontend/src/charts/SampleD3.tsx` | Small D3 SVG demo |
| `frontend/src/lib/labDemoQuery.ts` | Isolated Lab `useQuery` fetcher |
| `frontend/src/main.tsx` | Import `index.css`; mount router under QueryClient |
| `frontend/src/App.tsx` | Unchanged dashboard (still imports slate styles via existing path) |
| `frontend/README.md` | Document stack, `/lab`, visual language, slate coexistence |

---

### Task 1: Install dependencies and wire Tailwind v4

**Files:**
- Modify: `frontend/package.json`
- Modify: `frontend/vite.config.ts`
- Create: `frontend/src/index.css`
- Modify: `frontend/src/main.tsx`
- Modify: `frontend/README.md` (stack bullets only; finish in Task 5)

**Interfaces:**
- Consumes: existing Vite React setup
- Produces: Tailwind utilities available app-wide via `src/index.css`; Vite plugin active

- [ ] **Step 1: Install packages from `frontend/`**

```bash
cd frontend
npm install react-router-dom d3 recharts geist lucide-react class-variance-authority clsx tailwind-merge
npm install -D tailwindcss @tailwindcss/vite @types/d3
```

Expected: `package.json` lists those deps; lockfile updates; exit 0.

- [ ] **Step 2: Register Tailwind Vite plugin**

Replace `frontend/vite.config.ts` with:

```ts
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const rootDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(rootDir, "..");

export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "./",
  resolve: {
    alias: {
      "@": path.resolve(rootDir, "./src"),
    },
  },
  server: {
    fs: {
      allow: [repoRoot],
    },
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
});
```

- [ ] **Step 3: Create `frontend/src/index.css` with Tailwind + visual-language tokens**

```css
@import "tailwindcss";
@import "geist/dist/fonts/geist-sans/style.css";
@import "geist/dist/fonts/geist-mono/style.css";

/* shadcn-compatible neutral tokens + HoopVista visual language */
:root {
  --radius: 0.5rem;
  --background: oklch(1 0 0);
  --foreground: oklch(0.145 0 0);
  --card: oklch(1 0 0);
  --card-foreground: oklch(0.145 0 0);
  --popover: oklch(1 0 0);
  --popover-foreground: oklch(0.145 0 0);
  --primary: oklch(0.205 0 0);
  --primary-foreground: oklch(0.985 0 0);
  --secondary: oklch(0.97 0 0);
  --secondary-foreground: oklch(0.205 0 0);
  --muted: oklch(0.97 0 0);
  --muted-foreground: oklch(0.556 0 0);
  --accent: oklch(0.97 0 0);
  --accent-foreground: oklch(0.205 0 0);
  --destructive: oklch(0.577 0.245 27.325);
  --border: oklch(0.88 0 0);
  --input: oklch(0.88 0 0);
  --ring: oklch(0.708 0 0);
  /* League badge accent — only intentional non-neutral color for new UI */
  --league-badge: oklch(0.55 0.14 250);
  --league-badge-foreground: oklch(0.98 0 0);
  --font-sans: "Geist", ui-sans-serif, system-ui, sans-serif;
  --font-mono: "Geist Mono", ui-monospace, monospace;
}

@theme inline {
  --radius-sm: calc(var(--radius) - 4px);
  --radius-md: calc(var(--radius) - 2px);
  --radius-lg: var(--radius);
  --radius-xl: calc(var(--radius) + 4px);
  --color-background: var(--background);
  --color-foreground: var(--foreground);
  --color-card: var(--card);
  --color-card-foreground: var(--card-foreground);
  --color-popover: var(--popover);
  --color-popover-foreground: var(--popover-foreground);
  --color-primary: var(--primary);
  --color-primary-foreground: var(--primary-foreground);
  --color-secondary: var(--secondary);
  --color-secondary-foreground: var(--secondary-foreground);
  --color-muted: var(--muted);
  --color-muted-foreground: var(--muted-foreground);
  --color-accent: var(--accent);
  --color-accent-foreground: var(--accent-foreground);
  --color-destructive: var(--destructive);
  --color-border: var(--border);
  --color-input: var(--input);
  --color-ring: var(--ring);
  --color-league-badge: var(--league-badge);
  --color-league-badge-foreground: var(--league-badge-foreground);
  --font-sans: var(--font-sans);
  --font-mono: var(--font-mono);
}

@layer base {
  * {
    @apply border-border;
  }
  body {
    @apply bg-background text-foreground font-sans antialiased;
  }
}
```

If `geist/dist/fonts/...` import paths fail at build time, open `node_modules/geist/package.json` `exports` / `files` and adjust the two `@import` paths to the package’s actual CSS entry points — do not switch fonts.

- [ ] **Step 4: Import `index.css` in `main.tsx` (keep `slate.css` for now)**

Update imports at top of `frontend/src/main.tsx` to:

```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClientProvider } from "@tanstack/react-query";
import App from "./App";
import { queryClient } from "@/lib/queryClient";
import "./index.css";
import "./styles/slate.css";
```

Leave the render tree as `QueryClientProvider` → `App` until Task 4.

- [ ] **Step 5: Verify build**

```bash
cd frontend && npm run build
```

Expected: `tsc -b && vite build` succeeds. If Geist CSS path errors, fix imports per Step 3 note and rebuild.

- [ ] **Step 6: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/vite.config.ts frontend/src/index.css frontend/src/main.tsx
git commit -m "$(cat <<'EOF'
Add Tailwind v4, router/charts deps, and theme tokens to frontend.

EOF
)"
```

---

### Task 2: Add shadcn `cn`, Button, and Card

**Files:**
- Create: `frontend/components.json`
- Create: `frontend/src/lib/utils.ts`
- Create: `frontend/src/components/ui/button.tsx`
- Create: `frontend/src/components/ui/card.tsx`

**Interfaces:**
- Consumes: Tailwind theme tokens from Task 1 (`border`, `primary`, `muted-foreground`, radius)
- Produces:
  - `cn(...inputs: ClassValue[]): string` from `@/lib/utils`
  - `Button` from `@/components/ui/button` (variants: default, outline, ghost, link; sizes: default, sm, lg, icon)
  - `Card`, `CardHeader`, `CardTitle`, `CardDescription`, `CardContent` from `@/components/ui/card`

- [ ] **Step 1: Create `frontend/components.json`**

```json
{
  "$schema": "https://ui.shadcn.com/schema.json",
  "style": "new-york",
  "rsc": false,
  "tsx": true,
  "tailwind": {
    "config": "",
    "css": "src/index.css",
    "baseColor": "neutral",
    "cssVariables": true,
    "prefix": ""
  },
  "aliases": {
    "components": "@/components",
    "utils": "@/lib/utils",
    "ui": "@/components/ui",
    "lib": "@/lib",
    "hooks": "@/hooks"
  },
  "iconLibrary": "lucide"
}
```

- [ ] **Step 2: Create `frontend/src/lib/utils.ts`**

```ts
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
```

- [ ] **Step 3: Create `frontend/src/components/ui/button.tsx`**

```tsx
import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary/90",
        outline:
          "border border-border bg-background hover:bg-accent hover:text-accent-foreground",
        ghost: "hover:bg-accent hover:text-accent-foreground",
        link: "text-primary underline-offset-4 hover:underline",
      },
      size: {
        default: "h-9 px-4 py-2",
        sm: "h-8 rounded-md px-3 text-xs",
        lg: "h-10 rounded-md px-8",
        icon: "h-9 w-9",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, ...props }, ref) => (
    <button
      className={cn(buttonVariants({ variant, size, className }))}
      ref={ref}
      {...props}
    />
  ),
);
Button.displayName = "Button";

export { Button, buttonVariants };
```

- [ ] **Step 4: Create `frontend/src/components/ui/card.tsx`**

```tsx
import * as React from "react";
import { cn } from "@/lib/utils";

const Card = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div
      ref={ref}
      className={cn(
        "rounded-lg border border-border bg-card text-card-foreground",
        className,
      )}
      {...props}
    />
  ),
);
Card.displayName = "Card";

const CardHeader = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn("flex flex-col space-y-1.5 p-6", className)} {...props} />
  ),
);
CardHeader.displayName = "CardHeader";

const CardTitle = React.forwardRef<HTMLHeadingElement, React.HTMLAttributes<HTMLHeadingElement>>(
  ({ className, ...props }, ref) => (
    <h3
      ref={ref}
      className={cn("text-lg font-semibold leading-none tracking-tight", className)}
      {...props}
    />
  ),
);
CardTitle.displayName = "CardTitle";

const CardDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => (
  <p ref={ref} className={cn("text-sm text-muted-foreground", className)} {...props} />
));
CardDescription.displayName = "CardDescription";

const CardContent = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => (
    <div ref={ref} className={cn("p-6 pt-0", className)} {...props} />
  ),
);
CardContent.displayName = "CardContent";

export { Card, CardHeader, CardTitle, CardDescription, CardContent };
```

- [ ] **Step 5: Verify build**

```bash
cd frontend && npm run build
```

Expected: success (unused files are fine; TypeScript still typechecks `src/`).

- [ ] **Step 6: Commit**

```bash
git add frontend/components.json frontend/src/lib/utils.ts frontend/src/components/ui/button.tsx frontend/src/components/ui/card.tsx
git commit -m "$(cat <<'EOF'
Add shadcn utils, Button, and Card primitives.

EOF
)"
```

---

### Task 3: Chart samples + Lab demo query

**Files:**
- Create: `frontend/src/charts/SampleRecharts.tsx`
- Create: `frontend/src/charts/SampleD3.tsx`
- Create: `frontend/src/lib/labDemoQuery.ts`

**Interfaces:**
- Consumes: `recharts`, `d3`, `@tanstack/react-query` (via Lab later)
- Produces:
  - `export function SampleRecharts(): JSX.Element`
  - `export function SampleD3(): JSX.Element`
  - `export const labDemoQueryKey = ["lab", "demo"] as const`
  - `export async function fetchLabDemo(): Promise<{ label: string; value: number }>`
  - `export function useLabDemoQuery()` returning TanStack `useQuery` result for that key

- [ ] **Step 1: Create `frontend/src/lib/labDemoQuery.ts`**

```ts
import { useQuery } from "@tanstack/react-query";

export const labDemoQueryKey = ["lab", "demo"] as const;

export type LabDemoDatum = { label: string; value: number };

export async function fetchLabDemo(): Promise<LabDemoDatum> {
  // Isolated demo — no slate/props APIs
  await new Promise((r) => setTimeout(r, 200));
  return { label: "stack-ready", value: 1 };
}

export function useLabDemoQuery() {
  return useQuery({
    queryKey: labDemoQueryKey,
    queryFn: fetchLabDemo,
  });
}
```

- [ ] **Step 2: Create `frontend/src/charts/SampleRecharts.tsx`**

```tsx
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";

const DATA = [
  { name: "Mon", pts: 12 },
  { name: "Tue", pts: 18 },
  { name: "Wed", pts: 9 },
  { name: "Thu", pts: 22 },
  { name: "Fri", pts: 15 },
];

export function SampleRecharts() {
  if (!DATA.length) {
    return <p className="text-sm text-muted-foreground">No chart data.</p>;
  }

  return (
    <div className="h-56 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={DATA}>
          <CartesianGrid stroke="currentColor" className="text-border" vertical={false} />
          <XAxis dataKey="name" tick={{ fill: "currentColor" }} className="text-muted-foreground text-xs" />
          <YAxis tick={{ fill: "currentColor" }} className="text-muted-foreground text-xs" />
          <Tooltip />
          <Bar dataKey="pts" fill="oklch(0.45 0 0)" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
```

- [ ] **Step 3: Create `frontend/src/charts/SampleD3.tsx`**

```tsx
import { useEffect, useRef } from "react";
import * as d3 from "d3";

const VALUES = [4, 8, 15, 16, 23, 42];

export function SampleD3() {
  const ref = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    const svg = d3.select(ref.current);
    svg.selectAll("*").remove();

    if (!VALUES.length) return;

    const width = 320;
    const height = 120;
    const margin = { top: 8, right: 8, bottom: 24, left: 28 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    svg.attr("viewBox", `0 0 ${width} ${height}`);

    const g = svg
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const x = d3
      .scaleBand()
      .domain(VALUES.map((_, i) => String(i)))
      .range([0, innerW])
      .padding(0.2);

    const y = d3
      .scaleLinear()
      .domain([0, d3.max(VALUES) ?? 0])
      .nice()
      .range([innerH, 0]);

    g.selectAll("rect")
      .data(VALUES)
      .join("rect")
      .attr("x", (_, i) => x(String(i)) ?? 0)
      .attr("y", (d) => y(d))
      .attr("width", x.bandwidth())
      .attr("height", (d) => innerH - y(d))
      .attr("fill", "oklch(0.45 0 0)")
      .attr("rx", 3);

    g.append("g")
      .attr("transform", `translate(0,${innerH})`)
      .call(d3.axisBottom(x).tickFormat((d) => String(Number(d) + 1)))
      .attr("color", "oklch(0.55 0 0)");

    g.append("g")
      .call(d3.axisLeft(y).ticks(4))
      .attr("color", "oklch(0.55 0 0)");
  }, []);

  if (!VALUES.length) {
    return <p className="text-sm text-muted-foreground">No chart data.</p>;
  }

  return <svg ref={ref} className="w-full max-w-sm" role="img" aria-label="D3 sample bar chart" />;
}
```

- [ ] **Step 4: Verify build**

```bash
cd frontend && npm run build
```

Expected: success.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/charts/SampleRecharts.tsx frontend/src/charts/SampleD3.tsx frontend/src/lib/labDemoQuery.ts
git commit -m "$(cat <<'EOF'
Add Lab demo query and Recharts/D3 sample chart components.

EOF
)"
```

---

### Task 4: Router, Lab page, NotFound

**Files:**
- Create: `frontend/src/AppRouter.tsx`
- Create: `frontend/src/pages/LabPage.tsx`
- Create: `frontend/src/pages/NotFoundPage.tsx`
- Modify: `frontend/src/main.tsx`
- Modify: `frontend/src/App.tsx` — only if slate import should move; prefer: keep `slate.css` import in `main.tsx` for now (dashboard still works; Lab inherits global slate rules — acceptable for this pass per CSS coexistence). Do **not** restyle dashboard components.

**Interfaces:**
- Consumes: `Button`, `Card*`, `SampleRecharts`, `SampleD3`, `useLabDemoQuery`, `react-router-dom`
- Produces: routes `/` → `App`, `/lab` → `LabPage`, `*` → `NotFoundPage`

- [ ] **Step 1: Create `frontend/src/pages/NotFoundPage.tsx`**

```tsx
import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";

export function NotFoundPage() {
  return (
    <main className="mx-auto flex min-h-svh max-w-lg flex-col justify-center gap-4 px-6">
      <h1 className="text-2xl font-bold tracking-tight">Page not found</h1>
      <p className="text-muted-foreground">That route does not exist.</p>
      <Link
        to="/"
        className="inline-flex items-center gap-1 text-sm font-medium text-foreground hover:underline"
      >
        Back to dashboard
        <ArrowRight className="size-4" />
      </Link>
    </main>
  );
}
```

- [ ] **Step 2: Create `frontend/src/pages/LabPage.tsx`**

```tsx
import { Link } from "react-router-dom";
import { ArrowRight, ArrowUpRight, FlaskConical } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { SampleRecharts } from "@/charts/SampleRecharts";
import { SampleD3 } from "@/charts/SampleD3";
import { useLabDemoQuery } from "@/lib/labDemoQuery";

export function LabPage() {
  const demo = useLabDemoQuery();

  return (
    <main className="mx-auto max-w-3xl space-y-10 px-6 py-12">
      <header className="space-y-3">
        <div className="flex items-center gap-2">
          <FlaskConical className="size-5 text-muted-foreground" />
          <span className="rounded-md bg-league-badge px-2 py-0.5 text-xs font-medium text-league-badge-foreground">
            NBA
          </span>
        </div>
        <h1 className="text-3xl font-bold tracking-tight">Stack lab</h1>
        <p className="text-muted-foreground">
          Tailwind v4, shadcn, Query, Recharts, and D3 — ready for product UI.
        </p>
        <Link
          to="/"
          className="inline-flex items-center gap-1 text-sm font-medium hover:underline"
        >
          Open dashboard
          <ArrowRight className="size-4" />
        </Link>
      </header>

      <Card>
        <CardHeader>
          <CardTitle>Controls</CardTitle>
          <CardDescription>shadcn Button + arrow CTA pattern</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-3">
          <Button type="button">Primary</Button>
          <Button type="button" variant="outline">
            Outline
          </Button>
          <Button type="button" variant="ghost" className="gap-1">
            External pattern
            <ArrowUpRight className="size-4" />
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>TanStack Query</CardTitle>
          <CardDescription>Isolated demo key — not slate/props</CardDescription>
        </CardHeader>
        <CardContent>
          {demo.isPending && (
            <p className="text-sm text-muted-foreground">Loading…</p>
          )}
          {demo.isError && (
            <p className="text-sm text-muted-foreground">Demo query failed.</p>
          )}
          {demo.isSuccess && (
            <p className="text-sm">
              <span className="font-medium">{demo.data.label}</span>
              <span className="text-muted-foreground"> · value {demo.data.value}</span>
            </p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Recharts</CardTitle>
          <CardDescription>Muted neutral series</CardDescription>
        </CardHeader>
        <CardContent>
          <SampleRecharts />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>D3</CardTitle>
          <CardDescription>Small SVG sample</CardDescription>
        </CardHeader>
        <CardContent>
          <SampleD3 />
        </CardContent>
      </Card>
    </main>
  );
}
```

- [ ] **Step 3: Create `frontend/src/AppRouter.tsx`**

```tsx
import { Routes, Route } from "react-router-dom";
import App from "@/App";
import { LabPage } from "@/pages/LabPage";
import { NotFoundPage } from "@/pages/NotFoundPage";

export function AppRouter() {
  return (
    <Routes>
      <Route path="/" element={<App />} />
      <Route path="/lab" element={<LabPage />} />
      <Route path="*" element={<NotFoundPage />} />
    </Routes>
  );
}
```

- [ ] **Step 4: Wire router in `frontend/src/main.tsx`**

```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter } from "react-router-dom";
import { AppRouter } from "./AppRouter";
import { queryClient } from "@/lib/queryClient";
import "./index.css";
import "./styles/slate.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AppRouter />
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);
```

- [ ] **Step 5: Verify build + smoke routes**

```bash
cd frontend && npm run build
```

Expected: success.

Then:

```bash
cd frontend && npm run dev
```

Manually check:
- `http://localhost:5173/` — existing dashboard
- `http://localhost:5173/lab` — Lab showcase (cards, badge, charts, query)
- `http://localhost:5173/does-not-exist` — NotFound with arrow link

Stop the dev server when done.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/AppRouter.tsx frontend/src/pages/LabPage.tsx frontend/src/pages/NotFoundPage.tsx frontend/src/main.tsx
git commit -m "$(cat <<'EOF'
Wire React Router with Lab showcase and NotFound routes.

EOF
)"
```

---

### Task 5: README update + final verification

**Files:**
- Modify: `frontend/README.md`

**Interfaces:**
- Consumes: completed stack from Tasks 1–4
- Produces: docs that match running app

- [ ] **Step 1: Rewrite `frontend/README.md` structure section and add stack notes**

Replace the file with:

```markdown
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
```

- [ ] **Step 2: Final build verification**

```bash
cd frontend && npm run build
```

Expected: exit 0.

- [ ] **Step 3: Commit**

```bash
git add frontend/README.md
git commit -m "$(cat <<'EOF'
Document frontend stack, /lab route, and visual language.

EOF
)"
```

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| Tailwind v4 + `@tailwindcss/vite` | 1 |
| Geist + theme tokens / visual language | 1, 4 |
| react-router-dom, d3, recharts, lucide, cva/clsx/twMerge | 1–3 |
| shadcn Button + Card + `cn` + `components.json` | 2 |
| `/` existing App, `/lab` showcase, 404 | 4 |
| Lab Query / Recharts / D3 demos | 3–4 |
| Keep slate.css, proxy, `@/` | 1, 4 |
| README update | 5 |
| `npm run build` passes | every task |

## Plan self-review

1. **Spec coverage:** All locked decisions mapped to tasks above; no dark-mode/auth/dashboard restyle tasks.
2. **Placeholders:** None — concrete file contents and commands included.
3. **Types:** `LabDemoDatum`, `useLabDemoQuery`, `SampleRecharts` / `SampleD3` names consistent across Tasks 3–4.
4. **Geist path risk:** Task 1 notes inspecting `node_modules/geist` if CSS import paths differ — still `geist` package per spec.
