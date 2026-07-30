# HoopVista About Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a `/about` page with shared home chrome (nav + ticker), HoopVista-specific About copy, and NBA/WNBA league pills matching the approved About mockup structure.

**Architecture:** Extract `HomeChromeLayout` (HomeNav + LiveTicker + Outlet). Nest `/` and `/about` under it. About UI lives in `frontend/src/components/about/AboutContent.tsx`; thin `AboutPage` composes it. HomeNav uses React Router `Link` + `useLocation` for the active About pill.

**Tech Stack:** React 19 · TypeScript · Vite 6 · Tailwind CSS v4 · React Router 7 · Vitest · Testing Library · lucide-react · Geist

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-29-hoopvista-about-page-design.md`
- Brand: HoopVista (not boxseats)
- Route: `/about` under shared chrome with `/`
- League pills: NBA + WNBA only
- No Contributors section
- Copy: HoopVista-specific basketball analytics / props (static strings)
- About components folder: `frontend/src/components/about/`
- Fonts: existing Geist only; no new npm deps; no API calls
- Verify with `npm run test` and `npm run build` in `frontend/`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `frontend/src/components/about/AboutContent.tsx` | Badge, headline, NBA/WNBA pills, body copy |
| `frontend/src/components/about/AboutContent.test.tsx` | About content unit tests |
| `frontend/src/pages/AboutPage.tsx` | Thin page wrapping `AboutContent` |
| `frontend/src/layouts/HomeChromeLayout.tsx` | Nav + ticker + `<Outlet />` |
| `frontend/src/components/home/HomeNav.tsx` | Link to `/about`, active state, `/#live-now` league links |
| `frontend/src/components/home/HomeNav.test.tsx` | Nav link + active-state tests |
| `frontend/src/pages/HomePage.tsx` | Home sections only (no nav/ticker) |
| `frontend/src/AppRouter.tsx` | Nest `/` and `/about` under chrome layout |
| `frontend/src/AppRouter.test.tsx` | Assert `/about` renders About content |

---

### Task 1: AboutContent component

**Files:**
- Create: `frontend/src/components/about/AboutContent.tsx`
- Create: `frontend/src/components/about/AboutContent.test.tsx`

**Interfaces:**
- Produces: `export function AboutContent(): JSX.Element`

- [ ] **Step 1: Write the failing test**

```tsx
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { AboutContent } from "./AboutContent";

describe("AboutContent", () => {
  it("renders badge, headline, league pills, and body copy", () => {
    render(<AboutContent />);

    expect(screen.getByText(/sports analytics/i)).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: /about hoopvista/i }),
    ).toBeInTheDocument();
    expect(screen.getByText("NBA")).toBeInTheDocument();
    expect(screen.getByText("WNBA")).toBeInTheDocument();
    expect(
      screen.getByText(/basketball analytics/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/plain-language/i)).toBeInTheDocument();
    expect(screen.getByText(/still in beta/i)).toBeInTheDocument();
    expect(screen.queryByText(/contributors/i)).not.toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm run test -- src/components/about/AboutContent.test.tsx`

Expected: FAIL (module or export not found)

- [ ] **Step 3: Implement `AboutContent`**

```tsx
const LEAGUES = [
  {
    id: "nba",
    label: "NBA",
    className: "border-sky-500/60 text-sky-300",
  },
  {
    id: "wnba",
    label: "WNBA",
    className: "border-violet-500/60 text-violet-300",
  },
] as const;

export function AboutContent() {
  return (
    <main className="mx-auto max-w-3xl px-4 py-16 sm:px-6">
      <p className="inline-flex rounded-full border border-white/20 px-3 py-1 text-[11px] font-medium tracking-wide text-white/50 uppercase">
        Sports Analytics · Beta
      </p>

      <h1 className="mt-6 font-heading text-4xl font-semibold tracking-tight text-white sm:text-5xl">
        About HoopVista.
      </h1>

      <ul className="mt-6 flex flex-wrap gap-2" aria-label="Leagues">
        {LEAGUES.map((league) => (
          <li
            key={league.id}
            className={`rounded-full border px-3 py-1 text-xs font-medium ${league.className}`}
          >
            {league.label}
          </li>
        ))}
      </ul>

      <div className="mt-8 space-y-5 text-base leading-relaxed text-white/55">
        <p>
          HoopVista is an interactive basketball analytics site for the NBA and
          WNBA — live trackers, props context, and visualizations that help you
          see the game from a better seat.
        </p>
        <p>
          We design for any fan. Every stat comes with a plain-language
          explainer, and charts stay honest about what the numbers do and do not
          say.
        </p>
        <p>
          The site is still in beta. We are actively adding tools and polishing
          what is already here.
        </p>
      </div>
    </main>
  );
}
```

Note: badge visible text can be `Sports Analytics · Beta` (middle dot). The test uses `/sports analytics/i` so casing is flexible; keep “Beta” in the badge string.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm run test -- src/components/about/AboutContent.test.tsx`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/about/AboutContent.tsx frontend/src/components/about/AboutContent.test.tsx
git commit -m "Add HoopVista AboutContent with badge, pills, and copy."
```

---

### Task 2: HomeNav About link + active state

**Files:**
- Modify: `frontend/src/components/home/HomeNav.tsx`
- Create: `frontend/src/components/home/HomeNav.test.tsx`

**Interfaces:**
- Consumes: `react-router-dom` `Link`, `useLocation`
- Produces: About `Link` to `/about`; active pill when pathname is `/about`; league anchors `href="/#live-now"`

- [ ] **Step 1: Write the failing tests**

```tsx
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { HomeNav } from "./HomeNav";

function renderNav(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <HomeNav />
    </MemoryRouter>,
  );
}

describe("HomeNav", () => {
  it("links About to /about", () => {
    renderNav("/");
    expect(screen.getByRole("link", { name: "About" })).toHaveAttribute(
      "href",
      "/about",
    );
  });

  it("marks About as current on /about", () => {
    renderNav("/about");
    expect(screen.getByRole("link", { name: "About" })).toHaveAttribute(
      "aria-current",
      "page",
    );
  });

  it("does not mark About current on home", () => {
    renderNav("/");
    expect(screen.getByRole("link", { name: "About" })).not.toHaveAttribute(
      "aria-current",
    );
  });

  it("points league links at /#live-now", () => {
    renderNav("/about");
    expect(screen.getByRole("link", { name: "NBA" })).toHaveAttribute(
      "href",
      "/#live-now",
    );
    expect(screen.getByRole("link", { name: "WNBA" })).toHaveAttribute(
      "href",
      "/#live-now",
    );
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npm run test -- src/components/home/HomeNav.test.tsx`

Expected: FAIL (About still `href="#"` or missing `aria-current`)

- [ ] **Step 3: Update `HomeNav`**

Replace the About `<a href="#">` and league `#live-now` anchors with:

```tsx
import { Link, useLocation } from "react-router-dom";
// ... existing icon imports and leagues array ...

export function HomeNav() {
  const { pathname } = useLocation();
  const aboutActive = pathname === "/about";

  return (
    <header className="border-b border-white/10 bg-black">
      <div className="mx-auto flex h-12 max-w-6xl items-center justify-between gap-4 px-4 sm:px-6">
        <Link to="/" className="flex items-center gap-2 text-white no-underline">
          {/* existing logo mark + HoopVista wordmark */}
        </Link>

        <div className="flex items-center gap-4">
          <nav className="flex items-center gap-3" aria-label="Leagues">
            {leagues.map((league) => (
              <a
                key={league.id}
                href="/#live-now"
                className="flex items-center gap-2 text-[14px] font-medium text-white/80 no-underline transition-colors hover:text-white"
              >
                <img
                  src={league.icon}
                  alt=""
                  aria-hidden
                  className="size-4 shrink-0 object-contain"
                />
                {league.label}
              </a>
            ))}
            <Link
              to="/about"
              aria-current={aboutActive ? "page" : undefined}
              className={
                aboutActive
                  ? "rounded-md bg-neutral-600/80 px-2.5 py-1 text-[14px] font-medium text-white/90 no-underline"
                  : "rounded-md px-2.5 py-1 text-[14px] font-medium text-white/80 no-underline transition-colors hover:bg-neutral-600/50 hover:text-white"
              }
            >
              About
            </Link>
          </nav>

          {/* existing settings button unchanged */}
        </div>
      </div>
    </header>
  );
}
```

Keep existing logo/`BarChart3`/`Settings` markup; only change routing and About active styles as above.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npm run test -- src/components/home/HomeNav.test.tsx`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/home/HomeNav.tsx frontend/src/components/home/HomeNav.test.tsx
git commit -m "Wire HomeNav About link and active state to /about."
```

---

### Task 3: Chrome layout, AboutPage, router wiring

**Files:**
- Create: `frontend/src/layouts/HomeChromeLayout.tsx`
- Create: `frontend/src/pages/AboutPage.tsx`
- Modify: `frontend/src/pages/HomePage.tsx`
- Modify: `frontend/src/AppRouter.tsx`
- Modify: `frontend/src/AppRouter.test.tsx`

**Interfaces:**
- Consumes: `AboutContent`, `HomeNav`, `LiveTicker`, `HomePage`
- Produces: `HomeChromeLayout`, `AboutPage`; routes `/` and `/about` under chrome

- [ ] **Step 1: Extend `AppRouter.test.tsx` with a failing `/about` case**

Add:

```tsx
it("renders about at /about", () => {
  render(
    <MemoryRouter initialEntries={["/about"]}>
      <AppRouter />
    </MemoryRouter>,
  );
  expect(
    screen.getByRole("heading", { name: /about hoopvista/i }),
  ).toBeInTheDocument();
  expect(screen.getByText("No live games")).toBeInTheDocument();
});
```

Keep existing `/` and unknown-path tests. Home at `/` should still find a hoopvista heading (hero); if that assertion becomes ambiguous after layout changes, scope it with `getByRole("heading", { name: /^hoopvista$/i })` or the hero tagline already present — prefer the existing assertion if it still passes.

- [ ] **Step 2: Run test to verify `/about` fails**

Run: `cd frontend && npm run test -- src/AppRouter.test.tsx`

Expected: FAIL (NotFound or missing About heading)

- [ ] **Step 3: Add layout + About page + slim HomePage + router**

`frontend/src/layouts/HomeChromeLayout.tsx`:

```tsx
import { Outlet } from "react-router-dom";
import { HomeNav } from "@/components/home/HomeNav";
import { LiveTicker } from "@/components/home/LiveTicker";

export function HomeChromeLayout() {
  return (
    <div className="min-h-screen bg-black text-white">
      <HomeNav />
      <LiveTicker />
      <Outlet />
    </div>
  );
}
```

`frontend/src/pages/AboutPage.tsx`:

```tsx
import { AboutContent } from "@/components/about/AboutContent";

export function AboutPage() {
  return <AboutContent />;
}
```

`frontend/src/pages/HomePage.tsx` — remove `HomeNav` and `LiveTicker`; drop the outer `min-h-screen bg-black text-white` wrapper (layout owns it):

```tsx
import { TicketHero } from "@/components/home/TicketHero";
import { LiveNowSection } from "@/components/home/LiveNowSection";
import { StoriesSection } from "@/components/home/StoriesSection";
import { ExploreSection } from "@/components/home/ExploreSection";
import { LearnTheGameSection } from "@/components/home/LearnTheGameSection";

export function HomePage() {
  return (
    <>
      <TicketHero />
      <LiveNowSection />
      <StoriesSection />
      <ExploreSection />
      <LearnTheGameSection />
    </>
  );
}
```

`frontend/src/AppRouter.tsx`:

```tsx
import { Routes, Route } from "react-router-dom";
import { HomeChromeLayout } from "@/layouts/HomeChromeLayout";
import { HomePage } from "@/pages/HomePage";
import { AboutPage } from "@/pages/AboutPage";
import { NotFoundPage } from "@/pages/NotFoundPage";

export function AppRouter() {
  return (
    <Routes>
      <Route element={<HomeChromeLayout />}>
        <Route path="/" element={<HomePage />} />
        <Route path="/about" element={<AboutPage />} />
      </Route>
      <Route path="*" element={<NotFoundPage />} />
    </Routes>
  );
}
```

- [ ] **Step 4: Run router + about + nav tests**

Run: `cd frontend && npm run test -- src/AppRouter.test.tsx src/components/about/AboutContent.test.tsx src/components/home/HomeNav.test.tsx`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/layouts/HomeChromeLayout.tsx frontend/src/pages/AboutPage.tsx frontend/src/pages/HomePage.tsx frontend/src/AppRouter.tsx frontend/src/AppRouter.test.tsx
git commit -m "Add /about route under shared home chrome layout."
```

---

### Task 4: Full test suite + build verification

**Files:**
- Verify only (no intentional product changes)

**Interfaces:**
- Consumes: Tasks 1–3

- [ ] **Step 1: Run full frontend tests**

Run: `cd frontend && npm run test`

Expected: PASS (all existing + new tests)

- [ ] **Step 2: Run production build**

Run: `cd frontend && npm run build`

Expected: exit 0

- [ ] **Step 3: Manual smoke (optional but recommended)**

With `npm run dev`: open `/` (home + nav), click About → `/about` (badge, headline, pills, copy, ticker still visible), logo → `/`.

- [ ] **Step 4: Commit only if Step 1–2 required small fixes**

```bash
# only if fixes were needed
git add -u frontend/src
git commit -m "Fix About page wiring after test and build verification."
```

If no fixes, skip commit.

---

## Spec coverage checklist

| Spec requirement | Task |
| --- | --- |
| `/about` route | Task 3 |
| Shared HomeNav + LiveTicker chrome | Task 3 |
| Badge `SPORTS ANALYTICS · BETA` | Task 1 |
| Headline `About HoopVista.` | Task 1 |
| NBA + WNBA pills only | Task 1 |
| HoopVista-specific body copy | Task 1 |
| No Contributors | Task 1 assertion |
| About active nav pill | Task 2 |
| League links `/#live-now` | Task 2 |
| Components under `components/about/` | Task 1 |
| NotFound outside chrome | Task 3 (`*` sibling route) |
