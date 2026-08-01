# Homepage Apple-Style Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the HoopVista homepage to a calm Apple-style dark marketing page with BrandHero (logo slideshow), Live (team icons kept), Stories, Feature strip, and League CTAs.

**Architecture:** Replace `TicketHero` with `BrandHero` + `LeagueLogoSlideshow`. Restyle existing home chrome/sections; wire `HomePage` to the new section order; leave Explore/Learn/TicketHero unused on the page.

**Tech Stack:** React 19 · TypeScript · Vite · Tailwind v4 · Vitest · React Router · Geist · lucide-react

## Global Constraints

- Dark-only this pass
- No new APIs; keep `useWnbaScoreboard`
- Live now retains team icons
- Stories after Live; default max 3 stories
- Prefer theme tokens / opacity utilities over new hardcoded hex
- TDD for new components; update existing tests as needed
- Spec: `docs/superpowers/specs/2026-08-01-homepage-apple-style-rebuild-design.md`

## File map

| File | Responsibility |
| --- | --- |
| `frontend/src/components/home/LeagueLogoSlideshow.tsx` | NBA↔WNBA floating logo crossfade |
| `frontend/src/components/home/LeagueLogoSlideshow.test.tsx` | Slideshow + reduced-motion tests |
| `frontend/src/components/home/BrandHero.tsx` | Split hero: copy left, slideshow right |
| `frontend/src/components/home/BrandHero.test.tsx` | Hero copy + CTA `#live-now` |
| `frontend/src/components/home/FeatureStrip.tsx` | Props / Edges / Explain |
| `frontend/src/components/home/FeatureStrip.test.tsx` | Feature strip render |
| `frontend/src/components/home/LeagueCtaSection.tsx` | NBA / WNBA entry links |
| `frontend/src/components/home/LeagueCtaSection.test.tsx` | CTA hrefs |
| `frontend/src/pages/HomePage.tsx` | Section composition |
| `frontend/src/pages/HomePage.test.tsx` | Page section presence (new) |
| `frontend/src/components/home/LiveNowSection.tsx` | Quieter restyle; keep icons |
| `frontend/src/components/home/StoriesSection.tsx` | Quieter restyle; cap defaults at 3 |
| `frontend/src/components/home/SectionHeading.tsx` | Title style closer to display hierarchy |
| `frontend/src/components/home/HomeNav.tsx` | Quieter active states |
| `frontend/src/components/home/LiveTicker.tsx` | Quieter muted colors |
| `frontend/src/index.css` | Optional logo fade keyframes + reduced-motion |

---

### Task 1: LeagueLogoSlideshow

**Files:**
- Create: `frontend/src/components/home/LeagueLogoSlideshow.tsx`
- Test: `frontend/src/components/home/LeagueLogoSlideshow.test.tsx`

**Interfaces:**
- Produces: `export function LeagueLogoSlideshow(): JSX.Element`

- [ ] **Step 1: Write failing tests**

```tsx
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { LeagueLogoSlideshow } from "./LeagueLogoSlideshow";

describe("LeagueLogoSlideshow", () => {
  beforeEach(() => {
    Object.defineProperty(window, "matchMedia", {
      writable: true,
      value: vi.fn().mockImplementation((query: string) => ({
        matches: query.includes("prefers-reduced-motion"),
        media: query,
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        addListener: vi.fn(),
        removeListener: vi.fn(),
        dispatchEvent: vi.fn(),
      })),
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders NBA and WNBA logo images", () => {
    // matchMedia returns matches:false for reduced motion unless query includes it — override
    window.matchMedia = vi.fn().mockImplementation(() => ({
      matches: false,
      media: "",
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn(),
    }));
    render(<LeagueLogoSlideshow />);
    expect(screen.getByAltText("NBA")).toBeInTheDocument();
    expect(screen.getByAltText("WNBA")).toBeInTheDocument();
  });

  it("shows only the first logo when prefers-reduced-motion is set", () => {
    window.matchMedia = vi.fn().mockImplementation((query: string) => ({
      matches: query === "(prefers-reduced-motion: reduce)",
      media: query,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn(),
    }));
    render(<LeagueLogoSlideshow />);
    expect(screen.getByAltText("NBA")).toBeInTheDocument();
    expect(screen.queryByAltText("WNBA")).not.toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `cd frontend && npm test -- src/components/home/LeagueLogoSlideshow.test.tsx`

- [ ] **Step 3: Implement**

Use existing assets `@/assets/basketball.png` and `@/assets/wnba_basketball.png`. No border/card. Absolute-positioned images with CSS animation classes from `index.css` (or Tailwind `animate-*` if defined). Under reduced motion, render only NBA image.

- [ ] **Step 4: Run tests — PASS**

- [ ] **Step 5: Commit** `feat: add league logo slideshow for homepage hero`

---

### Task 2: BrandHero

**Files:**
- Create: `frontend/src/components/home/BrandHero.tsx`
- Test: `frontend/src/components/home/BrandHero.test.tsx`
- Modify: `frontend/src/index.css` (keyframes if not added in Task 1)

**Interfaces:**
- Consumes: `LeagueLogoSlideshow`
- Produces: `export function BrandHero(): JSX.Element`

- [ ] **Step 1: Failing test**

```tsx
it("renders brand copy and CTA to live now", () => {
  render(<BrandHero />);
  expect(screen.getByRole("heading", { name: /hoopvista/i })).toBeInTheDocument();
  expect(screen.getByText(/season pass/i)).toBeInTheDocument();
  expect(screen.getByRole("link", { name: /see what.?s live/i })).toHaveAttribute(
    "href",
    "#live-now",
  );
});
```

- [ ] **Step 2: Implement** — grid `lg:grid-cols-2`; copy left; slideshow right; mobile stacks copy then logos; generous padding; no ticket chrome.

- [ ] **Step 3: Tests pass + commit** `feat: add BrandHero with split logo slideshow`

---

### Task 3: FeatureStrip + LeagueCtaSection

**Files:**
- Create: `frontend/src/components/home/FeatureStrip.tsx` (+ test)
- Create: `frontend/src/components/home/LeagueCtaSection.tsx` (+ test)

- [ ] FeatureStrip: heading “Built for clarity”; three columns Props / Edges / Explain with short copy from the mockup.
- [ ] LeagueCtaSection: “Enter a league”; links to `/nba/matchups` and `/wnba/matchups` (MemoryRouter in tests).
- [ ] Commit `feat: add homepage feature strip and league CTAs`

---

### Task 4: Restyle LiveNow, Stories, SectionHeading

**Files:**
- Modify: `LiveNowSection.tsx`, `StoriesSection.tsx`, `SectionHeading.tsx`
- Update tests if copy/structure assertions break

- [ ] LiveNow: quieter league label (`text-white/40` text, not bright pill); softer score treatment; keep `TeamAbbrevAvatar`; keep `#live-now`.
- [ ] Stories: typography-led cards; muted league label; drop loud graphic icons OR keep very muted; **default list capped to first 3** of `DEFAULT_STORIES`.
- [ ] SectionHeading: sentence-case / larger title (`text-2xl font-semibold tracking-tight text-white`) instead of all-caps tiny tracking — update tests looking for exact heading text (“Live Now”, “Stories”).
- [ ] Commit `feat: quiet restyle for live and stories sections`

---

### Task 5: Quiet HomeNav + LiveTicker

**Files:**
- Modify: `HomeNav.tsx`, `LiveTicker.tsx` (+ tests if class assertions fail)

- [ ] Nav: active league = subtle `text-white` + underline or soft `bg-white/10` pill (not sky/violet fills).
- [ ] Ticker: replace sky/rose abbrev colors with `text-white/80` / `text-white/55`; keep structure and reduced-motion behavior.
- [ ] Commit `feat: quiet homepage nav and ticker chrome`

---

### Task 6: Wire HomePage

**Files:**
- Modify: `frontend/src/pages/HomePage.tsx`
- Create: `frontend/src/pages/HomePage.test.tsx`

- [ ] Composition order: BrandHero → LiveNowSection → StoriesSection → FeatureStrip → LeagueCtaSection
- [ ] Remove TicketHero, Explore, Learn imports
- [ ] Test: renders Live Now, Stories, Built for clarity, Enter a league; CTA to `#live-now`; no “Learn the Game” / Explore headings
- [ ] Run full `cd frontend && npm test`
- [ ] Commit `feat: wire Apple-style homepage section stack`

---

## Self-review

- Spec coverage: hero slideshow, live+icons, stories after live, features, league CTAs, quiet chrome, dark-only, reduced-motion, tests — all tasked
- No light mode / league page work — correctly out of plan
