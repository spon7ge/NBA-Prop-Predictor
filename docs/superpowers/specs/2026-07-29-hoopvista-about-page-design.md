# HoopVista About Page Design

Date: 2026-07-29  
Status: Approved for planning

## Goal

Add an About experience matching the boxseats About mockup structure, branded as **HoopVista**: shared home chrome (nav + ticker), dedicated `/about` route, beta badge, headline, NBA/WNBA league pills, and product-specific body copy. No Contributors section in v1.

## Decisions

| Topic | Choice |
| --- | --- |
| Routing | Separate route `/about` (not in-page tab or home anchor) |
| Chrome | Shared layout: `HomeNav` + `LiveTicker` for `/` and `/about` |
| League pills | NBA + WNBA only |
| Contributors | Out of scope for v1 |
| Copy | HoopVista-specific (basketball analytics / props for NBA + WNBA) |
| Component folder | `frontend/src/components/about/` (not under `home/`) |

## Routing

| Path | Element |
| --- | --- |
| `/` | Home sections under chrome layout |
| `/about` | About content under chrome layout |
| `*` | Unchanged `NotFoundPage` |

Router nests home and about under a chrome layout route so nav and ticker are not duplicated:

```tsx
<Routes>
  <Route element={<HomeChromeLayout />}>
    <Route path="/" element={<HomePage />} />
    <Route path="/about" element={<AboutPage />} />
  </Route>
  <Route path="*" element={<NotFoundPage />} />
</Routes>
```

## Page structure

**HomeChromeLayout**

1. `HomeNav`
2. `LiveTicker`
3. `<Outlet />` for page body

**About page body** (centered column, `max-w` ~2xl–3xl, generous top padding, left-aligned on black):

1. **Badge** — pill: `SPORTS ANALYTICS • BETA`
2. **Headline** — `About HoopVista.`
3. **League pills** — NBA and WNBA with colored borders / accents consistent with home
4. **Body** — 2–3 light-gray paragraphs:
   - What HoopVista is (NBA/WNBA basketball analytics and props)
   - Approachability (plain-language explainers, honest charts)
   - Beta note (still building and polishing)

No cards. Static copy in the About component (no API/CMS).

## Nav behavior

- Logo → `/`
- NBA / WNBA → `/#live-now` (works from home and about)
- About → `/about`; active pill styling when `pathname === "/about"`
- Settings gear remains decorative (`aria-label`, no modal)

`HomePage` no longer renders its own `HomeNav` / `LiveTicker`; those move into the layout.

## File layout

```
frontend/src/layouts/HomeChromeLayout.tsx
frontend/src/pages/AboutPage.tsx
frontend/src/pages/HomePage.tsx          # sections only (hero onward)
frontend/src/components/about/           # About UI lives here
  AboutContent.tsx                       # badge, headline, pills, copy
frontend/src/components/home/HomeNav.tsx # Link to /about + active state
frontend/src/AppRouter.tsx               # nest / and /about under chrome
```

`AboutPage` is a thin page that renders `AboutContent` from `components/about/`.

## Visual system

- Same dark landing tokens as home: black background, white/gray text, Geist Sans.
- Pill badge and league chips with rounded-full borders; no card containers.
- League accents for NBA / WNBA only (reuse existing home accent pattern / icons where practical).

## Data

- No API calls. Static strings in `AboutContent`.
- Ticker empty state unchanged (optional props for future data).

## Out of scope

- Contributors section
- Settings modal
- Real live scores
- Leagues beyond NBA / WNBA
- CMS or editable copy
- Shared shell with NotFound (NotFound stays outside chrome layout)

## Success criteria

- Visiting `/about` shows nav + ticker + About content (badge, headline, two pills, body copy).
- Visiting `/` still shows the full home landing under the same chrome.
- About nav control links to `/about` and appears active on that route.
- Logo returns to `/`; league links go to `/#live-now`.
- No Contributors block; build passes with tests for nav active link and About content presence.
