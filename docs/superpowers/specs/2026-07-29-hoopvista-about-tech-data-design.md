# HoopVista About Tech Stack & Data Sources Design

Date: 2026-07-29  
Status: Approved for implementation

## Goal

Extend `/about` with **Tech stack** and **Data sources** sections matching the boxseats About mockup layout, using HoopVista branding and current-truth stack/source lists. No system-design CTA.

## Decisions

| Topic | Choice |
| --- | --- |
| Honesty | Current truth only (no aspirational / removed deps) |
| System-design CTA | Out of scope |
| Data sources | Pipeline truth (NBA Stats, Odds API, BRef WNBA, Supabase) |
| Structure | Split components under `components/about/` |
| Width | Widen About page column to `max-w-4xl` |

## Page composition (`AboutContent`)

1. Existing intro (badge, headline, NBA/WNBA pills, body copy)
2. **TechStackSection**
3. **DataSourcesSection**

## Tech stack

- Heading: `Tech stack`
- Dark rounded bordered card; three rows with label + wrapping tech list
- Labels: `FRONTEND`, `BACKEND`, `INFRA & TOOLING` (muted uppercase)
- Tech text: light gray, mono if available (`font-mono`)

**Lists (v1):**

| Row | Items |
| --- | --- |
| Frontend | React 19, TypeScript, Vite, Tailwind CSS v4, React Router, Geist, lucide-react, Vitest |
| Backend | Python, FastAPI, Pandas, NumPy, scikit-learn, XGBoost, nba_api, SQLAlchemy, Supabase/PostgreSQL, joblib |
| Infra & tooling | GitHub Actions, GitHub Pages |

No CTA bar under the card.

## Data sources

- Heading: `Data sources`
- Subcopy: e.g. “It’s all public or licensed data feeds we use in the pipeline. Every chart should be able to say where its numbers came from.”
- List rows: league badge(s) + title + description + external-link icon (`target="_blank"`, `rel="noopener noreferrer"`)

**Rows (v1):**

| Badge | Title | Description | Href |
| --- | --- | --- | --- |
| NBA | NBA Stats API | Schedules, box scores, team and player game logs via `nba_api`. | https://github.com/swar/nba_api |
| NBA / WNBA | The Odds API | Player prop lines for NBA and WNBA. | https://the-odds-api.com |
| WNBA | Basketball-Reference | WNBA per-game / position tables used for player context. | https://www.basketball-reference.com/wnba/ |
| Shared | Supabase (PostgreSQL) | Stored raw tables and engineered features. | https://supabase.com |

League badge colors: NBA sky, WNBA violet (match About pills). Shared badge: muted gray.

## File layout

```
frontend/src/components/about/AboutContent.tsx      # compose intro + sections; max-w-4xl
frontend/src/components/about/TechStackSection.tsx
frontend/src/components/about/TechStackSection.test.tsx
frontend/src/components/about/DataSourcesSection.tsx
frontend/src/components/about/DataSourcesSection.test.tsx
frontend/src/components/about/AboutContent.test.tsx  # extend or keep section tests primary
```

## Out of scope

- System-design CTA / docs page
- Listing removed frontend deps (TanStack Query, D3, Recharts, shadcn)
- Contributors
- New routes
- Editable CMS

## Success criteria

- `/about` shows intro, then Tech stack card with three truthful rows, then Data sources list with working external links
- No system-design CTA
- Tests cover stack labels/tech strings and source titles/links
- `npm run test` and `npm run build` pass in `frontend/`
