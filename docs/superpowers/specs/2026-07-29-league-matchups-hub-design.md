# League matchups hub (WNBA + NBA stub)

Date: 2026-07-29  
Status: Approved for planning

## Goal

When a user clicks **WNBA** (or **NBA**) in `HomeNav`, navigate to a league hub that matches the provided boxseats-style matchups mockup: league hero, Explore/Learn subnav, and a Matchups panel. Ship **WNBA Matchups** with live scoreboard data (including venue and W-L records). Ship **NBA** as the same shell with a coming-soon empty state. Brand as **HoopVista**.

## Decisions

| Topic | Choice |
| --- | --- |
| Scope | Full visual page; Matchups live for WNBA; other Explore/Learn tabs visible but disabled |
| Date switcher | Today only; prev/next arrows disabled (no dated API in v1) |
| Venue + records | Extend scoreboard payload from ESPN |
| Route | `/:league/matchups` — concrete paths `/wnba/matchups` and `/nba/matchups` |
| Branding | HoopVista everywhere (including “HoopVista Picks”) |
| NBA | Same layout shell; Matchups body is coming-soon (no NBA scoreboard) |
| Architecture | Shared `LeagueMatchupsPage` (Approach 1) |
| Chrome | Remains under `HomeChromeLayout` (nav + live ticker) |

## Architecture

```
HomeNav WNBA / NBA click
        │
        ▼
  /:league/matchups     (HomeChromeLayout — nav + ticker stay)
        │
        ├── league === "wnba"
        │     useWnbaScoreboard() → MatchupsPanel (LIVE NOW + REST OF THE SLATE)
        │
        └── league === "nba"
              same hero + subnav; coming-soon empty body
```

### Routing

| Path | Element |
| --- | --- |
| `/wnba/matchups` | `LeagueMatchupsPage` with WNBA content + live Matchups |
| `/nba/matchups` | `LeagueMatchupsPage` with NBA shell + coming-soon body |
| Other `/:league/matchups` | 404 (`NotFoundPage` or explicit guard) |

- HomeNav WNBA → `Link` to `/wnba/matchups`; active (violet pill) when pathname starts with `/wnba`.
- HomeNav NBA → `Link` to `/nba/matchups`; active (sky pill) when pathname starts with `/nba`.
- Matchup cards → `/games/:espnEventId` when `espnEventId` is set (existing game detail).
- Back from game detail remains `/` per existing game-detail spec (unchanged in this work).

## Page structure

Vertical stack under nav + ticker:

1. **LeagueHero** — rounded dark banner: league pill, weekday/date label, title (“Women’s Basketball” / “Men’s Basketball”), short supporting blurb, faint basketball watermark.
2. **LeagueSubnav** — EXPLORE + LEARN pill groups. **Matchups** is the only active/enabled item. Others are visible, muted, and non-navigating (`aria-disabled` / `disabled` buttons or spans): HoopVista Picks, Leaders, Standings, Playoff race, Clutch, How it works, Glossary.
3. **MatchupsPanel**
   - Header: “Matchups” + date control labeled **Today** with disabled chevrons.
   - Subcopy: `{n} games · open a card for box score, play-by-play & win probability`.
   - **LIVE NOW** — games with status `live` or `halftime`.
   - **REST OF THE SLATE** — `scheduled` and `final` (and any other non-in-progress).
   - Cards: status/clock (live dot for in-progress), venue · city when available, team rows (letter avatar, abbrev, name, record, amber score box), chevron; left accent bar (WNBA violet; optional gradient on finals).
   - Loading / empty / error: skeletons or muted copy; never replace good data with an error wipe (same pattern as home LIVE NOW).
4. **NBA body** — hero + subnav as above; instead of cards, a short “NBA matchups coming soon” empty state.

### Visual system

- Dark HoopVista landing: black page, charcoal cards (`~#141414`), white/gray text.
- Existing Geist Sans + Geist Mono (no new Inter font).
- WNBA accent violet; NBA accent sky — pills, active nav, live indicators, card accent bars.
- Light motion: pulsing live dot only in v1.

## Data & API

Keep `GET /api/wnba/scoreboard/today`. Extend models:

### Team (`WnbaTeam`)

| Field | Type | Notes |
| --- | --- | --- |
| `abbrev`, `name`, `score` | existing | unchanged |
| `record` | `string \| null` | e.g. `"19-8"` from ESPN competitor records |

### Game (`WnbaGame`)

| Field | Type | Notes |
| --- | --- | --- |
| existing fields | — | unchanged |
| `venue` | `string \| null` | ESPN competition venue `fullName` / `name` |
| `venue_city` | `string \| null` | ESPN venue address city when present |

### Mapping rules

- Parse venue + records in `normalize_espn_scoreboard`.
- Stats normalize leaves `venue`, `venue_city`, and team `record` as `null`.
- `merge_games` uses `prefer_complete` so ESPN extras survive when stats wins id/scores.
- Frontend maps into a `MatchupGame` (or extended `LiveGame`) with `venue`, `venueCity`, and team `record`.
- Polling: reuse `useWnbaScoreboard` (~18s while any game is in progress).
- NBA page: no scoreboard fetch in v1.

### Display rules

- Venue line: `venue` alone, or `venue · city` when city present; omit line if both null.
- Record: show when non-null; omit (no placeholder dash) when null.
- Scores: show for live/halftime/final; scheduled may show `–` or hide per existing home card pattern (prefer amber box with `–` for layout stability).
- Winning record styling (green) is optional polish; gray/default is fine if win/loss parsing is awkward — prefer simple muted record text unless W-L is trivially comparable.

## File layout

```
frontend/src/pages/LeagueMatchupsPage.tsx
frontend/src/components/league/
  LeagueHero.tsx
  LeagueSubnav.tsx
  MatchupsPanel.tsx
  MatchupGameCard.tsx
  types.ts
frontend/src/components/home/HomeNav.tsx          # Links + active pills
frontend/src/AppRouter.tsx                       # /:league/matchups
frontend/src/components/home/mapScoreboard.ts     # venue/record mapping
frontend/src/components/home/types.ts             # or league/types shared shape
frontend/src/lib/api.ts                          # scoreboard TypeScript types

backend/app/schemas/wnba_scoreboard.py           # + venue, venue_city, record
backend/app/services/wnba_scoreboard.py           # ESPN parse + merge
```

## Testing

### Backend

- ESPN fixture → `venue`, `venue_city`, and team `record` populated when present.
- Merge: stats-preferred game still retains ESPN `espn_event_id`, venue, and records when ESPN had them.
- Existing scoreboard route tests still pass (`200`, stale-while-error behavior unchanged).

### Frontend

- HomeNav: WNBA/NBA link to the correct matchups paths; active pill on league routes; inactive styling elsewhere.
- Router: `/wnba/matchups` mounts hub; `/nba/matchups` shows coming-soon; invalid league → not found.
- MatchupsPanel: splits live vs rest; cards link when `espnEventId` set; venue/record render when present; date arrows disabled.
- Subnav: only Matchups is interactive; others are disabled.
- `npm run build` (frontend) and relevant backend scoreboard tests pass.

## Out of scope

- Real date navigation or `GET /api/wnba/scoreboard?date=`
- Implementing Explore/Learn destinations (Picks, Leaders, Standings, etc.)
- NBA live scoreboard or ESPN NBA summary
- Win probability charts, box-score tables beyond existing `/games/:espnEventId`
- Multi-sport leagues beyond NBA / WNBA
- Changing game-detail Back target to return to matchups (can be a later polish)

## Success criteria

- Clicking WNBA in the header opens `/wnba/matchups` matching the mockup structure (hero, subnav, Matchups list).
- WNBA Matchups shows today’s live and non-live games from the scoreboard with venue/records when ESPN provides them.
- Cards with ESPN ids open the existing game detail page.
- Clicking NBA opens `/nba/matchups` with the same chrome and a clear coming-soon body.
- Other subnav items are visible but do not navigate.
- Date control shows Today with disabled arrows.
