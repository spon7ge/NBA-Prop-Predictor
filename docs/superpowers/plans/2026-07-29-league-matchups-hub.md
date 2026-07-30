# League Matchups Hub Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clicking WNBA/NBA in `HomeNav` opens `/:league/matchups` with a boxseats-style hub (hero, Explore/Learn subnav, Matchups panel); WNBA uses live scoreboard data including venue and W-L records; NBA is the same shell with a coming-soon body.

**Architecture:** Shared `LeagueMatchupsPage` under `HomeChromeLayout`. Extend `GET /api/wnba/scoreboard/today` with `venue`, `venue_city`, and team `record` from ESPN (preserved through merge). Frontend maps all games to `MatchupGame`, splits LIVE NOW vs REST OF THE SLATE, and links cards to `/games/:espnEventId` when present.

**Tech Stack:** FastAPI · Pydantic · React 19 · TypeScript · Vite 6 · TanStack Query · React Router · Vitest · Testing Library · Tailwind 4

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-29-league-matchups-hub-design.md`
- Coding standards: `CLAUDE.md` (small focused modules, strong typing, tests with code)
- Brand: **HoopVista** everywhere (including “HoopVista Picks”)
- Routes: `/wnba/matchups`, `/nba/matchups`; other leagues → 404
- Date control: **Today** only; prev/next arrows **disabled**
- Explore/Learn tabs other than Matchups: visible, muted, non-navigating
- Scoreboard endpoint stays `GET /api/wnba/scoreboard/today` (no dated API)
- Reuse `useWnbaScoreboard` polling; NBA page fetches no scoreboard
- Chrome: keep `HomeChromeLayout` (nav + ticker)
- Verify backend: `python3 -m pytest backend/tests/test_wnba_scoreboard_normalize.py -v`
- Verify frontend: `cd frontend && npm run test && npm run build`

---

## File Structure

| File | Responsibility |
|------|----------------|
| `backend/app/schemas/wnba_scoreboard.py` | Add `record` on team; `venue` / `venue_city` on game |
| `backend/app/services/wnba_scoreboard.py` | ESPN parse venue/records; merge preserves them |
| `backend/tests/fixtures/espn_wnba_scoreboard.json` | Add venue + records to fixture |
| `backend/tests/test_wnba_scoreboard_normalize.py` | Normalize + merge coverage |
| `frontend/src/lib/api.ts` | Mirror new scoreboard fields |
| `frontend/src/components/league/types.ts` | `MatchupGame`, league meta |
| `frontend/src/components/home/mapScoreboard.ts` | `mapToMatchupGames` |
| `frontend/src/components/home/mapScoreboard.test.ts` | Mapper tests |
| `frontend/src/components/league/MatchupGameCard.tsx` | Full-width matchup card |
| `frontend/src/components/league/MatchupGameCard.test.tsx` | Card / link / venue / record |
| `frontend/src/components/league/MatchupsPanel.tsx` | Header, date control, LIVE NOW / REST |
| `frontend/src/components/league/MatchupsPanel.test.tsx` | Split + empty/loading |
| `frontend/src/components/league/LeagueHero.tsx` | League banner |
| `frontend/src/components/league/LeagueHero.test.tsx` | Title / pill by league |
| `frontend/src/components/league/LeagueSubnav.tsx` | Explore/Learn pills |
| `frontend/src/components/league/LeagueSubnav.test.tsx` | Matchups active; others disabled |
| `frontend/src/pages/LeagueMatchupsPage.tsx` | Compose hub; WNBA vs NBA body |
| `frontend/src/AppRouter.tsx` | `/:league/matchups` under chrome |
| `frontend/src/AppRouter.test.tsx` | Route smoke |
| `frontend/src/components/home/HomeNav.tsx` | Links + active pills |
| `frontend/src/components/home/HomeNav.test.tsx` | Href + aria-current |

---

### Task 1: Scoreboard venue + records (backend)

**Files:**
- Modify: `backend/app/schemas/wnba_scoreboard.py`
- Modify: `backend/app/services/wnba_scoreboard.py`
- Modify: `backend/tests/fixtures/espn_wnba_scoreboard.json`
- Modify: `backend/tests/test_wnba_scoreboard_normalize.py`

**Interfaces:**
- Produces: `WnbaTeam.record: str | None`; `WnbaGame.venue: str | None`; `WnbaGame.venue_city: str | None`
- Consumes: existing `normalize_espn_scoreboard`, `normalize_stats_scoreboard`, `merge_games`, `prefer_complete`

- [ ] **Step 1: Extend the ESPN fixture**

In `backend/tests/fixtures/espn_wnba_scoreboard.json`, add venue on the competition and records on each competitor:

```json
{
  "events": [
    {
      "id": "401749001",
      "date": "2026-07-29T23:00Z",
      "competitions": [
        {
          "venue": {
            "fullName": "College Park Center",
            "address": { "city": "Arlington" }
          },
          "competitors": [
            {
              "homeAway": "home",
              "score": "44",
              "records": [{ "type": "total", "summary": "18-10" }],
              "team": {
                "abbreviation": "DAL",
                "displayName": "Dallas Wings"
              }
            },
            {
              "homeAway": "away",
              "score": "36",
              "records": [{ "type": "total", "summary": "17-10" }],
              "team": {
                "abbreviation": "ATL",
                "displayName": "Atlanta Dream"
              }
            }
          ]
        }
      ],
      "status": {
        "type": {
          "state": "in",
          "completed": false,
          "name": "STATUS_IN_PROGRESS",
          "shortDetail": "7:13 - 3rd",
          "detail": "7:13 - 3rd Quarter"
        },
        "period": 3,
        "displayClock": "7:13"
      }
    }
  ]
}
```

- [ ] **Step 2: Write the failing tests**

Add to `backend/tests/test_wnba_scoreboard_normalize.py`:

```python
def test_normalize_espn_sets_venue_and_records():
    payload = json.loads((FIXTURES / "espn_wnba_scoreboard.json").read_text())
    g = normalize_espn_scoreboard(payload, date_et="2026-07-29")[0]
    assert g.venue == "College Park Center"
    assert g.venue_city == "Arlington"
    assert g.away.record == "17-10"
    assert g.home.record == "18-10"


def test_merge_preserves_venue_and_records_when_stats_id_wins():
    espn = [
        WnbaGame(
            id="espn-401749001",
            espn_event_id="401749001",
            status="live",
            status_label="Q3 7:13",
            away=WnbaTeam(
                abbrev="ATL", name="Atlanta Dream", score=36, record="17-10"
            ),
            home=WnbaTeam(
                abbrev="DAL", name="Dallas Wings", score=44, record="18-10"
            ),
            start_time_et="2026-07-29T23:00:00Z",
            venue="College Park Center",
            venue_city="Arlington",
        )
    ]
    stats = [
        WnbaGame(
            id="1022600123",
            espn_event_id=None,
            status="live",
            status_label="Q3 7:13",
            away=WnbaTeam(abbrev="ATL", name="Atlanta Dream", score=36),
            home=WnbaTeam(abbrev="DAL", name="Dallas Wings", score=44),
            start_time_et="2026-07-29T23:00:00Z",
        )
    ]
    merged = merge_games(espn, stats)
    assert merged[0].id == "1022600123"
    assert merged[0].venue == "College Park Center"
    assert merged[0].venue_city == "Arlington"
    assert merged[0].away.record == "17-10"
    assert merged[0].home.record == "18-10"
```

Also update the inline `espn_event()` helper (or leave it) so existing tests that construct `WnbaGame` / `WnbaTeam` still type-check — new fields default to `None`.

- [ ] **Step 3: Run tests to verify they fail**

Run: `python3 -m pytest backend/tests/test_wnba_scoreboard_normalize.py::test_normalize_espn_sets_venue_and_records backend/tests/test_wnba_scoreboard_normalize.py::test_merge_preserves_venue_and_records_when_stats_id_wins -v`

Expected: FAIL (missing fields / AttributeError / validation)

- [ ] **Step 4: Minimal schema + normalize + merge**

In `backend/app/schemas/wnba_scoreboard.py`:

```python
class WnbaTeam(BaseModel):
    abbrev: str
    name: str
    score: int | None = None
    record: str | None = None


class WnbaGame(BaseModel):
    id: str
    espn_event_id: str | None = None
    league: Literal["wnba"] = "wnba"
    status: GameStatus
    status_label: str
    away: WnbaTeam
    home: WnbaTeam
    start_time_et: str
    venue: str | None = None
    venue_city: str | None = None
```

In `backend/app/services/wnba_scoreboard.py`, add helpers and wire them into ESPN normalize + merge:

```python
def _espn_team_record(competitor: dict) -> str | None:
    for rec in competitor.get("records") or []:
        if not isinstance(rec, dict):
            continue
        if rec.get("type") == "total" or str(rec.get("name") or "").lower() in {
            "total",
            "overall",
            "ytd",
        }:
            summary = rec.get("summary")
            if summary:
                return str(summary)
    for rec in competitor.get("records") or []:
        if isinstance(rec, dict) and rec.get("summary"):
            return str(rec["summary"])
    return None


def _espn_venue(comps: dict) -> tuple[str | None, str | None]:
    venue = comps.get("venue") or {}
    if not isinstance(venue, dict):
        return None, None
    name = venue.get("fullName") or venue.get("name")
    city = (venue.get("address") or {}).get("city") if isinstance(
        venue.get("address"), dict
    ) else None
    return (str(name) if name else None, str(city) if city else None)
```

In `normalize_espn_scoreboard`, when building each team include `record=_espn_team_record(c)`, and on `WnbaGame` set `venue` / `venue_city` from `_espn_venue(comps)`.

In `merge_games`, when constructing the merged `WnbaGame`:

```python
away=WnbaTeam(
    abbrev=str(prefer_complete(a.away.abbrev, g.away.abbrev)),
    name=str(prefer_complete(a.away.name, g.away.name)),
    score=prefer_complete(a.away.score, g.away.score),
    record=prefer_complete(a.away.record, g.away.record) or None,
),
home=WnbaTeam(
    abbrev=str(prefer_complete(a.home.abbrev, g.home.abbrev)),
    name=str(prefer_complete(a.home.name, g.home.name)),
    score=prefer_complete(a.home.score, g.home.score),
    record=prefer_complete(a.home.record, g.home.record) or None,
),
start_time_et=str(prefer_complete(a.start_time_et, g.start_time_et)),
venue=prefer_complete(a.venue, g.venue) or None,
venue_city=prefer_complete(a.venue_city, g.venue_city) or None,
```

Note: `prefer_complete` currently accepts `str | int | None`. Passing `None` for missing records/venues is fine; empty string should coerce to `None` via `or None`. If type checker complains about `prefer_complete(a.venue, g.venue)` when both can be `None`, cast or extend overloads to include `str | None` pairs (empty string treated as incomplete — already handled).

Stats normalize needs no venue/record assignment (defaults `None`).

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest backend/tests/test_wnba_scoreboard_normalize.py -v`

Expected: PASS (all normalize/merge tests)

- [ ] **Step 6: Commit**

```bash
git add backend/app/schemas/wnba_scoreboard.py backend/app/services/wnba_scoreboard.py \
  backend/tests/fixtures/espn_wnba_scoreboard.json \
  backend/tests/test_wnba_scoreboard_normalize.py
git commit -m "$(cat <<'EOF'
Add venue and team records to WNBA scoreboard.

EOF
)"
```

---

### Task 2: Frontend scoreboard types + `mapToMatchupGames`

**Files:**
- Modify: `frontend/src/lib/api.ts`
- Create: `frontend/src/components/league/types.ts`
- Modify: `frontend/src/components/home/mapScoreboard.ts`
- Modify: `frontend/src/components/home/mapScoreboard.test.ts`

**Interfaces:**
- Consumes: `ApiWnbaGame` with new fields
- Produces:
  - `MatchupTeam = { abbrev: string; name: string; score: number | null; record?: string | null }`
  - `MatchupGame = { id; espnEventId?; league; status; statusLabel; venue?; venueCity?; away; home }`
  - `mapToMatchupGames(games: ApiWnbaGame[]): MatchupGame[]` — **all** games (not filtered)

- [ ] **Step 1: Write the failing mapper test**

Add to `frontend/src/components/home/mapScoreboard.test.ts`:

```ts
it("mapToMatchupGames maps all games with venue and records", () => {
  const games: ApiWnbaGame[] = [
    {
      id: "1",
      espn_event_id: "401",
      league: "wnba",
      status: "final",
      status_label: "Final",
      away: {
        abbrev: "ATL",
        name: "Atlanta Dream",
        score: 82,
        record: "17-10",
      },
      home: {
        abbrev: "DAL",
        name: "Dallas Wings",
        score: 81,
        record: "18-10",
      },
      start_time_et: "2026-07-29T23:00:00Z",
      venue: "College Park Center",
      venue_city: "Arlington",
    },
  ];
  expect(mapToMatchupGames(games)).toEqual([
    {
      id: "1",
      espnEventId: "401",
      league: "wnba",
      status: "final",
      statusLabel: "Final",
      venue: "College Park Center",
      venueCity: "Arlington",
      away: {
        abbrev: "ATL",
        name: "Atlanta Dream",
        score: 82,
        record: "17-10",
      },
      home: {
        abbrev: "DAL",
        name: "Dallas Wings",
        score: 81,
        record: "18-10",
      },
    },
  ]);
});
```

Import `mapToMatchupGames` from `./mapScoreboard`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/components/home/mapScoreboard.test.ts`

Expected: FAIL (`mapToMatchupGames` not exported / types missing)

- [ ] **Step 3: Implement types + mapper**

Update `frontend/src/lib/api.ts`:

```ts
export type ApiWnbaTeam = {
  abbrev: string;
  name: string;
  score: number | null;
  record?: string | null;
};

export type ApiWnbaGame = {
  id: string;
  espn_event_id: string | null;
  league: "wnba";
  status: ApiGameStatus;
  status_label: string;
  away: ApiWnbaTeam;
  home: ApiWnbaTeam;
  start_time_et: string;
  venue?: string | null;
  venue_city?: string | null;
};
```

Create `frontend/src/components/league/types.ts`:

```ts
import type { GameStatus, HomeLeague } from "@/components/home/types";

export type LeagueSlug = "nba" | "wnba";

export type MatchupTeam = {
  abbrev: string;
  name: string;
  score: number | null;
  record?: string | null;
};

export type MatchupGame = {
  id: string;
  espnEventId?: string | null;
  league: HomeLeague;
  status: GameStatus;
  statusLabel: string;
  venue?: string | null;
  venueCity?: string | null;
  away: MatchupTeam;
  home: MatchupTeam;
};
```

Add to `frontend/src/components/home/mapScoreboard.ts`:

```ts
import type { MatchupGame } from "@/components/league/types";

export function mapToMatchupGames(games: ApiWnbaGame[]): MatchupGame[] {
  return games.map((g) => ({
    id: g.id,
    espnEventId: g.espn_event_id,
    league: g.league,
    statusLabel: g.status_label,
    status: g.status,
    venue: g.venue ?? null,
    venueCity: g.venue_city ?? null,
    away: {
      abbrev: g.away.abbrev,
      name: g.away.name,
      score: g.away.score,
      record: g.away.record ?? null,
    },
    home: {
      abbrev: g.home.abbrev,
      name: g.home.name,
      score: g.home.score,
      record: g.home.record ?? null,
    },
  }));
}
```

Keep existing `mapToLiveGames` / `mapToTickerGames` unchanged (home LIVE NOW does not need venue/records).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run src/components/home/mapScoreboard.test.ts`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/api.ts frontend/src/components/league/types.ts \
  frontend/src/components/home/mapScoreboard.ts \
  frontend/src/components/home/mapScoreboard.test.ts
git commit -m "$(cat <<'EOF'
Map scoreboard venue and records into MatchupGame.

EOF
)"
```

---

### Task 3: `MatchupGameCard`

**Files:**
- Create: `frontend/src/components/league/MatchupGameCard.tsx`
- Create: `frontend/src/components/league/MatchupGameCard.test.tsx`

**Interfaces:**
- Consumes: `MatchupGame`
- Produces: `MatchupGameCard({ game }: { game: MatchupGame })` — `Link` when `espnEventId` set, else `<article>`

- [ ] **Step 1: Write the failing tests**

```tsx
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { MatchupGameCard } from "./MatchupGameCard";
import type { MatchupGame } from "./types";

const liveGame: MatchupGame = {
  id: "1",
  espnEventId: "401857098",
  league: "wnba",
  status: "live",
  statusLabel: "3:31 - 4th",
  venue: "Mortgage Matchup Center",
  venueCity: "Phoenix",
  away: {
    abbrev: "GS",
    name: "Golden State Valkyries",
    score: 77,
    record: "19-8",
  },
  home: {
    abbrev: "PHX",
    name: "Phoenix Mercury",
    score: 78,
    record: "10-18",
  },
};

function renderCard(game: MatchupGame) {
  return render(
    <MemoryRouter>
      <MatchupGameCard game={game} />
    </MemoryRouter>,
  );
}

describe("MatchupGameCard", () => {
  it("links to game detail when espnEventId is set", () => {
    renderCard(liveGame);
    expect(
      screen.getByRole("link", { name: /golden state valkyries/i }),
    ).toHaveAttribute("href", "/games/401857098");
  });

  it("shows venue · city, status, records, and scores", () => {
    renderCard(liveGame);
    expect(screen.getByText("3:31 - 4th")).toBeInTheDocument();
    expect(
      screen.getByText("Mortgage Matchup Center · Phoenix"),
    ).toBeInTheDocument();
    expect(screen.getByText("19-8")).toBeInTheDocument();
    expect(screen.getByText("10-18")).toBeInTheDocument();
    expect(screen.getByText("77")).toBeInTheDocument();
    expect(screen.getByText("78")).toBeInTheDocument();
  });

  it("omits venue line and records when absent", () => {
    renderCard({
      ...liveGame,
      espnEventId: null,
      venue: null,
      venueCity: null,
      away: { ...liveGame.away, record: null },
      home: { ...liveGame.home, record: null },
    });
    expect(screen.queryByText(/Mortgage/)).not.toBeInTheDocument();
    expect(screen.queryByText("19-8")).not.toBeInTheDocument();
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/components/league/MatchupGameCard.test.tsx`

Expected: FAIL (module not found)

- [ ] **Step 3: Implement the card**

Create `MatchupGameCard.tsx` matching the mockup structure:

- Outer: `Link` or `article` with `rounded-xl border border-white/10 bg-[#141414]`, left accent via `border-l-2` or absolute bar (`border-l-violet-500` for live WNBA; `border-l-violet-400/80` or subtle gradient class for final).
- Top row: live pulsing violet/red dot + `statusLabel` on the left; venue line on the right (`MapPin` optional from lucide — only if already used nearby; otherwise plain text). Format venue as `${venue} · ${venueCity}` when both set, else venue alone.
- Team rows: letter avatar circle, abbrev, name, muted `record` when present, amber score box (`bg-black text-amber-300` or amber outlined box per mockup — prefer existing LIVE NOW amber score box pattern for consistency: `rounded-md bg-black font-mono text-amber-300`).
- Trailing chevron (`ChevronRight` from lucide-react) for affordance.
- Accessible name: include both team names in link text (visible names are enough for Testing Library).

Do **not** invent green winning-record logic unless trivial; use `text-white/45` for records.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run src/components/league/MatchupGameCard.test.tsx`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/league/MatchupGameCard.tsx \
  frontend/src/components/league/MatchupGameCard.test.tsx
git commit -m "$(cat <<'EOF'
Add MatchupGameCard for the league hub slate.

EOF
)"
```

---

### Task 4: `MatchupsPanel`

**Files:**
- Create: `frontend/src/components/league/MatchupsPanel.tsx`
- Create: `frontend/src/components/league/MatchupsPanel.test.tsx`

**Interfaces:**
- Consumes: `MatchupGame[]`, loading/error flags
- Produces: `MatchupsPanel({ games, isLoading?, isError? })`
- Uses `isInProgressStatus` from `@/components/home/mapScoreboard`

- [ ] **Step 1: Write the failing tests**

```tsx
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { MatchupsPanel } from "./MatchupsPanel";
import type { MatchupGame } from "./types";

const live: MatchupGame = {
  id: "live-1",
  espnEventId: "1",
  league: "wnba",
  status: "live",
  statusLabel: "Q4 3:31",
  away: { abbrev: "GS", name: "Golden State Valkyries", score: 77 },
  home: { abbrev: "PHX", name: "Phoenix Mercury", score: 78 },
};

const finalGame: MatchupGame = {
  id: "final-1",
  espnEventId: "2",
  league: "wnba",
  status: "final",
  statusLabel: "Final",
  away: { abbrev: "ATL", name: "Atlanta Dream", score: 82 },
  home: { abbrev: "DAL", name: "Dallas Wings", score: 81 },
};

function renderPanel(games: MatchupGame[], props = {}) {
  return render(
    <MemoryRouter>
      <MatchupsPanel games={games} {...props} />
    </MemoryRouter>,
  );
}

describe("MatchupsPanel", () => {
  it("splits live and rest, shows count and disabled Today control", () => {
    renderPanel([live, finalGame]);
    expect(screen.getByRole("heading", { name: "Matchups" })).toBeInTheDocument();
    expect(
      screen.getByText(/2 games · open a card for box score/i),
    ).toBeInTheDocument();
    expect(screen.getByText("LIVE NOW")).toBeInTheDocument();
    expect(screen.getByText("REST OF THE SLATE")).toBeInTheDocument();
    expect(screen.getByText("Today")).toBeInTheDocument();
    const prev = screen.getByRole("button", { name: /previous day/i });
    const next = screen.getByRole("button", { name: /next day/i });
    expect(prev).toBeDisabled();
    expect(next).toBeDisabled();
  });

  it("hides LIVE NOW when no in-progress games", () => {
    renderPanel([finalGame]);
    expect(screen.queryByText("LIVE NOW")).not.toBeInTheDocument();
    expect(screen.getByText("REST OF THE SLATE")).toBeInTheDocument();
  });

  it("shows muted empty copy when no games and not loading", () => {
    renderPanel([]);
    expect(screen.getByText(/no games/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/components/league/MatchupsPanel.test.tsx`

Expected: FAIL (module not found)

- [ ] **Step 3: Implement the panel**

Layout:

```tsx
type MatchupsPanelProps = {
  games: MatchupGame[];
  isLoading?: boolean;
  isError?: boolean;
};

export function MatchupsPanel({ games, isLoading, isError }: MatchupsPanelProps) {
  const live = games.filter((g) => isInProgressStatus(g.status));
  const rest = games.filter((g) => !isInProgressStatus(g.status));
  // header with Matchups + disabled ChevronLeft/Right + "Today"
  // subcopy with games.length
  // if isLoading && games.length === 0 → 2–3 skeleton cards
  // if isError && games.length === 0 → "Unable to load matchups"
  // if !isLoading && games.length === 0 → "No games on today's slate"
  // else render LIVE NOW section (if live.length) + REST OF THE SLATE (if rest.length)
}
```

Use `MatchupGameCard` for each game. Section labels: small uppercase tracking like home LIVE NOW (`text-xs font-semibold tracking-wider text-white/50`).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run src/components/league/MatchupsPanel.test.tsx`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/league/MatchupsPanel.tsx \
  frontend/src/components/league/MatchupsPanel.test.tsx
git commit -m "$(cat <<'EOF'
Add MatchupsPanel with live and rest-of-slate sections.

EOF
)"
```

---

### Task 5: `LeagueHero` + `LeagueSubnav`

**Files:**
- Create: `frontend/src/components/league/LeagueHero.tsx`
- Create: `frontend/src/components/league/LeagueHero.test.tsx`
- Create: `frontend/src/components/league/LeagueSubnav.tsx`
- Create: `frontend/src/components/league/LeagueSubnav.test.tsx`

**Interfaces:**
- Consumes: `league: LeagueSlug`
- Produces: `LeagueHero({ league })`, `LeagueSubnav({ league })`

- [ ] **Step 1: Write failing tests**

`LeagueHero.test.tsx`:

```tsx
it("renders WNBA hero copy", () => {
  render(<LeagueHero league="wnba" />);
  expect(screen.getByText("WNBA")).toBeInTheDocument();
  expect(
    screen.getByRole("heading", { name: /women.?s basketball/i }),
  ).toBeInTheDocument();
});

it("renders NBA hero copy", () => {
  render(<LeagueHero league="nba" />);
  expect(
    screen.getByRole("heading", { name: /men.?s basketball/i }),
  ).toBeInTheDocument();
});
```

Date label: format **today** in local or America/New_York as `WED, JUL 29` (uppercase). Prefer a small helper using `Intl.DateTimeFormat` with `weekday: "short", month: "short", day: "numeric"` then uppercasing — pin the test with `vi.useFakeTimers` set to `2026-07-29T18:00:00-07:00` if asserting exact string, or assert with a regex `/[A-Z]{3}, [A-Z]{3} \d{1,2}/`.

`LeagueSubnav.test.tsx`:

```tsx
it("marks Matchups active and disables other items", () => {
  render(<LeagueSubnav league="wnba" />);
  expect(screen.getByRole("button", { name: "Matchups" })).toHaveAttribute(
    "aria-current",
    "page",
  );
  expect(screen.getByRole("button", { name: "HoopVista Picks" })).toBeDisabled();
  expect(screen.getByRole("button", { name: "Leaders" })).toBeDisabled();
  expect(screen.getByRole("button", { name: "Glossary" })).toBeDisabled();
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npx vitest run src/components/league/LeagueHero.test.tsx src/components/league/LeagueSubnav.test.tsx`

Expected: FAIL

- [ ] **Step 3: Implement hero + subnav**

`LeagueHero` content by league:

| League | Pill | Title | Blurb |
| --- | --- | --- | --- |
| wnba | violet “WNBA” | Women’s Basketball | Tonight's matchups, league leaders, and standings, plus the playoff race and a clutch tab for who delivers late in tight games. |
| nba | sky “NBA” | Men’s Basketball | Same structural blurb adapted for NBA (one short sentence). |

Rounded charcoal banner (`rounded-2xl bg-[#121212] border border-white/10`), max-width aligned with home (`max-w-6xl mx-auto px-4`). Optional faint basketball watermark: reuse `@/assets/wnba_basketball.png` / `@/assets/basketball.png` at low opacity on the right — decorative `aria-hidden`.

`LeagueSubnav`: horizontal scroll-friendly pill bar; EXPLORE group then LEARN group with tiny labels. Items:

EXPLORE: Matchups (active), HoopVista Picks, Leaders, Standings, Playoff race, Clutch  
LEARN: How it works, Glossary  

Active Matchups: `bg-violet-600` for WNBA, `bg-sky-600` for NBA. Others: `disabled` buttons with muted text.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/components/league/LeagueHero.test.tsx src/components/league/LeagueSubnav.test.tsx`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/league/LeagueHero.tsx \
  frontend/src/components/league/LeagueHero.test.tsx \
  frontend/src/components/league/LeagueSubnav.tsx \
  frontend/src/components/league/LeagueSubnav.test.tsx
git commit -m "$(cat <<'EOF'
Add league hero and Explore/Learn subnav chrome.

EOF
)"
```

---

### Task 6: `LeagueMatchupsPage` + router

**Files:**
- Create: `frontend/src/pages/LeagueMatchupsPage.tsx`
- Modify: `frontend/src/AppRouter.tsx`
- Modify: `frontend/src/AppRouter.test.tsx`

**Interfaces:**
- Consumes: `league: LeagueSlug` prop, `useWnbaScoreboard`, `mapToMatchupGames`
- Routes (explicit, under `HomeChromeLayout`): `/wnba/matchups` and `/nba/matchups` → `<LeagueMatchupsPage league="…" />`

- [ ] **Step 1: Write failing router tests**

Add to `AppRouter.test.tsx` (reuse existing fetch mock for scoreboard):

```ts
it("renders WNBA matchups hub at /wnba/matchups", async () => {
  renderWithProviders(["/wnba/matchups"]);
  expect(
    await screen.findByRole("heading", { name: /women.?s basketball/i }),
  ).toBeInTheDocument();
  expect(screen.getByRole("heading", { name: "Matchups" })).toBeInTheDocument();
});

it("renders NBA coming-soon hub at /nba/matchups", async () => {
  renderWithProviders(["/nba/matchups"]);
  expect(
    await screen.findByRole("heading", { name: /men.?s basketball/i }),
  ).toBeInTheDocument();
  expect(screen.getByText(/coming soon/i)).toBeInTheDocument();
});

it("renders not found for unknown league matchups", () => {
  renderWithProviders(["/mlb/matchups"]);
  expect(
    screen.getByRole("heading", { name: /page not found/i }),
  ).toBeInTheDocument();
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npx vitest run src/AppRouter.test.tsx`

Expected: FAIL (404 or missing headings)

- [ ] **Step 3: Implement page + route**

`LeagueMatchupsPage.tsx`:

```tsx
import { Navigate, useParams } from "react-router-dom";
import { LeagueHero } from "@/components/league/LeagueHero";
import { LeagueSubnav } from "@/components/league/LeagueSubnav";
import { MatchupsPanel } from "@/components/league/MatchupsPanel";
import type { LeagueSlug } from "@/components/league/types";
import { mapToMatchupGames } from "@/components/home/mapScoreboard";
import { useWnbaScoreboard } from "@/hooks/useWnbaScoreboard";

const LEAGUES = new Set<LeagueSlug>(["nba", "wnba"]);

export function LeagueMatchupsPage() {
  const { league: raw } = useParams();
  if (!raw || !LEAGUES.has(raw as LeagueSlug)) {
    return <Navigate to="/404" replace />; // or render NotFoundPage inline
  }
  const league = raw as LeagueSlug;

  return (
    <div className="mx-auto max-w-6xl space-y-6 px-4 py-6 sm:px-6">
      <LeagueHero league={league} />
      <LeagueSubnav league={league} />
      {league === "wnba" ? <WnbaMatchupsBody /> : (
        <p className="text-sm text-white/50">NBA matchups coming soon.</p>
      )}
    </div>
  );
}

function WnbaMatchupsBody() {
  const { games, isLoading, hasNeverLoaded } = useWnbaScoreboard();
  return (
    <MatchupsPanel
      games={mapToMatchupGames(games)}
      isLoading={isLoading}
      isError={hasNeverLoaded}
    />
  );
}
```

Prefer rendering `<NotFoundPage />` for invalid leagues **without** leaving `HomeChromeLayout` if that’s how other 404s work — check `AppRouter`: currently `*` is outside chrome. Matching the existing “unknown path → NotFoundPage without chrome” behavior: for invalid league, either:

1. `<Navigate to="/not-a-real-path" replace />` (hacky), or  
2. Register only explicit routes:

```tsx
<Route path="/wnba/matchups" element={<LeagueMatchupsPage />} />
<Route path="/nba/matchups" element={<LeagueMatchupsPage />} />
```

**Prefer explicit routes** (option 2) so `/mlb/matchups` hits `*` → `NotFoundPage`. Pass league via `useParams` only if using `/:league/matchups` with an in-page guard that returns `<NotFoundPage />` (will still show chrome). Spec allows either; **use explicit `/wnba/matchups` and `/nba/matchups` routes** and derive league from `useLocation().pathname` or two thin wrappers. Simplest:

```tsx
<Route path="/wnba/matchups" element={<LeagueMatchupsPage league="wnba" />} />
<Route path="/nba/matchups" element={<LeagueMatchupsPage league="nba" />} />
```

Change page props to `league: LeagueSlug` instead of `useParams` — clearer and makes `/mlb/matchups` a true 404.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/AppRouter.test.tsx`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/LeagueMatchupsPage.tsx \
  frontend/src/AppRouter.tsx frontend/src/AppRouter.test.tsx
git commit -m "$(cat <<'EOF'
Route WNBA and NBA matchups hubs under home chrome.

EOF
)"
```

---

### Task 7: Wire `HomeNav` league links + active pills

**Files:**
- Modify: `frontend/src/components/home/HomeNav.tsx`
- Modify: `frontend/src/components/home/HomeNav.test.tsx`

**Interfaces:**
- WNBA → `/wnba/matchups` (violet active pill when `pathname.startsWith("/wnba")`)
- NBA → `/nba/matchups` (sky active pill when `pathname.startsWith("/nba")`)

- [ ] **Step 1: Update failing expectations in HomeNav tests**

Replace the `/#live-now` test:

```tsx
it("points league links at matchups hubs", () => {
  renderNav("/about");
  expect(screen.getByRole("link", { name: "NBA" })).toHaveAttribute(
    "href",
    "/nba/matchups",
  );
  expect(screen.getByRole("link", { name: "WNBA" })).toHaveAttribute(
    "href",
    "/wnba/matchups",
  );
});

it("marks WNBA current on /wnba/matchups", () => {
  renderNav("/wnba/matchups");
  expect(screen.getByRole("link", { name: "WNBA" })).toHaveAttribute(
    "aria-current",
    "page",
  );
  expect(screen.getByRole("link", { name: "NBA" })).not.toHaveAttribute(
    "aria-current",
  );
});

it("marks NBA current on /nba/matchups", () => {
  renderNav("/nba/matchups");
  expect(screen.getByRole("link", { name: "NBA" })).toHaveAttribute(
    "aria-current",
    "page",
  );
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npx vitest run src/components/home/HomeNav.test.tsx`

Expected: FAIL (still `/#live-now`)

- [ ] **Step 3: Implement nav Links**

Replace `<a href="/#live-now">` with `Link` to `/${league.id}/matchups`. Active styles:

```tsx
const active =
  league.id === "wnba"
    ? pathname.startsWith("/wnba")
    : pathname.startsWith("/nba");

className={
  active
    ? league.id === "wnba"
      ? "flex items-center gap-2 rounded-full bg-violet-600 px-3 py-1 text-[14px] font-medium text-white no-underline"
      : "flex items-center gap-2 rounded-full bg-sky-600 px-3 py-1 text-[14px] font-medium text-white no-underline"
    : "flex items-center gap-2 text-[14px] font-medium text-white/80 no-underline transition-colors hover:text-white"
}
aria-current={active ? "page" : undefined}
```

Keep icons. Mobile: keep `hidden sm:flex` on the league group.

- [ ] **Step 4: Run HomeNav + full frontend verification**

Run: `cd frontend && npx vitest run src/components/home/HomeNav.test.tsx && npm run test && npm run build`

Expected: all PASS; build succeeds

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/home/HomeNav.tsx \
  frontend/src/components/home/HomeNav.test.tsx
git commit -m "$(cat <<'EOF'
Point HomeNav leagues at matchups hubs with active pills.

EOF
)"
```

---

## Spec coverage checklist

| Spec requirement | Task |
| --- | --- |
| `/wnba/matchups` + `/nba/matchups` | 6 |
| HomeNav links + active pills | 7 |
| League hero | 5 |
| Explore/Learn subnav; only Matchups enabled | 5 |
| Matchups panel + Today disabled | 4 |
| LIVE NOW / REST OF THE SLATE | 4 |
| Cards → `/games/:espnEventId` | 3 |
| Venue + records on scoreboard | 1 |
| Frontend mapping | 2 |
| NBA coming soon | 6 |
| Invalid league → 404 | 6 |
| HoopVista branding | 5 (Picks label), 7 (wordmark unchanged) |
| HomeChromeLayout retained | 6 |

## Out of scope (do not implement)

- Dated scoreboard / working date arrows  
- Real Explore/Learn destinations  
- NBA scoreboard data  
- Changing game-detail Back to return to matchups  
