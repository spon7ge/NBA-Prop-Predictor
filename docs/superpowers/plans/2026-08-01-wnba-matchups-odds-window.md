# WNBA Matchups Odds Window + DK/FD Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show Sharp team odds on `/wnba/matchups` for today through day+2 (ET), preferring DraftKings and falling back to FanDuel per game when DK has no usable line.

**Architecture:** Extend `sharp_odds` to fetch DK + FD in parallel, normalize each book (with `game_date` from `event_id` and per-game `sportsbook`), then merge DK-over-FD by `(away, home, game_date)`. Keep `GET /api/wnba/odds/today`. Frontend opens the odds merge gate to a 3-day ET window and prefers date-keyed matching when `game_date` is present.

**Tech Stack:** FastAPI · pytest · httpx · React 19 · TypeScript · Vitest · openapi-typescript

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-01-wnba-matchups-odds-window-design.md`
- Window: today, today+1, today+2 (ET slate dates); hide past and day+3+
- Primary book: DraftKings; fallback: FanDuel, **per game**, whole game from one book
- Route stays `GET /api/wnba/odds/today`
- No sportsbook badge on matchup cards
- TDD: failing test before implementation each task
- Commit `frontend/openapi.json` + `api.schema.d.ts` when schema changes

## File map

| File | Responsibility |
| --- | --- |
| `backend/app/schemas/wnba_odds.py` | Optional `game_date`, `sportsbook` on `WnbaOddsGame` |
| `backend/app/services/sharp_odds.py` | Parse date; sportsbook-aware fetch/normalize; DK/FD merge |
| `backend/tests/test_sharp_odds.py` | Normalize, merge, dual-fetch, route tests |
| `frontend/openapi.json` / `api.schema.d.ts` | Schema sync |
| `frontend/src/components/league/matchupSlateDate.ts` | `isOddsWindowDate` |
| `frontend/src/components/league/mergeMatchupOdds.ts` | Date-aware merge |
| `frontend/src/pages/LeagueMatchupsPage.tsx` | 3-day odds gate |

---

### Task 1: Schema + `game_date` / sportsbook on normalize

**Files:**
- Modify: `backend/app/schemas/wnba_odds.py`
- Modify: `backend/app/services/sharp_odds.py`
- Test: `backend/tests/test_sharp_odds.py`

**Interfaces:**
- Consumes: existing Sharp row shape (`event_id`, markets, teams)
- Produces:
  - `WnbaOddsGame.game_date: str | None = None`
  - `WnbaOddsGame.sportsbook: str | None = None`
  - `_EVENT_DATE_RE` + `_game_date_from_event_id(event_id: str) -> str | None`
  - `normalize_sharp_odds(rows, sportsbook: str | None = None) -> list[WnbaOddsGame]`  
    (sets `game_date` from `event_id`; sets `sportsbook` on each game when provided)

- [ ] **Step 1: Write the failing tests**

Add to `backend/tests/test_sharp_odds.py`:

```python
def test_normalize_parses_game_date_from_event_id():
    rows = json.loads(FIXTURE.read_text())["data"]
    games = svc.normalize_sharp_odds(rows, sportsbook="draftkings")
    atl = next(g for g in games if g.home_abbrev == "ATL")
    assert atl.game_date == "2026-07-31"
    assert atl.sportsbook == "draftkings"


def test_normalize_omits_game_date_when_event_id_has_none():
    rows = [
        {
            "event_id": "wnba_no_date_here",
            "is_main_line": True,
            "market_type": "total_points",
            "line": 170.5,
            "home": {"abbreviation": "ATL"},
            "away": {"abbreviation": "SEA"},
            "home_team": "ATL Dream",
            "away_team": "SEA Storm",
        }
    ]
    games = svc.normalize_sharp_odds(rows, sportsbook="fanduel")
    assert len(games) == 1
    assert games[0].game_date is None
    assert games[0].sportsbook == "fanduel"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_sharp_odds.py::test_normalize_parses_game_date_from_event_id tests/test_sharp_odds.py::test_normalize_omits_game_date_when_event_id_has_none -v`

Expected: FAIL (missing attrs / unexpected kwargs)

- [ ] **Step 3: Minimal schema + normalize implementation**

In `backend/app/schemas/wnba_odds.py`, add to `WnbaOddsGame`:

```python
game_date: str | None = None
sportsbook: str | None = None
```

In `backend/app/services/sharp_odds.py`:

```python
_EVENT_DATE_RE = re.compile(r"(20\d{2}-\d{2}-\d{2})")


def _game_date_from_event_id(event_id: str | None) -> str | None:
    if not event_id:
        return None
    match = _EVENT_DATE_RE.search(str(event_id))
    return match.group(1) if match else None
```

Update `normalize_sharp_odds` signature to `normalize_sharp_odds(rows: list[dict[str, Any]], sportsbook: str | None = None) -> list[WnbaOddsGame]`.

In the per-event bucket, also store `game_date` from the first row’s `event_id` (`_game_date_from_event_id`). When appending `WnbaOddsGame`, pass `game_date=bucket.get("game_date")` and `sportsbook=sportsbook`.

Update existing fixture-based tests if they assert exact model equality (they should still pass; optional fields default `None` until sportsbook arg is passed — update the favorite-spread test to optionally assert `game_date == "2026-07-31"` after implementation).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_sharp_odds.py -v`

Expected: PASS (all existing + new)

- [ ] **Step 5: Commit**

```bash
git add backend/app/schemas/wnba_odds.py backend/app/services/sharp_odds.py backend/tests/test_sharp_odds.py
git commit -m "feat: add game_date and sportsbook to Sharp odds normalize"
```

---

### Task 2: Dual-book fetch + per-game DK/FD merge

**Files:**
- Modify: `backend/app/services/sharp_odds.py`
- Test: `backend/tests/test_sharp_odds.py`

**Interfaces:**
- Consumes: `normalize_sharp_odds` from Task 1
- Produces:
  - `async def fetch_sharp_odds_rows(sportsbook: str = "draftkings") -> list[dict]`
  - `def _odds_merge_key(game: WnbaOddsGame) -> tuple[str, str, str]`
  - `def merge_odds_prefer_primary(primary: list[WnbaOddsGame], fallback: list[WnbaOddsGame]) -> list[WnbaOddsGame]`
  - `get_today_odds` fetches DK + FD in parallel (`asyncio.gather(..., return_exceptions=True)`), normalizes each, merges DK then FD fill; if one book errors, still serve the other; both error → stale cache or empty + `error`

- [ ] **Step 1: Write the failing tests**

```python
def test_merge_odds_prefer_primary_keeps_dk_over_fd():
    from app.schemas.wnba_odds import WnbaOddsGame

    dk = [
        WnbaOddsGame(
            home_abbrev="ATL",
            away_abbrev="SEA",
            spread_team_abbrev="ATL",
            spread_line=-12.5,
            total=179.5,
            game_date="2026-07-31",
            sportsbook="draftkings",
        )
    ]
    fd = [
        WnbaOddsGame(
            home_abbrev="ATL",
            away_abbrev="SEA",
            spread_team_abbrev="ATL",
            spread_line=-11.5,
            total=180.5,
            game_date="2026-07-31",
            sportsbook="fanduel",
        )
    ]
    merged = svc.merge_odds_prefer_primary(dk, fd)
    assert len(merged) == 1
    assert merged[0].sportsbook == "draftkings"
    assert merged[0].spread_line == -12.5


def test_merge_odds_prefer_primary_fills_missing_game_from_fd():
    from app.schemas.wnba_odds import WnbaOddsGame

    dk = [
        WnbaOddsGame(
            home_abbrev="ATL",
            away_abbrev="SEA",
            spread_line=-12.5,
            total=179.5,
            game_date="2026-07-31",
            sportsbook="draftkings",
            spread_team_abbrev="ATL",
        )
    ]
    fd = [
        WnbaOddsGame(
            home_abbrev="WAS",
            away_abbrev="DAL",
            spread_line=-3.5,
            total=167.5,
            game_date="2026-07-31",
            sportsbook="fanduel",
            spread_team_abbrev="DAL",
        )
    ]
    merged = svc.merge_odds_prefer_primary(dk, fd)
    assert {g.home_abbrev for g in merged} == {"ATL", "WAS"}
    was = next(g for g in merged if g.home_abbrev == "WAS")
    assert was.sportsbook == "fanduel"


def test_get_today_odds_uses_fd_when_dk_fetch_fails():
    from app.schemas.wnba_odds import WnbaOddsGame

    fd_rows = json.loads(FIXTURE.read_text())["data"]

    async def fake_fetch(sportsbook: str = "draftkings"):
        if sportsbook == "draftkings":
            raise RuntimeError("dk down")
        return fd_rows

    with (
        patch.object(svc, "SHARP_API_KEY", "sk_test"),
        patch.object(svc, "fetch_sharp_odds_rows", side_effect=fake_fetch),
    ):
        body = __import__("asyncio").run(svc.get_today_odds())

    assert len(body.games) >= 1
    assert all(g.sportsbook == "fanduel" for g in body.games)
```

Use `from app.schemas.wnba_odds import WnbaOddsGame` in all three tests (do not invent `svc.WnbaOddsGame`).

Update `test_odds_route_returns_games_when_fetch_ok` so `fake_fetch` accepts `sportsbook: str = "draftkings"` and returns fixture data for both (or only when `draftkings` / `fanduel`). Same for stale-cache test.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_sharp_odds.py::test_merge_odds_prefer_primary_keeps_dk_over_fd tests/test_sharp_odds.py::test_merge_odds_prefer_primary_fills_missing_game_from_fd tests/test_sharp_odds.py::test_get_today_odds_uses_fd_when_dk_fetch_fails -v`

Expected: FAIL (`merge_odds_prefer_primary` missing / `fetch` arity)

- [ ] **Step 3: Implement fetch param, merge, and dual `get_today_odds`**

Change fetch:

```python
async def fetch_sharp_odds_rows(sportsbook: str = "draftkings") -> list[dict[str, Any]]:
    ...
    params_base = {
        "league": "wnba",
        "sportsbook": sportsbook,
        "market": "point_spread,total_points",
        "is_main_line": "true",
        "limit": str(PAGE_LIMIT),
    }
```

Add:

```python
def _odds_merge_key(game: WnbaOddsGame) -> tuple[str, str, str]:
    return (game.away_abbrev, game.home_abbrev, game.game_date or "")


def merge_odds_prefer_primary(
    primary: list[WnbaOddsGame],
    fallback: list[WnbaOddsGame],
) -> list[WnbaOddsGame]:
    by_key: dict[tuple[str, str, str], WnbaOddsGame] = {}
    for game in primary:
        by_key[_odds_merge_key(game)] = game
    for game in fallback:
        key = _odds_merge_key(game)
        if key not in by_key:
            by_key[key] = game
    games = list(by_key.values())
    games.sort(key=lambda g: (g.game_date or "", g.home_abbrev, g.away_abbrev))
    return games
```

Rewrite `get_today_odds` success path (keep cache / no-key / stale behavior):

```python
import asyncio
...
dk_result, fd_result = await asyncio.gather(
    fetch_sharp_odds_rows("draftkings"),
    fetch_sharp_odds_rows("fanduel"),
    return_exceptions=True,
)
errors: list[str] = []
dk_games: list[WnbaOddsGame] = []
fd_games: list[WnbaOddsGame] = []
if isinstance(dk_result, BaseException):
    errors.append(f"draftkings: {dk_result}")
else:
    dk_games = normalize_sharp_odds(dk_result, sportsbook="draftkings")
if isinstance(fd_result, BaseException):
    errors.append(f"fanduel: {fd_result}")
else:
    fd_games = normalize_sharp_odds(fd_result, sportsbook="fanduel")

if not dk_games and not fd_games:
    # both failed or empty — raise or return error path like today
    if errors:
        raise RuntimeError("; ".join(errors))
    games = []
else:
    games = merge_odds_prefer_primary(dk_games, fd_games)

response = WnbaOddsResponse(
    as_of=_utcnow_iso(),
    games=games,
    error="; ".join(errors) if errors else None,
)
```

Wrap the gather block in the existing try/except so total failure still serves stale cache.

- [ ] **Step 4: Run full sharp odds tests**

Run: `cd backend && python -m pytest tests/test_sharp_odds.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/sharp_odds.py backend/tests/test_sharp_odds.py
git commit -m "feat: merge Sharp WNBA odds with DraftKings over FanDuel"
```

---

### Task 3: Regenerate OpenAPI + frontend schema

**Files:**
- Modify: `frontend/openapi.json`
- Modify: `frontend/src/lib/api.schema.d.ts`
- Test: `backend/tests/test_export_openapi.py` (existing; should still pass)

**Interfaces:**
- Produces: `WnbaOddsGame` in OpenAPI/types includes optional `game_date` and `sportsbook`

- [ ] **Step 1: Export OpenAPI**

Run from repo root:

```bash
python scripts/export_openapi.py
```

- [ ] **Step 2: Generate TypeScript types**

Run: `cd frontend && npm run generate:api`

Confirm `WnbaOddsGame` in `api.schema.d.ts` includes:

```typescript
game_date?: string | null;
sportsbook?: string | null;
```

(or required-null fields matching FastAPI’s serialization defaults — match whatever export emits).

- [ ] **Step 3: Verify export test**

Run: `cd backend && python -m pytest tests/test_export_openapi.py -v`

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add frontend/openapi.json frontend/src/lib/api.schema.d.ts
git commit -m "chore: sync OpenAPI for odds game_date and sportsbook"
```

---

### Task 4: Date-aware `mergeMatchupOdds`

**Files:**
- Modify: `frontend/src/components/league/mergeMatchupOdds.ts`
- Test: `frontend/src/components/league/mergeMatchupOdds.test.ts`

**Interfaces:**
- Consumes: `ApiWnbaOddsGame` with optional `game_date`
- Produces:
  - `mergeMatchupOdds(games, oddsGames, slateDate?: string): MatchupGame[]`
  - When `slateDate` is set: match rows with `game_date === slateDate` first; else rows with missing `game_date`; never attach a row whose `game_date` is a different day
  - When `slateDate` omitted: keep abbrev-only behavior (ignore `game_date` for keying, last-write wins by abbrev — same as today)

- [ ] **Step 1: Write the failing tests**

Append to `mergeMatchupOdds.test.ts`:

```typescript
it("prefers odds whose game_date matches the slate", () => {
  const merged = mergeMatchupOdds(
    [baseGame],
    [
      {
        home_abbrev: "ATL",
        away_abbrev: "SEA",
        spread_team_abbrev: "ATL",
        spread_line: -10.5,
        total: 170.5,
        game_date: "2026-07-30",
      },
      {
        home_abbrev: "ATL",
        away_abbrev: "SEA",
        spread_team_abbrev: "ATL",
        spread_line: -12.5,
        total: 179.5,
        game_date: "2026-07-31",
      },
    ],
    "2026-07-31",
  );
  expect(merged[0].odds?.spreadLine).toBe(-12.5);
});

it("does not use odds from a different game_date", () => {
  const merged = mergeMatchupOdds(
    [baseGame],
    [
      {
        home_abbrev: "ATL",
        away_abbrev: "SEA",
        spread_team_abbrev: "ATL",
        spread_line: -12.5,
        total: 179.5,
        game_date: "2026-07-30",
      },
    ],
    "2026-07-31",
  );
  expect(merged[0].odds).toBeNull();
});

it("falls back to undated odds when slateDate is set", () => {
  const merged = mergeMatchupOdds(
    [baseGame],
    [
      {
        home_abbrev: "ATL",
        away_abbrev: "SEA",
        spread_team_abbrev: "ATL",
        spread_line: -12.5,
        total: 179.5,
      },
    ],
    "2026-07-31",
  );
  expect(merged[0].odds?.spreadLine).toBe(-12.5);
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npx vitest run src/components/league/mergeMatchupOdds.test.ts`

Expected: FAIL on date preference / wrong-day rejection

- [ ] **Step 3: Implement date-aware merge**

```typescript
export function mergeMatchupOdds(
  games: MatchupGame[],
  oddsGames: ApiWnbaOddsGame[] | undefined,
  slateDate?: string,
): MatchupGame[] {
  if (!oddsGames || oddsGames.length === 0) {
    return games.map((game) => ({ ...game, odds: game.odds ?? null }));
  }

  const byDated = new Map<string, MatchupOdds>();
  const byUndated = new Map<string, MatchupOdds>();

  for (const row of oddsGames) {
    const odds = toMatchupOdds(row);
    if (!odds) continue;
    const key = oddsKey(row.home_abbrev, row.away_abbrev);
    if (row.game_date) {
      byDated.set(`${row.game_date}|${key}`, odds);
    } else {
      byUndated.set(key, odds);
    }
  }

  return games.map((game) => {
    const key = oddsKey(game.home.abbrev, game.away.abbrev);
    let odds: MatchupOdds | null = null;
    if (slateDate) {
      odds = byDated.get(`${slateDate}|${key}`) ?? byUndated.get(key) ?? null;
    } else {
      odds = byUndated.get(key) ?? null;
      if (!odds) {
        // abbrev-only legacy: accept any dated row for this matchup
        for (const [datedKey, value] of byDated) {
          if (datedKey.endsWith(`|${key}`)) {
            odds = value;
            break;
          }
        }
      }
    }
    return { ...game, odds };
  });
}
```

Keep existing tests passing (call without `slateDate`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/components/league/mergeMatchupOdds.test.ts`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/league/mergeMatchupOdds.ts frontend/src/components/league/mergeMatchupOdds.test.ts
git commit -m "feat: match merge odds by slate game_date when present"
```

---

### Task 5: Three-day odds window on matchups page

**Files:**
- Modify: `frontend/src/components/league/matchupSlateDate.ts`
- Modify: `frontend/src/pages/LeagueMatchupsPage.tsx`
- Modify: `frontend/src/components/league/matchupSlateDate.test.ts`
- Test: helper coverage in `matchupSlateDate.test.ts` (page already mocks odds; helper is the gate under test)

**Interfaces:**
- Consumes: `shiftEtDate`
- Produces:
  - `isOddsWindowDate(selectedDate: string, today: string): boolean` — true iff `selectedDate` is `today`, `today+1`, or `today+2`
  - Page: `mergeMatchupOdds(mapToMatchupGames(games), showOdds ? oddsQuery.data?.games : undefined, selectedDate)` where `showOdds = isOddsWindowDate(selectedDate, today)`

- [ ] **Step 1: Write the failing helper tests**

Append to `frontend/src/components/league/matchupSlateDate.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import { isOddsWindowDate } from "./matchupSlateDate";

describe("isOddsWindowDate", () => {
  const today = "2026-08-01";

  it("includes today through day+2", () => {
    expect(isOddsWindowDate("2026-08-01", today)).toBe(true);
    expect(isOddsWindowDate("2026-08-02", today)).toBe(true);
    expect(isOddsWindowDate("2026-08-03", today)).toBe(true);
  });

  it("excludes past and day+3", () => {
    expect(isOddsWindowDate("2026-07-31", today)).toBe(false);
    expect(isOddsWindowDate("2026-08-04", today)).toBe(false);
  });
});
```

- [ ] **Step 2: Run helper test to verify it fails**

Run: `cd frontend && npx vitest run src/components/league/matchupSlateDate.test.ts -t isOddsWindowDate`

Expected: FAIL (`isOddsWindowDate` not exported)

- [ ] **Step 3: Implement helper + wire page**

In `matchupSlateDate.ts`:

```typescript
export function isOddsWindowDate(
  selectedDate: string,
  today: string,
): boolean {
  return (
    selectedDate === today ||
    selectedDate === shiftEtDate(today, 1) ||
    selectedDate === shiftEtDate(today, 2)
  );
}
```

In `LeagueMatchupsPage.tsx`:

```typescript
import {
  isOddsWindowDate,
  isValidEtDate,
  parseMatchupDateParam,
  shiftEtDate,
  slateEtDate,
} from "@/components/league/matchupSlateDate";

// inside WnbaMatchupsPage — replace isToday-only odds gate:
const showOdds = isOddsWindowDate(selectedDate, today);
const matchupGames = mergeMatchupOdds(
  mapToMatchupGames(games),
  showOdds ? oddsQuery.data?.games : undefined,
  selectedDate,
);
```

Remove unused `isToday` if nothing else needs it on that page (Hero still uses `selectedDate` / `data?.date`).

- [ ] **Step 4: Run frontend tests**

Run:

```bash
cd frontend && npx vitest run \
  src/components/league/matchupSlateDate.test.ts \
  src/components/league/mergeMatchupOdds.test.ts \
  src/pages/LeagueMatchupsPage.test.tsx
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/league/matchupSlateDate.ts \
  frontend/src/components/league/matchupSlateDate.test.ts \
  frontend/src/pages/LeagueMatchupsPage.tsx
git commit -m "feat: show matchup odds for today through day after"
```

---

### Task 6: Manual verification checklist

**Files:** none (verification only)

- [ ] **Step 1: Backend smoke**

With `SHARP_API_KEY` set and API running:

```bash
curl -s http://127.0.0.1:8000/api/wnba/odds/today | python -m json.tool | head -80
```

Confirm some games include `game_date` / `sportsbook`, and FD-only games appear if DK is sparse.

- [ ] **Step 2: UI smoke**

On `/wnba/matchups`:

1. Today — odds pills when lines exist  
2. Next day / day after — pills when Sharp has those games  
3. Day+3 or a past date — no pills  
4. Network tab: still `GET /api/wnba/odds/today` (no new route)

- [ ] **Step 3: Final commit only if checklist found fixes**

Do not create an empty commit. If fixes were needed, commit them with a focused message.

---

## Spec coverage (self-review)

| Spec requirement | Task |
| --- | --- |
| 3-day ET window | Task 5 |
| DK primary / FD per-game fallback | Task 2 |
| Whole game from one book | Task 2 (`merge_odds_prefer_primary`) |
| `game_date` from `event_id` | Task 1 |
| Per-game `sportsbook` | Task 1–2 |
| Keep `/odds/today` | Tasks 1–2 (unchanged route) |
| Date-keyed frontend merge | Task 4 |
| Undated abbrev fallback | Task 4 |
| One book fails → other serves | Task 2 |
| Both fail → stale / empty + error | Task 2 |
| No card sportsbook badge | (no UI task) |
| OpenAPI sync | Task 3 |
| Past / day+3 hide odds | Task 5 |

## Out of scope (do not implement)

- Past-date odds, ParlayAPI cutover, NBA matchups, dated odds route, mixing DK+FD markets on one card
