# WNBA Matchups Date Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable Previous / Next day on `/wnba/matchups` so users can browse any ET slate via `?date=YYYY-MM-DD`, backed by a dated scoreboard API.

**Architecture:** Add `GET /api/wnba/scoreboard?date=` reusing `_merged_games_for_date` (no overnight carryover, ~5 min per-date cache). Frontend owns URL search params, mirrors `slate_et_date` for “today,” fetches dated boards off today, and merges odds only on the current slate.

**Tech Stack:** FastAPI · pytest · React 19 · TypeScript · React Router · TanStack Query · Vitest · Testing Library

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-01-wnba-matchups-date-nav-design.md`
- Unlimited date range; empty slate is OK
- Odds only when viewing today’s slate date
- URL: `/wnba/matchups` today; `/wnba/matchups?date=YYYY-MM-DD` otherwise
- Keep `/api/wnba/scoreboard/today` unchanged (ticker + carryover)
- Empty copy: “No games on this slate”
- TDD: failing test before implementation each task
- Do not regenerate unrelated OpenAPI churn beyond the new path (update `frontend/openapi.json` + `api.schema.d.ts` in the API client task)

## File map

| File | Responsibility |
| --- | --- |
| `backend/app/services/wnba_scoreboard.py` | `get_scoreboard_for_date`, per-date cache |
| `backend/app/api/routes/wnba_scoreboard.py` | `GET /wnba/scoreboard?date=` |
| `backend/tests/test_wnba_scoreboard_route.py` | Dated route + cache tests |
| `frontend/src/components/league/matchupSlateDate.ts` | Slate “today,” ±1 day, label, URL parse |
| `frontend/src/lib/api.ts` | `fetchWnbaScoreboardByDate` |
| `frontend/src/hooks/useWnbaScoreboard.ts` | Optional `dateEt`; today vs dated query |
| `frontend/src/components/league/MatchupsPanel.tsx` | Enabled chevrons + center Today button |
| `frontend/src/pages/LeagueMatchupsPage.tsx` | URL state, odds gate, wire panel |
| `frontend/openapi.json` / `api.schema.d.ts` | Document new endpoint |

---

### Task 1: Backend dated scoreboard service

**Files:**
- Modify: `backend/app/services/wnba_scoreboard.py`
- Test: `backend/tests/test_wnba_scoreboard_route.py` (service-level assertions via route in Task 2; unit tests for service cache here or in same file)

**Interfaces:**
- Consumes: `_merged_games_for_date(date_et: str) -> tuple[list[WnbaGame], bool]`
- Produces:
  - `DATED_CACHE_TTL_SECONDS = 300`
  - `_date_cache: dict[str, dict]` keyed by `date_et` with `response`, `expires_at`
  - `async def get_scoreboard_for_date(date_et: str) -> WnbaScoreboardResponse`
  - No overnight carryover; raises `RuntimeError` when no usable source and no fresh/stale cache for that date

- [ ] **Step 1: Write the failing tests**

Add to `backend/tests/test_wnba_scoreboard_route.py` (or new `test_wnba_scoreboard_dated.py` if preferred — same fixtures pattern):

```python
def test_get_scoreboard_for_date_returns_requested_date_without_carryover():
    today = "2026-07-30"
    target = "2026-07-28"

    async def fake_fetch_espn(date_et: str):
        assert date_et == target
        return {"events": []}

    async def fake_fetch_stats(date_et: str):
        assert date_et == target
        return {"scoreboard": {"games": []}}

    svc._date_cache.clear()
    with (
        patch.object(svc, "fetch_espn_scoreboard", side_effect=fake_fetch_espn),
        patch.object(svc, "fetch_stats_scoreboard", side_effect=fake_fetch_stats),
    ):
        body = asyncio.run(svc.get_scoreboard_for_date(target))
    assert body.date == target
    assert body.games == []


def test_get_scoreboard_for_date_uses_per_date_cache():
    target = "2026-07-28"
    calls = {"n": 0}

    async def fake_fetch_espn(date_et: str):
        calls["n"] += 1
        return {"events": []}

    async def fake_fetch_stats(date_et: str):
        return {"scoreboard": {"games": []}}

    svc._date_cache.clear()
    with (
        patch.object(svc, "fetch_espn_scoreboard", side_effect=fake_fetch_espn),
        patch.object(svc, "fetch_stats_scoreboard", side_effect=fake_fetch_stats),
    ):
        first = asyncio.run(svc.get_scoreboard_for_date(target))
        second = asyncio.run(svc.get_scoreboard_for_date(target))
    assert first.date == second.date == target
    assert calls["n"] == 1
```

Also clear `_date_cache` in the existing `clear_cache` autouse fixture:

```python
@pytest.fixture(autouse=True)
def clear_cache():
    svc._cache.clear()
    if hasattr(svc, "_date_cache"):
        svc._date_cache.clear()
    yield
    svc._cache.clear()
    if hasattr(svc, "_date_cache"):
        svc._date_cache.clear()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_wnba_scoreboard_route.py::test_get_scoreboard_for_date_returns_requested_date_without_carryover tests/test_wnba_scoreboard_route.py::test_get_scoreboard_for_date_uses_per_date_cache -v`

Expected: FAIL (`get_scoreboard_for_date` / `_date_cache` missing)

- [ ] **Step 3: Implement `get_scoreboard_for_date`**

Near the today-cache helpers in `wnba_scoreboard.py`:

```python
DATED_CACHE_TTL_SECONDS = 300
_date_cache: dict[str, dict] = {}


async def get_scoreboard_for_date(date_et: str) -> WnbaScoreboardResponse:
    """Scoreboard for one ET calendar day (no overnight live carryover)."""
    cached = _date_cache.get(date_et)
    if cached is not None and time.time() < float(cached.get("expires_at") or 0):
        return cached["response"]

    games, usable = await _merged_games_for_date(date_et)
    if not usable:
        if cached is not None:
            return cached["response"]
        raise RuntimeError(f"No usable WNBA scoreboard source for {date_et}")

    response = WnbaScoreboardResponse(
        date=date_et,
        games=games,
        fetched_at=datetime.now(tz=ET).isoformat(),
    )
    _date_cache[date_et] = {
        "response": response,
        "expires_at": time.time() + DATED_CACHE_TTL_SECONDS,
    }
    return response
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_wnba_scoreboard_route.py::test_get_scoreboard_for_date_returns_requested_date_without_carryover tests/test_wnba_scoreboard_route.py::test_get_scoreboard_for_date_uses_per_date_cache -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/wnba_scoreboard.py backend/tests/test_wnba_scoreboard_route.py
git commit -m "feat: add dated WNBA scoreboard service with per-date cache"
```

---

### Task 2: Backend dated scoreboard route

**Files:**
- Modify: `backend/app/api/routes/wnba_scoreboard.py`
- Modify: `backend/tests/test_wnba_scoreboard_route.py`

**Interfaces:**
- Consumes: `get_scoreboard_for_date(date_et: str)`
- Produces: `GET /api/wnba/scoreboard?date=YYYY-MM-DD`
  - required `date` query with pattern `^\d{4}-\d{2}-\d{2}$`
  - invalid → 422
  - success → 200 + `Cache-Control: no-store`
  - upstream failure → 502 + `Cache-Control: no-store`

- [ ] **Step 1: Write the failing tests**

```python
def test_scoreboard_by_date_returns_requested_day():
    target = "2026-07-28"
    espn = json.loads((FIXTURES / "espn_wnba_scoreboard.json").read_text())

    async def fake_fetch_espn(date_et: str):
        assert date_et == target
        return espn

    async def fake_fetch_stats(date_et: str):
        assert date_et == target
        return {"scoreboard": {"games": []}}

    with (
        patch.object(svc, "fetch_espn_scoreboard", side_effect=fake_fetch_espn),
        patch.object(svc, "fetch_stats_scoreboard", side_effect=fake_fetch_stats),
    ):
        client = TestClient(app)
        res = client.get("/api/wnba/scoreboard", params={"date": target})
    assert res.status_code == 200
    assert res.headers.get("cache-control") == "no-store"
    assert res.json()["date"] == target
    assert len(res.json()["games"]) >= 1


def test_scoreboard_by_date_422_on_bad_date():
    client = TestClient(app)
    res = client.get("/api/wnba/scoreboard", params={"date": "07-28-2026"})
    assert res.status_code == 422


def test_scoreboard_by_date_empty_slate_ok():
    target = "2026-01-01"

    async def empty_espn(date_et: str):
        return {"events": []}

    async def empty_stats(date_et: str):
        return {"scoreboard": {"games": []}}

    with (
        patch.object(svc, "fetch_espn_scoreboard", side_effect=empty_espn),
        patch.object(svc, "fetch_stats_scoreboard", side_effect=empty_stats),
    ):
        client = TestClient(app)
        res = client.get("/api/wnba/scoreboard", params={"date": target})
    assert res.status_code == 200
    assert res.json()["date"] == target
    assert res.json()["games"] == []


def test_scoreboard_by_date_502_when_upstream_fails():
    async def boom(date_et: str):
        raise RuntimeError("upstream down")

    with (
        patch.object(svc, "fetch_espn_scoreboard", side_effect=boom),
        patch.object(svc, "fetch_stats_scoreboard", side_effect=boom),
    ):
        client = TestClient(app)
        res = client.get("/api/wnba/scoreboard", params={"date": "2026-07-28"})
    assert res.status_code == 502
    assert res.headers.get("cache-control") == "no-store"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_wnba_scoreboard_route.py::test_scoreboard_by_date_returns_requested_day tests/test_wnba_scoreboard_route.py::test_scoreboard_by_date_422_on_bad_date tests/test_wnba_scoreboard_route.py::test_scoreboard_by_date_empty_slate_ok tests/test_wnba_scoreboard_route.py::test_scoreboard_by_date_502_when_upstream_fails -v`

Expected: FAIL (404 on `/api/wnba/scoreboard`)

- [ ] **Step 3: Add the route**

In `backend/app/api/routes/wnba_scoreboard.py`:

```python
from fastapi import APIRouter, HTTPException, Query, Response

from app.services.wnba_scoreboard import get_scoreboard_for_date, get_today_scoreboard

@router.get("/wnba/scoreboard", response_model=WnbaScoreboardResponse)
async def wnba_scoreboard_by_date(
    response: Response,
    date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
) -> WnbaScoreboardResponse:
    response.headers["Cache-Control"] = "no-store"
    try:
        return await get_scoreboard_for_date(date)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("WNBA scoreboard unavailable for %s: %s", date, exc)
        raise HTTPException(
            status_code=502,
            detail="WNBA scoreboard is temporarily unavailable",
            headers=_NO_STORE,
        ) from exc
```

Keep the existing `/wnba/scoreboard/today` handler unchanged.

- [ ] **Step 4: Run dated + today regression tests**

Run: `cd backend && python -m pytest tests/test_wnba_scoreboard_route.py -v`

Expected: PASS (all today + dated tests)

- [ ] **Step 5: Commit**

```bash
git add backend/app/api/routes/wnba_scoreboard.py backend/tests/test_wnba_scoreboard_route.py
git commit -m "feat: expose GET /api/wnba/scoreboard?date="
```

---

### Task 3: Frontend slate date helpers

**Files:**
- Create: `frontend/src/components/league/matchupSlateDate.ts`
- Create: `frontend/src/components/league/matchupSlateDate.test.ts`

**Interfaces:**
- Produces:
  - `SLATE_ROLLOVER_HOUR_ET = 3`
  - `export function slateEtDate(now?: Date): string`
  - `export function shiftEtDate(dateIso: string, deltaDays: number): string`
  - `export function isValidEtDate(value: string | null | undefined): value is string`
  - `export function parseMatchupDateParam(raw: string | null, today: string): string`
  - `export function formatMatchupNavLabel(dateIso: string, today: string): string` → `"Today"` or `"Jul 28"`

Reuse calendar math like `nextEtDate` in `filterPropLines.ts` (UTC date arithmetic on YYYY-MM-DD). Do **not** import from filterPropLines into matchups (keep prop filter module focused); duplicate the tiny ±1 day helper inside `shiftEtDate`.

- [ ] **Step 1: Write the failing test**

```ts
import { describe, expect, it } from "vitest";
import {
  formatMatchupNavLabel,
  isValidEtDate,
  parseMatchupDateParam,
  shiftEtDate,
  slateEtDate,
} from "./matchupSlateDate";

describe("matchupSlateDate", () => {
  it("uses ET calendar date and lags before 3:00 AM ET", () => {
    // 2026-08-01 02:30 ET = 06:30 UTC
    expect(slateEtDate(new Date("2026-08-01T06:30:00Z"))).toBe("2026-07-31");
    // 2026-08-01 03:00 ET = 07:00 UTC
    expect(slateEtDate(new Date("2026-08-01T07:00:00Z"))).toBe("2026-08-01");
  });

  it("shifts YYYY-MM-DD by calendar days", () => {
    expect(shiftEtDate("2026-07-31", 1)).toBe("2026-08-01");
    expect(shiftEtDate("2026-08-01", -1)).toBe("2026-07-31");
  });

  it("validates and parses date params", () => {
    expect(isValidEtDate("2026-07-28")).toBe(true);
    expect(isValidEtDate("07-28-2026")).toBe(false);
    expect(parseMatchupDateParam("2026-07-28", "2026-08-01")).toBe("2026-07-28");
    expect(parseMatchupDateParam("nope", "2026-08-01")).toBe("2026-08-01");
    expect(parseMatchupDateParam(null, "2026-08-01")).toBe("2026-08-01");
  });

  it("labels today vs short month day", () => {
    expect(formatMatchupNavLabel("2026-08-01", "2026-08-01")).toBe("Today");
    expect(formatMatchupNavLabel("2026-07-28", "2026-08-01")).toBe("Jul 28");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test -- src/components/league/matchupSlateDate.test.ts`

Expected: FAIL (module not found)

- [ ] **Step 3: Implement helpers**

```ts
const ET = "America/New_York";
export const SLATE_ROLLOVER_HOUR_ET = 3;
const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;

export function slateEtDate(now: Date = new Date()): string {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: ET,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "numeric",
    hour12: false,
  }).formatToParts(now);
  const get = (type: string) =>
    Number(parts.find((p) => p.type === type)?.value);
  const year = get("year");
  const month = get("month");
  const day = get("day");
  let hour = get("hour");
  // en-US hourCycle can yield 24 for midnight — normalize
  if (hour === 24) hour = 0;
  const iso = `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
  if (hour < SLATE_ROLLOVER_HOUR_ET) {
    return shiftEtDate(iso, -1);
  }
  return iso;
}

export function shiftEtDate(dateIso: string, deltaDays: number): string {
  const [y, m, d] = dateIso.split("-").map(Number);
  const utc = new Date(Date.UTC(y!, (m ?? 1) - 1, d ?? 1));
  utc.setUTCDate(utc.getUTCDate() + deltaDays);
  return utc.toISOString().slice(0, 10);
}

export function isValidEtDate(
  value: string | null | undefined,
): value is string {
  if (!value || !DATE_RE.test(value)) return false;
  const [y, m, d] = value.split("-").map(Number);
  const dt = new Date(Date.UTC(y!, (m ?? 1) - 1, d ?? 1));
  return dt.toISOString().slice(0, 10) === value;
}

export function parseMatchupDateParam(
  raw: string | null,
  today: string,
): string {
  return isValidEtDate(raw) ? raw : today;
}

export function formatMatchupNavLabel(
  dateIso: string,
  today: string,
): string {
  if (dateIso === today) return "Today";
  const date = new Date(`${dateIso}T12:00:00-04:00`);
  return new Intl.DateTimeFormat("en-US", {
    timeZone: ET,
    month: "short",
    day: "numeric",
  }).format(date);
}
```

If `slateEtDate` hour parsing is flaky under `hour12: false`, switch to `en-CA` date parts + a separate `hour`/`hourCycle: "h23"` formatter — tests above are the contract.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test -- src/components/league/matchupSlateDate.test.ts`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/league/matchupSlateDate.ts frontend/src/components/league/matchupSlateDate.test.ts
git commit -m "feat: add matchup slate date helpers"
```

---

### Task 4: API client + scoreboard hook for optional date

**Files:**
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/hooks/useWnbaScoreboard.ts`
- Create: `frontend/src/hooks/useWnbaScoreboard.test.tsx` (if none exists; otherwise extend)
- Modify: `frontend/openapi.json` (add `/api/wnba/scoreboard` path mirroring today response)
- Modify: `frontend/src/lib/api.schema.d.ts` via `npm run generate:api` **or** minimal hand edit if generate is blocked — prefer generate from updated openapi.json

**Interfaces:**
- Produces:
  - `export async function fetchWnbaScoreboardByDate(dateEt: string): Promise<WnbaScoreboardResponse>`
  - `export function useWnbaScoreboard(dateEt?: string)` — when omitted or equal to `slateEtDate()`, uses `/today` + live polling; otherwise dated fetch, no polling
- Existing callers (`HomeChromeLayout`, `HomePage`, prop picks) keep calling `useWnbaScoreboard()` with no args

- [ ] **Step 1: Write the failing hook/API tests**

```ts
// frontend/src/lib/api.scoreboardDate.test.ts
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { fetchWnbaScoreboardByDate } from "./api";

describe("fetchWnbaScoreboardByDate", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ date: "2026-07-28", games: [], fetched_at: "" }),
      }),
    );
  });
  afterEach(() => vi.unstubAllGlobals());

  it("calls /api/wnba/scoreboard?date=", async () => {
    await fetchWnbaScoreboardByDate("2026-07-28");
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining("/api/wnba/scoreboard?date=2026-07-28"),
      expect.any(Object),
    );
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test -- src/lib/api.scoreboardDate.test.ts`

Expected: FAIL (`fetchWnbaScoreboardByDate` not exported)

- [ ] **Step 3: Implement client + hook**

```ts
// api.ts
export async function fetchWnbaScoreboardByDate(
  dateEt: string,
): Promise<WnbaScoreboardResponse> {
  const res = await fetch(
    `${API_BASE}/api/wnba/scoreboard?date=${encodeURIComponent(dateEt)}`,
    { headers: { Accept: "application/json" }, cache: "no-store" },
  );
  if (!res.ok) {
    throw new Error(`Scoreboard request failed: ${res.status}`);
  }
  return res.json();
}
```

```ts
// useWnbaScoreboard.ts
import { slateEtDate } from "@/components/league/matchupSlateDate";
import { fetchWnbaScoreboard, fetchWnbaScoreboardByDate } from "@/lib/api";

export function useWnbaScoreboard(dateEt?: string) {
  const today = slateEtDate();
  const selected = dateEt ?? today;
  const isToday = selected === today;

  const query = useQuery({
    queryKey: isToday
      ? ["wnba", "scoreboard", "today"]
      : ["wnba", "scoreboard", selected],
    queryFn: () =>
      isToday
        ? fetchWnbaScoreboard()
        : fetchWnbaScoreboardByDate(selected),
    refetchInterval: (q) =>
      isToday && shouldPollScoreboard(q.state.data?.games)
        ? REFETCH_MS
        : false,
  });
  // ... same return shape as today
}
```

Add OpenAPI path `/api/wnba/scoreboard` with required `date` query (same 200 schema as today), then `cd frontend && npm run generate:api`.

- [ ] **Step 4: Run tests**

Run: `cd frontend && npm test -- src/lib/api.scoreboardDate.test.ts src/hooks/useWnbaScoreboard`

Expected: PASS (existing layout tests still pass if run full suite later)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/api.ts frontend/src/hooks/useWnbaScoreboard.ts frontend/src/lib/api.scoreboardDate.test.ts frontend/openapi.json frontend/src/lib/api.schema.d.ts
git commit -m "feat: fetch WNBA scoreboard by date from the client"
```

---

### Task 5: Enable MatchupsPanel date controls

**Files:**
- Modify: `frontend/src/components/league/MatchupsPanel.tsx`
- Modify: `frontend/src/components/league/MatchupsPanel.test.tsx`

**Interfaces:**
- Consumes: `formatMatchupNavLabel` (or receives `navLabel` as prop)
- Produces props:
  ```ts
  type MatchupsPanelProps = {
    games: MatchupGame[];
    isLoading?: boolean;
    isError?: boolean;
    selectedDate: string;
    todayDate: string;
    onPrevDay: () => void;
    onNextDay: () => void;
    onGoToday: () => void;
  };
  ```

- [ ] **Step 1: Update tests first (failing until props wired)**

Replace the disabled-Today assertion with:

```tsx
it("enables day navigation and returns to today from center control", async () => {
  const user = userEvent.setup();
  const onPrevDay = vi.fn();
  const onNextDay = vi.fn();
  const onGoToday = vi.fn();
  renderPanel([live], {
    selectedDate: "2026-07-28",
    todayDate: "2026-08-01",
    onPrevDay,
    onNextDay,
    onGoToday,
  });
  expect(screen.getByRole("button", { name: /previous day/i })).toBeEnabled();
  expect(screen.getByRole("button", { name: /next day/i })).toBeEnabled();
  expect(screen.getByRole("button", { name: "Jul 28" })).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: /previous day/i }));
  await user.click(screen.getByRole("button", { name: /next day/i }));
  await user.click(screen.getByRole("button", { name: "Jul 28" }));
  expect(onPrevDay).toHaveBeenCalledOnce();
  expect(onNextDay).toHaveBeenCalledOnce();
  expect(onGoToday).toHaveBeenCalledOnce();
});

it("shows Today label on the slate date", () => {
  renderPanel([live], {
    selectedDate: "2026-08-01",
    todayDate: "2026-08-01",
    onPrevDay: () => {},
    onNextDay: () => {},
    onGoToday: () => {},
  });
  expect(screen.getByRole("button", { name: "Today" })).toBeInTheDocument();
});

it("shows empty slate copy when no games", () => {
  renderPanel([], {
    selectedDate: "2026-07-28",
    todayDate: "2026-08-01",
    onPrevDay: () => {},
    onNextDay: () => {},
    onGoToday: () => {},
  });
  expect(screen.getByText("No games on this slate")).toBeInTheDocument();
});
```

Import `userEvent` from `@testing-library/user-event` and `vi` from `vitest`. Pass new required props through `renderPanel` defaults for other tests.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npm test -- src/components/league/MatchupsPanel.test.tsx`

Expected: FAIL (missing props / still disabled)

- [ ] **Step 3: Implement panel controls**

```tsx
export function MatchupsPanel({
  games,
  isLoading = false,
  isError = false,
  selectedDate,
  todayDate,
  onPrevDay,
  onNextDay,
  onGoToday,
}: MatchupsPanelProps) {
  const navLabel = formatMatchupNavLabel(selectedDate, todayDate);
  // ...
  <button type="button" aria-label="Previous day" onClick={onPrevDay}
    className="flex size-8 items-center justify-center rounded-md border border-white/10 text-white/70 hover:bg-white/5">
    <ChevronLeft ... />
  </button>
  <button type="button" onClick={onGoToday}
    className="min-w-14 text-center text-sm font-medium text-white/55 hover:text-white/80">
    {navLabel}
  </button>
  <button type="button" aria-label="Next day" onClick={onNextDay} ...>
    <ChevronRight ... />
  </button>
  // empty: "No games on this slate"
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npm test -- src/components/league/MatchupsPanel.test.tsx`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/league/MatchupsPanel.tsx frontend/src/components/league/MatchupsPanel.test.tsx
git commit -m "feat: enable matchups day navigation controls"
```

---

### Task 6: Wire LeagueMatchupsPage URL + odds gate

**Files:**
- Modify: `frontend/src/pages/LeagueMatchupsPage.tsx`
- Create: `frontend/src/pages/LeagueMatchupsPage.test.tsx` (or extend `AppRouter.test.tsx` if that already covers matchups — prefer a focused page test with mocked hooks)

**Interfaces:**
- Consumes: `useSearchParams`, `matchupSlateDate` helpers, `useWnbaScoreboard(selectedDate)`, `useWnbaOdds` (enabled only when today)
- Behavior:
  - `today = slateEtDate()`
  - `selected = parseMatchupDateParam(searchParams.get("date"), today)`
  - If raw param invalid, `setSearchParams` to clear `date`
  - Prev → `shiftEtDate(selected, -1)`; Next → `+1`; if result === today, delete `date`, else set `date`
  - Odds: `mergeMatchupOdds(..., isToday ? odds.data?.games : undefined)`
  - Hero `dateEt={data?.date ?? selected}`

- [ ] **Step 1: Write the failing page test**

```tsx
import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { LeagueMatchupsPage } from "./LeagueMatchupsPage";

vi.mock("@/hooks/useWnbaScoreboard", () => ({
  useWnbaScoreboard: (dateEt?: string) => ({
    games: [],
    isLoading: false,
    hasNeverLoaded: false,
    data: { date: dateEt ?? "2026-08-01", games: [], fetched_at: "" },
  }),
}));

vi.mock("@/hooks/useWnbaOdds", () => ({
  useWnbaOdds: () => ({ data: undefined }),
}));

vi.mock("@/components/league/matchupSlateDate", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/league/matchupSlateDate")>();
  return { ...actual, slateEtDate: () => "2026-08-01" };
});

function renderAt(path: string) {
  const client = new QueryClient();
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/wnba/matchups" element={<LeagueMatchupsPage league="wnba" />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("LeagueMatchupsPage date nav", () => {
  it("writes ?date when moving off today and clears it returning", async () => {
    const user = userEvent.setup();
    renderAt("/wnba/matchups");
    expect(screen.getByRole("button", { name: "Today" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /previous day/i }));
    expect(screen.getByRole("button", { name: "Jul 31" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Jul 31" }));
    expect(screen.getByRole("button", { name: "Today" })).toBeInTheDocument();
  });
});
```

Adjust label expectations if `formatMatchupNavLabel` produces `Jul 31` for `2026-07-31`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test -- src/pages/LeagueMatchupsPage.test.tsx`

Expected: FAIL (page still passes no nav props / TypeScript or runtime)

- [ ] **Step 3: Wire the page**

```tsx
function WnbaMatchupsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const today = slateEtDate();
  const raw = searchParams.get("date");
  const selectedDate = parseMatchupDateParam(raw, today);
  const isToday = selectedDate === today;

  useEffect(() => {
    if (raw !== null && !isValidEtDate(raw)) {
      setSearchParams({}, { replace: true });
    }
  }, [raw, setSearchParams]);

  const { games, isLoading, hasNeverLoaded, data } =
    useWnbaScoreboard(selectedDate);
  const oddsQuery = useWnbaOdds();
  const matchupGames = mergeMatchupOdds(
    mapToMatchupGames(games),
    isToday ? oddsQuery.data?.games : undefined,
  );

  const setDate = (next: string) => {
    if (next === today) setSearchParams({});
    else setSearchParams({ date: next });
  };

  return (
    <div className="space-y-0">
      <LeagueHero league="wnba" dateEt={data?.date ?? selectedDate} />
      <LeagueSubnav league="wnba" />
      <MatchupsPanel
        games={matchupGames}
        isLoading={isLoading}
        isError={hasNeverLoaded}
        selectedDate={selectedDate}
        todayDate={today}
        onPrevDay={() => setDate(shiftEtDate(selectedDate, -1))}
        onNextDay={() => setDate(shiftEtDate(selectedDate, 1))}
        onGoToday={() => setDate(today)}
      />
    </div>
  );
}
```

Optionally `enabled: isToday` on odds via changing `useWnbaOdds` — if that hook has no `enabled` flag yet, gating the merge is enough for this task (YAGNI on hook options).

- [ ] **Step 4: Run page + panel + helper tests**

Run: `cd frontend && npm test -- src/pages/LeagueMatchupsPage.test.tsx src/components/league/MatchupsPanel.test.tsx src/components/league/matchupSlateDate.test.ts`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/LeagueMatchupsPage.tsx frontend/src/pages/LeagueMatchupsPage.test.tsx
git commit -m "feat: wire WNBA matchups date nav to URL and odds gate"
```

---

### Task 7: Regression verification

**Files:** none (verification only)

- [ ] **Step 1: Run backend scoreboard suite**

Run: `cd backend && python -m pytest tests/test_wnba_scoreboard_route.py tests/test_wnba_scoreboard_normalize.py -v`

Expected: PASS

- [ ] **Step 2: Run frontend focused + build**

Run: `cd frontend && npm test -- src/components/league/ src/pages/LeagueMatchupsPage.test.tsx src/lib/api.scoreboardDate.test.ts src/hooks/useWnbaScoreboard && npm run build`

Expected: PASS / build succeeds

- [ ] **Step 3: Manual smoke (optional if local servers up)**

- Open `/wnba/matchups` → Today, chevrons work
- Prev → URL gains `?date=`; odds pills gone
- Center control → back to Today; odds return when books available
- Empty offseason day → “No games on this slate”

- [ ] **Step 4: Commit only if Step 2 produced leftover OpenAPI/docs fixes; otherwise done**

---

## Spec coverage checklist

| Spec requirement | Task |
| --- | --- |
| `GET /api/wnba/scoreboard?date=` | 1–2 |
| No overnight carryover on dated | 1 |
| Per-date ~5 min cache | 1 |
| `/today` unchanged | 2 regression |
| Unlimited ±1 day nav | 5–6 |
| `?date=` URL; omit on today | 6 |
| Center label Today / short date; click → today | 5–6 |
| Odds only on today | 6 |
| Empty “No games on this slate” | 5 |
| No polling off today | 4 |
| Invalid date → today | 3, 6 |
| Hero follows selected date | 6 |

## Self-review notes

- No TBD/placeholder steps; signatures consistent (`get_scoreboard_for_date`, `fetchWnbaScoreboardByDate`, `shiftEtDate`, panel callbacks).
- `useWnbaScoreboard(dateEt?)` keeps shared today query key so ticker + matchups share cache on today.
- OpenAPI update scoped to Task 4 so `check:api` stays green.
