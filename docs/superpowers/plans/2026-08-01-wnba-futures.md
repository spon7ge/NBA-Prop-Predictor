# WNBA Futures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Explore **Clutch** with **Futures** linking to `/wnba/futures`, backed by ESPN’s season futures API and a quiet Finals Winner odds board.

**Architecture:** Backend `wnba_futures` service fetches `sports.core.api.espn.com/.../wnba/seasons/{season}/futures`, resolves team `$ref`s, normalizes markets/entries, and caches ~5 min. Frontend adds subnav link, route, page, and TanStack Query hook mirroring standings patterns.

**Tech Stack:** FastAPI · httpx · pytest · React 19 · TypeScript · TanStack Query · Vitest · openapi-typescript

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-01-wnba-futures-design.md`
- Upstream: `GET https://sports.core.api.espn.com/v2/sports/basketball/leagues/wnba/seasons/{season}/futures`
- Season via `current_wnba_season_year()` from `wnba_standings` (reuse)
- Map market name `WNBA - Winner` → display **Finals Winner**
- Prefer provider name containing `ESPN BET` (case-insensitive)
- Sort entries by American odds ascending (favorites first)
- WNBA-only route; NBA Futures stays disabled
- Quiet Apple surfaces; `Cache-Control: no-store` on HTTP
- TDD each task; sync OpenAPI after backend route lands

## File map

| File | Responsibility |
| --- | --- |
| `backend/app/schemas/wnba_futures.py` | Response models |
| `backend/app/services/wnba_futures.py` | Fetch, team resolve, normalize, cache |
| `backend/app/api/routes/wnba_futures.py` | `GET /wnba/futures` |
| `backend/app/main.py` | Register router |
| `backend/app/openapi_export.py` | Add required path |
| `backend/tests/fixtures/espn_wnba_futures.json` | Sample payload |
| `backend/tests/test_wnba_futures.py` | Normalize + route tests |
| `frontend/src/lib/api.ts` + OpenAPI artifacts | `fetchWnbaFutures` |
| `frontend/src/hooks/useWnbaFutures.ts` | Query hook |
| `frontend/src/components/league/FuturesBoard.tsx` | Quiet odds board UI |
| `frontend/src/pages/LeagueFuturesPage.tsx` | Page shell |
| `frontend/src/components/league/LeagueSubnav.tsx` | Clutch → Futures |
| `frontend/src/AppRouter.tsx` | Route |

---

### Task 1: Schema + normalize + team resolve

**Files:**
- Create: `backend/app/schemas/wnba_futures.py`
- Create: `backend/app/services/wnba_futures.py`
- Create: `backend/tests/fixtures/espn_wnba_futures.json`
- Create: `backend/tests/fixtures/espn_wnba_team_8.json` (minimal team payload)
- Test: `backend/tests/test_wnba_futures.py`

**Interfaces:**
- Produces:
  - `WnbaFuturesEntry(team_id, abbrev, name, logo_url: str | None, odds_american: str)`
  - `WnbaFuturesMarket(id, name, display_name, provider, entries: list[WnbaFuturesEntry])`
  - `WnbaFuturesResponse(season: int, as_of: str, markets: list[WnbaFuturesMarket], error: str | None = None)`
  - `display_name_for_market(name: str) -> str`
  - `parse_american_odds(value: str) -> int | None`
  - `pick_provider(futures: list[dict]) -> dict | None`
  - `async def resolve_team(ref_or_id: str, client: httpx.AsyncClient) -> dict | None`
  - `async def normalize_futures_payload(payload: dict, season: int, client: httpx.AsyncClient) -> WnbaFuturesResponse`

- [ ] **Step 1: Write fixtures + failing tests**

Create a trimmed fixture from live ESPN shape (one market `WNBA - Winner`, one provider `ESPN BET`, 3–4 books with team `$ref`s). Team fixture: `{ "id": "8", "abbreviation": "NYL", "displayName": "New York Liberty", "logos": [{"href": "https://example.com/nyl.png"}] }`.

```python
def test_display_name_maps_winner():
    assert svc.display_name_for_market("WNBA - Winner") == "Finals Winner"
    assert svc.display_name_for_market("Other Market") == "Other Market"


def test_parse_american_odds():
    assert svc.parse_american_odds("+250") == 250
    assert svc.parse_american_odds("-150") == -150
    assert svc.parse_american_odds("even") is None


def test_normalize_sorts_favorites_first_and_maps_teams(monkeypatch):
    payload = json.loads(FIXTURE.read_text())
    async def fake_resolve(ref, client):
        # return dict keyed by team id parsed from ref
        ...
    # asyncio.run normalize with mocked resolve
    # assert display_name Finals Winner, provider ESPN BET
    # assert first entry has shortest odds
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `cd backend && python3 -m pytest tests/test_wnba_futures.py -v`

- [ ] **Step 3: Implement schema + service helpers + normalize**

```python
# schemas/wnba_futures.py — Pydantic models matching spec JSON

# services/wnba_futures.py
FUTURES_URL = (
    "https://sports.core.api.espn.com/v2/sports/basketball/leagues/"
    "wnba/seasons/{season}/futures"
)
CACHE_TTL_SECONDS = 300.0

def display_name_for_market(name: str) -> str:
    if name.strip() == "WNBA - Winner":
        return "Finals Winner"
    return name.strip() or "Futures"

def parse_american_odds(value: str) -> int | None:
    text = str(value or "").strip().replace("−", "-")
    if not text or text[0] not in "+-" and not text.isdigit():
        # allow +250 / -150 / 250
        ...
    try:
        return int(text)
    except ValueError:
        return None

def pick_provider(futures: list) -> dict | None:
    # prefer active + name contains "espn bet"; else first active; else first
```

Team resolve: GET `$ref` URL (http→https), parse `id`/`abbreviation`/`displayName`/`logos[0].href`. In-memory `_team_cache: dict[str, dict]`.

Normalize: for each `items[]` market, pick provider, resolve each book’s team, build entries, sort by `parse_american_odds` (None last), append market. `as_of` = UTC ISO Z.

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add backend/app/schemas/wnba_futures.py backend/app/services/wnba_futures.py \
  backend/tests/fixtures/espn_wnba_futures.json backend/tests/fixtures/espn_wnba_team_8.json \
  backend/tests/test_wnba_futures.py
git commit -m "feat: normalize ESPN WNBA futures markets and team odds"
```

---

### Task 2: Fetch + cache + route

**Files:**
- Modify: `backend/app/services/wnba_futures.py`
- Create: `backend/app/api/routes/wnba_futures.py`
- Modify: `backend/app/main.py`
- Modify: `backend/tests/test_wnba_futures.py`

**Interfaces:**
- Consumes: `normalize_futures_payload`, `current_wnba_season_year` from `wnba_standings`
- Produces: `async def get_wnba_futures() -> WnbaFuturesResponse`
- Route: `GET /api/wnba/futures` → 200 or 502

- [ ] **Step 1: Failing route/cache tests**

```python
def test_futures_route_ok(monkeypatch):
    # patch get_wnba_futures to return fixture response
    res = client.get("/api/wnba/futures")
    assert res.status_code == 200
    assert res.headers["cache-control"] == "no-store"
    assert res.json()["markets"][0]["display_name"] == "Finals Winner"


def test_get_wnba_futures_uses_cache(monkeypatch):
    calls = {"n": 0}
    async def fake_fetch(season):
        calls["n"] += 1
        return json.loads(FIXTURE.read_text())
    # first + second call → calls["n"] == 1
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement fetch/cache/route**

```python
async def fetch_espn_futures(season: int) -> dict:
    url = FUTURES_URL.format(season=season)
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, params={"limit": 200, "lang": "en", "region": "us"})
        r.raise_for_status()
        return r.json()

async def get_wnba_futures() -> WnbaFuturesResponse:
    # TTL cache; on error return stale if same season else raise
    season = current_wnba_season_year()
    ...
```

Route mirrors standings (`HTTPException` 502, `Cache-Control: no-store`). Register in `main.py`.

- [ ] **Step 4: Run full `test_wnba_futures.py` — PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add GET /api/wnba/futures with ESPN fetch and cache"
```

---

### Task 3: OpenAPI + frontend API client

**Files:**
- Modify: `backend/app/openapi_export.py` — add `"/api/wnba/futures"` to `REQUIRED_WNBA_PATHS`
- Modify: `frontend/openapi.json`, `frontend/src/lib/api.schema.d.ts`
- Modify: `frontend/src/lib/api.ts`
- Test: `backend/tests/test_export_openapi.py`; optional `frontend/src/lib/api.test.ts` futures case

**Interfaces:**
- Produces: `fetchWnbaFutures(): Promise<ApiWnbaFuturesResponse>`
- Types: `ApiWnbaFuturesResponse`, `ApiWnbaFuturesMarket`, `ApiWnbaFuturesEntry`

- [ ] **Step 1: Add required path; fail export test if missing**

- [ ] **Step 2: `python scripts/export_openapi.py` then `cd frontend && npm run generate:api`**

- [ ] **Step 3: Add fetch helper**

```typescript
export async function fetchWnbaFutures(): Promise<ApiWnbaFuturesResponse> {
  const res = await fetch(`${API_BASE}/api/wnba/futures`, {
    headers: { Accept: "application/json" },
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`Futures request failed: ${res.status}`);
  return res.json();
}
```

- [ ] **Step 4: Pytest export + optional vitest api — PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "chore: sync OpenAPI and fetch helper for WNBA futures"
```

---

### Task 4: Subnav Futures link

**Files:**
- Modify: `frontend/src/components/league/LeagueSubnav.tsx`
- Modify: `frontend/src/components/league/LeagueSubnav.test.tsx`

**Interfaces:**
- `"Futures"` replaces `"Clutch"`; WNBA → `/wnba/futures`; NBA disabled; active on `/futures`

- [ ] **Step 1: Failing tests**

```typescript
it("links Futures for WNBA and leaves it disabled for NBA", () => {
  // wnba: getByRole link Futures href /wnba/futures
  // nba: getByRole button Futures disabled; no Clutch
});
```

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement subnav changes**

- [ ] **Step 4: Run — PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: replace Clutch with Futures in league Explore subnav"
```

---

### Task 5: Futures board UI + page + route

**Files:**
- Create: `frontend/src/hooks/useWnbaFutures.ts`
- Create: `frontend/src/components/league/FuturesBoard.tsx`
- Create: `frontend/src/components/league/FuturesBoard.test.tsx`
- Create: `frontend/src/pages/LeagueFuturesPage.tsx`
- Create: `frontend/src/pages/LeagueFuturesPage.test.tsx` (optional thin)
- Modify: `frontend/src/AppRouter.tsx`, `frontend/src/AppRouter.test.tsx`

**Interfaces:**
- `useWnbaFutures()` → query + `hasNeverLoaded` (standings pattern)
- `FuturesBoard({ markets, isLoading, isError, season })`
- `LeagueFuturesPage` wires subnav + board

- [ ] **Step 1: Failing board + router tests**

```typescript
it("renders Finals Winner rows with odds and provider", () => {
  render(
    <FuturesBoard
      season={2026}
      markets={[
        {
          id: "8146",
          name: "WNBA - Winner",
          display_name: "Finals Winner",
          provider: "ESPN BET",
          entries: [
            {
              team_id: "8",
              abbrev: "NYL",
              name: "New York Liberty",
              logo_url: null,
              odds_american: "+250",
            },
          ],
        },
      ]}
      isLoading={false}
      isError={false}
    />,
  );
  expect(screen.getByText("Finals Winner")).toBeInTheDocument();
  expect(screen.getByText("New York Liberty")).toBeInTheDocument();
  expect(screen.getByText("+250")).toBeInTheDocument();
  expect(screen.getByText(/Odds by/)).toBeInTheDocument();
  expect(screen.getByText("ESPN BET")).toBeInTheDocument();
});
```

AppRouter test: navigate `/wnba/futures` shows Futures content (mock hook if needed).

- [ ] **Step 2: Run — FAIL**

- [ ] **Step 3: Implement board, hook, page, route**

Board: for each market, heading + `Odds by {provider}` + rows (`TeamAbbrevAvatar`, name, mono odds). Skeletons when loading; “Unable to load futures” when `isError`; “No futures listed” when empty.

```tsx
// LeagueFuturesPage
export function LeagueFuturesPage() {
  const { data, isLoading, hasNeverLoaded } = useWnbaFutures();
  return (
    <div className="space-y-0">
      <LeagueSubnav league="wnba" />
      <FuturesBoard
        season={data?.season ?? new Date().getFullYear()}
        markets={data?.markets ?? []}
        isLoading={isLoading && !data}
        isError={hasNeverLoaded}
      />
    </div>
  );
}
```

- [ ] **Step 4: Vitest board + AppRouter + subnav — PASS**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: add WNBA futures page with Finals Winner odds board"
```

---

### Task 6: Manual smoke

- [ ] Restart API; `curl -s localhost:8000/api/wnba/futures | python3 -m json.tool | head`
- [ ] Open `/wnba/futures` — Finals Winner list, provider caption, subnav active
- [ ] NBA hub: Futures still disabled
- [ ] Commit only if smoke required fixes

---

## Spec coverage

| Spec item | Task |
| --- | --- |
| Clutch → Futures subnav | 4 |
| `/wnba/futures` page | 5 |
| ESPN core API | 1–2 |
| Finals Winner display map | 1 |
| Provider preference | 1 |
| Sort favorites first | 1 |
| Team resolve + logos | 1 |
| Cache ~5 min + 502 | 2 |
| OpenAPI | 3 |
| NBA disabled | 4 |

## Out of scope (do not implement)

NBA futures, season picker, Sharp books, HTML scrape, sportsbook deep links
