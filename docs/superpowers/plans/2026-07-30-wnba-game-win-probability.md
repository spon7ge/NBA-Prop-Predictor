# WNBA Game Win Probability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a shared win-probability panel beneath the WNBA game-detail shot chart and play-by-play sections using ESPN summary data, with graceful fallbacks when ESPN omits predictor or team-stat data.

**Architecture:** Keep a single `GET /api/wnba/games/{espnEventId}` data flow and extend its normalized payload with a `win_probability` object. Map that payload into frontend game-detail types and render a dedicated `WinProbabilityPanel` beneath the existing two-column layout, preserving current page behavior when win-probability data is unavailable.

**Tech Stack:** FastAPI, Pydantic, pytest, React, TypeScript, Vitest, Testing Library, Tailwind utility classes

## Global Constraints

- Keep one shared full-width panel below `Shot chart` and `Play-by-play`; do not duplicate the content under each existing panel.
- Extend the existing backend ESPN summary normalization instead of adding a second API endpoint.
- The chart must be interactive on hover/focus and show the corresponding score state plus both teams' win percentages.
- Team stats are limited to `Field goal %`, `Three point %`, `Free throw %`, `Rebounds`, `Offensive rebounds`, and `Assists`.
- If the timeline or stat block is missing, render the available subsection; if both are missing, degrade gracefully without breaking the page.
- Follow TDD strictly: no production code before a failing test.

---

## File Structure

- Modify `backend/app/schemas/wnba_game_detail.py` to add Pydantic models for normalized win-probability points, stat rows, and the optional parent block.
- Modify `backend/app/services/wnba_game_detail.py` to parse ESPN predictor/stat data into the new models without changing existing header/shot/play behavior.
- Modify `backend/tests/test_wnba_game_detail_normalize.py` to cover normalization of full, partial, and missing win-probability data.
- Modify `backend/tests/test_wnba_game_detail_route.py` to verify the route serializes the new field and remains backward-compatible.
- Modify `frontend/src/lib/api.ts` to type the new API shape.
- Modify `frontend/src/components/game/types.ts` and `frontend/src/components/game/mapGameDetail.ts` to carry the new camelCase data into the UI layer.
- Create `frontend/src/components/game/WinProbabilityPanel.tsx` for the interactive chart/stats UI.
- Create `frontend/src/components/game/WinProbabilityPanel.test.tsx` for rendering, hover/focus, and fallback coverage.
- Modify `frontend/src/pages/GameDetailPage.tsx` to place the panel beneath the existing grid.
- Modify `frontend/src/components/game/testFixtures.ts`, `frontend/src/components/game/mapGameDetail.test.ts`, and `frontend/src/AppRouter.test.tsx` only if needed to support new UI test data without duplicating fixture setup.

### Task 1: Backend schema and normalization

**Files:**
- Modify: `backend/app/schemas/wnba_game_detail.py`
- Modify: `backend/app/services/wnba_game_detail.py`
- Test: `backend/tests/test_wnba_game_detail_normalize.py`

**Interfaces:**
- Consumes: existing `normalize_espn_summary(payload: dict, *, espn_event_id: str, fetched_at: str) -> WnbaGameDetail`
- Produces: `GameDetailWinProbabilityPoint`, `GameDetailTeamStat`, `GameDetailWinProbability`, and `WnbaGameDetail.win_probability: GameDetailWinProbability | None`

- [ ] **Step 1: Write the failing normalization test**

```python
def test_normalize_includes_win_probability_and_team_stats():
    payload = load_fixture("espn_wnba_summary_with_predictor.json")

    detail = normalize_espn_summary(
        payload,
        espn_event_id="401857098",
        fetched_at="2026-07-30T00:00:00-04:00",
    )

    assert detail.win_probability is not None
    assert detail.win_probability.summary == "Above the midline favors PHX"
    assert detail.win_probability.timeline[-1].home_win_pct == 54
    assert detail.win_probability.timeline[-1].away_score == 10
    assert detail.win_probability.team_stats == [
        GameDetailTeamStat(
            key="field_goal_pct",
            label="Field goal %",
            away_value=41,
            home_value=49,
        ),
        GameDetailTeamStat(
            key="three_point_pct",
            label="Three point %",
            away_value=36,
            home_value=31,
        ),
        GameDetailTeamStat(
            key="free_throw_pct",
            label="Free throw %",
            away_value=79,
            home_value=74,
        ),
        GameDetailTeamStat(
            key="rebounds",
            label="Rebounds",
            away_value=33,
            home_value=34,
        ),
        GameDetailTeamStat(
            key="offensive_rebounds",
            label="Offensive rebounds",
            away_value=13,
            home_value=6,
        ),
        GameDetailTeamStat(
            key="assists",
            label="Assists",
            away_value=24,
            home_value=19,
        ),
    ]
```

- [ ] **Step 2: Run the normalization test to verify it fails**

Run: `PYTHONPATH=backend pytest backend/tests/test_wnba_game_detail_normalize.py::test_normalize_includes_win_probability_and_team_stats -v`

Expected: FAIL because `WnbaGameDetail` has no `win_probability` field or the normalization output omits it.

- [ ] **Step 3: Write the minimal schema and parser implementation**

```python
class GameDetailWinProbabilityPoint(BaseModel):
    id: str
    period: int
    clock: str
    away_score: int
    home_score: int
    away_win_pct: int
    home_win_pct: int
    team_id: str | None


class GameDetailTeamStat(BaseModel):
    key: str
    label: str
    away_value: int
    home_value: int


class GameDetailWinProbability(BaseModel):
    summary: str | None
    timeline: list[GameDetailWinProbabilityPoint]
    team_stats: list[GameDetailTeamStat]


def _normalize_win_probability(payload: dict) -> GameDetailWinProbability | None:
    predictor = payload.get("predictor") or {}
    graph = predictor.get("gameFlow") or predictor.get("homeTeamGameProjection") or []
    raw_stats = predictor.get("teamStats") or []

    timeline = [
        GameDetailWinProbabilityPoint(
            id=str(point.get("id") or f"wp-{index}"),
            period=int(point.get("period") or 0),
            clock=str(point.get("clock") or ""),
            away_score=int(point.get("awayScore") or 0),
            home_score=int(point.get("homeScore") or 0),
            away_win_pct=int(round(float(point.get("awayWinPct") or 0))),
            home_win_pct=int(round(float(point.get("homeWinPct") or 0))),
            team_id=str(point.get("teamId") or "") or None,
        )
        for index, point in enumerate(graph)
        if point.get("awayWinPct") is not None or point.get("homeWinPct") is not None
    ]

    allowed_stats = {
        "field_goal_pct": "Field goal %",
        "three_point_pct": "Three point %",
        "free_throw_pct": "Free throw %",
        "rebounds": "Rebounds",
        "offensive_rebounds": "Offensive rebounds",
        "assists": "Assists",
    }
    team_stats = [
        GameDetailTeamStat(
            key=key,
            label=label,
            away_value=int(raw["away"]),
            home_value=int(raw["home"]),
        )
        for key, label in allowed_stats.items()
        if (raw := predictor.get("teamStatsMap", {}).get(key))
    ]

    if not timeline and not team_stats:
        return None

    return GameDetailWinProbability(
        summary=str(predictor.get("summary") or "") or None,
        timeline=timeline,
        team_stats=team_stats,
    )
```

- [ ] **Step 4: Wire the new field into `normalize_espn_summary`**

```python
win_probability = _normalize_win_probability(payload)

return WnbaGameDetail(
    espn_event_id=espn_event_id,
    status=status,
    status_label=status_label,
    venue=str(venue) if venue else None,
    away=team(away_c, FALLBACK_AWAY_COLOR),
    home=team(home_c, FALLBACK_HOME_COLOR),
    fg_made=sum(1 for s in shots if s.made),
    fg_attempted=len(shots),
    latest_play=latest,
    shots=shots,
    plays=list(reversed(plays)),
    win_probability=win_probability,
    fetched_at=fetched_at,
)
```

- [ ] **Step 5: Run the normalization test to verify it passes**

Run: `PYTHONPATH=backend pytest backend/tests/test_wnba_game_detail_normalize.py::test_normalize_includes_win_probability_and_team_stats -v`

Expected: PASS

- [ ] **Step 6: Add partial and missing-data regression tests**

```python
def test_normalize_win_probability_allows_partial_sections():
    payload = load_fixture("espn_wnba_summary_with_predictor.json")
    payload["predictor"]["teamStatsMap"] = {}

    detail = normalize_espn_summary(
        payload,
        espn_event_id="401857098",
        fetched_at="2026-07-30T00:00:00-04:00",
    )

    assert detail.win_probability is not None
    assert len(detail.win_probability.timeline) > 0
    assert detail.win_probability.team_stats == []


def test_normalize_win_probability_returns_none_when_missing_everything():
    payload = load_fixture("espn_wnba_summary.json")

    detail = normalize_espn_summary(
        payload,
        espn_event_id="401857098",
        fetched_at="2026-07-30T00:00:00-04:00",
    )

    assert detail.win_probability is None
```

- [ ] **Step 7: Run the full normalization test file**

Run: `PYTHONPATH=backend pytest backend/tests/test_wnba_game_detail_normalize.py -v`

Expected: PASS with existing normalization tests still green.

### Task 2: Backend route serialization coverage

**Files:**
- Modify: `backend/tests/test_wnba_game_detail_route.py`
- Test: `backend/tests/test_wnba_game_detail_route.py`

**Interfaces:**
- Consumes: `GET /api/wnba/games/{espnEventId}` returning `WnbaGameDetail`
- Produces: route-level confidence that `win_probability` serializes and that `null` is allowed

- [ ] **Step 1: Write the failing route test for populated predictor data**

```python
def test_game_detail_route_includes_win_probability(client, monkeypatch):
    async def fake_fetch(_: str) -> dict:
        return load_fixture("espn_wnba_summary_with_predictor.json")

    monkeypatch.setattr(game_detail_service, "fetch_espn_summary", fake_fetch)
    game_detail_service.clear_game_detail_cache()

    response = client.get("/api/wnba/games/401857098")

    assert response.status_code == 200
    body = response.json()
    assert body["win_probability"]["summary"] == "Above the midline favors PHX"
    assert body["win_probability"]["timeline"][-1]["home_win_pct"] == 54
    assert body["win_probability"]["team_stats"][0]["label"] == "Field goal %"
```

- [ ] **Step 2: Run the route test to verify it fails**

Run: `PYTHONPATH=backend pytest backend/tests/test_wnba_game_detail_route.py::test_game_detail_route_includes_win_probability -v`

Expected: FAIL because the route payload does not yet include the new field or fixture wiring is incomplete.

- [ ] **Step 3: Add a null-shape route test**

```python
def test_game_detail_route_allows_null_win_probability(client, monkeypatch):
    async def fake_fetch(_: str) -> dict:
        return load_fixture("espn_wnba_summary.json")

    monkeypatch.setattr(game_detail_service, "fetch_espn_summary", fake_fetch)
    game_detail_service.clear_game_detail_cache()

    response = client.get("/api/wnba/games/401857098")

    assert response.status_code == 200
    assert response.json()["win_probability"] is None
```

- [ ] **Step 4: Run the targeted route tests to verify they pass**

Run: `PYTHONPATH=backend pytest backend/tests/test_wnba_game_detail_route.py::test_game_detail_route_includes_win_probability backend/tests/test_wnba_game_detail_route.py::test_game_detail_route_allows_null_win_probability -v`

Expected: PASS

- [ ] **Step 5: Run the full backend game-detail route suite**

Run: `PYTHONPATH=backend pytest backend/tests/test_wnba_game_detail_route.py -v`

Expected: PASS with existing cache/error-route expectations preserved.

### Task 3: Frontend API and mapping types

**Files:**
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/components/game/types.ts`
- Modify: `frontend/src/components/game/mapGameDetail.ts`
- Modify: `frontend/src/components/game/mapGameDetail.test.ts`

**Interfaces:**
- Consumes: backend JSON field `win_probability`
- Produces: `GameDetail["winProbability"]` with camelCase `timeline` and `teamStats`

- [ ] **Step 1: Write the failing mapper test**

```typescript
it("maps win probability into camelCase detail data", () => {
  const mapped = mapGameDetail(
    buildApiDetail({
      win_probability: {
        summary: "Above the midline favors PHX",
        timeline: [
          {
            id: "p-1",
            period: 1,
            clock: "4:29",
            away_score: 10,
            home_score: 8,
            away_win_pct: 46,
            home_win_pct: 54,
            team_id: "129153",
          },
        ],
        team_stats: [
          {
            key: "field_goal_pct",
            label: "Field goal %",
            away_value: 41,
            home_value: 49,
          },
        ],
      },
    }),
  );

  expect(mapped.winProbability?.summary).toBe("Above the midline favors PHX");
  expect(mapped.winProbability?.timeline[0]).toEqual({
    id: "p-1",
    period: 1,
    clock: "4:29",
    awayScore: 10,
    homeScore: 8,
    awayWinPct: 46,
    homeWinPct: 54,
    teamId: "129153",
  });
  expect(mapped.winProbability?.teamStats[0].label).toBe("Field goal %");
});
```

- [ ] **Step 2: Run the mapper test to verify it fails**

Run: `cd frontend && npm run test -- src/components/game/mapGameDetail.test.ts`

Expected: FAIL because the API and UI types do not include `win_probability` / `winProbability`.

- [ ] **Step 3: Add the API and UI types**

```typescript
export type ApiGameDetailWinProbabilityPoint = {
  id: string;
  period: number;
  clock: string;
  away_score: number;
  home_score: number;
  away_win_pct: number;
  home_win_pct: number;
  team_id: string | null;
};

export type ApiGameDetailTeamStat = {
  key: string;
  label: string;
  away_value: number;
  home_value: number;
};

export type ApiGameDetailWinProbability = {
  summary: string | null;
  timeline: ApiGameDetailWinProbabilityPoint[];
  team_stats: ApiGameDetailTeamStat[];
};

export type ApiWnbaGameDetail = {
  // existing fields...
  win_probability: ApiGameDetailWinProbability | null;
  fetched_at: string;
};
```

- [ ] **Step 4: Implement the mapper changes**

```typescript
export type GameDetailWinProbabilityPoint = {
  id: string;
  period: number;
  clock: string;
  awayScore: number;
  homeScore: number;
  awayWinPct: number;
  homeWinPct: number;
  teamId: string | null;
};

export type GameDetailTeamStat = {
  key: string;
  label: string;
  awayValue: number;
  homeValue: number;
};

export type GameDetailWinProbability = {
  summary: string | null;
  timeline: GameDetailWinProbabilityPoint[];
  teamStats: GameDetailTeamStat[];
};

winProbability: detail.win_probability
  ? {
      summary: detail.win_probability.summary,
      timeline: detail.win_probability.timeline.map((point) => ({
        id: point.id,
        period: point.period,
        clock: point.clock,
        awayScore: point.away_score,
        homeScore: point.home_score,
        awayWinPct: point.away_win_pct,
        homeWinPct: point.home_win_pct,
        teamId: point.team_id,
      })),
      teamStats: detail.win_probability.team_stats.map((stat) => ({
        key: stat.key,
        label: stat.label,
        awayValue: stat.away_value,
        homeValue: stat.home_value,
      })),
    }
  : null,
```

- [ ] **Step 5: Run the mapper test to verify it passes**

Run: `cd frontend && npm run test -- src/components/game/mapGameDetail.test.ts`

Expected: PASS

- [ ] **Step 6: Run related frontend mapper/component tests**

Run: `cd frontend && npm run test -- src/components/game/mapGameDetail.test.ts src/components/game/GameHeader.test.tsx src/components/game/ShotChart.test.tsx src/components/game/PlayByPlay.test.tsx`

Expected: PASS, confirming the new types did not break existing component consumers.

### Task 4: Win probability panel UI

**Files:**
- Create: `frontend/src/components/game/WinProbabilityPanel.tsx`
- Create: `frontend/src/components/game/WinProbabilityPanel.test.tsx`
- Modify: `frontend/src/components/game/testFixtures.ts`

**Interfaces:**
- Consumes: `detail.away`, `detail.home`, and `detail.winProbability`
- Produces: `WinProbabilityPanel({ detail }: { detail: GameDetail })`

- [ ] **Step 1: Write the failing component test for default render**

```typescript
it("renders the latest win probability point and team stats by default", () => {
  render(<WinProbabilityPanel detail={buildGameDetailFixture()} />);

  expect(screen.getByText("Win probability")).toBeInTheDocument();
  expect(screen.getByText("Above the midline favors PHX")).toBeInTheDocument();
  expect(screen.getByText("10-8")).toBeInTheDocument();
  expect(screen.getByText("PHX 54%")).toBeInTheDocument();
  expect(screen.getByText("GS 46%")).toBeInTheDocument();
  expect(screen.getByText("Field goal %")).toBeInTheDocument();
  expect(screen.getByText("49")).toBeInTheDocument();
  expect(screen.getByText("41")).toBeInTheDocument();
});
```

- [ ] **Step 2: Write the failing interaction test**

```typescript
it("updates the active score snapshot when hovering a timeline point", async () => {
  render(<WinProbabilityPanel detail={buildGameDetailFixture()} />);

  const point = screen.getByRole("button", { name: /q1 4:29/i });
  await userEvent.hover(point);

  expect(screen.getByText("10-8")).toBeInTheDocument();
  expect(screen.getByText("PHX 54%")).toBeInTheDocument();
  expect(screen.getByText("GS 46%")).toBeInTheDocument();
});
```

- [ ] **Step 3: Write the failing fallback test**

```typescript
it("shows an unavailable message when win probability data is missing", () => {
  render(
    <WinProbabilityPanel
      detail={buildGameDetailFixture({ winProbability: null })}
    />,
  );

  expect(
    screen.getByText("Win probability unavailable for this game yet."),
  ).toBeInTheDocument();
});
```

- [ ] **Step 4: Run the new component test file to verify it fails**

Run: `cd frontend && npm run test -- src/components/game/WinProbabilityPanel.test.tsx`

Expected: FAIL because the component and supporting fixture data do not exist yet.

- [ ] **Step 5: Add fixture data and implement the minimal component**

```typescript
export function WinProbabilityPanel({ detail }: { detail: GameDetail }) {
  const data = detail.winProbability;
  const latestPoint = data?.timeline[data.timeline.length - 1] ?? null;
  const [activePoint, setActivePoint] = useState(latestPoint);

  useEffect(() => {
    setActivePoint(data?.timeline[data.timeline.length - 1] ?? null);
  }, [data]);

  if (!data) {
    return (
      <section className="rounded-xl border border-white/10 bg-[#141414] p-4">
        <h2 className="text-sm font-semibold text-white">Win probability</h2>
        <p className="mt-2 text-sm text-white/50">
          Win probability unavailable for this game yet.
        </p>
      </section>
    );
  }

  return (
    <section className="rounded-xl border border-white/10 bg-[#141414] p-4">
      <h2 className="text-sm font-semibold text-white">Win probability</h2>
      {data.summary ? (
        <p className="mt-1 text-xs text-white/50">{data.summary}</p>
      ) : null}
      <div className="mt-4">
        <div className="flex items-baseline gap-2 text-sm text-white">
          <span>{activePoint ? `${activePoint.awayScore}-${activePoint.homeScore}` : "—"}</span>
          {activePoint ? <span>{detail.home.abbrev} {activePoint.homeWinPct}%</span> : null}
          {activePoint ? <span>{detail.away.abbrev} {activePoint.awayWinPct}%</span> : null}
        </div>
        <div className="mt-4 flex gap-2">
          {data.timeline.map((point) => (
            <button
              key={point.id}
              type="button"
              aria-label={`Q${point.period} ${point.clock}`}
              onMouseEnter={() => setActivePoint(point)}
              onFocus={() => setActivePoint(point)}
              className="size-3 rounded-full"
              style={{ backgroundColor: point.homeWinPct >= 50 ? detail.home.color : detail.away.color }}
            />
          ))}
        </div>
      </div>
      {data.teamStats.length > 0 ? (
        <div className="mt-6 space-y-3">
          {data.teamStats.map((stat) => (
            <div key={stat.key} className="grid grid-cols-[40px_1fr_40px] items-center gap-3 text-sm">
              <span className="text-white/80">{stat.awayValue}</span>
              <span className="text-center text-white/50">{stat.label}</span>
              <span className="text-right text-white/80">{stat.homeValue}</span>
            </div>
          ))}
        </div>
      ) : null}
    </section>
  );
}
```

- [ ] **Step 6: Replace the placeholder dots with the final chart interaction**

```typescript
const points = data.timeline;
const path = buildWinProbabilityPath(points, chartWidth, chartHeight);

<svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} className="mt-4 w-full">
  <line
    x1={0}
    x2={chartWidth}
    y1={chartHeight / 2}
    y2={chartHeight / 2}
    stroke="rgba(255,255,255,0.15)"
    strokeDasharray="4 4"
  />
  <path d={path.area} fill={`${detail.home.color}22`} />
  <path d={path.line} fill="none" stroke={detail.home.color} strokeWidth={2} />
  {points.map((point, index) => (
    <circle
      key={point.id}
      cx={xForIndex(index, points.length, chartWidth)}
      cy={yForPct(point.homeWinPct, chartHeight)}
      r={index === activeIndex ? 5 : 3}
      fill={detail.home.color}
      tabIndex={0}
      role="button"
      aria-label={`Q${point.period} ${point.clock}`}
      onMouseEnter={() => setActiveIndex(index)}
      onFocus={() => setActiveIndex(index)}
    />
  ))}
</svg>
```

- [ ] **Step 7: Run the component tests to verify they pass**

Run: `cd frontend && npm run test -- src/components/game/WinProbabilityPanel.test.tsx`

Expected: PASS

- [ ] **Step 8: Run the full game-component test set**

Run: `cd frontend && npm run test -- src/components/game/*.test.tsx src/components/game/*.test.ts`

Expected: PASS with the new panel coexisting cleanly with the existing game-detail components.

### Task 5: Page integration and end-to-end page behavior

**Files:**
- Modify: `frontend/src/pages/GameDetailPage.tsx`
- Modify: `frontend/src/AppRouter.test.tsx` (only if route fixture assertions need updating)
- Test: `frontend/src/pages/GameDetailPage.tsx` via existing routed tests or a new focused component test

**Interfaces:**
- Consumes: `WinProbabilityPanel`
- Produces: final page layout order `GameHeader -> grid -> WinProbabilityPanel`

- [ ] **Step 1: Write the failing integration test**

```typescript
it("renders win probability beneath shot chart and play-by-play", async () => {
  renderAppAt("/games/401857098");

  expect(await screen.findByText("Shot chart")).toBeInTheDocument();
  expect(await screen.findByText("Play-by-play")).toBeInTheDocument();
  expect(await screen.findByText("Win probability")).toBeInTheDocument();
  expect(screen.getByText("Field goal %")).toBeInTheDocument();
});
```

- [ ] **Step 2: Run the integration test to verify it fails**

Run: `cd frontend && npm run test -- src/AppRouter.test.tsx`

Expected: FAIL because the routed page does not yet render the new panel.

- [ ] **Step 3: Wire the panel into the page**

```typescript
import { WinProbabilityPanel } from "@/components/game/WinProbabilityPanel";

export function GameDetailPage() {
  // existing loading / error guards...
  const detail = mapGameDetail(data);

  return (
    <div className="mx-auto max-w-6xl space-y-4 px-4 py-6 sm:px-6">
      <GameHeader detail={detail} />
      <div className="grid gap-4 lg:grid-cols-2">
        <ShotChart detail={detail} />
        <PlayByPlay detail={detail} />
      </div>
      <WinProbabilityPanel detail={detail} />
    </div>
  );
}
```

- [ ] **Step 4: Run the routed integration test to verify it passes**

Run: `cd frontend && npm run test -- src/AppRouter.test.tsx`

Expected: PASS

- [ ] **Step 5: Run focused verification for the whole feature**

Run: `cd frontend && npm run test -- src/AppRouter.test.tsx src/components/game/*.test.tsx src/components/game/*.test.ts`

Expected: PASS

- [ ] **Step 6: Run backend and frontend verification together**

Run: `PYTHONPATH=backend pytest backend/tests/test_wnba_game_detail_normalize.py backend/tests/test_wnba_game_detail_route.py -v && cd frontend && npm run test -- src/AppRouter.test.tsx src/components/game/*.test.tsx src/components/game/*.test.ts`

Expected: PASS across both stacks with no new failures.
