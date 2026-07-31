# WNBA Game Header Logos Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show ESPN team logos next to team names on the WNBA game detail header by passing through `logos[].href` from the ESPN summary payload.

**Architecture:** Backend picks a logo URL from each competitor’s `team.logos` array (prefer `dark`, else `default`, else first `href`) and exposes `logo_url` on `GameDetailTeam`. Frontend maps it to `logoUrl` and renders it only in `GameHeader`, with `onError` hide fallback. No CDN URL invention; no matchup/ticker changes.

**Tech Stack:** FastAPI/Pydantic, pytest, React, TypeScript, Vitest, Testing Library, Tailwind

## Global Constraints

- Game detail header only — no matchup cards, live ticker, or other surfaces.
- Prefer ESPN `logos[]` entry whose `rel` includes `dark`; else `default`; else first non-empty `href`.
- Missing logos → `logo_url` / `logoUrl` is `null` (do not invent CDN URLs).
- Keep existing `color` behavior unchanged; do not add `alternateColor`.
- Decorative logo: `alt=""`; hide image on `onError`.
- Spec: `docs/superpowers/specs/2026-07-30-wnba-game-header-logos-design.md`.

---

## File Structure

- Modify `backend/app/schemas/wnba_game_detail.py` — add `logo_url: str | None` on `GameDetailTeam`.
- Modify `backend/app/services/wnba_game_detail.py` — add `_team_logo_url` helper; set `logo_url` in team normalization.
- Modify `backend/tests/fixtures/espn_wnba_summary.json` — add realistic `logos` arrays for home/away teams.
- Modify `backend/tests/test_wnba_game_detail_normalize.py` — assert dark preference and null-when-missing.
- Modify `frontend/src/lib/api.ts` — add `logo_url` on `ApiGameDetailTeam`.
- Modify `frontend/src/components/game/types.ts` — add `logoUrl` on `GameDetailTeam`.
- Modify `frontend/src/components/game/mapGameDetail.ts` — map `logo_url` → `logoUrl`.
- Modify `frontend/src/components/game/mapGameDetail.test.ts` — expect `logoUrl`.
- Modify `frontend/src/components/game/testFixtures.ts` — include `logoUrl` on teams.
- Modify `frontend/src/components/game/GameHeader.tsx` — render logo beside team name.
- Modify `frontend/src/components/game/GameHeader.test.tsx` — img present / absent cases.
- Modify other frontend fixtures that construct API/UI teams (`AppRouter.test.tsx`, `GameDetailPage.test.tsx`, `useGameDetail.test.tsx`, `PlayByPlay.test.tsx`) so TypeScript stays green (`logo_url: null` / `logoUrl: null`).

### Task 1: Backend `logo_url` from ESPN `logos[]`

**Files:**
- Modify: `backend/app/schemas/wnba_game_detail.py`
- Modify: `backend/app/services/wnba_game_detail.py`
- Modify: `backend/tests/fixtures/espn_wnba_summary.json`
- Modify: `backend/tests/test_wnba_game_detail_normalize.py`

**Interfaces:**
- Consumes: ESPN competitor `team.logos` list of `{ "href": str, "rel": list[str] | str }`
- Produces: `GameDetailTeam.logo_url: str | None`; helper `_team_logo_url(logos: object) -> str | None`

- [ ] **Step 1: Write the failing normalize tests**

Add to `backend/tests/test_wnba_game_detail_normalize.py`:

```python
def test_normalize_prefers_dark_logo_url():
    payload = load_fixture("espn_wnba_summary.json")
    detail = normalize_espn_summary(
        payload,
        espn_event_id="401857098",
        fetched_at="2026-07-29T19:00:00-04:00",
    )
    assert detail.away.logo_url == (
        "https://a.espncdn.com/i/teamlogos/wnba/500-dark/gs.png"
    )
    assert detail.home.logo_url == (
        "https://a.espncdn.com/i/teamlogos/wnba/500-dark/phx.png"
    )


def test_normalize_logo_url_null_when_logos_missing():
    payload = load_fixture("espn_wnba_summary.json")
    for competitor in payload["header"]["competitions"][0]["competitors"]:
        competitor["team"].pop("logos", None)
    detail = normalize_espn_summary(
        payload,
        espn_event_id="401857098",
        fetched_at="2026-07-29T19:00:00-04:00",
    )
    assert detail.away.logo_url is None
    assert detail.home.logo_url is None
```

Also extend `test_normalize_espn_summary_header_shots_plays` with:

```python
assert detail.away.logo_url is not None
assert detail.home.logo_url is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor && PYTHONPATH=backend python -m pytest backend/tests/test_wnba_game_detail_normalize.py::test_normalize_prefers_dark_logo_url backend/tests/test_wnba_game_detail_normalize.py::test_normalize_logo_url_null_when_logos_missing -v
```

Expected: FAIL (attribute / fixture missing `logos` or `logo_url` not on model).

- [ ] **Step 3: Add `logos` to the primary summary fixture**

In `backend/tests/fixtures/espn_wnba_summary.json`, on each competitor `team` object, add:

Home (`PHX`):

```json
"logos": [
  {
    "href": "https://a.espncdn.com/i/teamlogos/wnba/500/phx.png",
    "rel": ["full", "default"]
  },
  {
    "href": "https://a.espncdn.com/i/teamlogos/wnba/500-dark/phx.png",
    "rel": ["full", "dark"]
  }
]
```

Away (`GS`):

```json
"logos": [
  {
    "href": "https://a.espncdn.com/i/teamlogos/wnba/500/gs.png",
    "rel": ["full", "default"]
  },
  {
    "href": "https://a.espncdn.com/i/teamlogos/wnba/500-dark/gs.png",
    "rel": ["full", "dark"]
  }
]
```

Leave other fixtures without `logos` so null-path coverage remains natural for those tests.

- [ ] **Step 4: Extend schema and implement logo selection**

In `backend/app/schemas/wnba_game_detail.py`:

```python
class GameDetailTeam(BaseModel):
    id: str
    abbrev: str
    name: str
    score: int | None
    color: str
    logo_url: str | None = None
```

In `backend/app/services/wnba_game_detail.py`, add:

```python
def _rel_tokens(rel: object) -> set[str]:
    if isinstance(rel, list):
        return {str(x).lower() for x in rel}
    if isinstance(rel, str) and rel.strip():
        return {rel.strip().lower()}
    return set()


def _team_logo_url(logos: object) -> str | None:
    if not isinstance(logos, list):
        return None
    entries: list[tuple[set[str], str]] = []
    for item in logos:
        if not isinstance(item, dict):
            continue
        href = str(item.get("href") or "").strip()
        if not href:
            continue
        entries.append((_rel_tokens(item.get("rel")), href))
    if not entries:
        return None
    for token in ("dark", "default"):
        for rels, href in entries:
            if token in rels:
                return href
    return entries[0][1]
```

In the existing `team(...)` builder inside `normalize_espn_summary`, pass:

```python
logo_url=_team_logo_url(t.get("logos")),
```

- [ ] **Step 5: Run normalize tests to verify they pass**

Run:

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor && PYTHONPATH=backend python -m pytest backend/tests/test_wnba_game_detail_normalize.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/schemas/wnba_game_detail.py \
  backend/app/services/wnba_game_detail.py \
  backend/tests/fixtures/espn_wnba_summary.json \
  backend/tests/test_wnba_game_detail_normalize.py \
  docs/superpowers/specs/2026-07-30-wnba-game-header-logos-design.md
git commit -m "$(cat <<'EOF'
feat: pass ESPN team logo URLs through WNBA game detail

Prefer dark logo hrefs from summary competitors so the game header can render official marks without inventing CDN paths.
EOF
)"
```

### Task 2: Frontend types + `mapGameDetail` for `logoUrl`

**Files:**
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/components/game/types.ts`
- Modify: `frontend/src/components/game/mapGameDetail.ts`
- Modify: `frontend/src/components/game/mapGameDetail.test.ts`
- Modify: `frontend/src/components/game/testFixtures.ts`
- Modify: `frontend/src/AppRouter.test.tsx`
- Modify: `frontend/src/pages/GameDetailPage.test.tsx`
- Modify: `frontend/src/hooks/useGameDetail.test.tsx`
- Modify: `frontend/src/components/game/PlayByPlay.test.tsx`

**Interfaces:**
- Consumes: `ApiGameDetailTeam.logo_url: string | null`
- Produces: `GameDetailTeam.logoUrl: string | null` via `mapGameDetail`

- [ ] **Step 1: Write the failing mapper expectation**

In `frontend/src/components/game/mapGameDetail.test.ts`, add `logo_url` to `buildApiDetail` teams and expected mapped teams:

```typescript
away: {
  id: "away1",
  abbrev: "GS",
  name: "Golden State Valkyries",
  score: 10,
  color: "#5B2C6F",
  logo_url: "https://a.espncdn.com/i/teamlogos/wnba/500-dark/gs.png",
},
home: {
  id: "home1",
  abbrev: "PHX",
  name: "Phoenix Mercury",
  score: 9,
  color: "#E56020",
  logo_url: "https://a.espncdn.com/i/teamlogos/wnba/500-dark/phx.png",
},
```

Expected mapped:

```typescript
away: {
  id: "away1",
  abbrev: "GS",
  name: "Golden State Valkyries",
  score: 10,
  color: "#5B2C6F",
  logoUrl: "https://a.espncdn.com/i/teamlogos/wnba/500-dark/gs.png",
},
home: {
  id: "home1",
  abbrev: "PHX",
  name: "Phoenix Mercury",
  score: 9,
  color: "#E56020",
  logoUrl: "https://a.espncdn.com/i/teamlogos/wnba/500-dark/phx.png",
},
```

Add a focused case:

```typescript
it("maps null logo_url to null logoUrl", () => {
  const mapped = mapGameDetail(
    buildApiDetail({
      away: {
        id: "away1",
        abbrev: "GS",
        name: "Golden State Valkyries",
        score: 10,
        color: "#5B2C6F",
        logo_url: null,
      },
    }),
  );
  expect(mapped.away.logoUrl).toBeNull();
});
```

- [ ] **Step 2: Run mapper test to verify it fails**

Run:

```bash
npm --prefix "/Users/alexgonzalez/Documents/NBA-Prop-Predictor/frontend" run test -- src/components/game/mapGameDetail.test.ts
```

Expected: FAIL (missing `logoUrl` on mapped object / types).

- [ ] **Step 3: Update API + UI types and mapper**

`frontend/src/lib/api.ts`:

```typescript
export type ApiGameDetailTeam = {
  id: string;
  abbrev: string;
  name: string;
  score: number | null;
  color: string;
  logo_url: string | null;
};
```

`frontend/src/components/game/types.ts`:

```typescript
export type GameDetailTeam = {
  id: string;
  abbrev: string;
  name: string;
  score: number | null;
  color: string;
  logoUrl: string | null;
};
```

In `mapGameDetail.ts` for both `away` and `home`:

```typescript
logoUrl: detail.away.logo_url,
// ...
logoUrl: detail.home.logo_url,
```

- [ ] **Step 4: Update shared fixtures so TypeScript compiles**

Set `logoUrl: null` (UI fixtures) or `logo_url: null` (API mocks) on every constructed team in:

- `frontend/src/components/game/testFixtures.ts`
- `frontend/src/AppRouter.test.tsx`
- `frontend/src/pages/GameDetailPage.test.tsx`
- `frontend/src/hooks/useGameDetail.test.tsx`
- `frontend/src/components/game/PlayByPlay.test.tsx`

Optionally set a real dark URL on `testFixtures.detail.away/home` if Task 3 wants a default happy-path image; otherwise keep `null` and override in header tests.

- [ ] **Step 5: Run mapper tests to verify they pass**

Run:

```bash
npm --prefix "/Users/alexgonzalez/Documents/NBA-Prop-Predictor/frontend" run test -- src/components/game/mapGameDetail.test.ts
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add frontend/src/lib/api.ts \
  frontend/src/components/game/types.ts \
  frontend/src/components/game/mapGameDetail.ts \
  frontend/src/components/game/mapGameDetail.test.ts \
  frontend/src/components/game/testFixtures.ts \
  frontend/src/AppRouter.test.tsx \
  frontend/src/pages/GameDetailPage.test.tsx \
  frontend/src/hooks/useGameDetail.test.tsx \
  frontend/src/components/game/PlayByPlay.test.tsx
git commit -m "$(cat <<'EOF'
feat: map WNBA game detail logo_url to logoUrl

Carry ESPN logo URLs into the UI team model so GameHeader can render them.
EOF
)"
```

### Task 3: Render logos in `GameHeader`

**Files:**
- Modify: `frontend/src/components/game/GameHeader.tsx`
- Modify: `frontend/src/components/game/GameHeader.test.tsx`

**Interfaces:**
- Consumes: `GameDetailTeam.logoUrl: string | null`
- Produces: header team row with optional decorative logo image

- [ ] **Step 1: Write the failing header tests**

Add to `frontend/src/components/game/GameHeader.test.tsx`:

```typescript
it("renders team logos when logoUrl is set", () => {
  renderHeader({
    away: {
      ...detail.away,
      logoUrl: "https://a.espncdn.com/i/teamlogos/wnba/500-dark/gs.png",
    },
    home: {
      ...detail.home,
      logoUrl: "https://a.espncdn.com/i/teamlogos/wnba/500-dark/phx.png",
    },
  });
  const images = screen.getAllByRole("presentation");
  expect(images).toHaveLength(2);
  expect(images[0]).toHaveAttribute(
    "src",
    "https://a.espncdn.com/i/teamlogos/wnba/500-dark/gs.png",
  );
  expect(images[1]).toHaveAttribute(
    "src",
    "https://a.espncdn.com/i/teamlogos/wnba/500-dark/phx.png",
  );
});

it("omits logo images when logoUrl is null", () => {
  renderHeader({
    away: { ...detail.away, logoUrl: null },
    home: { ...detail.home, logoUrl: null },
  });
  expect(screen.queryByRole("presentation")).not.toBeInTheDocument();
});
```

If `role="presentation"` is awkward with `alt=""`, use:

```typescript
container.querySelectorAll("img")
```

via `const { container } = renderHeader(...)` instead — prefer whatever matches the implementation’s accessibility attributes. Keep assertions on `src` either way.

- [ ] **Step 2: Run header tests to verify they fail**

Run:

```bash
npm --prefix "/Users/alexgonzalez/Documents/NBA-Prop-Predictor/frontend" run test -- src/components/game/GameHeader.test.tsx
```

Expected: FAIL (no `<img>` yet).

- [ ] **Step 3: Implement logo rendering in `GameHeader`**

```tsx
import { useState } from "react";
import { Link } from "react-router-dom";
import type { GameDetail, GameDetailTeam } from "./types";

function TeamLogo({ url }: { url: string }) {
  const [failed, setFailed] = useState(false);
  if (failed) return null;
  return (
    <img
      src={url}
      alt=""
      role="presentation"
      className="size-8 shrink-0 object-contain"
      onError={() => setFailed(true)}
    />
  );
}

function TeamRow({ team }: { team: GameDetailTeam }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="flex min-w-0 items-center gap-2.5">
        {team.logoUrl ? <TeamLogo url={team.logoUrl} /> : null}
        <span
          className="truncate text-base font-semibold"
          style={{ color: team.color }}
        >
          {team.name}
        </span>
      </span>
      <ScoreBox score={team.score} />
    </div>
  );
}
```

Keep the rest of `GameHeader` unchanged.

- [ ] **Step 4: Run header + related game tests**

Run:

```bash
npm --prefix "/Users/alexgonzalez/Documents/NBA-Prop-Predictor/frontend" run test -- src/components/game/GameHeader.test.tsx src/components/game/mapGameDetail.test.ts src/pages/GameDetailPage.test.tsx
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/game/GameHeader.tsx \
  frontend/src/components/game/GameHeader.test.tsx
git commit -m "$(cat <<'EOF'
feat: show ESPN team logos on WNBA game header

Render dark-preferring logo URLs beside colored team names with a graceful image error fallback.
EOF
)"
```

---

## Spec coverage checklist

| Spec requirement | Task |
| --- | --- |
| Pass through ESPN `logos[].href` | Task 1 |
| Prefer `dark` > `default` > first | Task 1 `_team_logo_url` |
| `logo_url` null when missing | Task 1 |
| Frontend `logoUrl` mapping | Task 2 |
| `GameHeader` only | Task 3 |
| `alt=""` + `onError` hide | Task 3 |
| Fixture with dark+default logos | Task 1 |
| No matchup/ticker / no alternateColor / no CDN invent | Honored by scope (no tasks) |
