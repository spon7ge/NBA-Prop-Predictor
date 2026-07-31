# WNBA Scoreboard Logos (Live Now + Matchups) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show ESPN team logos in Live Now and Matchup hub letter-circle slots, with letter fallback when missing or on image error.

**Architecture:** Extend `WnbaTeam` with `logo_url` from ESPN scoreboard `team.logo`. Map to `logoUrl` for live and matchup UI models. Shared `TeamAbbrevAvatar` replaces letter circles in `LiveNowSection` and `MatchupGameCard`. Ticker and game-header logo selection stay unchanged.

**Tech Stack:** FastAPI/Pydantic, pytest, React, TypeScript, Vitest, Testing Library, Tailwind

## Global Constraints

- Surfaces: Live Now cards + Matchup hub cards only.
- Source: ESPN scoreboard `competitors[].team.logo` string — pass through non-empty trimmed URL.
- Missing logos → `logo_url` / `logoUrl` is `null`; letter avatar fallback (`abbrev.slice(0, 1)`).
- Stats-only rows → `logo_url` is `null`.
- No CDN URL invention from abbrev/id; no `/500/` → `/500-dark/` rewrite.
- Live ticker unchanged; game-detail summary `logos[]` selection unchanged.
- Spec: `docs/superpowers/specs/2026-07-30-wnba-scoreboard-logos-design.md`.

---

## File Structure

- Modify `backend/app/schemas/wnba_scoreboard.py` — add `logo_url` on `WnbaTeam`.
- Modify `backend/app/services/wnba_scoreboard.py` — set `logo_url` in ESPN/stats normalize + merge.
- Modify `backend/tests/fixtures/espn_wnba_scoreboard.json` — add `logo` on teams.
- Modify `backend/tests/test_wnba_scoreboard_normalize.py` — logo normalize + merge tests.
- Modify `frontend/src/lib/api.ts` — `logo_url` on `ApiWnbaTeam`.
- Modify `frontend/src/components/home/types.ts` — `logoUrl` on `LiveGameTeam`.
- Modify `frontend/src/components/league/types.ts` — `logoUrl` on `MatchupTeam`.
- Modify `frontend/src/components/home/mapScoreboard.ts` — map `logoUrl`.
- Modify `frontend/src/components/home/mapScoreboard.test.ts` — expectations + fixtures.
- Create `frontend/src/components/TeamAbbrevAvatar.tsx` — logo or letter circle.
- Create `frontend/src/components/TeamAbbrevAvatar.test.tsx` — logo / letter / error cases.
- Modify `frontend/src/components/home/LiveNowSection.tsx` — use avatar.
- Modify `frontend/src/components/home/LiveNowSection.test.tsx` — logo cases + `logoUrl` on fixtures.
- Modify `frontend/src/components/league/MatchupGameCard.tsx` — use avatar.
- Modify `frontend/src/components/league/MatchupGameCard.test.tsx` — logo cases + fixtures.

### Task 1: Backend scoreboard `logo_url`

**Files:**
- Modify: `backend/app/schemas/wnba_scoreboard.py`
- Modify: `backend/app/services/wnba_scoreboard.py`
- Modify: `backend/tests/fixtures/espn_wnba_scoreboard.json`
- Modify: `backend/tests/test_wnba_scoreboard_normalize.py`

**Interfaces:**
- Consumes: ESPN competitor `team.logo: str | missing`
- Produces: `WnbaTeam.logo_url: str | None`

- [ ] **Step 1: Write the failing normalize + merge tests**

Add to `backend/tests/test_wnba_scoreboard_normalize.py`:

```python
def test_normalize_espn_sets_logo_url_from_team_logo():
    payload = json.loads((FIXTURES / "espn_wnba_scoreboard.json").read_text())
    g = normalize_espn_scoreboard(payload, date_et="2026-07-29")[0]
    assert g.away.logo_url == (
        "https://a.espncdn.com/i/teamlogos/wnba/500/atl.png"
    )
    assert g.home.logo_url == (
        "https://a.espncdn.com/i/teamlogos/wnba/500/dal.png"
    )


def test_normalize_stats_logo_url_is_null():
    payload = json.loads((FIXTURES / "stats_wnba_scoreboard.json").read_text())
    g = normalize_stats_scoreboard(payload, date_et="2026-07-29")[0]
    assert g.away.logo_url is None
    assert g.home.logo_url is None


def test_merge_keeps_espn_logo_url_over_stats_null():
    espn = normalize_espn_scoreboard(
        json.loads((FIXTURES / "espn_wnba_scoreboard.json").read_text()),
        date_et="2026-07-29",
    )
    stats = normalize_stats_scoreboard(
        json.loads((FIXTURES / "stats_wnba_scoreboard.json").read_text()),
        date_et="2026-07-29",
    )
    merged = merge_games(espn, stats)[0]
    assert merged.away.logo_url == (
        "https://a.espncdn.com/i/teamlogos/wnba/500/atl.png"
    )
    assert merged.home.logo_url == (
        "https://a.espncdn.com/i/teamlogos/wnba/500/dal.png"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor && PYTHONPATH=backend python3 -m pytest backend/tests/test_wnba_scoreboard_normalize.py::test_normalize_espn_sets_logo_url_from_team_logo backend/tests/test_wnba_scoreboard_normalize.py::test_normalize_stats_logo_url_is_null backend/tests/test_wnba_scoreboard_normalize.py::test_merge_keeps_espn_logo_url_over_stats_null -v
```

Expected: FAIL (missing field / assertion).

- [ ] **Step 3: Add `logo` to the ESPN scoreboard fixture**

In `backend/tests/fixtures/espn_wnba_scoreboard.json`, on each competitor `team`:

Away (`ATL`):

```json
"logo": "https://a.espncdn.com/i/teamlogos/wnba/500/atl.png"
```

Home (`DAL`):

```json
"logo": "https://a.espncdn.com/i/teamlogos/wnba/500/dal.png"
```

- [ ] **Step 4: Schema + normalize + merge**

`backend/app/schemas/wnba_scoreboard.py`:

```python
class WnbaTeam(BaseModel):
    abbrev: str
    name: str
    score: int | None = None
    record: str | None = None
    logo_url: str | None = None
```

In `wnba_scoreboard.py`, helper:

```python
def _team_logo_from_espn(team: dict) -> str | None:
    logo = str(team.get("logo") or "").strip()
    return logo or None
```

In `normalize_espn_scoreboard` `team(...)`:

```python
return WnbaTeam(
    abbrev=str(t.get("abbreviation") or ""),
    name=str(t.get("displayName") or ""),
    score=score if status != "scheduled" else None,
    record=_espn_team_record(c),
    logo_url=_team_logo_from_espn(t),
)
```

In `normalize_stats_scoreboard` `team(...)`, omit `logo_url` (defaults `None`) or set `logo_url=None` explicitly.

In `merge_games` team rebuilds:

```python
away=WnbaTeam(
    abbrev=str(prefer_complete(a.away.abbrev, g.away.abbrev)),
    name=str(prefer_complete(a.away.name, g.away.name)),
    score=prefer_complete(a.away.score, g.away.score),
    record=prefer_complete(a.away.record, g.away.record) or None,
    logo_url=prefer_complete(a.away.logo_url, g.away.logo_url) or None,
),
home=WnbaTeam(
    abbrev=str(prefer_complete(a.home.abbrev, g.home.abbrev)),
    name=str(prefer_complete(a.home.name, g.home.name)),
    score=prefer_complete(a.home.score, g.home.score),
    record=prefer_complete(a.home.record, g.home.record) or None,
    logo_url=prefer_complete(a.home.logo_url, g.home.logo_url) or None,
),
```

(`a` is ESPN when matched; empty stats logo yields ESPN URL via `prefer_complete`.)

- [ ] **Step 5: Run scoreboard normalize tests**

Run:

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor && PYTHONPATH=backend python3 -m pytest backend/tests/test_wnba_scoreboard_normalize.py -q
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/schemas/wnba_scoreboard.py \
  backend/app/services/wnba_scoreboard.py \
  backend/tests/fixtures/espn_wnba_scoreboard.json \
  backend/tests/test_wnba_scoreboard_normalize.py \
  docs/superpowers/specs/2026-07-30-wnba-scoreboard-logos-design.md
git commit -m "$(cat <<'EOF'
feat: pass ESPN scoreboard team logos through WnbaTeam

Expose logo_url from team.logo so Live Now and matchup cards can render official marks with letter fallback.
EOF
)"
```

### Task 2: Frontend types + scoreboard mappers

**Files:**
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/components/home/types.ts`
- Modify: `frontend/src/components/league/types.ts`
- Modify: `frontend/src/components/home/mapScoreboard.ts`
- Modify: `frontend/src/components/home/mapScoreboard.test.ts`

**Interfaces:**
- Consumes: `ApiWnbaTeam.logo_url: string | null`
- Produces: `LiveGameTeam.logoUrl` / `MatchupTeam.logoUrl`

- [ ] **Step 1: Write failing mapper expectations**

Update `apiGame` helper teams to include `logo_url: null` by default.

Add:

```typescript
it("maps logo_url to logoUrl for live and matchup games", () => {
  const game = apiGame({
    away: {
      abbrev: "ATL",
      name: "Atlanta Dream",
      score: 36,
      logo_url: "https://a.espncdn.com/i/teamlogos/wnba/500/atl.png",
    },
    home: {
      abbrev: "DAL",
      name: "Dallas Wings",
      score: 44,
      logo_url: "https://a.espncdn.com/i/teamlogos/wnba/500/dal.png",
    },
  });
  const live = mapToLiveGames([game])[0];
  expect(live.away.logoUrl).toBe(
    "https://a.espncdn.com/i/teamlogos/wnba/500/atl.png",
  );
  expect(live.home.logoUrl).toBe(
    "https://a.espncdn.com/i/teamlogos/wnba/500/dal.png",
  );
  const matchup = mapToMatchupGames([game])[0];
  expect(matchup.away.logoUrl).toBe(
    "https://a.espncdn.com/i/teamlogos/wnba/500/atl.png",
  );
  expect(matchup.home.logoUrl).toBe(
    "https://a.espncdn.com/i/teamlogos/wnba/500/dal.png",
  );
});
```

Also add:

```typescript
it("maps null logo_url to null logoUrl", () => {
  const live = mapToLiveGames([apiGame()])[0];
  expect(live.away.logoUrl).toBeNull();
  expect(live.home.logoUrl).toBeNull();
});
```

- [ ] **Step 2: Run mapper tests to verify fail**

Run:

```bash
npm --prefix "/Users/alexgonzalez/Documents/NBA-Prop-Predictor/frontend" run test -- src/components/home/mapScoreboard.test.ts
```

Expected: FAIL (missing `logoUrl`).

- [ ] **Step 3: Update types and mappers**

`ApiWnbaTeam`:

```typescript
export type ApiWnbaTeam = {
  abbrev: string;
  name: string;
  score: number | null;
  record?: string | null;
  logo_url: string | null;
};
```

`LiveGameTeam` / `MatchupTeam`: add `logoUrl: string | null`.

In `mapToLiveGames` / `mapToMatchupGames` team objects:

```typescript
logoUrl: g.away.logo_url,
// ...
logoUrl: g.home.logo_url,
```

- [ ] **Step 4: Fix any TypeScript breaks in tests that construct `ApiWnbaTeam` / live / matchup teams**

Add `logo_url: null` or `logoUrl: null` wherever required. Do not stage unrelated WIP files beyond what Task 2 lists unless a listed test file needs the field.

- [ ] **Step 5: Run mapper tests**

Run:

```bash
npm --prefix "/Users/alexgonzalez/Documents/NBA-Prop-Predictor/frontend" run test -- src/components/home/mapScoreboard.test.ts
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add frontend/src/lib/api.ts \
  frontend/src/components/home/types.ts \
  frontend/src/components/league/types.ts \
  frontend/src/components/home/mapScoreboard.ts \
  frontend/src/components/home/mapScoreboard.test.ts
git commit -m "$(cat <<'EOF'
feat: map scoreboard logo_url into live and matchup UI types

Carry ESPN team logos into Live Now and Matchups mappers for avatar rendering.
EOF
)"
```

If other test fixtures must change for compile (e.g. LiveNow fixtures), include only those necessary compile fixes in this commit — prefer deferring UI fixture updates to Task 3 when possible. If TypeScript requires `logoUrl` on `LiveGame` fixtures immediately, update them with `logoUrl: null` here and list those files in the commit.

### Task 3: Shared avatar + Live Now + Matchup cards

**Files:**
- Create: `frontend/src/components/TeamAbbrevAvatar.tsx`
- Create: `frontend/src/components/TeamAbbrevAvatar.test.tsx`
- Modify: `frontend/src/components/home/LiveNowSection.tsx`
- Modify: `frontend/src/components/home/LiveNowSection.test.tsx`
- Modify: `frontend/src/components/league/MatchupGameCard.tsx`
- Modify: `frontend/src/components/league/MatchupGameCard.test.tsx`

**Interfaces:**
- Consumes: `{ abbrev: string; logoUrl: string | null; sizeClassName?: string }`
- Produces: logo `<img>` or letter circle

- [ ] **Step 1: Write failing `TeamAbbrevAvatar` tests**

```tsx
import { describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { TeamAbbrevAvatar } from "./TeamAbbrevAvatar";

describe("TeamAbbrevAvatar", () => {
  it("renders a logo image when logoUrl is set", () => {
    const { container } = render(
      <TeamAbbrevAvatar
        abbrev="ATL"
        logoUrl="https://a.espncdn.com/i/teamlogos/wnba/500/atl.png"
      />,
    );
    const img = container.querySelector("img");
    expect(img).toHaveAttribute(
      "src",
      "https://a.espncdn.com/i/teamlogos/wnba/500/atl.png",
    );
    expect(img).toHaveAttribute("alt", "");
  });

  it("renders the abbrev letter when logoUrl is null", () => {
    render(<TeamAbbrevAvatar abbrev="ATL" logoUrl={null} />);
    expect(screen.getByText("A")).toBeInTheDocument();
    expect(screen.queryByRole("presentation")).not.toBeInTheDocument();
  });

  it("falls back to the letter after image error", () => {
    const { container } = render(
      <TeamAbbrevAvatar
        abbrev="ATL"
        logoUrl="https://a.espncdn.com/i/teamlogos/wnba/500/atl.png"
      />,
    );
    fireEvent.error(container.querySelector("img")!);
    expect(screen.getByText("A")).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run avatar tests to verify fail**

Run:

```bash
npm --prefix "/Users/alexgonzalez/Documents/NBA-Prop-Predictor/frontend" run test -- src/components/TeamAbbrevAvatar.test.tsx
```

Expected: FAIL (module missing).

- [ ] **Step 3: Implement `TeamAbbrevAvatar`**

```tsx
import { useState } from "react";

type TeamAbbrevAvatarProps = {
  abbrev: string;
  logoUrl: string | null;
  /** Tailwind size classes for the circle/img, e.g. `size-7` or `size-8`. */
  sizeClassName?: string;
};

export function TeamAbbrevAvatar({
  abbrev,
  logoUrl,
  sizeClassName = "size-7",
}: TeamAbbrevAvatarProps) {
  const [failed, setFailed] = useState(false);
  const letter = abbrev.slice(0, 1);
  const showLogo = Boolean(logoUrl) && !failed;

  if (showLogo && logoUrl) {
    return (
      <img
        src={logoUrl}
        alt=""
        role="presentation"
        className={`${sizeClassName} shrink-0 object-contain`}
        onError={() => setFailed(true)}
      />
    );
  }

  return (
    <span
      className={`flex ${sizeClassName} shrink-0 items-center justify-center rounded-full bg-white/10 text-[10px] font-bold text-white/70`}
    >
      {letter}
    </span>
  );
}
```

- [ ] **Step 4: Run avatar tests to verify pass**

Run:

```bash
npm --prefix "/Users/alexgonzalez/Documents/NBA-Prop-Predictor/frontend" run test -- src/components/TeamAbbrevAvatar.test.tsx
```

Expected: PASS

- [ ] **Step 5: Wire Live Now + Matchups + tests**

In `LiveNowSection.tsx`, replace the letter `<span>` with:

```tsx
<TeamAbbrevAvatar
  abbrev={team.abbrev}
  logoUrl={team.logoUrl}
  sizeClassName="size-7"
/>
```

In `MatchupGameCard.tsx`:

```tsx
<TeamAbbrevAvatar
  abbrev={team.abbrev}
  logoUrl={team.logoUrl}
  sizeClassName="size-8"
/>
```

Update fixtures with `logoUrl: null` by default.

Add Live Now test:

```tsx
it("renders team logos when logoUrl is set", () => {
  const { container } = render(
    <MemoryRouter>
      <LiveNowSection
        games={[
          {
            ...linkedLiveGame,
            away: {
              ...linkedLiveGame.away,
              logoUrl: "https://a.espncdn.com/i/teamlogos/wnba/500/atl.png",
            },
            home: {
              ...linkedLiveGame.home,
              logoUrl: "https://a.espncdn.com/i/teamlogos/wnba/500/dal.png",
            },
          },
        ]}
      />
    </MemoryRouter>,
  );
  const images = container.querySelectorAll("img");
  expect(images).toHaveLength(2);
  expect(images[0]).toHaveAttribute(
    "src",
    "https://a.espncdn.com/i/teamlogos/wnba/500/atl.png",
  );
});
```

Add MatchupGameCard test with the same pattern (expect 2 imgs when both `logoUrl`s set; expect letter `G` when null).

- [ ] **Step 6: Run UI tests**

Run:

```bash
npm --prefix "/Users/alexgonzalez/Documents/NBA-Prop-Predictor/frontend" run test -- src/components/TeamAbbrevAvatar.test.tsx src/components/home/LiveNowSection.test.tsx src/components/league/MatchupGameCard.test.tsx src/components/home/mapScoreboard.test.ts
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/TeamAbbrevAvatar.tsx \
  frontend/src/components/TeamAbbrevAvatar.test.tsx \
  frontend/src/components/home/LiveNowSection.tsx \
  frontend/src/components/home/LiveNowSection.test.tsx \
  frontend/src/components/league/MatchupGameCard.tsx \
  frontend/src/components/league/MatchupGameCard.test.tsx
git commit -m "$(cat <<'EOF'
feat: show ESPN logos on Live Now and matchup cards

Replace letter circles with TeamAbbrevAvatar, falling back to the abbrev letter when logoUrl is missing or fails.
EOF
)"
```

---

## Spec coverage checklist

| Spec requirement | Task |
| --- | --- |
| `WnbaTeam.logo_url` from `team.logo` | Task 1 |
| Stats → null | Task 1 |
| Merge keeps ESPN logo | Task 1 |
| Live/Matchup `logoUrl` mapping | Task 2 |
| Letter fallback + onError | Task 3 |
| Live Now + Matchup cards | Task 3 |
| Ticker / game-header / no CDN invent | Honored by scope |
