# Prop Picks Hide Final Games Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On `/wnba/prop_picks`, hide prop rows for players whose game has `status === "final"`.

**Architecture:** Frontend-only. A pure helper builds a set of team abbrevs from final scoreboard games and filters props by `team_abbrev`. `LeaguePropPicksPage` loads scoreboard via `useWnbaScoreboard`, excludes finals before user filters and filter-chip options. No API changes.

**Tech Stack:** React, TypeScript, Vitest, TanStack Query, existing `useWnbaScoreboard` / `useWnbaProps`

## Global Constraints

- Hide only when `status === "final"` (keep live / scheduled / halftime).
- Match via prop `team_abbrev` ↔ scoreboard home/away `abbrev`.
- Missing `team_abbrev` → keep row.
- Missing / empty scoreboard games → no exclusion (show all props).
- Filter chips built from post-exclusion props.
- No backend / schema changes.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `frontend/src/components/league/filterPropLines.ts` | Add `excludePropsFromFinalGames` |
| `frontend/src/components/league/filterPropLines.test.ts` | Unit tests for exclusion helper |
| `frontend/src/pages/LeaguePropPicksPage.tsx` | Wire scoreboard + exclusion before filters |

---

### Task 1: `excludePropsFromFinalGames` helper

**Files:**
- Modify: `frontend/src/components/league/filterPropLines.ts`
- Modify: `frontend/src/components/league/filterPropLines.test.ts`

**Interfaces:**
- Consumes: `ApiWnbaPropLine`, `ApiWnbaGame` from `@/lib/api`
- Produces: `excludePropsFromFinalGames(props: ApiWnbaPropLine[], games: ApiWnbaGame[] | undefined | null): ApiWnbaPropLine[]`

- [ ] **Step 1: Write the failing tests**

Append to `filterPropLines.test.ts`:

```ts
import type { ApiWnbaGame } from "@/lib/api";
import { excludePropsFromFinalGames } from "./filterPropLines";

function game(
  partial: Partial<ApiWnbaGame> & {
    status: ApiWnbaGame["status"];
    homeAbbrev: string;
    awayAbbrev: string;
  },
): ApiWnbaGame {
  return {
    id: partial.id ?? "g1",
    espn_event_id: null,
    league: "wnba",
    status: partial.status,
    status_label: partial.status_label ?? partial.status,
    start_time_et: "7:00 PM ET",
    away: {
      abbrev: partial.awayAbbrev,
      name: partial.awayAbbrev,
      score: null,
      logo_url: null,
    },
    home: {
      abbrev: partial.homeAbbrev,
      name: partial.homeAbbrev,
      score: null,
      logo_url: null,
    },
  };
}

describe("excludePropsFromFinalGames", () => {
  it("removes props for both teams in a final game", () => {
    const games = [game({ status: "final", homeAbbrev: "ATL", awayAbbrev: "SEA" })];
    const out = excludePropsFromFinalGames(rows, games);
    expect(out.map((r) => r.player_name)).toEqual(["Unknown"]);
  });

  it("keeps props for live and scheduled games", () => {
    const games = [
      game({ status: "live", homeAbbrev: "ATL", awayAbbrev: "CHI", id: "live" }),
      game({
        status: "scheduled",
        homeAbbrev: "SEA",
        awayAbbrev: "LAS",
        id: "sched",
      }),
    ];
    const out = excludePropsFromFinalGames(rows, games);
    expect(out).toEqual(rows);
  });

  it("keeps rows with null team_abbrev", () => {
    const games = [game({ status: "final", homeAbbrev: "ATL", awayAbbrev: "SEA" })];
    const out = excludePropsFromFinalGames(rows, games);
    expect(out.some((r) => r.team_abbrev == null)).toBe(true);
  });

  it("does not filter when games are empty or undefined", () => {
    expect(excludePropsFromFinalGames(rows, [])).toEqual(rows);
    expect(excludePropsFromFinalGames(rows, undefined)).toEqual(rows);
    expect(excludePropsFromFinalGames(rows, null)).toEqual(rows);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npx vitest run src/components/league/filterPropLines.test.ts`

Expected: FAIL — `excludePropsFromFinalGames` is not exported / not defined.

- [ ] **Step 3: Implement the helper**

Add to `filterPropLines.ts`:

```ts
import type { ApiWnbaGame, ApiWnbaPropLine } from "@/lib/api";

export function excludePropsFromFinalGames(
  props: ApiWnbaPropLine[],
  games: ApiWnbaGame[] | undefined | null,
): ApiWnbaPropLine[] {
  if (!games || games.length === 0) return props;

  const finalTeams = new Set<string>();
  for (const g of games) {
    if (g.status !== "final") continue;
    if (g.home?.abbrev) finalTeams.add(g.home.abbrev);
    if (g.away?.abbrev) finalTeams.add(g.away.abbrev);
  }
  if (finalTeams.size === 0) return props;

  return props.filter(
    (row) => !row.team_abbrev || !finalTeams.has(row.team_abbrev),
  );
}
```

Keep existing exports (`filterPropLines`, `collectStatOptions`, `collectTeamOptions`) unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/components/league/filterPropLines.test.ts`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/league/filterPropLines.ts frontend/src/components/league/filterPropLines.test.ts
git commit -m "$(cat <<'EOF'
feat: exclude prop picks for teams in final games

EOF
)"
```

---

### Task 2: Wire exclusion into `LeaguePropPicksPage`

**Files:**
- Modify: `frontend/src/pages/LeaguePropPicksPage.tsx`
- Test: covered by Task 1 unit tests; smoke via existing router test if needed

**Interfaces:**
- Consumes: `excludePropsFromFinalGames` from `./filterPropLines` (via `@/components/league/filterPropLines`)
- Consumes: `useWnbaScoreboard` → `games: ApiWnbaGame[]`
- Produces: page uses `activeProps = excludePropsFromFinalGames(props, games)` for table filters and chip options

- [ ] **Step 1: Update the page**

Replace `LeaguePropPicksPage.tsx` body logic so scoreboard games feed exclusion before filters:

```tsx
import { useState } from "react";
import { LeagueSubnav } from "@/components/league/LeagueSubnav";
import { PropPicksFilters } from "@/components/league/PropPicksFilters";
import { PropPicksTable } from "@/components/league/PropPicksTable";
import {
  collectStatOptions,
  collectTeamOptions,
  excludePropsFromFinalGames,
  filterPropLines,
} from "@/components/league/filterPropLines";
import { useWnbaProps } from "@/hooks/useWnbaProps";
import { useWnbaScoreboard } from "@/hooks/useWnbaScoreboard";

export function LeaguePropPicksPage() {
  const { data, isLoading, isError, isFetched } = useWnbaProps();
  const { games } = useWnbaScoreboard();
  const props = data?.props ?? [];
  const activeProps = excludePropsFromFinalGames(props, games);
  const showError = isError && !data;
  const showLoading = isLoading && !isFetched;
  const apiEmpty =
    showError || Boolean(data && props.length === 0 && data.error);

  const [selectedStats, setSelectedStats] = useState<Set<string>>(
    () => new Set(),
  );
  const [selectedSides, setSelectedSides] = useState<Set<string>>(
    () => new Set(),
  );
  const [selectedTeams, setSelectedTeams] = useState<Set<string>>(
    () => new Set(),
  );

  const filtersActive =
    selectedStats.size > 0 ||
    selectedSides.size > 0 ||
    selectedTeams.size > 0;

  const filtered = filterPropLines(activeProps, {
    stats: selectedStats,
    sides: selectedSides,
    teams: selectedTeams,
  });

  return (
    <div className="space-y-6 py-6">
      <LeagueSubnav league="wnba" />
      <PropPicksTable
        props={filtered}
        isLoading={showLoading}
        isError={apiEmpty}
        filtersActive={filtersActive && !apiEmpty && activeProps.length > 0}
        toolbar={
          !showLoading && !apiEmpty && activeProps.length > 0 ? (
            <PropPicksFilters
              stats={collectStatOptions(activeProps)}
              teams={collectTeamOptions(activeProps)}
              selectedStats={selectedStats}
              selectedSides={selectedSides}
              selectedTeams={selectedTeams}
              onStatsChange={setSelectedStats}
              onSidesChange={setSelectedSides}
              onTeamsChange={setSelectedTeams}
              onClear={() => {
                setSelectedStats(new Set());
                setSelectedSides(new Set());
                setSelectedTeams(new Set());
              }}
            />
          ) : null
        }
      />
    </div>
  );
}
```

Notes:
- `apiEmpty` still keys off raw `props` + API error so a day where every game is final (all props excluded) can show an empty table with filters hidden (`activeProps.length === 0`), not the API-error copy — unless the API itself failed.
- When all props are finals-excluded, table gets `filtered = []` and `filtersActive` is false → shows muted “Prop lines unavailable”. That is acceptable for v1 (all games done).

- [ ] **Step 2: Run unit + router smoke tests**

Run:

```bash
cd frontend && npx vitest run src/components/league/filterPropLines.test.ts src/AppRouter.test.tsx
```

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add frontend/src/pages/LeaguePropPicksPage.tsx
git commit -m "$(cat <<'EOF'
feat: hide finished-game players on WNBA prop picks

EOF
)"
```

---

## Spec coverage check

| Spec requirement | Task |
| --- | --- |
| Hide only `final` | Task 1 |
| Frontend filter via team abbrev | Task 1 + 2 |
| Keep live/scheduled | Task 1 |
| Keep null `team_abbrev` | Task 1 |
| Empty scoreboard → no filter | Task 1 |
| Chips from post-exclusion props | Task 2 |
| No API changes | Both tasks |
