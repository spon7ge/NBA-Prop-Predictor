# Prop Picks — hide players from finished games

Date: 2026-07-31  
Status: Approved for planning

## Goal

On `/wnba/prop_picks`, remove prop rows for players whose game has finished (`status === "final"`), so the table only shows props for games still upcoming or in progress.

## Decisions

| Topic | Choice |
| --- | --- |
| Trigger | `status === "final"` only (keep live / scheduled) |
| Where | Frontend filter (Approach 1) |
| Match key | Prop `team_abbrev` ↔ scoreboard home/away `abbrev` |
| Scoreboard failure | Do not filter — show all props |
| Missing `team_abbrev` | Keep the row |
| Filter chips | Build options from post-exclusion props so finished teams disappear |

## Behavior

1. `LeaguePropPicksPage` loads props (`useWnbaProps`) and today’s scoreboard (`useWnbaScoreboard`) in parallel.
2. Collect team abbrevs from games where `status === "final"` (both home and away).
3. Drop any prop whose `team_abbrev` is in that set.
4. Apply existing stat / side / team filters to the remaining rows.
5. Live, scheduled, and unknown statuses stay visible.
6. If scoreboard data is missing or errored, skip exclusion and show all props.

## Implementation

- Pure helper in `filterPropLines.ts` (or adjacent), e.g. `excludePropsFromFinalGames(props, games)`.
- Wire in `LeaguePropPicksPage` before `filterPropLines` and before `collectStatOptions` / `collectTeamOptions`.
- No backend API or schema changes.

## Testing

- Final game → both teams’ prop rows removed.
- Live / scheduled → rows kept.
- Missing `team_abbrev` → row kept.
- Empty / undefined games → no filtering.
- Optional page-level: when scoreboard reports a final, those players are absent from the table.

## Out of scope

- Backend filtering in `/api/wnba/props/today`
- Hiding live / halftime games
- NBA Prop Picks
- Changing scoreboard or Sharp fetch behavior

## Success criteria

- Finished-game players do not appear on `/wnba/prop_picks`.
- Live and upcoming game props still appear.
- Scoreboard outage does not blank the prop table.
- Helper unit tests pass.
