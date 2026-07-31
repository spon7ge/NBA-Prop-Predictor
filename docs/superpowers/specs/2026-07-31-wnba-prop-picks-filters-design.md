# WNBA Prop Picks filters

Date: 2026-07-31  
Status: Approved for planning

## Goal

Add client-side filters on `/wnba/prop_picks` for Stat, Over/Under, and Team (multi-select each). Show a disabled **+EV · Soon** control as a placeholder until EV data exists.

## Decisions

| Topic | Choice |
| --- | --- |
| Selection mode | Multi-select for Stat, O/U, and Team |
| Control UI | Dropdown menus with multi-check |
| Filtering | Client-side only on already-fetched props |
| Empty filter | No restriction for that dimension (show all) |
| Combine filters | AND across dimensions; OR within a dimension |
| +EV | Visible, disabled, labeled `+EV · Soon`; no filtering |
| URL sync | Out of scope for v1 |
| Backend | No API changes |

## UI

Toolbar under the Prop Picks heading, above the table, in the same `max-w-6xl` container.

Left → right:

1. **Stat** — multi-check dropdown; options = sorted unique `stat` values from loaded props  
2. **O/U** — multi-check dropdown; options = Over, Under  
3. **Team** — multi-check dropdown; options = sorted unique `team_abbrev` values that are non-null; show team logo beside abbrev in the menu when `logo_url` is known for that team  
4. **+EV** — disabled control: `+EV · Soon`

Trigger button labels:

- No selection: `Stat`, `O/U`, `Team`  
- With selection: e.g. `Stat (2)`, `O/U (1)`, `Team (3)` (count of selected values)

When any of Stat / O/U / Team has a non-empty selection, show a text **Clear filters** control that resets those three to empty.

Styling: match existing dark WNBA chrome (border `white/10`, `#121212` / `#141414` surfaces, violet accent for checked/active where needed). Do not introduce a new card aesthetic beyond a compact toolbar.

## Filter logic

Given loaded `props: ApiWnbaPropLine[]` and selections:

```
stats: Set<string>     // empty → pass all
sides: Set<'over'|'under'>  // empty → pass all
teams: Set<string>     // empty → pass all
```

A row is visible if:

- (`stats` empty OR `row.stat` ∈ `stats`) AND  
- (`sides` empty OR `row.side` ∈ `sides`) AND  
- (`teams` empty OR (`row.team_abbrev` ≠ null AND `row.team_abbrev` ∈ `teams`))

Rows with `team_abbrev == null` never match a non-empty Team filter.

### Empty / error copy

| State | Copy |
| --- | --- |
| Loading | Existing skeleton |
| Fetch error / zero props from API | `Prop lines unavailable` |
| Props loaded, filters hide all rows | `No props match these filters` |

## Architecture

```
LeaguePropPicksPage
  ├── useWnbaProps() → all props
  ├── PropPicksFilters (local state)
  │     options derived from props
  │     +EV disabled
  └── PropPicksTable(filteredProps)
```

Pure helper (unit-tested): `filterPropLines(props, { stats, sides, teams })`.

Optional small `MultiSelectFilter` used by Stat / O/U / Team.

## File layout

```
frontend/src/components/league/filterPropLines.ts (+ test)
frontend/src/components/league/PropPicksFilters.tsx (+ test)
frontend/src/components/league/PropPicksTable.tsx   # empty-filter copy if needed
frontend/src/pages/LeaguePropPicksPage.tsx          # wire filters
```

No backend changes.

## Testing

- `filterPropLines`: empty selections = identity; AND across filters; OR within; null team excluded when Team filter active  
- Filters UI: options from props; multi-toggle; Clear resets; +EV disabled with Soon  
- Table/page: filtered empty state vs unavailable state

## Out of scope

- Server-side filter query params  
- URL query sync / shareable filter links  
- Real +EV filtering or EV calculation  
- Persist filter state across navigations  
- NBA Prop Picks

## Success criteria

- Users can multi-filter today’s WNBA props by stat, O/U side, and team.  
- +EV appears disabled with Soon.  
- Empty filter selections mean “all.”  
- Filtered-empty and API-empty messages are distinct.  
- Relevant frontend tests pass.
