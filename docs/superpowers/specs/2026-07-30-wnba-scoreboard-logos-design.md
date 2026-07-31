# WNBA scoreboard logos (Live Now + Matchups)

Date: 2026-07-30  
Status: Approved for planning

## Goal

Replace letter-circle avatars with ESPN team logos on **Live Now** home cards and **league matchup** cards, using logo URLs from the ESPN scoreboard payload. Keep letter-circle fallback when a logo is missing or fails to load. Live ticker stays letter-free / abbrev-only (unchanged).

## Decisions

| Topic | Choice |
| --- | --- |
| Surfaces | Live Now cards + Matchup hub cards (Approach B) |
| Source | ESPN scoreboard `competitors[].team.logo` string (Approach 1) |
| Missing logos | `logo_url` / `logoUrl` is `null` — letter avatar fallback |
| Stats-only rows | `null` (stats.wnba.com has no logo field) |
| URL rewriting | None — pass ESPN `logo` through as-is (default `/500/` path) |
| Dark variant | Not applied on scoreboard (unlike game-detail `logos[]` dark preference) |
| Ticker | Out of scope |
| Game header | Already ships logos via summary `logos[]` — unchanged |

### Relationship to game-header logos

| Surface | Payload | Selection |
| --- | --- | --- |
| Game detail header | ESPN summary `team.logos[]` | Prefer `dark` → `default` → first `href` |
| Live Now / Matchups | ESPN scoreboard `team.logo` | Pass through non-empty string |

Do not invent CDN URLs from abbrev or team id.

## Architecture

```
ESPN …/wnba/scoreboard
        │
        ▼
  normalize_espn_scoreboard → WnbaTeam.logo_url
  normalize_stats_scoreboard → logo_url = null
  merge_games → keep ESPN logo_url when present
        │
        ▼
  GET /api/wnba/scoreboard/today
        │
        ├── mapToLiveGames → LiveGameTeam.logoUrl → LiveNowSection
        └── mapToMatchupGames → MatchupTeam.logoUrl → MatchupGameCard
```

## Data & API

### `WnbaTeam` (backend)

| Field | Type | Notes |
| --- | --- | --- |
| existing fields | unchanged | abbrev, name, score, record |
| `logo_url` | `str \| None` | new; absolute ESPN CDN URL or `null` |

### Selection helper (scoreboard)

Given competitor `team`:

1. Read `team.logo` (string).
2. If stripped non-empty → use as `logo_url`.
3. Else if `team.logos` is a non-empty list, reuse the same dark→default→first helper as game detail (`_team_logo_url`) when practical — optional; live scoreboard currently has empty `logos` and a populated `logo`.
4. Else `null`.

Minimum required for v1: step 1–2 + 4. Prefer extracting a tiny shared `_espn_logo_url(team: dict) -> str | None` that checks `logo` first, then `logos[]`, so both scoreboard and (optionally later) other ESPN shapes share one path. **Do not** change game-detail summary selection rules in this work.

### Merge behavior

When merging ESPN + stats games, preserve `logo_url` from the ESPN-sided team when present; do not overwrite a non-null ESPN logo with stats `null`.

### Frontend types

| Type | Field |
| --- | --- |
| `ApiWnbaTeam` | `logo_url: string \| null` |
| `LiveGameTeam` | `logoUrl: string \| null` |
| `MatchupTeam` | `logoUrl: string \| null` |

Mappers (`mapToLiveGames`, `mapToMatchupGames`) pass through `logo_url` → `logoUrl`.

## UI

### Avatar behavior (both surfaces)

In the existing letter-circle slot (~size-7 Live Now, ~size-8 Matchups):

1. If `logoUrl` is set → render `<img>` (`object-contain`, decorative `alt=""`).
2. On `onError` or `logoUrl === null` → render the existing letter circle (`abbrev.slice(0, 1)`).

Optional: extract a small shared `TeamLetterLogo` (or similar) used by `LiveNowSection` and `MatchupGameCard` to avoid duplicating error-state logic. Game header may keep its own `TeamLogo` (no letter fallback there — name is adjacent).

### Out of scope

- Live ticker chips
- NBA scoreboard logos
- Changing game-detail logo selection
- Prefetch / self-hosting logos

## Testing

| Layer | Cases |
| --- | --- |
| Backend ESPN normalize | Fixture with `team.logo` → `logo_url` equals that href |
| Backend stats normalize | `logo_url` is `null` |
| Backend merge | ESPN logo retained over stats null |
| Frontend mappers | `logo_url` → `logoUrl` for live + matchup maps |
| `LiveNowSection` | Renders img when `logoUrl` set; letter when null |
| `MatchupGameCard` | Same |

## Non-goals

- Rewriting `/500/` to `/500-dark/`
- Building logos from abbrev/id
- Ticker logos
