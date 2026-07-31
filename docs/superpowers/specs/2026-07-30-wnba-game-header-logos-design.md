# WNBA game header logos (ESPN summary)

Date: 2026-07-30  
Status: Approved for planning

## Goal

Show official WNBA team logos next to team names on the game detail header, using logo URLs already present in the ESPN summary payload. Scope is game detail header only; matchup cards and live ticker stay on letter avatars.

## Decisions

| Topic | Choice |
| --- | --- |
| Source | ESPN summary `competitors[].team.logos[]` `href` (Approach 1) |
| Variant | Prefer `rel` containing `dark`, else `default`, else first `href` |
| Missing logos | `logo_url` / `logoUrl` is `null` — do not invent CDN URLs |
| Surfaces | `GameHeader` only |
| Colors | Unchanged (`color` already from ESPN); no `alternateColor` in this work |
| Fallback UI | Hide broken `<img>` via `onError`; keep colored team name |

### Rejected alternatives

| Approach | Why not |
| --- | --- |
| Build CDN URL from ESPN team id | WNBA paths like `/wnba/500/{id}.png` 404 for most teams |
| Build CDN URL from abbrev on frontend | Fragile with `LV`/`LVA`, `NY`/`NYL` and new franchises |
| Logos on matchup cards / ticker | Deferred; letter circles remain |

## Architecture

```
ESPN …/wnba/summary?event=…
        │
        ▼
  normalize GameDetailTeam
    + logo_url from logos[] (dark > default > first)
        │
        ▼
  GET /api/wnba/games/{espnEventId}
        │
        ▼
  mapGameDetail → away/home.logoUrl
        │
        ▼
  GameHeader TeamRow: [logo?] name … score
```

## Data & API

Extend existing game-detail team model only (no new endpoints).

### `GameDetailTeam` (backend)

| Field | Type | Notes |
| --- | --- | --- |
| `id` | `str` | unchanged |
| `abbrev` | `str` | unchanged |
| `name` | `str` | unchanged |
| `score` | `int \| None` | unchanged |
| `color` | `str` | unchanged (`#RRGGBB`) |
| `logo_url` | `str \| None` | new; absolute ESPN CDN `https://…` URL or `null` |

### Frontend `GameDetailTeam`

| Field | Type | Notes |
| --- | --- | --- |
| `logoUrl` | `string \| null` | mapped from `logo_url` |

### Logo selection helper

Given `team.logos` (list of objects with optional `rel` and `href`):

1. Prefer entry whose `rel` list/string includes `dark` and has a non-empty `href`.
2. Else prefer entry whose `rel` includes `default` and has a non-empty `href`.
3. Else first entry with a non-empty `href`.
4. Else `null`.

Do not rewrite or proxy the URL; pass ESPN’s `href` through as-is (including `500-dark/…` paths).

## UI

### `GameHeader` team row

Layout (left → right):

1. Optional logo image (~28–32px, `object-contain`, shrink-0).
2. Team `name` in existing team `color`.
3. Existing amber score box.

Accessibility:

- `alt=""` (decorative; name is adjacent text).
- If `logoUrl` is null, omit the `<img>` entirely.
- On `onError`, remove/hide the image so a broken icon does not remain.

Out of scope: matchup cards, live ticker, season leaders, projected starters, win-probability chrome.

## Testing

| Layer | Cases |
| --- | --- |
| Backend normalize | Fixture with both `dark` and `default` → picks dark `href`; fixture with no `logos` → `logo_url` is `null` |
| Frontend map | `logo_url` maps to `logoUrl` |
| `GameHeader` | Renders `<img>` with expected `src` when `logoUrl` set; no `<img>` when null |

## Non-goals

- Caching or downloading logo binaries
- NBA logos
- Letter-avatar replacement on home / matchups / ticker
- `alternateColor` plumbing
- Licensing / self-hosted brand assets

## Implementation notes

- Update at least one ESPN summary fixture used by normalize tests to include a realistic `logos` array (mirror live ESPN: `full`+`default`, `full`+`dark`).
- Existing fixtures without `logos` must still normalize with `logo_url: null`.
