# WNBA player bio showcase (header enrichment)

Date: 2026-08-01  
Status: Approved for planning

## Goal

Enrich `/wnba/player/:playerId` with an ESPN-style left bio column: jersey, full position, height, birthdate (with age), college, and draft info — while keeping season averages on the right. Data stays on the existing `commonplayerinfo` fetch (Approach 1).

## Decisions

| Topic | Choice |
| --- | --- |
| Layout | ESPN-style: wider left bio column; averages remain right |
| Fields | Jersey #, full position, height, birthdate (+ age), college, draft info |
| Missing data | Omit the row / subtitle piece; do not show empty labels |
| Data source | Extend existing `commonplayerinfo` parsing (no ESPN athlete API) |
| API | Same `GET /api/wnba/player/{player_id}`; add nullable bio fields |
| Position display | Prefer full name from upstream; else map `G`/`F`/`C` (and common variants) → Guard / Forward / Center |
| Draft display | Assemble `"{year}: Rd {round}, Pk {number} ({team})"` when parts exist |
| Birthdate | `M/D/YYYY (age)` with age computed from birth date |
| Visual language | Existing charcoal league hub; larger headshot |
| Out of scope | ESPN ID mapping, weight/country, career timeline |

## Architecture

```
GET /api/wnba/player/{id}  (existing)
        │
        ├── leaguedashplayerstats → identity + averages (unchanged)
        ├── playergamelog → games (unchanged)
        └── commonplayerinfo → position + NEW bio fields
                │
                ▼
PlayerHeader
  left: headshot, name, # · position · team, fact rows
  right: PTS / REB / AST / FG% / 3P%
```

## Backend

### Response additions (`WnbaPlayerResponse`)

All new fields nullable strings (pre-formatted for display where useful):

```json
{
  "jersey": "5",
  "position": "Guard",
  "height": "5' 10\"",
  "birthdate": "1/29/2003 (23)",
  "college": "TCU",
  "draft_info": "2026: Rd 1, Pk 2 (MIN)"
}
```

Notes:
- `jersey` is the bare number; UI prefixes `#`.
- `position` replaces/enhances the existing field to the display form (full name when possible). Keep backward-compatible: still a single `position` string on the response.
- Upstream `commonplayerinfo` headers typically include variants of: `JERSEY`, `HEIGHT`, `BIRTHDATE`, `SCHOOL` / `LAST_AFFILIATION`, `DRAFT_YEAR`, `DRAFT_ROUND`, `DRAFT_NUMBER`, `TEAM_ABBREVIATION`, `POSITION`. Map defensively to whatever headers exist in the live payload.
- If draft year missing but round/number present (or vice versa), include only the available pieces; if nothing usable → `draft_info: null`.

### Normalize helpers

- `format_height(raw) -> str | None`
- `format_birthdate(raw, *, today=None) -> str | None` — parse ISO or stats date; append `(age)`
- `format_position(raw) -> str | None` — expand abbrevs
- `format_draft(year, round_, number, team_abbrev) -> str | None`
- `format_jersey(raw) -> str | None` — digits only, no `#`

## Frontend

### `PlayerHeader` layout

**Desktop (`md+`):** two columns — left bio (flex-1), right avg tiles.

**Left:**
1. Larger headshot (e.g. ~96px) + name
2. Subtitle: join non-empty of `#${jersey}`, `position`, `team_name` with ` · `
3. Definition list / stacked rows for Height, Birthdate, College, Draft Info — only render rows whose value is non-null

**Right:** existing five average tiles unchanged.

**Mobile:** bio stack first, averages below.

### Tests

- Backend: fixture with full bio fields → formatted strings; sparse info → nulls, no crash; abbrev position → full name; birthdate age.
- Frontend: header shows jersey/position/team line and fact rows; omits missing rows; averages still render.

## Out of scope

- ESPN athlete endpoint / ID mapping
- Weight, country, experience years (unless already trivial from same row — do not add in v1)
- Changing Recent games section
- NBA player pages

## Spec coverage note

This is an additive change to the existing player profile feature (`2026-08-01-wnba-player-page-design.md`). Suitable for a short follow-up plan.
