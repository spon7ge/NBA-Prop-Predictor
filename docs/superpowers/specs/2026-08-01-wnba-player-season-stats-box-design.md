# WNBA player season stats box

Date: 2026-08-01  
Status: Approved for planning

## Goal

Replace the five separate average tiles on the player header with one boxed “regular season stats” card matching the structure of the provided mock (header bar + equal columns), adapted to HoopVista charcoal chrome. Stats remain PTS · REB · AST · FG% · 3P%.

## Decisions

| Topic | Choice |
| --- | --- |
| Stats | PTS, REB, AST, FG%, 3P% (no API / STL change) |
| Visual | Charcoal hub adaptation (Approach A) — muted header bar, dark body |
| Structure | Single rounded box: header strip + 5 equal columns |
| Header copy | `{season} REGULAR SEASON STATS` (uppercase) |
| Placement | Right side of `PlayerHeader` (replaces tile grid); stacks under bio on mobile |
| Implementation | Restyle inside `PlayerHeader` (Approach 1) |
| Out of scope | Blue/light ESPN colors, STL, new backend fields |

## UI

```
┌─────────────────────────────────────┐
│  2026 REGULAR SEASON STATS          │  ← muted charcoal bar, white/near-white text
├─────────────────────────────────────┤
│  PTS    REB    AST    FG%    3P%    │  ← small muted labels
│  19.4   4.7    5.8    48.2   33.1   │  ← large bold tabular values
└─────────────────────────────────────┘
```

- Outer: `rounded-xl border border-white/10`, subtle lift optional via existing border (no heavy shadow)
- Header: filled bar using existing white/opacity or slightly stronger surface (e.g. `bg-white/10`), not bright blue
- Columns: equal width, centered, padding similar to mock
- Values from `player.averages` / `player.season` (already on API)

## Tests

- `PlayerHeader` still shows all five labels and values
- Assert header text includes `REGULAR SEASON STATS` and season year
- Bio / omit-empty / headshot tests unchanged in intent

## Out of scope

- Backend changes
- Matching mock’s white card / blue header hex exactly
- Adding STL or other categories
