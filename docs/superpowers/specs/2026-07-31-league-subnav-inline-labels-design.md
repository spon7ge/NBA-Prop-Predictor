# League subnav inline Explore / Learn labels

Date: 2026-07-31  
Status: Approved for planning

## Goal

Put the **Explore** and **Learn** section labels on the same line as their pills (Matchups, HoopVista Picks, etc. / How it works, Glossary), with a thin vertical divider between the two groups. Applies to all league hubs that use `LeagueSubnav` (WNBA and NBA).

## Decisions

| Topic | Choice |
| --- | --- |
| Approach | Flex row groups: label + pills, `items-center` (Approach 1) |
| Divider | Thin vertical rule between Explore and Learn |
| Scope | `LeagueSubnav` layout only — no route / enablement changes |
| Leagues | Shared component — both WNBA and NBA |

## Visual structure

```
┌─────────────────────────────────────────────────────────────┐
│ Explore  [Matchups] [Picks] [Leaders] … │ Learn  [How…] [Gloss] │
└─────────────────────────────────────────────────────────────┘
```

- Outer bar: existing charcoal rounded container, horizontal scroll when needed.
- Each group: `flex items-center gap-2 shrink-0` — muted uppercase label, then pill row.
- Label styling: keep `text-[10px] font-semibold tracking-[0.18em] text-white/35 uppercase`; remove stacked `mb-2`.
- Divider: vertical `border-white/10` (e.g. `border-l pl-6` on Learn group, or a `w-px` separator).
- Pill link/button styles, active colors, and enabled/disabled routes unchanged.

## Files touched

| File | Change |
| --- | --- |
| `frontend/src/components/league/LeagueSubnav.tsx` | Inline label + divider layout |
| `frontend/src/components/league/LeagueSubnav.test.tsx` | Assert Explore/Learn text still present; links unchanged |

## Testing

- Explore and Learn labels still render.
- WNBA: Matchups / Leaders / Standings links + active state unchanged; other pills disabled.
- NBA: Matchups link; Leaders / Standings remain disabled.
- Optional: assert Learn group has a left border / separator class for the divider.

## Out of scope

- Enabling new Explore/Learn destinations
- Changing pill copy or order
- LeagueHero or page chrome outside the subnav

## Success criteria

- Explore sits on the same line as its pills; Learn on the same line as its pills.
- A thin divider separates the two groups.
- Navigation behavior matches current subnav.
