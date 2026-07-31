# Rename HoopVista Picks → Prop Picks

Date: 2026-07-31  
Status: Approved for planning

## Goal

Rename the Explore subnav pill label from **HoopVista Picks** to **Prop Picks**.

## Decisions

| Topic | Choice |
| --- | --- |
| New copy | `Prop Picks` |
| Behavior | Still disabled / non-navigating |
| Scope | `LeagueSubnav` + its test only |
| Docs | Do not rewrite historical plans/specs |

## Files touched

| File | Change |
| --- | --- |
| `frontend/src/components/league/LeagueSubnav.tsx` | `exploreItems` string |
| `frontend/src/components/league/LeagueSubnav.test.tsx` | button name assertion |

## Success criteria

- Subnav shows **Prop Picks**; test looks up that name; no new routes.
