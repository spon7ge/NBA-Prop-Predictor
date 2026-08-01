# Prop Picks “Last updated” timestamp

Date: 2026-07-31  
Status: Approved

## Goal

Show when the WNBA Prop Picks page last successfully called the props API, next to the “Prop Picks” title, in a style that matches the existing dark league UI.

## Decisions

| Topic | Choice |
| --- | --- |
| Timestamp source | React Query `dataUpdatedAt` from `useWnbaProps()` (client last successful fetch / poll) |
| Placement | Same row as the `h2` “Prop Picks” title, immediately to the right |
| Layout | Flex row: `items-baseline gap-3` |
| Copy | `Last updated {date} {time}` in the user’s local timezone |
| Format | Short local datetime, e.g. `Jul 31, 11:54 PM` via `Intl.DateTimeFormat("en-US", …)` |
| Style | `text-sm text-white/40` (muted secondary chrome) |
| Visibility | Hide when `dataUpdatedAt` is `0` / unavailable (no successful fetch yet) |
| Scope | `LeaguePropPicksPage` passes `lastUpdatedAt`; `PropPicksTable` renders it |

## Out of scope

- API `as_of` / book scrape times
- Relative time (“2 min ago”)
- Changing refetch interval behavior

## Test plan

- Unit: `PropPicksTable` shows formatted “Last updated …” when `lastUpdatedAt` is set; omits it when unset.
- Manual: load `/wnba/prop_picks`, confirm timestamp appears beside the title and refreshes after the 60s poll.
