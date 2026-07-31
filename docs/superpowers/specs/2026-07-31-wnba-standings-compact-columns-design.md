# WNBA standings compact table columns

Date: 2026-07-31  
Status: Approved for planning

## Goal

Keep East | West side-by-side standings on desktop while making all columns (`#` through `Strk`, including Away / L10 / Diff / Strk) visible without relying on horizontal scroll. Free space by tightening TEAM→W-L gap and slightly smaller type.

## Decisions

| Topic | Choice |
| --- | --- |
| Approach | Compact table in `StandingsConferenceCard` (Approach 1) |
| Layout | Keep `lg:grid-cols-2` East \| West; do not stack solely to fit columns |
| Team cell | Keep logo + colored abbrev + full name (no truncate in v1) |
| Overflow | Remove forced `min-w-[720px]`; retain `overflow-x-auto` as last-resort safety |
| Scope | Frontend card styling only — no API / schema changes |
| Fallback | If mid-width still clips, a follow-up may add name truncate (Approach 2) — out of scope unless needed |

## Visual changes

In `StandingsConferenceCard.tsx`:

1. **Table width** — Drop `min-w-[720px]` so the table can shrink to the card. Keep `w-full` and `overflow-x-auto` on the wrapper.
2. **Type** — Body/table from `text-sm` → `text-xs`; header stays uppercase muted (slightly smaller tracking-friendly size if needed).
3. **Padding** — Tighten horizontal padding, especially on the TEAM cell (`pr-2` → `pr-1`) and other stat columns (`pr-1` / `pr-1.5`). Prefer `tabular-nums` on numeric/record columns for alignment.
4. **Unchanged** — Diff green/red rules; Strk W/L color rules; conference card chrome; grid / page structure; data mapping.

## Files touched

| File | Change |
| --- | --- |
| `frontend/src/components/league/StandingsConferenceCard.tsx` | Compact sizing / padding; remove min-width |
| `frontend/src/components/league/StandingsGrid.test.tsx` | Assert Away / L10 / Diff / Strk still render (if not already) |

## Testing

- Conference card still shows Away, L10, Diff, Strk values from fixtures.
- Diff/Strk color classes still applied for `+` / `-` and `W` / `L`.
- No horizontal-scroll requirement for the default `max-w-6xl` + `lg` two-column layout in typical desktop widths (manual check).
- Existing StandingsGrid / page tests still pass.

## Out of scope

- Stacking conferences to gain width
- Truncating or hiding full team names
- Widening `max-w-6xl` page container
- Backend or ESPN mapping changes
- NBA standings

## Success criteria

- On desktop with East | West side-by-side, Away / L10 / Diff / Strk are visible without scrolling the table.
- Team full names remain shown.
- Diff and Strk retain existing color semantics.
