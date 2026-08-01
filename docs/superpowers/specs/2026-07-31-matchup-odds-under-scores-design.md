# Matchup card — odds under scores (live + final)

Date: 2026-07-31  
Status: Approved

## Goal

On WNBA matchup cards, when a game has started (`status !== "scheduled"`), place the DraftKings team-lines pill under the score column instead of beside the home row.

## Decisions

| Topic | Choice |
| --- | --- |
| Trigger | Live + final (same as `showScores`) |
| Layout | Approach 1 — shared score column with odds stacked under it |
| Scheduled | Unchanged — pill beside home row |
| Odds data / copy | No change |

## Layout

**Live / final (scores visible):**

```
status                          venue
[away …]                    [score]
[home …]                    [score]
                            [odds pill]
                            Odds by DK
```

**Scheduled (no scores):** keep current home-row side pill.

## Out of scope

- Prop picks table pills
- Odds fetch / merge logic
- NBA cards (same component will inherit if used)
