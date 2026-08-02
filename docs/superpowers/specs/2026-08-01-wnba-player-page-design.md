# WNBA player profile page

Date: 2026-08-01  
Status: Approved for planning

## Goal

Ship `/wnba/player/:playerId` so users can open a player from WNBA Leaders and see identity + season averages, then a recent-games table (last 5 by default, expand in place for the full current-season log).

## Decisions

| Topic | Choice |
| --- | --- |
| Route | `/wnba/player/:playerId` |
| Player ID | Same `player_id` as `/wnba/leaders` (stats.wnba.com `PLAYER_ID`) |
| Entry points (v1) | Leaders only — name links in `LeaderCategoryCard` |
| Architecture | Dedicated backend proxy `GET /api/wnba/player/{player_id}` (Approach 1) |
| Data source | stats.wnba.com (`leaguedashplayerstats`, `playergamelog`, `commonplayerinfo`) |
| Season | Current WNBA season only; no season picker |
| Header | Left: headshot, name, position, team; right: PTS, REB, AST, FG%, 3P% season avgs |
| Recent games default | Last 5, newest first |
| See more | Expand in place on the same page to full current-season log; “Show less” collapses |
| Game row columns | Box-score style: Date, Matchup, MIN, PTS, FG, 3PT, FT, REB, AST, TO, STL, BLK |
| Page chrome | `HomeChromeLayout` + `LeagueSubnav` (no new subnav tab; no `LeagueHero`) |
| Attribution | `Data: stats.wnba.com` |
| Cache | ~10 minutes in-process (aligned with leaders) |
| NBA | No `/nba/player/...` in v1 |

## Architecture

```
LeaderCategoryCard name Link
        │
        ▼
/wnba/player/:playerId  →  LeaguePlayerPage
        │
        ├── LeagueSubnav (no player tab; user came from Leaders)
        ├── PlayerHeader (bio left, averages right)
        ├── PlayerRecentGames (5 → expand full log)
        └── "Data: stats.wnba.com"
        │
        ▼
useWnbaPlayer(playerId) → GET /api/wnba/player/{player_id}
        │
        ▼
wnba_player service
  leaguedashplayerstats → identity + season averages (filter PLAYER_ID)
  commonplayerinfo → position (and bio gaps)
  playergamelog → current-season games (newest first)
  headshot_url from CDN pattern using player_id (nullable)
```

### Routing

| Path | Element |
| --- | --- |
| `/wnba/player/:playerId` | `LeaguePlayerPage` under `HomeChromeLayout` |
| `/nba/player/:playerId` | Not registered (404) |

- HomeNav WNBA active pill continues to cover any pathname starting with `/wnba`.
- Leaders rows: player **name** is a `Link` to `/wnba/player/{player_id}`; rest of the row is not a link.

## Page structure

Vertical stack under nav + ticker:

1. **LeagueSubnav** — existing Explore/Learn; no dedicated Player item.
2. **PlayerHeader**
   - **Left:** headshot (or placeholder), display name, position, team name (abbrev optional as secondary).
   - **Right:** five current-season per-game averages — PTS, REB, AST, FG%, 3P%.
3. **PlayerRecentGames**
   - Section title: “Recent games”.
   - Table columns: Date | Matchup | MIN | PTS | FG | 3PT | FT | REB | AST | TO | STL | BLK.
   - Default: first 5 games from API `games` (already newest-first).
   - Control: “See more” expands to full `games` array; “Show less” returns to 5.
   - Hide the control when `games.length <= 5`.
4. **Attribution** — `Data: stats.wnba.com`.

### Visual language

Match existing WNBA league hubs (leaders/standings): charcoal surfaces, muted labels, bold primary values. Not a marketing/landing layout.

## Backend

### Route

`GET /api/wnba/player/{player_id}`

Response shape:

```json
{
  "player_id": "1628932",
  "name": "A'ja Wilson",
  "position": "F",
  "team_name": "Las Vegas Aces",
  "team_abbrev": "LVA",
  "headshot_url": "https://...",
  "season": 2026,
  "averages": {
    "pts": "26.4",
    "reb": "10.1",
    "ast": "2.8",
    "fg_pct": "48.2",
    "fg3_pct": "33.1"
  },
  "games": [
    {
      "game_id": "1022600123",
      "game_date": "2026-07-28",
      "matchup": "LVA vs NYL",
      "min": "34",
      "pts": "28",
      "fg": "11-20",
      "three_pt": "1-3",
      "ft": "5-6",
      "reb": "12",
      "ast": "3",
      "to": "2",
      "stl": "1",
      "blk": "2"
    }
  ],
  "source_label": "stats.wnba.com"
}
```

### Upstream

| Need | stats.wnba.com endpoint |
| --- | --- |
| Identity + season avgs (PTS/REB/AST/FG%/3P%) | `leaguedashplayerstats` (PerGame, LeagueID=10); filter row by `PLAYER_ID` |
| Position / bio gaps | `commonplayerinfo` |
| Game log | `playergamelog` for current season; newest first |

- Shooting percentages in `averages` are display strings (one decimal), same spirit as leaders values (e.g. `"48.2"` meaning 48.2%).
- Made-attempt strings (`fg`, `three_pt`, `ft`) use `"m-a"` format.
- `headshot_url`: construct from known WNBA/NBA CDN pattern using `player_id`. Frontend treats broken images as placeholder (onError). Backend may still return the URL; null only if no pattern applies.
- `team_name`: prefer full name when available; otherwise fall back to abbrev-only display on the frontend.

### Errors and cache

| Case | Behavior |
| --- | --- |
| Unknown / no matching player | **404** |
| Upstream failure, no cache | **502** (or project-standard upstream error) |
| Upstream failure, fresh-enough cache | Return cached payload (same pattern as other WNBA services when applicable) |
| Cache TTL | ~10 minutes |

Frontend receives the **full** season `games` array in one response. “See more” is UI-only — no second request.

## Frontend

| Piece | Role |
| --- | --- |
| `LeaguePlayerPage` | Route page; fetch + compose header + games + attribution |
| `PlayerHeader` | Bio left, averages right |
| `PlayerRecentGames` | Table + expand/collapse |
| `useWnbaPlayer(playerId)` | Client for `GET /api/wnba/player/{id}` |
| `LeaderCategoryCard` | Name → `Link` to player route |

### States

| State | UI |
| --- | --- |
| Loading | Skeleton/placeholder for header + table |
| 404 | “Player not found” |
| Upstream / fetch error | Short error message; retry if existing league pages already offer it |
| Empty `games` | Header still renders; table empty-state copy (“No games yet”) |
| ≤5 games | No “See more” control |

## Testing

- **Backend:** Normalize fixtures for dash stats + game log + info → response shape; 404 when player missing; route smoke test.
- **Frontend:** Page renders header avgs and 5-row default; “See more” expands; Leaders name navigates to `/wnba/player/{id}`; broken headshot falls back to placeholder.

## Out of scope (v1)

- NBA player pages
- Links from prop picks, box score, starters, injuries
- Career / prior-season stats or season picker
- Props, predictions, or shot charts on the profile
- ESPN athlete ID mapping
- New LeagueSubnav “Player” item
