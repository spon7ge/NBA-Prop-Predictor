# WNBA projected starters from RotoWire

Date: 2026-07-31  
Status: Approved for planning

## Goal

For **scheduled** WNBA game detail, show **RotoWire expected starting fives** (e.g. Angel Reese for Atlanta) instead of ESPN last-game boxscore starters (which incorrectly surface one-off starters like Madina Okot when a regular starter sat out). Keep ESPN last-game starters as a **fallback** when RotoWire is unavailable.

Live / halftime / final behavior is unchanged (`projected_starters` remains null).

## Decisions

| Topic | Choice |
| --- | --- |
| Primary source | Live scrape [RotoWire WNBA lineups](https://www.rotowire.com/wnba/lineups.php) |
| Integration style | Import existing `WNBADailyLineups` from `src/scrapers/rotowire_starters_scraper.py` (run sync work in `asyncio.to_thread`) |
| Jersey numbers | Enrich from ESPN team roster by normalized name match; `jersey: null` if no match |
| Position | Prefer RotoWire lineup position (G/F/C) when parseable; else ESPN roster position; else null |
| Failure mode | Fall back to existing ESPN `lastFiveGames` → prior boxscore starters |
| Page cache | In-memory parsed RotoWire lineups, TTL ~3 minutes, keyed by ET calendar date |
| Roster cache | In-memory ESPN roster jersey/position maps by team id, short TTL (~10 minutes) |
| UI note (RotoWire) | `"RotoWire expected lineup"` |
| UI note (fallback) | `"from each team's last game"` (unchanged) |
| Frontend layout | Unchanged beyond the note string |

## Architecture

```
GET /api/wnba/games/{espnEventId}
        │
        ▼
  ESPN summary (existing)
        │
        ▼
  status == scheduled?
        │
        ├─ no → projected_starters = null (as today)
        │
        └─ yes
              │
              ├─ RotoWire WNBADailyLineups (cached by ET date)
              │     match away/home abbrev → ordered expected five
              │     enrich jersey (+ position fallback) via ESPN roster
              │     on success → projected_starters (RotoWire note)
              │
              └─ on scrape / parse / match / incomplete five failure
                    → existing lastFiveGames prior-summary fan-out
                    → projected_starters (last-game note) or null
```

### Components

| Unit | Responsibility |
| --- | --- |
| `WNBADailyLineups` (existing) | Fetch + parse RotoWire HTML into per-team expected starters |
| `wnba_rotowire_lineups` helper (new, thin backend wrapper) | Async entrypoint, date-keyed cache, abbrev → starters lookup for a matchup |
| ESPN roster fetch (new small helper) | `GET .../wnba/teams/{id}/roster` → name → `{jersey, position}` map with TTL cache |
| `get_game_detail` / normalize (existing) | Prefer RotoWire projected starters; else prior-game path |

Do **not** rewrite the Airflow/ML `team_info.updateTeamInfo()` path in this change. Optionally extend `WNBADailyLineups` parse slightly so the API can get **ordered** starters with **positions** (today `getDict()` stores unordered name sets for ML). Keep ML `updateTeamInfo()` behavior compatible.

### Name matching

- Prefer RotoWire `a[title]` full name when present (page link text is often abbreviated).
- Normalize with the same accent/casefold approach as `src/utils/team_info._norm_player_name` for roster jersey lookup.
- If roster match fails, still return the starter with `jersey: null`.

### Ordering and caps

- Preserve RotoWire DOM order for the expected lineup.
- Cap at 5 per team.
- All-or-nothing across teams for the RotoWire path: if either side lacks a usable five, treat as RotoWire miss and try ESPN fallback.

## Data schema

No new response fields. Existing `projected_starters` shape stays:

```json
{
  "projected_starters": {
    "note": "RotoWire expected lineup",
    "away": [
      { "jersey": "14", "name": "Dominique Malonga", "position": "C" }
    ],
    "home": [
      { "jersey": "5", "name": "Angel Reese", "position": "F" }
    ]
  }
}
```

## Backend rules

1. Only attempt RotoWire / ESPN-starter fan-out when detail `status == "scheduled"`.
2. Run `WNBADailyLineups` off the event loop (`asyncio.to_thread`); never block the loop on `requests`.
3. Cache the parsed RotoWire payload for ~3 minutes per ET date so concurrent scheduled game detail requests share one scrape.
4. Cache ESPN roster maps by team id (~10 minutes).
5. Add `beautifulsoup4` and `requests` to `backend/requirements.txt` (already used by ETL/Airflow).
6. Backend must import `src.scrapers.rotowire_starters_scraper`. Today local uvicorn often uses `PYTHONPATH=backend`, which cannot see `src/`. Fix by including the repo root on `PYTHONPATH` (e.g. `PYTHONPATH=.:backend` or project-root only with backend as a package path) and update any run docs/scripts that start the API. Prefer env/`sys.path` bootstrap only if unavoidable.
7. RotoWire errors must not fail the detail response; fall back to ESPN last-game starters, then `null` if that also fails.
8. Do not scrape RotoWire for live/halftime/final.

## UI

`ProjectedStarters` already renders optional jersey and position. No layout change required; note string comes from the API.

## File layout

```
src/scrapers/rotowire_starters_scraper.py     # optional: ordered starters + position for API use
backend/app/services/wnba_rotowire_lineups.py # cache + async wrapper + matchup lookup
backend/app/services/wnba_espn_roster.py      # roster fetch + jersey map cache (or colocated)
backend/app/services/wnba_game_detail.py      # prefer RotoWire, else last-game
backend/requirements.txt                      # + beautifulsoup4, requests
backend/tests/fixtures/rotowire_wnba_lineups.html
backend/tests/fixtures/espn_wnba_roster_atl.json  # (and/or sea)
backend/tests/test_wnba_rotowire_lineups.py
backend/tests/test_wnba_game_detail_*.py      # update starter expectations
```

## Testing

- HTML fixture for SEA @ ATL expected lineups → ATL includes Angel Reese (not Madina Okot); positions present when in fixture.
- Roster fixture enrich → `#5` on Angel Reese when roster has jersey `5`.
- Unmatched name → starter still returned, `jersey: null`.
- Mocked RotoWire failure / missing team → ESPN prior-game path still populates starters (existing fixtures).
- Live status → no RotoWire call, `projected_starters` null.
- Cache: two scheduled detail fetches within TTL → single RotoWire scrape.

## Out of scope

- NBA RotoWire lineups on game detail
- Showing GTD / OUT / “may not play” inside the starters panel (injury card remains the injury surface)
- Changing Airflow scrape schedule or `team_info.py` write format beyond optional shared parse helpers
- Frontend calling RotoWire or ESPN directly
- Confirmed-vs-expected lineup status badges

## Success criteria

- Scheduled SEA @ ATL (or fixture equivalent) shows RotoWire Atlanta five including Angel Reese with jersey when roster enrich works.
- When RotoWire is down, UI still shows last-game starters with the existing note.
- Live/final game detail unchanged.
- Backend tests pass without live network access to RotoWire.
