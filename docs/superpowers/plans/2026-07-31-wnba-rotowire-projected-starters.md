# WNBA RotoWire Projected Starters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For scheduled WNBA game detail, prefer live RotoWire expected starting fives (with ESPN roster jersey enrichment), falling back to ESPN last-game starters when RotoWire fails.

**Architecture:** Thin backend wrapper imports existing `WNBADailyLineups`, runs it in `asyncio.to_thread`, caches parsed lineups by ET date (~3 min). Enrich jerseys from ESPN team roster (cached ~10 min). `get_game_detail` tries RotoWire first for `scheduled` games; on miss, keeps today’s `lastFiveGames` prior-summary path.

**Tech Stack:** FastAPI, httpx, requests, BeautifulSoup4, pytest, existing `src/scrapers/rotowire_starters_scraper.py`

## Global Constraints

- Primary source: live scrape `https://www.rotowire.com/wnba/lineups.php` via existing `WNBADailyLineups`.
- UI note on success: `"RotoWire expected lineup"`; fallback note: `"from each team's last game"`.
- Jersey from ESPN roster name match; `jersey: null` if unmatched.
- Position: RotoWire `lineup__pos` first; else ESPN roster abbreviation; else null.
- All-or-nothing per RotoWire attempt: both sides need a usable five of 5, else ESPN fallback.
- RotoWire errors must not fail the detail response.
- Do not scrape RotoWire for live/halftime/final.
- Do not rewrite Airflow/`updateTeamInfo()` ML path beyond optional shared ordered-starter helper.
- No frontend layout changes (note comes from API).
- TDD: failing test before production code each task.
- Backend tests must not hit live RotoWire/ESPN network.

---

## File Structure

- Modify: `src/scrapers/rotowire_starters_scraper.py` — add ordered starters + position parser (keep `getDict` / `updateTeamInfo` compatible)
- Create: `backend/app/services/wnba_rotowire_lineups.py` — ET-date cache + async lookup by team abbrevs
- Create: `backend/app/services/wnba_espn_roster.py` — roster fetch + jersey/position map cache + name normalize
- Modify: `backend/app/services/wnba_game_detail.py` — prefer RotoWire projected starters before prior-game fan-out
- Modify: `backend/requirements.txt` — add `beautifulsoup4`, `requests`
- Modify: `backend/README.md` — document `PYTHONPATH` / import of `src`
- Create: `backend/tests/fixtures/rotowire_wnba_lineups_sea_atl.html` — minimal SEA@ATL card HTML
- Create: `backend/tests/fixtures/espn_wnba_roster_atl.json` — roster with Angel Reese `#5`
- Create: `backend/tests/fixtures/espn_wnba_roster_sea.json` — roster for away enrich
- Create: `backend/tests/test_wnba_rotowire_lineups.py`
- Create: `backend/tests/test_wnba_espn_roster.py`
- Modify: `backend/tests/test_wnba_game_detail_route.py` — RotoWire success + fallback cases
- Optional: `frontend` fixture note strings may stay as last-game; no required FE change

---

### Task 1: Ordered RotoWire starters with positions (scraper)

**Files:**
- Modify: `src/scrapers/rotowire_starters_scraper.py`
- Create: `backend/tests/fixtures/rotowire_wnba_lineups_sea_atl.html`
- Create: `backend/tests/test_wnba_rotowire_lineups.py` (parser section first; wrapper tests in Task 2)

**Interfaces:**
- Consumes: existing `DailyLineups` HTML parse helpers
- Produces:
  - `DailyLineups.expected_starters_by_abbr() -> dict[str, list[dict[str, str | None]]]`
    - each starter: `{"name": str, "position": str | None}`
    - DOM order preserved, max 5, title `"Very Likely To Play"`, skip `has-injury-status`
  - `getDict()` / `updateTeamInfo()` still work (sets of names unchanged)

- [ ] **Step 1: Write minimal HTML fixture**

Create `backend/tests/fixtures/rotowire_wnba_lineups_sea_atl.html` with one `div.lineup` card containing visit SEA + home ATL lists. Include ATL expected five in this order with `div.lineup__pos` + `a[title]`:

1. Allisha Gray G  
2. Jordin Canada G  
3. Rhyne Howard G  
4. Naz Hillmon F  
5. Angel Reese F  

SEA five can be Natisha Hiedeman, Jade Melbourne, Flau'jae Johnson, Awa Fam, Dominique Malonga (any valid five). Mirror real markup:

```html
<li class="lineup__player is-pct-play-100" title="Very Likely To Play">
  <div class="lineup__pos">F</div>
  <a href="/wnba/player/angel-reese-915" title="Angel Reese">Angel Reese</a>
</li>
```

Include `a.lineup__team.is-visit` / `.is-home` with `.lineup__abbr` text `SEA` / `ATL`.

- [ ] **Step 2: Write failing parser test**

In `backend/tests/test_wnba_rotowire_lineups.py`:

```python
from pathlib import Path
from src.scrapers.rotowire_starters_scraper import WNBADailyLineups

FIXTURES = Path(__file__).parent / "fixtures"


def test_expected_starters_by_abbr_preserves_order_and_positions(monkeypatch):
    html = (FIXTURES / "rotowire_wnba_lineups_sea_atl.html").read_text()

    class FakeLineups(WNBADailyLineups):
        def _get_soup(self):
            from bs4 import BeautifulSoup

            return BeautifulSoup(html, "html.parser")

    scraped = FakeLineups()
    by_abbr = scraped.expected_starters_by_abbr()
    atl = by_abbr["ATL"]
    assert [p["name"] for p in atl] == [
        "Allisha Gray",
        "Jordin Canada",
        "Rhyne Howard",
        "Naz Hillmon",
        "Angel Reese",
    ]
    assert atl[-1]["position"] == "F"
    assert "Madina Okot" not in [p["name"] for p in atl]
    assert len(by_abbr["SEA"]) == 5
```

- [ ] **Step 3: Run test to verify it fails**

Run:

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor && PYTHONPATH=.:backend python3 -m pytest backend/tests/test_wnba_rotowire_lineups.py::test_expected_starters_by_abbr_preserves_order_and_positions -v
```

Expected: FAIL (`expected_starters_by_abbr` missing) or import/deps error until Task 2 adds requirements — if `bs4` missing, install deps first from Task 2 Step 1 then re-run this red test.

- [ ] **Step 4: Implement `expected_starters_by_abbr`**

Add to `DailyLineups` in `src/scrapers/rotowire_starters_scraper.py`:

```python
@staticmethod
def _ordered_expected_starters(lineup_list) -> list[dict[str, str | None]]:
    if lineup_list is None:
        return []
    out: list[dict[str, str | None]] = []
    for item in lineup_list.find_all("li", {"title": "Very Likely To Play"}):
        classes = item.get("class") or []
        if "has-injury-status" in classes or not item.a:
            continue
        name = (item.a.get("title") or item.a.get_text(strip=True) or "").strip()
        if not name:
            continue
        pos_el = item.find(class_="lineup__pos")
        position = pos_el.get_text(strip=True) if pos_el else None
        out.append({"name": name, "position": position or None})
        if len(out) == 5:
            break
    return out


def expected_starters_by_abbr(self) -> dict[str, list[dict[str, str | None]]]:
    """Ordered expected fives keyed by team abbreviation (WNBA/NBA)."""
    if not self.data:
        self.getDict()
    result: dict[str, list[dict[str, str | None]]] = {}
    for matchup_el, parsed in zip(self._matchup_cards(), self.data):
        for side, list_cls in (
            ("away", "lineup__list is-visit"),
            ("home", "lineup__list is-home"),
        ):
            abbr = parsed[side].get("abbr")
            if not abbr:
                continue
            ul = matchup_el.find("ul", {"class": list_cls})
            starters = self._ordered_expected_starters(ul)
            if starters:
                result[abbr] = starters
    return result
```

Prefer a cleaner approach if `zip` is fragile: re-walk cards inside `expected_starters_by_abbr` the same way `getDict` does (duplicate loop is OK; keep ML `getDict` untouched).

- [ ] **Step 5: Run test to verify it passes**

Run the same pytest command as Step 3. Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/scrapers/rotowire_starters_scraper.py \
  backend/tests/fixtures/rotowire_wnba_lineups_sea_atl.html \
  backend/tests/test_wnba_rotowire_lineups.py
git commit -m "feat: parse ordered RotoWire WNBA starters with positions"
```

---

### Task 2: Backend deps + async RotoWire cache wrapper

**Files:**
- Modify: `backend/requirements.txt`
- Modify: `backend/README.md`
- Create: `backend/app/services/wnba_rotowire_lineups.py`
- Modify: `backend/tests/test_wnba_rotowire_lineups.py`

**Interfaces:**
- Consumes: `WNBADailyLineups.expected_starters_by_abbr()`
- Produces:
  - `async def get_rotowire_starters_for_matchup(*, away_abbr: str, home_abbr: str) -> dict[str, list[dict[str, str | None]]] | None`
  - returns `{"away": [...5...], "home": [...5...]}` or `None`
  - `clear_rotowire_lineups_cache() -> None`
  - TTL 180 seconds; cache key = ET calendar date string `YYYY-MM-DD`

- [ ] **Step 1: Add dependencies**

Append to `backend/requirements.txt`:

```
beautifulsoup4>=4.12.0
requests>=2.31.0
```

Install into the env used for tests:

```bash
pip install 'beautifulsoup4>=4.12.0' 'requests>=2.31.0'
```

- [ ] **Step 2: Document PYTHONPATH**

In `backend/README.md` Run section, replace the uvicorn snippet with:

```bash
# from repo root — root must be on PYTHONPATH so `src.scrapers` imports work
PYTHONPATH=.:backend uvicorn app.main:app --reload --port 8000
```

Note that WNBA game detail may call ESPN + RotoWire for scheduled projected starters.

- [ ] **Step 3: Write failing wrapper tests**

```python
import asyncio
from pathlib import Path
from unittest.mock import patch

import app.services.wnba_rotowire_lineups as rw


FIXTURES = Path(__file__).parent / "fixtures"


def test_get_rotowire_starters_for_matchup_sea_atl(monkeypatch):
    rw.clear_rotowire_lineups_cache()
    html = (FIXTURES / "rotowire_wnba_lineups_sea_atl.html").read_text()

    def fake_fetch():
        from bs4 import BeautifulSoup
        from src.scrapers.rotowire_starters_scraper import WNBADailyLineups

        class Fake(WNBADailyLineups):
            def _get_soup(self):
                return BeautifulSoup(html, "html.parser")

        return Fake().expected_starters_by_abbr()

    monkeypatch.setattr(rw, "_scrape_starters_by_abbr", fake_fetch)
    result = asyncio.run(
        rw.get_rotowire_starters_for_matchup(away_abbr="SEA", home_abbr="ATL")
    )
    assert result is not None
    assert result["home"][-1]["name"] == "Angel Reese"
    assert len(result["away"]) == 5


def test_get_rotowire_starters_returns_none_if_team_missing(monkeypatch):
    rw.clear_rotowire_lineups_cache()
    monkeypatch.setattr(rw, "_scrape_starters_by_abbr", lambda: {"ATL": [{"name": "X", "position": "F"}] * 5})
    result = asyncio.run(
        rw.get_rotowire_starters_for_matchup(away_abbr="SEA", home_abbr="ATL")
    )
    assert result is None


def test_rotowire_cache_reuses_scrape(monkeypatch):
    rw.clear_rotowire_lineups_cache()
    calls = {"n": 0}

    def fake_fetch():
        calls["n"] += 1
        five = [{"name": f"P{i}", "position": "G"} for i in range(5)]
        return {"SEA": five, "ATL": five}

    monkeypatch.setattr(rw, "_scrape_starters_by_abbr", fake_fetch)
    asyncio.run(rw.get_rotowire_starters_for_matchup(away_abbr="SEA", home_abbr="ATL"))
    asyncio.run(rw.get_rotowire_starters_for_matchup(away_abbr="SEA", home_abbr="ATL"))
    assert calls["n"] == 1
```

- [ ] **Step 4: Run tests — expect fail**

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor && PYTHONPATH=.:backend python3 -m pytest backend/tests/test_wnba_rotowire_lineups.py -v
```

Expected: FAIL on missing module/`get_rotowire_starters_for_matchup`.

- [ ] **Step 5: Implement wrapper**

Create `backend/app/services/wnba_rotowire_lineups.py`:

```python
from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
ROTOWIRE_TTL_SECONDS = 180

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_cache: dict[str, object] = {"date_et": None, "expires_at": 0.0, "by_abbr": None}


def clear_rotowire_lineups_cache() -> None:
    _cache.update({"date_et": None, "expires_at": 0.0, "by_abbr": None})


def _scrape_starters_by_abbr() -> dict[str, list[dict[str, str | None]]]:
    from src.scrapers.rotowire_starters_scraper import WNBADailyLineups

    return WNBADailyLineups().expected_starters_by_abbr()


def _cached_by_abbr() -> dict[str, list[dict[str, str | None]]] | None:
    now = time.time()
    date_et = datetime.now(ET).strftime("%Y-%m-%d")
    if (
        _cache["by_abbr"] is not None
        and _cache["date_et"] == date_et
        and float(_cache["expires_at"]) > now
    ):
        return _cache["by_abbr"]  # type: ignore[return-value]
    try:
        by_abbr = _scrape_starters_by_abbr()
    except Exception:
        return None
    _cache["date_et"] = date_et
    _cache["expires_at"] = now + ROTOWIRE_TTL_SECONDS
    _cache["by_abbr"] = by_abbr
    return by_abbr


async def get_rotowire_starters_for_matchup(
    *, away_abbr: str, home_abbr: str
) -> dict[str, list[dict[str, str | None]]] | None:
    by_abbr = await asyncio.to_thread(_cached_by_abbr)
    if not by_abbr:
        return None
    away = by_abbr.get(away_abbr.upper())
    home = by_abbr.get(home_abbr.upper())
    if not away or not home or len(away) != 5 or len(home) != 5:
        return None
    return {"away": away, "home": home}
```

- [ ] **Step 6: Run tests — expect pass**

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor && PYTHONPATH=.:backend python3 -m pytest backend/tests/test_wnba_rotowire_lineups.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/requirements.txt backend/README.md \
  backend/app/services/wnba_rotowire_lineups.py \
  backend/tests/test_wnba_rotowire_lineups.py
git commit -m "feat: add cached async RotoWire WNBA lineups helper"
```

---

### Task 3: ESPN roster jersey enrichment

**Files:**
- Create: `backend/app/services/wnba_espn_roster.py`
- Create: `backend/tests/fixtures/espn_wnba_roster_atl.json`
- Create: `backend/tests/fixtures/espn_wnba_roster_sea.json`
- Create: `backend/tests/test_wnba_espn_roster.py`

**Interfaces:**
- Consumes: ESPN `GET https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/teams/{team_id}/roster`
- Produces:
  - `async def fetch_espn_roster(team_id: str) -> dict`
  - `def roster_player_index(payload: dict) -> dict[str, dict[str, str | None]]`  
    keyed by normalized name → `{"jersey": str | None, "position": str | None}`
  - `def enrich_starters(starters: list[dict], index: dict) -> list[GameDetailStarter]`  
    or return plain dicts that game_detail converts
  - `async def get_roster_index(team_id: str) -> dict[str, dict[str, str | None]]` with 600s TTL cache
  - `clear_roster_cache() -> None`
  - `norm_player_name(name: str) -> str` (NFKD + casefold, same idea as `team_info._norm_player_name`)

- [ ] **Step 1: Add roster fixtures**

`espn_wnba_roster_atl.json` minimal:

```json
{
  "athletes": [
    {
      "displayName": "Angel Reese",
      "jersey": "5",
      "position": { "abbreviation": "F" }
    },
    {
      "displayName": "Naz Hillmon",
      "jersey": "00",
      "position": { "abbreviation": "F" }
    },
    {
      "displayName": "Allisha Gray",
      "jersey": "15",
      "position": { "abbreviation": "G" }
    },
    {
      "displayName": "Jordin Canada",
      "jersey": "3",
      "position": { "abbreviation": "G" }
    },
    {
      "displayName": "Rhyne Howard",
      "jersey": "10",
      "position": { "abbreviation": "G" }
    }
  ]
}
```

SEA fixture: five players matching the HTML fixture names with any jerseys.

- [ ] **Step 2: Write failing tests**

```python
import json
from pathlib import Path

from app.services.wnba_espn_roster import (
    enrich_starters,
    norm_player_name,
    roster_player_index,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_roster_index_and_enrich_angel_reese_jersey():
    payload = json.loads((FIXTURES / "espn_wnba_roster_atl.json").read_text())
    index = roster_player_index(payload)
    assert index[norm_player_name("Angel Reese")]["jersey"] == "5"
    starters = [
        {"name": "Allisha Gray", "position": "G"},
        {"name": "Jordin Canada", "position": "G"},
        {"name": "Rhyne Howard", "position": "G"},
        {"name": "Naz Hillmon", "position": "F"},
        {"name": "Angel Reese", "position": "F"},
    ]
    enriched = enrich_starters(starters, index)
    assert enriched[-1].name == "Angel Reese"
    assert enriched[-1].jersey == "5"
    assert enriched[-1].position == "F"


def test_enrich_starters_null_jersey_when_unmatched():
    enriched = enrich_starters(
        [{"name": "Unknown Player", "position": "G"}],
        {},
    )
    assert enriched[0].jersey is None
    assert enriched[0].name == "Unknown Player"
```

- [ ] **Step 3: Run — expect fail**

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor && PYTHONPATH=.:backend python3 -m pytest backend/tests/test_wnba_espn_roster.py -v
```

Expected: FAIL (module missing).

- [ ] **Step 4: Implement roster helper**

Create `backend/app/services/wnba_espn_roster.py` with:

- `httpx` async fetch to  
  `https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/teams/{team_id}/roster`
- timeout 8s (match game detail)
- `roster_player_index` walks `athletes[]` (top-level list as in live ESPN)
- `enrich_starters` builds `GameDetailStarter` list; prefer RotoWire position when non-empty, else roster position
- in-memory `_roster_cache: dict[str, {"expires_at": float, "index": dict}]`

- [ ] **Step 5: Run — expect pass**

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor && PYTHONPATH=.:backend python3 -m pytest backend/tests/test_wnba_espn_roster.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/wnba_espn_roster.py \
  backend/tests/test_wnba_espn_roster.py \
  backend/tests/fixtures/espn_wnba_roster_atl.json \
  backend/tests/fixtures/espn_wnba_roster_sea.json
git commit -m "feat: enrich RotoWire starters with ESPN roster jerseys"
```

---

### Task 4: Wire RotoWire into `get_game_detail` with ESPN fallback

**Files:**
- Modify: `backend/app/services/wnba_game_detail.py`
- Modify: `backend/tests/test_wnba_game_detail_route.py`
- Modify: `backend/tests/test_wnba_game_detail_normalize.py` (keep prior-summary unit tests unchanged)

**Interfaces:**
- Consumes:
  - `get_rotowire_starters_for_matchup(away_abbr=..., home_abbr=...)`
  - `get_roster_index(team_id)`
  - existing `_fetch_prior_game_summaries` / `_normalize_projected_starters`
- Produces: `WnbaGameDetail.projected_starters` with RotoWire note when primary path succeeds

- [ ] **Step 1: Write failing integration test**

In `backend/tests/test_wnba_game_detail_route.py` (or new test file):

```python
def test_get_game_detail_prefers_rotowire_starters(monkeypatch):
    svc.clear_game_detail_cache()
    scheduled = load_fixture("espn_wnba_summary_scheduled_preview.json")
    # Force abbrevs SEA/ATL on competitors for matchup keying
    # (mutate fixture competitors abbreviations + ids as needed)

    async def fake_espn(event_id: str):
        return scheduled

    async def fake_rw(*, away_abbr: str, home_abbr: str):
        return {
            "away": [{"name": f"Away{i}", "position": "G"} for i in range(5)],
            "home": [
                {"name": "Allisha Gray", "position": "G"},
                {"name": "Jordin Canada", "position": "G"},
                {"name": "Rhyne Howard", "position": "G"},
                {"name": "Naz Hillmon", "position": "F"},
                {"name": "Angel Reese", "position": "F"},
            ],
        }

    async def fake_roster(team_id: str):
        if team_id.endswith("home") or team_id == "home1":
            return roster_player_index(
                json.loads((FIXTURES / "espn_wnba_roster_atl.json").read_text())
            )
        return {}

    monkeypatch.setattr(svc, "fetch_espn_summary", fake_espn)
    monkeypatch.setattr(
        "app.services.wnba_rotowire_lineups.get_rotowire_starters_for_matchup",
        fake_rw,
    )
    monkeypatch.setattr(
        "app.services.wnba_espn_roster.get_roster_index",
        fake_roster,
    )
    # Also patch wherever get_game_detail imports the symbols

    detail = asyncio.run(svc.get_game_detail("401857099"))
    assert detail.projected_starters is not None
    assert detail.projected_starters.note == "RotoWire expected lineup"
    assert detail.projected_starters.home[-1].name == "Angel Reese"
    assert detail.projected_starters.home[-1].jersey == "5"


def test_get_game_detail_falls_back_to_prior_starters_when_rotowire_misses(monkeypatch):
    svc.clear_game_detail_cache()
    scheduled = load_fixture("espn_wnba_summary_scheduled_preview.json")
    prior_away = load_fixture("espn_wnba_summary_prior_away.json")
    prior_home = load_fixture("espn_wnba_summary_prior_home.json")

    async def fake_fetch(event_id: str):
        return {
            "401857099": scheduled,
            "401857069": prior_away,
            "401857060": prior_home,
        }[event_id]

    async def fake_rw(**kwargs):
        return None

    monkeypatch.setattr(svc, "fetch_espn_summary", fake_fetch)
    monkeypatch.setattr(
        "app.services.wnba_rotowire_lineups.get_rotowire_starters_for_matchup",
        fake_rw,
    )
    detail = asyncio.run(svc.get_game_detail("401857099"))
    assert detail.projected_starters is not None
    assert detail.projected_starters.note == "from each team's last game"
```

Adjust fixture event ids to match `lastFiveGames` in `espn_wnba_summary_scheduled_preview.json` (existing test already maps `401857069` / `401857060`).

- [ ] **Step 2: Run — expect fail**

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor && PYTHONPATH=.:backend python3 -m pytest backend/tests/test_wnba_game_detail_route.py::test_get_game_detail_prefers_rotowire_starters -v
```

Expected: FAIL (still last-game only / Angel Reese missing).

- [ ] **Step 3: Wire into `get_game_detail`**

In `backend/app/services/wnba_game_detail.py` inside the scheduled branch:

1. Read `away.abbrev` / `home.abbrev` and team ids from the summary (after status known).
2. `rw = await get_rotowire_starters_for_matchup(away_abbr=..., home_abbr=...)`.
3. If `rw` is not None:
   - `away_idx, home_idx = await asyncio.gather(get_roster_index(away_id), get_roster_index(home_id))`  
     (roster failures → empty index, still OK)
   - Build `GameDetailProjectedStarters(note="RotoWire expected lineup", away=enrich_starters(...), home=enrich_starters(...))`
   - Pass into `normalize_espn_summary` as a new optional arg `projected_starters=` **or** set on the detail after normalize.
4. Else: keep existing `prior_game_summaries = await _fetch_prior_game_summaries(...)`.

Cleanest API: extend `normalize_espn_summary(..., projected_starters: GameDetailProjectedStarters | None = None)` so when provided it wins over prior-summary normalization:

```python
if projected_starters is not None:
    resolved = projected_starters
else:
    resolved = _normalize_projected_starters(...)
```

Do **not** call prior-game fan-out when RotoWire succeeded.

- [ ] **Step 4: Run route + normalize tests**

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor && PYTHONPATH=.:backend python3 -m pytest \
  backend/tests/test_wnba_game_detail_route.py \
  backend/tests/test_wnba_game_detail_normalize.py \
  backend/tests/test_wnba_rotowire_lineups.py \
  backend/tests/test_wnba_espn_roster.py -v
```

Expected: PASS (existing prior-summary normalize tests still green).

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/wnba_game_detail.py \
  backend/tests/test_wnba_game_detail_route.py
git commit -m "feat: prefer RotoWire projected starters on scheduled WNBA games"
```

---

### Task 5: Smoke verification + docs touch-up

**Files:**
- Modify only if needed: `docs/superpowers/specs/2026-07-30-wnba-scheduled-matchup-preview-design.md` is historical — do **not** rewrite; the new 2026-07-31 spec is source of truth for starters.

- [ ] **Step 1: Full backend game-detail related suite**

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor && PYTHONPATH=.:backend python3 -m pytest \
  backend/tests/test_wnba_game_detail_normalize.py \
  backend/tests/test_wnba_game_detail_route.py \
  backend/tests/test_wnba_rotowire_lineups.py \
  backend/tests/test_wnba_espn_roster.py -v
```

Expected: all PASS.

- [ ] **Step 2: Manual smoke (optional, network)**

```bash
cd /Users/alexgonzalez/Documents/NBA-Prop-Predictor && PYTHONPATH=.:backend uvicorn app.main:app --reload --port 8000
curl -sS 'http://127.0.0.1:8000/api/wnba/games/401857102' | python3 -m json.tool | head
```

Confirm `projected_starters.note` is `RotoWire expected lineup` and Atlanta includes Angel Reese with jersey `5` when RotoWire/ESPN are reachable.

- [ ] **Step 3: Commit only if smoke prompted doc/script fixes**

Otherwise no commit.

---

## Self-review (plan vs spec)

| Spec requirement | Task |
| --- | --- |
| Live RotoWire via existing scraper | 1–2 |
| Ordered starters + positions | 1 |
| Async `to_thread` + 3 min ET-date cache | 2 |
| ESPN roster jersey enrich + 10 min cache | 3 |
| Prefer RotoWire; fallback last-game | 4 |
| Notes: RotoWire vs last-game | 4 |
| `beautifulsoup4` + `requests` + PYTHONPATH/`src` import | 2 |
| No live/final scrape | 4 (scheduled gate only) |
| Tests without live network | 1–4 fixtures/mocks |
| No FE layout change | — (API note only) |
| ML `updateTeamInfo` untouched | 1 (additive method only) |

No TBD placeholders. Types use existing `GameDetailStarter` / `GameDetailProjectedStarters`.
