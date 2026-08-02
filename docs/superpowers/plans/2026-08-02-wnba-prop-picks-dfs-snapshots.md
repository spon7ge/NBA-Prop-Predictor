# WNBA Prop Picks DFS Snapshots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `/wnba/prop_picks` DFS-first: show latest Supabase PrizePicks (standard) and Underdog main lines, then attach matching US sportsbook quotes from Parlay; drop sportsbook-only rows.

**Architecture:** Extend `parlay_props.get_today_props`: keep market allowlist + Parlay main-line normalize for US books only; seed the board from Supabase PP∪UD via `attach_dfs_snapshots`; match sportsbook lines to the DFS target line. Same `GET /api/wnba/props/today` and `WnbaPropLine` shape.

**Tech Stack:** FastAPI, Pydantic, pytest, React, Vitest, existing `odds_snapshots` + Parlay client

**Spec:** `docs/superpowers/specs/2026-08-02-wnba-prop-picks-dfs-snapshots-design.md`

## Global Constraints

- Endpoint remains `GET /api/wnba/props/today`
- DFS source: `fetch_latest_prizepicks` / `fetch_latest_underdog` only (not Parlay DFS bookmakers)
- PrizePicks: `odds_type == "standard"` only
- Underdog: one quote per `(norm_player, stat_key, side)`
- Board keep rule: row must have `prizepicks` and/or `underdog`
- US books attached: `fanduel`, `draftkings`, `caesars`, `betmgm`, `pinnacle`, `bet365`, `novig` only
- Never invent DFS lines from Parlay; empty snapshots → `props: []`
- Parlay down + snapshots OK → DFS rows with blank US cells
- Missing `PARLAY_API_KEY` → existing error payload (no change)
- Include in-progress market allowlist (drop milestone / `*_alt`)

## File structure

| File | Responsibility |
| --- | --- |
| `backend/app/services/prop_stat_keys.py` | Canonical stat keys + aliases for PP / UD / Parlay |
| `backend/app/services/dfs_attach.py` | Seed DFS board + attach closest US quotes |
| `backend/app/services/parlay_props.py` | Orchestrate fetch → normalize US → attach DFS; strip Parlay PP/UD |
| `backend/tests/test_prop_stat_keys.py` | Stat key mapping tests |
| `backend/tests/test_dfs_attach.py` | Attach / filter behavior tests |
| `backend/tests/test_parlay_props.py` | Pipeline + allowlist + snapshot wiring tests |
| `frontend/src/pages/LeaguePropPicksPage.tsx` | Default Book filter = PrizePicks + Underdog |
| `frontend/src/pages/LeaguePropPicksPage.test.tsx` (or filters test) | Default books assertion |

---

### Task 1: Canonical stat keys

**Files:**
- Create: `backend/app/services/prop_stat_keys.py`
- Create: `backend/tests/test_prop_stat_keys.py`

**Interfaces:**
- Produces: `canonical_stat_key_from_pp(stat_type: str) -> str | None`
- Produces: `canonical_stat_key_from_ud(stat_name: str) -> str | None`
- Produces: `canonical_stat_key_from_parlay_market(market_key: str) -> str | None`
- Produces: `display_stat_label(stat_key: str, fallback: str | None = None) -> str`
- Note: return `None` when the DFS stat is intentionally unmatched to Parlay (Fantasy Score, Combo, OReb/DReb). Callers still may keep the DFS row using a raw fallback key prefixed for display-only rows.

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_prop_stat_keys.py
from app.services.prop_stat_keys import (
    canonical_stat_key_from_pp,
    canonical_stat_key_from_ud,
    canonical_stat_key_from_parlay_market,
)

def test_pp_core_stats():
    assert canonical_stat_key_from_pp("Points") == "points"
    assert canonical_stat_key_from_pp("Pts+Rebs+Asts") == "pts_rebs_asts"
    assert canonical_stat_key_from_pp("3-PT Made") == "threes"
    assert canonical_stat_key_from_pp("Pts+Rebs") == "pts_rebs"

def test_pp_unmatched_returns_none():
    assert canonical_stat_key_from_pp("Fantasy Score") is None
    assert canonical_stat_key_from_pp("Points (Combo)") is None
    assert canonical_stat_key_from_pp("Defensive Rebounds") is None

def test_ud_core_stats():
    assert canonical_stat_key_from_ud("points") == "points"
    assert canonical_stat_key_from_ud("three_points_made") == "threes"
    assert canonical_stat_key_from_ud("pts_rebs_asts") == "pts_rebs_asts"

def test_parlay_markets():
    assert canonical_stat_key_from_parlay_market("player_points") == "points"
    assert canonical_stat_key_from_parlay_market("player_threes") == "threes"
    assert canonical_stat_key_from_parlay_market("player_three_pointers_made") == "threes"
    assert canonical_stat_key_from_parlay_market("player_pra") == "pts_rebs_asts"
    assert canonical_stat_key_from_parlay_market("player_points_rebounds_assists") == "pts_rebs_asts"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_prop_stat_keys.py -v`  
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `prop_stat_keys.py`**

```python
# backend/app/services/prop_stat_keys.py
from __future__ import annotations

_PP_ALIASES: dict[str, str] = {
    "points": "points",
    "rebounds": "rebounds",
    "assists": "assists",
    "3-pt_made": "threes",
    "3_pt_made": "threes",
    "pts_rebs": "pts_rebs",
    "pts_asts": "pts_asts",
    "rebs_asts": "rebs_asts",
    "pts_rebs_asts": "pts_rebs_asts",
}

_UD_ALIASES: dict[str, str] = {
    "points": "points",
    "rebounds": "rebounds",
    "assists": "assists",
    "three_points_made": "threes",
    "pts_rebs": "pts_rebs",
    "pts_asts": "pts_asts",
    "rebs_asts": "rebs_asts",
    "pts_rebs_asts": "pts_rebs_asts",
}

_PARLAY_ALIASES: dict[str, str] = {
    "player_points": "points",
    "player_rebounds": "rebounds",
    "player_assists": "assists",
    "player_threes": "threes",
    "player_three_pointers": "threes",
    "player_three_pointers_made": "threes",
    "player_pts_rebs": "pts_rebs",
    "player_points_rebounds": "pts_rebs",
    "player_pts_asts": "pts_asts",
    "player_points_assists": "pts_asts",
    "player_rebs_asts": "rebs_asts",
    "player_assists_rebounds": "rebs_asts",
    "player_pra": "pts_rebs_asts",
    "player_pts_rebs_asts": "pts_rebs_asts",
    "player_points_rebounds_assists": "pts_rebs_asts",
}

_LABELS: dict[str, str] = {
    "points": "Points",
    "rebounds": "Rebounds",
    "assists": "Assists",
    "threes": "3-PT Made",
    "pts_rebs": "Pts+Rebs",
    "pts_asts": "Pts+Asts",
    "rebs_asts": "Rebs+Asts",
    "pts_rebs_asts": "Pts+Rebs+Asts",
}


def _norm_pp(stat_type: str) -> str:
    return stat_type.strip().lower().replace(" ", "_").replace("+", "_")


def canonical_stat_key_from_pp(stat_type: str) -> str | None:
    return _PP_ALIASES.get(_norm_pp(stat_type))


def canonical_stat_key_from_ud(stat_name: str) -> str | None:
    return _UD_ALIASES.get(stat_name.strip().lower().replace(" ", "_"))


def canonical_stat_key_from_parlay_market(market_key: str) -> str | None:
    return _PARLAY_ALIASES.get(market_key.strip().lower())


def display_stat_label(stat_key: str, fallback: str | None = None) -> str:
    return _LABELS.get(stat_key) or fallback or stat_key.replace("_", " ").title()
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `cd backend && python -m pytest tests/test_prop_stat_keys.py -v`

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/prop_stat_keys.py backend/tests/test_prop_stat_keys.py
git commit -m "feat: add canonical prop stat keys for PP/UD/Parlay"
```

---

### Task 2: `attach_dfs_snapshots` (DFS-first merge)

**Files:**
- Create: `backend/app/services/dfs_attach.py`
- Create: `backend/tests/test_dfs_attach.py`

**Interfaces:**
- Consumes: Task 1 stat key helpers; `WnbaPropLine`, `WnbaPropBookQuote`, `PROP_SPORTSBOOKS`; `norm_player_name`
- Produces: `US_PROP_SPORTSBOOKS: tuple[str, ...]`
- Produces: `attach_dfs_snapshots(sportsbook_props: list[WnbaPropLine], pp_rows: list[dict], ud_rows: list[dict], player_teams: dict | None = None) -> list[WnbaPropLine]`

**Behavior checklist (must match spec):**
1. Ignore PP rows where `odds_type` (lower) ≠ `standard`
2. PP seeds both `over` and `under` with same line / `odds_american=None`
3. UD seeds its side only; last-write or first-write is fine if duplicates (prefer first non-null line)
4. Unmapped PP/UD stats still create rows (use raw display label; `market_type` = `prizepicks:{stat}` or `underdog:{stat}`; no US attach possible)
5. Index sportsbook quotes from `sportsbook_props` by `(norm_player, canonical_stat_key, side)` → per US book list of quotes
6. Target line = PP line if present else UD line; prefer exact match to either DFS line; else closest to PP (or UD if no PP)
7. Clear/ignore any incoming `prizepicks`/`underdog`/`betr`/`sleeper`/`pick6` on sportsbook props
8. Emit only buckets with PP and/or UD; sort player / market_type / side

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_dfs_attach.py
from app.schemas.wnba_props import WnbaPropBookQuote, WnbaPropLine
from app.services.dfs_attach import attach_dfs_snapshots

def _sb(player: str, market: str, side: str, **books) -> WnbaPropLine:
    return WnbaPropLine(
        player_name=player,
        stat=market.replace("player_", "").replace("_", " ").title(),
        market_type=market,
        side=side,
        **books,
    )

def test_drops_demon_and_keeps_standard_pp():
    props = attach_dfs_snapshots(
        [],
        [
            {"player_name": "A", "stat_type": "Points", "line_score": 19.5, "odds_type": "demon"},
            {"player_name": "A", "stat_type": "Points", "line_score": 18.5, "odds_type": "standard"},
        ],
        [],
    )
    assert len(props) == 2  # over+under
    assert all(p.prizepicks and p.prizepicks.line == 18.5 for p in props)

def test_pp_only_and_ud_only_kept():
    pp = attach_dfs_snapshots([], [{"player_name": "A", "stat_type": "Points", "line_score": 10.5, "odds_type": "standard"}], [])
    assert {p.side for p in pp} == {"over", "under"}
    ud = attach_dfs_snapshots([], [], [{"player_name": "B", "stat_name": "assists", "line_score": 5.5, "side": "over", "american_price": -110}])
    assert len(ud) == 1 and ud[0].underdog.line == 5.5

def test_sportsbook_only_dropped():
    sb = [_sb("A", "player_points", "over", fanduel=WnbaPropBookQuote(line=20.5, odds_american=-110))]
    assert attach_dfs_snapshots(sb, [], []) == []

def test_exact_dfs_line_wins_over_farther_alt():
    sb = [
        _sb("Caitlin Clark", "player_points", "over",
            fanduel=WnbaPropBookQuote(line=19.5, odds_american=-110),
            draftkings=WnbaPropBookQuote(line=20.5, odds_american=-105)),
    ]
    # Simulate two FD alts by two rows same book different lines via index builder —
    # sportsbook_props already collapsed per book; pass one line per book.
    # Closest: FD 19.5 matches PP 19.5; DK 20.5 kept as only DK quote.
    out = attach_dfs_snapshots(
        sb,
        [{"player_name": "Caitlin Clark", "stat_type": "Points", "line_score": 19.5, "odds_type": "standard"}],
        [{"player_name": "Caitlin Clark", "stat_name": "points", "line_score": 19.5, "side": "over", "american_price": -108}],
    )
    over = next(p for p in out if p.side == "over")
    assert over.prizepicks.line == 19.5
    assert over.underdog.odds_american == -108
    assert over.fanduel.line == 19.5
    assert over.draftkings.line == 20.5

def test_prefers_exact_match_when_multiple_quotes_indexed():
    # Index must support multiple candidate lines per book (from raw normalize
    # before DFS strip). If normalize collapses to one, attach still works.
    # This test builds the multi-quote path via sportsbook_props list with
    # duplicate player/market/side is impossible on WnbaPropLine — instead
    # export _pick_quote and unit-test it, OR pass props where FD=20.5 and
    # assert closest-to-19.5 picks 20.5 when 19.5 absent.
    sb = [_sb("A", "player_points", "over", fanduel=WnbaPropBookQuote(line=20.5, odds_american=-110))]
    out = attach_dfs_snapshots(
        sb,
        [{"player_name": "A", "stat_type": "Points", "line_score": 19.5, "odds_type": "standard"}],
        [],
    )
    over = next(p for p in out if p.side == "over")
    assert over.fanduel.line == 20.5  # closest available

def test_parlay_dfs_quotes_ignored():
    sb = [_sb("A", "player_points", "over",
              prizepicks=WnbaPropBookQuote(line=99.0),
              underdog=WnbaPropBookQuote(line=99.0, odds_american=-100),
              fanduel=WnbaPropBookQuote(line=19.5, odds_american=-110))]
    out = attach_dfs_snapshots(
        sb,
        [{"player_name": "A", "stat_type": "Points", "line_score": 19.5, "odds_type": "standard"}],
        [],
    )
    over = next(p for p in out if p.side == "over")
    assert over.prizepicks.line == 19.5
    assert over.underdog is None
```

Also add a focused unit test for quote picking with multiple candidates — export `pick_closest_quote(quotes: list[WnbaPropBookQuote], targets: list[float]) -> WnbaPropBookQuote | None`:

```python
def test_pick_closest_prefers_exact_either_target():
    from app.services.dfs_attach import pick_closest_quote
    q19 = WnbaPropBookQuote(line=19.5, odds_american=-110)
    q21 = WnbaPropBookQuote(line=21.5, odds_american=-105)
    assert pick_closest_quote([q21, q19], [19.5, 20.5]) is q19
```

- [ ] **Step 2: Run — expect FAIL**

`cd backend && python -m pytest tests/test_dfs_attach.py -v`

- [ ] **Step 3: Implement `dfs_attach.py`**

Core sketch:

```python
US_PROP_SPORTSBOOKS = (
    "fanduel", "draftkings", "caesars", "betmgm", "pinnacle", "bet365", "novig",
)

def pick_closest_quote(
    quotes: list[WnbaPropBookQuote], targets: list[float]
) -> WnbaPropBookQuote | None:
    if not quotes:
        return None
    exact = {t for t in targets}
    for q in quotes:
        if q.line in exact:
            return q
    primary = targets[0]
    return min(quotes, key=lambda q: abs(q.line - primary))

def attach_dfs_snapshots(...) -> list[WnbaPropLine]:
    # 1) build quote index from sportsbook_props for US books only
    #    key: (norm_player, stat_key, side) -> book -> list[WnbaPropBookQuote]
    # 2) seed buckets from PP (standard) then UD
    # 3) for each bucket attach US books via pick_closest_quote
    # 4) apply roster team/logo when missing
    # 5) filter to PP|UD present; sort; return
```

For multi-line candidates: when indexing, if the same book already has a quote, append to a list (normalize currently one line per book — `pick_closest_quote` still works with length-1 lists; leave list structure for future).

Unmapped stats: use `stat_key = f"raw:{normalized}"` so they never collide with Parlay keys; `market_type` = `prizepicks:{stat_type}` or `underdog:{stat_name}`.

- [ ] **Step 4: Run — expect PASS**

`cd backend && python -m pytest tests/test_dfs_attach.py -v`

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/dfs_attach.py backend/tests/test_dfs_attach.py
git commit -m "feat: attach Supabase PP/UD snapshots onto Parlay sportsbook quotes"
```

---

### Task 3: Wire `get_today_props` + strip Parlay DFS + allowlist

**Files:**
- Modify: `backend/app/services/parlay_props.py`
- Modify: `backend/tests/test_parlay_props.py`

**Interfaces:**
- Consumes: `attach_dfs_snapshots`, `fetch_latest_prizepicks`, `fetch_latest_underdog`
- Modifies: `normalize_parlay_props` — do not store quotes for `prizepicks`, `underdog`, `betr`, `sleeper`, `pick6` (skip those bookmakers when assigning)
- Modifies: `get_today_props` — after normalize, fetch snapshots, call `attach_dfs_snapshots`; on Parlay fetch failure, still try snapshots and return DFS-only board when possible

- [ ] **Step 1: Extend failing tests in `test_parlay_props.py`**

Keep existing allowlist test (`test_normalize_drops_milestone_and_alt_markets`).

Add:

```python
def test_normalize_strips_parlay_dfs_books():
    rows = _rows()  # fixture includes prizepicks
    # ensure fixture has a prizepicks row OR append one
    props = svc.normalize_parlay_props(rows)
    assert all(p.prizepicks is None and p.underdog is None for p in props)

def test_get_today_props_attaches_snapshots(monkeypatch):
    async def fake_fetch():
        return _rows()

    svc._cache.clear()
    with (
        patch.object(svc, "PARLAY_API_KEY", "pk_test"),
        patch.object(svc, "fetch_parlay_prop_rows", side_effect=fake_fetch),
        patch("src.odds.load_snapshots.maybe_persist_parlay_props"),
        patch.object(svc, "build_player_team_index", return_value={}),
        patch("app.services.parlay_props.fetch_latest_prizepicks", return_value=[
            {"player_name": "Rhyne Howard", "stat_type": "Assists", "line_score": 3.5, "odds_type": "standard"},
        ]),
        patch("app.services.parlay_props.fetch_latest_underdog", return_value=[]),
    ):
        import asyncio
        body = asyncio.get_event_loop().run_until_complete(svc.get_today_props())
    assert body.error is None
    assert body.props
    assert all(p.prizepicks is not None or p.underdog is not None for p in body.props)
    assert all(p.stat.lower() != "assists" or p.prizepicks is not None for p in body.props if "howard" in p.player_name.lower())

def test_get_today_props_parlay_fail_still_returns_dfs():
    async def boom():
        raise RuntimeError("parlay down")

    svc._cache.clear()
    with (
        patch.object(svc, "PARLAY_API_KEY", "pk_test"),
        patch.object(svc, "fetch_parlay_prop_rows", side_effect=boom),
        patch("app.services.parlay_props.fetch_latest_prizepicks", return_value=[
            {"player_name": "A", "stat_type": "Points", "line_score": 1.5, "odds_type": "standard"},
        ]),
        patch("app.services.parlay_props.fetch_latest_underdog", return_value=[]),
    ):
        import asyncio
        body = asyncio.get_event_loop().run_until_complete(svc.get_today_props())
    assert body.props
    assert all(p.fanduel is None for p in body.props)
    assert all(p.prizepicks is not None for p in body.props)
```

Use the same async test style already used in `test_parlay_props.py` (prefer existing helpers / `pytest.mark.asyncio` if present).

- [ ] **Step 2: Run — expect FAIL**

`cd backend && python -m pytest tests/test_parlay_props.py -v`

- [ ] **Step 3: Implement wiring**

In `normalize_parlay_props`, after resolving `book`:

```python
_DFS_OR_SKIP_BOOKS = frozenset({"prizepicks", "underdog", "betr", "sleeper", "pick6"})
# ...
if book in _DFS_OR_SKIP_BOOKS:
    continue
```

In `get_today_props`:

```python
from app.services.odds_snapshots import fetch_latest_prizepicks, fetch_latest_underdog
from app.services.dfs_attach import attach_dfs_snapshots

# Pseudocode:
try:
    rows = await fetch_parlay_prop_rows()
    # persist + roster as today
    sportsbook_props = normalize_parlay_props(rows, player_teams=player_teams)
    parlay_error = None
except Exception as exc:
    sportsbook_props = []
    player_teams = {}
    parlay_error = str(exc)
    rows = []

pp_rows = fetch_latest_prizepicks("wnba")
ud_rows = fetch_latest_underdog("wnba")
props = attach_dfs_snapshots(sportsbook_props, pp_rows, ud_rows, player_teams=player_teams)

if not props and parlay_error and not pp_rows and not ud_rows:
    # fall back to stale cache / error as today
    ...
response = WnbaPropsResponse(as_of=..., props=props, error=parlay_error if not props else None)
```

Refine to match existing stale-cache behavior: if Parlay fails and snapshots empty and cache exists, return cached; if Parlay fails and snapshots non-empty, return DFS board with `error=None` (or optional warning — prefer `error=None` for clean UI).

Ensure allowlist changes already in the working tree remain (do not revert).

- [ ] **Step 4: Run full related tests — expect PASS**

`cd backend && python -m pytest tests/test_parlay_props.py tests/test_dfs_attach.py tests/test_prop_stat_keys.py -v`

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/parlay_props.py backend/tests/test_parlay_props.py
git commit -m "feat: DFS-first WNBA props from Supabase with Parlay US books"
```

---

### Task 4: Default Book filter to PrizePicks + Underdog

**Files:**
- Modify: `frontend/src/pages/LeaguePropPicksPage.tsx`
- Modify: existing page/filter test if present; else extend `PropPicksFilters.test.tsx` or add a small page test

**Interfaces:**
- Initial `selectedBooks` = `new Set(["prizepicks", "underdog"])`
- Empty set previously meant “all books”; with a non-empty default, `filterPropLines` keeps rows that have **either** PP or UD (OR semantics already). That matches the backend board (all rows already have PP|UD), and hides other book columns until the user clears/adds books.

- [ ] **Step 1: Update page default**

```tsx
const [selectedBooks, setSelectedBooks] = useState<Set<string>>(
  () => new Set(["prizepicks", "underdog"]),
);
```

- [ ] **Step 2: Add/adjust test** asserting the page (or documenting filter behavior) starts with those two books selected — follow existing Vitest patterns in `frontend/src/pages` / `PropPicksFilters.test.tsx`.

- [ ] **Step 3: Run**

`cd frontend && npm test -- --run src/components/league/PropPicksFilters.test.tsx src/components/league/filterPropLines.test.ts`

If a page test is added, include it.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/pages/LeaguePropPicksPage.tsx frontend/src/components/league/*.test.tsx
git commit -m "feat: default prop picks book filter to PrizePicks and Underdog"
```

---

### Task 5: Manual verification checklist

- [ ] **Step 1:** With backend running and Supabase configured, `GET /api/wnba/props/today` returns only rows with `prizepicks` and/or `underdog`.
- [ ] **Step 2:** Spot-check a known player from today’s scrape JSON against the API.
- [ ] **Step 3:** Open `/wnba/prop_picks` — PP/UD columns visible by default; US books appear when matched.
- [ ] **Step 4:** No commit required unless fixes arise; if fixes, commit with `fix:` message.

---

## Self-review (plan vs spec)

| Spec requirement | Task |
| --- | --- |
| Supabase PP/UD latest snapshots | Task 3 |
| PP standard only | Task 2 |
| DFS-first board / drop sportsbook-only | Task 2 |
| Match US books to DFS line | Task 2 |
| Strip Parlay PP/UD | Task 2 + 3 |
| Market allowlist | Task 3 (keep WIP) |
| Parlay fail → DFS still shown | Task 3 |
| Empty snapshots → empty props | Task 2 + 3 |
| Default PP+UD book filter | Task 4 |
| Stat mapping points/PRA/threes | Task 1 |

No TBD/placeholder steps remain after inline review.
