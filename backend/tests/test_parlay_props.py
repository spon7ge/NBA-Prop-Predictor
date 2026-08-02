from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.schemas.wnba_props import PROP_SPORTSBOOKS
from app.services import parlay_props as svc

FIXTURE = Path(__file__).parent / "fixtures" / "parlay_wnba_props.json"


def _rows() -> list[dict]:
    return json.loads(FIXTURE.read_text())


def test_normalize_parlay_props_main_line_and_sides():
    props = svc.normalize_parlay_props(_rows())
    over = next(
        p
        for p in props
        if p.player_name == "Rhyne Howard"
        and p.market_type == "player_assists"
        and p.side == "over"
    )
    under = next(
        p
        for p in props
        if p.player_name == "Rhyne Howard"
        and p.market_type == "player_assists"
        and p.side == "under"
    )

    assert over.fanduel is not None
    assert over.fanduel.line == 3.5
    assert over.fanduel.odds_american == -114
    assert over.draftkings is not None
    assert over.draftkings.odds_american == -120
    assert over.pinnacle is not None
    assert over.pinnacle.odds_american == -108
    assert over.caesars is None
    assert over.betmgm is None
    assert over.bet365 is None

    assert under.fanduel is not None
    assert under.fanduel.odds_american == -110

    assert all(
        not (p.player_name == "Rhyne Howard" and p.fanduel and p.fanduel.line == 4.5)
        for p in props
    )

    assert all(p.prizepicks is None and p.underdog is None for p in props)


def test_normalize_includes_caesars_only_player():
    props = svc.normalize_parlay_props(_rows())
    gray = next(
        p for p in props if p.player_name == "Allisha Gray" and p.side == "over"
    )
    assert gray.caesars is not None
    assert gray.caesars.line == 14.5
    assert gray.fanduel is None


def test_normalize_strips_parlay_dfs_books():
    props = svc.normalize_parlay_props(_rows())
    assert all(p.prizepicks is None and p.underdog is None for p in props)


def test_normalize_drops_milestone_and_alt_markets():
    rows = _rows() + [
        {
            "bookmaker": "fanduel",
            "player": "Rhyne Howard",
            "market_key": "player_threes_made_milestones_1_or_more",
            "market": "Threes Made Milestones 1 Or More",
            "line": 1.0,
            "over_price": -200,
            "under_price": None,
            "home_team": "Atlanta Dream",
            "away_team": "Chicago Sky",
        },
        {
            "bookmaker": "draftkings",
            "player": "Rhyne Howard",
            "market_key": "player_points_alt",
            "market": "Player Points Milestones 20 Or More",
            "line": 20.0,
            "over_price": 150,
            "under_price": None,
            "home_team": "Atlanta Dream",
            "away_team": "Chicago Sky",
        },
    ]
    props = svc.normalize_parlay_props(rows)
    assert all("milestone" not in p.stat.lower() for p in props)
    assert all(not p.market_type.endswith("_alt") for p in props)
    assert all("milestones" not in p.market_type for p in props)


def test_get_today_props_missing_key():
    with patch.object(svc, "PARLAY_API_KEY", None):
        svc._cache.clear()
        client = TestClient(app)
        res = client.get("/api/wnba/props/today")
    assert res.status_code == 200
    body = res.json()
    assert body["props"] == []
    assert body["error"] == "PARLAY_API_KEY is not configured"


def _pp_snapshots() -> list[dict]:
    return [
        {
            "player_name": "Rhyne Howard",
            "stat_type": "Assists",
            "line_score": 3.5,
            "odds_type": "standard",
        },
        {
            "player_name": "Allisha Gray",
            "stat_type": "Points",
            "line_score": 14.5,
            "odds_type": "standard",
        },
    ]


def test_get_today_props_success():
    rows = _rows()

    async def fake_fetch():
        return rows

    with (
        patch.object(svc, "PARLAY_API_KEY", "pk_test"),
        patch.object(svc, "fetch_parlay_prop_rows", side_effect=fake_fetch),
        patch.object(svc, "build_player_team_index", return_value={}),
        patch.object(svc, "fetch_latest_prizepicks", return_value=_pp_snapshots()),
        patch.object(svc, "fetch_latest_underdog", return_value=[]),
        patch(
            "src.odds.load_snapshots.maybe_persist_parlay_props",
            return_value={b: 0 for b in PROP_SPORTSBOOKS},
        ),
    ):
        svc._cache.clear()
        client = TestClient(app)
        res = client.get("/api/wnba/props/today")

    assert res.status_code == 200
    body = res.json()
    assert body["error"] is None
    assert len(body["props"]) >= 2
    assert body["sportsbooks"] == list(PROP_SPORTSBOOKS)
    assert "prizepicks" in body["sportsbooks"]
    assert "pick6" in body["sportsbooks"]


def test_get_today_props_persist_failure_still_returns():
    rows = _rows()

    async def fake_fetch():
        return rows

    def boom(*_a, **_k):
        raise RuntimeError("db down")

    with (
        patch.object(svc, "PARLAY_API_KEY", "pk_test"),
        patch.object(svc, "fetch_parlay_prop_rows", side_effect=fake_fetch),
        patch.object(svc, "build_player_team_index", return_value={}),
        patch.object(svc, "fetch_latest_prizepicks", return_value=_pp_snapshots()),
        patch.object(svc, "fetch_latest_underdog", return_value=[]),
        patch("src.odds.load_snapshots.maybe_persist_parlay_props", side_effect=boom),
    ):
        svc._cache.clear()
        client = TestClient(app)
        res = client.get("/api/wnba/props/today")

    assert res.status_code == 200
    body = res.json()
    assert body["error"] is None
    assert len(body["props"]) >= 1


def test_get_today_props_attaches_snapshots():
    rows = _rows()

    async def fake_fetch():
        return rows

    svc._cache.clear()
    with (
        patch.object(svc, "PARLAY_API_KEY", "pk_test"),
        patch.object(svc, "fetch_parlay_prop_rows", side_effect=fake_fetch),
        patch("src.odds.load_snapshots.maybe_persist_parlay_props"),
        patch.object(svc, "build_player_team_index", return_value={}),
        patch.object(
            svc,
            "fetch_latest_prizepicks",
            return_value=[
                {
                    "player_name": "Rhyne Howard",
                    "stat_type": "Assists",
                    "line_score": 3.5,
                    "odds_type": "standard",
                },
            ],
        ),
        patch.object(svc, "fetch_latest_underdog", return_value=[]),
    ):
        client = TestClient(app)
        res = client.get("/api/wnba/props/today")

    assert res.status_code == 200
    body = res.json()
    assert body["error"] is None
    assert body["props"]
    assert all(
        p.get("prizepicks") is not None or p.get("underdog") is not None
        for p in body["props"]
    )
    howard_assists = [
        p
        for p in body["props"]
        if "howard" in p["player_name"].lower() and p["stat"].lower() == "assists"
    ]
    assert howard_assists
    assert all(p["prizepicks"] is not None for p in howard_assists)


def test_get_today_props_parlay_fail_still_returns_dfs():
    async def boom():
        raise RuntimeError("parlay down")

    svc._cache.clear()
    with (
        patch.object(svc, "PARLAY_API_KEY", "pk_test"),
        patch.object(svc, "fetch_parlay_prop_rows", side_effect=boom),
        patch.object(
            svc,
            "fetch_latest_prizepicks",
            return_value=[
                {
                    "player_name": "A",
                    "stat_type": "Points",
                    "line_score": 1.5,
                    "odds_type": "standard",
                },
            ],
        ),
        patch.object(svc, "fetch_latest_underdog", return_value=[]),
    ):
        client = TestClient(app)
        res = client.get("/api/wnba/props/today")

    assert res.status_code == 200
    body = res.json()
    assert body["error"] is None
    assert body["props"]
    assert all(p["fanduel"] is None for p in body["props"])
    assert all(p["prizepicks"] is not None for p in body["props"])
