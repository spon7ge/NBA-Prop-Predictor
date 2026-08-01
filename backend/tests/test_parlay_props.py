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

    pp_over = next(
        p
        for p in props
        if p.player_name == "Rhyne Howard"
        and p.market_type == "player_assists"
        and p.side == "over"
        and p.prizepicks is not None
    )
    assert pp_over.prizepicks.line == 3.5
    assert all(
        not (
            p.player_name == "Rhyne Howard"
            and p.prizepicks
            and p.prizepicks.line == 5.5
        )
        for p in props
    )


def test_normalize_includes_caesars_only_player():
    props = svc.normalize_parlay_props(_rows())
    gray = next(
        p for p in props if p.player_name == "Allisha Gray" and p.side == "over"
    )
    assert gray.caesars is not None
    assert gray.caesars.line == 14.5
    assert gray.fanduel is None


def test_get_today_props_missing_key():
    with patch.object(svc, "PARLAY_API_KEY", None):
        svc._cache.clear()
        client = TestClient(app)
        res = client.get("/api/wnba/props/today")
    assert res.status_code == 200
    body = res.json()
    assert body["props"] == []
    assert body["error"] == "PARLAY_API_KEY is not configured"


def test_get_today_props_success():
    rows = _rows()

    async def fake_fetch():
        return rows

    with (
        patch.object(svc, "PARLAY_API_KEY", "pk_test"),
        patch.object(svc, "fetch_parlay_prop_rows", side_effect=fake_fetch),
        patch.object(svc, "build_player_team_index", return_value={}),
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
        patch("src.odds.load_snapshots.maybe_persist_parlay_props", side_effect=boom),
    ):
        svc._cache.clear()
        client = TestClient(app)
        res = client.get("/api/wnba/props/today")

    assert res.status_code == 200
    body = res.json()
    assert body["error"] is None
    assert len(body["props"]) >= 1
