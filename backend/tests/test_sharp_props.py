from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import sharp_props as svc

FIXTURE = Path(__file__).parent / "fixtures" / "sharp_wnba_props.json"


@pytest.fixture(autouse=True)
def clear_cache():
    svc._cache.clear()
    yield
    svc._cache.clear()


def test_merge_snapshot_props_fills_prizepicks_and_underdog():
    rows = json.loads(FIXTURE.read_text())["data"]
    player_teams = {
        "rhyne howard": ("ATL", "https://cdn.sharpapi.io/teams/basketball/48.png"),
    }
    sharp_props = svc.normalize_sharp_props(rows, player_teams=player_teams)
    pp_rows = [
        {
            "player_name": "Rhyne Howard",
            "stat_type": "Assists",
            "line_score": 3.5,
            "odds_type": "standard",
        }
    ]
    ud_rows = [
        {
            "player_name": "Rhyne Howard",
            "stat_name": "assists",
            "line_score": 3.5,
            "side": "over",
            "american_price": -108,
        },
        {
            "player_name": "Rhyne Howard",
            "stat_name": "assists",
            "line_score": 3.5,
            "side": "under",
            "american_price": -112,
        },
    ]
    props = svc.merge_snapshot_props(
        sharp_props, pp_rows, ud_rows, player_teams=player_teams
    )
    over = next(
        p for p in props if p.player_name == "Rhyne Howard" and p.side == "over"
    )
    under = next(
        p for p in props if p.player_name == "Rhyne Howard" and p.side == "under"
    )
    assert over.fanduel is not None
    assert over.draftkings is not None
    assert over.prizepicks is not None
    assert over.prizepicks.line == 3.5
    assert over.prizepicks.odds_american is None
    assert over.underdog is not None
    assert over.underdog.odds_american == -108
    assert under.prizepicks is not None
    assert under.prizepicks.line == 3.5
    assert under.underdog is not None
    assert under.underdog.odds_american == -112
    assert not hasattr(over, "betmgm")
    assert not hasattr(over, "betrivers")


def test_merge_snapshot_props_pp_only_creates_both_sides():
    pp_rows = [
        {
            "player_name": "Caitlin Clark",
            "stat_type": "Points",
            "line_score": 19.5,
            "odds_type": "standard",
        }
    ]
    props = svc.merge_snapshot_props([], pp_rows, [])
    assert len(props) == 2
    sides = {p.side for p in props}
    assert sides == {"over", "under"}
    row = props[0]
    assert row.stat == "Points"
    assert row.market_type == "prizepicks:Points"
    assert row.prizepicks.line == 19.5
    assert row.fanduel is None
    assert row.draftkings is None
    assert row.underdog is None


def test_normalize_merges_books_and_both_sides():
    rows = json.loads(FIXTURE.read_text())["data"]
    player_teams = {
        "rhyne howard": ("ATL", "https://cdn.sharpapi.io/teams/basketball/48.png"),
    }
    props = svc.normalize_sharp_props(rows, player_teams=player_teams)
    over = next(
        p for p in props if p.player_name == "Rhyne Howard" and p.side == "over"
    )
    under = next(
        p for p in props if p.player_name == "Rhyne Howard" and p.side == "under"
    )
    assert over.stat == "Assists"
    assert over.market_type == "player_assists"
    assert over.team_abbrev == "ATL"
    assert over.logo_url == "https://cdn.sharpapi.io/teams/basketball/48.png"
    assert over.model_prediction is None
    assert over.over_under_pct is None
    assert over.ev is None
    assert over.fanduel is not None
    assert over.fanduel.line == 3.5
    assert over.fanduel.odds_american == -114
    assert over.draftkings is not None
    assert over.draftkings.line == 3.5
    assert over.draftkings.odds_american == -120
    assert over.underdog is None
    assert over.prizepicks is None
    assert under.fanduel is not None
    assert under.draftkings is not None
    assert under.draftkings.odds_american == -110


def test_normalize_keeps_row_when_one_book_missing():
    rows = json.loads(FIXTURE.read_text())["data"]
    props = svc.normalize_sharp_props(rows)
    gray = next(
        p for p in props if p.player_name == "Allisha Gray" and p.side == "over"
    )
    assert gray.fanduel is not None
    assert gray.fanduel.line == 2.5
    assert gray.draftkings is None
    assert gray.team_abbrev is None
    assert gray.logo_url is None


def test_normalize_ignores_non_props_and_alternates():
    rows = json.loads(FIXTURE.read_text())["data"]
    props = svc.normalize_sharp_props(rows)
    assert all(p.market_type.startswith("player_") for p in props)
    assert {p.player_name for p in props} == {"Rhyne Howard", "Allisha Gray"}
    howard = [p for p in props if p.player_name == "Rhyne Howard"]
    assert len(howard) == 2
    assert {p.side for p in howard} == {"over", "under"}


def test_normalize_sort_order():
    rows = json.loads(FIXTURE.read_text())["data"]
    props = svc.normalize_sharp_props(rows)
    keys = [(p.player_name, p.market_type, p.side) for p in props]
    assert keys == sorted(
        keys, key=lambda k: (k[0], k[1], 0 if k[2] == "over" else 1)
    )


def test_props_route_returns_props_when_fetch_ok():
    payload = json.loads(FIXTURE.read_text())

    async def fake_fetch():
        return payload["data"]

    async def fake_teams(_rows):
        return {
            "rhyne howard": ("ATL", "https://cdn.sharpapi.io/teams/basketball/48.png"),
        }

    with (
        patch.object(svc, "SHARP_API_KEY", "sk_test"),
        patch.object(svc, "fetch_sharp_prop_rows", side_effect=fake_fetch),
        patch.object(svc, "build_player_team_index", side_effect=fake_teams),
        patch.object(svc, "fetch_latest_prizepicks", return_value=[]),
        patch.object(svc, "fetch_latest_underdog", return_value=[]),
        patch(
            "src.odds.load_snapshots.maybe_persist_sharp_props",
            return_value={"fanduel": 0, "draftkings": 0},
        ),
    ):
        client = TestClient(app)
        res = client.get("/api/wnba/props/today")

    assert res.status_code == 200
    assert res.headers.get("cache-control") == "no-store"
    body = res.json()
    assert body["sportsbooks"] == [
        "fanduel",
        "draftkings",
        "prizepicks",
        "underdog",
    ]
    assert body["as_of"]
    assert len(body["props"]) == 3
    over = next(
        p
        for p in body["props"]
        if p["player_name"] == "Rhyne Howard" and p["side"] == "over"
    )
    assert over["fanduel"]["odds_american"] == -114
    assert over["draftkings"]["odds_american"] == -120
    assert over["team_abbrev"] == "ATL"
    assert over["logo_url"] == "https://cdn.sharpapi.io/teams/basketball/48.png"


def test_props_route_ok_when_snapshot_persist_raises():
    payload = json.loads(FIXTURE.read_text())

    async def fake_fetch():
        return payload["data"]

    async def fake_teams(_rows):
        return {}

    def boom(*_a, **_k):
        raise RuntimeError("db unavailable")

    with (
        patch.object(svc, "SHARP_API_KEY", "sk_test"),
        patch.object(svc, "fetch_sharp_prop_rows", side_effect=fake_fetch),
        patch.object(svc, "build_player_team_index", side_effect=fake_teams),
        patch.object(svc, "fetch_latest_prizepicks", return_value=[]),
        patch.object(svc, "fetch_latest_underdog", return_value=[]),
        patch("src.odds.load_snapshots.maybe_persist_sharp_props", side_effect=boom),
    ):
        client = TestClient(app)
        res = client.get("/api/wnba/props/today")

    assert res.status_code == 200
    assert len(res.json()["props"]) >= 1


def test_props_route_empty_when_no_key():
    with patch.object(svc, "SHARP_API_KEY", None):
        client = TestClient(app)
        res = client.get("/api/wnba/props/today")

    assert res.status_code == 200
    body = res.json()
    assert body["props"] == []
    assert body["error"]


def test_props_route_stale_cache_on_error():
    payload = json.loads(FIXTURE.read_text())

    async def ok():
        return payload["data"]

    async def boom():
        raise RuntimeError("sharp down")

    async def no_teams(_rows):
        return {}

    with (
        patch.object(svc, "SHARP_API_KEY", "sk_test"),
        patch.object(svc, "fetch_sharp_prop_rows", side_effect=ok),
        patch.object(svc, "build_player_team_index", side_effect=no_teams),
        patch.object(svc, "fetch_latest_prizepicks", return_value=[]),
        patch.object(svc, "fetch_latest_underdog", return_value=[]),
        patch(
            "src.odds.load_snapshots.maybe_persist_sharp_props",
            return_value={"fanduel": 0, "draftkings": 0},
        ),
    ):
        client = TestClient(app)
        assert client.get("/api/wnba/props/today").status_code == 200

    svc._cache["expires_at"] = 0

    with (
        patch.object(svc, "SHARP_API_KEY", "sk_test"),
        patch.object(svc, "fetch_sharp_prop_rows", side_effect=boom),
    ):
        res = client.get("/api/wnba/props/today")

    assert res.status_code == 200
    assert len(res.json()["props"]) == 3


def test_fetch_prop_rows_fetches_books_separately_and_stops_without_next_offset():
    import asyncio

    # Each book: page0 has_more+next, page1 has_more but no next_offset → stop.
    pages_by_book = {
        "fanduel": [
            {
                "data": [{"id": "fd-1", "sportsbook": "fanduel"}],
                "pagination": {"has_more": True, "next_offset": 200},
            },
            {
                "data": [{"id": "fd-2", "sportsbook": "fanduel"}],
                "pagination": {"has_more": True, "next_offset": None},
            },
        ],
        "draftkings": [
            {
                "data": [{"id": "dk-1", "sportsbook": "draftkings"}],
                "pagination": {"has_more": False},
            },
        ],
    }
    calls: list[str] = []

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self._idx = {book: 0 for book in pages_by_book}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, **kwargs):
            params = kwargs.get("params") or {}
            book = str(params.get("sportsbook"))
            calls.append(book)
            pages = pages_by_book[book]
            i = self._idx[book]
            self._idx[book] = i + 1
            return FakeResp(pages[i])

    async def run():
        with (
            patch.object(svc, "SHARP_API_KEY", "sk_test"),
            patch.object(svc.httpx, "AsyncClient", FakeClient),
        ):
            return await svc.fetch_sharp_prop_rows()

    rows = asyncio.run(run())
    assert calls == [
        "fanduel",
        "fanduel",
        "draftkings",
    ]
    assert [r["id"] for r in rows] == ["fd-1", "fd-2", "dk-1"]
