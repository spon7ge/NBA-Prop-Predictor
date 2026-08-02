from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.mlb_scoreboard import MlbGame, MlbScoreboardResponse, MlbTeam


@pytest.fixture
def client():
    return TestClient(app)


def _sample_response() -> MlbScoreboardResponse:
    team = MlbTeam(abbrev="NYY", name="New York Yankees", score=1)
    home = MlbTeam(abbrev="BOS", name="Boston Red Sox", score=0)
    game = MlbGame(
        id="mlb-1",
        mlb_game_pk="1",
        status="live",
        status_label="Top 1st",
        away=team,
        home=home,
        start_time_et="2026-08-02T17:00:00Z",
    )
    return MlbScoreboardResponse(
        date="2026-08-02",
        games=[game],
        fetched_at="2026-08-02T12:00:00+00:00",
    )


def test_mlb_scoreboard_today_ok(client):
    with patch(
        "app.api.routes.mlb_scoreboard.get_today_scoreboard",
        new=AsyncMock(return_value=_sample_response()),
    ):
        res = client.get("/api/mlb/scoreboard/today")
    assert res.status_code == 200
    assert res.headers.get("cache-control") == "no-store"
    body = res.json()
    assert body["games"][0]["league"] == "mlb"
    assert body["games"][0]["mlb_game_pk"] == "1"


def test_mlb_scoreboard_today_upstream_failure_is_502(client):
    with patch(
        "app.api.routes.mlb_scoreboard.get_today_scoreboard",
        new=AsyncMock(side_effect=RuntimeError("upstream down")),
    ):
        res = client.get("/api/mlb/scoreboard/today")
    assert res.status_code == 502
    assert res.headers.get("cache-control") == "no-store"
