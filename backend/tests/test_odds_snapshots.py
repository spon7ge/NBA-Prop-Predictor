from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.services import odds_snapshots as svc


def _mock_engine(rows: list[dict]):
    conn = MagicMock()
    result = MagicMock()
    result.__iter__ = lambda self: iter(
        [MagicMock(_mapping=row) for row in rows]
    )
    conn.execute.return_value = result
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    engine = MagicMock()
    engine.connect.return_value = conn
    return engine, conn


def test_fetch_latest_prizepicks_returns_rows():
    rows = [
        {
            "player_name": "A'ja Wilson",
            "stat_type": "Points",
            "line_score": 22.5,
            "odds_type": "standard",
        }
    ]
    engine, conn = _mock_engine(rows)

    with patch("src.utils.db.get_engine", return_value=engine):
        out = svc.fetch_latest_prizepicks("wnba")

    assert out == rows
    conn.execute.assert_called_once()
    params = conn.execute.call_args[0][1]
    assert params["league"] == "wnba"


def test_fetch_latest_underdog_returns_rows():
    rows = [
        {
            "player_name": "Breanna Stewart",
            "stat_name": "Rebounds",
            "line_score": 8.5,
            "side": "over",
            "american_price": -112,
        }
    ]
    engine, conn = _mock_engine(rows)

    with patch("src.utils.db.get_engine", return_value=engine):
        out = svc.fetch_latest_underdog("wnba")

    assert out == rows
    conn.execute.assert_called_once()
    params = conn.execute.call_args[0][1]
    assert params["league"] == "wnba"


def test_fetch_latest_prizepicks_empty_on_db_error():
    with patch("src.utils.db.get_engine", side_effect=RuntimeError("no db")):
        assert svc.fetch_latest_prizepicks() == []


def test_fetch_latest_underdog_empty_on_query_error():
    engine = MagicMock()
    conn = MagicMock()
    conn.execute.side_effect = Exception("query failed")
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    engine.connect.return_value = conn

    with patch("src.utils.db.get_engine", return_value=engine):
        assert svc.fetch_latest_underdog() == []
