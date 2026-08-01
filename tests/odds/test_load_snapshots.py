import os
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.odds import load_snapshots


SCRAPED = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)

PRIZEPICKS_PROJECTIONS = [
    {
        "player": "A'ja Wilson",
        "stat_type": "Points",
        "line_score": 22.5,
        "odds_type": "standard",
        "updated_at": "2026-07-31T12:00:00-04:00",
    }
]

UNDERDOG_PICKS = [
    {
        "full_name": "Caitlin Clark",
        "stat_name": "points",
        "stat_value": "19.5",
        "choice": "over",
        "american_price": "-130",
        "payout_multiplier": "0.94",
        "updated_at": "2026-07-31T23:57:11Z",
    }
]


@pytest.fixture
def mock_upsert(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(load_snapshots, "upsert_df", mock)
    return mock


def test_load_prizepicks_snapshot_calls_upsert(mock_upsert):
    count = load_snapshots.load_prizepicks_snapshot(
        PRIZEPICKS_PROJECTIONS, league="wnba", scraped_at=SCRAPED
    )

    assert count == 1
    mock_upsert.assert_called_once()
    table, df = mock_upsert.call_args[0]
    assert table == "wnba_prizepicks"
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["player_name"] == "A'ja Wilson"
    assert df.iloc[0]["league"] == "wnba"
    assert df.iloc[0]["line_score"] == 22.5

    kwargs = mock_upsert.call_args[1]
    assert kwargs["schema"] == "odds"
    assert kwargs["lineage_col"] == "fetched_at"
    assert kwargs["conflict_cols"] == [
        "league",
        "player_name",
        "stat_type",
        "odds_type",
        "line_score",
        "scraped_at",
    ]


def test_load_prizepicks_snapshot_empty_returns_zero(mock_upsert):
    count = load_snapshots.load_prizepicks_snapshot([], league="wnba", scraped_at=SCRAPED)
    assert count == 0
    mock_upsert.assert_not_called()


def test_load_prizepicks_snapshot_skip_db(monkeypatch, mock_upsert):
    monkeypatch.setenv("PRIZEPICKS_SKIP_DB", "1")
    count = load_snapshots.load_prizepicks_snapshot(
        PRIZEPICKS_PROJECTIONS, league="wnba", scraped_at=SCRAPED
    )
    assert count == 0
    mock_upsert.assert_not_called()


@pytest.mark.parametrize("skip_value", ["true", "yes", "TRUE"])
def test_load_prizepicks_snapshot_skip_db_truthy(monkeypatch, mock_upsert, skip_value):
    monkeypatch.setenv("PRIZEPICKS_SKIP_DB", skip_value)
    count = load_snapshots.load_prizepicks_snapshot(
        PRIZEPICKS_PROJECTIONS, league="wnba", scraped_at=SCRAPED
    )
    assert count == 0
    mock_upsert.assert_not_called()


def test_load_underdog_snapshot_calls_upsert(mock_upsert):
    count = load_snapshots.load_underdog_snapshot(
        UNDERDOG_PICKS, league="wnba", scraped_at=SCRAPED
    )

    assert count == 1
    mock_upsert.assert_called_once()
    table, df = mock_upsert.call_args[0]
    assert table == "wnba_underdogs"
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["player_name"] == "Caitlin Clark"
    assert df.iloc[0]["side"] == "over"
    assert df.iloc[0]["line_score"] == 19.5
    assert df.iloc[0]["payout_multiplier"] == 0.94

    kwargs = mock_upsert.call_args[1]
    assert kwargs["schema"] == "odds"
    assert kwargs["lineage_col"] == "fetched_at"
    assert kwargs["conflict_cols"] == [
        "league",
        "player_name",
        "stat_name",
        "side",
        "line_score",
        "scraped_at",
    ]


def test_load_underdog_snapshot_empty_returns_zero(mock_upsert):
    count = load_snapshots.load_underdog_snapshot([], league="wnba", scraped_at=SCRAPED)
    assert count == 0
    mock_upsert.assert_not_called()


def test_load_underdog_snapshot_skip_db(monkeypatch, mock_upsert):
    monkeypatch.setenv("UNDERDOG_SKIP_DB", "1")
    count = load_snapshots.load_underdog_snapshot(
        UNDERDOG_PICKS, league="wnba", scraped_at=SCRAPED
    )
    assert count == 0
    mock_upsert.assert_not_called()


def test_load_prizepicks_coerces_string_line_score(mock_upsert):
    projections = [
        {
            "player": "Test Player",
            "stat_type": "Rebounds",
            "line_score": "10.5",
            "odds_type": "standard",
        }
    ]
    load_snapshots.load_prizepicks_snapshot(projections, league="nba", scraped_at=SCRAPED)
    df = mock_upsert.call_args[0][1]
    assert df.iloc[0]["line_score"] == 10.5
