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


SHARP_ROWS = [
    {
        "sportsbook": "fanduel",
        "is_main_line": True,
        "market_type": "player_assists",
        "selection_type": "over",
        "player_name": "Rhyne Howard",
        "stat_category": "assists",
        "line": 3.5,
        "odds_american": -114,
    },
    {
        "sportsbook": "draftkings",
        "is_main_line": True,
        "market_type": "player_assists",
        "selection_type": "under",
        "player_name": "Rhyne Howard",
        "stat_category": "assists",
        "line": 3.5,
        "odds_american": -110,
    },
]


def test_load_sharp_book_snapshot_fanduel(mock_upsert):
    count = load_snapshots.load_sharp_book_snapshot(
        SHARP_ROWS, sportsbook="fanduel", league="wnba", scraped_at=SCRAPED
    )
    assert count == 1
    table, df = mock_upsert.call_args[0]
    assert table == "wnba_fanduel"
    assert df.iloc[0]["player_name"] == "Rhyne Howard"
    assert df.iloc[0]["american_price"] == -114
    assert mock_upsert.call_args[1]["conflict_cols"] == [
        "league",
        "player_name",
        "market_type",
        "side",
        "line_score",
        "scraped_at",
    ]


def test_load_sharp_book_snapshot_skip_db(monkeypatch, mock_upsert):
    monkeypatch.setenv("SHARP_PROPS_SKIP_DB", "1")
    count = load_snapshots.load_sharp_book_snapshot(
        SHARP_ROWS, sportsbook="draftkings", league="wnba", scraped_at=SCRAPED
    )
    assert count == 0
    mock_upsert.assert_not_called()


def test_should_persist_when_no_prior_snapshot(monkeypatch):
    monkeypatch.setattr(load_snapshots, "latest_sharp_props_scraped_at", lambda league: None)
    assert load_snapshots.should_persist_sharp_props(league="wnba") is True


def test_should_persist_false_when_recent(monkeypatch):
    now = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    recent = datetime(2026, 8, 1, 11, 45, tzinfo=timezone.utc)
    monkeypatch.setattr(
        load_snapshots, "latest_sharp_props_scraped_at", lambda league: recent
    )
    assert (
        load_snapshots.should_persist_sharp_props(
            league="wnba", now=now, interval_minutes=30
        )
        is False
    )


def test_should_persist_true_when_stale(monkeypatch):
    now = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    stale = datetime(2026, 8, 1, 11, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(
        load_snapshots, "latest_sharp_props_scraped_at", lambda league: stale
    )
    assert (
        load_snapshots.should_persist_sharp_props(
            league="wnba", now=now, interval_minutes=30
        )
        is True
    )


def test_maybe_persist_sharp_props_writes_both_books(monkeypatch, mock_upsert):
    monkeypatch.setattr(load_snapshots, "should_persist_sharp_props", lambda **kw: True)
    counts = load_snapshots.maybe_persist_sharp_props(
        SHARP_ROWS, league="wnba", scraped_at=SCRAPED
    )
    assert counts == {"fanduel": 1, "draftkings": 1}
    assert mock_upsert.call_count == 2
    tables = {call.args[0] for call in mock_upsert.call_args_list}
    assert tables == {"wnba_fanduel", "wnba_draftkings"}


def test_maybe_persist_sharp_props_skips_when_throttled(monkeypatch, mock_upsert):
    monkeypatch.setattr(load_snapshots, "should_persist_sharp_props", lambda **kw: False)
    counts = load_snapshots.maybe_persist_sharp_props(SHARP_ROWS, league="wnba")
    assert counts == {"fanduel": 0, "draftkings": 0}
    mock_upsert.assert_not_called()


def test_maybe_persist_sharp_props_swallows_write_errors(monkeypatch, mock_upsert):
    monkeypatch.setattr(load_snapshots, "should_persist_sharp_props", lambda **kw: True)
    mock_upsert.side_effect = RuntimeError("db down")
    counts = load_snapshots.maybe_persist_sharp_props(SHARP_ROWS, league="wnba")
    assert counts == {"fanduel": 0, "draftkings": 0}


PARLAY_ROWS = [
    {
        "bookmaker": "fanduel",
        "player": "Rhyne Howard",
        "market_key": "player_assists",
        "market": "Assists",
        "line": 3.5,
        "over_price": -114,
        "under_price": -110,
    },
    {
        "bookmaker": "draftkings",
        "player": "Rhyne Howard",
        "market_key": "player_assists",
        "market": "Assists",
        "line": 3.5,
        "over_price": -120,
        "under_price": -110,
    },
    {
        "bookmaker": "pinnacle",
        "player": "Rhyne Howard",
        "market_key": "player_assists",
        "market": "Assists",
        "line": 3.5,
        "over_price": -108,
        "under_price": -112,
    },
    {
        "bookmaker": "prizepicks",
        "player": "Rhyne Howard",
        "market_key": "player_assists",
        "market": "Assists",
        "line": 3.5,
        "over_price": -100,
        "under_price": -100,
    },
    {
        "bookmaker": "novig",
        "player": "Rhyne Howard",
        "market_key": "player_assists",
        "market": "Assists",
        "line": 3.5,
        "over_price": -110,
        "under_price": -110,
    },
]


def test_maybe_persist_parlay_props_writes_books(monkeypatch, mock_upsert):
    monkeypatch.setattr(load_snapshots, "should_persist_parlay_props", lambda **kw: True)
    counts = load_snapshots.maybe_persist_parlay_props(
        PARLAY_ROWS, league="wnba", scraped_at=SCRAPED
    )
    assert counts["fanduel"] == 2
    assert counts["draftkings"] == 2
    assert counts["pinnacle"] == 2
    assert counts["prizepicks"] == 2
    assert counts["novig"] == 2
    assert counts["caesars"] == 0
    assert counts["betr"] == 0
    tables = {call.args[0] for call in mock_upsert.call_args_list}
    assert tables == {
        "wnba_fanduel",
        "wnba_draftkings",
        "wnba_pinnacle",
        "wnba_prizepicks_parlay",
        "wnba_novig",
    }
    assert set(counts) == set(load_snapshots.PARLAY_PROP_SPORTSBOOKS)
    assert len(load_snapshots.PARLAY_PROP_SPORTSBOOKS) == 12


def test_maybe_persist_parlay_props_skips_when_throttled(monkeypatch, mock_upsert):
    monkeypatch.setattr(load_snapshots, "should_persist_parlay_props", lambda **kw: False)
    counts = load_snapshots.maybe_persist_parlay_props(PARLAY_ROWS, league="wnba")
    assert all(v == 0 for v in counts.values())
    mock_upsert.assert_not_called()


def test_should_persist_parlay_false_when_recent(monkeypatch):
    now = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    recent = datetime(2026, 8, 1, 11, 45, tzinfo=timezone.utc)
    monkeypatch.setattr(
        load_snapshots, "latest_parlay_props_scraped_at", lambda league: recent
    )
    assert (
        load_snapshots.should_persist_parlay_props(
            league="wnba", now=now, interval_minutes=30
        )
        is False
    )
