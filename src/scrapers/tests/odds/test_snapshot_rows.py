from datetime import datetime, timezone

import pytest

from src.odds.snapshot_rows import (
    parse_american_price,
    prizepicks_projections_to_rows,
    sharp_props_to_book_rows,
    underdog_picks_to_rows,
)


def test_parse_american_price():
    assert parse_american_price("+477") == 477
    assert parse_american_price("-130") == -130
    assert parse_american_price(None) is None


def test_prizepicks_mapper():
    scraped = datetime(2026, 8, 1, tzinfo=timezone.utc)
    rows = prizepicks_projections_to_rows(
        [{"player": "A'ja Wilson", "stat_type": "Points", "line_score": 22.5,
          "odds_type": "standard", "updated_at": "2026-07-31T12:00:00-04:00", "league": "WNBA"}],
        league="wnba",
        scraped_at=scraped,
    )
    assert rows[0]["player_name"] == "A'ja Wilson"
    assert rows[0]["league"] == "wnba"
    assert rows[0]["line_score"] == 22.5
    assert rows[0]["scraped_at"] == scraped


def test_underdog_mapper():
    scraped = datetime(2026, 8, 1, tzinfo=timezone.utc)
    rows = underdog_picks_to_rows(
        [{"full_name": "Caitlin Clark", "stat_name": "points", "stat_value": "19.5",
          "choice": "over", "american_price": "-130", "payout_multiplier": "0.94",
          "updated_at": "2026-07-31T23:57:11Z"}],
        league="wnba",
        scraped_at=scraped,
    )
    assert rows[0]["side"] == "over"
    assert rows[0]["american_price"] == -130
    assert float(rows[0]["line_score"]) == 19.5


def test_sharp_props_to_book_rows_filters_and_maps():
    scraped = datetime(2026, 8, 1, tzinfo=timezone.utc)
    rows = sharp_props_to_book_rows(
        [
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
                "selection_type": "over",
                "player_name": "Rhyne Howard",
                "stat_category": "assists",
                "line": 3.5,
                "odds_american": -120,
            },
            {
                "sportsbook": "fanduel",
                "is_main_line": False,
                "market_type": "player_assists",
                "selection_type": "over",
                "player_name": "Alt Line",
                "line": 4.5,
                "odds_american": -110,
            },
            {
                "sportsbook": "fanduel",
                "is_main_line": True,
                "market_type": "team_total",
                "selection_type": "over",
                "player_name": "Team",
                "line": 80.5,
                "odds_american": -110,
            },
        ],
        sportsbook="fanduel",
        league="WNBA",
        scraped_at=scraped,
    )
    assert len(rows) == 1
    assert rows[0] == {
        "league": "wnba",
        "player_name": "Rhyne Howard",
        "market_type": "player_assists",
        "stat_category": "assists",
        "side": "over",
        "line_score": 3.5,
        "american_price": -114,
        "scraped_at": scraped,
    }


def test_sharp_props_to_book_rows_rejects_unknown_book():
    scraped = datetime(2026, 8, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="unsupported sportsbook"):
        sharp_props_to_book_rows(
            [], sportsbook="betmgm", league="wnba", scraped_at=scraped
        )
