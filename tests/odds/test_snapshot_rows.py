from datetime import datetime, timezone

from src.odds.snapshot_rows import (
    parse_american_price,
    prizepicks_projections_to_rows,
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
