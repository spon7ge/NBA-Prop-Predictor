"""Tests for Parlay main-line selection (sportsbook balance + sharp DFS match)."""

from __future__ import annotations

from src.odds.parlay_main_lines import SHARPNESS_ORDER, select_parlay_main_lines


def test_sharpness_order_lists_all_sportsbooks():
    assert set(SHARPNESS_ORDER) == {
        "pinnacle",
        "novig",
        "bet365",
        "draftkings",
        "fanduel",
        "caesars",
        "betmgm",
    }
    assert SHARPNESS_ORDER[0] == "pinnacle"


def test_sportsbook_keeps_balanced_main_drops_alt():
    rows = [
        {
            "bookmaker": "fanduel",
            "player": "A'ja Wilson",
            "market_key": "player_assists",
            "line": 3.5,
            "over_price": -110,
            "under_price": -110,
        },
        {
            "bookmaker": "fanduel",
            "player": "A'ja Wilson",
            "market_key": "player_assists",
            "line": 5.5,
            "over_price": 150,
            "under_price": -200,
        },
    ]
    out = select_parlay_main_lines(rows)
    assert len(out) == 1
    assert out[0]["line"] == 3.5


def test_dfs_matches_sharpest_sportsbook_line_when_offered():
    rows = [
        {
            "bookmaker": "pinnacle",
            "player": "A'ja Wilson",
            "market_key": "player_assists",
            "line": 4.5,
            "over_price": -110,
            "under_price": -110,
        },
        {
            "bookmaker": "fanduel",
            "player": "A'ja Wilson",
            "market_key": "player_assists",
            "line": 3.5,
            "over_price": -114,
            "under_price": -110,
        },
        {
            "bookmaker": "prizepicks",
            "player": "A'ja Wilson",
            "market_key": "player_assists",
            "line": 3.5,
            "over_price": -100,
            "under_price": -100,
        },
        {
            "bookmaker": "prizepicks",
            "player": "A'ja Wilson",
            "market_key": "player_assists",
            "line": 4.5,
            "over_price": -100,
            "under_price": -100,
        },
        {
            "bookmaker": "prizepicks",
            "player": "A'ja Wilson",
            "market_key": "player_assists",
            "line": 7.5,
            "over_price": -100,
            "under_price": -100,
        },
    ]
    out = select_parlay_main_lines(rows)
    pp = [r for r in out if r["bookmaker"] == "prizepicks"]
    assert len(pp) == 1
    assert pp[0]["line"] == 4.5  # pinnacle wins over FanDuel


def test_dfs_walks_down_sharpness_when_sharper_line_missing_on_dfs():
    rows = [
        {
            "bookmaker": "pinnacle",
            "player": "A'ja Wilson",
            "market_key": "player_assists",
            "line": 4.5,
            "over_price": -110,
            "under_price": -110,
        },
        {
            "bookmaker": "fanduel",
            "player": "A'ja Wilson",
            "market_key": "player_assists",
            "line": 3.5,
            "over_price": -114,
            "under_price": -110,
        },
        {
            "bookmaker": "prizepicks",
            "player": "A'ja Wilson",
            "market_key": "player_assists",
            "line": 3.5,
            "over_price": -100,
            "under_price": -100,
        },
        {
            "bookmaker": "prizepicks",
            "player": "A'ja Wilson",
            "market_key": "player_assists",
            "line": 7.5,
            "over_price": -100,
            "under_price": -100,
        },
    ]
    out = select_parlay_main_lines(rows)
    pp = next(r for r in out if r["bookmaker"] == "prizepicks")
    # Pinnacle 4.5 not on PP → FanDuel 3.5 matches.
    assert pp["line"] == 3.5


def test_dfs_falls_back_when_no_sportsbook_line_matches():
    rows = [
        {
            "bookmaker": "fanduel",
            "player": "A'ja Wilson",
            "market_key": "player_assists",
            "line": 3.5,
            "over_price": -114,
            "under_price": -110,
        },
        {
            "bookmaker": "prizepicks",
            "player": "A'ja Wilson",
            "market_key": "player_assists",
            "line": 5.5,
            "over_price": -100,
            "under_price": -100,
        },
        {
            "bookmaker": "prizepicks",
            "player": "A'ja Wilson",
            "market_key": "player_assists",
            "line": 7.5,
            "over_price": -100,
            "under_price": -100,
        },
    ]
    out = select_parlay_main_lines(rows)
    pp = next(r for r in out if r["bookmaker"] == "prizepicks")
    # No exact match to 3.5 → keep DFS fallback (balance / smaller line).
    assert pp["line"] == 5.5


def test_dfs_falls_back_without_any_sportsbook():
    rows = [
        {
            "bookmaker": "prizepicks",
            "player": "A'ja Wilson",
            "market_key": "player_assists",
            "line": 3.5,
            "over_price": -100,
            "under_price": -100,
        },
        {
            "bookmaker": "prizepicks",
            "player": "A'ja Wilson",
            "market_key": "player_assists",
            "line": 7.5,
            "over_price": -100,
            "under_price": -100,
        },
    ]
    out = select_parlay_main_lines(rows)
    assert len(out) == 1
    assert out[0]["line"] == 3.5


def test_books_filter_still_uses_excluded_sportsbooks_for_match():
    rows = [
        {
            "bookmaker": "pinnacle",
            "player": "A'ja Wilson",
            "market_key": "player_assists",
            "line": 3.5,
            "over_price": -108,
            "under_price": -112,
        },
        {
            "bookmaker": "underdog",
            "player": "A'ja Wilson",
            "market_key": "player_assists",
            "line": 3.5,
            "over_price": -110,
            "under_price": None,
        },
        {
            "bookmaker": "underdog",
            "player": "A'ja Wilson",
            "market_key": "player_assists",
            "line": 6.5,
            "over_price": -110,
            "under_price": None,
        },
    ]
    out = select_parlay_main_lines(rows, books=frozenset({"underdog"}))
    assert len(out) == 1
    assert out[0]["bookmaker"] == "underdog"
    assert out[0]["line"] == 3.5
