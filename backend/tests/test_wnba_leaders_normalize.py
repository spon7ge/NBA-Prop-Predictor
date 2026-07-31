from __future__ import annotations

import json
from pathlib import Path

from app.services.wnba_leaders import normalize_leaguedashplayerstats

FIXTURES = Path(__file__).parent / "fixtures"


def _payload():
    return json.loads(
        (FIXTURES / "stats_wnba_leaguedashplayerstats.json").read_text()
    )


def test_normalize_six_categories_top_ten_order():
    result = normalize_leaguedashplayerstats(_payload(), season=2026)
    assert result.season == 2026
    assert result.pace == "per_game"
    keys = [c.key for c in result.categories]
    assert keys == [
        "points",
        "rebounds",
        "assists",
        "steals",
        "blocks",
        "three_pointers",
    ]
    assert [c.stat for c in result.categories] == [
        "PTS",
        "REB",
        "AST",
        "STL",
        "BLK",
        "3PM",
    ]
    for cat in result.categories:
        assert 1 <= len(cat.leaders) <= 10
        assert [r.rank for r in cat.leaders] == list(
            range(1, len(cat.leaders) + 1)
        )


def test_normalize_points_leader_and_truncation():
    result = normalize_leaguedashplayerstats(_payload(), season=2026)
    points = result.categories[0]
    assert points.leaders[0].name == "A'ja Wilson"
    assert points.leaders[0].team_abbrev == "LVA"
    assert points.leaders[0].gp == 25
    assert points.leaders[0].value == "26.2"
    assert points.leaders[0].player_id == "1001"
    assert len(points.leaders) == 10


def test_normalize_skips_incomplete_rows():
    result = normalize_leaguedashplayerstats(_payload(), season=2026)
    names = {
        row.name
        for cat in result.categories
        for row in cat.leaders
    }
    assert "Incomplete Row" not in names


def test_normalize_empty_result_set():
    empty = {
        "resultSets": [
            {"name": "LeagueDashPlayerStats", "headers": [], "rowSet": []}
        ]
    }
    result = normalize_leaguedashplayerstats(empty, season=2026)
    assert len(result.categories) == 6
    for cat in result.categories:
        assert cat.leaders == []
