from __future__ import annotations

import json
from pathlib import Path

from app.services.wnba_standings import normalize_espn_standings

FIXTURES = Path(__file__).parent / "fixtures"


def _payload():
    return json.loads((FIXTURES / "espn_wnba_standings.json").read_text())


def test_normalize_east_then_west_order_and_season():
    result = normalize_espn_standings(_payload())
    assert result.season == 2026
    assert [c.key for c in result.conferences] == ["east", "west"]
    assert [c.label for c in result.conferences] == [
        "Eastern Conference",
        "Western Conference",
    ]
    assert len(result.conferences[0].teams) == 2
    assert len(result.conferences[1].teams) == 2


def test_normalize_maps_row_fields():
    east = normalize_espn_standings(_payload()).conferences[0]
    ind = east.teams[0]
    assert ind.rank == 1
    assert ind.team_id == "5"
    assert ind.abbrev == "IND"
    assert ind.name == "Indiana Fever"
    assert ind.logo_url == "https://a.espncdn.com/i/teamlogos/wnba/500/ind.png"
    assert ind.wins == 18
    assert ind.losses == 10
    assert ind.wl == "18-10"
    assert ind.pct == ".643"
    assert ind.gb == "-"
    assert ind.home == "11-5"
    assert ind.away == "7-5"
    assert ind.l10 == "8-2"
    assert ind.diff == "+169"
    assert ind.streak == "W4"


def test_normalize_skips_incomplete_and_null_logo():
    result = normalize_espn_standings(_payload())
    east_names = [t.name for t in result.conferences[0].teams]
    assert "Broken Entry" not in east_names
    lv = result.conferences[1].teams[1]
    assert lv.abbrev == "LV"
    assert lv.logo_url is None


def test_normalize_empty_children():
    result = normalize_espn_standings({"season": {"year": 2026}, "children": []})
    assert result.season == 2026
    assert result.conferences == []
