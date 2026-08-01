from __future__ import annotations

import json
from pathlib import Path

from app.services import parlay_odds as svc

FIXTURE = Path(__file__).parent / "fixtures" / "parlay_wnba_odds.json"


def _events() -> list[dict]:
    return json.loads(FIXTURE.read_text())


def test_normalize_prefers_pinnacle():
    games, sportsbook = svc.normalize_parlay_odds(_events())
    assert sportsbook == "pinnacle"
    atl = next(g for g in games if g.home_abbrev == "ATL")
    assert atl.away_abbrev == "CHI"
    assert atl.spread_team_abbrev == "ATL"
    assert atl.spread_line == -4.5
    assert atl.total == 162.5


def test_normalize_falls_back_to_draftkings():
    games, sportsbook = svc.normalize_parlay_odds(_events())
    nyl = next(g for g in games if g.home_abbrev == "NYL")
    assert nyl.spread_line == -8.5
    assert nyl.total == 170.5
    assert sportsbook == "pinnacle"


def test_normalize_dk_only_when_no_pinnacle():
    events = [
        {
            "home_team": "Seattle Storm",
            "away_team": "Phoenix Mercury",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "markets": [
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "Seattle Storm", "price": -110, "point": -2.5},
                                {"name": "Phoenix Mercury", "price": -110, "point": 2.5},
                            ],
                        }
                    ],
                }
            ],
        }
    ]
    games, sportsbook = svc.normalize_parlay_odds(events)
    assert sportsbook == "draftkings"
    assert games[0].home_abbrev == "SEA"
    assert games[0].spread_line == -2.5


def test_normalize_falls_back_beyond_draftkings():
    events = [
        {
            "home_team": "Minnesota Lynx",
            "away_team": "Indiana Fever",
            "bookmakers": [
                {
                    "key": "novig",
                    "markets": [
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "Minnesota Lynx", "price": -110, "point": -5.5},
                                {"name": "Indiana Fever", "price": -110, "point": 5.5},
                            ],
                        },
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "price": -110, "point": 193.5},
                                {"name": "Under", "price": -110, "point": 193.5},
                            ],
                        },
                    ],
                }
            ],
        }
    ]
    games, sportsbook = svc.normalize_parlay_odds(events)
    assert sportsbook == "novig"
    assert games[0].home_abbrev == "MIN"
    assert games[0].spread_line == -5.5
    assert games[0].total == 193.5
