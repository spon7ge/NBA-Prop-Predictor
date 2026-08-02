from __future__ import annotations

import json
from pathlib import Path

from app.services.mlb_scoreboard import normalize_mlb_schedule

FIXTURES = Path(__file__).parent / "fixtures"


def test_normalize_mlb_schedule_maps_preview_live_final():
    payload = json.loads((FIXTURES / "mlb_statsapi_schedule.json").read_text())
    games = normalize_mlb_schedule(payload, date_et="2026-08-02")
    by_pk = {g.mlb_game_pk: g for g in games}

    preview = by_pk["824900"]
    assert preview.league == "mlb"
    assert preview.id == "mlb-824900"
    assert preview.status == "scheduled"
    assert preview.away.abbrev == "NYY"
    assert preview.home.abbrev == "NYM"
    assert preview.away.score is None  # hide scores for scheduled
    assert preview.away.logo_url == "https://www.mlbstatic.com/team-logos/147.svg"
    assert preview.venue == "Citi Field"

    live = by_pk["824971"]
    assert live.status == "live"
    assert live.status_label == "Top 8th"
    assert live.away.score == 9
    assert live.home.score == 0
    assert live.away.record == "40-60"

    final = by_pk["824807"]
    assert final.status == "final"
    assert final.status_label == "Final"
    assert final.away.score == 8


def test_normalize_postponed_is_scheduled_with_label():
    payload = {
        "dates": [
            {
                "date": "2026-08-02",
                "games": [
                    {
                        "gamePk": 1,
                        "gameDate": "2026-08-02T23:00:00Z",
                        "status": {
                            "abstractGameState": "Final",
                            "detailedState": "Postponed",
                            "codedGameState": "D",
                        },
                        "teams": {
                            "away": {
                                "team": {
                                    "id": 1,
                                    "abbreviation": "AAA",
                                    "name": "Away",
                                }
                            },
                            "home": {
                                "team": {
                                    "id": 2,
                                    "abbreviation": "HHH",
                                    "name": "Home",
                                }
                            },
                        },
                    }
                ],
            }
        ]
    }
    games = normalize_mlb_schedule(payload, date_et="2026-08-02")
    assert len(games) == 1
    assert games[0].status == "scheduled"
    assert games[0].status_label == "Postponed"
