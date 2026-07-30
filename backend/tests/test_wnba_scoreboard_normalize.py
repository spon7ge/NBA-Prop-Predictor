from __future__ import annotations

import json
from pathlib import Path

from app.services.wnba_scoreboard import normalize_espn_scoreboard

FIXTURES = Path(__file__).parent / "fixtures"


def test_normalize_espn_live_game():
    payload = json.loads((FIXTURES / "espn_wnba_scoreboard.json").read_text())
    games = normalize_espn_scoreboard(payload, date_et="2026-07-29")
    assert len(games) == 1
    g = games[0]
    assert g.id == "espn-401749001"
    assert g.league == "wnba"
    assert g.status == "live"
    assert g.status_label == "Q3 7:13"
    assert g.away.abbrev == "ATL"
    assert g.away.name == "Atlanta Dream"
    assert g.away.score == 36
    assert g.home.abbrev == "DAL"
    assert g.home.score == 44
