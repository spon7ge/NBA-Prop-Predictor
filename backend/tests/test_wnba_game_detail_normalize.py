import json
from pathlib import Path

from app.services.wnba_game_detail import normalize_espn_summary

FIXTURES = Path(__file__).parent / "fixtures"


def test_normalize_espn_summary_header_shots_plays():
    payload = json.loads((FIXTURES / "espn_wnba_summary.json").read_text())
    detail = normalize_espn_summary(
        payload, espn_event_id="401857098", fetched_at="2026-07-29T19:00:00-04:00"
    )
    assert detail.espn_event_id == "401857098"
    assert detail.league == "wnba"
    assert detail.status == "live"
    assert detail.status_label == "4:13 - 1st"
    assert detail.venue == "Mortgage Matchup Center"
    assert detail.away.abbrev == "GS"
    assert detail.away.score == 10
    assert detail.away.color.startswith("#")
    assert detail.home.abbrev == "PHX"
    assert detail.home.score == 9
    assert detail.fg_attempted == 2
    assert detail.fg_made == 1
    assert len(detail.shots) == 2
    made = next(s for s in detail.shots if s.made)
    assert made.player_name == "Laeticia Amihere"
    assert made.x == 25
    assert made.y == 5
    assert detail.latest_play is not None
    assert "Burton" in detail.latest_play.text
    assert len(detail.plays) == 3
    scoring = [p for p in detail.plays if p.scoring]
    assert len(scoring) == 1
    assert scoring[0].away_score == 10
    assert scoring[0].home_score == 8
