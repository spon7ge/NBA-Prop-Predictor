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
    # 5 raw plays total, but free throws and shots without real coordinates
    # are excluded from `shots` (fg_attempted stays 2, not 4).
    assert len(detail.plays) == 5
    made_shot_texts = {s.player_name for s in detail.shots}
    assert "Veronica Burton" not in made_shot_texts
    assert "Kahleah Copper" not in made_shot_texts
    field_goal_scoring = [
        p for p in detail.plays if p.scoring and "free throw" not in p.text.lower()
    ]
    assert len(field_goal_scoring) == 1
    assert field_goal_scoring[0].away_score == 10
    assert field_goal_scoring[0].home_score == 8


def test_normalize_excludes_free_throws_and_missing_coordinates_from_shots():
    payload = json.loads((FIXTURES / "espn_wnba_summary.json").read_text())
    detail = normalize_espn_summary(
        payload, espn_event_id="401857098", fetched_at="2026-07-29T19:00:00-04:00"
    )
    shot_ids = {s.id for s in detail.shots}
    # Free throw (has coordinate {0,0} but is a free throw) is excluded.
    assert "40185709812" not in shot_ids
    # Shot with no coordinate object at all is excluded, not coerced to (0, 0).
    assert "40185709813" not in shot_ids
    assert detail.fg_made == 1
    assert detail.fg_attempted == 2


def test_normalize_excludes_null_coordinates_from_shots():
    payload = json.loads((FIXTURES / "espn_wnba_summary.json").read_text())
    payload["plays"].append(
        {
            "id": "40185709814",
            "text": "Alyssa Thomas misses layup",
            "awayScore": 10,
            "homeScore": 9,
            "scoringPlay": False,
            "shootingPlay": True,
            "scoreValue": 0,
            "period": {"number": 1},
            "clock": {"displayValue": "3:55"},
            "coordinate": {"x": None, "y": None},
            "team": {"id": "21"},
            "participants": [],
        }
    )
    detail = normalize_espn_summary(
        payload, espn_event_id="401857098", fetched_at="2026-07-29T19:00:00-04:00"
    )
    shot_ids = {s.id for s in detail.shots}
    assert "40185709814" not in shot_ids
    assert detail.fg_made == 1
    assert detail.fg_attempted == 2
