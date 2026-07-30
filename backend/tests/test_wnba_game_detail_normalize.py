import json
from pathlib import Path

from app.schemas.wnba_game_detail import GameDetailTeamStat
from app.services.wnba_game_detail import normalize_espn_summary

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_normalize_includes_win_probability_and_team_stats():
    payload = load_fixture("espn_wnba_summary_with_winprobability.json")

    detail = normalize_espn_summary(
        payload,
        espn_event_id="401857098",
        fetched_at="2026-07-30T00:00:00-04:00",
    )

    assert detail.win_probability is not None
    assert detail.win_probability.summary is None
    assert len(detail.win_probability.timeline) == 2

    first = detail.win_probability.timeline[0]
    assert first.id == "40185709810"
    assert first.period == 1
    assert first.clock == "4:29"
    assert first.away_score == 10
    assert first.home_score == 8
    assert first.home_win_pct == 46
    assert first.away_win_pct == 54
    assert first.team_id == "129153"

    last = detail.win_probability.timeline[-1]
    assert last.id == "40185709811"
    assert last.home_win_pct == 54
    assert last.away_win_pct == 46
    assert last.away_score == 10
    assert last.home_score == 9

    assert detail.win_probability.team_stats == [
        GameDetailTeamStat(
            key="field_goal_pct",
            label="Field goal %",
            away_value=41,
            home_value=49,
        ),
        GameDetailTeamStat(
            key="three_point_pct",
            label="Three point %",
            away_value=36,
            home_value=31,
        ),
        GameDetailTeamStat(
            key="free_throw_pct",
            label="Free throw %",
            away_value=79,
            home_value=74,
        ),
        GameDetailTeamStat(
            key="rebounds",
            label="Rebounds",
            away_value=33,
            home_value=34,
        ),
        GameDetailTeamStat(
            key="offensive_rebounds",
            label="Offensive rebounds",
            away_value=13,
            home_value=6,
        ),
        GameDetailTeamStat(
            key="assists",
            label="Assists",
            away_value=24,
            home_value=19,
        ),
    ]


def test_normalize_win_probability_allows_timeline_only():
    payload = load_fixture("espn_wnba_summary_with_winprobability.json")
    del payload["boxscore"]

    detail = normalize_espn_summary(
        payload,
        espn_event_id="401857098",
        fetched_at="2026-07-30T00:00:00-04:00",
    )

    assert detail.win_probability is not None
    assert len(detail.win_probability.timeline) == 2
    assert detail.win_probability.team_stats == []


def test_normalize_win_probability_allows_team_stats_only():
    payload = load_fixture("espn_wnba_summary_with_winprobability.json")
    del payload["winprobability"]

    detail = normalize_espn_summary(
        payload,
        espn_event_id="401857098",
        fetched_at="2026-07-30T00:00:00-04:00",
    )

    assert detail.win_probability is not None
    assert detail.win_probability.timeline == []
    assert len(detail.win_probability.team_stats) == 6
    assert detail.win_probability.team_stats[0].label == "Field goal %"


def test_normalize_win_probability_returns_none_when_missing_everything():
    payload = load_fixture("espn_wnba_summary.json")

    detail = normalize_espn_summary(
        payload,
        espn_event_id="401857098",
        fetched_at="2026-07-30T00:00:00-04:00",
    )

    assert detail.win_probability is None


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


def test_normalize_includes_matchup_prediction_leaders_injuries():
    payload = load_fixture("espn_wnba_summary_scheduled_preview.json")
    detail = normalize_espn_summary(
        payload,
        espn_event_id="401857099",
        fetched_at="2026-07-30T00:00:00-04:00",
    )
    assert detail.status == "scheduled"
    assert detail.matchup_prediction is not None
    assert detail.matchup_prediction.away_win_pct == 67
    assert detail.matchup_prediction.home_win_pct == 33
    assert detail.matchup_prediction.source_label == "ESPN game projection"
    assert detail.season_leaders is not None
    assert [r.stat for r in detail.season_leaders.away] == [
        "points",
        "assists",
        "rebounds",
    ]
    assert detail.season_leaders.away[0].name == "Olivia Miles"
    assert detail.season_leaders.away[0].value == "19.5"
    assert detail.season_leaders.home[0].name == "Marina Mabrey"
    assert detail.injuries is not None
    assert detail.injuries.away == []
    assert detail.injuries.home[0].name == "Nyara Sabally"
    assert detail.injuries.home[0].status == "Out"
    assert detail.injuries.home[0].detail == "Ribs"
    assert detail.injuries.home[0].position == "F"
    assert detail.projected_starters is None


def test_normalize_injuries_null_when_both_sides_empty():
    payload = load_fixture("espn_wnba_summary_scheduled_preview.json")
    payload["injuries"] = [
        {"team": {"id": "away1"}, "injuries": []},
        {"team": {"id": "home1"}, "injuries": []},
    ]
    detail = normalize_espn_summary(
        payload,
        espn_event_id="401857099",
        fetched_at="2026-07-30T00:00:00-04:00",
    )
    assert detail.injuries is None


def test_normalize_preview_fields_null_when_missing():
    payload = load_fixture("espn_wnba_summary.json")
    detail = normalize_espn_summary(
        payload,
        espn_event_id="401749001",
        fetched_at="2026-07-30T00:00:00-04:00",
    )
    assert detail.matchup_prediction is None
    assert detail.season_leaders is None
    assert detail.injuries is None
    assert detail.projected_starters is None


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
