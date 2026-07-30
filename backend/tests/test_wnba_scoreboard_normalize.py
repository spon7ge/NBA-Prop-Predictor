from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.schemas.wnba_scoreboard import WnbaGame, WnbaTeam
from app.services.wnba_scoreboard import (
    cache_ttl_seconds,
    canonical_abbrev,
    format_tip_label,
    merge_games,
    normalize_espn_scoreboard,
    normalize_stats_scoreboard,
    prefer_complete,
)

FIXTURES = Path(__file__).parent / "fixtures"


def espn_event(
    *,
    status_type: dict,
    period: int | None = None,
    clock: str = "",
    away_score: str | None = None,
    home_score: str | None = None,
    date: str = "2026-07-29T23:00Z",
    away_abbrev: str = "ATL",
    home_abbrev: str = "DAL",
    event_id: str = "401749001",
) -> dict:
    """Build a minimal ESPN scoreboard payload with one event."""
    return {
        "events": [
            {
                "id": event_id,
                "date": date,
                "competitions": [
                    {
                        "competitors": [
                            {
                                "homeAway": "home",
                                "score": home_score,
                                "team": {
                                    "abbreviation": home_abbrev,
                                    "displayName": "Dallas Wings",
                                },
                            },
                            {
                                "homeAway": "away",
                                "score": away_score,
                                "team": {
                                    "abbreviation": away_abbrev,
                                    "displayName": "Atlanta Dream",
                                },
                            },
                        ]
                    }
                ],
                "status": {
                    "type": status_type,
                    "period": period,
                    "displayClock": clock,
                },
            }
        ]
    }


def test_normalize_espn_sets_espn_event_id():
    payload = json.loads((FIXTURES / "espn_wnba_scoreboard.json").read_text())
    g = normalize_espn_scoreboard(payload, date_et="2026-07-29")[0]
    assert g.espn_event_id == "401749001"


def test_merge_preserves_espn_event_id_when_stats_id_wins():
    espn = [
        WnbaGame(
            id="espn-401749001",
            espn_event_id="401749001",
            status="live",
            status_label="Q3 7:13",
            away=WnbaTeam(abbrev="ATL", name="Atlanta Dream", score=36),
            home=WnbaTeam(abbrev="DAL", name="Dallas Wings", score=44),
            start_time_et="2026-07-29T23:00:00Z",
        )
    ]
    stats = [
        WnbaGame(
            id="1022600123",
            espn_event_id=None,
            status="live",
            status_label="Q3 7:13",
            away=WnbaTeam(abbrev="ATL", name="Atlanta Dream", score=36),
            home=WnbaTeam(abbrev="DAL", name="Dallas Wings", score=44),
            start_time_et="2026-07-29T23:00:00Z",
        )
    ]
    merged = merge_games(espn, stats)
    assert len(merged) == 1
    assert merged[0].id == "1022600123"
    assert merged[0].espn_event_id == "401749001"


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


def test_normalize_stats_live_game():
    payload = json.loads((FIXTURES / "stats_wnba_scoreboard.json").read_text())
    games = normalize_stats_scoreboard(payload, date_et="2026-07-29")
    assert len(games) == 1
    g = games[0]
    assert g.id == "1022600123"
    assert g.status == "live"
    assert g.away.abbrev == "ATL"
    assert g.home.score == 45


def test_merge_prefers_non_null_and_richer_fields():
    espn = [
        WnbaGame(
            id="espn-1",
            status="live",
            status_label="Q3 7:13",
            away=WnbaTeam(abbrev="ATL", name="Atlanta Dream", score=36),
            home=WnbaTeam(abbrev="DAL", name="Dallas Wings", score=44),
            start_time_et="2026-07-29T23:00:00Z",
        )
    ]
    stats = [
        WnbaGame(
            id="1022600123",
            status="live",
            status_label="Q3 7:10",
            away=WnbaTeam(abbrev="ATL", name="Atlanta Dream", score=36),
            home=WnbaTeam(abbrev="DAL", name="Dallas Wings", score=45),
            start_time_et="2026-07-29T23:00:00Z",
        )
    ]
    merged = merge_games(espn, stats)
    assert len(merged) == 1
    assert merged[0].id == "1022600123"  # prefer stats id
    assert merged[0].home.score == 45  # prefer non-stale higher completeness: non-null from stats


def test_merge_fills_null_score_from_other_source():
    espn = [
        WnbaGame(
            id="espn-1",
            status="live",
            status_label="Q3 7:13",
            away=WnbaTeam(abbrev="ATL", name="Atlanta Dream", score=36),
            home=WnbaTeam(abbrev="DAL", name="Dallas Wings", score=None),
            start_time_et="2026-07-29T23:00:00Z",
        )
    ]
    stats = [
        WnbaGame(
            id="1022600123",
            status="live",
            status_label="Q3 7:10",
            away=WnbaTeam(abbrev="ATL", name="Atlanta Dream", score=36),
            home=WnbaTeam(abbrev="DAL", name="Dallas Wings", score=45),
            start_time_et="2026-07-29T23:00:00Z",
        )
    ]
    merged = merge_games(espn, stats)
    assert merged[0].home.score == 45


def test_merge_prefers_richer_team_name():
    espn = [
        WnbaGame(
            id="espn-1",
            status="live",
            status_label="Q3 7:13",
            away=WnbaTeam(abbrev="ATL", name="Dream", score=36),
            home=WnbaTeam(abbrev="DAL", name="Wings", score=44),
            start_time_et="2026-07-29T23:00:00Z",
        )
    ]
    stats = [
        WnbaGame(
            id="1022600123",
            status="live",
            status_label="Q3 7:10",
            away=WnbaTeam(abbrev="ATL", name="Atlanta Dream", score=36),
            home=WnbaTeam(abbrev="DAL", name="Dallas Wings", score=45),
            start_time_et="2026-07-29T23:00:00Z",
        )
    ]
    merged = merge_games(espn, stats)
    assert merged[0].away.name == "Atlanta Dream"
    assert merged[0].home.name == "Dallas Wings"


def test_merge_keeps_unmatched_game():
    espn = [
        WnbaGame(
            id="espn-1",
            status="scheduled",
            status_label="Scheduled",
            away=WnbaTeam(abbrev="ATL", name="Atlanta Dream", score=None),
            home=WnbaTeam(abbrev="DAL", name="Dallas Wings", score=None),
            start_time_et="2026-07-29T23:00:00Z",
        )
    ]
    stats = [
        WnbaGame(
            id="1022600999",
            status="scheduled",
            status_label="Scheduled",
            away=WnbaTeam(abbrev="NYL", name="New York Liberty", score=None),
            home=WnbaTeam(abbrev="LVA", name="Las Vegas Aces", score=None),
            start_time_et="2026-07-29T01:00:00Z",
        )
    ]
    merged = merge_games(espn, stats)
    assert len(merged) == 2
    abbrevs = {(g.away.abbrev, g.home.abbrev) for g in merged}
    assert ("ATL", "DAL") in abbrevs
    assert ("NYL", "LVA") in abbrevs


def test_merge_status_label_coherent_on_final_vs_live():
    espn = [
        WnbaGame(
            id="espn-1",
            status="final",
            status_label="Final",
            away=WnbaTeam(abbrev="ATL", name="Atlanta Dream", score=80),
            home=WnbaTeam(abbrev="DAL", name="Dallas Wings", score=75),
            start_time_et="2026-07-29T23:00:00Z",
        )
    ]
    stats = [
        WnbaGame(
            id="1022600123",
            status="live",
            status_label="Q4 0:12 — very detailed live label",
            away=WnbaTeam(abbrev="ATL", name="Atlanta Dream", score=78),
            home=WnbaTeam(abbrev="DAL", name="Dallas Wings", score=73),
            start_time_et="2026-07-29T23:00:00Z",
        )
    ]
    merged = merge_games(espn, stats)
    assert merged[0].status == "final"
    assert merged[0].status_label == "Final"


def test_prefer_complete_helpers():
    assert prefer_complete(None, 45) == 45
    assert prefer_complete(36, None) == 36
    assert prefer_complete(36, 45) == 45
    assert prefer_complete("", "Dallas Wings") == "Dallas Wings"
    assert prefer_complete("Dream", "Atlanta Dream") == "Atlanta Dream"


def test_normalize_stats_final_game():
    payload = {
        "scoreboard": {
            "games": [
                {
                    "gameId": "1022600456",
                    "gameStatus": 3,
                    "gameStatusText": "Final",
                    "gameTimeUTC": "2026-07-29T23:00:00Z",
                    "homeTeam": {
                        "teamTricode": "DAL",
                        "teamName": "Wings",
                        "teamCity": "Dallas",
                        "score": 75,
                    },
                    "awayTeam": {
                        "teamTricode": "ATL",
                        "teamName": "Dream",
                        "teamCity": "Atlanta",
                        "score": 80,
                    },
                }
            ]
        }
    }
    games = normalize_stats_scoreboard(payload, date_et="2026-07-29")
    assert len(games) == 1
    g = games[0]
    assert g.status == "final"
    assert g.status_label == "Final"
    assert g.away.score == 80
    assert g.home.score == 75


def test_normalize_stats_halftime_game():
    payload = {
        "scoreboard": {
            "games": [
                {
                    "gameId": "1022600789",
                    "gameStatus": 2,
                    "gameStatusText": "Halftime",
                    "gameTimeUTC": "2026-07-29T23:00:00Z",
                    "homeTeam": {
                        "teamTricode": "DAL",
                        "teamName": "Wings",
                        "teamCity": "Dallas",
                        "score": 40,
                    },
                    "awayTeam": {
                        "teamTricode": "ATL",
                        "teamName": "Dream",
                        "teamCity": "Atlanta",
                        "score": 38,
                    },
                }
            ]
        }
    }
    games = normalize_stats_scoreboard(payload, date_et="2026-07-29")
    assert len(games) == 1
    g = games[0]
    assert g.status == "halftime"
    assert g.status_label == "Halftime"
    assert g.away.score == 38
    assert g.home.score == 40


def test_cache_ttl_live_vs_final():
    live = [
        WnbaGame(
            id="1",
            status="live",
            status_label="Q1 10:00",
            away=WnbaTeam(abbrev="ATL", name="A", score=0),
            home=WnbaTeam(abbrev="DAL", name="D", score=0),
            start_time_et="2026-07-29T23:00:00Z",
        )
    ]
    final = [
        WnbaGame(
            id="1",
            status="final",
            status_label="Final",
            away=WnbaTeam(abbrev="ATL", name="A", score=80),
            home=WnbaTeam(abbrev="DAL", name="D", score=75),
            start_time_et="2026-07-29T23:00:00Z",
        )
    ]
    assert cache_ttl_seconds(live) == 30
    assert cache_ttl_seconds(final) == 60
    assert cache_ttl_seconds([]) == 60


def test_cache_ttl_halftime():
    halftime = [
        WnbaGame(
            id="1",
            status="halftime",
            status_label="Halftime",
            away=WnbaTeam(abbrev="ATL", name="A", score=40),
            home=WnbaTeam(abbrev="DAL", name="D", score=38),
            start_time_et="2026-07-29T23:00:00Z",
        )
    ]
    assert cache_ttl_seconds(halftime) == 30


def test_normalize_espn_scheduled_game_uses_et_tip_label():
    payload = espn_event(
        status_type={
            "state": "pre",
            "completed": False,
            "name": "STATUS_SCHEDULED",
            "shortDetail": "7/29 - 10:00 PM EDT",
        },
        date="2026-07-29T23:00Z",
    )
    games = normalize_espn_scoreboard(payload, date_et="2026-07-29")
    assert len(games) == 1
    g = games[0]
    assert g.status == "scheduled"
    assert g.status_label == "7:00 PM ET"
    assert g.away.score is None
    assert g.home.score is None


def test_normalize_espn_halftime_game():
    payload = espn_event(
        status_type={
            "state": "in",
            "completed": False,
            "name": "STATUS_HALFTIME",
            "shortDetail": "Halftime",
        },
        period=2,
        clock="0:00",
        away_score="38",
        home_score="40",
    )
    games = normalize_espn_scoreboard(payload, date_et="2026-07-29")
    g = games[0]
    assert g.status == "halftime"
    assert g.status_label == "Halftime"
    assert g.away.score == 38
    assert g.home.score == 40


def test_normalize_espn_final_game():
    payload = espn_event(
        status_type={
            "state": "post",
            "completed": True,
            "name": "STATUS_FINAL",
            "shortDetail": "Final",
        },
        period=4,
        away_score="80",
        home_score="75",
    )
    games = normalize_espn_scoreboard(payload, date_et="2026-07-29")
    g = games[0]
    assert g.status == "final"
    assert g.status_label == "Final"
    assert g.away.score == 80
    assert g.home.score == 75


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("STATUS_POSTPONED", "Postponed"),
        ("STATUS_CANCELED", "Canceled"),
        ("STATUS_CANCELLED", "Canceled"),
        ("STATUS_SUSPENDED", "Suspended"),
    ],
)
def test_normalize_espn_non_result_states_are_not_final(name, label):
    payload = espn_event(
        status_type={
            "state": "post",
            "completed": True,
            "name": name,
            "shortDetail": label,
        },
    )
    g = normalize_espn_scoreboard(payload, date_et="2026-07-29")[0]
    assert g.status == "scheduled"
    assert g.status_label == label


def test_normalize_stats_scheduled_uses_et_tip_label():
    payload = {
        "scoreboard": {
            "games": [
                {
                    "gameId": "1022600321",
                    "gameStatus": 1,
                    "gameStatusText": "7:00 pm ET",
                    "gameTimeUTC": "2026-07-29T23:00:00Z",
                    "homeTeam": {
                        "teamTricode": "DAL",
                        "teamName": "Wings",
                        "teamCity": "Dallas",
                        "score": 0,
                    },
                    "awayTeam": {
                        "teamTricode": "ATL",
                        "teamName": "Dream",
                        "teamCity": "Atlanta",
                        "score": 0,
                    },
                }
            ]
        }
    }
    g = normalize_stats_scoreboard(payload, date_et="2026-07-29")[0]
    assert g.status == "scheduled"
    assert g.status_label == "7:00 PM ET"
    assert g.away.score is None


def test_format_tip_label_strips_leading_zero_and_handles_junk():
    assert format_tip_label("2026-07-30T00:30:00Z") == "8:30 PM ET"
    assert format_tip_label("2026-07-29T23:00Z") == "7:00 PM ET"
    assert format_tip_label("not-a-date") is None
    assert format_tip_label("") is None


def test_canonical_abbrev_aliases_espn_short_codes():
    assert canonical_abbrev("GS") == "GSV"
    assert canonical_abbrev("LA") == "LAS"
    assert canonical_abbrev("LV") == "LVA"
    assert canonical_abbrev("NY") == "NYL"
    assert canonical_abbrev("PHX") == "PHO"
    assert canonical_abbrev("WSH") == "WAS"
    # Canonical spellings and unknown codes pass through unchanged.
    assert canonical_abbrev("GSV") == "GSV"
    assert canonical_abbrev("atl") == "ATL"


def test_merge_matches_espn_short_code_to_stats_tricode():
    espn = [
        WnbaGame(
            id="espn-1",
            status="live",
            status_label="Q3 7:13",
            away=WnbaTeam(abbrev="GS", name="Golden State Valkyries", score=36),
            home=WnbaTeam(abbrev="NY", name="New York Liberty", score=44),
            start_time_et="2026-07-29T23:00:00Z",
        )
    ]
    stats = [
        WnbaGame(
            id="1022600123",
            status="live",
            status_label="Q3 7:10",
            away=WnbaTeam(abbrev="GSV", name="Golden State Valkyries", score=36),
            home=WnbaTeam(abbrev="NYL", name="New York Liberty", score=45),
            start_time_et="2026-07-29T23:00:00Z",
        )
    ]
    merged = merge_games(espn, stats)
    assert len(merged) == 1
    assert merged[0].id == "1022600123"
    assert merged[0].home.score == 45


def test_merge_falls_back_to_tip_time_window_when_abbrevs_miss():
    espn = [
        WnbaGame(
            id="espn-1",
            status="live",
            status_label="Q3 7:13",
            away=WnbaTeam(abbrev="ZZZ", name="Atlanta Dream", score=36),
            home=WnbaTeam(abbrev="DAL", name="Dallas Wings", score=44),
            start_time_et="2026-07-29T23:00:00Z",
        )
    ]
    stats = [
        WnbaGame(
            id="1022600123",
            status="live",
            status_label="Q3 7:10",
            away=WnbaTeam(abbrev="ATL", name="Atlanta Dream", score=36),
            home=WnbaTeam(abbrev="DAL", name="Dallas Wings", score=45),
            start_time_et="2026-07-29T23:10:00Z",
        )
    ]
    merged = merge_games(espn, stats)
    assert len(merged) == 1
    assert merged[0].id == "1022600123"
    assert merged[0].home.score == 45


def test_merge_keeps_games_apart_when_tip_times_are_far_off():
    espn = [
        WnbaGame(
            id="espn-1",
            status="scheduled",
            status_label="7:00 PM ET",
            away=WnbaTeam(abbrev="ZZZ", name="Atlanta Dream", score=None),
            home=WnbaTeam(abbrev="QQQ", name="Dallas Wings", score=None),
            start_time_et="2026-07-29T23:00:00Z",
        )
    ]
    stats = [
        WnbaGame(
            id="1022600123",
            status="scheduled",
            status_label="10:00 PM ET",
            away=WnbaTeam(abbrev="ATL", name="Atlanta Dream", score=None),
            home=WnbaTeam(abbrev="DAL", name="Dallas Wings", score=None),
            start_time_et="2026-07-30T02:00:00Z",
        )
    ]
    assert len(merge_games(espn, stats)) == 2


def test_merge_espn_game_pairs_with_only_one_stats_game():
    espn = [
        WnbaGame(
            id="espn-1",
            status="live",
            status_label="Q1 5:00",
            away=WnbaTeam(abbrev="GS", name="Golden State Valkyries", score=10),
            home=WnbaTeam(abbrev="NY", name="New York Liberty", score=12),
            start_time_et="2026-07-29T23:00:00Z",
        )
    ]
    stats = [
        WnbaGame(
            id="1022600001",
            status="live",
            status_label="Q1 4:55",
            away=WnbaTeam(abbrev="GSV", name="Golden State Valkyries", score=10),
            home=WnbaTeam(abbrev="NYL", name="New York Liberty", score=13),
            start_time_et="2026-07-29T23:00:00Z",
        ),
        WnbaGame(
            id="1022600002",
            status="live",
            status_label="Q1 4:55",
            away=WnbaTeam(abbrev="ATL", name="Atlanta Dream", score=8),
            home=WnbaTeam(abbrev="DAL", name="Dallas Wings", score=11),
            start_time_et="2026-07-29T23:00:00Z",
        ),
    ]
    merged = merge_games(espn, stats)
    assert len(merged) == 2
    ids = {g.id for g in merged}
    assert ids == {"1022600001", "1022600002"}
