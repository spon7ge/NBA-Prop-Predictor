from __future__ import annotations

import json
from pathlib import Path

from app.schemas.wnba_scoreboard import WnbaGame, WnbaTeam
from app.services.wnba_scoreboard import (
    cache_ttl_seconds,
    merge_games,
    normalize_espn_scoreboard,
    normalize_stats_scoreboard,
    prefer_complete,
)

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
