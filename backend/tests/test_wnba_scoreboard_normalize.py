from __future__ import annotations

import json
from pathlib import Path

from app.schemas.wnba_scoreboard import WnbaGame, WnbaTeam
from app.services.wnba_scoreboard import (
    cache_ttl_seconds,
    merge_games,
    normalize_espn_scoreboard,
    normalize_stats_scoreboard,
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
