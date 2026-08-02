from __future__ import annotations

import json
from pathlib import Path

from app.services import wnba_player as svc

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_format_pct_handles_fraction_and_percent():
    assert svc.format_pct(0.482) == "48.2"
    assert svc.format_pct(48.2) == "48.2"
    assert svc.format_pct(None) is None


def test_made_attempt():
    assert svc.made_attempt(11, 20) == "11-20"


def test_normalize_player_happy_path():
    result = svc.normalize_wnba_player(
        player_id="1628932",
        season=2026,
        dash=_load("stats_wnba_player_dash.json"),
        info=_load("stats_wnba_player_info.json"),
        gamelog=_load("stats_wnba_player_gamelog.json"),
    )
    assert result is not None
    assert result.player_id == "1628932"
    assert result.name == "A'ja Wilson"
    assert result.position  # from info
    assert result.team_abbrev == "LVA"
    assert result.averages.pts  # one-decimal string
    assert result.averages.fg_pct
    assert result.averages.fg3_pct
    assert len(result.games) >= 6
    g0 = result.games[0]
    assert g0.fg  # "m-a"
    assert g0.three_pt
    assert g0.ft
    assert result.source_label == "stats.wnba.com"
    assert result.headshot_url  # non-empty CDN URL containing player_id


def test_normalize_unknown_player_returns_none():
    result = svc.normalize_wnba_player(
        player_id="99999999",
        season=2026,
        dash=_load("stats_wnba_player_dash.json"),
        info=_load("stats_wnba_player_info.json"),
        gamelog=_load("stats_wnba_player_gamelog.json"),
    )
    assert result is None
