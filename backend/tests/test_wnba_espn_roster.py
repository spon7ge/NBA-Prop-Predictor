import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import app.services.wnba_espn_roster as roster_svc
from app.services.wnba_espn_roster import (
    enrich_starters,
    norm_player_name,
    roster_player_index,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_roster_index_and_enrich_angel_reese_jersey():
    payload = json.loads((FIXTURES / "espn_wnba_roster_atl.json").read_text())
    index = roster_player_index(payload)
    assert index[norm_player_name("Angel Reese")]["jersey"] == "5"
    starters = [
        {"name": "Allisha Gray", "position": "G"},
        {"name": "Jordin Canada", "position": "G"},
        {"name": "Rhyne Howard", "position": "G"},
        {"name": "Naz Hillmon", "position": "F"},
        {"name": "Angel Reese", "position": "F"},
    ]
    enriched = enrich_starters(starters, index)
    assert enriched[-1].name == "Angel Reese"
    assert enriched[-1].jersey == "5"
    assert enriched[-1].position == "F"


def test_enrich_starters_null_jersey_when_unmatched():
    enriched = enrich_starters(
        [{"name": "Unknown Player", "position": "G"}],
        {},
    )
    assert enriched[0].jersey is None
    assert enriched[0].name == "Unknown Player"


def test_roster_index_sea_flaujae_accent_match():
    payload = json.loads((FIXTURES / "espn_wnba_roster_sea.json").read_text())
    index = roster_player_index(payload)
    key = norm_player_name("Flau'jae Johnson")
    assert index[key]["jersey"] == "22"
    assert index[key]["position"] == "G"


def test_enrich_prefers_rotowire_position_over_roster():
    payload = json.loads((FIXTURES / "espn_wnba_roster_atl.json").read_text())
    index = roster_player_index(payload)
    enriched = enrich_starters(
        [{"name": "Angel Reese", "position": "C"}],
        index,
    )
    assert enriched[0].position == "C"


def test_enrich_falls_back_to_roster_position():
    payload = json.loads((FIXTURES / "espn_wnba_roster_atl.json").read_text())
    index = roster_player_index(payload)
    enriched = enrich_starters(
        [{"name": "Angel Reese", "position": ""}],
        index,
    )
    assert enriched[0].position == "F"


def test_get_roster_index_caches_fetch():
    roster_svc.clear_roster_cache()
    payload = json.loads((FIXTURES / "espn_wnba_roster_atl.json").read_text())
    calls = {"n": 0}

    async def fake_fetch(team_id: str):
        calls["n"] += 1
        return payload

    with patch.object(roster_svc, "fetch_espn_roster", side_effect=fake_fetch):
        asyncio.run(roster_svc.get_roster_index("20"))
        asyncio.run(roster_svc.get_roster_index("20"))
    assert calls["n"] == 1
