"""Shared WNBA team name → abbreviation helpers for odds providers."""

from __future__ import annotations

import re

from app.services.wnba_scoreboard import canonical_abbrev

_TRICODE_RE = re.compile(r"^[A-Z]{2,3}$")

NAME_TO_ABBREV = {
    "atlanta dream": "ATL",
    "atl dream": "ATL",
    "chicago sky": "CHI",
    "chi sky": "CHI",
    "connecticut sun": "CON",
    "con sun": "CON",
    "dallas wings": "DAL",
    "dal wings": "DAL",
    "golden state valkyries": "GSV",
    "gs valkyries": "GSV",
    "indiana fever": "IND",
    "ind fever": "IND",
    "las vegas aces": "LVA",
    "lv aces": "LVA",
    "los angeles sparks": "LAS",
    "la sparks": "LAS",
    "minnesota lynx": "MIN",
    "min lynx": "MIN",
    "new york liberty": "NYL",
    "ny liberty": "NYL",
    "toronto tempo": "TOR",
    "tor tempo": "TOR",
    "phoenix mercury": "PHO",
    "phx mercury": "PHO",
    "portland fire": "PDX",
    "por fire": "PDX",
    "pdx fire": "PDX",
    "seattle storm": "SEA",
    "sea storm": "SEA",
    "washington mystics": "WAS",
    "was mystics": "WAS",
    "wsh mystics": "WAS",
}


def abbrev_from_team_name(label: str | None) -> str | None:
    """Map a full team name or leading tricode to a canonical WNBA abbrev."""
    text = str(label or "").strip()
    if not text:
        return None

    mapped = NAME_TO_ABBREV.get(text.lower())
    if mapped:
        return canonical_abbrev(mapped)

    first = text.split()[0].upper()
    if _TRICODE_RE.match(first):
        return canonical_abbrev(first)

    return None
