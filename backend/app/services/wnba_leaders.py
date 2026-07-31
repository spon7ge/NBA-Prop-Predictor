from __future__ import annotations

import logging
from typing import Any

from app.schemas.wnba_leaders import (
    WnbaLeaderCategory,
    WnbaLeaderRow,
    WnbaLeadersResponse,
)

logger = logging.getLogger(__name__)

_CATEGORY_SPECS: list[tuple[str, str, str, str]] = [
    # key, label, display_stat, upstream_header
    ("points", "Points", "PTS", "PTS"),
    ("rebounds", "Rebounds", "REB", "REB"),
    ("assists", "Assists", "AST", "AST"),
    ("steals", "Steals", "STL", "STL"),
    ("blocks", "Blocks", "BLK", "BLK"),
    ("three_pointers", "3-Pointers", "3PM", "FG3M"),
]

TOP_N = 10


def _rows_as_dicts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    sets = payload.get("resultSets") or []
    if not sets:
        return []
    block = sets[0] or {}
    headers = [str(h) for h in (block.get("headers") or [])]
    if not headers:
        return []
    out: list[dict[str, Any]] = []
    for raw in block.get("rowSet") or []:
        if not isinstance(raw, (list, tuple)):
            continue
        out.append({headers[i]: raw[i] for i in range(min(len(headers), len(raw)))})
    return out


def _format_value(raw: Any) -> str | None:
    try:
        num = float(raw)
    except (TypeError, ValueError):
        return None
    return f"{num:.1f}"


def _leader_row(rank: int, player: dict[str, Any], value: str) -> WnbaLeaderRow | None:
    player_id = player.get("PLAYER_ID")
    name = str(player.get("PLAYER_NAME") or "").strip()
    abbrev = str(player.get("TEAM_ABBREVIATION") or "").strip().upper()
    gp_raw = player.get("GP")
    try:
        gp = int(gp_raw)
    except (TypeError, ValueError):
        return None
    if player_id is None or not name or not abbrev:
        return None
    return WnbaLeaderRow(
        rank=rank,
        player_id=str(player_id),
        name=name,
        team_abbrev=abbrev,
        gp=gp,
        value=value,
    )


def normalize_leaguedashplayerstats(
    payload: dict[str, Any], *, season: int
) -> WnbaLeadersResponse:
    players = _rows_as_dicts(payload)
    categories: list[WnbaLeaderCategory] = []
    for key, label, stat, header in _CATEGORY_SPECS:
        scored: list[tuple[float, dict[str, Any], str]] = []
        for player in players:
            formatted = _format_value(player.get(header))
            if formatted is None:
                continue
            scored.append((float(formatted), player, formatted))
        scored.sort(key=lambda item: item[0], reverse=True)
        leaders: list[WnbaLeaderRow] = []
        for idx, (_num, player, formatted) in enumerate(scored[:TOP_N], start=1):
            row = _leader_row(idx, player, formatted)
            if row is not None:
                leaders.append(row)
        categories.append(
            WnbaLeaderCategory(
                key=key,  # type: ignore[arg-type]
                label=label,
                stat=stat,
                leaders=leaders,
            )
        )
    return WnbaLeadersResponse(season=season, pace="per_game", categories=categories)
