from __future__ import annotations

from datetime import datetime
from typing import Any

from app.schemas.wnba_player import (
    WnbaPlayerAverages,
    WnbaPlayerGame,
    WnbaPlayerResponse,
)

HEADSHOT_URL_TEMPLATE = (
    "https://cdn.wnba.com/headshots/wnba/latest/1040x760/{player_id}.png"
)


def rows_as_dicts(payload: dict[str, Any]) -> list[dict[str, Any]]:
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


def format_avg(raw: Any) -> str | None:
    try:
        num = float(raw)
    except (TypeError, ValueError):
        return None
    return f"{num:.1f}"


def format_pct(raw: Any) -> str | None:
    try:
        num = float(raw)
    except (TypeError, ValueError):
        return None
    if 0 < num <= 1:
        num *= 100
    return f"{num:.1f}"


def made_attempt(made: Any, attempted: Any) -> str:
    try:
        m = int(float(made))
        a = int(float(attempted))
    except (TypeError, ValueError):
        return "0-0"
    return f"{m}-{a}"


def headshot_url_for(player_id: str) -> str:
    return HEADSHOT_URL_TEMPLATE.format(player_id=player_id)


def _format_game_stat(raw: Any) -> str:
    try:
        num = float(raw)
    except (TypeError, ValueError):
        return "0"
    if num == int(num):
        return str(int(num))
    return f"{num:.1f}"


def _normalize_game_date(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        return text
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return text


def _game_id(row: dict[str, Any]) -> str:
    for key in ("Game_ID", "GAME_ID"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _position_from_info(info_rows: list[dict[str, Any]]) -> str | None:
    if not info_rows:
        return None
    row = info_rows[0]
    for key in ("POSITION", "POSITION_ABBREVIATION"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return None


def _team_name(
    dash_row: dict[str, Any], info_rows: list[dict[str, Any]], team_abbrev: str
) -> str:
    dash_name = str(dash_row.get("TEAM_NAME") or "").strip()
    if dash_name:
        return dash_name
    if info_rows:
        info_name = str(info_rows[0].get("TEAM_NAME") or "").strip()
        if info_name:
            return info_name
    return team_abbrev


def _normalize_games(gamelog: dict[str, Any]) -> list[WnbaPlayerGame]:
    rows = rows_as_dicts(gamelog)
    games: list[WnbaPlayerGame] = []
    for row in rows:
        game_id = _game_id(row)
        if not game_id:
            continue
        games.append(
            WnbaPlayerGame(
                game_id=game_id,
                game_date=_normalize_game_date(row.get("GAME_DATE")),
                matchup=str(row.get("MATCHUP") or "").strip(),
                min=_format_game_stat(row.get("MIN")),
                pts=_format_game_stat(row.get("PTS")),
                fg=made_attempt(row.get("FGM"), row.get("FGA")),
                three_pt=made_attempt(row.get("FG3M"), row.get("FG3A")),
                ft=made_attempt(row.get("FTM"), row.get("FTA")),
                reb=_format_game_stat(row.get("REB")),
                ast=_format_game_stat(row.get("AST")),
                to=_format_game_stat(row.get("TOV")),
                stl=_format_game_stat(row.get("STL")),
                blk=_format_game_stat(row.get("BLK")),
            )
        )
    games.sort(key=lambda g: g.game_date, reverse=True)
    return games


def normalize_wnba_player(
    *,
    player_id: str,
    season: int,
    dash: dict[str, Any],
    info: dict[str, Any],
    gamelog: dict[str, Any],
) -> WnbaPlayerResponse | None:
    dash_rows = rows_as_dicts(dash)
    dash_row = next(
        (row for row in dash_rows if str(row.get("PLAYER_ID")) == player_id),
        None,
    )
    if dash_row is None:
        return None

    info_rows = rows_as_dicts(info)
    name = str(dash_row.get("PLAYER_NAME") or "").strip()
    team_abbrev = str(dash_row.get("TEAM_ABBREVIATION") or "").strip().upper()
    if not name or not team_abbrev:
        return None

    pts = format_avg(dash_row.get("PTS"))
    reb = format_avg(dash_row.get("REB"))
    ast = format_avg(dash_row.get("AST"))
    fg_pct = format_pct(dash_row.get("FG_PCT"))
    fg3_pct = format_pct(dash_row.get("FG3_PCT"))
    if not all([pts, reb, ast, fg_pct, fg3_pct]):
        return None

    return WnbaPlayerResponse(
        player_id=player_id,
        name=name,
        position=_position_from_info(info_rows),
        team_name=_team_name(dash_row, info_rows, team_abbrev),
        team_abbrev=team_abbrev,
        headshot_url=headshot_url_for(player_id),
        season=season,
        averages=WnbaPlayerAverages(
            pts=pts,
            reb=reb,
            ast=ast,
            fg_pct=fg_pct,
            fg3_pct=fg3_pct,
        ),
        games=_normalize_games(gamelog),
    )
