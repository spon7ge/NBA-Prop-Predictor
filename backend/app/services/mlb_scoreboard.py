from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from app.schemas.mlb_scoreboard import GameStatus, MlbGame, MlbTeam

ET = ZoneInfo("America/New_York")

TEAM_LOGO = "https://www.mlbstatic.com/team-logos/{id}.svg"

_NON_RESULT_KEYWORDS = ("postponed", "cancelled", "canceled", "suspended")


def _parse_start(start: str) -> datetime | None:
    raw = str(start or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ET)
    return parsed


def format_tip_label(start: str) -> str | None:
    """Render a scheduled first pitch as an ET wall-clock label, e.g. ``7:00 PM ET``."""
    parsed = _parse_start(start)
    if parsed is None:
        return None
    local = parsed.astimezone(ET)
    return f"{local.strftime('%I:%M %p').lstrip('0')} ET"


def _team_logo_url(team_id: int | None) -> str | None:
    if team_id is None:
        return None
    return TEAM_LOGO.format(id=team_id)


def _mlb_team_record(side: dict) -> str | None:
    record = side.get("leagueRecord") or {}
    if not isinstance(record, dict):
        return None
    wins = record.get("wins")
    losses = record.get("losses")
    if wins is None or losses is None:
        return None
    return f"{wins}-{losses}"


def _mlb_status(
    status: dict,
    linescore: dict | None,
    game_date: str,
) -> tuple[GameStatus, str]:
    detailed = str(status.get("detailedState") or "").strip()
    detailed_lower = detailed.lower()

    for keyword in _NON_RESULT_KEYWORDS:
        if keyword in detailed_lower:
            return "scheduled", detailed

    abstract = str(status.get("abstractGameState") or "").strip()
    if abstract == "Final":
        return "final", "Final"
    if abstract == "Live":
        if isinstance(linescore, dict):
            inning_state = str(linescore.get("inningState") or "").strip()
            inning_ordinal = str(linescore.get("currentInningOrdinal") or "").strip()
            if inning_state and inning_ordinal:
                return "live", f"{inning_state} {inning_ordinal}"
        return "live", "Live"

    tip = format_tip_label(game_date)
    return "scheduled", tip or detailed or "Scheduled"


def _mlb_venue(game: dict) -> tuple[str | None, str | None]:
    venue = game.get("venue") or {}
    if not isinstance(venue, dict):
        return None, None
    name = venue.get("name")
    address = venue.get("address") or {}
    city = address.get("city") if isinstance(address, dict) else None
    return (str(name) if name else None, str(city) if city else None)


def normalize_mlb_schedule(payload: dict, *, date_et: str) -> list[MlbGame]:
    games: list[MlbGame] = []
    for day in payload.get("dates") or []:
        if not isinstance(day, dict):
            continue
        for game in day.get("games") or []:
            if not isinstance(game, dict):
                continue
            game_pk = game.get("gamePk")
            if game_pk is None:
                continue
            pk_str = str(game_pk)
            game_date = str(game.get("gameDate") or "")
            status, label = _mlb_status(
                game.get("status") or {},
                game.get("linescore"),
                game_date,
            )
            venue, venue_city = _mlb_venue(game)
            teams = game.get("teams") or {}
            away_side = teams.get("away") or {}
            home_side = teams.get("home") or {}

            def team(side: dict) -> MlbTeam:
                team_info = side.get("team") or {}
                team_id = team_info.get("id")
                raw_score = side.get("score")
                score = (
                    int(raw_score)
                    if raw_score is not None and status != "scheduled"
                    else None
                )
                return MlbTeam(
                    abbrev=str(team_info.get("abbreviation") or ""),
                    name=str(team_info.get("name") or ""),
                    score=score,
                    record=_mlb_team_record(side),
                    logo_url=_team_logo_url(team_id),
                )

            games.append(
                MlbGame(
                    id=f"mlb-{pk_str}",
                    mlb_game_pk=pk_str,
                    status=status,
                    status_label=label,
                    away=team(away_side),
                    home=team(home_side),
                    start_time_et=game_date,
                    venue=venue,
                    venue_city=venue_city,
                )
            )
    return sorted(games, key=lambda g: (g.start_time_et or "", g.id))
