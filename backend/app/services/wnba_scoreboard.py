from __future__ import annotations

from app.schemas.wnba_scoreboard import GameStatus, WnbaGame, WnbaTeam


def _espn_status(status_block: dict) -> tuple[GameStatus, str]:
    typ = status_block.get("type") or {}
    name = str(typ.get("name") or "")
    state = str(typ.get("state") or "")
    short = str(typ.get("shortDetail") or typ.get("detail") or "")
    period = status_block.get("period")
    clock = str(status_block.get("displayClock") or "").strip()

    if typ.get("completed") or name == "STATUS_FINAL" or state == "post":
        return "final", "Final"
    if "HALFTIME" in name.upper() or short.lower() == "halftime":
        return "halftime", "Halftime"
    if state == "in" or name == "STATUS_IN_PROGRESS":
        # Prefer compact Qn clock when period + clock present
        if isinstance(period, int) and period > 0 and clock:
            return "live", f"Q{period} {clock}"
        return "live", short or "Live"
    # scheduled
    label = short or "Scheduled"
    return "scheduled", label


def normalize_espn_scoreboard(payload: dict, *, date_et: str) -> list[WnbaGame]:
    games: list[WnbaGame] = []
    for event in payload.get("events") or []:
        comps = (event.get("competitions") or [{}])[0]
        teams = {c.get("homeAway"): c for c in (comps.get("competitors") or [])}
        away_c, home_c = teams.get("away") or {}, teams.get("home") or {}
        status, label = _espn_status(event.get("status") or {})
        start = str(event.get("date") or "")

        def team(c: dict) -> WnbaTeam:
            t = c.get("team") or {}
            raw = c.get("score")
            score = int(raw) if raw not in (None, "") else None
            return WnbaTeam(
                abbrev=str(t.get("abbreviation") or ""),
                name=str(t.get("displayName") or ""),
                score=score if status != "scheduled" else None,
            )

        games.append(
            WnbaGame(
                id=f"espn-{event.get('id')}",
                status=status,
                status_label=label,
                away=team(away_c),
                home=team(home_c),
                start_time_et=start,
            )
        )
    return games
