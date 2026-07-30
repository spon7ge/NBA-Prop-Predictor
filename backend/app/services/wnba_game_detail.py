from __future__ import annotations

from app.schemas.wnba_game_detail import (
    GameDetailLatestPlay,
    GameDetailPlay,
    GameDetailShot,
    GameDetailTeam,
    WnbaGameDetail,
)
from app.schemas.wnba_scoreboard import GameStatus

FALLBACK_AWAY_COLOR = "#7C3AED"
FALLBACK_HOME_COLOR = "#EA580C"

_ESPN_NON_RESULT_LABELS = {
    "STATUS_POSTPONED": "Postponed",
    "STATUS_CANCELED": "Canceled",
    "STATUS_CANCELLED": "Canceled",
    "STATUS_SUSPENDED": "Suspended",
    "STATUS_DELAYED": "Delayed",
}


def _hex_color(raw: str | None, fallback: str) -> str:
    s = str(raw or "").strip().lstrip("#")
    if len(s) == 6 and all(c in "0123456789abcdefABCDEF" for c in s):
        return f"#{s.upper()}"
    return fallback


def _player_name_from_text(text: str) -> str:
    for verb in (" makes ", " misses ", " shooting ", " defensive ", " offensive "):
        if verb in text:
            return text.split(verb, 1)[0].strip()
    return text.split(" ", 2)[0] if text else ""


def _detail_status(status_block: dict) -> tuple[GameStatus, str]:
    typ = status_block.get("type") or {}
    name = str(typ.get("name") or "").upper()
    state = str(typ.get("state") or "")
    short = str(typ.get("shortDetail") or typ.get("detail") or "")

    if name in _ESPN_NON_RESULT_LABELS:
        return "scheduled", _ESPN_NON_RESULT_LABELS[name]
    if typ.get("completed") or name == "STATUS_FINAL" or state == "post":
        return "final", "Final"
    if "HALFTIME" in name or short.lower() == "halftime":
        return "halftime", "Halftime"
    if state == "in" or name == "STATUS_IN_PROGRESS":
        return "live", short or "Live"
    return "scheduled", short or "Scheduled"


def normalize_espn_summary(
    payload: dict, *, espn_event_id: str, fetched_at: str
) -> WnbaGameDetail:
    header = payload.get("header") or {}
    comp = (header.get("competitions") or [{}])[0]
    status_block = comp.get("status") or {}
    teams = {c.get("homeAway"): c for c in (comp.get("competitors") or [])}
    away_c, home_c = teams.get("away") or {}, teams.get("home") or {}
    venue = ((payload.get("gameInfo") or {}).get("venue") or {}).get("fullName")
    status, status_label = _detail_status(status_block)

    def team(c: dict, fallback_color: str) -> GameDetailTeam:
        t = c.get("team") or {}
        raw = c.get("score")
        score = int(raw) if raw not in (None, "") else None
        return GameDetailTeam(
            id=str(t.get("id") or ""),
            abbrev=str(t.get("abbreviation") or ""),
            name=str(t.get("displayName") or ""),
            score=score if status != "scheduled" else None,
            color=_hex_color(t.get("color"), fallback_color),
        )

    raw_plays = payload.get("plays") or []
    plays: list[GameDetailPlay] = []
    shots: list[GameDetailShot] = []
    for p in raw_plays:
        period = int((p.get("period") or {}).get("number") or 0)
        clock = str((p.get("clock") or {}).get("displayValue") or "")
        team_id = str((p.get("team") or {}).get("id") or "") or None
        text = str(p.get("text") or "")
        shooting = bool(p.get("shootingPlay"))
        scoring = bool(p.get("scoringPlay"))
        play = GameDetailPlay(
            id=str(p.get("id") or ""),
            team_id=team_id,
            period=period,
            clock=clock,
            text=text,
            scoring=scoring,
            away_score=int(p.get("awayScore") or 0),
            home_score=int(p.get("homeScore") or 0),
            shooting=shooting,
        )
        plays.append(play)
        if shooting:
            coord = p.get("coordinate") or {}
            shots.append(
                GameDetailShot(
                    id=play.id,
                    team_id=team_id or "",
                    player_name=_player_name_from_text(text),
                    made=scoring,
                    x=float(coord.get("x") or 0),
                    y=float(coord.get("y") or 0),
                    period=period,
                    clock=clock,
                )
            )

    display_clock = str(status_block.get("displayClock") or "").strip()
    current_period = status_block.get("period")
    latest_src = None
    if raw_plays and display_clock:
        for p in raw_plays:
            p_clock = str((p.get("clock") or {}).get("displayValue") or "")
            p_period = int((p.get("period") or {}).get("number") or 0)
            if p_clock == display_clock and p_period == current_period:
                latest_src = p
                break
    if latest_src is None and raw_plays:
        latest_src = raw_plays[-1]
    latest = None
    if latest_src is not None:
        latest = GameDetailLatestPlay(
            id=str(latest_src.get("id") or ""),
            clock=str((latest_src.get("clock") or {}).get("displayValue") or ""),
            period=int((latest_src.get("period") or {}).get("number") or 0),
            text=str(latest_src.get("text") or ""),
            team_id=str((latest_src.get("team") or {}).get("id") or "") or None,
        )

    return WnbaGameDetail(
        espn_event_id=espn_event_id,
        status=status,
        status_label=status_label,
        venue=str(venue) if venue else None,
        away=team(away_c, FALLBACK_AWAY_COLOR),
        home=team(home_c, FALLBACK_HOME_COLOR),
        fg_made=sum(1 for s in shots if s.made),
        fg_attempted=len(shots),
        latest_play=latest,
        shots=shots,
        plays=list(reversed(plays)),
        fetched_at=fetched_at,
    )
