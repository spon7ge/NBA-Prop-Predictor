from __future__ import annotations

import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

from app.schemas.wnba_game_detail import (
    GameDetailLatestPlay,
    GameDetailPlay,
    GameDetailShot,
    GameDetailTeam,
    GameDetailTeamStat,
    GameDetailWinProbability,
    GameDetailWinProbabilityPoint,
    WnbaGameDetail,
)
from app.schemas.wnba_scoreboard import GameStatus

ET = ZoneInfo("America/New_York")

ESPN_SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/summary"
)
ESPN_TIMEOUT_SECONDS = 8.0

FALLBACK_AWAY_COLOR = "#7C3AED"
FALLBACK_HOME_COLOR = "#EA580C"

_cache: dict[str, dict] = {}

_EVENT_ID_PATTERN = re.compile(r"^\d{6,12}$")
_NOT_FOUND_CACHE_TTL_SECONDS = 45

_ESPN_NON_RESULT_LABELS = {
    "STATUS_POSTPONED": "Postponed",
    "STATUS_CANCELED": "Canceled",
    "STATUS_CANCELLED": "Canceled",
    "STATUS_SUSPENDED": "Suspended",
    "STATUS_DELAYED": "Delayed",
}


def clear_game_detail_cache() -> None:
    _cache.clear()


def is_valid_espn_event_id(espn_event_id: str) -> bool:
    return bool(_EVENT_ID_PATTERN.match(espn_event_id))


def _cache_not_found(espn_event_id: str, *, now: float) -> None:
    _cache[espn_event_id] = {
        "not_found": True,
        "expires_at": now + _NOT_FOUND_CACHE_TTL_SECONDS,
    }


def cache_ttl_seconds(detail: WnbaGameDetail) -> int:
    if detail.status in ("live", "halftime"):
        return 15
    return 60


async def fetch_espn_summary(espn_event_id: str) -> dict:
    async with httpx.AsyncClient(timeout=ESPN_TIMEOUT_SECONDS) as client:
        response = await client.get(
            ESPN_SUMMARY_URL, params={"event": espn_event_id}
        )
        response.raise_for_status()
        return response.json()


def _is_not_found_payload(payload: dict) -> bool:
    header = payload.get("header")
    if not isinstance(header, dict):
        return True

    competitions = header.get("competitions")
    if not isinstance(competitions, list) or not competitions:
        return True

    competition = competitions[0]
    competitors = (
        competition.get("competitors") if isinstance(competition, dict) else None
    )
    if not isinstance(competitors, list):
        return True

    teams = {
        competitor.get("homeAway"): competitor
        for competitor in competitors
        if isinstance(competitor, dict)
    }
    for side in ("away", "home"):
        team = (teams.get(side) or {}).get("team")
        if not isinstance(team, dict) or not any(
            team.get(field) for field in ("id", "abbreviation", "displayName")
        ):
            return True
    return False


async def get_game_detail(espn_event_id: str) -> WnbaGameDetail:
    now = time.time()
    cached = _cache.get(espn_event_id)
    if cached and cached["expires_at"] > now:
        if cached.get("not_found"):
            raise LookupError(espn_event_id)
        return cached["response"]

    # A stale positive cache entry is still usable as a stale-while-error
    # fallback below; a stale/expired negative entry is not.
    stale_fallback = cached if cached and not cached.get("not_found") else None

    if not is_valid_espn_event_id(espn_event_id):
        _cache_not_found(espn_event_id, now=now)
        raise LookupError(espn_event_id)

    try:
        payload = await fetch_espn_summary(espn_event_id)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in (400, 404):
            _cache_not_found(espn_event_id, now=now)
            raise LookupError(espn_event_id) from exc
        if stale_fallback:
            return stale_fallback["response"]
        raise
    except Exception:
        if stale_fallback:
            return stale_fallback["response"]
        raise

    if _is_not_found_payload(payload):
        _cache_not_found(espn_event_id, now=now)
        raise LookupError(espn_event_id)

    try:
        detail = normalize_espn_summary(
            payload,
            espn_event_id=espn_event_id,
            fetched_at=datetime.now(ET).isoformat(),
        )
    except Exception:
        if stale_fallback:
            return stale_fallback["response"]
        raise

    _cache[espn_event_id] = {
        "response": detail,
        "expires_at": now + cache_ttl_seconds(detail),
    }
    return detail


def _hex_color(raw: str | None, fallback: str) -> str:
    s = str(raw or "").strip().lstrip("#")
    if len(s) == 6 and all(c in "0123456789abcdefABCDEF" for c in s):
        return f"#{s.upper()}"
    return fallback


def _is_numeric_coordinate(value: object) -> bool:
    if value is None:
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _has_real_coordinate(coord: object) -> bool:
    return (
        isinstance(coord, dict)
        and _is_numeric_coordinate(coord.get("x"))
        and _is_numeric_coordinate(coord.get("y"))
    )


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


def _normalize_win_probability(payload: dict) -> GameDetailWinProbability | None:
    predictor = payload.get("predictor") or {}
    graph = (
        predictor.get("gameFlow")
        or predictor.get("homeTeamGameProjection")
        or []
    )

    timeline = [
        GameDetailWinProbabilityPoint(
            id=str(point.get("id") or f"wp-{index}"),
            period=int(point.get("period") or 0),
            clock=str(point.get("clock") or ""),
            away_score=int(point.get("awayScore") or 0),
            home_score=int(point.get("homeScore") or 0),
            away_win_pct=int(round(float(point.get("awayWinPct") or 0))),
            home_win_pct=int(round(float(point.get("homeWinPct") or 0))),
            team_id=str(point.get("teamId") or "") or None,
        )
        for index, point in enumerate(graph)
        if point.get("awayWinPct") is not None
        or point.get("homeWinPct") is not None
    ]

    allowed_stats = {
        "field_goal_pct": "Field goal %",
        "three_point_pct": "Three point %",
        "free_throw_pct": "Free throw %",
        "rebounds": "Rebounds",
        "offensive_rebounds": "Offensive rebounds",
        "assists": "Assists",
    }
    team_stats = [
        GameDetailTeamStat(
            key=key,
            label=label,
            away_value=int(raw["away"]),
            home_value=int(raw["home"]),
        )
        for key, label in allowed_stats.items()
        if (raw := predictor.get("teamStatsMap", {}).get(key))
    ]

    if not timeline and not team_stats:
        return None

    return GameDetailWinProbability(
        summary=str(predictor.get("summary") or "") or None,
        timeline=timeline,
        team_stats=team_stats,
    )


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

        is_free_throw = "free throw" in text.lower()
        coord = p.get("coordinate")
        if shooting and not is_free_throw and _has_real_coordinate(coord):
            shots.append(
                GameDetailShot(
                    id=play.id,
                    team_id=team_id or "",
                    player_name=_player_name_from_text(text),
                    made=scoring,
                    x=float(coord["x"]),
                    y=float(coord["y"]),
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

    win_probability = _normalize_win_probability(payload)

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
        win_probability=win_probability,
        fetched_at=fetched_at,
    )
