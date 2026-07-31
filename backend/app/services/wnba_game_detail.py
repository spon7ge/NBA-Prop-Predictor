from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx

from app.schemas.wnba_game_detail import (
    GameDetailInjuries,
    GameDetailInjury,
    GameDetailLatestPlay,
    GameDetailMatchupPrediction,
    GameDetailPlay,
    GameDetailProjectedStarters,
    GameDetailSeasonLeader,
    GameDetailSeasonLeaders,
    GameDetailShot,
    GameDetailStarter,
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

    status_block = (
        ((payload.get("header") or {}).get("competitions") or [{}])[0].get("status")
        or {}
    )
    status, _ = _detail_status(status_block)
    away_id, home_id = _competitor_team_ids(payload)
    prior_game_summaries: dict[str, dict] | None = None
    if status == "scheduled" and away_id and home_id:
        prior_game_summaries = await _fetch_prior_game_summaries(
            payload, away_id=away_id, home_id=home_id
        )

    try:
        detail = normalize_espn_summary(
            payload,
            espn_event_id=espn_event_id,
            fetched_at=datetime.now(ET).isoformat(),
            prior_game_summaries=prior_game_summaries,
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


def _rel_tokens(rel: object) -> set[str]:
    if isinstance(rel, list):
        return {str(x).lower() for x in rel}
    if isinstance(rel, str) and rel.strip():
        return {rel.strip().lower()}
    return set()


def _team_logo_url(logos: object) -> str | None:
    if not isinstance(logos, list):
        return None
    entries: list[tuple[set[str], str]] = []
    for item in logos:
        if not isinstance(item, dict):
            continue
        href = str(item.get("href") or "").strip()
        if not href:
            continue
        entries.append((_rel_tokens(item.get("rel")), href))
    if not entries:
        return None
    for token in ("dark", "default"):
        for rels, href in entries:
            if token in rels:
                return href
    return entries[0][1]


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


_TEAM_STAT_BY_NAME = {
    "fieldgoalpct": ("field_goal_pct", "Field goal %"),
    "threepointfieldgoalpct": ("three_point_pct", "Three point %"),
    "freethrowpct": ("free_throw_pct", "Free throw %"),
    "totalrebounds": ("rebounds", "Rebounds"),
    "offensiverebounds": ("offensive_rebounds", "Offensive rebounds"),
    "assists": ("assists", "Assists"),
}

_TEAM_STAT_BY_LABEL = {
    "field goal %": ("field_goal_pct", "Field goal %"),
    "three point %": ("three_point_pct", "Three point %"),
    "free throw %": ("free_throw_pct", "Free throw %"),
    "rebounds": ("rebounds", "Rebounds"),
    "offensive rebounds": ("offensive_rebounds", "Offensive rebounds"),
    "assists": ("assists", "Assists"),
}


def _clamp_pct(value: int) -> int:
    return max(0, min(100, value))


def _parse_stat_int(raw: object) -> int | None:
    if raw is None or raw == "":
        return None
    try:
        return int(round(float(str(raw).replace("%", "").strip())))
    except (TypeError, ValueError):
        return None


def _resolve_team_stat_key(stat: dict) -> tuple[str, str] | None:
    name_key = str(stat.get("name") or "").replace("-", "").replace("_", "").lower()
    if name_key in _TEAM_STAT_BY_NAME:
        return _TEAM_STAT_BY_NAME[name_key]
    label_key = str(stat.get("label") or "").strip().lower()
    return _TEAM_STAT_BY_LABEL.get(label_key)


def _normalize_team_stats(payload: dict) -> list[GameDetailTeamStat]:
    teams = ((payload.get("boxscore") or {}).get("teams") or [])
    by_side: dict[str, dict[str, int]] = {"away": {}, "home": {}}
    for team in teams:
        side = str(team.get("homeAway") or "").lower()
        if side not in by_side:
            continue
        for stat in team.get("statistics") or []:
            resolved = _resolve_team_stat_key(stat if isinstance(stat, dict) else {})
            if not resolved:
                continue
            key, _label = resolved
            value = _parse_stat_int(stat.get("displayValue"))
            if value is None:
                value = _parse_stat_int(stat.get("value"))
            if value is None:
                continue
            by_side[side][key] = value

    ordered: list[GameDetailTeamStat] = []
    for key, label in (
        ("field_goal_pct", "Field goal %"),
        ("three_point_pct", "Three point %"),
        ("free_throw_pct", "Free throw %"),
        ("rebounds", "Rebounds"),
        ("offensive_rebounds", "Offensive rebounds"),
        ("assists", "Assists"),
    ):
        if key not in by_side["away"] or key not in by_side["home"]:
            continue
        away_value = by_side["away"][key]
        home_value = by_side["home"][key]
        if key.endswith("_pct"):
            away_value = _clamp_pct(away_value)
            home_value = _clamp_pct(home_value)
        ordered.append(
            GameDetailTeamStat(
                key=key,
                label=label,
                away_value=away_value,
                home_value=home_value,
            )
        )
    return ordered


def _normalize_win_probability(payload: dict) -> GameDetailWinProbability | None:
    plays_by_id = {
        str(play.get("id")): play
        for play in (payload.get("plays") or [])
        if play.get("id") is not None
    }

    timeline: list[GameDetailWinProbabilityPoint] = []
    for index, point in enumerate(payload.get("winprobability") or []):
        if not isinstance(point, dict):
            continue
        if point.get("homeWinPercentage") is None:
            continue
        play_id = str(point.get("playId") or "")
        play = plays_by_id.get(play_id)
        if play is None:
            continue

        home_win_pct = _clamp_pct(
            int(round(float(point["homeWinPercentage"]) * 100))
        )
        period = play.get("period") or {}
        clock = play.get("clock") or {}
        team = play.get("team") or {}
        timeline.append(
            GameDetailWinProbabilityPoint(
                id=play_id or f"wp-{index}",
                period=int(period.get("number") or 0),
                clock=str(clock.get("displayValue") or ""),
                away_score=int(play.get("awayScore") or 0),
                home_score=int(play.get("homeScore") or 0),
                home_win_pct=home_win_pct,
                away_win_pct=_clamp_pct(100 - home_win_pct),
                team_id=str(team.get("id") or "") or None,
            )
        )

    team_stats = _normalize_team_stats(payload)

    if not timeline and not team_stats:
        return None

    return GameDetailWinProbability(
        summary=None,
        timeline=timeline,
        team_stats=team_stats,
    )


_LEADER_STAT_MAP = {
    "pointsPerGame": ("points", "Points"),
    "assistsPerGame": ("assists", "Assists"),
    "reboundsPerGame": ("rebounds", "Rebounds"),
}


def _normalize_matchup_prediction(payload: dict) -> GameDetailMatchupPrediction | None:
    predictor = payload.get("predictor")
    if not isinstance(predictor, dict):
        return None
    try:
        away = float((predictor.get("awayTeam") or {}).get("gameProjection"))
        home = float((predictor.get("homeTeam") or {}).get("gameProjection"))
    except (TypeError, ValueError):
        return None
    return GameDetailMatchupPrediction(
        away_win_pct=round(away),
        home_win_pct=round(home),
        source_label="ESPN game projection",
    )


def _leaders_for_team(blocks: list, team_id: str) -> list[GameDetailSeasonLeader]:
    rows: list[GameDetailSeasonLeader] = []
    for block in blocks:
        if str((block.get("team") or {}).get("id") or "") != team_id:
            continue
        for cat in block.get("leaders") or []:
            mapped = _LEADER_STAT_MAP.get(str(cat.get("name") or ""))
            if not mapped:
                continue
            stat, label = mapped
            entry = (cat.get("leaders") or [None])[0] or {}
            athlete = entry.get("athlete") or {}
            name = str(athlete.get("displayName") or "").strip()
            value = str(entry.get("displayValue") or "").strip()
            if not name or not value:
                continue
            rows.append(
                GameDetailSeasonLeader(stat=stat, label=label, name=name, value=value)
            )
    return rows


def _normalize_season_leaders(
    payload: dict, *, away_id: str, home_id: str
) -> GameDetailSeasonLeaders | None:
    blocks = payload.get("leaders")
    if not isinstance(blocks, list) or not blocks:
        return None
    away = _leaders_for_team(blocks, away_id)
    home = _leaders_for_team(blocks, home_id)
    if not away and not home:
        return None
    return GameDetailSeasonLeaders(away=away, home=home)


def _injuries_for_team(blocks: list, team_id: str) -> list[GameDetailInjury]:
    rows: list[GameDetailInjury] = []
    for block in blocks:
        if str((block.get("team") or {}).get("id") or "") != team_id:
            continue
        for item in block.get("injuries") or []:
            athlete = item.get("athlete") or {}
            name = str(athlete.get("displayName") or "").strip()
            if not name:
                continue
            pos = athlete.get("position") or {}
            position = str(pos.get("abbreviation") or "").strip() or None
            details = item.get("details") or {}
            detail = str(details.get("type") or "").strip() or None
            status = str(item.get("status") or "").strip() or "Unknown"
            rows.append(
                GameDetailInjury(
                    name=name, position=position, status=status, detail=detail
                )
            )
    return rows


def _normalize_injuries(
    payload: dict, *, away_id: str, home_id: str
) -> GameDetailInjuries | None:
    blocks = payload.get("injuries")
    if not isinstance(blocks, list):
        return None
    away = _injuries_for_team(blocks, away_id)
    home = _injuries_for_team(blocks, home_id)
    if not away and not home:
        return None
    return GameDetailInjuries(away=away, home=home)


def _competitor_team_ids(payload: dict) -> tuple[str, str]:
    comp = ((payload.get("header") or {}).get("competitions") or [{}])[0]
    teams = {c.get("homeAway"): c for c in (comp.get("competitors") or [])}
    away_c, home_c = teams.get("away") or {}, teams.get("home") or {}
    away_id = str((away_c.get("team") or {}).get("id") or "")
    home_id = str((home_c.get("team") or {}).get("id") or "")
    return away_id, home_id


def _prior_event_ids_by_team(payload: dict) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for block in payload.get("lastFiveGames") or []:
        team_id = str((block.get("team") or {}).get("id") or "")
        events = block.get("events") or []
        if not team_id or not events:
            continue
        event_id = str(events[0].get("id") or "").strip()
        if event_id:
            mapping[team_id] = event_id
    return mapping


async def _fetch_prior_game_summaries(
    payload: dict, *, away_id: str, home_id: str
) -> dict[str, dict] | None:
    prior_ids = _prior_event_ids_by_team(payload)
    away_event = prior_ids.get(away_id)
    home_event = prior_ids.get(home_id)
    if not away_event or not home_event:
        return None
    try:
        away_summary, home_summary = await asyncio.gather(
            fetch_espn_summary(away_event),
            fetch_espn_summary(home_event),
        )
    except Exception:
        return None
    return {away_id: away_summary, home_id: home_summary}


def _starters_from_summary(summary: dict, *, team_id: str) -> list[GameDetailStarter] | None:
    players = (summary.get("boxscore") or {}).get("players")
    if not isinstance(players, list):
        return None
    for block in players:
        if str((block.get("team") or {}).get("id") or "") != team_id:
            continue
        stats = block.get("statistics") or []
        if not stats:
            return None
        athletes = stats[0].get("athletes") or []
        starters: list[GameDetailStarter] = []
        for row in athletes:
            if not row.get("starter"):
                continue
            athlete = row.get("athlete") or {}
            name = str(athlete.get("displayName") or "").strip()
            if not name:
                continue
            jersey = athlete.get("jersey")
            jersey_s = (
                str(jersey).strip()
                if jersey is not None and str(jersey).strip()
                else None
            )
            pos = athlete.get("position") or {}
            position = str(pos.get("abbreviation") or "").strip() or None
            starters.append(
                GameDetailStarter(jersey=jersey_s, name=name, position=position)
            )
            if len(starters) == 5:
                break
        return starters if len(starters) == 5 else None
    return None


def _normalize_projected_starters(
    *,
    status: GameStatus,
    away_id: str,
    home_id: str,
    prior_game_summaries: dict[str, dict] | None,
) -> GameDetailProjectedStarters | None:
    if status != "scheduled" or not prior_game_summaries:
        return None
    away_summary = prior_game_summaries.get(away_id)
    home_summary = prior_game_summaries.get(home_id)
    if not away_summary or not home_summary:
        return None
    away = _starters_from_summary(away_summary, team_id=away_id)
    home = _starters_from_summary(home_summary, team_id=home_id)
    if away is None or home is None:
        return None
    return GameDetailProjectedStarters(
        note="from each team's last game",
        away=away,
        home=home,
    )


def normalize_espn_summary(
    payload: dict,
    *,
    espn_event_id: str,
    fetched_at: str,
    prior_game_summaries: dict[str, dict] | None = None,
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
            logo_url=_team_logo_url(t.get("logos")),
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

    away_id = str((away_c.get("team") or {}).get("id") or "")
    home_id = str((home_c.get("team") or {}).get("id") or "")
    matchup_prediction = _normalize_matchup_prediction(payload)
    season_leaders = _normalize_season_leaders(
        payload, away_id=away_id, home_id=home_id
    )
    injuries = _normalize_injuries(payload, away_id=away_id, home_id=home_id)
    projected_starters = _normalize_projected_starters(
        status=status,
        away_id=away_id,
        home_id=home_id,
        prior_game_summaries=prior_game_summaries,
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
        win_probability=win_probability,
        matchup_prediction=matchup_prediction,
        projected_starters=projected_starters,
        season_leaders=season_leaders,
        injuries=injuries,
        fetched_at=fetched_at,
    )
