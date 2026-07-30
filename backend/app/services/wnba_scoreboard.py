from __future__ import annotations

import re

from app.schemas.wnba_scoreboard import GameStatus, WnbaGame, WnbaTeam

_STATUS_MAP = {1: "scheduled", 2: "live", 3: "final"}


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


def _parse_iso_clock(game_clock: str | None) -> str | None:
    if not game_clock:
        return None
    # PT7M10.00S → 7:10
    m = re.match(r"PT(?:(\d+)M)?(?:(\d+)(?:\.\d+)?S)?", game_clock)
    if not m:
        return None
    mins = int(m.group(1) or 0)
    secs = int(float(m.group(2) or 0))
    return f"{mins}:{secs:02d}"


def _stats_status(game: dict) -> tuple[GameStatus, str]:
    code = int(game.get("gameStatus") or 1)
    text = str(game.get("gameStatusText") or "").strip()
    if code == 3 or text.lower() == "final":
        return "final", "Final"
    if "half" in text.lower():
        return "halftime", "Halftime"
    if code == 2:
        period = game.get("period")
        clock = _parse_iso_clock(game.get("gameClock"))
        if isinstance(period, int) and period > 0 and clock:
            return "live", f"Q{period} {clock}"
        return "live", text or "Live"
    return "scheduled", text or "Scheduled"


def normalize_stats_scoreboard(payload: dict, *, date_et: str) -> list[WnbaGame]:
    board = payload.get("scoreboard") or payload
    games: list[WnbaGame] = []
    for g in board.get("games") or []:
        status, label = _stats_status(g)
        away, home = g.get("awayTeam") or {}, g.get("homeTeam") or {}

        def team(t: dict) -> WnbaTeam:
            city = str(t.get("teamCity") or "").strip()
            name = str(t.get("teamName") or "").strip()
            full = f"{city} {name}".strip()
            raw = t.get("score")
            score = int(raw) if raw is not None and status != "scheduled" else None
            return WnbaTeam(
                abbrev=str(t.get("teamTricode") or ""),
                name=full,
                score=score,
            )

        games.append(
            WnbaGame(
                id=str(g.get("gameId")),
                status=status,
                status_label=label,
                away=team(away),
                home=team(home),
                start_time_et=str(g.get("gameTimeUTC") or ""),
            )
        )
    return games


def _match_key(game: WnbaGame) -> tuple[str, str]:
    return (game.away.abbrev.upper(), game.home.abbrev.upper())


def _score_richness(label: str) -> int:
    return len(label or "")


def merge_games(espn: list[WnbaGame], stats: list[WnbaGame]) -> list[WnbaGame]:
    by_key: dict[tuple[str, str], WnbaGame] = {}
    for g in espn:
        by_key[_match_key(g)] = g
    for g in stats:
        key = _match_key(g)
        if key not in by_key:
            by_key[key] = g
            continue
        a = by_key[key]
        # Prefer stats id when present and not espn-prefixed
        game_id = g.id if not g.id.startswith("espn-") else a.id
        if a.id.startswith("espn-") and not g.id.startswith("espn-"):
            game_id = g.id
        status = g.status if g.status != "scheduled" or a.status == "scheduled" else a.status
        # Prefer richer status_label
        status_label = (
            g.status_label
            if _score_richness(g.status_label) >= _score_richness(a.status_label)
            else a.status_label
        )

        def pick_score(x: int | None, y: int | None) -> int | None:
            if x is None:
                return y
            if y is None:
                return x
            return y  # prefer stats when both present

        by_key[key] = WnbaGame(
            id=game_id,
            status=status if status in ("scheduled", "live", "halftime", "final") else a.status,
            status_label=status_label,
            away=WnbaTeam(
                abbrev=g.away.abbrev or a.away.abbrev,
                name=g.away.name if len(g.away.name) >= len(a.away.name) else a.away.name,
                score=pick_score(a.away.score, g.away.score),
            ),
            home=WnbaTeam(
                abbrev=g.home.abbrev or a.home.abbrev,
                name=g.home.name if len(g.home.name) >= len(a.home.name) else a.home.name,
                score=pick_score(a.home.score, g.home.score),
            ),
            start_time_et=g.start_time_et or a.start_time_et,
        )
    return sorted(by_key.values(), key=lambda g: g.start_time_et or g.id)


def cache_ttl_seconds(games: list[WnbaGame]) -> int:
    if any(g.status in ("live", "halftime") for g in games):
        return 30
    return 60
