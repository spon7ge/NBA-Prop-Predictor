from __future__ import annotations

from typing import Any

from app.schemas.wnba_props import (
    PROP_SPORTSBOOKS,
    WnbaPropBookQuote,
    WnbaPropLine,
)
from app.services.prop_stat_keys import (
    canonical_stat_key_from_parlay_market,
    canonical_stat_key_from_pp,
    canonical_stat_key_from_ud,
    display_stat_label,
)
from app.services.wnba_espn_roster import norm_player_name

US_PROP_SPORTSBOOKS: tuple[str, ...] = (
    "fanduel",
    "draftkings",
    "caesars",
    "betmgm",
    "betrivers",
    "pinnacle",
    "bet365",
    "novig",
)

# Attach a US book only when it matches the DFS slot line or is within this
# half-point / point neighborhood ("similar").
SIMILAR_LINE_MAX_DELTA = 1.0

_VALID_SIDES = frozenset({"over", "under"})

PlayerTeamIndex = dict[str, tuple[str, str | None]]


def _line_key(line: float) -> float:
    return round(float(line), 2)


def pick_closest_quote(
    quotes: list[WnbaPropBookQuote], targets: list[float]
) -> WnbaPropBookQuote | None:
    if not quotes or not targets:
        return None
    exact = {_line_key(t) for t in targets}
    for quote in quotes:
        if _line_key(quote.line) in exact:
            return quote
    primary = targets[0]
    best = min(quotes, key=lambda q: abs(q.line - primary))
    if abs(best.line - primary) > SIMILAR_LINE_MAX_DELTA:
        return None
    return best


def _norm_pp_stat(stat_type: str) -> str:
    return stat_type.strip().lower().replace(" ", "_").replace("+", "_")


def _norm_ud_stat(stat_name: str) -> str:
    return stat_name.strip().lower().replace(" ", "_")


def _pp_stat_key(stat_type: str) -> str:
    canonical = canonical_stat_key_from_pp(stat_type)
    if canonical:
        return canonical
    return f"raw:{_norm_pp_stat(stat_type)}"


def _ud_stat_key(stat_name: str) -> str:
    canonical = canonical_stat_key_from_ud(stat_name)
    if canonical:
        return canonical
    return f"raw:{_norm_ud_stat(stat_name)}"


def _pp_display_stat(stat_type: str, stat_key: str) -> str:
    if stat_key.startswith("raw:"):
        return stat_type
    return display_stat_label(stat_key, fallback=stat_type)


def _ud_display_stat(stat_name: str, stat_key: str) -> str:
    if stat_key.startswith("raw:"):
        return stat_name.replace("_", " ").title()
    return display_stat_label(stat_key, fallback=stat_name.replace("_", " ").title())


def _player_side_key(
    player_name: str, stat_key: str, side: str
) -> tuple[str, str, str]:
    return (norm_player_name(player_name), stat_key, side)


def _slot_key(
    player_name: str, stat_key: str, side: str, line: float
) -> tuple[str, str, str, float]:
    return (*_player_side_key(player_name, stat_key, side), _line_key(line))


def _empty_bucket(
    player_name: str,
    stat: str,
    market_type: str,
    side: str,
    line: float,
) -> dict[str, Any]:
    return {
        "player_name": player_name,
        "stat": stat,
        "market_type": market_type,
        "side": side,
        "slot_line": _line_key(line),
        "team_abbrev": None,
        "logo_url": None,
        "game_date": None,
        "commence_time": None,
        **{book_id: None for book_id in PROP_SPORTSBOOKS},
    }


def _apply_roster(bucket: dict[str, Any], teams: PlayerTeamIndex) -> None:
    if bucket.get("team_abbrev"):
        return
    hit = teams.get(norm_player_name(bucket["player_name"]))
    if hit:
        bucket["team_abbrev"], bucket["logo_url"] = hit


def _attach_us_quotes(
    bucket: dict[str, Any],
    quote_index: dict[tuple[str, str, str], dict[str, list[WnbaPropBookQuote]]],
) -> None:
    slot_line = bucket.get("slot_line")
    if slot_line is None:
        return
    key = _player_side_key(bucket["player_name"], bucket["stat_key"], bucket["side"])
    book_quotes = quote_index.get(key, {})
    targets = [float(slot_line)]
    for book in US_PROP_SPORTSBOOKS:
        candidates = book_quotes.get(book, [])
        bucket[book] = pick_closest_quote(candidates, targets)


def _build_quote_index(
    sportsbook_props: list[WnbaPropLine],
) -> dict[tuple[str, str, str], dict[str, list[WnbaPropBookQuote]]]:
    index: dict[tuple[str, str, str], dict[str, list[WnbaPropBookQuote]]] = {}
    for prop in sportsbook_props:
        stat_key = canonical_stat_key_from_parlay_market(prop.market_type)
        if stat_key is None:
            continue
        key = _player_side_key(prop.player_name, stat_key, prop.side)
        for book in US_PROP_SPORTSBOOKS:
            quote = getattr(prop, book, None)
            if quote is not None:
                index.setdefault(key, {}).setdefault(book, []).append(quote)
    return index


def _parse_american_price(raw: Any) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _pp_target_lines(
    pp_rows: list[dict[str, Any]],
) -> dict[tuple[str, str], float]:
    """Map (norm_player, stat_key) → PrizePicks standard line when present."""
    targets: dict[tuple[str, str], float] = {}
    for row in pp_rows:
        if str(row.get("odds_type") or "").lower() != "standard":
            continue
        player = str(row.get("player_name") or "").strip()
        stat_type = str(row.get("stat_type") or "").strip()
        line_raw = row.get("line_score")
        if not player or not stat_type or line_raw is None:
            continue
        try:
            line_f = float(line_raw)
        except (TypeError, ValueError):
            continue
        targets[(norm_player_name(player), _pp_stat_key(stat_type))] = line_f
    return targets


def _select_underdog_mains(
    ud_rows: list[dict[str, Any]],
    pp_targets: dict[tuple[str, str], float],
) -> list[dict[str, Any]]:
    """One Underdog row per player + stat + side (drop alts).

    Prefer a line that matches PrizePicks when available; otherwise the price
    closest to -110 (typical main).
    """
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in ud_rows:
        player = str(row.get("player_name") or "").strip()
        stat_name = str(row.get("stat_name") or "").strip()
        side = str(row.get("side") or "").lower()
        line_raw = row.get("line_score")
        if not player or not stat_name or side not in _VALID_SIDES or line_raw is None:
            continue
        try:
            line_f = float(line_raw)
        except (TypeError, ValueError):
            continue
        stat_key = _ud_stat_key(stat_name)
        key = (norm_player_name(player), stat_key, side)
        groups.setdefault(key, []).append(
            {
                **row,
                "_line_f": line_f,
                "_odds_i": _parse_american_price(row.get("american_price")),
                "_stat_key": stat_key,
            }
        )

    selected: list[dict[str, Any]] = []
    for (norm_player, stat_key, _side), candidates in groups.items():
        pp_line = pp_targets.get((norm_player, stat_key))

        def rank(c: dict[str, Any]) -> tuple[float, float]:
            line_f = float(c["_line_f"])
            odds_i = c["_odds_i"]
            price_dist = (
                abs(int(odds_i) - (-110)) if odds_i is not None else 10_000.0
            )
            if pp_line is not None:
                return (abs(line_f - pp_line), price_dist)
            return (price_dist, abs(line_f))

        best = min(candidates, key=rank)
        selected.append(best)
    return selected


def attach_dfs_snapshots(
    sportsbook_props: list[WnbaPropLine],
    pp_rows: list[dict[str, Any]],
    ud_rows: list[dict[str, Any]],
    player_teams: PlayerTeamIndex | None = None,
) -> list[WnbaPropLine]:
    """Seed DFS slots; PP+UD share a slot only when their lines match."""
    teams = player_teams or {}
    quote_index = _build_quote_index(sportsbook_props)
    buckets: dict[tuple[str, str, str, float], dict[str, Any]] = {}
    pp_targets = _pp_target_lines(pp_rows)

    for row in pp_rows:
        if str(row.get("odds_type") or "").lower() != "standard":
            continue
        player = str(row.get("player_name") or "").strip()
        stat_type = str(row.get("stat_type") or "").strip()
        line_raw = row.get("line_score")
        if not player or not stat_type or line_raw is None:
            continue
        try:
            line_f = float(line_raw)
        except (TypeError, ValueError):
            continue

        stat_key = _pp_stat_key(stat_type)
        stat = _pp_display_stat(stat_type, stat_key)
        market_type = f"prizepicks:{stat_type}"
        # Standard PrizePicks entries price as even money (+100) vs the line.
        quote = WnbaPropBookQuote(line=line_f, odds_american=100)
        for side in _VALID_SIDES:
            key = _slot_key(player, stat_key, side, line_f)
            bucket = buckets.get(key)
            if bucket is None:
                bucket = _empty_bucket(player, stat, market_type, side, line_f)
                bucket["stat_key"] = stat_key
                buckets[key] = bucket
            bucket["prizepicks"] = quote
            _apply_roster(bucket, teams)

    for row in _select_underdog_mains(ud_rows, pp_targets):
        player = str(row.get("player_name") or "").strip()
        stat_name = str(row.get("stat_name") or "").strip()
        side = str(row.get("side") or "").lower()
        line_f = float(row["_line_f"])
        odds_i = row["_odds_i"]
        stat_key = str(row["_stat_key"])

        key = _slot_key(player, stat_key, side, line_f)
        bucket = buckets.get(key)
        if bucket is None:
            bucket = _empty_bucket(
                player,
                _ud_display_stat(stat_name, stat_key),
                f"underdog:{stat_name}",
                side,
                line_f,
            )
            bucket["stat_key"] = stat_key
            buckets[key] = bucket
        if bucket.get("underdog") is None:
            bucket["underdog"] = WnbaPropBookQuote(line=line_f, odds_american=odds_i)
        _apply_roster(bucket, teams)

    for bucket in buckets.values():
        _attach_us_quotes(bucket, quote_index)

    props: list[WnbaPropLine] = []
    for bucket in buckets.values():
        if bucket.get("prizepicks") is None and bucket.get("underdog") is None:
            continue
        props.append(
            WnbaPropLine(
                player_name=bucket["player_name"],
                team_abbrev=bucket.get("team_abbrev"),
                logo_url=bucket.get("logo_url"),
                stat=bucket["stat"],
                market_type=bucket["market_type"],
                side=bucket["side"],
                game_date=bucket.get("game_date"),
                commence_time=bucket.get("commence_time"),
                **{book_id: bucket.get(book_id) for book_id in PROP_SPORTSBOOKS},
            )
        )

    props.sort(
        key=lambda p: (
            p.player_name.lower(),
            p.stat.lower(),
            0 if p.side == "over" else 1,
            (p.prizepicks.line if p.prizepicks else None)
            or (p.underdog.line if p.underdog else 0.0),
        )
    )
    return props
