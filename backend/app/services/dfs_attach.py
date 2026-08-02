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
    "pinnacle",
    "bet365",
    "novig",
)

_VALID_SIDES = frozenset({"over", "under"})

PlayerTeamIndex = dict[str, tuple[str, str | None]]


def pick_closest_quote(
    quotes: list[WnbaPropBookQuote], targets: list[float]
) -> WnbaPropBookQuote | None:
    if not quotes:
        return None
    exact = set(targets)
    for quote in quotes:
        if quote.line in exact:
            return quote
    primary = targets[0]
    return min(quotes, key=lambda q: abs(q.line - primary))


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


def _merge_key(player_name: str, stat_key: str, side: str) -> tuple[str, str, str]:
    return (norm_player_name(player_name), stat_key, side)


def _empty_bucket(
    player_name: str,
    stat: str,
    market_type: str,
    side: str,
) -> dict[str, Any]:
    return {
        "player_name": player_name,
        "stat": stat,
        "market_type": market_type,
        "side": side,
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


def _target_lines(bucket: dict[str, Any]) -> list[float]:
    targets: list[float] = []
    pp = bucket.get("prizepicks")
    if pp is not None:
        targets.append(pp.line)
    ud = bucket.get("underdog")
    if ud is not None:
        targets.append(ud.line)
    return targets


def _attach_us_quotes(
    bucket: dict[str, Any],
    quote_index: dict[tuple[str, str, str], dict[str, list[WnbaPropBookQuote]]],
) -> None:
    targets = _target_lines(bucket)
    if not targets:
        return
    key = _merge_key(bucket["player_name"], bucket["stat_key"], bucket["side"])
    book_quotes = quote_index.get(key, {})
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
        key = _merge_key(prop.player_name, stat_key, prop.side)
        for book in US_PROP_SPORTSBOOKS:
            quote = getattr(prop, book, None)
            if quote is not None:
                index.setdefault(key, {}).setdefault(book, []).append(quote)
    return index


def attach_dfs_snapshots(
    sportsbook_props: list[WnbaPropLine],
    pp_rows: list[dict[str, Any]],
    ud_rows: list[dict[str, Any]],
    player_teams: PlayerTeamIndex | None = None,
) -> list[WnbaPropLine]:
    teams = player_teams or {}
    quote_index = _build_quote_index(sportsbook_props)
    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}

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
        quote = WnbaPropBookQuote(line=line_f, odds_american=None)
        for side in _VALID_SIDES:
            key = _merge_key(player, stat_key, side)
            bucket = buckets.get(key)
            if bucket is None:
                bucket = _empty_bucket(player, stat, market_type, side)
                bucket["stat_key"] = stat_key
                buckets[key] = bucket
            bucket["prizepicks"] = quote
            _apply_roster(bucket, teams)

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

        odds_raw = row.get("american_price")
        odds_i: int | None
        if odds_raw is None:
            odds_i = None
        else:
            try:
                odds_i = int(odds_raw)
            except (TypeError, ValueError):
                odds_i = None

        stat_key = _ud_stat_key(stat_name)
        key = _merge_key(player, stat_key, side)
        bucket = buckets.get(key)
        if bucket is None:
            bucket = _empty_bucket(
                player,
                _ud_display_stat(stat_name, stat_key),
                f"underdog:{stat_name}",
                side,
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
            p.market_type,
            0 if p.side == "over" else 1,
        )
    )
    return props
