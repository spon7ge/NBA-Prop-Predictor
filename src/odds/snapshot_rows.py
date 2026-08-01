"""Map scraper JSON projections/picks to odds table row dicts."""

from __future__ import annotations

from datetime import datetime


def parse_american_price(raw: str | int | None) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    text = str(raw).strip()
    if not text:
        return None
    return int(text.replace("+", ""))


def _parse_line_updated_at(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def prizepicks_projections_to_rows(
    projections: list[dict],
    *,
    league: str,
    scraped_at: datetime,
) -> list[dict]:
    rows: list[dict] = []
    league_key = league.lower()

    for projection in projections:
        player_name = projection.get("player")
        stat_type = projection.get("stat_type")
        line_score = projection.get("line_score")

        if not player_name or not stat_type or line_score is None:
            continue

        rows.append(
            {
                "league": league_key,
                "player_name": player_name,
                "stat_type": stat_type,
                "line_score": line_score,
                "odds_type": projection.get("odds_type") or "standard",
                "line_updated_at": _parse_line_updated_at(projection.get("updated_at")),
                "scraped_at": scraped_at,
            }
        )

    return rows


def underdog_picks_to_rows(
    picks: list[dict],
    *,
    league: str,
    scraped_at: datetime,
) -> list[dict]:
    rows: list[dict] = []
    league_key = league.lower()

    for pick in picks:
        player_name = pick.get("full_name")
        stat_name = pick.get("stat_name")
        stat_value = pick.get("stat_value")
        side = pick.get("choice")

        if not player_name or not stat_name or stat_value is None or not side:
            continue

        rows.append(
            {
                "league": league_key,
                "player_name": player_name,
                "stat_name": stat_name,
                "line_score": stat_value,
                "side": side,
                "american_price": parse_american_price(pick.get("american_price")),
                "payout_multiplier": pick.get("payout_multiplier"),
                "line_updated_at": _parse_line_updated_at(pick.get("updated_at")),
                "scraped_at": scraped_at,
            }
        )

    return rows


_VALID_SIDES = frozenset({"over", "under"})
_SHARP_BOOKS = frozenset({"fanduel", "draftkings"})
_PARLAY_BOOKS = frozenset(
    {
        "fanduel",
        "draftkings",
        "caesars",
        "betmgm",
        "pinnacle",
        "bet365",
        "prizepicks",
        "underdog",
        "betr",
        "novig",
        "sleeper",
        "pick6",
    }
)


def _sharp_player_name(row: dict) -> str | None:
    name = row.get("player_name") or row.get("selection")
    if not name:
        return None
    text = str(name).strip()
    return text or None


def sharp_props_to_book_rows(
    rows: list[dict],
    *,
    sportsbook: str,
    league: str,
    scraped_at: datetime,
) -> list[dict]:
    """Map Sharp prop API rows to odds.wnba_{fanduel|draftkings} row dicts."""
    book = sportsbook.lower().strip()
    if book not in _SHARP_BOOKS:
        raise ValueError(f"unsupported sportsbook: {sportsbook}")

    out: list[dict] = []
    league_key = league.lower()

    for row in rows:
        if not row.get("is_main_line", False):
            continue
        if str(row.get("sportsbook") or "").lower() != book:
            continue
        market = str(row.get("market_type") or "")
        if not market.startswith("player_"):
            continue
        side = str(row.get("selection_type") or "").lower()
        if side not in _VALID_SIDES:
            continue
        player = _sharp_player_name(row)
        if not player:
            continue
        line_raw = row.get("line")
        odds_raw = row.get("odds_american")
        if line_raw is None or odds_raw is None:
            continue
        try:
            line_score = float(line_raw)
            american_price = int(odds_raw)
        except (TypeError, ValueError):
            continue

        stat_category = row.get("stat_category")
        if stat_category is not None:
            stat_category = str(stat_category).strip() or None

        out.append(
            {
                "league": league_key,
                "player_name": player,
                "market_type": market,
                "stat_category": stat_category,
                "side": side,
                "line_score": line_score,
                "american_price": american_price,
                "scraped_at": scraped_at,
            }
        )

    return out


def parlay_props_to_book_rows(
    rows: list[dict],
    *,
    sportsbook: str,
    league: str,
    scraped_at: datetime,
) -> list[dict]:
    """Map Parlay prop API rows to odds.wnba_{book} row dicts (main lines only)."""
    from src.odds.parlay_main_lines import select_parlay_main_lines

    book = sportsbook.lower().strip()
    if book not in _PARLAY_BOOKS:
        raise ValueError(f"unsupported sportsbook: {sportsbook}")

    out: list[dict] = []
    league_key = league.lower()

    for row in select_parlay_main_lines(rows, books=frozenset({book})):
        player = str(row.get("player") or "").strip()
        market = str(row.get("market_key") or "").strip()
        if not player or not market.startswith("player_"):
            continue
        try:
            line_score = float(row["line"])
        except (KeyError, TypeError, ValueError):
            continue

        sides: list[tuple[str, int]] = []
        for side, raw in (
            ("over", row.get("over_price")),
            ("under", row.get("under_price")),
        ):
            if raw is None:
                continue
            try:
                sides.append((side, int(raw)))
            except (TypeError, ValueError):
                continue
        if not sides:
            continue

        market_label = str(row.get("market") or "").strip() or None
        for side, price in sides:
            out.append(
                {
                    "league": league_key,
                    "player_name": player,
                    "market_type": market,
                    "stat_category": market_label,
                    "side": side,
                    "line_score": line_score,
                    "american_price": price,
                    "scraped_at": scraped_at,
                }
            )

    return out
