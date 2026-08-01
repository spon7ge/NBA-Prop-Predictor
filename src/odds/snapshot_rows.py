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
