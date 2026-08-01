"""Select ParlayAPI main prop lines.

Sportsbooks: closest to balanced −110/−110.
DFS books: exact match to the sharpest sportsbook main that DFS also offers;
otherwise keep a DFS fallback line.
"""

from __future__ import annotations

from typing import Any

# Sharpest → softest among books we display. Used to pick DFS set lines.
SHARPNESS_ORDER: tuple[str, ...] = (
    "pinnacle",
    "novig",
    "bet365",
    "draftkings",
    "fanduel",
    "caesars",
    "betmgm",
)

SPORTSBOOK_BOOKS = frozenset(SHARPNESS_ORDER)

DFS_BOOKS = frozenset(
    {
        "prizepicks",
        "underdog",
        "betr",
        "sleeper",
        "pick6",
    }
)

PARLAY_PROP_BOOKS = SPORTSBOOK_BOOKS | DFS_BOOKS

_LINE_EPS = 1e-6


def _american_to_implied_prob(price: int) -> float:
    if price >= 100:
        return 100.0 / (price + 100.0)
    if price <= -100:
        return abs(price) / (abs(price) + 100.0)
    return 0.5


def balance_score(over_price: int, under_price: int) -> float:
    """Lower is closer to a balanced (~−110/−110) main line."""
    over_dist = abs(over_price - (-110))
    under_dist = abs(under_price - (-110))
    gap = abs(_american_to_implied_prob(over_price) - 0.5) + abs(
        _american_to_implied_prob(under_price) - 0.5
    )
    return float(over_dist + under_dist) + gap


def _lines_equal(a: float, b: float) -> bool:
    return abs(a - b) < _LINE_EPS


def _parse_prices(
    row: dict[str, Any],
) -> tuple[float, int | None, int | None, int, int] | None:
    """Return (line, over_keep, under_keep, over_for_score, under_for_score)."""
    line_raw = row.get("line")
    over_raw = row.get("over_price")
    under_raw = row.get("under_price")
    if line_raw is None:
        return None
    if over_raw is None and under_raw is None:
        return None
    try:
        line_f = float(line_raw)
        over_keep = int(over_raw) if over_raw is not None else None
        under_keep = int(under_raw) if under_raw is not None else None
    except (TypeError, ValueError):
        return None
    over_i = over_keep if over_keep is not None else -110
    under_i = under_keep if under_keep is not None else -110
    return line_f, over_keep, under_keep, over_i, under_i


def _kept_row(
    row: dict[str, Any],
    over_raw: int | None,
    under_raw: int | None,
) -> dict[str, Any]:
    kept = dict(row)
    if over_raw is None:
        kept.pop("over_price", None)
    else:
        kept["over_price"] = over_raw
    if under_raw is None:
        kept.pop("under_price", None)
    else:
        kept["under_price"] = under_raw
    return kept


def _sportsbook_main_lines(
    sportsbook_mains: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[tuple[str, str], dict[str, float]]:
    """Map (player, market) → {book: main_line}."""
    out: dict[tuple[str, str], dict[str, float]] = {}
    for (player, market, book), row in sportsbook_mains.items():
        try:
            line_f = float(row["line"])
        except (KeyError, TypeError, ValueError):
            continue
        out.setdefault((player, market), {})[book] = line_f
    return out


def _pick_dfs_row(
    candidates: list[tuple[float, int, int, dict[str, Any]]],
    book_lines: dict[str, float] | None,
) -> dict[str, Any] | None:
    """Match DFS line to sharpest sportsbook main; else DFS fallback."""
    if not candidates:
        return None

    if book_lines:
        for book in SHARPNESS_ORDER:
            target = book_lines.get(book)
            if target is None:
                continue
            matched = [c for c in candidates if _lines_equal(c[0], target)]
            if matched:
                matched.sort(key=lambda c: balance_score(c[1], c[2]))
                return matched[0][3]

    # No sharp match: keep whatever DFS offered (balance, then smaller line).
    best_row: dict[str, Any] | None = None
    best_rank: tuple[float, float] | None = None
    for line_f, over_i, under_i, kept in candidates:
        rank = (balance_score(over_i, under_i), abs(line_f))
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_row = kept
    return best_row


def select_parlay_main_lines(
    rows: list[dict[str, Any]],
    *,
    books: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Keep one main line per (player, market_key, bookmaker).

    Sportsbook mains use price balance. DFS mains walk ``SHARPNESS_ORDER`` and
    keep the first sportsbook main line that DFS also offers; if none match,
    keep a DFS fallback line. Sportsbook rows are always scanned for matching
    even if ``books`` excludes them from the returned set.
    """
    allowed = books if books is not None else PARLAY_PROP_BOOKS

    sportsbook_best: dict[tuple[str, str, str], tuple[float, dict[str, Any]]] = {}
    dfs_candidates: dict[
        tuple[str, str, str], list[tuple[float, int, int, dict[str, Any]]]
    ] = {}

    for row in rows:
        book = str(row.get("bookmaker") or "").lower().strip()
        if book not in PARLAY_PROP_BOOKS:
            continue
        player = str(row.get("player") or "").strip()
        market = str(row.get("market_key") or "").strip()
        if not player or not market.startswith("player_"):
            continue
        parsed = _parse_prices(row)
        if parsed is None:
            continue
        line_f, over_keep, under_keep, over_i, under_i = parsed
        player_key = player.lower()
        kept = _kept_row(row, over_keep, under_keep)

        if book in SPORTSBOOK_BOOKS:
            key = (player_key, market, book)
            score = balance_score(over_i, under_i)
            prev = sportsbook_best.get(key)
            if prev is None or score < prev[0]:
                sportsbook_best[key] = (score, kept)
        elif book in DFS_BOOKS:
            if book not in allowed:
                continue
            key = (player_key, market, book)
            dfs_candidates.setdefault(key, []).append(
                (line_f, over_i, under_i, kept)
            )

    mains_by_player_market = _sportsbook_main_lines(
        {k: v[1] for k, v in sportsbook_best.items()}
    )

    out: list[dict[str, Any]] = []
    for (player_key, market, book), (_score, kept) in sportsbook_best.items():
        if book in allowed:
            out.append(kept)

    for key, candidates in dfs_candidates.items():
        player_key, market, _book = key
        picked = _pick_dfs_row(
            candidates, mains_by_player_market.get((player_key, market))
        )
        if picked is not None:
            out.append(picked)

    return out
