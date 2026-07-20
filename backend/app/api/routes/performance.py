"""GET /api/performance — live prop hit rates from ml.*_live_prop_grades."""
from __future__ import annotations

import datetime
from collections import defaultdict

from fastapi import APIRouter, HTTPException, Query

from app.core import db
from app.schemas.performance import (
    BookDailyTrend,
    DailyHitRate,
    GradedLeg,
    GradedParlay,
    GradedPick,
    HitRateBucket,
    ParlaySummary,
    PerformanceResponse,
)

router = APIRouter(tags=["performance"])

_VALID_LEAGUES = {"nba", "wnba"}

# Slate row bookmaker is a slug; grades / leg.platform use display names.
_BOOK_SLUG_TO_DISPLAY = {
    "prizepicks": "PrizePicks",
    "underdog": "Underdog",
    "draftkings": "DraftKings Pick6",
    "betr": "Betr DFS",
}
_BOOK_DISPLAY_TO_SLUG = {v.lower(): k for k, v in _BOOK_SLUG_TO_DISPLAY.items()}
for _slug, _name in _BOOK_SLUG_TO_DISPLAY.items():
    _BOOK_DISPLAY_TO_SLUG[_slug] = _slug


def _rate(hits: int, n: int) -> float | None:
    if n <= 0:
        return None
    return round(hits / n, 4)


def _bucket(key: str, hits: int, n: int, dnps: int = 0) -> HitRateBucket:
    return HitRateBucket(key=key, hits=hits, n=n, hit_rate=_rate(hits, n), dnps=dnps)


def _trend_from_day_map(
    by_day: dict[datetime.date, list[bool]],
) -> list[DailyHitRate]:
    return [
        DailyHitRate(
            game_date=d,
            hits=sum(1 for v in vals if v),
            n=len(vals),
            hit_rate=_rate(sum(1 for v in vals if v), len(vals)),
        )
        for d, vals in sorted(by_day.items())
    ]


def _trend_by_book_from_bools(
    by_book_day: dict[str, dict[datetime.date, list[bool]]],
) -> list[BookDailyTrend]:
    """Build per-book daily series, largest sample first."""
    series: list[BookDailyTrend] = []
    for book, by_day in sorted(
        by_book_day.items(),
        key=lambda kv: (-sum(len(v) for v in kv[1].values()), kv[0]),
    ):
        series.append(BookDailyTrend(bookmaker=book, points=_trend_from_day_map(by_day)))
    return series


def _norm_book(raw: str | None) -> str:
    """Normalize bookmaker / platform to a canonical display name."""
    if not raw:
        return ""
    s = str(raw).strip()
    low = s.lower()
    slug = _BOOK_DISPLAY_TO_SLUG.get(low)
    if slug:
        return _BOOK_SLUG_TO_DISPLAY.get(slug, s)
    return s


def _norm_name(raw: object) -> str:
    return str(raw or "").strip().lower()


def _line_key(line: object) -> float | None:
    if line is None:
        return None
    try:
        return round(float(line), 1)
    except (TypeError, ValueError):
        return None


def _grade_lookup_key(
    game_date: object,
    player: object,
    market: object,
    bookmaker: object,
    line: object = None,
) -> tuple:
    return (
        game_date,
        _norm_name(player),
        str(market or "").upper(),
        _norm_book(str(bookmaker) if bookmaker else "").lower(),
        _line_key(line),
    )


def _build_grades_sql(league: str) -> str:
    table = f"ml.{league}_live_prop_grades"
    return f"""
SELECT
    graded_at,
    run_at,
    game_date,
    player_name,
    team_abbr,
    opponent_abbr,
    market,
    bookmaker,
    line,
    side,
    stat_q10,
    stat_q50,
    p_over,
    p_under,
    actual_stat,
    actual_min,
    hit,
    miss_reason,
    abs_error
FROM {table}
WHERE game_date >= %(since)s
ORDER BY game_date DESC, player_name, market, bookmaker
"""


def _build_slates_sql(league: str) -> str:
    table = f"ml.{league}_live_slates"
    return f"""
SELECT
    run_at,
    game_date,
    bookmaker,
    n_legs,
    parlays
FROM {table}
WHERE game_date >= %(since)s
ORDER BY game_date DESC, n_legs, bookmaker
"""


def _dedupe_grades(rows: list[dict]) -> list[dict]:
    best: dict[tuple, dict] = {}
    for r in rows:
        key = (
            r["game_date"],
            r.get("player_name"),
            r.get("market"),
            r.get("bookmaker"),
        )
        prev = best.get(key)
        if prev is None or (r.get("run_at") or "") > (prev.get("run_at") or ""):
            best[key] = r
    return list(best.values())


def _index_grades(rows: list[dict]) -> dict[tuple, dict]:
    """Index by (date, player, market, book, line) and fallback without line."""
    idx: dict[tuple, dict] = {}
    for r in rows:
        full = _grade_lookup_key(
            r["game_date"],
            r.get("player_name"),
            r.get("market"),
            r.get("bookmaker"),
            r.get("line"),
        )
        idx[full] = r
        no_line = _grade_lookup_key(
            r["game_date"],
            r.get("player_name"),
            r.get("market"),
            r.get("bookmaker"),
            None,
        )
        # Prefer exact line match; only set no-line fallback once.
        idx.setdefault(no_line, r)
    return idx


def _find_grade(idx: dict[tuple, dict], game_date, player, market, bookmaker, line) -> dict | None:
    full = _grade_lookup_key(game_date, player, market, bookmaker, line)
    if full in idx:
        return idx[full]
    return idx.get(_grade_lookup_key(game_date, player, market, bookmaker, None))


def _leg_side(leg: dict) -> str:
    raw = leg.get("side")
    if not raw and isinstance(leg.get("model"), dict):
        raw = leg["model"].get("lean")
    return str(raw or "over").strip().lower()


def _grade_parlays(
    slate_rows: list[dict],
    grade_idx: dict[tuple, dict],
) -> tuple[list[GradedParlay], ParlaySummary]:
    # Latest slate run per game_date
    latest_run: dict[object, object] = {}
    for row in slate_rows:
        d = row["game_date"]
        ra = row.get("run_at")
        if d not in latest_run or (ra or "") > (latest_run[d] or ""):
            latest_run[d] = ra

    graded: list[GradedParlay] = []
    sum_cashed = 0
    sum_decided = 0
    sum_legs_hit = 0
    sum_legs_scored = 0

    for row in slate_rows:
        if row.get("run_at") != latest_run.get(row["game_date"]):
            continue
        game_date = row["game_date"]
        book_slug = str(row.get("bookmaker") or "").lower()
        book_display = _BOOK_SLUG_TO_DISPLAY.get(book_slug, book_slug)
        n_legs = int(row.get("n_legs") or 0)
        raw_parlays = row.get("parlays") or []
        if not isinstance(raw_parlays, list):
            continue

        for p in raw_parlays:
            if not isinstance(p, dict):
                continue
            legs_raw = p.get("LEGS") or p.get("legs") or []
            if not isinstance(legs_raw, list) or not legs_raw:
                continue

            out_legs: list[GradedLeg] = []
            legs_hit = 0
            legs_scored = 0
            legs_pending = 0
            any_miss = False

            for leg in legs_raw:
                if not isinstance(leg, dict):
                    continue
                player = leg.get("player") or leg.get("display_name") or ""
                market = str(leg.get("market") or "").upper()
                line = leg.get("dfs_line")
                if line is None:
                    line = leg.get("line")
                platform = leg.get("platform") or book_display
                side = _leg_side(leg)
                team = leg.get("team_abbr") or leg.get("team")

                g = _find_grade(grade_idx, game_date, player, market, platform, line)
                if g is None:
                    legs_pending += 1
                    out_legs.append(
                        GradedLeg(
                            player_name=str(player),
                            team_abbr=team,
                            market=market,
                            line=float(line) if line is not None else None,
                            side=side,
                            actual_stat=None,
                            hit=None,
                            miss_reason="ungraded",
                        )
                    )
                    continue

                reason = g.get("miss_reason") or ""
                # DNP kills the ticket — count as a miss, not "pending".
                if reason == "dnp":
                    any_miss = True
                    legs_scored += 1
                    out_legs.append(
                        GradedLeg(
                            player_name=str(player),
                            team_abbr=g.get("team_abbr") or team,
                            market=market,
                            line=g.get("line") if g.get("line") is not None else (
                                float(line) if line is not None else None
                            ),
                            side=side,
                            actual_stat=None,
                            hit=False,
                            miss_reason="dnp",
                        )
                    )
                    continue

                hit = bool(g.get("hit"))
                legs_scored += 1
                if hit:
                    legs_hit += 1
                else:
                    any_miss = True
                out_legs.append(
                    GradedLeg(
                        player_name=str(player),
                        team_abbr=g.get("team_abbr") or team,
                        market=market,
                        line=g.get("line") if g.get("line") is not None else (
                            float(line) if line is not None else None
                        ),
                        side=side,
                        actual_stat=g.get("actual_stat"),
                        hit=hit,
                        miss_reason=reason or None,
                    )
                )

            if not out_legs:
                continue

            # Ticket hits only if EVERY leg hit. Any miss/DNP → miss.
            # Still-waiting (ungraded) with no misses yet → pending.
            n_legs_total = len(out_legs)
            all_hit = (
                n_legs_total > 0
                and legs_pending == 0
                and not any_miss
                and legs_hit == n_legs_total
            )
            if all_hit:
                cashed: bool | None = True
            elif any_miss:
                cashed = False
            else:
                cashed = None  # ungraded remaining, no miss yet

            if cashed is not None:
                sum_decided += 1
                if cashed:
                    sum_cashed += 1
            sum_legs_hit += legs_hit
            sum_legs_scored += legs_scored

            graded.append(
                GradedParlay(
                    game_date=game_date,
                    bookmaker=book_display,
                    n_legs=n_legs or n_legs_total,
                    legs_hit=legs_hit,
                    legs_scored=legs_scored,
                    legs_pending=legs_pending,
                    cashed=cashed,
                    parlay_prob=p.get("PARLAY_PROB") if p.get("PARLAY_PROB") is not None else p.get("parlay_prob"),
                    ev=p.get("EV") if p.get("EV") is not None else p.get("ev"),
                    legs=out_legs,
                )
            )

    graded.sort(
        key=lambda gp: (
            0 if gp.legs_scored > 0 else 1,
            0 if gp.cashed is True else (1 if gp.cashed is False else 2),
            -(gp.game_date.toordinal() if isinstance(gp.game_date, datetime.date) else 0),
            -(gp.legs_hit / gp.legs_scored if gp.legs_scored else 0),
            gp.bookmaker,
            gp.n_legs,
        ),
    )

    summary = ParlaySummary(
        cashed=sum_cashed,
        decided=sum_decided,
        cash_rate=_rate(sum_cashed, sum_decided),
        legs_hit=sum_legs_hit,
        legs_scored=sum_legs_scored,
        leg_hit_rate=_rate(sum_legs_hit, sum_legs_scored),
    )
    return graded, summary


def _parlay_summary_from(parlays: list[GradedParlay]) -> ParlaySummary:
    cashed = sum(1 for p in parlays if p.cashed is True)
    decided = sum(1 for p in parlays if p.cashed is not None)
    legs_hit = sum(p.legs_hit for p in parlays)
    legs_scored = sum(p.legs_scored for p in parlays)
    return ParlaySummary(
        cashed=cashed,
        decided=decided,
        cash_rate=_rate(cashed, decided),
        legs_hit=legs_hit,
        legs_scored=legs_scored,
        leg_hit_rate=_rate(legs_hit, legs_scored),
    )


def _aggregate_parlay_mode(
    parlays: list[GradedParlay],
    *,
    days: int,
) -> PerformanceResponse:
    """Accuracy from parlay cash / leg outcomes (used when filtering to N-leg)."""
    today = datetime.date.today()
    last_night = today - datetime.timedelta(days=1)
    decided = [p for p in parlays if p.cashed is not None]
    dates_present = sorted({p.game_date for p in parlays}, reverse=True)
    headline_date = last_night if last_night in dates_present else (
        dates_present[0] if dates_present else last_night
    )

    night = [p for p in decided if p.game_date == headline_date]
    nh = sum(1 for p in night if p.cashed)
    nn = len(night)
    wh = sum(1 for p in decided if p.cashed)
    wn = len(decided)

    by_market: dict[str, list[bool]] = defaultdict(list)
    by_book: dict[str, list[bool]] = defaultdict(list)
    by_side: dict[str, list[bool]] = defaultdict(list)
    by_day: dict[datetime.date, list[bool]] = defaultdict(list)
    by_book_day: dict[str, dict[datetime.date, list[bool]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for p in decided:
        by_book[p.bookmaker].append(bool(p.cashed))
        by_day[p.game_date].append(bool(p.cashed))
        by_book_day[p.bookmaker][p.game_date].append(bool(p.cashed))

    for p in parlays:
        for leg in p.legs:
            # Only count fully cashed tickets' legs toward market accuracy,
            # OR count every scored leg? User wants ticket-level: skip partials.
            # Use only legs from settled tickets; hits only when ticket cashed.
            if p.cashed is None:
                continue
            if leg.hit is None:
                continue
            # For market breakdown: individual leg outcomes on settled tickets
            by_market[leg.market or "?"].append(bool(leg.hit))
            by_side[(leg.side or "?").lower()].append(bool(leg.hit))

    def bucket_bools(key: str, vals: list[bool]) -> HitRateBucket:
        hits = sum(1 for v in vals if v)
        return _bucket(key, hits, len(vals), 0)

    market_buckets = []
    for key in ("PTS", "REB", "AST"):
        if key in by_market:
            market_buckets.append(bucket_bools(key, by_market[key]))
    for key, vals in sorted(by_market.items()):
        if key in ("PTS", "REB", "AST"):
            continue
        market_buckets.append(bucket_bools(key, vals))

    book_buckets = [
        bucket_bools(k, v)
        for k, v in sorted(by_book.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    ]
    side_buckets = []
    for key in ("over", "under"):
        if key in by_side:
            side_buckets.append(bucket_bools(key, by_side[key]))

    trend = _trend_from_day_map(by_day)
    trend_by_book = _trend_by_book_from_bools(by_book_day)

    # Flatten scored legs from settled tickets into recent_picks
    recent: list[GradedPick] = []
    seen: set[tuple] = set()
    for p in sorted(parlays, key=lambda x: x.game_date, reverse=True):
        if p.cashed is None:
            continue
        for leg in p.legs:
            if leg.hit is None:
                continue
            key = (p.game_date, leg.player_name, leg.market, p.bookmaker, leg.side)
            if key in seen:
                continue
            seen.add(key)
            recent.append(
                GradedPick(
                    game_date=p.game_date,
                    player_name=leg.player_name,
                    team_abbr=leg.team_abbr,
                    market=leg.market,
                    bookmaker=p.bookmaker,
                    line=leg.line,
                    side=leg.side,
                    actual_stat=leg.actual_stat,
                    hit=bool(leg.hit),
                    miss_reason=leg.miss_reason or ("clean_hit" if leg.hit else "miss"),
                )
            )
            if len(recent) >= 40:
                break
        if len(recent) >= 40:
            break
    summary = _parlay_summary_from(parlays)
    night_key = "last_night"
    if headline_date != last_night:
        night_key = f"slate_{headline_date.isoformat()}"

    return PerformanceResponse(
        generated_at=datetime.datetime.now(datetime.timezone.utc),
        league="",
        days=days,
        last_night=_bucket(night_key, nh, nn, 0),
        last_n_days=_bucket(f"last_{days}_days", wh, wn, 0),
        by_market=market_buckets,
        by_book=book_buckets,
        by_side=side_buckets,
        trend=trend,
        trend_by_book=trend_by_book,
        brier_score=None,
        recent_picks=recent,
        parlay_summary=summary,
        graded_parlays=parlays[:30],
    )


def _aggregate(
    rows: list[dict],
    *,
    days: int,
    graded_parlays: list[GradedParlay] | None = None,
    parlay_summary: ParlaySummary | None = None,
) -> PerformanceResponse:
    today = datetime.date.today()
    last_night = today - datetime.timedelta(days=1)

    dates_present = sorted({r["game_date"] for r in rows}, reverse=True)
    headline_date = last_night if last_night in dates_present else (
        dates_present[0] if dates_present else last_night
    )

    def scored(rs: list[dict]) -> list[dict]:
        return [r for r in rs if r.get("miss_reason") != "dnp"]

    def tally(rs: list[dict]) -> tuple[int, int, int]:
        s = scored(rs)
        hits = sum(1 for r in s if r.get("hit"))
        dnps = sum(1 for r in rs if r.get("miss_reason") == "dnp")
        return hits, len(s), dnps

    night_rows = [r for r in rows if r["game_date"] == headline_date]
    nh, nn, nd = tally(night_rows)
    wh, wn, wd = tally(rows)

    by_market: dict[str, list[dict]] = defaultdict(list)
    by_book: dict[str, list[dict]] = defaultdict(list)
    by_side: dict[str, list[dict]] = defaultdict(list)
    by_day: dict[datetime.date, list[dict]] = defaultdict(list)
    by_book_day: dict[str, dict[datetime.date, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for r in rows:
        book = str(r.get("bookmaker") or "?")
        by_market[str(r.get("market") or "?")].append(r)
        by_book[book].append(r)
        by_side[str(r.get("side") or "?").lower()].append(r)
        by_day[r["game_date"]].append(r)
        by_book_day[book][r["game_date"]].append(r)

    market_buckets = []
    for key in ("PTS", "REB", "AST"):
        if key in by_market:
            h, n, d = tally(by_market[key])
            market_buckets.append(_bucket(key, h, n, d))
    for key, rs in sorted(by_market.items()):
        if key in ("PTS", "REB", "AST"):
            continue
        h, n, d = tally(rs)
        market_buckets.append(_bucket(key, h, n, d))

    book_buckets = []
    for key, rs in sorted(by_book.items(), key=lambda kv: (-len(scored(kv[1])), kv[0])):
        h, n, d = tally(rs)
        book_buckets.append(_bucket(key, h, n, d))

    side_buckets = []
    for key in ("over", "under"):
        if key in by_side:
            h, n, d = tally(by_side[key])
            side_buckets.append(_bucket(key, h, n, d))

    trend = []
    for d in sorted(by_day.keys()):
        h, n, _ = tally(by_day[d])
        trend.append(DailyHitRate(game_date=d, hits=h, n=n, hit_rate=_rate(h, n)))

    trend_by_book: list[BookDailyTrend] = []
    for book, day_map in sorted(
        by_book_day.items(),
        key=lambda kv: (-sum(len(scored(rs)) for rs in kv[1].values()), kv[0]),
    ):
        points: list[DailyHitRate] = []
        for d in sorted(day_map.keys()):
            h, n, _ = tally(day_map[d])
            points.append(DailyHitRate(game_date=d, hits=h, n=n, hit_rate=_rate(h, n)))
        trend_by_book.append(BookDailyTrend(bookmaker=book, points=points))

    brier: float | None = None
    sq_err = 0.0
    brier_n = 0
    for r in scored(rows):
        p = r.get("p_over")
        actual = r.get("actual_stat")
        line = r.get("line")
        if p is None or actual is None or line is None:
            continue
        y = 1.0 if float(actual) > float(line) else 0.0
        sq_err += (float(p) - y) ** 2
        brier_n += 1
    if brier_n > 0:
        brier = round(sq_err / brier_n, 4)

    recent_src = sorted(
        scored(rows),
        key=lambda r: (
            r["game_date"],
            0 if r.get("hit") else 1,
            r.get("player_name") or "",
        ),
        reverse=True,
    )[:40]

    recent = [
        GradedPick(
            game_date=r["game_date"],
            player_name=r.get("player_name") or "",
            team_abbr=r.get("team_abbr"),
            market=r.get("market") or "",
            bookmaker=r.get("bookmaker") or "",
            line=r.get("line"),
            side=str(r.get("side") or "").lower(),
            stat_q50=r.get("stat_q50"),
            p_over=r.get("p_over"),
            actual_stat=r.get("actual_stat"),
            hit=bool(r.get("hit")),
            miss_reason=r.get("miss_reason") or "",
            abs_error=r.get("abs_error"),
        )
        for r in recent_src
    ]

    night_key = "last_night"
    if headline_date != last_night:
        night_key = f"slate_{headline_date.isoformat()}"

    return PerformanceResponse(
        generated_at=datetime.datetime.now(datetime.timezone.utc),
        league="",
        days=days,
        last_night=_bucket(night_key, nh, nn, nd),
        last_n_days=_bucket(f"last_{days}_days", wh, wn, wd),
        by_market=market_buckets,
        by_book=book_buckets,
        by_side=side_buckets,
        trend=trend,
        trend_by_book=trend_by_book,
        brier_score=brier,
        recent_picks=recent,
        parlay_summary=parlay_summary or ParlaySummary(),
        graded_parlays=(graded_parlays or [])[:30],
    )


_VALID_LEGS = {"all", "singles", "2", "3", "5", "6"}


@router.get("/performance", response_model=PerformanceResponse)
def get_performance(
    league: str = Query(default="wnba", description="nba | wnba"),
    days: int = Query(default=7, ge=1, le=90, description="Lookback window in days"),
    since: str | None = Query(
        default=None,
        description="Optional start date YYYY-MM-DD (overrides days).",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    ),
    book: str | None = Query(
        default=None,
        description=(
            "Optional book filter: prizepicks | underdog | draftkings | betr "
            "(or display name). When set, hit rates / parlays / Brier are scoped "
            "to that book only."
        ),
    ),
    legs: str = Query(
        default="all",
        description="all | singles | 2 | 3 | 5 | 6 — scopes lists and accuracy.",
    ),
) -> PerformanceResponse:
    """Hit rates for graded live props (DNPs excluded from denominators).

    Also joins latest Top Legs parlays to grades so each parlay shows per-leg
    hit/miss and whether the ticket cashed.
    """
    league_lc = league.lower()
    if league_lc not in _VALID_LEAGUES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid league '{league}'. Valid: {sorted(_VALID_LEAGUES)}",
        )

    legs_mode = (legs or "all").strip().lower()
    if legs_mode not in _VALID_LEGS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid legs '{legs}'. Valid: {sorted(_VALID_LEGS)}",
        )
    n_legs_filter: int | None = int(legs_mode) if legs_mode in {"2", "3", "5", "6"} else None

    book_filter: str | None = None
    book_slug: str | None = None
    if book:
        book_filter = _norm_book(book)
        low = book.strip().lower()
        book_slug = _BOOK_DISPLAY_TO_SLUG.get(low) or _BOOK_DISPLAY_TO_SLUG.get(
            book_filter.lower()
        )
        if book_slug is None and low in _BOOK_SLUG_TO_DISPLAY:
            book_slug = low
            book_filter = _BOOK_SLUG_TO_DISPLAY[low]
        if book_slug is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Invalid book '{book}'. Valid: "
                    f"{sorted(_BOOK_SLUG_TO_DISPLAY)} (or display names)."
                ),
            )

    if since:
        since_date = datetime.date.fromisoformat(since)
        lookback_days = max((datetime.date.today() - since_date).days, 1)
    else:
        lookback_days = days
        since_date = datetime.date.today() - datetime.timedelta(days=days)

    try:
        rows = db.query(
            _build_grades_sql(league_lc),
            {"since": since_date.isoformat()},
        )
    except HTTPException as exc:
        detail = str(exc.detail) if exc.detail else ""
        if exc.status_code == 500 and "does not exist" in detail:
            rows = []
        else:
            raise

    all_deduped = _dedupe_grades(rows)
    deduped = all_deduped
    if book_filter:
        want = book_filter.lower()
        deduped = [
            r
            for r in all_deduped
            if _norm_book(str(r.get("bookmaker") or "")).lower() == want
        ]

    grade_idx = _index_grades(all_deduped)

    graded_parlays: list[GradedParlay] = []
    parlay_summary = ParlaySummary()
    if legs_mode != "singles":
        try:
            slate_rows = db.query(
                _build_slates_sql(league_lc),
                {"since": since_date.isoformat()},
            )
            if book_slug:
                slate_rows = [
                    r
                    for r in slate_rows
                    if str(r.get("bookmaker") or "").lower() == book_slug
                ]
            if n_legs_filter is not None:
                slate_rows = [
                    r
                    for r in slate_rows
                    if int(r.get("n_legs") or 0) == n_legs_filter
                ]
            graded_parlays, parlay_summary = _grade_parlays(slate_rows, grade_idx)
        except HTTPException as exc:
            detail = str(exc.detail) if exc.detail else ""
            if not (exc.status_code == 500 and "does not exist" in detail):
                raise

    if n_legs_filter is not None:
        resp = _aggregate_parlay_mode(graded_parlays, days=lookback_days)
    elif legs_mode == "singles":
        resp = _aggregate(
            deduped,
            days=lookback_days,
            graded_parlays=[],
            parlay_summary=ParlaySummary(),
        )
    else:
        resp = _aggregate(
            deduped,
            days=lookback_days,
            graded_parlays=graded_parlays,
            parlay_summary=parlay_summary,
        )
    resp.league = league_lc
    return resp
