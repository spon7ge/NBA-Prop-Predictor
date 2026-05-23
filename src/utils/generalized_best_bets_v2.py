"""
src/utils/generalized_best_bets_v2.py

Enriches DFS picks (PrizePicks, Underdog, Betr DFS, DraftKings Pick6) with
sharp-book context from NBA_US_* CSVs.

Data sources:
  - data/raw/player_lines/NBA_DFS_*.csv  : DFS platform lines to evaluate
  - data/raw/player_lines/NBA_US_*.csv   : Sharp book reference lines
  - data/props/circa+betonline_team_lines/*.json : Game spreads/totals (optional)

Same enrichment/tier logic as underdog_lines.py; generalised to all DFS platforms
and the unified CSV schema (BOOKMAKER, CATEGORY, NAME, OVER/UNDER, LINE, ODDS).

Tier values:
  "sharp_verified" : real sharp book (Pinnacle/FD/DK) agrees with model lean
  "dfs_only"       : no sharp coverage; model agrees with DFS platform's own no-vig
  "conflict"       : model lean disagrees with sharp lean
  "no_model"       : no model row for this pick (line gap too large, or non-PTS/AST/REB)
"""
from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SHARP_BOOK_ORDER = ["Pinnacle", "FanDuel", "DraftKings", "BetMGM", "BetOnline.ag", "Bovada"]
TEAM_BOOK_ORDER  = ["Circa", "BetOnline.ag", "FanDuel", "DraftKings"]
CIRCA_BETONLINE_TEAM_LINES_DIR = Path("data/props/circa+betonline_team_lines")
DK_FD_TEAM_LINES_DIR = Path("data/props/dk+fd_team_lines")

# CATEGORY column in CSVs → internal market key
CATEGORY_TO_MARKET: dict[str, str] = {
    "player_points":                  "PTS",
    "player_rebounds":                "REB",
    "player_assists":                 "AST",
    "player_threes":                  "3PM",
    "player_blocks":                  "BLK",
    "player_steals":                  "STL",
    "player_turnovers":               "TOV",
    "player_field_goals":             "FGM",
    "player_frees_made":              "FTM",
    "player_frees_attempts":          "FTA",
    "player_points_rebounds_assists": "PRA",
    "player_points_rebounds":         "PR",
    "player_points_assists":          "PA",
    "player_rebounds_assists":        "RA",
    "player_blocks_steals":           "BS",
}

# Markets covered by all_line_probs (P_OVER / P_UNDER available)
MODEL_MARKETS: frozenset[str] = frozenset({"PTS", "AST", "REB"})

# Market → base_df stat column (for form / vs_opp; only modelled markets)
MARKET_TO_BASE_COL: dict[str, str] = {"PTS": "PTS", "AST": "AST", "REB": "REB"}

TEAM_ABBR_TO_FULL = {
    "ATL": "Atlanta Hawks",          "BOS": "Boston Celtics",      "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",      "CHI": "Chicago Bulls",       "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",       "DEN": "Denver Nuggets",      "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",  "HOU": "Houston Rockets",     "IND": "Indiana Pacers",
    "LAC": "LA Clippers",            "LAL": "Los Angeles Lakers",  "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",             "MIL": "Milwaukee Bucks",     "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",   "NYK": "New York Knicks",     "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",          "PHI": "Philadelphia 76ers",  "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings",    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",        "UTA": "Utah Jazz",           "WAS": "Washington Wizards",
}
TEAM_FULL_TO_ABBR = {v: k for k, v in TEAM_ABBR_TO_FULL.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
_SUFFIX_RE = re.compile(r"\s+(jr|sr|ii|iii|iv)\.?$", re.IGNORECASE)
_PUNCT_RE  = re.compile(r"[.\'\-'`]")


def normalize_name(name: Any) -> str:
    """Lowercase, strip accents, drop punctuation + name suffixes."""
    if not isinstance(name, str):
        return ""
    n = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    n = n.lower().strip()
    n = _SUFFIX_RE.sub("", n)
    n = _PUNCT_RE.sub("", n)
    return re.sub(r"\s+", " ", n).strip()


def american_to_implied(american) -> np.ndarray:
    """Vectorized American-odds → implied probability."""
    s = pd.to_numeric(pd.Series(american), errors="coerce").to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(
            s > 0,
            100.0 / (s + 100.0),
            np.where(s < 0, -s / (-s + 100.0), np.nan),
        )


def _to_opt_float(v) -> float | None:
    """Return float or None — drops NaN/inf so JSON stays clean."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (pd.isna(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if (np.isnan(o) or np.isinf(o)) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    raise TypeError(f"not JSON serializable: {type(o)}")


# ─────────────────────────────────────────────────────────────────────────────
# NBA-API team ratings
# Disk cache at data/raw/cache/team_ratings.csv — survives restarts and API
# failures. Re-fetches only when the file is missing or explicitly stale.
# ─────────────────────────────────────────────────────────────────────────────
_TEAM_RATINGS_CACHE = Path("data/raw/cache/team_ratings.csv")


def fetch_team_ratings(*, force_refresh: bool = True) -> pd.DataFrame:
    """Return team DEF_RATING / PACE indexed by TEAM_NAME.

    Resolution order:
      1. In-process memory cache (fastest)
      2. Disk cache at data/raw/cache/team_ratings.csv
      3. Live NBA API call (writes result to disk for next time)
    """
    if not force_refresh and getattr(fetch_team_ratings, "_cache", None) is not None:
        return fetch_team_ratings._cache

    if not force_refresh and _TEAM_RATINGS_CACHE.is_file():
        out = pd.read_csv(_TEAM_RATINGS_CACHE, index_col="TEAM_NAME")
        fetch_team_ratings._cache = out
        return out

    from nba_api.stats.endpoints import leaguedashteamstats
    df = leaguedashteamstats.LeagueDashTeamStats(
        league_id_nullable="00",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced",
        season_type_all_star="Playoffs"
    ).get_data_frames()[0]
    out = (
        df[["TEAM_NAME", "DEF_RATING", "DEF_RATING_RANK", "PACE", "PACE_RANK"]]
        .set_index("TEAM_NAME")
    )
    _TEAM_RATINGS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(_TEAM_RATINGS_CACHE)
    fetch_team_ratings._cache = out
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Opponent resolution
# ─────────────────────────────────────────────────────────────────────────────
_schedule_cache: dict[str, pd.DataFrame] = {}


def _get_schedule_for(date_str: str) -> pd.DataFrame:
    if date_str not in _schedule_cache:
        from nba_api.stats.endpoints import scheduleleaguev2
        sched = scheduleleaguev2.ScheduleLeagueV2().get_data_frames()[0]
        sched["gameDate"] = pd.to_datetime(sched["gameDate"]).dt.strftime("%Y-%m-%d")
        _schedule_cache[date_str] = sched
    return _schedule_cache[date_str]


def find_opp(player_canonical: str, players_df: pd.DataFrame,
             game_date: str, max_days_ahead: int = 3) -> tuple[str | None, int | None]:
    rows = players_df.loc[players_df["PLAYER_NAME"] == player_canonical, "TEAM_ABBREVIATION"]
    if rows.empty:
        return None, None
    player_team = rows.iloc[0]
    base = datetime.strptime(game_date, "%Y-%m-%d")
    for i in range(max_days_ahead + 1):
        d = (base + timedelta(days=i)).strftime("%Y-%m-%d")
        sched   = _get_schedule_for(d)
        sched_f = sched[sched["gameDate"] == d]
        homes   = sched_f["homeTeam_teamTricode"].unique().tolist()
        aways   = sched_f["awayTeam_teamTricode"].unique().tolist()
        if player_team in homes:
            return aways[homes.index(player_team)], 1
        if player_team in aways:
            return homes[aways.index(player_team)], 0
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Team line index — compatible with team-lines JSON format
# ─────────────────────────────────────────────────────────────────────────────
def _load_team_odds_json(team_odds_source) -> dict:
    """
    Accepts a path to a JSON file, a directory of JSON files, or an already-loaded dict.
    Directories resolve to their most recently modified JSON file.
    """
    if team_odds_source is None:
        return {"records": []}
    if not isinstance(team_odds_source, (str, Path)):
        return team_odds_source

    path = Path(team_odds_source)
    if path.is_dir():
        files = list(path.glob("*.json"))
        if not files:
            return {"records": []}
        path = max(files, key=lambda f: f.stat().st_mtime)

    with open(path) as f:
        return json.load(f)


def build_team_line_index(team_odds_source) -> dict:
    """
    Accepts a path to a JSON file or an already-loaded dict.
    Returns dict keyed by frozenset({home_abbr, away_abbr}) with team spread
    and total lines. Book priority: Circa → BetOnline.ag → FanDuel → DraftKings.
    """
    primary_source = (
        team_odds_source
        if team_odds_source is not None
        else CIRCA_BETONLINE_TEAM_LINES_DIR
    )
    sources = [_load_team_odds_json(primary_source)]
    if DK_FD_TEAM_LINES_DIR.is_dir():
        sources.append(_load_team_odds_json(DK_FD_TEAM_LINES_DIR))

    idx: dict = {}
    for data in sources:
        for rec in data.get("records", []):
            home_full = rec.get("HOME")
            away_full = rec.get("AWAY")
            if home_full not in TEAM_FULL_TO_ABBR or away_full not in TEAM_FULL_TO_ABBR:
                continue
            home_ab = TEAM_FULL_TO_ABBR[home_full]
            away_ab = TEAM_FULL_TO_ABBR[away_full]
            key = frozenset({home_ab, away_ab})
            if key not in idx:
                idx[key] = {
                    "HOME_abbr": home_ab, "AWAY_abbr": away_ab,
                    "HOME_full": home_full, "AWAY_full": away_full,
                }
            bk = rec.get("BOOKMAKER")
            if bk not in TEAM_BOOK_ORDER:
                continue
            slot = idx[key].setdefault(bk, {"spread_line": None, "total_line": None})
            mkt  = rec.get("MARKET")
            line = rec.get("LINE")
            if mkt == "Spread":
                slot["spread_line"] = line      # home-team perspective
            elif mkt == "Totals":
                slot["total_line"] = line
    return idx


def get_team_game_context(team_line_idx: dict, player_team_abbr: str,
                          opp_abbr: str, is_home: int) -> dict:
    """Use team-book priority order. Converts spread to player-team perspective."""
    blank = {"book_used": None, "game_total": None, "spread": None, "team_total_implied": None}
    if player_team_abbr is None or opp_abbr is None:
        return blank
    info = team_line_idx.get(frozenset({player_team_abbr, opp_abbr}))
    if info is None:
        return blank
    for book in TEAM_BOOK_ORDER:
        slot = info.get(book)
        if not slot:
            continue
        spread_home = slot.get("spread_line")
        total_line  = slot.get("total_line")
        if spread_home is None or total_line is None:
            continue
        spread_for_team = spread_home if is_home else -spread_home
        team_total_implied = (total_line / 2.0) - (spread_for_team / 2.0)
        return {
            "book_used":          book,
            "game_total":         _to_opt_float(total_line),
            "spread":             _to_opt_float(spread_for_team),
            "team_total_implied": _to_opt_float(team_total_implied),
        }
    return blank


# ─────────────────────────────────────────────────────────────────────────────
# CSV pivoting — shared schema for NBA_DFS_* and NBA_US_*
# ─────────────────────────────────────────────────────────────────────────────
def pivot_odds_csv(df: pd.DataFrame, *,
                   bookmakers: list[str] | None = None) -> pd.DataFrame:
    """
    Input:  long-format CSV with columns BOOKMAKER, CATEGORY, NAME, OVER/UNDER, LINE, ODDS
    Output: wide-format, one row per (BOOKMAKER, CATEGORY, NAME, LINE):
              BOOKMAKER, CATEGORY, NAME, MARKET, NORM_NAME, LINE,
              AMERICAN_OVER, AMERICAN_UNDER, NO_VIG_IMPLIED_OVER, NO_VIG_IMPLIED_UNDER
    """
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    if bookmakers:
        work = work[work["BOOKMAKER"].isin(bookmakers)]
    if work.empty:
        return pd.DataFrame()

    work["LINE"] = pd.to_numeric(work["LINE"], errors="coerce")
    work["ODDS"] = pd.to_numeric(work["ODDS"], errors="coerce")

    over_df  = work[work["OVER/UNDER"] == "Over" ].rename(columns={"ODDS": "AMERICAN_OVER"})
    under_df = work[work["OVER/UNDER"] == "Under"].rename(columns={"ODDS": "AMERICAN_UNDER"})

    key_cols = ["BOOKMAKER", "CATEGORY", "NAME", "LINE"]
    wide = over_df[key_cols + ["AMERICAN_OVER"]].merge(
        under_df[key_cols + ["AMERICAN_UNDER"]],
        on=key_cols, how="outer",
    )

    imp_over  = american_to_implied(wide["AMERICAN_OVER"].to_numpy())
    imp_under = american_to_implied(wide["AMERICAN_UNDER"].to_numpy())
    total     = imp_over + imp_under
    with np.errstate(invalid="ignore", divide="ignore"):
        wide["NO_VIG_IMPLIED_OVER"]  = np.where(total > 0, imp_over  / total, np.nan)
        wide["NO_VIG_IMPLIED_UNDER"] = np.where(total > 0, imp_under / total, np.nan)

    wide["MARKET"]    = wide["CATEGORY"].map(CATEGORY_TO_MARKET)
    wide["NORM_NAME"] = wide["NAME"].map(normalize_name)

    return wide.dropna(subset=["LINE", "MARKET"]).reset_index(drop=True)


def build_sharp_books_from_csv(us_wide: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split a pivoted NBA_US_* DataFrame into per-book DataFrames."""
    needed = ["NORM_NAME", "MARKET", "LINE",
              "AMERICAN_OVER", "AMERICAN_UNDER",
              "NO_VIG_IMPLIED_OVER", "NO_VIG_IMPLIED_UNDER"]
    books: dict[str, pd.DataFrame] = {}
    for book in SHARP_BOOK_ORDER:
        sub = us_wide[us_wide["BOOKMAKER"] == book][needed].copy().reset_index(drop=True)
        if not sub.empty:
            books[book] = sub
    return books


# ─────────────────────────────────────────────────────────────────────────────
# Sharp line lookup + consensus
# ─────────────────────────────────────────────────────────────────────────────
def get_sharp_line(sharp_books: dict[str, pd.DataFrame],
                   norm_name: str, market: str, dfs_line: float,
                   fallback_row: dict, *, fallback_source: str) -> dict:
    """Pinnacle → FanDuel → DraftKings → BetMGM → BetOnline.ag → Bovada; else DFS fallback."""
    for book in SHARP_BOOK_ORDER:
        df = sharp_books.get(book)
        if df is None or df.empty:
            continue
        hits = df[(df["NORM_NAME"] == norm_name) & (df["MARKET"] == market)]
        if hits.empty:
            continue
        idx = (hits["LINE"] - dfs_line).abs().idxmin()
        row = hits.loc[idx]
        return {
            "source":         book,
            "is_fallback":    False,
            "line":           _to_opt_float(row["LINE"]),
            "no_vig_over":    _to_opt_float(row["NO_VIG_IMPLIED_OVER"]),
            "no_vig_under":   _to_opt_float(row["NO_VIG_IMPLIED_UNDER"]),
            "american_over":  _to_opt_float(row["AMERICAN_OVER"]),
            "american_under": _to_opt_float(row["AMERICAN_UNDER"]),
        }
    return {
        "source":         fallback_source,
        "is_fallback":    True,
        "line":           _to_opt_float(dfs_line),
        "no_vig_over":    _to_opt_float(fallback_row.get("NO_VIG_IMPLIED_OVER")),
        "no_vig_under":   _to_opt_float(fallback_row.get("NO_VIG_IMPLIED_UNDER")),
        "american_over":  _to_opt_float(fallback_row.get("AMERICAN_OVER")),
        "american_under": _to_opt_float(fallback_row.get("AMERICAN_UNDER")),
    }


def collect_offerings(sharp_books: dict[str, pd.DataFrame],
                      norm_name: str, market: str) -> list[dict]:
    """Every book/line that covers this player+market."""
    out: list[dict] = []
    for book, df in sharp_books.items():
        if df is None or df.empty:
            continue
        hits = df[(df["NORM_NAME"] == norm_name) & (df["MARKET"] == market)]
        for _, r in hits.iterrows():
            out.append({
                "book":           book,
                "line":           _to_opt_float(r["LINE"]),
                "no_vig_over":    _to_opt_float(r["NO_VIG_IMPLIED_OVER"]),
                "no_vig_under":   _to_opt_float(r["NO_VIG_IMPLIED_UNDER"]),
                "american_over":  _to_opt_float(r["AMERICAN_OVER"]),
                "american_under": _to_opt_float(r["AMERICAN_UNDER"]),
            })
    return out


def compute_consensus(offerings: list[dict], dfs_line: float) -> dict:
    """Same-line and any-line consensus on no-vig OVER probability."""
    if not offerings:
        return {
            "n_books_same_line":          0, "mean_no_vig_over_same_line": None,
            "n_books_any_line":           0, "mean_no_vig_over_any_line":  None,
        }
    same = [o["no_vig_over"] for o in offerings
            if o["line"] is not None
            and abs(o["line"] - dfs_line) < 1e-9
            and o["no_vig_over"] is not None]
    any_ = [o["no_vig_over"] for o in offerings if o["no_vig_over"] is not None]
    return {
        "n_books_same_line":          len(same),
        "mean_no_vig_over_same_line": float(np.mean(same)) if same else None,
        "n_books_any_line":           len(any_),
        "mean_no_vig_over_any_line":  float(np.mean(any_)) if any_ else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Recent form + vs-opp (base_df must be sorted GAME_DATE desc)
# ─────────────────────────────────────────────────────────────────────────────
def compute_form_and_vsopp(player_log: pd.DataFrame, stat_col: str,
                           dfs_line: float, opp_abbr: str | None) -> dict:
    out = {
        "form":   {"over_l5": None, "over_l10": None, "over_l15": None},
        "vs_opp": {"n_games": 0, "avg_stat": None, "over_rate_at_line": None},
    }
    if player_log.empty or stat_col not in player_log.columns:
        return out

    s = pd.to_numeric(player_log[stat_col], errors="coerce")

    def _over_rate(n: int):
        x = s.head(n).dropna()
        return float((x > dfs_line).mean()) if len(x) else None

    out["form"] = {
        "over_l5":  _over_rate(5),
        "over_l10": _over_rate(10),
        "over_l15": _over_rate(15),
    }
    if opp_abbr and "OPP_OPP_ABBREVIATION_base" in player_log.columns:
        m  = player_log["OPP_OPP_ABBREVIATION_base"] == opp_abbr
        vs = pd.to_numeric(player_log.loc[m, stat_col], errors="coerce").dropna()
        if len(vs):
            out["vs_opp"] = {
                "n_games":          int(len(vs)),
                "avg_stat":         float(vs.mean()),
                "over_rate_at_line": float((vs > dfs_line).mean()),
            }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Core enrichment — one pick
# ─────────────────────────────────────────────────────────────────────────────
def _emit_enriched_pick(
    *,
    platform: str,
    display_name: str,
    norm: str,
    market: str,
    dfs_line: float,
    dfs_odds_payload: dict,
    base: pd.DataFrame,
    alp: pd.DataFrame,
    sharp_books: dict[str, pd.DataFrame],
    team_line_idx: Any,
    team_ratings: pd.DataFrame,
    current_date: str,
    verbose: bool,
) -> dict | None:
    plog = base[base["NORM_NAME"] == norm]
    if plog.empty:
        if verbose:
            print(f"  [skip] {display_name}: not in base_df")
        return None

    player_canonical = plog["PLAYER_NAME"].iloc[0]
    player_team_abbr = plog["TEAM_ABBREVIATION"].iloc[0]
    player_team_full = TEAM_ABBR_TO_FULL.get(player_team_abbr)

    try:
        opp_abbr, is_home = find_opp(player_canonical, base, current_date)
    except Exception as exc:
        if verbose:
            print(f"  [warn] findOpp failed for {player_canonical}: {exc}")
        opp_abbr, is_home = None, None
    opp_full = TEAM_ABBR_TO_FULL.get(opp_abbr) if opp_abbr else None

    game_ctx = (
        get_team_game_context(team_line_idx, player_team_abbr, opp_abbr, is_home)
        if opp_abbr is not None and is_home is not None
        else {"book_used": None, "game_total": None, "spread": None, "team_total_implied": None}
    )

    # --- Model probs (PTS / AST / REB only) ---
    p_over = p_under = model_lean = None
    min_q10 = min_q50 = min_q90 = stat_q10 = stat_q50 = stat_q90 = None

    if market in MODEL_MARKETS:
        m_hit = alp[
            (alp["NORM_NAME"] == norm)
            & (alp["MARKET"] == market)
            & (np.isclose(alp["LINE"], dfs_line))
        ]
        if m_hit.empty:
            # DFS lines can sit 0.5–1.5 pts away from model lines — use nearest within 3
            cands = alp[(alp["NORM_NAME"] == norm) & (alp["MARKET"] == market)]
            if not cands.empty:
                nearest_idx = (cands["LINE"] - dfs_line).abs().idxmin()
                if abs(cands.loc[nearest_idx, "LINE"] - dfs_line) <= 3.0:
                    m_hit = cands.loc[[nearest_idx]]
        if not m_hit.empty:
            r = m_hit.iloc[0]
            p_over  = _to_opt_float(r["P_OVER"])
            p_under = _to_opt_float(r["P_UNDER"])
            min_q10  = _to_opt_float(r["MIN_Q10"])  if "MIN_Q10"  in m_hit.columns else None
            min_q50  = _to_opt_float(r["MIN_Q50"])  if "MIN_Q50"  in m_hit.columns else None
            min_q90  = _to_opt_float(r["MIN_Q90"])  if "MIN_Q90"  in m_hit.columns else None
            stat_q10 = _to_opt_float(r["STAT_Q10"]) if "STAT_Q10" in m_hit.columns else None
            stat_q50 = _to_opt_float(r["STAT_Q50"]) if "STAT_Q50" in m_hit.columns else None
            stat_q90 = _to_opt_float(r["STAT_Q90"]) if "STAT_Q90" in m_hit.columns else None
            if p_over is not None and p_under is not None:
                model_lean = "OVER" if p_over > p_under else "UNDER"

    # --- Sharp line + lean ---
    sharp      = get_sharp_line(sharp_books, norm, market, dfs_line,
                                dfs_odds_payload, fallback_source=platform)
    sharp_line = sharp["line"]

    if sharp_line is not None:
        if sharp_line > dfs_line:
            sharp_lean = "OVER"
        elif sharp_line < dfs_line:
            sharp_lean = "UNDER"
        elif sharp["no_vig_over"] is not None and sharp["no_vig_under"] is not None:
            sharp_lean = "OVER" if sharp["no_vig_over"] > sharp["no_vig_under"] else "UNDER"
        else:
            sharp_lean = None
    else:
        sharp_lean = None

    offerings = collect_offerings(sharp_books, norm, market)
    consensus  = compute_consensus(offerings, dfs_line)

    stat_col = MARKET_TO_BASE_COL.get(market)
    fv = (
        compute_form_and_vsopp(plog, stat_col, dfs_line, opp_abbr)
        if stat_col
        else {"form": {"over_l5": None, "over_l10": None, "over_l15": None},
              "vs_opp": {"n_games": 0, "avg_stat": None, "over_rate_at_line": None}}
    )

    team_rate = team_ratings.loc[player_team_full].to_dict() if player_team_full in team_ratings.index else {}
    opp_rate  = team_ratings.loc[opp_full].to_dict()         if opp_full  in team_ratings.index else {}

    is_conflict   = model_lean is not None and sharp_lean is not None and model_lean != sharp_lean
    sharp_aligned = model_lean is not None and sharp_lean is not None and model_lean == sharp_lean

    if model_lean is None:
        tier = "no_model"
    elif is_conflict:
        tier = "conflict"
    elif sharp_aligned and not sharp["is_fallback"]:
        tier = "sharp_verified"
    elif sharp_aligned and sharp["is_fallback"]:
        tier = "dfs_only"
    else:
        tier = "no_model"

    return {
        "platform":      platform,
        "player":        player_canonical,
        "display_name":  display_name,
        "team_abbr":     player_team_abbr,
        "team":          player_team_full,
        "opponent_abbr": opp_abbr,
        "opponent":      opp_full,
        "is_home":       is_home,
        "market":        market,
        "dfs_line":      dfs_line,

        "dfs_odds": dfs_odds_payload,

        "model": {
            "p_over":   p_over,
            "p_under":  p_under,
            "lean":     model_lean,
            "min_q10":  min_q10,
            "min_q50":  min_q50,
            "min_q90":  min_q90,
            "stat_q10": stat_q10,
            "stat_q50": stat_q50,
            "stat_q90": stat_q90,
        },

        "sharp": {**sharp, "lean": sharp_lean},

        "consensus": {**consensus, "offerings": offerings},

        "game_context": {
            **game_ctx,
            "team_pace":            _to_opt_float(team_rate.get("PACE")),
            "team_pace_rank":       _to_opt_float(team_rate.get("PACE_RANK")),
            "team_def_rating":      _to_opt_float(team_rate.get("DEF_RATING")),
            "team_def_rating_rank": _to_opt_float(team_rate.get("DEF_RATING_RANK")),
            "opp_pace":             _to_opt_float(opp_rate.get("PACE")),
            "opp_pace_rank":        _to_opt_float(opp_rate.get("PACE_RANK")),
            "opp_def_rating":       _to_opt_float(opp_rate.get("DEF_RATING")),
            "opp_def_rating_rank":  _to_opt_float(opp_rate.get("DEF_RATING_RANK")),
        },

        "form":   fv["form"],
        "vs_opp": fv["vs_opp"],

        "is_conflict":   bool(is_conflict),
        "sharp_aligned": bool(sharp_aligned),
        "tier":          tier,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def enrich_dfs_picks(
    *,
    dfs_df: pd.DataFrame,
    us_df: pd.DataFrame,
    base_df: pd.DataFrame,
    all_line_probs: pd.DataFrame,
    team_odds_source: Any = None,           # path or dict; None skips game context
    dfs_platforms: list[str] | None = None, # filter DFS books; None = all four
    current_date: str,
    output_dir: str | Path = "data/props/enriched",
    verbose: bool = True,
) -> tuple[Path, Path, pd.DataFrame]:
    """
    Enrich NBA_DFS_* picks with NBA_US_* sharp context and model probabilities.

    Parameters
    ----------
    dfs_df          : raw NBA_DFS_* CSV loaded into a DataFrame (long format)
    us_df           : raw NBA_US_* CSV loaded into a DataFrame (long format)
    base_df         : player game-log DataFrame (PLAYER_NAME, TEAM_ABBREVIATION, GAME_DATE, ...)
    all_line_probs  : model output with P_OVER / P_UNDER per player/market/line
    team_odds_source: path or dict for circa+betonline_team_lines JSON (optional)
    dfs_platforms   : list of BOOKMAKER values to include, e.g. ["PrizePicks", "Underdog"]
    current_date    : "YYYY-MM-DD" for opponent lookup and output filenames
    output_dir      : where to write enriched JSONs
    verbose         : print progress

    Returns
    -------
    (enriched_path, aligned_path, enriched_df)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dfs_wide = pivot_odds_csv(dfs_df, bookmakers=dfs_platforms)
    if dfs_wide.empty:
        raise ValueError("No DFS picks after pivoting — check dfs_df and dfs_platforms")

    us_wide     = pivot_odds_csv(us_df)
    sharp_books = build_sharp_books_from_csv(us_wide)

    if verbose:
        platforms = dfs_wide["BOOKMAKER"].unique().tolist()
        print(f"  DFS rows: {len(dfs_wide)} across {platforms}")
        print(f"  Sharp books available: {list(sharp_books.keys())}")

    team_line_idx = build_team_line_index(team_odds_source)

    if verbose:
        print("  Fetching league team ratings (NBA API)...")
    try:
        team_ratings = fetch_team_ratings()
    except Exception as exc:
        if verbose:
            print(f"  [warn] leaguedashteamstats failed: {exc}")
        team_ratings = pd.DataFrame(columns=["DEF_RATING", "DEF_RATING_RANK", "PACE", "PACE_RANK"])

    alp = all_line_probs.copy()
    alp["NORM_NAME"] = alp["PLAYER_NAME"].map(normalize_name)
    alp["LINE"]      = pd.to_numeric(alp["LINE"], errors="coerce")

    base = base_df.copy()
    base["NORM_NAME"] = base["PLAYER_NAME"].map(normalize_name)
    base["GAME_DATE"] = pd.to_datetime(base["GAME_DATE"], errors="coerce")
    base = base.sort_values("GAME_DATE", ascending=False).reset_index(drop=True)

    enriched_rows: list[dict] = []
    seen: set[tuple] = set()  # deduplicate (platform, norm, market, line)

    for _, row in dfs_wide.iterrows():
        platform = str(row["BOOKMAKER"])
        norm     = str(row["NORM_NAME"])
        market   = str(row["MARKET"])
        dfs_line = float(row["LINE"])
        key      = (platform, norm, market, dfs_line)
        if key in seen:
            continue
        seen.add(key)

        dfs_odds_payload = {
            "american_over":  _to_opt_float(row.get("AMERICAN_OVER")),
            "american_under": _to_opt_float(row.get("AMERICAN_UNDER")),
            "no_vig_over":    _to_opt_float(row.get("NO_VIG_IMPLIED_OVER")),
            "no_vig_under":   _to_opt_float(row.get("NO_VIG_IMPLIED_UNDER")),
        }
        rec = _emit_enriched_pick(
            platform=platform,
            display_name=str(row["NAME"]),
            norm=norm,
            market=market,
            dfs_line=dfs_line,
            dfs_odds_payload=dfs_odds_payload,
            base=base,
            alp=alp,
            sharp_books=sharp_books,
            team_line_idx=team_line_idx,
            team_ratings=team_ratings,
            current_date=current_date,
            verbose=verbose,
        )
        if rec is not None:
            enriched_rows.append(rec)

    generated_at = datetime.utcnow()
    today_str    = current_date.replace("-", "")
    run_ts       = generated_at.strftime("%H%M%S")
    enriched_path = output_dir / f"dfs_enriched_{today_str}_{run_ts}.json"
    aligned_path  = output_dir / f"dfs_sharp_aligned_{today_str}_{run_ts}.json"
    enriched_latest_path = output_dir / "dfs_enriched_latest.json"
    aligned_latest_path  = output_dir / "dfs_sharp_aligned_latest.json"

    meta = {
        "generated_at": generated_at.isoformat() + "Z",
        "current_date": current_date,
        "n_dfs_picks":  int(len(dfs_wide)),
        "n_enriched":   int(len(enriched_rows)),
    }
    with open(enriched_path, "w") as f:
        json.dump({**meta, "picks": enriched_rows}, f, indent=2, default=_json_default)
    with open(enriched_latest_path, "w") as f:
        json.dump({**meta, "picks": enriched_rows}, f, indent=2, default=_json_default)

    aligned = [r for r in enriched_rows if r["sharp_aligned"]]
    with open(aligned_path, "w") as f:
        json.dump(
            {**meta, "n_sharp_aligned": len(aligned), "picks": aligned},
            f, indent=2, default=_json_default,
        )
    with open(aligned_latest_path, "w") as f:
        json.dump(
            {**meta, "n_sharp_aligned": len(aligned), "picks": aligned},
            f, indent=2, default=_json_default,
        )

    if verbose:
        n_verified = sum(1 for r in enriched_rows if r["tier"] == "sharp_verified")
        n_dfs_only = sum(1 for r in enriched_rows if r["tier"] == "dfs_only")
        n_conflict = sum(1 for r in enriched_rows if r["tier"] == "conflict")
        n_nomodel  = sum(1 for r in enriched_rows if r["tier"] == "no_model")
        print(f"  Enriched: {len(enriched_rows)} | sharp_verified: {n_verified} | "
              f"dfs_only: {n_dfs_only} | conflict: {n_conflict} | no_model: {n_nomodel}")
        print(f"  → {enriched_path}")
        print(f"  → {aligned_path}")
        print(f"  → {enriched_latest_path} (latest)")
        print(f"  → {aligned_latest_path} (latest)")

    return enriched_path, aligned_path, pd.DataFrame(enriched_rows)
