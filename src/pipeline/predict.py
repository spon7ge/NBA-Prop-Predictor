"""
predict.py — live prediction entry point.

Public API
----------
load_latest_odds(league, region, prop)
    Pulls the most recent prop lines from raw Supabase tables.

predict_rate(names, current_date, prop, *, league, min_bundle, rate_bundle)
    Loads the most recent silver season, selects the correct rate pipeline
    from ``prop``, and returns quantile predictions:
        PLAYER_NAME, MARKET,
        MIN_Q10, MIN_Q50, MIN_Q90, MIN_HISTORY,
        RATE_Q10, RATE_Q50, RATE_Q90, RATE_HISTORY

line_probs_for_market(preds_df, lines_df, sim_fn, n_sims=10_000)
    Scores each prediction against the book line:
        PLAYER_NAME, MARKET, LINE,
        MIN_Q10, MIN_Q50, MIN_Q90,
        STAT_Q10, STAT_Q50, STAT_Q90,
        P_OVER, P_UNDER
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.live_pipeline.apm_pipeline import apm_pipeline
from src.live_pipeline.min_pipeline import min_pipeline
from src.live_pipeline.ppm_pipeline import ppm_pipeline
from src.live_pipeline.rpm_pipeline import rpm_pipeline
from src.utils.context import coerce_nonneg_monotone_quantiles
from src.utils.db import read_df
from src.utils.helper_functions import findOpp


def load_opp_def_ratings(season_type: str = "Regular Season") -> tuple[dict, float, float]:
    """Fetch per-team DEF_RATING and PACE from nba_api for the current season_type.

    Returns
    -------
    ratings           : dict[TEAM_ID (int) -> {"DEF_RATING": float, "PACE": float}]
    league_avg_def_rtg: float
    league_avg_pace   : float
    """
    from nba_api.stats.endpoints import leaguedashteamstats
    import pandas as _pd

    df = leaguedashteamstats.LeagueDashTeamStats(
        league_id_nullable="00",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced",
        season_type_all_star=season_type,
    ).get_data_frames()[0]

    ratings = {
        int(row["TEAM_ID"]): {
            "DEF_RATING": float(row["DEF_RATING"]),
            "PACE":       float(row["PACE"]),
        }
        for _, row in df.iterrows()
    }
    league_avg_def_rtg = float(df["DEF_RATING"].mean())
    league_avg_pace    = float(df["PACE"].mean())
    print(
        f"  Loaded def ratings for {len(ratings)} teams"
        f"  |  avg DEF_RTG={league_avg_def_rtg:.1f}"
        f"  |  avg PACE={league_avg_pace:.1f}"
    )
    return ratings, league_avg_def_rtg, league_avg_pace


# NBA team abbreviation -> numeric team ID (matches nba_api TEAM_ID).
# Inlined here to avoid importing src.utils.helpers which pulls heavy / broken deps.
_TEAM_ABV_TO_ID: dict[str, int] = {
    "ATL": 1610612737, "BKN": 1610612751, "BOS": 1610612738, "CHA": 1610612766,
    "CHI": 1610612741, "CLE": 1610612739, "DAL": 1610612742, "DEN": 1610612743,
    "DET": 1610612765, "GSW": 1610612744, "HOU": 1610612745, "IND": 1610612754,
    "LAC": 1610612746, "LAL": 1610612747, "MEM": 1610612763, "MIA": 1610612748,
    "MIL": 1610612749, "MIN": 1610612750, "NOP": 1610612740, "NYK": 1610612752,
    "OKC": 1610612760, "ORL": 1610612753, "PHI": 1610612755, "PHX": 1610612756,
    "POR": 1610612757, "SAC": 1610612758, "SAS": 1610612759, "TOR": 1610612761,
    "UTA": 1610612762, "WAS": 1610612764,
}

# ── quantile key aliases ──────────────────────────────────────────────────────
_Q10, _Q50, _Q90 = "q_0.10", "q_0.50", "q_0.90"

# ── odds-API prop category → pipeline / market label / rate column ────────────
_PROP_PIPELINE: dict[str, object] = {
    "player_points":   ppm_pipeline,
    "player_assists":  apm_pipeline,
    "player_rebounds": rpm_pipeline,
}
_PROP_MARKET: dict[str, str] = {
    "player_points":   "PTS",
    "player_assists":  "AST",
    "player_rebounds": "REB",
}
_PROP_RATE_COL: dict[str, str] = {
    "player_points":   "pts_per_min",
    "player_assists":  "ast_per_min",
    "player_rebounds": "reb_per_min",
}

_SILVER_TABLE: dict[str, str] = {
    "nba":  "nba_player_gamelogs",
    "wnba": "wnba_player_gamelogs",
}

# DB snake_case → CSV-style UPPER_CASE output columns
_ODDS_COL_RENAME: dict[str, str] = {
    "bookmaker":     "BOOKMAKER",
    "category":      "CATEGORY",
    "name":          "NAME",
    "over_under":    "OVER_UNDER",
    "line":          "LINE",
    "odds":          "ODDS",
    "commence_time": "COMMENCE_TIME",
    "last_update":   "LAST_UPDATE",
    "data_pulled_at":"DATA_PULLED_AT",
}


# ── public API ────────────────────────────────────────────────────────────────

def load_latest_odds(
    *,
    league: str = "nba",
    region: str = "dfs",
    prop: str | None = None,
) -> pd.DataFrame:
    """Load the most recently pulled prop lines from Supabase.

    Parameters
    ----------
    league:
        ``'nba'`` or ``'wnba'``.
    region:
        ``'dfs'`` — DFS platforms (PrizePicks, Underdog, …).
        ``'us'``  — US sportsbooks with real odds (DraftKings, FanDuel, …).
        ``'both'``— concatenation of dfs + us tables.
    prop:
        Optional odds-API category filter, e.g. ``'player_points'``.
        When omitted all categories are returned.

    Returns
    -------
    DataFrame with UPPER_CASE columns matching the PropFinder CSV format::

        BOOKMAKER, CATEGORY, NAME, OVER_UNDER, LINE, ODDS,
        COMMENCE_TIME, LAST_UPDATE, DATA_PULLED_AT
    """
    if league not in ("nba", "wnba"):
        raise ValueError(f"Unknown league {league!r}; expected 'nba' or 'wnba'")
    if region not in ("dfs", "us", "both"):
        raise ValueError(f"Unknown region {region!r}; expected 'dfs', 'us', or 'both'")

    regions = ("dfs", "us") if region == "both" else (region,)
    frames: list[pd.DataFrame] = []

    for rgn in regions:
        table = f"{league}_props_{rgn}"
        # subquery keeps only the latest pull without loading the full history
        where = f"data_pulled_at = (SELECT MAX(data_pulled_at) FROM raw.{table})"
        df = read_df(table, schema="raw", where=where)
        if df.empty:
            print(f"  ⚠ raw.{table}: no rows found")
            continue
        df = df.rename(columns={k: v for k, v in _ODDS_COL_RENAME.items() if k in df.columns})
        print(
            f"  raw.{table}: {len(df):,} rows"
            f"  (pulled {df['DATA_PULLED_AT'].max() if 'DATA_PULLED_AT' in df.columns else '?'})"
        )
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=list(_ODDS_COL_RENAME.values()))

    out = pd.concat(frames, ignore_index=True)

    if prop is not None:
        if "CATEGORY" in out.columns:
            out = out[out["CATEGORY"] == prop].reset_index(drop=True)

    return out


# ── internal helpers ──────────────────────────────────────────────────────────

def _infer_season(league: str) -> str:
    """Return the current season string for the given league.

    NBA  : ``'2025-26'`` format (Oct–Sep straddles two calendar years).
    WNBA : ``'2026'`` format (single calendar year, May–Oct).
    """
    today = date.today()
    if league == "wnba":
        return str(today.year)
    # NBA: season starts in October → if before October use prior start year
    start_year = today.year if today.month >= 10 else today.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _load_silver(league: str) -> pd.DataFrame:
    """Fetch the most recent season from ``silver.*_player_gamelogs``."""
    if league not in _SILVER_TABLE:
        raise ValueError(f"Unknown league {league!r}; expected 'nba' or 'wnba'")

    season = _infer_season(league)
    table  = _SILVER_TABLE[league]

    df = read_df(
        table,
        schema="silver",
        where="season_year = %(season)s",
        params={"season": season},
    )
    if df.empty:
        raise ValueError(
            f"No rows found in silver.{table} for season_year='{season}'"
        )
    print(f"  silver.{table}: {len(df):,} rows (season={season})")
    return df


# ── public API ────────────────────────────────────────────────────────────────

def predict_rate(
    names: list[str],
    current_date: str,
    prop: str,
    *,
    league: str,
    min_bundle: dict,
    rate_bundle: dict,
    def_ratings: dict | None = None,
    league_avg_def_rtg: float | None = None,
    league_avg_pace: float | None = None,
    verbose: bool = True,
    n_games: int = 15,
) -> pd.DataFrame:
    """Predict minute and rate quantiles for each player.

    Parameters
    ----------
    names:
        Player names to predict (matched against ``player_name``).
    current_date:
        Prediction date string ``'YYYY-MM-DD'``.
    prop:
        Odds-API category string: ``'player_points'``, ``'player_assists'``,
        or ``'player_rebounds'``.
    league:
        ``'nba'`` or ``'wnba'``.
    min_bundle:
        Joblib bundle for the minutes model
        (keys: ``quantile_models``, ``feature_names``).
    rate_bundle:
        Joblib bundle for the rate model matching *prop*.
    def_ratings:
        Optional dict keyed by TEAM_ID → ``{"DEF_RATING": float, "PACE": float}``.
        Load once via ``load_opp_def_ratings()`` and pass in.  When omitted
        ``OPP_DEF_RATING`` and ``OPP_PACE`` are returned as NaN.
    league_avg_def_rtg:
        League-average defensive rating (from ``load_opp_def_ratings()``).
    league_avg_pace:
        League-average pace (from ``load_opp_def_ratings()``).
    n_games:
        Games of history to include in ``*_HISTORY`` lists.

    Returns
    -------
    DataFrame with columns::

        PLAYER_NAME, PLAYER_TEAM, OPP_TEAM, HOME, MARKET,
        MIN_Q10, MIN_Q50, MIN_Q90, MIN_HISTORY,
        RATE_Q10, RATE_Q50, RATE_Q90, RATE_HISTORY,
        OPP_DEF_RATING, OPP_PACE,
        LEAGUE_AVG_DEF_RATING, LEAGUE_AVG_PACE
    """
    if prop not in _PROP_PIPELINE:
        raise ValueError(
            f"Unknown prop {prop!r}; expected one of {sorted(_PROP_PIPELINE)}"
        )

    rate_pipeline = _PROP_PIPELINE[prop]
    market        = _PROP_MARKET[prop]
    rate_col      = _PROP_RATE_COL[prop]

    min_models  = min_bundle["quantile_models"]
    rate_models = rate_bundle["quantile_models"]

    df = _load_silver(league)

    records: list[dict] = []
    for name in names:
        try:
            min_feats  = min_pipeline(df, name, current_date)
            if min_feats is None:
                raise ValueError("min_pipeline: need ≥ 10 games")

            rate_feats = rate_pipeline(df, name, current_date)
            if rate_feats is None:
                raise ValueError("rate_pipeline: need ≥ 10 games")

            min_arr  = np.asarray(min_feats,  dtype=float).reshape(1, -1)
            rate_arr = np.asarray(rate_feats, dtype=float).reshape(1, -1)

            m10 = float(min_models[_Q10].predict(min_arr)[0])
            m50 = float(min_models[_Q50].predict(min_arr)[0])
            m90 = float(min_models[_Q90].predict(min_arr)[0])

            r10 = float(rate_models[_Q10].predict(rate_arr)[0])
            r50 = float(rate_models[_Q50].predict(rate_arr)[0])
            r90 = float(rate_models[_Q90].predict(rate_arr)[0])

            m10, m50, m90 = coerce_nonneg_monotone_quantiles(m10, m50, m90)
            r10, r50, r90 = coerce_nonneg_monotone_quantiles(r10, r50, r90)

            pdf = df[df["player_name"] == name].sort_values("game_date")
            if rate_col not in pdf.columns:
                raise KeyError(f"'{rate_col}' missing from silver data")

            min_history  = pdf["min"].dropna().tail(n_games).tolist()
            rate_history = pdf[rate_col].dropna().tail(n_games).tolist()

            # ── opponent / home context ────────────────────────────────────────
            opp_abv, home = findOpp(name, df, current_date)
            player_team = (
                pdf["team_abbreviation"].iloc[-1] if not pdf.empty else np.nan
            )

            opp_team_id = _TEAM_ABV_TO_ID.get(opp_abv) if opp_abv else None
            opp_info    = (
                def_ratings.get(opp_team_id)
                if def_ratings and opp_team_id
                else None
            )

        except Exception as exc:
            if verbose:
                print(f"[SKIP] {name}: {exc}")
            continue

        records.append({
            "PLAYER_NAME":           name,
            "PLAYER_TEAM":           player_team,
            "OPP_TEAM":              opp_abv,
            "HOME":                  home,
            "MARKET":                market,
            "MIN_Q10":               round(m10, 2),
            "MIN_Q50":               round(m50, 2),
            "MIN_Q90":               round(m90, 2),
            "MIN_HISTORY":           min_history,
            "RATE_Q10":              round(r10, 4),
            "RATE_Q50":              round(r50, 4),
            "RATE_Q90":              round(r90, 4),
            "RATE_HISTORY":          rate_history,
            "OPP_DEF_RATING":        opp_info["DEF_RATING"] if opp_info else np.nan,
            "OPP_PACE":              opp_info["PACE"]       if opp_info else np.nan,
            "LEAGUE_AVG_DEF_RATING": league_avg_def_rtg if league_avg_def_rtg is not None else np.nan,
            "LEAGUE_AVG_PACE":       league_avg_pace    if league_avg_pace    is not None else np.nan,
        })

    return pd.DataFrame(records)


def line_probs_for_market(
    preds_df: pd.DataFrame,
    lines_df: pd.DataFrame,
    sim_fn,
    n_sims: int = 10_000,
) -> pd.DataFrame:
    """Score each player's quantile prediction against the book line.

    Parameters
    ----------
    preds_df:
        Output of ``predict_rate``.
    lines_df:
        Prop-line CSV rows from PropFinder. Must contain ``NAME`` and ``LINE``.
    sim_fn:
        Simulation callable, e.g. ``run_pts_simulation`` from
        ``src.utils.distributions``. Signature: ``sim_fn(row, n_sims) → ndarray``.
    n_sims:
        Monte-Carlo draws per player.

    Returns
    -------
    DataFrame with columns::

        PLAYER_NAME, MARKET, LINE,
        MIN_Q10, MIN_Q50, MIN_Q90,
        STAT_Q10, STAT_Q50, STAT_Q90,
        P_OVER, P_UNDER
    """
    line_by_name: dict[str, float] = {
        str(r["NAME"]).strip(): float(r["LINE"])
        for _, r in lines_df.iterrows()
        if pd.notna(r.get("LINE"))
    }

    rows: list[dict] = []
    for _, row in preds_df.iterrows():
        name   = row["PLAYER_NAME"]
        market = row["MARKET"]
        line   = line_by_name.get(name)

        m10 = float(row.get("MIN_Q10",  np.nan))
        m50 = float(row.get("MIN_Q50",  np.nan))
        m90 = float(row.get("MIN_Q90",  np.nan))
        r10 = float(row.get("RATE_Q10", np.nan))
        r50 = float(row.get("RATE_Q50", np.nan))
        r90 = float(row.get("RATE_Q90", np.nan))

        s10, s50, s90 = coerce_nonneg_monotone_quantiles(
            m10 * r10, m50 * r50, m90 * r90
        )

        if line is None or not np.isfinite(m50) or not np.isfinite(r50):
            rows.append({
                "PLAYER_NAME": name, "MARKET": market, "LINE": np.nan,
                "MIN_Q10": m10, "MIN_Q50": m50, "MIN_Q90": m90,
                "STAT_Q10": s10, "STAT_Q50": s50, "STAT_Q90": s90,
                "P_OVER": np.nan, "P_UNDER": np.nan,
            })
            continue

        sim_row = {**row.to_dict(), "STAT_Q10": s10, "STAT_Q50": s50, "STAT_Q90": s90}
        sims    = sim_fn(sim_row, n_sims=n_sims)
        line_f  = float(line)

        rows.append({
            "PLAYER_NAME": name,
            "MARKET":      market,
            "LINE":        line_f,
            "MIN_Q10":     round(m10, 2),
            "MIN_Q50":     round(m50, 2),
            "MIN_Q90":     round(m90, 2),
            "STAT_Q10":    round(s10, 2),
            "STAT_Q50":    round(s50, 2),
            "STAT_Q90":    round(s90, 2),
            "P_OVER":      round(float(np.mean(sims > line_f)), 3),
            "P_UNDER":     round(float(np.mean(sims < line_f)), 3),
        })

    return pd.DataFrame(rows)
