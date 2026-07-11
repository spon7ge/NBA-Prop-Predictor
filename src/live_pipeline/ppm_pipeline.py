import numpy as np
import pandas as pd

from src.features.ppm_features import PPM_FEATURES
from src.utils.helper_functions import findOpp

# ── Expected df columns for opponent lookups ───────────────────────────────────
# OPP_PTS        : pts allowed per game (on opponent team rows)
# OPP_FG_PCT     : FG% allowed
# OPP_DEF_RATING : defensive rating
# OPP_FG3A       : 3PA allowed per game
# OPP_FTA        : FTA allowed per game
# ──────────────────────────────────────────────────────────────────────────────


# ── Helpers ───────────────────────────────────────────────────────────────────

def _nanfloat(x) -> float:
    return float(x) if pd.notna(x) else float("nan")


def _season_expanding_tail(pdf: pd.DataFrame, col: str) -> float:
    """Expanding season mean of col, respects SEASON_YEAR grouping when available."""
    if col not in pdf.columns:
        return float("nan")
    if "SEASON_YEAR" in pdf.columns and "PLAYER_ID" in pdf.columns:
        v = pdf.groupby(["PLAYER_ID", "SEASON_YEAR"], group_keys=False)[col].transform(
            lambda x: x.astype(float).expanding().mean().round(2)
        )
    else:
        v = pdf[col].astype(float).expanding().mean().round(2)
    return _nanfloat(v.iloc[-1])


def _expanding_std(pdf: pd.DataFrame, col: str) -> float:
    """Expanding std of col, respects PLAYER_ID grouping when available."""
    if col not in pdf.columns:
        return float("nan")
    if "PLAYER_ID" in pdf.columns:
        v = pdf.groupby("PLAYER_ID", group_keys=False)[col].transform(
            lambda x: x.astype(float).expanding().std()
        )
    else:
        v = pdf[col].astype(float).expanding().std()
    return _nanfloat(v.iloc[-1])


def _opp_team_prior_metric(
    df: pd.DataFrame,
    opp_abbr: str | None,
    raw_col: str,
    *,
    as_of_date: str | None,
    decimals: int = 3,
) -> float:
    """
    Expanding mean of raw_col on opponent team rows, filtered to games
    strictly before as_of_date. Respects SEASON_YEAR when present.
    """
    if opp_abbr is None or raw_col not in df.columns:
        return float("nan")
    opp = df[df["TEAM_ABBREVIATION"] == opp_abbr].drop_duplicates(
        subset=["TEAM_ID", "GAME_ID"]
    )
    if opp.empty:
        return float("nan")
    if as_of_date is not None:
        cutoff = pd.to_datetime(as_of_date)
        dated = opp[pd.to_datetime(opp["GAME_DATE"], errors="coerce") < cutoff]
        opp = dated if len(dated) > 0 else opp
    opp = opp.sort_values("GAME_DATE").copy()
    if "SEASON_YEAR" not in opp.columns:
        vals = opp[raw_col].astype(float).expanding().mean().round(decimals)
        return _nanfloat(vals.iloc[-1])
    last_season = opp.iloc[-1]["SEASON_YEAR"]
    season = opp[opp["SEASON_YEAR"] == last_season]
    vals = season.groupby(["TEAM_ID", "SEASON_YEAR"], sort=False)[raw_col].transform(
        lambda x: x.astype(float).expanding().mean().round(decimals)
    )
    return _nanfloat(season.assign(_m=vals)["_m"].iloc[-1])


def _ewm10(pdf: pd.DataFrame, col: str) -> float:
    """EWM span=10 of col, unshifted (includes latest game)."""
    if col not in pdf.columns:
        return float("nan")
    v = pdf[col].astype(float).ewm(span=10, adjust=False).mean().iloc[-1]
    return _nanfloat(v)


def _roll_last(series: pd.Series, window: int, min_periods: int = 3) -> float:
    if series.empty:
        return float("nan")
    v = series.astype(float).rolling(window, min_periods=min_periods).mean().iloc[-1]
    return _nanfloat(v)


def _interaction(player_val: float, opp_val: float) -> float:
    """Multiply two scalars; return nan if either is nan."""
    if pd.isna(player_val) or pd.isna(opp_val):
        return float("nan")
    return float(player_val * opp_val)


def _team_stat_rank_l10(
    df: pd.DataFrame,
    player_name: str,
    team_id,
    stat_col: str,
) -> float:
    """
    Rank the player among teammates by L10 rolling avg of stat_col.
    Rank 1 = highest value on team.
    """
    team_df = df[df["TEAM_ID"] == team_id]
    if team_df.empty or stat_col not in team_df.columns:
        return float("nan")
    l10_avg = (
        team_df.sort_values("GAME_DATE")
        .groupby("PLAYER_NAME")[stat_col]
        .apply(lambda s: s.astype(float).tail(10).mean())
    )
    if player_name not in l10_avg.index:
        return float("nan")
    return _nanfloat(l10_avg.rank(ascending=False, method="min")[player_name])


# ── Main pipeline ─────────────────────────────────────────────────────────────

def ppm_pipeline(df: pd.DataFrame, name: str, date: str):
    """
    Parameters
    ----------
    df   : full player + team game-log dataframe
    name : PLAYER_NAME to look up
    date : prediction date string 'YYYY-MM-DD' (passed to findOpp & date filters)
    """
    pdf = df[df["PLAYER_NAME"] == name].sort_values("GAME_DATE").copy()
    if len(pdf) < 10:
        return None

    last = pdf.iloc[-1]

    opp_abbr, _ = findOpp(name, pdf, date, max_days_ahead=3)

    # ── Derived per-minute columns ────────────────────────────────────────────
    min_s = pdf["MIN"].replace(0, np.nan).astype(float)

    if "PTS_PER_MIN" not in pdf.columns:
        pdf["PTS_PER_MIN"] = pdf["PTS"].astype(float) / min_s
    if "FGA_PER_MIN" not in pdf.columns:
        pdf["FGA_PER_MIN"] = pdf["FGA"].astype(float) / min_s
    if "FTA_PER_MIN" not in pdf.columns:
        pdf["FTA_PER_MIN"] = pdf["FTA"].astype(float) / min_s
    if "CFGA_PER_MIN" not in pdf.columns:
        pdf["CFGA_PER_MIN"] = pdf["CFGA"].astype(float) / min_s
    if "3PA_PER_MIN" not in pdf.columns:
        pdf["3PA_PER_MIN"] = pdf["FG3A"].astype(float) / min_s

    # ── Player season average (expanding) ─────────────────────────────────────
    ppm_season_avg = _season_expanding_tail(pdf, "PTS_PER_MIN")

    # ── Player EWM(span=10) — used as player side of interaction terms ─────────
    ppm_ewm10   = _ewm10(pdf, "PTS_PER_MIN")
    fga_ewm10   = _ewm10(pdf, "FGA_PER_MIN")
    cfga_ewm10  = _ewm10(pdf, "CFGA_PER_MIN")
    fta_ewm10   = _ewm10(pdf, "FTA_PER_MIN")
    fg3a_ewm10  = _ewm10(pdf, "3PA_PER_MIN")

    # ── Opponent defensive stats (prior expanding mean from df) ───────────────
    opp_pts_allowed  = _opp_team_prior_metric(df, opp_abbr, "OPP_PTS",        as_of_date=date)
    opp_fg_pct       = _opp_team_prior_metric(df, opp_abbr, "OPP_FG_PCT",     as_of_date=date)
    opp_def_rating   = _opp_team_prior_metric(df, opp_abbr, "OPP_DEF_RATING", as_of_date=date)
    opp_fg3a_allowed = _opp_team_prior_metric(df, opp_abbr, "OPP_FG3A",       as_of_date=date)
    opp_fta_allowed  = _opp_team_prior_metric(df, opp_abbr, "OPP_FTA",        as_of_date=date)

    # ── PPM season std ────────────────────────────────────────────────────────
    ppm_season_std = _expanding_std(pdf, "PTS_PER_MIN")

    # ── Team ranking: PTS_PER_MIN L10 ────────────────────────────────────────
    team_df = df[df["TEAM_ID"] == last["TEAM_ID"]]
    teammate_roll10 = {}
    for player, grp in team_df.groupby("PLAYER_NAME"):
        grp_sorted = grp.sort_values("GAME_DATE").copy()
        if "PTS_PER_MIN" not in grp_sorted.columns:
            grp_sorted["PTS_PER_MIN"] = (
                grp_sorted["PTS"].astype(float)
                / grp_sorted["MIN"].replace(0, np.nan)
            )
        teammate_roll10[player] = _roll_last(grp_sorted["PTS_PER_MIN"], 10)
    sorted_teammates = sorted(
        teammate_roll10.items(),
        key=lambda x: (x[1] if pd.notna(x[1]) else -np.inf),
        reverse=True,
    )
    ppm_rank = {p: float(i + 1) for i, (p, _) in enumerate(sorted_teammates)}.get(
        name, float("nan")
    )

    # ── Team ranking: USG & MIN L10 ───────────────────────────────────────────
    usg_col       = "USG_PCT" if "USG_PCT" in df.columns else "USG"
    team_usg_rank = _team_stat_rank_l10(df, name, last["TEAM_ID"], usg_col)
    team_min_rank = _team_stat_rank_l10(df, name, last["TEAM_ID"], "MIN")

    # ── PPM percentiles over last 10 games ────────────────────────────────────
    ppm_l10 = pdf["PTS_PER_MIN"].tail(10).astype(float)
    ppm_p10 = float(ppm_l10.quantile(0.10))
    ppm_p90 = float(ppm_l10.quantile(0.90))

    # ── Assemble result in feature order ──────────────────────────────────────
    return [
        ppm_season_avg,                                  # PTS_PER_MIN_season_avg
        _interaction(ppm_ewm10,  opp_pts_allowed),       # PTS_PER_MIN_X_OPP_PTS_ALLOWED
        _interaction(cfga_ewm10, opp_fg_pct),           # CFGA_PER_MIN_X_OPP_FG_PCT_ALLOWED
        _interaction(fga_ewm10,  opp_def_rating),         # FGA_PER_MIN_X_OPP_DEF_RATING
        _interaction(fg3a_ewm10, opp_fg3a_allowed),       # 3PA_PER_MIN_X_OPP_TEAM_FG3A_ALLOWED
        _interaction(fta_ewm10,  opp_fta_allowed),       # FTA_PER_MIN_X_OPP_FTA_ALLOWED
        ppm_season_std,                                  # PPM_SEASON_STD
        ppm_rank,                                        # TEAM_PTS_PER_MIN_RANK_L10
        team_usg_rank,                                   # TEAM_USG_RANK_L10
        team_min_rank,                                   # TEAM_MIN_RANK_L10
        ppm_p10,                                         # PPM_P10_L10
        ppm_p90,                                         # PPM_P90_L10
    ]