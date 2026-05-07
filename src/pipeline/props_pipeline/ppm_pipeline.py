import unicodedata

import numpy as np
import pandas as pd

from src.utils.helper_functions import findOpp
from src.utils.team_info import projectedStartingFive, team3StarsPerTeam

# Order must match training / saved quantile bundle `feature_names`.
PPM_FEATURES = [
    "PTS_PER_MIN_season_avg",
    "USG_PCT_season_avg",
    "PTS_PER_MIN_lag1",
    "USG_PCT_5_ewm",
    "PTS_PER_MIN_X_OPP_DEF_RATING_L10",
    "USG_PCT_X_OPP_DEF_RATING_L10",
    "MIN_season_avg",
    "TEAM_PTS_PER_MIN_RANK_L5",
    "TS_PCT_X_USG_PCT",
    "EXPECTED_PACE",
    "OPP_DEF_RATING_roll10",
    "FTA_PER_MIN_X_OPP_FTA_ALLOWED",
    "ACTIVE_STARS_COUNT",
]


def _norm_name(name: str) -> str:
    """Loose match for NBA names (accents, punctuation) vs team_info strings."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    s = s.lower().replace(".", "").replace("-", " ").replace("'", "").strip()
    return " ".join(s.split())


def _active_stars_from_projected_lineup(team_abbrev: str) -> int:
    """
    Count team3StarsPerTeam entries that appear in projectedStartingFive for the
    same team (name-normalized). Ignores stars with no team entry.
    """
    stars = team3StarsPerTeam.get(team_abbrev) or []
    projected = projectedStartingFive.get(team_abbrev) or []
    if not stars or not projected:
        return 0
    proj_set = {_norm_name(p) for p in projected if _norm_name(p)}
    n = 0
    for s in stars:
        sn = _norm_name(s)
        if sn and sn in proj_set:
            n += 1
    return int(min(n, 3))


def _team_game_logs(df: pd.DataFrame, team_abbr: str) -> pd.DataFrame:
    g = df[df["TEAM_ABBREVIATION"] == team_abbr]
    if g.empty:
        return g
    if "GAME_ID" in g.columns:
        g = g.drop_duplicates(subset=["GAME_ID"])
    return g.sort_values("GAME_DATE")


def _opp_roll_metric(team_games: pd.DataFrame, col: str, window: int) -> float:
    """Match ppm_features._opponent_stats: shift(1).rolling(w, min_periods=1).mean().round(2)."""
    if team_games.empty or col not in team_games.columns:
        return float("nan")
    s = team_games[col].astype(float)
    return float(s.shift(1).rolling(window, min_periods=1).mean().round(2).iloc[-1])


def _shift_expanding_mean_round(series: pd.Series, ndigits: int = 2) -> float:
    if series.empty:
        return float("nan")
    v = series.astype(float).shift(1).expanding().mean().round(ndigits).iloc[-1]
    return float(v) if pd.notna(v) else float("nan")


def _shift_ewm_last(series: pd.Series, span: int) -> float:
    if series.empty:
        return float("nan")
    v = series.astype(float).shift(1).ewm(span=span, adjust=False).mean().iloc[-1]
    return float(v) if pd.notna(v) else float("nan")


def _shift_roll_last(series: pd.Series, window: int, min_periods: int = 3) -> float:
    if series.empty:
        return float("nan")
    v = series.astype(float).shift(1).rolling(window, min_periods=min_periods).mean().iloc[-1]
    return float(v) if pd.notna(v) else float("nan")


def ppm_pipeline(df, name, current_date):
    pdf = df[df["PLAYER_NAME"] == name].sort_values("GAME_DATE").copy()
    if len(pdf) < 10:
        return None

    res = []
    last = pdf.iloc[-1]
    player_team_abbr = str(last["TEAM_ABBREVIATION"])
    opp_abbr, _ = findOpp(name, df, current_date, max_days_ahead=3)

    if "PTS_PER_MIN" not in pdf.columns:
        pdf["PTS_PER_MIN"] = pdf["PTS"].astype(float) / pdf["MIN"].replace(0, np.nan)
    if "FTA_PER_MIN" not in pdf.columns and "FTA" in pdf.columns:
        pdf["FTA_PER_MIN"] = pdf["FTA"].astype(float) / pdf["MIN"].replace(0, np.nan)

    # ── PTS_PER_MIN_season_avg, USG_PCT_season_avg ────────────────────────────
    res.append(_shift_expanding_mean_round(pdf["PTS_PER_MIN"], 2))
    res.append(_shift_expanding_mean_round(pdf["USG_PCT"], 2))

    # ── PTS_PER_MIN_lag1 (next-game feature ≡ last logged game) ───────────────
    lag_ppm = pdf["PTS_PER_MIN"].astype(float).iloc[-1]
    res.append(float(lag_ppm) if pd.notna(lag_ppm) else float("nan"))

    # ── USG_PCT_5_ewm ─────────────────────────────────────────────────────────
    res.append(_shift_ewm_last(pdf["USG_PCT"], 5))

    # ── Opponent DEF_RATING roll10 ────────────────────────────────────────────
    opp_def_r10 = float("nan")
    if opp_abbr is not None:
        opp_g = _team_game_logs(df, opp_abbr)
        opp_def_r10 = _opp_roll_metric(opp_g, "TEAM_DEF_RATING", 10)

    ppm_10_ewm = _shift_ewm_last(pdf["PTS_PER_MIN"], 10)
    usg_r10 = _shift_roll_last(pdf["USG_PCT"], 10)

    res.append(ppm_10_ewm * opp_def_r10 if pd.notna(ppm_10_ewm) and pd.notna(opp_def_r10) else float("nan"))
    res.append(usg_r10 * opp_def_r10 if pd.notna(usg_r10) and pd.notna(opp_def_r10) else float("nan"))

    # ── MIN_season_avg ────────────────────────────────────────────────────────
    res.append(_shift_expanding_mean_round(pdf["MIN"], 2))

    # ── TEAM_PTS_PER_MIN_RANK_L5 (dense rank, high PPM roll5 = rank 1) ────────
    team_df = df[df["TEAM_ID"] == last["TEAM_ID"]]
    teammate_roll5 = {}
    for player, grp in team_df.groupby("PLAYER_NAME"):
        grp_sorted = grp.sort_values("GAME_DATE")
        if "PTS_PER_MIN" not in grp_sorted.columns:
            grp_sorted = grp_sorted.copy()
            grp_sorted["PTS_PER_MIN"] = grp_sorted["PTS"].astype(float) / grp_sorted["MIN"].replace(0, np.nan)
        r5 = _shift_roll_last(grp_sorted["PTS_PER_MIN"].astype(float), 5)
        teammate_roll5[player] = r5
    sorted_teammates = sorted(teammate_roll5.items(), key=lambda x: (x[1] if pd.notna(x[1]) else -np.inf), reverse=True)
    rank_map = {p: float(i + 1) for i, (p, _) in enumerate(sorted_teammates)}
    team_pts_rank = rank_map.get(name, float("nan"))
    res.append(team_pts_rank if pd.notna(team_pts_rank) else float("nan"))

    # ── TS_PCT_X_USG_PCT (season_avg × season_avg) ────────────────────────────
    ts_sa = _shift_expanding_mean_round(pdf["TS_PCT"], 2)
    usg_sa = _shift_expanding_mean_round(pdf["USG_PCT"], 2)
    res.append(ts_sa * usg_sa if pd.notna(ts_sa) and pd.notna(usg_sa) else float("nan"))

    # ── EXPECTED_PACE ─────────────────────────────────────────────────────────
    team_g = _team_game_logs(df, player_team_abbr)
    team_pace_r10 = _opp_roll_metric(team_g, "TEAM_PACE", 10)
    opp_pace_r10 = float("nan")
    if opp_abbr is not None:
        opp_pace_r10 = _opp_roll_metric(_team_game_logs(df, opp_abbr), "TEAM_PACE", 10)
    if pd.notna(team_pace_r10) and pd.notna(opp_pace_r10):
        res.append(round((team_pace_r10 + opp_pace_r10) / 2, 2))
    else:
        res.append(float("nan"))

    res.append(opp_def_r10)

    # ── FTA_PER_MIN_X_OPP_FTA_ALLOWED (training: FTA_PER_MIN_10_ewm * OPP_TEAM_FTA_ALLOWED) ─
    fta_10_ewm = _shift_ewm_last(pdf["FTA_PER_MIN"], 10)
    opp_fta_allowed = float("nan")
    if opp_abbr is not None:
        og = _team_game_logs(df, opp_abbr)
        if not og.empty and "OPP_FTA" in og.columns:
            opp_fta_allowed = float(
                og["OPP_FTA"].astype(float).shift(1).expanding().mean().round(3).iloc[-1]
            )
            if pd.isna(opp_fta_allowed):
                opp_fta_allowed = float("nan")
    res.append(
        fta_10_ewm * opp_fta_allowed
        if pd.notna(fta_10_ewm) and pd.notna(opp_fta_allowed)
        else float("nan")
    )

    # ── ACTIVE_STARS_COUNT: stars listed in team_info also in projected starters ─
    res.append(float(_active_stars_from_projected_lineup(player_team_abbr)))

    return res
