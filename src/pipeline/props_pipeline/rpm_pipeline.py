import numpy as np
import pandas as pd

from src.utils.helper_functions import findOpp


def _nanfloat(x):
    return float(x) if pd.notna(x) else float("nan")


def _season_expanding_tail(
    pdf: pd.DataFrame, col: str, seasons: bool
) -> float:
    s = pdf[col].astype(float)
    ok_season = (
        seasons
        and "SEASON_YEAR" in pdf.columns
        and "PLAYER_ID" in pdf.columns
    )
    if ok_season:
        v = pdf.groupby(["PLAYER_ID", "SEASON_YEAR"], group_keys=False)[col].transform(
            lambda x: x.astype(float).expanding().mean().round(2)
        )
    else:
        v = s.expanding().mean().round(2)
    return _nanfloat(v.iloc[-1])


def _opp_team_prior_metric(
    df: pd.DataFrame,
    opp_abbr: str,
    raw_col: str,
    *,
    as_of_date: str | None,
    decimals: int = 3,
) -> float:
    """RPM _team_allowed_context: prior expanding mean of raw_col on opp team-season rows."""
    if raw_col not in df.columns:
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
    opp = opp.sort_values(["GAME_DATE"]).copy()
    round_fn = decimals is not None
    if "SEASON_YEAR" not in opp.columns:
        ser = opp[raw_col].astype(float)
        vals = (
            ser.expanding().mean().round(decimals)
            if round_fn
            else ser.expanding().mean()
        )
        return _nanfloat(vals.iloc[-1])

    last = opp.iloc[-1]
    mask = opp["SEASON_YEAR"] == last["SEASON_YEAR"]
    season = opp.loc[mask]
    vals = (
        season.groupby(["TEAM_ID", "SEASON_YEAR"], sort=False)[raw_col]
        .transform(lambda x: x.astype(float).expanding().mean().round(decimals))
    )
    return _nanfloat(season.assign(_m=vals)["_m"].iloc[-1])


def _reb_roll10_slope_last(pdf: pd.DataFrame) -> float:
    r10 = (
        pdf["REB_PER_MIN"]
        .astype(float)
        .rolling(10)
        .mean()
        .round(2)
    )

    def _slope(win: np.ndarray) -> float:
        y = win.astype(np.float64)
        if y.size != 5 or not np.all(np.isfinite(y)):
            return np.nan
        return float(np.polyfit(np.arange(len(y)), y, 1)[0])

    out = (
        r10.rolling(5)
        .apply(lambda a: _slope(a), raw=True)
        .iloc[-1]
    )
    return float(out) if pd.notna(out) else float("nan")


def _team_pace_shift_roll_tail(
    df: pd.DataFrame,
    abbrev: str,
    window: int,
    *,
    as_of_date: str | None,
    min_periods: int | None = None,
) -> float:
    """shift(1).rolling(w) on TEAM_PACE (rpm _team_context + _opponent_stats windows)."""
    if "TEAM_PACE" not in df.columns:
        return float("nan")
    t = df[df["TEAM_ABBREVIATION"] == abbrev].drop_duplicates(
        subset=["TEAM_ID", "GAME_ID"]
    )
    if t.empty:
        return float("nan")
    if as_of_date is not None:
        cutoff = pd.to_datetime(as_of_date)
        dated = t[pd.to_datetime(t["GAME_DATE"], errors="coerce") < cutoff]
        if len(dated) > 0:
            t = dated
    t = t.sort_values("GAME_DATE")
    mp = window if min_periods is None else min_periods
    roll = (
        t["TEAM_PACE"]
        .astype(float)
        .rolling(window, min_periods=mp)
        .mean()
        .round(2)
    )
    return _nanfloat(roll.iloc[-1])


def _reb_per_min_roll5_last(pdf: pd.DataFrame) -> float:
    reb5 = pdf["REB"].astype(float).rolling(5).mean().round(2)
    mn5 = pdf["MIN"].astype(float).rolling(5).mean().round(2)
    v = (reb5.iloc[-1] / mn5.iloc[-1]) if mn5.iloc[-1] != 0 and pd.notna(mn5.iloc[-1]) else np.nan
    return float(v) if pd.notna(v) else float("nan")


def _opp_fga_roll10_last(df: pd.DataFrame, opp_abbr: str, *, as_of_date: str | None) -> float:
    """_opponent_stats OPP_FGA_roll10 analogue for TEAM_FGA rolling 10 on opponent abbreviation."""
    if "TEAM_FGA" not in df.columns:
        return float("nan")
    t = df[df["TEAM_ABBREVIATION"] == opp_abbr].drop_duplicates(
        subset=["TEAM_ID", "GAME_ID"]
    )
    if t.empty:
        return float("nan")
    if as_of_date is not None:
        cutoff = pd.to_datetime(as_of_date)
        dated = t[pd.to_datetime(t["GAME_DATE"], errors="coerce") < cutoff]
        if len(dated) > 0:
            t = dated
    t = t.sort_values("GAME_DATE")
    roll = (
        t.groupby("TEAM_ID", sort=False)["TEAM_FGA"]
        .transform(
            lambda x: x.astype(float)
            .rolling(10, min_periods=1)
            .mean()
            .round(2)
        )
    )
    val = pd.Series(roll.astype(float)).iloc[-1]
    return _nanfloat(val)


def rpm_pipeline(df, name, date):
    pdf = df[df["PLAYER_NAME"] == name].sort_values("GAME_DATE").copy()
    if len(pdf) < 10:
        return None

    res = []
    last = pdf.iloc[-1]
    seasons = "SEASON_YEAR" in pdf.columns

    opp_abbr, _ = findOpp(name, pdf, date, max_days_ahead=3)
    player_team = last["TEAM_ABBREVIATION"]

    # 1 — REB_PER_MIN_season_avg
    res.append(_season_expanding_tail(pdf, "REB_PER_MIN", seasons))

    # 2 — REB_PER_MIN_10_ewm
    reb_per_min = pdf['REB_PER_MIN'].astype(float).ewm(span=10, adjust=False).mean().iloc[-1]
    res.append(float(reb_per_min) if pd.notna(reb_per_min) else float("nan"))

    # 2b — RBC_PER_MIN_10_ewm
    rbc_per_min = pdf['RBC'] / pdf['MIN'].replace(0, np.nan)
    rbc_10_ewm = rbc_per_min.astype(float).ewm(span=10, adjust=False).mean().iloc[-1]
    res.append(float(rbc_10_ewm) if pd.notna(rbc_10_ewm) else float("nan"))


    # 3 — POSITION_ENC (notebook); fall back to POSITION_ENCODED when only that exists on raw logs
    if "POSITION_ENC" in last.index and pd.notna(last.get("POSITION_ENC")):
        res.append(float(last["POSITION_ENC"]))
    elif "POSITION_ENCODED" in pdf.columns:
        res.append(_nanfloat(float(pdf["POSITION_ENCODED"].iloc[-1])))
    else:
        res.append(float("nan"))

    # 4 — REB_PER_MIN_roll5
    res.append(_reb_per_min_roll5_last(pdf))

    # 5 — DREB_PCT_season_avg
    res.append(
        _season_expanding_tail(pdf, "DREB_PCT", seasons)
        if "DREB_PCT" in pdf.columns
        else float("nan")
    )

    # 6 — OPP_TEAM_DREB_ALLOWED
    if "OPP_TEAM_DREB_ALLOWED" in last.index and pd.notna(last.get("OPP_TEAM_DREB_ALLOWED")):
        res.append(float(last["OPP_TEAM_DREB_ALLOWED"]))
    else:
        res.append(
            _opp_team_prior_metric(
                df, opp_abbr, "OPP_DREB", as_of_date=date
            )
        )

    # 7 — REB_ROLL10_SLOPE
    res.append(_reb_roll10_slope_last(pdf))

    # 8 — EXPECTED_PACE
    if "EXPECTED_PACE" in last.index and pd.notna(last.get("EXPECTED_PACE")):
        res.append(_nanfloat(round(float(last["EXPECTED_PACE"]), 2)))
    else:
        tpace = _team_pace_shift_roll_tail(
            df,
            player_team,
            5,
            as_of_date=date,
            min_periods=5,
        )  # rpm _team_context: rolling(5) on TEAM pace (stored as TEAM_PACE_roll10)
        opace = _team_pace_shift_roll_tail(
            df,
            opp_abbr,
            10,
            as_of_date=date,
            min_periods=1,
        )  # _opponent_stats: rolling(..., min_periods=1) for opp rolls
        if pd.notna(tpace) and pd.notna(opace):
            res.append(round((tpace + opace) / 2.0, 2))
        else:
            res.append(float("nan"))

    # 9 — OPP_TEAM_OREB_ALLOWED
    if "OPP_TEAM_OREB_ALLOWED" in last.index and pd.notna(last.get("OPP_TEAM_OREB_ALLOWED")):
        res.append(float(last["OPP_TEAM_OREB_ALLOWED"]))
    else:
        res.append(_opp_team_prior_metric(df, opp_abbr, "OPP_OREB", as_of_date=date))

    # 10 — MIN_season_avg
    res.append(
        _season_expanding_tail(pdf, "MIN", seasons)
        if "MIN" in pdf.columns
        else float("nan")
    )

    # 11 — RPM_SEASON_STD
    if "PLAYER_ID" in pdf.columns:
        rpm_std_series = pdf.groupby("PLAYER_ID", group_keys=False)["REB_PER_MIN"].transform(
            lambda x: x.astype(float).expanding().std()
        )
    else:
        rpm_std_series = pdf["REB_PER_MIN"].astype(float).expanding().std()
    res.append(float(rpm_std_series.iloc[-1]) if pd.notna(rpm_std_series.iloc[-1]) else float("nan"))

    # 12 — OREB_PCT_season_avg
    res.append(
        _season_expanding_tail(pdf, "OREB_PCT", seasons)
        if "OREB_PCT" in pdf.columns
        else float("nan")
    )

    # 13 — DREB_PER_MIN_season_avg
    pdf['DREB_PER_MIN'] = pdf['DREB'] / pdf['MIN'].replace(0, np.nan)
    res.append(pdf['DREB_PER_MIN'].mean().astype(float))


    # 14 — OPP_FGA_roll10
    if "OPP_FGA_roll10" in last.index and pd.notna(last.get("OPP_FGA_roll10")):
        res.append(_nanfloat(last["OPP_FGA_roll10"]))
    else:
        res.append(_opp_fga_roll10_last(df, opp_abbr, as_of_date=date))

    return res
