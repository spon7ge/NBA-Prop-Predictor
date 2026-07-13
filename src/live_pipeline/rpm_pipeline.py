import numpy as np
import pandas as pd

from src.pipeline.features.rpm_features import RPM_FEATURES
from src.utils.helper_functions import findOpp

# Must match rpm_features.RPM_FEATURES / saved model feature order.
FEATURE_COLS = list(RPM_FEATURES)
assert FEATURE_COLS == list(RPM_FEATURES)

# Matches training: REB_PER_MIN_10_ewm / RBC_PER_MIN_10_ewm use span=10
_EWMA_SPAN = 10

# sklearn LabelEncoder sorts classes alphabetically when fit on PG/SG/SF/PF/C.
_POS_ENC = {"C": 0, "PF": 1, "PG": 2, "SF": 3, "SG": 4}


def _nanfloat(x) -> float:
    return float(x) if pd.notna(x) else float("nan")


def _roll_mean(series: pd.Series, window: int, min_periods: int = 1) -> float:
    """Unshifted rolling mean of the last `window` values (includes latest game)."""
    if series.empty or series.isna().all():
        return float("nan")
    v = series.astype(float).rolling(window, min_periods=min_periods).mean().iloc[-1]
    return _nanfloat(v)


def _roll_quantile(series: pd.Series, window: int, q: float, min_periods: int = 5) -> float:
    if series.empty or series.isna().all():
        return float("nan")
    v = (
        series.astype(float)
        .rolling(window, min_periods=min_periods)
        .quantile(q)
        .iloc[-1]
    )
    return _nanfloat(v)


def _ewma(series: pd.Series, span: int = _EWMA_SPAN) -> float:
    """Unshifted EWMA through the latest game."""
    if series.empty or series.isna().all():
        return float("nan")
    v = series.astype(float).ewm(span=span, min_periods=1).mean().iloc[-1]
    return _nanfloat(v)


def _expanding_mean(series: pd.Series) -> float:
    if series.empty or series.isna().all():
        return float("nan")
    return _nanfloat(series.astype(float).expanding(min_periods=1).mean().iloc[-1])


def _expanding_std(series: pd.Series) -> float:
    if series.empty or series.isna().all() or len(series.dropna()) < 2:
        return float("nan")
    return _nanfloat(series.astype(float).expanding(min_periods=2).std().iloc[-1])


def _season_series(pdf: pd.DataFrame, col: str) -> pd.Series:
    if col not in pdf.columns:
        return pd.Series(dtype=float)
    if "season_year" in pdf.columns:
        last_season = pdf.iloc[-1]["season_year"]
        return pdf.loc[pdf["season_year"] == last_season, col].astype(float)
    return pdf[col].astype(float)


def _season_mean(pdf: pd.DataFrame, col: str) -> float:
    return _expanding_mean(_season_series(pdf, col))


def _season_std(pdf: pd.DataFrame, col: str) -> float:
    return _expanding_std(_season_series(pdf, col))


def _rolling_slope(series: pd.Series, window: int = 10, min_periods: int = 5) -> float:
    """Unshifted linear slope over the last ``window`` observations."""
    if series.empty or series.isna().all():
        return float("nan")
    y    = series.astype(float).tail(window).to_numpy(dtype=float)
    mask = np.isfinite(y)
    if int(mask.sum()) < min_periods:
        return float("nan")
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x[mask], y[mask], 1)[0])


def _team_stat_roll10(
    df: pd.DataFrame,
    team_abbr: str | None,
    stat_col: str,
    *,
    as_of_date: str | None = None,
) -> float:
    """Team-level rolling-10 mean (one row per team_id/game_id), unshifted."""
    if team_abbr is None or stat_col not in df.columns:
        return float("nan")

    team = df[df["team_abbreviation"] == team_abbr].drop_duplicates(
        subset=["team_id", "game_id"]
    )
    if team.empty:
        return float("nan")

    if as_of_date is not None:
        cutoff = pd.to_datetime(as_of_date)
        dated  = team[pd.to_datetime(team["game_date"], errors="coerce") < cutoff]
        team   = dated if len(dated) > 0 else team

    team = team.sort_values("game_date")
    return _roll_mean(team[stat_col], 10)


def _position_enc(pdf: pd.DataFrame) -> float:
    """Encode position to integer. Checks silver 'pos' column and common variants."""
    for col in ("position_enc", "position_encoded", "pos"):
        if col in pdf.columns:
            raw = str(pdf[col].iloc[-1]).strip().upper()
            if raw in _POS_ENC:
                return float(_POS_ENC[raw])
            try:
                return float(raw)
            except ValueError:
                pass
    return float("nan")


def rpm_pipeline(df: pd.DataFrame, name: str, date: str):
    """
    Parameters
    ----------
    df   : full player + team game-log dataframe (silver snake_case columns)
    name : player_name to look up
    date : prediction date string 'YYYY-MM-DD'
    """
    pdf = df[df["player_name"] == name].sort_values("game_date").copy()
    if len(pdf) < 10:
        return None

    opp_abbr, _ = findOpp(name, pdf, date, max_days_ahead=3)

    # ── Derived columns ───────────────────────────────────────────────────────
    min_s = pdf["min"].replace(0, np.nan).astype(float)
    if "reb_per_min" not in pdf.columns and "reb" in pdf.columns:
        pdf["reb_per_min"] = pdf["reb"].astype(float) / min_s
    if "rbc_per_min" not in pdf.columns and "rbc" in pdf.columns:
        pdf["rbc_per_min"] = pdf["rbc"].astype(float) / min_s

    rpm = pdf["reb_per_min"].astype(float)

    # ── Rebound features ──────────────────────────────────────────────────────
    rpm_season_avg  = _season_mean(pdf, "reb_per_min")
    rpm_10_ewm      = _ewma(rpm, _EWMA_SPAN)

    if {"oreb", "dreb"}.issubset(pdf.columns):
        ratio = pdf["oreb"].astype(float) / pdf["dreb"].astype(float).replace(0, np.nan)
        pdf   = pdf.assign(_oreb_dreb_ratio=ratio)
        oreb_dreb_ratio = _season_mean(pdf, "_oreb_dreb_ratio")
    else:
        oreb_dreb_ratio = float("nan")

    pos_enc = _position_enc(pdf)
    rbc_per_min_10_ewm = (
        _ewma(pdf["rbc_per_min"], _EWMA_SPAN)
        if "rbc_per_min" in pdf.columns
        else float("nan")
    )
    rpm_season_std  = _season_std(pdf, "reb_per_min")
    reb_roll10_slope = _rolling_slope(rpm, 10)
    rpm_p10_l10     = _roll_quantile(rpm, 10, 0.1)
    rpm_p90_l10     = _roll_quantile(rpm, 10, 0.9)

    orbc_roll10 = _roll_mean(pdf["orbc"], 10) if "orbc" in pdf.columns else float("nan")
    drbc_roll10 = _roll_mean(pdf["drbc"], 10) if "drbc" in pdf.columns else float("nan")
    min_roll5   = _roll_mean(pdf["min"], 5)
    min_season  = _season_mean(pdf, "min")
    opp_reb_pct_roll10 = _team_stat_roll10(df, opp_abbr, "team_reb_pct", as_of_date=date)

    # Return order == FEATURE_COLS / RPM_FEATURES
    return [
        rpm_season_avg,        # REB_PER_MIN_season_avg
        rpm_10_ewm,            # REB_PER_MIN_10_ewm
        oreb_dreb_ratio,       # OREB_DREB_RATIO
        pos_enc,               # POSITION_ENC
        rbc_per_min_10_ewm,    # RBC_PER_MIN_10_ewm
        rpm_season_std,        # RPM_SEASON_STD
        reb_roll10_slope,      # REB_ROLL10_SLOPE
        rpm_p10_l10,           # REB_PER_MIN_P10_L10
        rpm_p90_l10,           # REB_PER_MIN_P90_L10
        orbc_roll10,           # player_ORBC_roll10_mean
        drbc_roll10,           # player_DRBC_roll10_mean
        min_roll5,             # player_MIN_roll5_mean
        min_season,            # player_MIN_season_mean
        opp_reb_pct_roll10,    # opp_team_TEAM_REB_PCT_roll10_mean
    ]
