import numpy as np
import pandas as pd

from src.pipeline.features.ppm_features import PPM_FEATURES
from src.utils.helper_functions import findOpp

# Must match ppm_features.PPM_FEATURES / saved model feature order.
FEATURE_COLS = list(PPM_FEATURES)
assert FEATURE_COLS == list(PPM_FEATURES)

# Matches training spans in ppm_features.EWM_SPANS / TEAM_EWM_SPANS
_EWM_SPANS        = (5, 10, 20)
_OPP_DEF_EWM_SPAN = 10


def _nanfloat(x) -> float:
    return float(x) if pd.notna(x) else float("nan")


def _ewm_mean(series: pd.Series, span: int, min_periods: int = 1) -> float:
    """Unshifted ewm through the latest game in history (includes most recent row)."""
    if series.empty or series.isna().all():
        return float("nan")
    v = series.astype(float).ewm(span=span, min_periods=min_periods).mean().iloc[-1]
    return _nanfloat(v)


def _expanding_mean(series: pd.Series) -> float:
    """Unshifted expanding mean through the latest game in history."""
    if series.empty or series.isna().all():
        return float("nan")
    return _nanfloat(series.astype(float).expanding(min_periods=1).mean().iloc[-1])


def _season_series(pdf: pd.DataFrame, col: str) -> pd.Series:
    if col not in pdf.columns:
        return pd.Series(dtype=float)
    if "season_year" in pdf.columns:
        last_season = pdf.iloc[-1]["season_year"]
        return pdf.loc[pdf["season_year"] == last_season, col].astype(float)
    return pdf[col].astype(float)


def _season_mean(pdf: pd.DataFrame, col: str) -> float:
    """Season-to-date expanding mean (unshifted), scoped to season_year when present."""
    return _expanding_mean(_season_series(pdf, col))


def _opp_team_def_rating_10_ewm(
    df: pd.DataFrame,
    opp_abbr: str | None,
    as_of_date: str | None,
) -> float:
    """Opponent team_def_rating span-10 ewm (unshifted, through latest opp game on/before date)."""
    if opp_abbr is None:
        return float("nan")

    def_col = "team_def_rating" if "team_def_rating" in df.columns else None
    if def_col is None:
        return float("nan")

    opp = df[df["team_abbreviation"] == opp_abbr].drop_duplicates(
        subset=["team_id", "game_id"]
    )
    if opp.empty:
        return float("nan")

    if as_of_date is not None:
        cutoff = pd.to_datetime(as_of_date)
        dated  = opp[pd.to_datetime(opp["game_date"], errors="coerce") < cutoff]
        opp    = dated if len(dated) > 0 else opp

    opp = opp.sort_values("game_date")
    return _ewm_mean(opp[def_col], _OPP_DEF_EWM_SPAN)


def _starting_rate_last10(pdf: pd.DataFrame) -> float:
    """Share of starts over the last 10 games (unshifted tail mean)."""
    starting_col = None
    if "starting" in pdf.columns:
        starting_col = "starting"
    elif "start_position" in pdf.columns:
        starting_col = "_starting_derived"
        pdf = pdf.copy()
        pdf[starting_col] = pdf["start_position"].notna().astype(int)

    if starting_col is None:
        return float("nan")

    st = pdf[starting_col].astype(float)
    n  = min(10, len(st))
    return _nanfloat(st.tail(n).mean())


def ppm_pipeline(df: pd.DataFrame, name: str, date: str):
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

    last     = pdf.iloc[-1]
    opp_abbr, _ = findOpp(name, pdf, date, max_days_ahead=3)

    # ── Derived columns ───────────────────────────────────────────────────────
    min_s = pdf["min"].replace(0, np.nan).astype(float)
    if "pts_per_min" not in pdf.columns:
        pdf["pts_per_min"] = pdf["pts"].astype(float) / min_s

    if "starting" not in pdf.columns and "start_position" in pdf.columns:
        pdf["starting"] = pdf["start_position"].notna().astype(int)

    usg_col = "usg_pct" if "usg_pct" in pdf.columns else "usg"

    # ── Player scoring rate ───────────────────────────────────────────────────
    ppm            = pdf["pts_per_min"].astype(float)
    ppm_expanding  = _expanding_mean(ppm)
    ppm_5_ewm      = _ewm_mean(ppm, 5)
    ppm_10_ewm     = _ewm_mean(ppm, 10)

    # ── Usage / volume / FT (season) + FGA 5_ewm ─────────────────────────────
    usg_season    = _season_mean(pdf, usg_col)
    fga_season    = _season_mean(pdf, "fga")
    fga_5_ewm     = _ewm_mean(pdf["fga"], 5) if "fga" in pdf.columns else float("nan")
    ft_pct_season = _season_mean(pdf, "ft_pct")

    # ── Efficiency ewms ───────────────────────────────────────────────────────
    fg3_10_ewm = (
        _ewm_mean(pdf["fg3_pct"], 10) if "fg3_pct" in pdf.columns else float("nan")
    )
    efg_10_ewm = (
        _ewm_mean(pdf["efg_pct"], 10) if "efg_pct" in pdf.columns else float("nan")
    )

    # ── Minutes + role trends (ewm5 − ewm20) ─────────────────────────────────
    min_season = _season_mean(pdf, "min")
    min_10_ewm = _ewm_mean(pdf["min"], 10)
    min_5_ewm  = _ewm_mean(pdf["min"], 5)
    min_20_ewm = _ewm_mean(pdf["min"], 20)
    min_trend  = (
        float(min_5_ewm - min_20_ewm)
        if pd.notna(min_5_ewm) and pd.notna(min_20_ewm)
        else float("nan")
    )

    if "ts_pct" in pdf.columns:
        ts_5_ewm  = _ewm_mean(pdf["ts_pct"], 5)
        ts_20_ewm = _ewm_mean(pdf["ts_pct"], 20)
        ts_trend  = (
            float(ts_5_ewm - ts_20_ewm)
            if pd.notna(ts_5_ewm) and pd.notna(ts_20_ewm)
            else float("nan")
        )
    else:
        ts_trend = float("nan")

    # ── Opponent context ──────────────────────────────────────────────────────
    opp_def_10_ewm = _opp_team_def_rating_10_ewm(df, opp_abbr, as_of_date=date)

    # ── Rest / role ───────────────────────────────────────────────────────────
    last_date = pd.to_datetime(last["game_date"])
    pred_date = pd.to_datetime(date)
    days_rest = float((pred_date - last_date).days)
    if days_rest < 0:
        days_rest = float("nan")

    starting_rate = _starting_rate_last10(pdf)

    # Return order == FEATURE_COLS / PPM_FEATURES
    return [
        ppm_expanding,    # player_PTS_PER_MIN_expanding_mean
        ppm_5_ewm,        # player_PTS_PER_MIN_5_ewm
        ppm_10_ewm,       # player_PTS_PER_MIN_10_ewm
        usg_season,       # player_USG_PCT_season_mean
        fga_season,       # player_FGA_season_mean
        fga_5_ewm,        # player_FGA_5_ewm
        ft_pct_season,    # player_FT_PCT_season_mean
        fg3_10_ewm,       # player_FG3_PCT_10_ewm
        efg_10_ewm,       # player_EFG_PCT_10_ewm
        min_season,       # player_MIN_season_mean
        min_10_ewm,       # player_MIN_10_ewm
        min_trend,        # MIN_TREND_5v20
        ts_trend,         # TS_TREND_5v20
        opp_def_10_ewm,   # opp_team_TEAM_DEF_RATING_10_ewm
        days_rest,        # DAYS_REST
        starting_rate,    # STARTING_rate_last10
    ]
