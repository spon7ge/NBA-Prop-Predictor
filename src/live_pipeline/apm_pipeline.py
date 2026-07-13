import numpy as np
import pandas as pd

from src.utils.helper_functions import findOpp

# Must match apm_features.APM_FEATURES / saved model feature order.
FEATURE_COLS = [
    "player_AST_PER_MIN_ewma",
    "player_PASS_roll10_mean",
    "player_PASS_season_mean",
    "player_SAST_roll10_mean",
    "player_AST_RATIO_roll10_mean",
    "player_MIN_roll5_mean",
    "MIN_TREND_5v20",
    "POSITION_ENCODED",
    "opp_team_TEAM_DEF_RATING_roll10_mean",
    "opp_team_TEAM_PACE_roll10_mean",
    "own_team_TEAM_AST_PCT_roll10_mean",
]

# Matches training: ewm(span=max(ROLLING_WINDOWS)) with ROLLING_WINDOWS = [5, 10, 20]
_EWMA_SPAN = 20

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


def _ewma(series: pd.Series, span: int = _EWMA_SPAN) -> float:
    """Unshifted EWMA through the latest game."""
    if series.empty or series.isna().all():
        return float("nan")
    v = series.astype(float).ewm(span=span, min_periods=1).mean().iloc[-1]
    return _nanfloat(v)


def _season_mean(pdf: pd.DataFrame, col: str) -> float:
    """Season-to-date expanding mean (unshifted), scoped to season_year when present."""
    if col not in pdf.columns:
        return float("nan")
    if "season_year" in pdf.columns:
        last_season = pdf.iloc[-1]["season_year"]
        series = pdf.loc[pdf["season_year"] == last_season, col]
    else:
        series = pdf[col]
    if series.empty:
        return float("nan")
    return _nanfloat(series.astype(float).expanding(min_periods=1).mean().iloc[-1])


def _team_stat_roll10(
    df: pd.DataFrame,
    team_abbr: str | None,
    stat_col: str,
    *,
    as_of_date: str | None = None,
) -> float:
    """
    Team-level rolling-10 mean of stat_col (one row per team_id/game_id).
    Unshifted; optionally restricted to games before as_of_date.
    """
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
    """Encode position to integer. Checks silver 'pos' column first."""
    for col in ("position_encoded", "position_enc", "pos"):
        if col in pdf.columns:
            raw = str(pdf[col].iloc[-1]).strip().upper()
            if raw in _POS_ENC:
                return float(_POS_ENC[raw])
            # try direct numeric
            try:
                return float(raw)
            except ValueError:
                pass
    return float("nan")


def apm_pipeline(df: pd.DataFrame, name: str, date: str):
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
    own_abbr = last.get("team_abbreviation")

    # ── Derived columns ───────────────────────────────────────────────────────
    min_s = pdf["min"].replace(0, np.nan).astype(float)
    if "ast_per_min" not in pdf.columns and "ast" in pdf.columns:
        pdf["ast_per_min"] = pdf["ast"].astype(float) / min_s

    # ── Player features (unshifted) ───────────────────────────────────────────
    apm_ewma = (
        _ewma(pdf["ast_per_min"], _EWMA_SPAN)
        if "ast_per_min" in pdf.columns
        else float("nan")
    )
    pass_roll10      = _roll_mean(pdf["pass"],      10) if "pass"      in pdf.columns else float("nan")
    pass_season      = _season_mean(pdf, "pass")
    sast_roll10      = _roll_mean(pdf["sast"],      10) if "sast"      in pdf.columns else float("nan")
    ast_ratio_roll10 = _roll_mean(pdf["ast_ratio"], 10) if "ast_ratio" in pdf.columns else float("nan")

    min_roll5  = _roll_mean(pdf["min"], 5)
    min_roll20 = _roll_mean(pdf["min"], 20)
    min_trend  = (
        float(min_roll5 - min_roll20)
        if pd.notna(min_roll5) and pd.notna(min_roll20)
        else float("nan")
    )

    pos_enc = _position_enc(pdf)

    # ── Team / matchup context ────────────────────────────────────────────────
    opp_def_roll10     = _team_stat_roll10(df, opp_abbr, "team_def_rating", as_of_date=date)
    opp_pace_roll10    = _team_stat_roll10(df, opp_abbr, "team_pace",       as_of_date=date)
    own_ast_pct_roll10 = _team_stat_roll10(df, own_abbr, "team_ast_pct",    as_of_date=date)

    # Return order == FEATURE_COLS / APM_FEATURES
    return [
        apm_ewma,             # player_AST_PER_MIN_ewma
        pass_roll10,          # player_PASS_roll10_mean
        pass_season,          # player_PASS_season_mean
        sast_roll10,          # player_SAST_roll10_mean
        ast_ratio_roll10,     # player_AST_RATIO_roll10_mean
        min_roll5,            # player_MIN_roll5_mean
        min_trend,            # MIN_TREND_5v20
        pos_enc,              # POSITION_ENCODED
        opp_def_roll10,       # opp_team_TEAM_DEF_RATING_roll10_mean
        opp_pace_roll10,      # opp_team_TEAM_PACE_roll10_mean
        own_ast_pct_roll10,   # own_team_TEAM_AST_PCT_roll10_mean
    ]
