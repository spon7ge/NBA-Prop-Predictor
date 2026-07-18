"""Shared live-feature helpers.

Training uses ``shift(1)`` then window on historical rows. Live predicts the
*next* game, so history is already prior-only — compute windows unshifted and
take ``iloc[-1]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Match training FeatureEngineer (avoid importing build_features → db deps).
EWM_MIN_PERIODS = 3

# sklearn LabelEncoder alphabetical order on PG/SG/SF/PF/C
_POS_ENC = {"C": 0, "PF": 1, "PG": 2, "SF": 3, "SG": 4}

MIN_GAMES = 10


def nanfloat(x) -> float:
    return float(x) if pd.notna(x) else float("nan")


def parse_minutes(series: pd.Series) -> pd.Series:
    """Parse ``MM:SS`` / numeric minutes to float."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    def _one(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        s = str(v).strip()
        if not s or s.lower() in ("nan", "none", ""):
            return np.nan
        if ":" in s:
            parts = s.split(":")
            try:
                if len(parts) == 2:
                    return float(parts[0]) + float(parts[1]) / 60.0
                if len(parts) == 3:
                    return (
                        float(parts[0]) * 60
                        + float(parts[1])
                        + float(parts[2]) / 60.0
                    )
            except ValueError:
                return np.nan
        try:
            return float(s)
        except ValueError:
            return np.nan

    return series.map(_one).astype(float)


def player_history(
    df: pd.DataFrame,
    name: str,
    as_of_date: str,
    *,
    min_games: int = MIN_GAMES,
) -> pd.DataFrame | None:
    """Completed games for ``name`` strictly before ``as_of_date``."""
    if "player_name" not in df.columns or "game_date" not in df.columns:
        return None
    cutoff = pd.to_datetime(as_of_date)
    pdf = df[df["player_name"] == name].copy()
    if pdf.empty:
        return None
    pdf["game_date"] = pd.to_datetime(pdf["game_date"], errors="coerce")
    pdf = pdf[pdf["game_date"] < cutoff].sort_values("game_date")
    if len(pdf) < min_games:
        return None
    return pdf.reset_index(drop=True)


def ensure_starting(pdf: pd.DataFrame) -> pd.DataFrame:
    """Derive ``starting`` from ``start_position`` / ``position`` when missing."""
    out = pdf
    if "starting" in out.columns:
        return out
    out = out.copy()
    for col in ("start_position", "position"):
        if col in out.columns:
            pos = out[col].astype(str).str.strip()
            out["starting"] = (
                out[col].notna()
                & pos.ne("")
                & pos.str.lower().ne("nan")
                & pos.str.lower().ne("none")
            ).astype(int)
            return out
    out["starting"] = 0
    return out


def ensure_per_min(pdf: pd.DataFrame, raw_cols: list[str]) -> pd.DataFrame:
    """Add ``{col}_per_min`` from counting stats / ``min`` when missing."""
    out = pdf.copy()
    if "min" not in out.columns:
        return out
    min_s = out["min"].replace(0, np.nan).astype(float)
    for col in raw_cols:
        dest = f"{col}_per_min"
        if dest not in out.columns and col in out.columns:
            out[dest] = out[col].astype(float) / min_s
    return out


def ewm_hl(series: pd.Series, halflife: int, min_periods: int = EWM_MIN_PERIODS) -> float:
    """Unshifted EWM mean through latest game (live ≡ training shift-then-ewm)."""
    if series is None or series.empty or series.isna().all():
        return float("nan")
    v = (
        series.astype(float)
        .ewm(halflife=halflife, min_periods=min_periods)
        .mean()
        .iloc[-1]
    )
    return nanfloat(v)


def season_avg(pdf: pd.DataFrame, col: str) -> float:
    """Unshifted expanding mean scoped to latest ``season_year`` when present."""
    if col not in pdf.columns:
        return float("nan")
    if "season_year" in pdf.columns:
        last_season = pdf.iloc[-1]["season_year"]
        s = pdf.loc[pdf["season_year"] == last_season, col].astype(float)
    else:
        s = pdf[col].astype(float)
    if s.empty or s.isna().all():
        return float("nan")
    return nanfloat(s.expanding(min_periods=1).mean().iloc[-1])


def lag1(pdf: pd.DataFrame, col: str) -> float:
    """Most recent completed-game value (live ≡ training ``shift(1)`` lag1)."""
    if col not in pdf.columns or pdf.empty:
        return float("nan")
    return nanfloat(pdf[col].iloc[-1])


def roll_mean(series: pd.Series, window: int, min_periods: int | None = None) -> float:
    """Unshifted rolling mean of last ``window`` values."""
    if series is None or series.empty or series.isna().all():
        return float("nan")
    mp = min_periods if min_periods is not None else max(2, window // 2)
    v = series.astype(float).rolling(window, min_periods=mp).mean().iloc[-1]
    return nanfloat(v)


def trend_5v20(series: pd.Series) -> float:
    a, b = ewm_hl(series, 5), ewm_hl(series, 20)
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    return float(a - b)


def starter_roll_pct(pdf: pd.DataFrame, window: int = 10, min_periods: int = 3) -> float:
    pdf = ensure_starting(pdf)
    return roll_mean(pdf["starting"].astype(float), window, min_periods=min_periods)


def position_encoded(pdf: pd.DataFrame) -> float:
    for col in ("position_encoded", "position_enc", "pos"):
        if col not in pdf.columns:
            continue
        raw = str(pdf[col].iloc[-1]).strip().upper()
        if raw in _POS_ENC:
            return float(_POS_ENC[raw])
        try:
            return float(raw)
        except ValueError:
            pass
    return float("nan")


def team_game_frame(
    df: pd.DataFrame,
    team_abbr: str | None,
    as_of_date: str,
) -> pd.DataFrame:
    """One row per team-game for ``team_abbr`` before ``as_of_date``."""
    if not team_abbr or "team_abbreviation" not in df.columns:
        return pd.DataFrame()
    cutoff = pd.to_datetime(as_of_date)
    team = df[df["team_abbreviation"] == team_abbr].copy()
    if team.empty:
        return team
    team["game_date"] = pd.to_datetime(team["game_date"], errors="coerce")
    team = team[team["game_date"] < cutoff]
    subset = ["team_id", "game_id"] if "team_id" in team.columns else ["game_id"]
    team = team.drop_duplicates(subset=subset).sort_values("game_date")
    return team.reset_index(drop=True)


def team_stat_ewm(
    df: pd.DataFrame,
    team_abbr: str | None,
    stat_col: str,
    as_of_date: str,
    *,
    halflife: int,
) -> float:
    """Unshifted team-level EWM for ``stat_col`` (used for own-team and opp)."""
    team = team_game_frame(df, team_abbr, as_of_date)
    if team.empty or stat_col not in team.columns:
        return float("nan")
    return ewm_hl(team[stat_col], halflife)


def team_rank_l10(
    df: pd.DataFrame,
    player_name: str,
    team_id,
    as_of_date: str,
    stat_col: str,
    *,
    halflife: int = 10,
) -> float:
    """Rank player among teammates by unshifted ``stat`` EWM (1 = highest)."""
    if team_id is None or stat_col not in df.columns:
        return float("nan")
    cutoff = pd.to_datetime(as_of_date)
    team_df = df[df["team_id"] == team_id].copy()
    if team_df.empty:
        return float("nan")
    team_df["game_date"] = pd.to_datetime(team_df["game_date"], errors="coerce")
    team_df = team_df[team_df["game_date"] < cutoff]
    if team_df.empty:
        return float("nan")

    scores = {}
    for pname, g in team_df.groupby("player_name", sort=False):
        g = g.sort_values("game_date")
        scores[pname] = ewm_hl(g[stat_col].astype(float), halflife)

    if player_name not in scores or not np.isfinite(scores[player_name]):
        return float("nan")
    series = pd.Series(scores, dtype=float).dropna()
    if series.empty:
        return float("nan")
    ranked = series.rank(ascending=False, method="dense")
    return float(ranked[player_name])


def ordered_features(feature_names: list[str], values: dict[str, float]) -> list[float]:
    """Return feature vector in ``feature_names`` order."""
    return [float(values.get(name, float("nan"))) for name in feature_names]
