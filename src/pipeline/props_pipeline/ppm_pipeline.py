import numpy as np
import pandas as pd

# Order must match training / saved quantile bundle `feature_names`.
PPM_FEATURES = [
    "PTS_PER_MIN_season_avg",
    "PTS_PER_MIN_5_ewm",
    "FGA_PER_MIN_5_ewm",
    "FTA_PER_MIN_5_ewm",
    "TS_PCT_5_ewm",
    "TS_PCT_season_avg",
    "PPM_STD_L10",
    "TS_PCT_STD_L10",
    "TEAM_PTS_PER_MIN_RANK_L10",
    "MIN_season_avg",
    "PPM_WHEN_USG_HIGH",
    "PPM_WHEN_USG_LOW",
    "PPM_WHEN_MIN_HIGH",
    "PPM_WHEN_MIN_LOW",
]


def _expanding_mean(series: pd.Series, ndigits: int = 2) -> float:
    if series.empty:
        return float("nan")
    v = series.astype(float).expanding().mean().round(ndigits).iloc[-1]
    return float(v) if pd.notna(v) else float("nan")


def _ewm_last(series: pd.Series, span: int) -> float:
    if series.empty:
        return float("nan")
    v = series.astype(float).ewm(span=span, adjust=False).mean().iloc[-1]
    return float(v) if pd.notna(v) else float("nan")


def _roll_last(series: pd.Series, window: int, min_periods: int = 3) -> float:
    if series.empty:
        return float("nan")
    v = series.astype(float).rolling(window, min_periods=min_periods).mean().iloc[-1]
    return float(v) if pd.notna(v) else float("nan")


def _roll_std(series: pd.Series, window: int, min_periods: int = 3) -> float:
    if series.empty:
        return float("nan")
    v = series.astype(float).rolling(window, min_periods=min_periods).std().iloc[-1]
    return float(v) if pd.notna(v) else float("nan")


def _conditional_ppm_last(
    series_stat: pd.Series,
    series_cond: pd.Series,
    threshold: float,
    comparison: str,
) -> float:
    if series_stat.empty or series_cond.empty:
        return float("nan")
    stat = series_stat.astype(float)
    cond = series_cond.astype(float)
    mask = (cond > threshold) if comparison == "gt" else (cond < threshold)
    v = stat.where(mask).expanding(min_periods=3).mean().iloc[-1]
    return float(v) if pd.notna(v) else float("nan")


def ppm_pipeline(df, name, current_date=None):
    pdf = df[df["PLAYER_NAME"] == name].sort_values("GAME_DATE").copy()
    if len(pdf) < 10:
        return None

    res = []
    last = pdf.iloc[-1]

    if "PTS_PER_MIN" not in pdf.columns:
        pdf["PTS_PER_MIN"] = pdf["PTS"].astype(float) / pdf["MIN"].replace(0, np.nan)

    if "FGA_PER_MIN" not in pdf.columns:
        pdf["FGA_PER_MIN"] = pdf["FGA"].astype(float) / pdf["MIN"].replace(0, np.nan)

    if "FTA_PER_MIN" not in pdf.columns:
        pdf["FTA_PER_MIN"] = pdf["FTA"].astype(float) / pdf["MIN"].replace(0, np.nan)

    # ── PTS_PER_MIN_season_avg ────────────────────────────────────────────────
    res.append(_expanding_mean(pdf["PTS_PER_MIN"], 2))

    # ── PTS_PER_MIN_5_ewm ─────────────────────────────────────────────────────
    res.append(_ewm_last(pdf["PTS_PER_MIN"], 5))

    # ── FGA_PER_MIN_5_ewm ─────────────────────────────────────────────────────
    res.append(_ewm_last(pdf["FGA_PER_MIN"], 5))

    # ── FTA_PER_MIN_5_ewm ─────────────────────────────────────────────────────
    res.append(_ewm_last(pdf["FTA_PER_MIN"], 5))

    # ── TS_PCT_5_ewm ──────────────────────────────────────────────────────────
    res.append(_ewm_last(pdf["TS_PCT"], 5))

    # ── TS_PCT_season_avg ─────────────────────────────────────────────────────
    res.append(_expanding_mean(pdf["TS_PCT"], 2))

    # ── PPM_STD_L10 ───────────────────────────────────────────────────────────
    res.append(_roll_std(pdf["PTS_PER_MIN"], 10))

    # ── TS_PCT_STD_L10 ────────────────────────────────────────────────────────
    res.append(_roll_std(pdf["TS_PCT"], 10))

    # ── TEAM_PTS_PER_MIN_RANK_L10 (dense rank, high roll10 = rank 1) ──────────
    team_df = df[df["TEAM_ID"] == last["TEAM_ID"]]
    teammate_roll10 = {}
    for player, grp in team_df.groupby("PLAYER_NAME"):
        grp_sorted = grp.sort_values("GAME_DATE")
        if "PTS_PER_MIN" not in grp_sorted.columns:
            grp_sorted = grp_sorted.copy()
            grp_sorted["PTS_PER_MIN"] = (
                grp_sorted["PTS"].astype(float) / grp_sorted["MIN"].replace(0, np.nan)
            )
        teammate_roll10[player] = _roll_last(grp_sorted["PTS_PER_MIN"].astype(float), 10)
    sorted_teammates = sorted(
        teammate_roll10.items(),
        key=lambda x: (x[1] if pd.notna(x[1]) else -np.inf),
        reverse=True,
    )
    rank_map = {p: float(i + 1) for i, (p, _) in enumerate(sorted_teammates)}
    res.append(rank_map.get(name, float("nan")))

    # ── MIN_season_avg ────────────────────────────────────────────────────────
    res.append(_expanding_mean(pdf["MIN"], 2))

    # ── PPM_WHEN_USG_HIGH (USG_PCT > 0.18) ───────────────────────────────────
    res.append(_conditional_ppm_last(pdf["PTS_PER_MIN"], pdf["USG_PCT"], 0.18, "gt"))

    # ── PPM_WHEN_USG_LOW  (USG_PCT < 0.17) ───────────────────────────────────
    res.append(_conditional_ppm_last(pdf["PTS_PER_MIN"], pdf["USG_PCT"], 0.17, "lt"))

    # ── PPM_WHEN_MIN_HIGH (MIN > 25) ──────────────────────────────────────────
    res.append(_conditional_ppm_last(pdf["PTS_PER_MIN"], pdf["MIN"], 25, "gt"))

    # ── PPM_WHEN_MIN_LOW  (MIN < 25) ──────────────────────────────────────────
    res.append(_conditional_ppm_last(pdf["PTS_PER_MIN"], pdf["MIN"], 25, "lt"))

    return res
