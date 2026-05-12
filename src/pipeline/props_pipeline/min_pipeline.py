import pandas as pd

# Order must match training / saved quantile bundle `feature_names`.
MIN_FEATURES = [
    "MIN_5_ewm",
    "MIN_MEDIAN_L15",
    "MIN_SEASON_MEAN",
    "STARTER_ROLL10_PCT",
    "CONSEC_STARTS",
    "MIN_TREND",
    "MIN_RATE_OF_CHANGE",
    "AVG_STARTERS_MIN_L15",
]


def _consec_starts(starting_series: pd.Series) -> float:
    """Consecutive-starts streak at the end of the series (current state, unshifted)."""
    vals = starting_series.astype(float).values
    count = 0
    for val in reversed(vals):
        if val == 1.0:
            count += 1
        else:
            break
    return float(count)


def _avg_starters_min_l15(df: pd.DataFrame, team_id) -> float:
    """Rolling 15-game average of starter minutes for a team (unshifted)."""
    team_df = df[df["TEAM_ID"] == team_id]
    if "STARTING" not in team_df.columns:
        return float("nan")

    starter_game_avg = (
        team_df[team_df["STARTING"] == 1]
        .groupby("GAME_ID")["MIN"]
        .mean()
        .reset_index(name="GAME_AVG_STARTER_MIN")
    )
    if starter_game_avg.empty:
        return float("nan")

    game_dates = df[["GAME_ID", "GAME_DATE"]].drop_duplicates()
    starter_game_avg = (
        starter_game_avg
        .merge(game_dates, on="GAME_ID", how="left")
        .sort_values("GAME_DATE")
    )
    rolling = starter_game_avg["GAME_AVG_STARTER_MIN"].rolling(15, min_periods=5).mean()
    v = rolling.iloc[-1]
    return float(v) if pd.notna(v) else float("nan")


def min_pipeline(df, name, current_date=None):
    pdf = df[df["PLAYER_NAME"] == name].sort_values("GAME_DATE").copy()
    if len(pdf) < 10:
        return None

    last = pdf.iloc[-1]
    min_s = pdf["MIN"].astype(float)

    # MIN_5_ewm — unshifted ewm(span=5), includes latest game
    min_5_ewm_val = float(min_s.ewm(span=5, adjust=False).mean().iloc[-1])

    # MIN_SEASON_MEAN
    season_mean = float(min_s.mean())

    # MIN_MEDIAN_L15
    min_median_l15 = float(min_s.tail(15).median())

    # STARTER_ROLL10_PCT — unshifted rolling mean of last 10 games
    if "STARTING" not in pdf.columns:
        starter_roll10 = float("nan")
        consec = float("nan")
    else:
        st = pdf["STARTING"].astype(float)
        n = min(10, len(st))
        starter_roll10 = round(float(st.tail(n).mean()), 2)
        consec = _consec_starts(st)

    # MIN_TREND — MIN_5_ewm minus season mean
    min_trend = (
        float(min_5_ewm_val - season_mean)
        if pd.notna(min_5_ewm_val) and pd.notna(season_mean)
        else float("nan")
    )

    # MIN_RATE_OF_CHANGE — fractional deviation of recent form from season baseline
    if pd.notna(season_mean) and season_mean != 0 and pd.notna(min_5_ewm_val):
        min_roc = float((min_5_ewm_val - season_mean) / season_mean)
    else:
        min_roc = float("nan")

    # AVG_STARTERS_MIN_L15 — team-level rolling starter minute tendency
    avg_starters_l15 = _avg_starters_min_l15(df, last["TEAM_ID"])

    return [
        min_5_ewm_val,
        min_median_l15,
        season_mean,
        starter_roll10,
        consec,
        min_trend,
        min_roc,
        avg_starters_l15,
    ]
