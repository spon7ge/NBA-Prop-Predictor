import pandas as pd

# Must match min_quantile_model FEATURE_COLS / saved model feature order.
FEATURE_COLS = [
    "CONSEC_STARTS",
    "MIN_10_ewm",
    "TEAM_MIN_RANK_L10",
    "STARTER_ROLL10_PCT",
    "MIN_SEASON_MEAN",
    "MIN_SUM_LAST_7_DAYS",
    "TEAM_USG_RANK_L10",
    "AST_PCT_lag2",
    "MIN_SEASON_STD",
    "GAMES_PLAYED_LAST_14_DAYS",
    "GAMES_PLAYED_LAST_7_DAYS",
    "MIN_RATE_OF_CHANGE",
    "PIE_lag2",
    "SPD_10_ewm",
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


def _team_min_rank_l10(df: pd.DataFrame, player_name: str, team_id) -> float:
    """
    Rank of the player's L10 avg minutes among all teammates (unshifted).
    Rank 1 = most minutes on the team.
    """
    team_df = df[df["team_id"] == team_id].copy()
    if team_df.empty:
        return float("nan")

    l10_avg = (
        team_df.sort_values("game_date")
        .groupby("player_name")["min"]
        .apply(lambda s: s.astype(float).tail(10).mean())
    )
    if l10_avg.empty or player_name not in l10_avg.index:
        return float("nan")

    ranked = l10_avg.rank(ascending=False, method="min")
    return float(ranked[player_name])


def _team_usg_rank_l10(df: pd.DataFrame, player_name: str, team_id) -> float:
    """
    Rank of the player's L10 avg usage among all teammates (unshifted).
    Rank 1 = highest usage on the team.
    """
    usg_col = "usg_pct" if "usg_pct" in df.columns else "usg" if "usg" in df.columns else None
    if usg_col is None:
        return float("nan")

    team_df = df[df["team_id"] == team_id].copy()
    if team_df.empty:
        return float("nan")

    l10_avg = (
        team_df.sort_values("game_date")
        .groupby("player_name")[usg_col]
        .apply(lambda s: s.astype(float).tail(10).mean())
    )
    if l10_avg.empty or player_name not in l10_avg.index:
        return float("nan")

    ranked = l10_avg.rank(ascending=False, method="min")
    return float(ranked[player_name])


def _lag_n(pdf: pd.DataFrame, col: str, n: int) -> float:
    """
    Live lag: nth most recent completed game (no shift).
    lag1 -> iloc[-1], lag2 -> iloc[-2], matching training shift(n) for the next game.
    """
    if col not in pdf.columns or len(pdf) < n:
        return float("nan")
    val = pdf[col].iloc[-n]
    return float(val) if pd.notna(val) else float("nan")


def _fatigue_windows(pdf: pd.DataFrame, as_of) -> tuple[float, float, float]:
    """
    Games / minutes in the 7- and 14-day windows before the prediction date.
    Includes all completed games (no shift / does not exclude a 'current' row).
    """
    as_of = pd.to_datetime(as_of)
    dates = pd.to_datetime(pdf["game_date"])
    prior = dates < as_of
    mins = pdf["min"].astype(float)

    mask_7  = prior & (dates >= as_of - pd.Timedelta(days=7))
    mask_14 = prior & (dates >= as_of - pd.Timedelta(days=14))

    games_7   = float(mask_7.sum())
    games_14  = float(mask_14.sum())
    min_sum_7 = float(round(mins[mask_7].sum(), 1))
    return games_7, games_14, min_sum_7


def min_pipeline(df, name, current_date=None):
    pdf = df[df["player_name"] == name].sort_values("game_date").copy()
    if len(pdf) < 10:
        return None

    last    = pdf.iloc[-1]
    min_s   = pdf["min"].astype(float)

    # MIN_10_ewm — unshifted ewm(span=10), includes latest game
    min_10_ewm_val = float(min_s.ewm(span=10, adjust=False).mean().iloc[-1])

    # MIN_SEASON_MEAN / MIN_SEASON_STD
    season_mean = float(min_s.mean())
    season_std  = float(min_s.std()) if len(min_s) > 1 else float("nan")

    # STARTING — derive from start_position if not already present
    if "starting" not in pdf.columns and "start_position" in pdf.columns:
        pdf["starting"] = pdf["start_position"].notna().astype(int)

    starting_col = "starting" if "starting" in pdf.columns else None

    # STARTER_ROLL10_PCT / CONSEC_STARTS — unshifted
    if starting_col is None:
        starter_roll10 = float("nan")
        consec         = float("nan")
    else:
        st = pdf[starting_col].astype(float)
        n  = min(10, len(st))
        starter_roll10 = round(float(st.tail(n).mean()), 2)
        consec         = _consec_starts(st)

    # MIN_RATE_OF_CHANGE — fractional deviation of recent form from season baseline
    if pd.notna(season_mean) and season_mean != 0 and pd.notna(min_10_ewm_val):
        min_roc = float((min_10_ewm_val - season_mean) / season_mean)
    else:
        min_roc = float("nan")

    team_min_rank = _team_min_rank_l10(df, name, last["team_id"])
    team_usg_rank = _team_usg_rank_l10(df, name, last["team_id"])

    # Fatigue windows relative to prediction date (default: day after last game)
    as_of = (
        current_date
        if current_date is not None
        else pd.to_datetime(last["game_date"]) + pd.Timedelta(days=1)
    )
    games_7, games_14, min_sum_7 = _fatigue_windows(pdf, as_of)

    # lag2 = second-most-recent completed game (live equivalent of shift(2))
    ast_pct_lag2 = _lag_n(pdf, "ast_pct", 2)
    pie_lag2     = _lag_n(pdf, "pie",     2)

    # SPD_10_ewm — unshifted ewm(span=10)
    if "spd" in pdf.columns:
        spd_10_ewm_val = float(
            pdf["spd"].astype(float).ewm(span=10, adjust=False).mean().iloc[-1]
        )
    else:
        spd_10_ewm_val = float("nan")

    # Return order == FEATURE_COLS
    return [
        consec,           # CONSEC_STARTS
        min_10_ewm_val,   # MIN_10_ewm
        team_min_rank,    # TEAM_MIN_RANK_L10
        starter_roll10,   # STARTER_ROLL10_PCT
        season_mean,      # MIN_SEASON_MEAN
        min_sum_7,        # MIN_SUM_LAST_7_DAYS
        team_usg_rank,    # TEAM_USG_RANK_L10
        ast_pct_lag2,     # AST_PCT_lag2
        season_std,       # MIN_SEASON_STD
        games_14,         # GAMES_PLAYED_LAST_14_DAYS
        games_7,          # GAMES_PLAYED_LAST_7_DAYS
        min_roc,          # MIN_RATE_OF_CHANGE
        pie_lag2,         # PIE_lag2
        spd_10_ewm_val,   # SPD_10_ewm
    ]
